# SPDX-License-Identifier: Apache-2.0
"""SGLang attention backend for the gfx906 Q8 FlashAttention kernel.

Wraps the vendored ``_gfx906_fa_C`` extension (see ``gfx906_fa/``). The KV
cache is token-major in SGLang, so we pass the request's token slot ids as a
block table with ``block_size = 1``: the extension's gather path supports
arbitrary block sizes, which avoids a 16-token paged KV layout for now.

Stage 1 uses the existing fp16 KV pool. Stage 2 (``key_cache_q8``) plugs the
gfx906 q8_0 K cache into the same call.
"""

from typing import Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.gfx906_fa import forward_paged
from sglang.srt.layers.attention.gfx906_fa.ext import load_ext
from sglang.srt.mem_cache.memory_pool import KVWriteLoc


class Gfx906FAttnBackend(AttentionBackend):
    # The FA wrapper/extension is used eagerly for now; graph capture needs
    # static gather/q-pad buffers and is added separately.
    needs_cpu_seq_lens = False
    supports_ragged_verify_graph = False

    def __init__(self, model_runner):
        super().__init__()
        self.device = model_runner.device
        self.dtype = model_runner.dtype
        self.data_type = model_runner.dtype
        self.max_context_len = model_runner.model_config.context_len
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.req_to_token = self.req_to_token_pool.req_to_token
        self.forward_metadata = None
        self._ext = load_ext()
        self._graph_max_bs = 0
        self._graph_block_table = None
        self._graph_cu_q = None
        self._graph_slot = None
        self._graph_seq_lens = None

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        self._graph_max_bs = max_bs
        self._graph_block_table = torch.zeros(
            (max_bs, self.max_context_len), dtype=torch.int32, device=self.device
        )
        self._graph_cu_q = torch.arange(
            max_bs + 1, dtype=torch.int32, device=self.device
        )
        # Static KV-write slots read by the captured q8 write path. The graph
        # runner updates out_cache_loc outside the graph; a capture-time
        # .to(int64) copy would freeze the dummy slot, so we refresh this
        # buffer in init_forward_metadata_out_graph before every replay.
        self._graph_slot = torch.zeros(max_bs, dtype=torch.int64, device=self.device)
        self._graph_seq_lens = torch.ones(
            max_bs, dtype=torch.int32, device=self.device
        )
        # Must be in place BEFORE capture: the write path's Python branch is
        # frozen into the graph, so setting this only on replay would bake the
        # capture-time `loc` tensor (always slot 0) instead.
        qm = self._quant_method()
        if qm is not None:
            qm.graph_slot = self._graph_slot

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def _build_metadata(self, forward_batch):
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens.to(torch.int32)
        batch_size = seq_lens.shape[0]

        # CUDA graph capture: the metadata tensors must be static, and
        # ``.item()`` (D2H sync) is not allowed while capturing. The block
        # table copy is captured, so replay refreshes it from the graph's
        # static req_pool_indices input buffer. Use the pool's max context as
        # the static KV tile width; the kernel still cuts at dynamic seq_lens.
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            if self._graph_block_table is None:
                raise RuntimeError("init_cuda_graph_state() was not called")
            # The graph reads this static table. It is refreshed eagerly in
            # init_forward_metadata_out_graph() before every replay (doing the
            # copy inside capture would bake the capture-time req_to_token
            # contents instead of following the live table).
            block_table = self._graph_block_table[:batch_size]
            if self._graph_seq_lens is None:
                raise RuntimeError("init_cuda_graph_state() was not called")
            seq_lens = self._graph_seq_lens[:batch_size]
            max_q = 1
            max_k = self.max_context_len
            cu_q = self._graph_cu_q[: batch_size + 1]
            self.forward_metadata = (block_table, seq_lens, cu_q, max_q, max_k)
            return self.forward_metadata

        block_table = self.req_to_token[req_pool_indices].to(torch.int32).contiguous()
        max_k = int(seq_lens.max().item())

        if forward_batch.forward_mode.is_decode():
            max_q = 1
            cu_q = torch.arange(
                batch_size + 1, device=self.device, dtype=torch.int32
            )
        else:
            extend_lens = forward_batch.extend_seq_lens
            max_q = int(extend_lens.max().item())
            cu_q = torch.zeros(batch_size + 1, device=self.device, dtype=torch.int32)
            cu_q[1:] = torch.cumsum(extend_lens.to(torch.int32), dim=0)

        self.forward_metadata = (block_table, seq_lens, cu_q, max_q, max_k)
        return self.forward_metadata

    def init_forward_metadata(self, forward_batch):
        self._build_metadata(forward_batch)
        qm = self._quant_method()
        if qm is not None and forward_batch.forward_mode.is_extend():
            # Prefill is eager; never reuse the decode graph's static slots.
            qm.graph_slot = None

    def _quant_method(self):
        try:
            return self.token_to_kv_pool.get_kv_cache_quant_method()
        except Exception:
            return None

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        self._build_metadata(forward_batch)
        # Replay path: refresh the static graph block table OUTSIDE the graph.
        # The captured graph reads this tensor; copying req_to_token before the
        # replay makes every new token visible to the captured FA metadata.
        if not in_capture and self._graph_block_table is not None:
            batch_size = forward_batch.seq_lens.shape[0]
            self._graph_block_table[:batch_size].copy_(
                self.req_to_token[
                    forward_batch.req_pool_indices, : self.max_context_len
                ]
            )
            if self._graph_seq_lens is not None:
                self._graph_seq_lens[:batch_size].copy_(
                    forward_batch.seq_lens[:batch_size].to(torch.int32)
                )
        qm = self._quant_method()
        if qm is not None and self._graph_slot is not None:
            qm.graph_slot = self._graph_slot
            if not in_capture:
                n = forward_batch.out_cache_loc.numel()
                self._graph_slot[:n].copy_(forward_batch.out_cache_loc)

    def _forward(self, q, k, v, layer, forward_batch, save_kv_cache):
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.reshape(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)

        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc.for_batch(forward_batch), k, v
            )

        if (
            self._graph_block_table is not None
            and self._graph_seq_lens is not None
            and forward_batch.forward_mode.is_decode()
        ):
            # Decode graph: always read the persistent graph buffers. They are
            # refreshed outside the graph before replay; self.forward_metadata
            # may have been overwritten by an eager init_forward_metadata pass
            # during capture, which would bake an eager tensor into the graph.
            b = forward_batch.seq_lens.shape[0]
            block_table = self._graph_block_table[:b]
            seq_lens = self._graph_seq_lens[:b]
            cu_q = self._graph_cu_q[: b + 1]
            max_q = 1
            max_k = self.max_context_len
        else:
            block_table, seq_lens, cu_q, max_q, max_k = self.forward_metadata
        key = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        val = self.token_to_kv_pool.get_value_buffer(layer.layer_id)

        k_cache_q8 = None
        if key.dtype == torch.uint8:
            hkv = layer.tp_k_head_num
            d = layer.qk_head_dim
            key = key.view(-1, 1, hkv, (d // 32) * 34)
            k_cache_q8 = key
            key_fp16 = None
        else:
            hkv = layer.tp_k_head_num
            d = layer.qk_head_dim
            key = key.view(-1, 1, hkv, d)
            key_fp16 = key

        val = val.view(-1, 1, hkv, layer.v_head_dim)
        # forward_paged wants the fp16 K argument even on the Q8 fast path;
        # it only reads it when key_cache_q8 is None.
        if key_fp16 is None:
            key_arg = val  # placeholder, never read
        else:
            key_arg = key_fp16

        out = forward_paged(
            q,
            key_arg,
            val,
            block_table,
            seq_lens,
            cu_q,
            max_q,
            max_k,
            scale=layer.scaling,
            key_cache_q8=k_cache_q8,
        )
        return out.to(q.dtype)

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache)

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self._forward(q, k, v, layer, forward_batch, save_kv_cache)

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def support_triton(self):
        return False
