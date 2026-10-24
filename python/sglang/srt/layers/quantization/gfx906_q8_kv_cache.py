# SPDX-License-Identifier: Apache-2.0
"""gfx906 Q8_0 K + fp16 V KV cache (the vllm-gfx906 recipe).

Storage per token/head:
  K: (head_dim / 32) * 34 bytes, q8_0 (fp16 scale + 32 int8 values)
  V: head_dim * 2 bytes, fp16

The gfx906 FA extension reads the packed K directly and keeps V in fp16, so
this preserves V accuracy while halving the K read bandwidth and saving ~25%
of the full-attention KV footprint.
"""

from typing import Optional

import torch

from sglang.srt.layers.attention.gfx906_fa.ext import load_ext
from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    KVCacheQuantMethodBase,
)


class Gfx906Q8KVCacheMethod(KVCacheQuantMethodBase):
    name = "gfx906_q8_0"
    SCALE_BLOCK_SIZE = 32

    def __init__(self, num_layers=None, device=None, page_size=None, **kwargs):
        self.num_layers = num_layers
        self.device = device
        self.page_size = page_size
        # Set by the graph backend: a static int64 slot buffer refreshed
        # outside the graph before each replay.
        self.graph_slot = None

    def create_buffers(self, size, head_num, head_dim, layer_num, device) -> dict:
        if head_dim % 32 != 0:
            raise ValueError(
                f"gfx906 q8_0 KV needs head_dim % 32 == 0, got {head_dim}"
            )
        bytes_per_head = (head_dim // 32) * 34
        k_buffer = [
            torch.zeros(
                (size, head_num, bytes_per_head), dtype=torch.uint8, device=device
            )
            for _ in range(layer_num)
        ]
        v_buffer = [
            torch.zeros(
                (size, head_num, head_dim), dtype=torch.float16, device=device
            )
            for _ in range(layer_num)
        ]
        # The pool's get_{key,value}_buffer() compares store_dtype to the
        # pool dtype and skips the byte-view when they match. Match the model
        # dtype (fp16) so the uint8 K buffer is returned untouched while the
        # fp16 V buffer is also returned as-is.
        return {
            "k_buffer": k_buffer,
            "v_buffer": v_buffer,
            "store_dtype": torch.float16,
        }

    def quantize_and_store(
        self,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        k_scale_buffer: Optional[torch.Tensor],
        v_scale_buffer: Optional[torch.Tensor],
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale=None,
        v_scale=None,
        **kwargs,
    ) -> None:
        ext = load_ext()
        size, hkv, bytes_per_head = k_buffer.shape
        d = cache_k.shape[-1]
        if bytes_per_head != (d // 32) * 34:
            raise ValueError(
                f"q8_0 K buffer last dim {bytes_per_head} != ({d}//32)*34"
            )
        if self.graph_slot is not None:
            slot = self.graph_slot[: loc.numel()]
        else:
            slot = loc.to(torch.int64)
        cache_k = cache_k.reshape(-1, hkv, d).to(torch.float16).contiguous()
        k_cache = k_buffer.view(size, 1, hkv, bytes_per_head)
        ext.reshape_and_cache_q8(cache_k, slot, k_cache)
        # V stays fp16; advanced indexing is a single scatter kernel.
        v_buffer.index_copy_(0, slot, cache_v.reshape(-1, hkv, d).to(torch.float16))

    def dequantize_prev_kv(self, *args, **kwargs):
        raise NotImplementedError("gfx906 q8_0 KV is consumed directly by FA")

    def compute_cell_size(self, head_num, head_dim, num_layers, kv_size) -> int:
        # ``kv_size`` here is torch's element size of the pool dtype (2 for
        # fp16), not a token count: the caller expects bytes per token.
        k_bytes = head_num * ((head_dim // 32) * 34)
        v_bytes = head_num * head_dim * 2
        return num_layers * (k_bytes + v_bytes)
