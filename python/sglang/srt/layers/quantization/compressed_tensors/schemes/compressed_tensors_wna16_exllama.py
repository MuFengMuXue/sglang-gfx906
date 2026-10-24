# SPDX-License-Identifier: Apache-2.0
#
# gfx906 (MI50/MI60) W4A16 path using the exllama GPTQ kernels.
#
# The Triton dequant+GEMM in compressed_tensors_wna16_triton.py is correct but
# ~7-17x slower than the tuned exllama kernel on gfx906 at decode (M=1).
# This scheme repacks the compressed-tensors `pack-quantized` layout into the
# layout expected by `torch.ops._C.gptq_gemm`:
#
#   weight_packed [N, K/8]  -> [K/8, N] int32, then gptq_shuffle(..., 4)
#   weight_scale  [N, G]    -> [G, N] fp16
#   weight_zp     [N/8, G]  -> [G, N/8] int32
#
# The kernel is the one from vllm-gfx906-performance
# (csrc/quantization/gptq/q_gemm.cu). It is a stable-ABI extension, so the
# prebuilt `.so` loads under torch 2.14 as well. Set SGLANG_GFX906_GPTQ_LIB to
# point at a different build; otherwise the known local build is used.

import os
from typing import Optional

import torch

from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsLinearScheme,
)

_DEFAULT_LIB = (
    "/home/neko/Projects/vllm-gfx906-performance/vllm/_C_stable_libtorch.abi3.so"
)

_loaded = False


def _load_gptq_lib() -> bool:
    """Load the GPTQ extension once. Returns True when gptq_gemm is usable."""
    global _loaded
    if _loaded:
        return True
    path = os.environ.get("SGLANG_GFX906_GPTQ_LIB", _DEFAULT_LIB)
    try:
        gptq = getattr(getattr(torch.ops, "_C", None), "gptq_gemm", None)
    except Exception:
        gptq = None
    if gptq is None:
        try:
            torch.ops.load_library(path)
        except Exception:
            return False
        try:
            torch.ops._C.gptq_gemm
            torch.ops._C.gptq_shuffle
        except Exception:
            return False
    _loaded = True
    return True


def gfx906_gptq_available() -> bool:
    return _load_gptq_lib()


class CompressedTensorsWNA16Exllama(CompressedTensorsLinearScheme):
    """W4A16 group-wise symmetric/asymmetric via the exllama GPTQ kernel."""

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: Optional[int],
        symmetric: Optional[bool],
        actorder=None,
    ):
        self.strategy = strategy
        self.num_bits = num_bits
        self.symmetric = symmetric if symmetric is not None else True
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == "group"
        self.pack_factor = 32 // num_bits

    def get_name(self) -> str:
        return "wNa16_exllama"

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.group_size

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )
        weight_scale = GroupQuantScaleParameter(
            output_dim=0,
            input_dim=1,
            weight_loader=weight_loader,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // group_size,
                dtype=torch.float16,
            ),
        )
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64), weight_loader=weight_loader
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            qzeros = PackedvLLMParameter(
                input_dim=1,
                output_dim=0,
                packed_dim=0,
                packed_factor=self.pack_factor,
                weight_loader=weight_loader,
                data=torch.zeros(
                    output_size_per_partition // self.pack_factor,
                    input_size_per_partition // group_size,
                    dtype=torch.int32,
                ),
            )
            layer.register_parameter("weight_zero_point", qzeros)
        if self.has_g_idx:
            raise NotImplementedError("g_idx not supported by the gfx906 GPTQ path")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not _load_gptq_lib():
            raise RuntimeError(
                "gfx906 GPTQ extension is not available; set SGLANG_GFX906_GPTQ_LIB"
            )
        # [N, K/8] -> [K/8, N], then shuffle into exllama's nibble order.
        wq = layer.weight_packed.data.t().contiguous()
        torch.ops._C.gptq_shuffle(wq, 4)
        layer.weight_packed = torch.nn.Parameter(wq, requires_grad=False)

        # [N, G] -> [G, N], fp16 (the kernel reads half scales).
        ws = layer.weight_scale.data.t().contiguous().to(torch.float16)
        layer.weight_scale = torch.nn.Parameter(ws, requires_grad=False)

        if self.symmetric:
            raise NotImplementedError(
                "gfx906 GPTQ path currently expects explicit zero points"
            )
        # [N/8, G] -> [G, N/8]
        wz = layer.weight_zero_point.data.t().contiguous()
        layer.weight_zero_point = torch.nn.Parameter(wz, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float16).contiguous()
        out = torch.ops._C.gptq_gemm(
            x2,
            layer.weight_packed,
            layer.weight_zero_point,
            layer.weight_scale,
            True,  # use_exllama
            True,  # use_v2_format (zero points stored directly)
            4,
        )
        out = out.reshape(*x.shape[:-1], out.shape[-1])
        if bias is not None:
            out = out + bias
        return out.to(x.dtype)
