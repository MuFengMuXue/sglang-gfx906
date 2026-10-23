# SPDX-License-Identifier: Apache-2.0
# Adapted from sglang-mi50 (https://github.com/NnnHU/sglang-mi50).
#
# gfx906 shim scheme: pure-Triton W4A16 (pack-quantized) for
# compressed-tensors checkpoints. Replaces the Marlin-based
# CompressedTensorsWNA16 on GPUs without CUDA marlin kernels
# (e.g. AMD MI50 / gfx906).

from typing import Optional

import os

import torch
import triton
import triton.language as tl

from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsLinearScheme,
)


# nibble order inside each int32 for AWQ-style packing: [0,4,1,5,2,6,3,7]
# order[j] == ((j & 1) << 2) | (j >> 1)


@triton.jit
def _ct_w4a16_gemm_kernel(
    a_ptr, qw_ptr, sc_ptr, qz_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_qwn, stride_qwk,
    stride_scn, stride_sck,
    stride_qzn, stride_qzk,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    kk = tl.arange(0, GROUP_SIZE)
    # inverse awq nibble order of (kk % 8): [0,2,4,6,1,3,5,7]
    j = kk % 8
    shift_k = 4 * j  # compressed-tensors pack-quantized: plain sequential nibble order

    for k0 in range(0, K, GROUP_SIZE):
        g = k0 // GROUP_SIZE

        # scales [BLOCK_N]
        sc = tl.load(
            sc_ptr + offs_n * stride_scn + g * stride_sck,
            mask=n_mask, other=0.0,
        ).to(tl.float32)

        # zero points: packed along N, 8 int4 per int32
        qz = tl.load(
            qz_ptr + (offs_n // 8) * stride_qzn + g * stride_qzk,
            mask=n_mask, other=0,
        )
        shift_n = 4 * (offs_n % 8)
        # compressed-tensors pack-quantized stores zero points directly (the
        # library dequant is (w - zp) * scale); do NOT bias them by +1.
        zero = ((qz >> shift_n) & 0xF).to(tl.float32)

        # unpack weights [BLOCK_N, GROUP_SIZE]
        wp_idx = (k0 + kk) // 8
        qw = tl.load(
            qw_ptr + offs_n[:, None] * stride_qwn + wp_idx[None, :] * stride_qwk,
            mask=n_mask[:, None], other=0,
        )
        wq4 = (qw >> shift_k[None, :]) & 0xF
        w = (wq4.to(tl.float32) - zero[:, None]) * sc[:, None]  # [BN, G]
        w = tl.trans(w).to(tl.float16)  # [G, BN]

        # activations [BLOCK_M, GROUP_SIZE]
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + (k0 + kk)[None, :] * stride_ak,
            mask=m_mask[:, None], other=0.0,
        ).to(tl.float16)

        acc += tl.dot(a, w)

    c = acc.to(c_ptr.dtype.element_ty)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        c += bias.to(c_ptr.dtype.element_ty)[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=m_mask[:, None] & n_mask[None, :])


_ACT_CLAMP = float(os.environ.get("SGLANG_FP16_ACT_CLAMP", "0") or 0)


def ct_w4a16_gemm(x, qweight, scales, qzeros, group_size: int, bias=None):
    M, K = x.shape
    N = qweight.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)
    import os as _os
    _clamp = float(_os.environ.get("SGLANG_FP16_ACT_CLAMP", "0") or 0)
    BM = 16 if M <= 16 else 64
    BN = 64
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    _ct_w4a16_gemm_kernel[grid](
        x, qweight, scales, qzeros, out, bias if bias is not None else x,
        M, N, K,
        x.stride(0), x.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0), scales.stride(1),
        qzeros.stride(0), qzeros.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        GROUP_SIZE=group_size,
        BLOCK_M=BM,
        BLOCK_N=BN,
        num_warps=4,
    )
    if _clamp > 0:
        out = out.clamp(-_clamp, _clamp)
    return out


class CompressedTensorsWNA16Triton(CompressedTensorsLinearScheme):
    """W4A16 scheme using a fused dequant+gemm triton kernel (gfx906)."""

    def __init__(self, strategy: str, num_bits: int, group_size: Optional[int],
                 symmetric: Optional[bool], actorder=None):
        self.strategy = strategy
        self.num_bits = num_bits
        self.symmetric = symmetric if symmetric is not None else True
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == "group"
        self.pack_factor = 32 // num_bits

    def get_name(self) -> str:
        return "wNa16_triton"

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight_packed = torch.nn.Parameter(layer.weight_packed.data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(layer.weight_scale.data, requires_grad=False)
        if hasattr(layer, "weight_zero_point"):
            layer.weight_zero_point = torch.nn.Parameter(layer.weight_zero_point.data, requires_grad=False)

    def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                       output_partition_sizes: list[int], input_size: int,
                       output_size: int, params_dtype: torch.dtype,
                       **extra_weight_attrs):
        weight_loader = extra_weight_attrs.get("weight_loader")
        output_size_per_partition = sum(output_partition_sizes)
        group_size = self.group_size

        weight = PackedvLLMParameter(
            input_dim=1, output_dim=0, weight_loader=weight_loader,
            packed_factor=self.pack_factor, packed_dim=1,
            data=torch.empty(output_size_per_partition,
                             input_size_per_partition // self.pack_factor,
                             dtype=torch.int32))

        weight_scale_args = {
            "weight_loader": weight_loader,
            "data": torch.empty(output_size_per_partition,
                                input_size_per_partition // group_size,
                                dtype=torch.float16),
        }
        zeros_args = {
            "weight_loader": weight_loader,
            "data": torch.zeros(output_size_per_partition // self.pack_factor,
                                input_size_per_partition // group_size,
                                dtype=torch.int32),
        }

        weight_scale = GroupQuantScaleParameter(output_dim=0, input_dim=1,
                                                **weight_scale_args)
        if not self.symmetric:
            qzeros = PackedvLLMParameter(input_dim=1, output_dim=0,
                                         packed_dim=0,
                                         packed_factor=self.pack_factor,
                                         **zeros_args)

        weight_shape = BasevLLMParameter(data=torch.empty(2, dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)
        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(
                data=torch.empty(input_size_per_partition, dtype=torch.int32),
                input_dim=0, weight_loader=weight_loader)
            layer.register_parameter("weight_g_idx", weight_g_idx)

    def apply(self, layer: torch.nn.Module, x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1]).to(torch.float16)
        out = ct_w4a16_gemm(
            x2,
            layer.weight_packed,
            layer.weight_scale,
            layer.weight_zero_point,
            self.group_size,
            bias,
        )
        out = out.to(x.dtype)
        return out.reshape(*x.shape[:-1], out.shape[-1])

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        return self.apply(layer, x, bias)
