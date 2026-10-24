# SPDX-License-Identifier: Apache-2.0
"""gfx906 fused ops backed by the vLLM stable-ABI extension.

The vLLM gfx906 build ships a stable-ABI extension that exports the fused
``rms_norm`` / ``fused_add_rms_norm`` / ``silu_and_mul`` kernels.  SGLang's
HIP layernorm/activation paths normally require the ``vllm`` Python package
(``vllm._custom_ops``), which is not installed here, so they fall back to the
multi-kernel native decomposition.  This module loads the extension directly
and exposes the same small wrappers, without importing the rest of vLLM.

Point ``SGLANG_GFX906_OPS_LIB`` (or ``SGLANG_GFX906_GPTQ_LIB``) at a different
build to override the default path.
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_DEFAULT_LIB = (
    "/home/neko/Projects/vllm-gfx906-performance/vllm/_C_stable_libtorch.abi3.so"
)

_loaded = False
_available = False


def _has_ops() -> bool:
    try:
        torch.ops._C.rms_norm
        torch.ops._C.fused_add_rms_norm
        torch.ops._C.silu_and_mul
        return True
    except Exception:
        return False


def available() -> bool:
    """Load the extension once and report whether all fused ops exist."""
    global _loaded, _available
    if os.environ.get("SGLANG_GFX906_FUSED_OPS", "1") == "0":
        return False
    if _loaded:
        return _available
    _loaded = True
    if _has_ops():
        _available = True
        return True

    path = (
        os.environ.get("SGLANG_GFX906_OPS_LIB")
        or os.environ.get("SGLANG_GFX906_GPTQ_LIB")
        or _DEFAULT_LIB
    )
    if not os.path.exists(path):
        logger.info("gfx906 fused ops: %s not found, using native fallback", path)
        return False
    try:
        torch.ops.load_library(path)
    except Exception as exc:
        logger.warning("gfx906 fused ops: failed to load %s: %r", path, exc)
        return False

    _available = _has_ops()
    if _available:
        logger.info("gfx906 fused ops: loaded %s", path)
    return _available


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Out-of-place RMSNorm (the vLLM ``rms_norm`` op writes ``out``)."""
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, weight, eps)
    return out


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place ``residual += x`` followed by RMSNorm of the sum.

    Matches the vLLM ``_custom_ops`` semantics: both ``x`` and ``residual``
    are updated in place and returned for convenience.
    """
    torch.ops._C.fused_add_rms_norm(x, residual, weight, eps)
    return x, residual


def silu_and_mul(
    x: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused ``silu(x[..., :d]) * x[..., d:]``."""
    d = x.shape[-1] // 2
    if out is None:
        out = torch.empty(
            *x.shape[:-1], d, dtype=x.dtype, device=x.device
        )
    torch.ops._C.silu_and_mul(out, x)
    return out
