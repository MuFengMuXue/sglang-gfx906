# SPDX-License-Identifier: Apache-2.0
"""Loader for the gfx906 LLMM1 skinny fp16 GEMV extension.

The extension is built from ``gfx906_llmm1/llmm1.cu`` plus ``bindings.cpp``;
see ``gfx906_llmm1/build.py``.  It is deliberately separate from the main
SGLang install so the decode fast path can use it without rebuilding all of
sgl-kernel.

Set ``SGLANG_GFX906_LLMM1_LIB`` to override the default library path.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_DEFAULT_LIB = os.path.join(
    os.path.dirname(__file__), "gfx906_llmm1", "gfx906_llmm1.so"
)

_loaded = False
_available = False


def _has_op() -> bool:
    try:
        torch.ops._gfx906_C.LLMM1
        return True
    except Exception:
        return False


def available() -> bool:
    global _loaded, _available
    if os.environ.get("SGLANG_GFX906_LLMM1", "1") == "0":
        return False
    if _loaded:
        return _available
    _loaded = True
    if _has_op():
        _available = True
        return True

    path = os.environ.get("SGLANG_GFX906_LLMM1_LIB", _DEFAULT_LIB)
    if not os.path.exists(path):
        logger.info("gfx906 LLMM1: %s not found, using F.linear", path)
        return False
    try:
        torch.ops.load_library(path)
    except Exception as exc:
        logger.warning("gfx906 LLMM1: failed to load %s: %r", path, exc)
        return False

    _available = _has_op()
    if _available:
        logger.info("gfx906 LLMM1: loaded %s", path)
    return _available


def llmm1(
    weight: torch.Tensor,
    x: torch.Tensor,
    rows_per_block: int = 4,
) -> torch.Tensor:
    """Return ``x @ weight.T`` for a single activation row.

    ``weight`` is ``[out_features, in_features]`` and ``x`` is
    ``[1, in_features]``; the result is ``[1, out_features]``.
    """
    return torch.ops._gfx906_C.LLMM1(weight, x, rows_per_block)
