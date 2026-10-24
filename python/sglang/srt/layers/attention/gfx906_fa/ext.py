# SPDX-License-Identifier: Apache-2.0
"""Loader for the gfx906 FlashAttention pybind11 extension.

The extension is a plain pybind11 module named ``_gfx906_fa_C`` (compiled by
the vllm-gfx906 tree). It is not a stable-ABI torch op library, but the
PyTorch C++ ABI it uses is compatible across the 2.13/2.14 builds we run.

Set ``SGLANG_GFX906_FA_LIB`` to point at a different build.
"""

import importlib.util
import os
import sys

_DEFAULT_LIB = (
    "/home/neko/Projects/vllm-gfx906-performance/vllm/"
    "_gfx906_fa_C.cpython-312-x86_64-linux-gnu.so"
)


def load_ext():
    name = "_gfx906_fa_C"
    mod = sys.modules.get(name)
    if mod is not None:
        return mod

    path = os.environ.get("SGLANG_GFX906_FA_LIB", _DEFAULT_LIB)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"gfx906 FA extension not found: {path}. "
            "Set SGLANG_GFX906_FA_LIB to the built _gfx906_fa_C*.so"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load gfx906 FA extension from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod
