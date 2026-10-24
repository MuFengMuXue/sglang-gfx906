#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the gfx906 LLMM1 extension against the current torch.

The kernel is extracted from vllm-gfx906-performance's
``csrc/rocm/skinny_gemms.cu``; only the LLMM1 half is used, so the fp8 /
wvSplitK dependencies are not needed.

Usage:
    python -m sglang.srt.layers.quantization.gfx906_llmm1.build

Environment:
    SGLANG_GFX906_CSRC       vLLM csrc root (default: known local checkout)
    SGLANG_GFX906_LLMM1_LIB  output .so path
    PYTORCH_ROCM_ARCH        target arch (default: gfx906)
"""

import os
import shutil
import tempfile
from pathlib import Path

from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent
DEFAULT_CSRC = "/home/neko/Projects/vllm-gfx906-performance/csrc"


def main() -> None:
    csrc = os.environ.get("SGLANG_GFX906_CSRC", DEFAULT_CSRC)
    out = os.environ.get("SGLANG_GFX906_LLMM1_LIB", str(HERE / "gfx906_llmm1.so"))
    os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx906")

    with tempfile.TemporaryDirectory(prefix="gfx906_llmm1_") as td:
        tmp = Path(td)
        shutil.copy2(HERE / "llmm1.cu", tmp / "llmm1.cu")
        shutil.copy2(HERE / "bindings.cpp", tmp / "bindings.cpp")
        mod = load(
            name="gfx906_llmm1",
            sources=[str(tmp / "bindings.cpp"), str(tmp / "llmm1.cu")],
            extra_include_paths=[csrc],
            extra_cflags=["-O3", "-DUSE_ROCM"],
            extra_cuda_cflags=["-O3", "-DUSE_ROCM", "--offload-arch=gfx906"],
            verbose=True,
        )
        shutil.copy2(mod.__file__, out)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
