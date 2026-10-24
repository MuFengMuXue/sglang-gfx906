# SPDX-License-Identifier: Apache-2.0
"""gfx906 Q8 FlashAttention wrapper (vendored from vllm-gfx906)."""

from sglang.srt.layers.attention.gfx906_fa.gfx906_fa_paged import forward_paged

__all__ = ["forward_paged"]
