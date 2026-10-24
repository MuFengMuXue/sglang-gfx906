// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: vLLM contributors
#include <torch/extension.h>

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME gfx906_llmm1
#endif

torch::Tensor LLMM1(at::Tensor& in_a, at::Tensor& in_b,
                    const int64_t rows_per_block);

namespace {

torch::Tensor llmm1(at::Tensor a, at::Tensor b, int64_t rows_per_block) {
  return LLMM1(a, b, rows_per_block);
}

}  // namespace

TORCH_LIBRARY(_gfx906_C, m) {
  m.def("LLMM1(Tensor a, Tensor b, int rows_per_block) -> Tensor", &llmm1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
