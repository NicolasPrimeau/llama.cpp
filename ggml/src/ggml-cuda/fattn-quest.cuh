#pragma once

#include "common.cuh"

// FlashAttention launcher that the sparse path hands the gathered pages to (the vector kernel).
typedef void (*ggml_cuda_fattn_launch_t)(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Quest sparse decode (see the CPU implementation in ggml-cpu/ops.cpp). Returns false when the dense kernels should
// run instead; the decision depends only on shapes and the environment, so it is stable across CUDA graph replays.
bool ggml_cuda_flash_attn_ext_quest(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_fattn_launch_t attend);
