#pragma once
#ifndef TENSORRT_LEARNING_PLUGINS_KERNELS_SWISH_CUH
#define TENSORRT_LEARNING_PLUGINS_KERNELS_SWISH_CUH

#include <cuda_runtime.h>

#include "Asuka/tensorrt_macro.hpp"

#ifdef __cplusplus
extern "C" {
#endif

AHRI_TENSORRT_API
void swish(const float* inputs, float* outputs, const int elements, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // !TENSORRT_LEARNING_PLUGINS_KERNELS_SWISH_CUH
