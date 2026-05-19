#pragma once
#ifndef TENSORRT_LEARN_PLUGINS_KERNELS_SWISH_CUH
#define TENSORRT_LEARN_PLUGINS_KERNELS_SWISH_CUH

#include <cuda_runtime.h>

#include "Ahri/Asuka/tensorrt_macro.hpp"

#ifdef __cplusplus
extern "C" {
#endif

AHRI_TENSORRT_API
void swish(const float* inputs, float* outputs, const int elements, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif  // !TENSORRT_LEARN_PLUGINS_KERNELS_SWISH_CUH
