#pragma once
#ifndef ASUKA_TENSORRT_MACRO_HPP
#define ASUKA_TENSORRT_MACRO_HPP

#include "Ahri/Ahri.cuh"

namespace Asuka::TensorRT {
#define ENGINE_EXTENSION ".engine"
#define TIMING_CACHE_EXTENSION ".timing_cache"
#define AHRI_TENSORRT_API AHRI_API
}  // namespace Asuka::TensorRT

#endif  // !ASUKA_TENSORRT_MACRO_HPP
