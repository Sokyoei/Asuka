#pragma once
#ifndef AHRI_ASUKA_INFERENCE_ENGINE_HPP
#define AHRI_ASUKA_INFERENCE_ENGINE_HPP

#include <cstdint>

namespace Ahri::Inference {
enum class InferenceType : std::uint8_t { ONNXRuntime, TensorRT, OpenVINO, OpenCV_DNN };
}  // namespace Ahri::Inference

#endif  // !AHRI_ASUKA_INFERENCE_ENGINE_HPP
