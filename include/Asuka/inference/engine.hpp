#pragma once
#ifndef ASUKA_INFERENCE_ENGINE_HPP
#define ASUKA_INFERENCE_ENGINE_HPP

#include <cstdint>

namespace Asuka::Inference {
enum class InferenceType : std::uint8_t { ONNXRuntime, TensorRT, OpenVINO, OpenCV_DNN };
}  // namespace Asuka::Inference

#endif  // !ASUKA_INFERENCE_ENGINE_HPP
