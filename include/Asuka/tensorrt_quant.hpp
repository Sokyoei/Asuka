#pragma once
#ifndef ASUKA_TENSORRT_QUANT_HPP
#define ASUKA_TENSORRT_QUANT_HPP

#include <NvInfer.h>

/// TensorRT Quantization
/// +--------------------------------------+-------------+
/// | @b nvinfer1::IInt8EntropyCalibrator  |             |
/// | @b nvinfer1::IInt8EntropyCalibrator2 |             |
/// | @b nvinfer1::IInt8MinMaxCalibrator   |             |
/// +--------------------------------------+-------------+
/// TensorRT 10.1+
/// Explicit Quantization

namespace Asuka::TensorRT {
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
private:
public:
    Int8EntropyCalibrator() {}
    ~Int8EntropyCalibrator() {}
};
}  // namespace Asuka::TensorRT

#endif  // !ASUKA_TENSORRT_QUANT_HPP
