#pragma once
#ifndef ASUKA_TENSORRT_PLUGIN_CUH
#define ASUKA_TENSORRT_PLUGIN_CUH

#include <cmath>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include "Asuka/tensorrt_utils.hpp"

namespace Asuka::TensorRT::Plugin {
/**
 * @brief Sample plugin impl
 */

///
/// +----------------------------------+-----------------------------------+
/// | ~nvinfer1::IPlugin~              |                                   |
/// | @b nvinfer1::IPluginV2           | 单一 input/output                 |
/// | @b nvinfer1::IPluginV2Ext        | 单一 input and mix output         |
/// | @b nvinfer1::IPluginV2IOExt      | mix input/output implicit batch   |
/// | @b nvinfer1::IPluginV2DynamicExt | mix input/output dynamic shape    |
/// +----------------------------------+-----------------------------------+
///

__global__ void custom_scalar_kernel(const float* inputs,
                                     float* outputs,
                                     const float scalar,
                                     const float scale,
                                     const int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    outputs[index] = (inputs[index] + scalar) * scale;
}

void custom_scalar(const float* inputs,
                   float* outputs,
                   const float scalar,
                   const float scale,
                   const int elements,
                   cudaStream_t stream) {
    dim3 block_size(256, 1, 1);
    dim3 grid_size(std::ceil(float(elements) / 256), 1, 1);
    custom_scalar_kernel<<<grid_size, block_size, 0, stream>>>(inputs, outputs, scalar, scale, elements);
}

#define PLUGIN_NAME "Scalar"
#define PLUGIN_NAMESPACE ""
#define PLUGIN_VERSION "1"

inline namespace V2 {
class CustomScalarPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    CustomScalarPlugin() = delete;

    /**
     * @brief parse constructor
     */
    CustomScalarPlugin(const std::string& name, float scalar, float scale) : _name(name) {
        _params.scalar = scalar;
        _params.scale = scale;
    }

    /**
     * @brief clone and deserialize constructor
     */
    CustomScalarPlugin(const std::string& name, const void* buffer, size_t size) : _name(name) {
        memcpy(&_params, buffer, sizeof(_params));
    }

    ~CustomScalarPlugin() {}

    PLUGIN_GET_TYPE_NAMESPACE_VERSION(PLUGIN_NAME, PLUGIN_NAMESPACE, PLUGIN_VERSION)

    int32_t getNbOutputs() const noexcept override { return 1; }

    size_t getSerializationSize() const noexcept override { return sizeof(_params); }

    nvinfer1::DimsExprs getOutputDimensions(int32_t output_index,
                                            nvinfer1::DimsExprs const* inputs,
                                            int32_t nb_inputs,
                                            nvinfer1::IExprBuilder& expr_builder) noexcept override {
        return inputs[0];
    }

    nvinfer1::DataType getOutputDataType(int32_t index,
                                         nvinfer1::DataType const* input_types,
                                         int32_t nb_inputs) const noexcept override {
        return input_types[0];
    }

    void setPluginNamespace(char const* plugin_namespace) noexcept override { _namespace = plugin_namespace; }

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs,
                            int32_t nb_inputs,
                            nvinfer1::PluginTensorDesc const* outputs,
                            int32_t nb_outputs) const noexcept override {
        return 0;
    }

    int32_t initialize() noexcept override { return 0; }

    void terminate() noexcept override {}

    void serialize(void* buffer) const noexcept override { memcpy(buffer, &_params, sizeof(_params)); }

    void destroy() noexcept override { delete this; }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* input_desc,
                    nvinfer1::PluginTensorDesc const* output_desc,
                    void const* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override {
        int elements = 1;
        for (int i = 0; i < input_desc[0].dims.nbDims; i++) {
            elements *= input_desc[0].dims.d[i];
        }

        custom_scalar(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), _params.scalar,
                      _params.scale, elements, stream);

        return 0;
    }

    IPluginV2DynamicExt* clone() const noexcept override {
        auto p = new CustomScalarPlugin(_name, &_params, sizeof(_params));
        p->setPluginNamespace(_namespace.c_str());
        return p;
    }

    bool supportsFormatCombination(int32_t pos,
                                   nvinfer1::PluginTensorDesc const* in_out,
                                   int32_t nb_inputs,
                                   int32_t nb_outputs) noexcept override {
        switch (pos) {
            case 0:
                return in_out[0].type == nvinfer1::DataType::kFLOAT &&
                       in_out[0].format == nvinfer1::TensorFormat::kLINEAR;
            case 1:
                return in_out[1].type == nvinfer1::DataType::kFLOAT &&
                       in_out[1].format == nvinfer1::TensorFormat::kLINEAR;
            default:
                return false;
        }
        return false;
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                         int32_t nb_inputs,
                         nvinfer1::DynamicPluginTensorDesc const* out,
                         int32_t nb_outputs) noexcept override {}

    virtual void attachToContext(cudnnContext* /*cudnn*/,
                                 cublasContext* /*cublas*/,
                                 nvinfer1::IGpuAllocator* /*allocator*/) noexcept override {}

    void detachFromContext() noexcept override {}

private:
    const std::string _name;
    std::string _namespace;
    struct {
        float scalar;
        float scale;
    } _params;
};

class CustomScalarPluginCreator : public nvinfer1::IPluginCreator {
public:
    CustomScalarPluginCreator() {
        _plugin_attributes.emplace_back(
            nvinfer1::PluginField("scalar", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        _plugin_attributes.emplace_back(
            nvinfer1::PluginField("scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        _fc.nbFields = _plugin_attributes.size();
        _fc.fields = _plugin_attributes.data();
    }

    ~CustomScalarPluginCreator() {}

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override {
        float scalar = 0;
        float scale = 0;
        std::map<std::string, float*> params{
            {"scalar", &scalar},
            { "scale",  &scale},
        };

        for (int i = 0; i < fc->nbFields; i++) {
            if (params.find(fc->fields[i].name) != params.end()) {
                *params[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
            }
        }

        AHRI_LOGGER_INFO("Creating Scalar plugin with params: scalar={}, scale={}", scalar, scale);

        return new CustomScalarPlugin(name, scalar, scale);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* name,
                                           void const* serial_data,
                                           size_t serial_length) noexcept override {
        return new CustomScalarPlugin(name, serial_data, serial_length);
    }

    nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &_fc; }

    void setPluginNamespace(char const* plugin_namespace) noexcept override { _namespace = plugin_namespace; }

    PLUGIN_GET_NAME_NAMESPACE_VERSION(PLUGIN_NAME, PLUGIN_NAMESPACE, PLUGIN_VERSION)

private:
    std::string _namespace;
    // 保存这个插件所需要的权重和参数，从 ONNX 中获取，在 parse 时使用
    std::vector<nvinfer1::PluginField> _plugin_attributes;
    // 接受 nvinfer1::PluginField 传入进来的权重和参数，并将信息传递给 Plugin, 内部通过 createPlugin 来创建带参数的
    // Plugin
    nvinfer1::PluginFieldCollection _fc{};
};

REGISTER_TENSORRT_PLUGIN(CustomScalarPluginCreator);
}  // namespace V2
}  // namespace Asuka::TensorRT::Plugin

#endif  // !ASUKA_TENSORRT_PLUGIN_CUH
