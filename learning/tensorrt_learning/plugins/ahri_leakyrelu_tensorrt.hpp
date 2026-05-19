/**
 * @brief LeakyReLU plugin using nvinfer1::IPluginV3/nvinfer1::IPluginV2DynamicExt
 */

#pragma once
#ifndef AHRI_LEAKYRELU_TENSORRT_HPP
#define AHRI_LEAKYRELU_TENSORRT_HPP

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <fmt/core.h>

#include "Ahri/Asuka/tensorrt_utils.hpp"
#include "kernels/leakyrelu.cuh"

namespace Ahri::TensorRT::Plugin {
static constexpr const char* leakyrelu_plugin_name{"AhriLeakyReLU"};
static constexpr const char* leakyrelu_plugin_namespace{""};
static constexpr const char* leakyrelu_plugin_version{"1"};

namespace V3 {
class AHRI_TENSORRT_API AhriLeakyReLUPlugin : public nvinfer1::IPluginV3,
                                              public nvinfer1::IPluginV3OneBuildV2,
                                              public nvinfer1::IPluginV3OneCore,
                                              public nvinfer1::IPluginV3OneRuntime {
public:
    AhriLeakyReLUPlugin(int32_t alpha) {
        _params.alpha = alpha;

        _plugin_attributes.clear();
        _plugin_attributes.emplace_back(
            nvinfer1::PluginField("alpha", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        _fc.nbFields = static_cast<int32_t>(_plugin_attributes.size());
        _fc.fields = _plugin_attributes.data();
    }

    ~AhriLeakyReLUPlugin() {}

    // nvinfer1::IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override {
        try {
            if (type == nvinfer1::PluginCapabilityType::kBUILD) {
                return static_cast<IPluginV3OneBuild*>(this);
            }
            if (type == nvinfer1::PluginCapabilityType::kRUNTIME) {
                return static_cast<IPluginV3OneRuntime*>(this);
            }
            assert(type == nvinfer1::PluginCapabilityType::kCORE);
            return static_cast<IPluginV3OneCore*>(this);
        } catch (std::exception const& e) {
            AHRI_LOGGER_ERROR("{}", e.what());
        }
        return nullptr;
    };

    nvinfer1::IPluginV3* clone() noexcept override { return new AhriLeakyReLUPlugin(_params.alpha); }

    // nvinfer1::IPluginV3OneBuildV2
    int32_t getAliasedInput(int32_t output_index) noexcept override { return -1; }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in,
                            int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* out,
                            int32_t nb_outputs) noexcept override {
        return 0;
    }

    int32_t getOutputDataTypes(nvinfer1::DataType* output_types,
                               int32_t nb_outputs,
                               const nvinfer1::DataType* input_types,
                               int32_t nb_inputs) const noexcept override {  // 确保输入输出数量符合预期
        assert(input_types[0] == nvinfer1::DataType::kFLOAT);
        output_types[0] = nvinfer1::DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs,
                            int32_t nb_inputs,
                            nvinfer1::DimsExprs const* shape_inputs,
                            int32_t nb_shape_inputs,
                            nvinfer1::DimsExprs* outputs,
                            int32_t nb_outputs,
                            nvinfer1::IExprBuilder& expr_builder) noexcept override {
        const nvinfer1::DimsExprs& input_shape = inputs[0];
        outputs[0].nbDims = 2;
        outputs[0].d[0] = input_shape.d[0];
        return 0;  // 返回 0 表示成功
    }

    bool supportsFormatCombination(int32_t pos,
                                   nvinfer1::DynamicPluginTensorDesc const* in_out,
                                   int32_t nb_inputs,
                                   int32_t nb_outputs) noexcept override {
        switch (pos) {
            case 0:
                return in_out[0].desc.type == nvinfer1::DataType::kFLOAT &&
                       in_out[0].desc.format == nvinfer1::TensorFormat::kLINEAR;
            case 1:
                return in_out[1].desc.type == nvinfer1::DataType::kFLOAT &&
                       in_out[1].desc.format == nvinfer1::TensorFormat::kLINEAR;
            default:
                return false;
        }
        return false;
    }

    int32_t getNbOutputs() const noexcept override { return 1; }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs,
                            int32_t nb_inputs,
                            nvinfer1::DynamicPluginTensorDesc const* outputs,
                            int32_t nb_outputs) const noexcept override {
        return sizeof(_params);
    }

    // int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override { return 0; }

    // int32_t getNbTactics() noexcept override { return 0; }

    // char const* getTimingCacheID() noexcept override { return nullptr; }

    // int32_t getFormatCombinationLimit() noexcept override { return kDEFAULT_FORMAT_COMBINATION_LIMIT; }

    // char const* getMetadataString() noexcept override { return nullptr; }

    // nvinfer1::IPluginV3OneCore
    PLUGIN_GET_NAME_NAMESPACE_VERSION(leakyrelu_plugin_name, leakyrelu_plugin_namespace, leakyrelu_plugin_version)

    // nvinfer1::IPluginV3OneRuntime
    // int32_t setTactic(int32_t tactic) noexcept override { return 0; }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in,
                          int32_t nb_inputs,
                          nvinfer1::PluginTensorDesc const* out,
                          int32_t nb_outputs) noexcept override {
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
                    nvinfer1::PluginTensorDesc const* outputDesc,
                    void const* const* inputs,
                    void* const* outputs,
                    void* workspace,
                    cudaStream_t stream) noexcept override {
        const float* xyz = static_cast<const float*>(inputs[0]);
        int64_t* idxs = static_cast<int64_t*>(outputs[0]);

        // 从PluginTensorDesc中提取维度信息
        int B = static_cast<int>(inputDesc[0].dims.d[0]);
        int N = static_cast<int>(inputDesc[0].dims.d[1]);
        int C = static_cast<int>(inputDesc[0].dims.d[2]);
        int S = static_cast<int>(outputDesc[0].dims.d[1]);

        // 计算所需的内存大小N
        size_t temp_size = B * N;
        size_t temp_size_byte = temp_size * sizeof(float);

        if (workspace == nullptr) {
            return -1;
        }
        cudaError_t err = cudaMemset(workspace, 10000, temp_size_byte);
        if (err != cudaSuccess) {
            return -1;
        }

        return 0;
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override { return &_fc; }

private:
    nvinfer1::PluginFieldCollection _fc;
    std::vector<nvinfer1::PluginField> _plugin_attributes;
    struct {
        float alpha;
    } _params;
};

class AHRI_TENSORRT_API AhriLeakyReLUPluginCreator : public nvinfer1::IPluginCreatorV3One {
public:
    AhriLeakyReLUPluginCreator() {
        _plugin_attributes.clear();
        _plugin_attributes.emplace_back(
            nvinfer1::PluginField("alpha", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        _fc.nbFields = _plugin_attributes.size();
        _fc.fields = _plugin_attributes.data();
    }

    ~AhriLeakyReLUPluginCreator() {}

    nvinfer1::IPluginV3* createPlugin(nvinfer1::AsciiChar const* name,
                                      nvinfer1::PluginFieldCollection const* fc,
                                      nvinfer1::TensorRTPhase phase) noexcept override {
        float alpha = 0;
        std::map<std::string, float*> params{
            {"alpha", &alpha},
        };

        for (int i = 0; i < fc->nbFields; i++) {
            if (params.find(fc->fields[i].name) != params.end()) {
                *params[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
            }
        }

        AHRI_LOGGER_INFO("Creating Scalar plugin with params: alpha={}", alpha);

        return new AhriLeakyReLUPlugin(alpha);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &_fc; }

    PLUGIN_GET_NAME_NAMESPACE_VERSION(leakyrelu_plugin_name, leakyrelu_plugin_namespace, leakyrelu_plugin_version)

private:
    nvinfer1::PluginFieldCollection _fc;
    std::vector<nvinfer1::PluginField> _plugin_attributes;
};
}  // namespace V3

inline namespace V2 {
class AHRI_TENSORRT_API AhriLeakyReLUPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    AhriLeakyReLUPlugin() = delete;

    /**
     * @brief parse constructor
     */
    AhriLeakyReLUPlugin(const std::string& name, float alpha = 0.01f) : _name(name) { _params.alpha = alpha; }

    /**
     * @brief clone and deserialize constructor
     */
    AhriLeakyReLUPlugin(const std::string& name, const void* buffer, size_t size) : _name(name) {
        memcpy(&_params, buffer, sizeof(_params));
    }

    ~AhriLeakyReLUPlugin() {}

    PLUGIN_GET_TYPE_NAMESPACE_VERSION(leakyrelu_plugin_name, leakyrelu_plugin_namespace, leakyrelu_plugin_version)

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

        leakyrelu(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), _params.alpha, elements,
                  stream);

        return 0;
    }

    IPluginV2DynamicExt* clone() const noexcept override {
        auto p = new AhriLeakyReLUPlugin(_name, &_params, sizeof(_params));
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
        float alpha;
    } _params;
};

class AHRI_TENSORRT_API AhriLeakyReLUPluginCreator : public nvinfer1::IPluginCreator {
public:
    AhriLeakyReLUPluginCreator() {
        _plugin_attributes.emplace_back(
            nvinfer1::PluginField("alpha", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        _fc.nbFields = _plugin_attributes.size();
        _fc.fields = _plugin_attributes.data();
    }

    ~AhriLeakyReLUPluginCreator() {}

    nvinfer1::IPluginV2* createPlugin(char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept override {
        float alpha = 0;
        std::map<std::string, float*> params{
            {"alpha", &alpha},
        };

        for (int i = 0; i < fc->nbFields; i++) {
            if (params.find(fc->fields[i].name) != params.end()) {
                *params[fc->fields[i].name] = *reinterpret_cast<const float*>(fc->fields[i].data);
            }
        }

        AHRI_LOGGER_INFO("Creating Scalar plugin with params: alpha={}", alpha);

        return new AhriLeakyReLUPlugin(name, alpha);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const* name,
                                           void const* serial_data,
                                           size_t serial_length) noexcept override {
        return new AhriLeakyReLUPlugin(name, serial_data, serial_length);
    }

    nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &_fc; }

    void setPluginNamespace(char const* plugin_namespace) noexcept override { _namespace = plugin_namespace; }

    PLUGIN_GET_NAME_NAMESPACE_VERSION(leakyrelu_plugin_name, leakyrelu_plugin_namespace, leakyrelu_plugin_version)

private:
    std::string _namespace;
    // 保存这个插件所需要的权重和参数，从 ONNX 中获取，在 parse 时使用
    std::vector<nvinfer1::PluginField> _plugin_attributes;
    // 接受 nvinfer1::PluginField 传入进来的权重和参数，并将信息传递给 Plugin, 内部通过 createPlugin 来创建带参数的
    // Plugin
    nvinfer1::PluginFieldCollection _fc{};
};
}  // namespace V2

// AHRI_REGISTER_TENSORRT_PLUGIN(AhriLeakyReLUPluginCreator);
REGISTER_TENSORRT_PLUGIN(AhriLeakyReLUPluginCreator);
}  // namespace Ahri::TensorRT::Plugin

#endif  // !AHRI_LEAKYRELU_TENSORRT_HPP
