/**
 * @file tensorrt_utils.cuh
 * @date 2024/05/16
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef AHRI_ASUKA_TENSORRT_UTILS_HPP
#define AHRI_ASUKA_TENSORRT_UTILS_HPP

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Ahri/Asuka.hpp"

// #ifdef __NVCC__
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#ifdef ASUKA_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "Ahri/Ahri.cuh"
#include "Ahri/Ceceilia/utils/logger_utils.hpp"
#include "Ahri/Asuka/tensorrt_macro.hpp"

#ifndef AHRI_CXX17
#error "requires compiler has C++17 or later."
#endif

namespace Ahri::TensorRT {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Declaration
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void print_network(nvinfer1::INetworkDefinition& network, nvinfer1::ICudaEngine& engine, bool optimized);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TensorRT Logger
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief tensorrt logger
 */
class TensorRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case Severity::kVERBOSE:
                AHRI_LOGGER_DEBUG(msg);
                break;
            case Severity::kINFO:
                AHRI_LOGGER_INFO(msg);
                break;
            case Severity::kWARNING:
                AHRI_LOGGER_WARN(msg);
                break;
            case Severity::kERROR:
                AHRI_LOGGER_ERROR(msg);
                break;
            case Severity::kINTERNAL_ERROR:
                AHRI_LOGGER_CRITICAL(msg);
                break;
            default:
                AHRI_LOGGER_INFO(msg);
        }
    }
};

inline TensorRTLogger trtlogger;

#define AHRI_TENSORRT_LOGGER_VERBOSE(...) trtlogger.log(Severity::kVERBOSE, fmt::format(__VA_ARGS__))
#define AHRI_TENSORRT_LOGGER_INFO(...) trtlogger.log(Severity::kINFO, fmt::format(__VA_ARGS__))
#define AHRI_TENSORRT_LOGGER_WARNING(...) trtlogger.log(Severity::kWARNING, fmt::format(__VA_ARGS__))
#define AHRI_TENSORRT_LOGGER_ERROR(...) trtlogger.log(Severity::kERROR, fmt::format(__VA_ARGS__))
#define AHRI_TENSORRT_LOGGER_INTERNAL_ERROR(...) trtlogger.log(Severity::kINTERNAL_ERROR, fmt::format(__VA_ARGS__))

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Inference
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class TensorRTModel {
public:
    TensorRTModel(std::filesystem::path onnx_path) : _onnx_path(onnx_path) {
        if (!std::filesystem::exists(_onnx_path)) {
            AHRI_LOGGER_ERROR("{} not found.", _onnx_path);
        }
        _engine_path = _onnx_path;
        _engine_path.replace_extension(ENGINE_EXTENSION);
        _timing_cache_path = _engine_path;
        _timing_cache_path.replace_extension(TIMING_CACHE_EXTENSION);

        cudaStreamCreate(&_stream);
    }

    ~TensorRTModel() { cudaStreamDestroy(_stream); }

    /**
     * @brief Build TensorRT engine from ONNX model.
     * @param build_flag Build flag @see nvinfer1::BuilderFlag.
     * @param enable_timing_cache Enable timing cache @see nvinfer1::ITimingCache.
     * @return True if build success.
     */
    bool build(nvinfer1::BuilderFlag build_flag = nvinfer1::BuilderFlag::kFP16, bool enable_timing_cache = true) {
        // initLibNvInferPlugins(&trtlogger, "samples");

        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtlogger));
        constexpr auto explicit_batch =
            1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));

        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trtlogger));
        parser->parseFromFile(_onnx_path.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR));
        for (int i = 0; i < parser->getNbErrors(); i++) {
            AHRI_LOGGER_ERROR("Parser ONNX error: {}", parser->getError(i)->desc());
        }
        AHRI_LOGGER_INFO("Parser ONNX success");

        // Builder Config
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
#if NV_TENSORRT_MAJOR >= 10
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
#else
        config->setMaxWorkspaceSize(1 << 28);
#endif
        config->setFlag(build_flag);
        config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);

        // Timing Cache
        std::unique_ptr<nvinfer1::ITimingCache> timing_cache;
        if (enable_timing_cache) {
            std::vector<uint8_t> timing_cache_data;

            // 尝试加载现有的 timing cache
            if (std::filesystem::exists(_timing_cache_path)) {
                timing_cache_data = load_binary_file(_timing_cache_path);
                if (!timing_cache_data.empty()) {
                    AHRI_LOGGER_INFO("Loaded timing cache from {}", _timing_cache_path);
                } else {
                    AHRI_LOGGER_WARN("Failed to load timing cache from {}, will create a new one", _timing_cache_path);
                }
            }

            // 创建或使用现有的 timing cache
            if (!timing_cache_data.empty()) {
                timing_cache.reset(config->createTimingCache(timing_cache_data.data(), timing_cache_data.size()));
            } else {
                timing_cache.reset(config->createTimingCache(nullptr, 0));
            }

            if (timing_cache) {
                config->setTimingCache(*timing_cache, false);
            }
        }

        auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
        auto model_stream = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

        // 保存 timing cache
        if (enable_timing_cache && timing_cache) {
            auto timing_cache_data = timing_cache->serialize();
            std::ofstream timing_cache_file(_timing_cache_path, std::ios::binary);
            if (timing_cache_file) {
                timing_cache_file.write(reinterpret_cast<const char*>(timing_cache_data->data()),
                                        timing_cache_data->size());
                timing_cache_file.close();
                AHRI_LOGGER_INFO("Saved timing cache to {}", _timing_cache_path);
            }
        }

        // save to file
        std::ofstream engine_file(_engine_path, std::ios::binary);
        if (!engine_file) {
            AHRI_LOGGER_ERROR("Could not open {}", _engine_path);
            return false;
        }
        engine_file.write(reinterpret_cast<const char*>(model_stream->data()), model_stream->size());
        engine_file.close();

        // print network
        print_network(*network, *engine, false);
        print_network(*network, *engine, true);

        return true;
    }

    std::vector<float> inference(std::vector<float>& input_host) {
        if (!_engine_loaded) {
            this->initialize_engine();
        }

        // alloc gpu memory
        // std::vector<float> input_host(input_size, 1.0);
        std::vector<float> output_host(_output_size, 0.0);
        void* input_device;
        void* output_device;
        cudaMalloc(&input_device, _input_size * sizeof(float));
        cudaMalloc(&output_device, _output_size * sizeof(float));

        _context->setInputTensorAddress(_input_name.c_str(), input_device);
        _context->setOutputTensorAddress(_output_name.c_str(), output_device);

        CUDA_CHECK(cudaMemcpyAsync(input_device, input_host.data(), _input_size * sizeof(float), cudaMemcpyHostToDevice,
                                   _stream));

        // void* bindings[] = {input_device, output_device};
        // context->enqueueV2(bindings, 0, nullptr);
        _context->enqueueV3(_stream);
        // cudaStreamSynchronize(stream);

        CUDA_CHECK(cudaMemcpyAsync(output_host.data(), output_device, _output_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, _stream));
        cudaStreamSynchronize(_stream);

        CUDA_CHECK(cudaFree(input_device));
        CUDA_CHECK(cudaFree(output_device));
        input_device = nullptr;
        output_device = nullptr;

        return output_host;
    }

#ifdef ASUKA_USE_OPENCV
    std::vector<float> inference(const cv::Mat& image) {
        if (!_engine_loaded) {
            this->initialize_engine();
        }

        std::vector<float> input_host(_input_size);
        std::memcpy(input_host.data(), image.data, image.total() * image.channels() * sizeof(float));
        return inference(input_host);
    }
#endif  // ASUKA_USE_OPENCV

// function alias
#define onnx2engine build
#define detect inference
#define infer inference

    bool initialize_engine() {
        if (!_engine_loaded) {
            AHRI_LOGGER_INFO("initialize engine");
            std::vector<uint8_t> model_data = load_binary_file(_engine_path);

            auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtlogger));
            _engine = std::unique_ptr<nvinfer1::ICudaEngine>(
                runtime->deserializeCudaEngine(model_data.data(), model_data.size()));

            // input output names
            _input_name = _engine->getIOTensorName(0);
            _output_name = _engine->getIOTensorName(1);
            AHRI_LOGGER_INFO("Input name: {}, output name: {}", _input_name, _output_name);

            _context = std::unique_ptr<nvinfer1::IExecutionContext>(_engine->createExecutionContext());

#if NV_TENSORRT_MAJOR >= 10
            nvinfer1::Dims input_dim_shapes = _engine->getTensorShape(_input_name.c_str());
            nvinfer1::Dims output_dim_shapes = _engine->getTensorShape(_output_name.c_str());
#else
            auto input_dims = _context->getBindingDimensions(0);
            auto output_dims = _context->getBindingDimensions(1);

            int input_index = _engine->getBindingIndex(_input_name.c_str());
            int output_index = _engine->getBindingIndex(_output_name.c_str());

            nvinfer1::Dims input_dim_shapes = _engine->getBindingDimensions(input_index);
            nvinfer1::Dims output_dim_shapes = _engine->getBindingDimensions(output_index);
#endif
            // check input and output dimensions
            if (input_dim_shapes.nbDims == -1 || output_dim_shapes.nbDims == -1) {
                throw std::runtime_error("Invalid input or output dimensions");
            }

            nvinfer1::Dims4 input_dims{1, input_dim_shapes.d[1], input_dim_shapes.d[2], input_dim_shapes.d[3]};
            nvinfer1::Dims4 output_dims{1, output_dim_shapes.d[1], output_dim_shapes.d[2], output_dim_shapes.d[3]};

            _context->setInputShape(_input_name.c_str(), input_dims);

            // 计算输入和输出数据大小
            for (int i = 0; i < input_dim_shapes.nbDims; ++i) {
                _input_size *= input_dim_shapes.d[i];
            }
            for (int i = 0; i < output_dim_shapes.nbDims; ++i) {
                _output_size *= output_dim_shapes.d[i];
            }

            AHRI_LOGGER_INFO("Input size: {}, output size: {}", _input_size, _output_size);
        }

        _engine_loaded = true;
        return true;
    }

private:
    std::vector<uint8_t> load_binary_file(std::filesystem::path path) {
        std::vector<uint8_t> data;
        std::ifstream file{path, std::ios::in | std::ios::binary};

        file.seekg(0, file.end);
        const auto size = file.tellg();
        file.seekg(0, file.beg);

        data.resize(size);
        file.read(reinterpret_cast<char*>(data.data()), size);
        file.close();
        return data;
    }

    std::map<std::string, nvinfer1::Weights> load_weights(std::filesystem::path path) {
        std::ifstream file{path};
        int32_t size;
        std::map<std::string, nvinfer1::Weights> weights;
        file >> size;
        if (size < 0) {
            AHRI_LOGGER_ERROR("No weights found in {}", path);
        }

        while (size > 0) {
            nvinfer1::Weights weight;
            std::string name;
            int weight_length;

            file >> name;
            file >> std::dec >> weight_length;

            uint32_t* values = new uint32_t[weight_length];
            for (int i = 0; i < weight_length; i++) {
                file >> std::hex >> values[i];
            }

            weight.type = nvinfer1::DataType::kFLOAT;
            weight.count = weight_length;
            weight.values = values;

            weights[name] = weight;
            size--;
        }

        return weights;
    }

private:
    std::filesystem::path _onnx_path;
    std::filesystem::path _engine_path;
    std::filesystem::path _timing_cache_path;
    cudaStream_t _stream;
    nvinfer1::Dims _input_dims;
    nvinfer1::Dims _output_dims;
    std::unique_ptr<nvinfer1::ICudaEngine> _engine;
    nvinfer1::DataType _data_type;
    bool _engine_loaded = false;
    std::unique_ptr<nvinfer1::IExecutionContext> _context;
    size_t _input_size = 1;
    size_t _output_size = 1;
    std::string _input_name;
    std::string _output_name;
};

using Model = TensorRTModel;

namespace examples {
inline bool build_weights(nvinfer1::INetworkDefinition& network, std::map<std::string, nvinfer1::Weights> weights) {
    auto data = network.addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 1, 1, 5});
#if NV_TENSORRT_MAJOR >= 10
    auto linear_weight = network.addConstant(nvinfer1::Dims{}, weights["linear.weight"]);
    auto linear_weight_tensor = linear_weight->getOutput(0);
    auto fc = network.addMatrixMultiply(*data, nvinfer1::MatrixOperation::kNONE, *linear_weight_tensor,
                                        nvinfer1::MatrixOperation::kTRANSPOSE);
#else
    auto fc = network.addFullyConnected(*data, 1, weights["linear.weight"], {});
#endif
    fc->setName("linear1");
    fc->getOutput(0)->setName("output0");
    network.markOutput(*fc->getOutput(0));
    return true;
}
}  // namespace examples

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Print helper
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inline std::string tensor_shape(nvinfer1::ITensor* tensor) {
    std::string shape;
    auto dims = tensor->getDimensions();
    for (int i = 0; i < dims.nbDims; i++) {
        if (i == 0) {
            shape = fmt::format("{}", dims.d[i]);
        } else {
            shape = fmt::format("{}x{}", shape, dims.d[i]);
        }
    }
    return shape;
}

inline void print_network(nvinfer1::INetworkDefinition& network, nvinfer1::ICudaEngine& engine, bool optimized) {
    int input_count = network.getNbInputs();
    int output_count = network.getNbOutputs();
    std::string layer_info;

    for (int i = 0; i < input_count; i++) {
        auto input = network.getInput(i);
        if (input != nullptr) {
            AHRI_LOGGER_INFO("Input info: {}: {}", input->getName(), tensor_shape(input));
        }
    }
    for (int i = 0; i < output_count; i++) {
        auto output = network.getOutput(i);
        if (output != nullptr) {
            AHRI_LOGGER_INFO("Output info: {}: {}", output->getName(), tensor_shape(output));
        }
    }

    int layer_count = optimized ? engine.getNbLayers() : network.getNbLayers();
    AHRI_LOGGER_INFO("network layer {}", layer_count);

    if (!optimized) {
        for (int i = 0; i < layer_count; i++) {
            auto layer = network.getLayer(i);
            if (layer == nullptr) {
                continue;
            }

            int nbInputs = layer->getNbInputs();
            int nbOutputs = layer->getNbOutputs();

            std::string input_shapes = "";
            std::string output_shapes = "";

            // 收集所有输入的形状信息
            for (int j = 0; j < nbInputs; j++) {
                auto input = layer->getInput(j);
                if (input != nullptr) {
                    if (j > 0) {
                        input_shapes += ", ";
                    }
                    input_shapes += tensor_shape(input);
                }
            }

            // 收集所有输出的形状信息
            for (int j = 0; j < nbOutputs; j++) {
                auto output = layer->getOutput(j);
                if (output != nullptr) {
                    if (j > 0) {
                        output_shapes += ", ";
                    }
                    output_shapes += tensor_shape(output);
                }
            }

            AHRI_LOGGER_INFO("Layer info: {}: inputs[{}] outputs[{}] {}", layer->getName(),
                             input_shapes.empty() ? "none" : input_shapes,
                             output_shapes.empty() ? "none" : output_shapes, static_cast<int>(layer->getPrecision()));
        }
    } else {
        auto inspector = std::unique_ptr<nvinfer1::IEngineInspector>(engine.createEngineInspector());
        for (int i = 0; i < layer_count; i++) {
            try {
                AHRI_LOGGER_INFO("Layer info: {}",
                                 inspector->getLayerInformation(i, nvinfer1::LayerInformationFormat::kJSON));
            } catch (...) {
                AHRI_LOGGER_WARN("Failed to get layer information for layer {}", i);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Plugin
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define PLUGIN_GET_TYPE_OR_NAME_NAMESPACE_VERSION(name, namespace, version, Type_or_Name) \
    nvinfer1::AsciiChar const* getPlugin##Type_or_Name() const noexcept override {        \
        return name;                                                                      \
    }                                                                                     \
    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {             \
        return namespace;                                                                 \
    }                                                                                     \
    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {               \
        return version;                                                                   \
    }

/**
 * @details for @c nvinfer1::IPluginV2DynamicExt
 */
#define PLUGIN_GET_TYPE_NAMESPACE_VERSION(name, namespace, version) \
    PLUGIN_GET_TYPE_OR_NAME_NAMESPACE_VERSION(name, namespace, version, Type)

/**
 * @details for @c nvinfer1::IPluginCreator
 */
#define PLUGIN_GET_NAME_NAMESPACE_VERSION(name, namespace, version) \
    PLUGIN_GET_TYPE_OR_NAME_NAMESPACE_VERSION(name, namespace, version, Name)

/**
 * @details register plugin for dynamic library
 */
#define AHRI_REGISTER_TENSORRT_PLUGIN(creator)                                 \
    extern "C" AHRI_TENSORRT_API nvinfer1::IPluginCreator* create##creator() { \
        return new creator();                                                  \
    }
}  // namespace Ahri::TensorRT

// #endif  // __NVCC__
#endif  // !AHRI_ASUKA_TENSORRT_UTILS_HPP
