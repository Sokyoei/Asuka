/**
 * @file openvino_utils.hpp
 * @date 2024/10/09
 * @author Sokyoei
 *
 *
 */

#pragma once
#ifndef AHRI_ASUKA_OPENVINO_UTILS_HPP
#define AHRI_ASUKA_OPENVINO_UTILS_HPP

#include <filesystem>
#include <memory>

#include "Ahri/Asuka.hpp"

#include <openvino/openvino.hpp>
#ifdef ASUKA_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#include "Ahri/Ceceilia/utils/fmt.hpp"
#include "Ahri/Ceceilia/utils/logger_utils.hpp"

namespace Ahri::OpenVINO {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define AHRI_OPENVINO_API AHRI_API

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Inference
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class OpenVINOModel {
public:
    /**
     * @param model_path ONNX or OpenVINO xml model file path
     */
    OpenVINOModel(std::filesystem::path model_path, std::string device = "CPU")
        : _model_path(model_path), _device(device) {
        load_model();
    }

    ~OpenVINOModel() {}

    void load_model() {
        if (_model_loaded) {
            return;
        }

        // load plugin
        try {
#ifdef _WIN32
            _core.add_extension("ahri_plugin_openvino.dll");
#elif defined(__linux__)
            _core.add_extension("libahri_plugin_openvino.so");
#endif
        } catch (const std::runtime_error& e) {
            AHRI_LOGGER_ERROR("Load OpenVINO plugin fail, reason: {}", e.what());
        }

        std::shared_ptr<ov::Model> model = _core.read_model(_model_path);
        _compiled_model = _core.compile_model(model, _device);
        _infer_request = _compiled_model.create_infer_request();

        // model information
        auto input_port = model->input();
        _input_shape = input_port.get_shape();
        auto input_type = input_port.get_element_type().to_string();
        AHRI_LOGGER_DEBUG("Input: shape: {}, type: {}", _input_shape, input_type);
        _input_size = ov::shape_size(_input_shape);

        auto output_port = model->output();
        _output_shape = output_port.get_shape();
        auto output_type = output_port.get_element_type().to_string();
        AHRI_LOGGER_DEBUG("Output: shape: {}, type: {}", _output_shape, output_type);
        _output_size = ov::shape_size(_output_shape);

        _model_loaded = true;
    }

    template <typename T>
    std::vector<T> inference(const std::vector<T>& input_data) {
        ov::Tensor input_tensor = _infer_request.get_input_tensor();

        // Check input data size
        if (input_data.size() != _input_size) {
            throw std::invalid_argument("Input data size mismatch");
        }

        // Copy input data directly to tensor (assuming T is compatible with tensor type)
        std::memcpy(input_tensor.data<T>(), input_data.data(), _input_size * sizeof(T));

        _infer_request.infer();

        ov::Tensor output_tensor = _infer_request.get_output_tensor();
        std::vector<T> output_data(_output_size);
        std::memcpy(output_data.data(), output_tensor.data<T>(), _output_size * sizeof(T));

        return output_data;
    }

#ifdef ASUKA_USE_OPENCV
    /**
     * @brief 使用 cv::Mat 输入进行推理
     * @param input_mat 输入图像 RGB
     * @return 推理结果向量
     */
    std::vector<float> inference(const cv::Mat& input_mat) {
        ov::Tensor input_tensor = _infer_request.get_input_tensor();
        std::memcpy(input_tensor.data<float>(), input_mat.data,
                    input_mat.total() * input_mat.channels() * sizeof(float));

        _infer_request.infer();

        ov::Tensor output_tensor = _infer_request.get_output_tensor();
        std::vector<float> output_data(_output_size);
        std::memcpy(output_data.data(), output_tensor.data<float>(), _output_size * sizeof(float));

        return output_data;
    }
#endif

private:
    std::filesystem::path _model_path;
    std::string _device;
    ov::Core _core;
    ov::CompiledModel _compiled_model;
    ov::InferRequest _infer_request;
    bool _model_loaded = false;

    ov::Shape _input_shape;
    ov::Shape _output_shape;
    size_t _input_size = 0;
    size_t _output_size = 0;
};

using Model = OpenVINOModel;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Print helper
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace Samples {}  // namespace Samples
}  // namespace Ahri::OpenVINO

AHRI_FMT_FORMATTER_OSTREAM(ov::Version);

#endif  // !AHRI_ASUKA_OPENVINO_UTILS_HPP
