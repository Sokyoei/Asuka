#pragma once
#ifndef AHRI_LEAKYRELU_OPENVINO_HPP
#define AHRI_LEAKYRELU_OPENVINO_HPP

#include <openvino/core/extension.hpp>
#include <openvino/core/visibility.hpp>
#include <openvino/op/op.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset.hpp>

namespace Ahri::OpenVINO::Plugin {
inline namespace V1 {
template <typename T>
void leaky_relu_kernel(const T* input, T* output, size_t count, float alpha) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = input[i] > T(0) ? input[i] : T(alpha * input[i]);
    }
}

class OPENVINO_CORE_EXPORTS AhriLeakyReLU : public ov::op::Op {
public:
    OPENVINO_OP("AhriLeakyReLU");

    AhriLeakyReLU() = default;

    AhriLeakyReLU(const ov::Output<ov::Node>& input, float alpha) : Op({input}), _alpha(alpha) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<AhriLeakyReLU>(new_args.at(0), _alpha);
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("alpha", _alpha);
        return true;
    }

    float get_alpha() const { return _alpha; }

    void set_alpha(float alpha) { _alpha = alpha; }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        const auto& input = inputs[0];
        auto& output = outputs[0];

        if (input.get_element_type() != output.get_element_type()) {
            return false;
        }

        switch (input.get_element_type()) {
            case ov::element::f32: {
                leaky_relu_kernel<float>(input.data<float>(), output.data<float>(), input.get_size(), _alpha);
                break;
            }
            case ov::element::f16: {
                using f16 = typename ov::element_type_traits<ov::element::f16>::value_type;
                leaky_relu_kernel<f16>(input.data<f16>(), output.data<f16>(), input.get_size(), _alpha);
                break;
            }
            default:
                return false;
        }
        return true;
    }

    bool evaluate(ov::TensorVector& outputs,
                  const ov::TensorVector& inputs,
                  const ov::EvaluationContext& context) const override {
        return evaluate(outputs, inputs);
    }

    std::string description() const override { return "AhriLeakyReLU"; }

    bool has_evaluate() const override { return true; }

private:
    float _alpha = 0.01f;
};
}  // namespace V1
}  // namespace Ahri::OpenVINO::Plugin

#endif  // !AHRI_LEAKYRELU_OPENVINO_HPP
