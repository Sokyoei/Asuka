#pragma once
#ifndef AHRI_SWISH_OPENVINO_HPP
#define AHRI_SWISH_OPENVINO_HPP

#include <openvino/core/extension.hpp>
#include <openvino/core/visibility.hpp>
#include <openvino/op/op.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset.hpp>

namespace Ahri::OpenVINO::Plugin {
inline namespace V1 {
template <typename T>
void swish_kernel(const T* input, T* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = input[i] * (1.0f / (1.0f + std::exp(-input[i])));
    }
}

class OPENVINO_CORE_EXPORTS AhriSwish : public ov::op::Op {
public:
    OPENVINO_OP("AhriSwish");

    AhriSwish() = default;

    AhriSwish(const ov::Output<ov::Node>& input) : Op({input}) { constructor_validate_and_infer_types(); }

    void validate_and_infer_types() override {
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override {
        return std::make_shared<AhriSwish>(new_args.at(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override { return true; }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        const auto& input = inputs[0];
        auto& output = outputs[0];

        if (input.get_element_type() != output.get_element_type()) {
            return false;
        }

        switch (input.get_element_type()) {
            case ov::element::f32: {
                swish_kernel<float>(input.data<float>(), output.data<float>(), input.get_size());
                break;
            }
            case ov::element::f16: {
                using f16 = typename ov::element_type_traits<ov::element::f16>::value_type;
                swish_kernel<f16>(input.data<f16>(), output.data<f16>(), input.get_size());
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

    std::string description() const override { return "AhriSwish"; }

    bool has_evaluate() const override { return true; }

private:
};
}  // namespace V1
}  // namespace Ahri::OpenVINO::Plugin

#endif  // !AHRI_SWISH_OPENVINO_HPP
