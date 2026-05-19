#include "pch.h"

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>
#include <openvino/frontend/node_context.hpp>

#include "ahri_leakyrelu_openvino.hpp"
#include "ahri_swish_openvino.hpp"

namespace Ahri::OpenVINO::Plugin {
OPENVINO_CREATE_EXTENSIONS(std::vector<ov::Extension::Ptr>({
    std::make_shared<ov::OpExtension<AhriLeakyReLU>>(),
    std::make_shared<ov::frontend::OpExtension<AhriLeakyReLU>>(),
    std::make_shared<ov::OpExtension<AhriSwish>>(),
    std::make_shared<ov::frontend::OpExtension<AhriSwish>>(),
}));
}  // namespace Ahri::OpenVINO::Plugin
