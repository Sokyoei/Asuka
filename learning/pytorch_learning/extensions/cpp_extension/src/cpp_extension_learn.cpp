#include "cpp_extension_learn.hpp"

torch::Tensor test_forward_cpu(const torch::Tensor& a, const torch::Tensor& b) {
    AT_ASSERTM(a.sizes() == b.sizes());
    torch::Tensor c = torch::zeros(a.sizes());
    c = 2 * a + b;
    return c;
}

std::vector<torch::Tensor> test_backward_cpu(const torch::Tensor& c) {
    torch::Tensor a_grad = 2 * c * torch::ones(c.sizes());
    torch::Tensor b_grad = c * torch::ones(c.sizes());
    return {a_grad, b_grad};
}

PYBIND11_MODULE(cpp_extension_learn, m) {
    m.def("forward", &test_forward_cpu, "test forward");
    m.def("backward", &test_backward_cpu, "test backward");
}
