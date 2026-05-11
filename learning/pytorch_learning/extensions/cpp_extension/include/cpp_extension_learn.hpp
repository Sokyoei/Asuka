#pragma once
#ifndef CPP_EXTENSION_LEARN_HPP
#define CPP_EXTENSION_LEARN_HPP

#include <vector>

#include <torch/extension.h>

torch::Tensor test_forward_cpu(const torch::Tensor& a, const torch::Tensor& b);

std::vector<torch::Tensor> test_backward_cpu(const torch::Tensor& c);

#endif  // !CPP_EXTENSION_LEARN_HPP
