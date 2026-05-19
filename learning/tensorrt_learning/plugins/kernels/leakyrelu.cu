#include "leakyrelu.cuh"

__global__ void leakyrelu_kernel(const float* inputs, float* outputs, const float alpha, const int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    outputs[index] = inputs[index] > 0.0f ? inputs[index] : (alpha * inputs[index]);
}

void leakyrelu(const float* inputs, float* outputs, const float alpha, const int elements, cudaStream_t stream) {
    dim3 block_size(256, 1, 1);
    dim3 grid_size(std::ceil(float(elements) / 256), 1, 1);
    leakyrelu_kernel<<<grid_size, block_size, 0, stream>>>(inputs, outputs, alpha, elements);
}
