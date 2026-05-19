#include "swish.cuh"

__global__ void swish_kernel(const float* inputs, float* outputs, const int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    outputs[index] = inputs[index] * (1.0f / (1.0f + std::exp(-inputs[index])));
}

void swish(const float* inputs, float* outputs, const int elements, cudaStream_t stream) {
    dim3 block_size(256, 1, 1);
    dim3 grid_size(std::ceil(float(elements) / 256), 1, 1);
    swish_kernel<<<grid_size, block_size, 0, stream>>>(inputs, outputs, elements);
}
