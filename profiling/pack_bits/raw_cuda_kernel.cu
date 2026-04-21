#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#include "common.hpp"

__global__ void pack_bits(const bool *input, uint8_t *output, int num_bytes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bytes) return;

    bool bits[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { bits[j] = input[8 * i + j]; }

    output[i] = (bits[0] << 0) | (bits[1] << 1) | (bits[2] << 2) |
                (bits[3] << 3) | (bits[4] << 4) | (bits[5] << 5) |
                (bits[6] << 6) | (bits[7] << 7);
}

int main(int argc, char *argv[]) {
    const int num_bytes = argc > 1 ? std::atoi(argv[1]) : (1 << 20);
    const int num_bits  = num_bytes * 8;

    auto h_bits = pack_bits_common::make_input_bits(num_bytes);

    bool *d_input     = nullptr;
    uint8_t *d_output = nullptr;
    cudaMalloc(&d_input, num_bits * sizeof(bool));
    cudaMalloc(&d_output, num_bytes * sizeof(uint8_t));

    cudaMemcpy(
      d_input, h_bits.data(), num_bits * sizeof(bool), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size  = (num_bytes + block_size - 1) / block_size;
    pack_bits<<<grid_size, block_size>>>(d_input, d_output, num_bytes);
    cudaDeviceSynchronize();

    std::vector<uint8_t> h_output(num_bytes);
    cudaMemcpy(h_output.data(),
               d_output,
               num_bytes * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return pack_bits_common::verify_and_print(
      "raw kernel", h_output, num_bytes);
}
