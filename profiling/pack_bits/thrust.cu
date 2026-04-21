#include <cstdint>
#include <cstdlib>
#include <vector>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/std/functional>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "common.hpp"

int main(int argc, char *argv[]) {
    const int num_bytes    = argc > 1 ? std::atoi(argv[1]) : (1 << 20);
    const int segment_size = 8;

    auto h_bits = pack_bits_common::make_input_bits(num_bytes);

    thrust::device_vector<bool> d_input(h_bits.begin(), h_bits.end());
    thrust::device_vector<uint8_t> d_output(num_bytes);

    // Build a transform iterator over (bit, global_index) that produces
    // bit ? (1u << (i % 8)) : 0.
    const bool *d_in_ptr = thrust::raw_pointer_cast(d_input.data());
    auto shifted_bits    = thrust::make_transform_iterator(
      thrust::counting_iterator<int>{0},
      [d_in_ptr] __host__ __device__(int i) -> uint8_t {
          return d_in_ptr[i] ? static_cast<uint8_t>(1u << (i % 8))
                                 : uint8_t{0};
      });

    // Fixed-size segmented reduce: num_bytes segments, each of size 8,
    // reduced with bit-or and initial value 0.
    void *d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                       temp_storage_bytes,
                                       shifted_bits,
                                       d_output.begin(),
                                       num_bytes,
                                       segment_size,
                                       cuda::std::bit_or<uint8_t>{},
                                       uint8_t{0});

    thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    cub::DeviceSegmentedReduce::Reduce(d_temp_storage,
                                       temp_storage_bytes,
                                       shifted_bits,
                                       d_output.begin(),
                                       num_bytes,
                                       segment_size,
                                       cuda::std::bit_or<uint8_t>{},
                                       uint8_t{0});
    cudaDeviceSynchronize();

    std::vector<uint8_t> h_output(num_bytes);
    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    return pack_bits_common::verify_and_print(
      "cub DeviceSegmentedReduce", h_output, num_bytes);
}
