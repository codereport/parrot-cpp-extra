#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>
#include <iostream>

struct running_average_transform {
    mutable int index;

    __host__ __device__ running_average_transform() : index(0) {}

    __host__ __device__ float operator()(const int& cumsum) const {
        return static_cast<float>(cumsum) / static_cast<float>(++index);
    }
};

int main() {
    const int N = 10000;

    // Create output vector for results
    thrust::device_vector<float> running_average(N, thrust::default_init);

    // Use counting iterator as input and transform_output_iterator to compute
    // running averages
    thrust::inclusive_scan(
      thrust::counting_iterator<int>(0),
      thrust::counting_iterator<int>(N),
      thrust::make_transform_output_iterator(running_average.begin(),
                                             running_average_transform()),
      thrust::plus<int>());

    // Print results
    thrust::host_vector<float> h_result = running_average;
    for (int i = 0; i < N; ++i) {
        std::cout << h_result[i] << " ";
        if ((i + 1) % 10 == 0)
            std::cout << std::endl;  // New line every 10 elements
    }

    return 0;
}