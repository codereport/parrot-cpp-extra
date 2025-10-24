#include <cmath>
#include <cub/device/device_segmented_reduce.cuh>
#include <iomanip>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

// Functor to compute row index from linear index
struct row_index_functor {
  int cols;
  row_index_functor(int cols) : cols(cols) {}
  __host__ __device__ int operator()(int idx) const { return idx / cols; }
};

// Functor to subtract row max from each element
struct subtract_row_max_functor {
  const float *data;
  const float *row_maxes;
  int cols;

  subtract_row_max_functor(const float *data, const float *row_maxes, int cols)
      : data(data), row_maxes(row_maxes), cols(cols) {}

  __host__ __device__ float operator()(int idx) const {
    int row = idx / cols;
    return data[idx] - row_maxes[row];
  }
};

// Functor to compute exp(input - row_max) for segmented sum
struct exp_minus_max_functor {
  const float *data;
  const float *row_maxes;
  int cols;

  exp_minus_max_functor(const float *data, const float *row_maxes, int cols)
      : data(data), row_maxes(row_maxes), cols(cols) {}

  __host__ __device__ float operator()(int idx) const {
    int row = idx / cols;
    return expf(data[idx] - row_maxes[row]);
  }
};

// Functor to compute full softmax: exp(input - row_max) / row_sum
struct fused_softmax_functor {
  const float *data;
  const float *row_maxes;
  const float *row_sums;
  int cols;

  fused_softmax_functor(const float *data, const float *row_maxes,
                        const float *row_sums, int cols)
      : data(data), row_maxes(row_maxes), row_sums(row_sums), cols(cols) {}

  __host__ __device__ float operator()(int idx) const {
    int row = idx / cols;
    return expf(data[idx] - row_maxes[row]) / row_sums[row];
  }
};

// Helper function to perform segmented reduction (row-wise max)
void segmented_max(const thrust::device_vector<float> &input,
                   thrust::device_vector<float> &output, int rows, int cols) {
  // Use transform_iterator for offset array (0, cols, 2*cols, ...) - computed
  // on-the-fly
  auto offset_iterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [cols] __device__(int i) { return i * cols; });

  // Perform segmented reduction to find max in each row
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Max(nullptr, temp_storage_bytes,
                                  thrust::raw_pointer_cast(input.data()),
                                  thrust::raw_pointer_cast(output.data()), rows,
                                  offset_iterator, offset_iterator + 1);

  thrust::device_vector<char> temp_storage(temp_storage_bytes);
  cub::DeviceSegmentedReduce::Max(thrust::raw_pointer_cast(temp_storage.data()),
                                  temp_storage_bytes,
                                  thrust::raw_pointer_cast(input.data()),
                                  thrust::raw_pointer_cast(output.data()), rows,
                                  offset_iterator, offset_iterator + 1);
}

// Helper function to perform segmented sum (row-wise sum)
void segmented_sum(const thrust::device_vector<float> &input,
                   thrust::device_vector<float> &output, int rows, int cols) {
  // Use transform_iterator for offset array (0, cols, 2*cols, ...) - computed
  // on-the-fly
  auto offset_iterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [cols] __device__(int i) { return i * cols; });

  // Perform segmented reduction to find sum in each row
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes,
                                  thrust::raw_pointer_cast(input.data()),
                                  thrust::raw_pointer_cast(output.data()), rows,
                                  offset_iterator, offset_iterator + 1);

  thrust::device_vector<char> temp_storage(temp_storage_bytes);
  cub::DeviceSegmentedReduce::Sum(thrust::raw_pointer_cast(temp_storage.data()),
                                  temp_storage_bytes,
                                  thrust::raw_pointer_cast(input.data()),
                                  thrust::raw_pointer_cast(output.data()), rows,
                                  offset_iterator, offset_iterator + 1);
}

// Template helper function to perform segmented sum from any iterator
template <typename InputIterator>
void segmented_sum_from_iterator(InputIterator input_iter,
                                 thrust::device_vector<float> &output, int rows,
                                 int cols) {
  // Use transform_iterator for offset array (0, cols, 2*cols, ...) - computed
  // on-the-fly
  auto offset_iterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [cols] __device__(int i) { return i * cols; });

  // Perform segmented reduction to find sum in each row
  size_t temp_storage_bytes = 0;
  cub::DeviceSegmentedReduce::Sum(nullptr, temp_storage_bytes, input_iter,
                                  thrust::raw_pointer_cast(output.data()), rows,
                                  offset_iterator, offset_iterator + 1);

  thrust::device_vector<char> temp_storage(temp_storage_bytes);
  cub::DeviceSegmentedReduce::Sum(thrust::raw_pointer_cast(temp_storage.data()),
                                  temp_storage_bytes, input_iter,
                                  thrust::raw_pointer_cast(output.data()), rows,
                                  offset_iterator, offset_iterator + 1);
}

int main() {
  const int rows = 100;
  const int cols = 100;
  const int size = rows * cols;

  // Create input matrix: range(10000) -> [0, 1, 2, ..., 9999] reshaped to
  // 100x100
  thrust::host_vector<float> h_input(size);
  for (int i = 0; i < size; ++i) {
    h_input[i] = static_cast<float>(i);
  }

  thrust::device_vector<float> d_input = h_input;

  // Step 1: Find row-wise maximum
  thrust::device_vector<float> d_row_max(rows, thrust::default_init);
  segmented_max(d_input, d_row_max, rows, cols);

  // Step 2: Create transform_iterator for exp(input - row_max) computation
  auto exp_minus_max_iterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      exp_minus_max_functor(thrust::raw_pointer_cast(d_input.data()),
                            thrust::raw_pointer_cast(d_row_max.data()), cols));

  // Step 3: Compute row-wise sum of exponentials directly from
  // transform_iterator
  thrust::device_vector<float> d_den(rows, thrust::default_init);
  segmented_sum_from_iterator(exp_minus_max_iterator, d_den, rows, cols);

  // Step 4: Create fully fused softmax transform_iterator
  auto softmax_iterator = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      fused_softmax_functor(thrust::raw_pointer_cast(d_input.data()),
                            thrust::raw_pointer_cast(d_row_max.data()),
                            thrust::raw_pointer_cast(d_den.data()), cols));

  // Materialize final result on device, then copy to host
  thrust::device_vector<float> d_result(size, thrust::default_init);
  thrust::copy(softmax_iterator, softmax_iterator + size, d_result.begin());
  thrust::host_vector<float> h_result = d_result;

  std::cout << std::fixed << std::setprecision(6);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cout << h_result[i * cols + j];
      if (j < cols - 1)
        std::cout << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
