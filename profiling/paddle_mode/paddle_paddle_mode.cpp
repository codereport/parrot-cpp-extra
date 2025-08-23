#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

// Simplified DenseTensor class to replace PaddlePaddle's version
class DenseTensor {
   public:
    thrust::device_vector<float> data_;
    std::vector<int64_t> shape_;

    DenseTensor() = default;

    void Resize(std::vector<int64_t> shape) {
        shape_             = shape;
        int64_t total_size = 1;
        for (auto dim : shape) { total_size *= dim; }
        data_.resize(total_size);
    }

    float* data() { return thrust::raw_pointer_cast(data_.data()); }
    const float* data() const { return thrust::raw_pointer_cast(data_.data()); }

    size_t size() const { return data_.size(); }
};

// Simplified GPUContext to replace PaddlePaddle's version
namespace phi {
class GPUContext {
   public:
    template <typename T>
    T* Alloc(DenseTensor* tensor) {
        return reinterpret_cast<T*>(tensor->data());
    }

    int GetPlace() const { return 0; }  // Simplified device ID
};

void Copy(const GPUContext& ctx,
          const DenseTensor& src,
          int place,
          bool async,
          DenseTensor* dst) {
    thrust::copy(src.data_.begin(), src.data_.end(), dst->data_.begin());
}
}  // namespace phi

// Simplified common namespace for dimension creation
namespace common {
std::vector<int64_t> make_ddim(std::initializer_list<int64_t> dims) {
    return std::vector<int64_t>(dims);
}
}  // namespace common

// The original function with minimal modifications
template <typename T>
static void GetModebySort(const phi::GPUContext& dev_ctx,
                          const DenseTensor* input_tensor,
                          const int64_t num_cols,
                          const int64_t num_rows,
                          T* out_tensor,
                          int64_t* indices_tensor) {
    DenseTensor input_tmp;
    input_tmp.Resize(common::make_ddim({num_rows, num_cols}));
    T* input_tmp_data = dev_ctx.Alloc<T>(&input_tmp);
    phi::Copy(dev_ctx, *input_tensor, dev_ctx.GetPlace(), false, &input_tmp);

    thrust::device_ptr<T> out_tensor_ptr(out_tensor);
    thrust::device_ptr<int64_t> indices_tensor_ptr(indices_tensor);

    for (int64_t i = 0; i < num_rows; ++i) {
        T* begin = input_tmp_data + num_cols * i;
        T* end   = input_tmp_data + num_cols * (i + 1);
        thrust::device_vector<int64_t> indices_data(num_cols);
        thrust::sequence(thrust::device,
                         indices_data.begin(),
                         indices_data.begin() + num_cols);
        thrust::sort_by_key(thrust::device, begin, end, indices_data.begin());
        int unique = 1 + thrust::inner_product(thrust::device,
                                               begin,
                                               end - 1,
                                               begin + 1,
                                               0,
                                               thrust::plus<int>(),
                                               thrust::not_equal_to<T>());
        thrust::device_vector<T> keys_data(unique);
        thrust::device_vector<int64_t> cnts_data(unique);
        thrust::reduce_by_key(thrust::device,
                              begin,
                              end,
                              thrust::constant_iterator<int>(1),
                              keys_data.begin(),
                              cnts_data.begin());
        auto it = thrust::max_element(
          thrust::device, cnts_data.begin(), cnts_data.begin() + unique);
        T mode                = keys_data[it - cnts_data.begin()];
        int64_t counts        = cnts_data[it - cnts_data.begin()];
        auto pos              = thrust::find(thrust::device, begin, end, mode);
        int64_t index         = indices_data[pos - begin + counts - 1];
        out_tensor_ptr[i]     = static_cast<T>(mode);
        indices_tensor_ptr[i] = static_cast<int64_t>(index);
    }
}

// Test function to demonstrate the mode computation
void test_mode_computation() {
    std::cout << "Testing Mode Computation\n";
    std::cout << "========================\n";

    const int64_t num_rows = 3;
    const int64_t num_cols = 6;

    // Create test data: each row has different patterns
    std::vector<float> test_data = {// Row 0: mode is 2.0 (appears 3 times)
                                    1.0f,
                                    2.0f,
                                    3.0f,
                                    2.0f,
                                    4.0f,
                                    2.0f,
                                    // Row 1: mode is 5.0 (appears 2 times)
                                    5.0f,
                                    1.0f,
                                    5.0f,
                                    3.0f,
                                    4.0f,
                                    6.0f,
                                    // Row 2: mode is 7.0 (appears 4 times)
                                    7.0f,
                                    8.0f,
                                    7.0f,
                                    9.0f,
                                    7.0f,
                                    7.0f};

    // Set up input tensor
    DenseTensor input_tensor;
    input_tensor.Resize({num_rows, num_cols});
    thrust::copy(
      test_data.begin(), test_data.end(), input_tensor.data_.begin());

    // Set up output tensors
    thrust::device_vector<float> modes(num_rows);
    thrust::device_vector<int64_t> indices(num_rows);

    // Create GPU context
    phi::GPUContext gpu_ctx;

    // Call the mode computation function
    GetModebySort(gpu_ctx,
                  &input_tensor,
                  num_cols,
                  num_rows,
                  thrust::raw_pointer_cast(modes.data()),
                  thrust::raw_pointer_cast(indices.data()));

    // Copy results back to host for display
    thrust::host_vector<float> h_modes     = modes;
    thrust::host_vector<int64_t> h_indices = indices;
    thrust::host_vector<float> h_input     = input_tensor.data_;

    // Display results
    std::cout << "Input data:\n";
    for (int64_t i = 0; i < num_rows; ++i) {
        std::cout << "Row " << i << ": ";
        for (int64_t j = 0; j < num_cols; ++j) {
            std::cout << h_input[i * num_cols + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nResults:\n";
    for (int64_t i = 0; i < num_rows; ++i) {
        std::cout << "Row " << i << " - Mode: " << h_modes[i]
                  << ", Original Index: " << h_indices[i] << "\n";
    }

    // Verify expected results
    std::vector<float> expected_modes = {2.0f, 5.0f, 7.0f};
    bool all_correct                  = true;

    for (int64_t i = 0; i < num_rows; ++i) {
        if (std::abs(h_modes[i] - expected_modes[i]) > 1e-6) {
            std::cout << "ERROR: Row " << i << " expected mode "
                      << expected_modes[i] << " but got " << h_modes[i] << "\n";
            all_correct = false;
        }
    }

    if (all_correct) {
        std::cout << "\n✓ All mode computations are correct!\n";
    } else {
        std::cout << "\n✗ Some mode computations failed!\n";
    }
}

int main() {
    std::cout << "PaddlePaddle Mode Computation Example\n";
    std::cout << "=====================================\n\n";

    try {
        test_mode_computation();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
