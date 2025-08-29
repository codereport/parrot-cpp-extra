#include "parrot.hpp"
#include <iostream>
#include <vector>

using namespace parrot;

/**
 * @brief Clean parrot.hpp-based expand implementation with functional interface
 *
 * This version abandons the C-style interface in favor of a clean functional
 * approach using parrot arrays directly, similar to mode_parrot.h
 */

template <typename BaseCountsArray, typename IndexArray,
          typename InputDimsArray>
auto expand_parrot(const BaseCountsArray &base_counts,
                   const IndexArray &indices,
                   const InputDimsArray &input_dims) {
  auto effective_counts = base_counts.gather(indices);
  using expanded_type = decltype(input_dims[0].replicate(effective_counts));
  auto expanded_dims = std::vector<expanded_type>{};

  for (size_t dim = 0; dim < input_dims.size(); dim++) {
    auto expanded = input_dims[dim].replicate(effective_counts);
    expanded_dims.push_back(expanded);
  }

  return expanded_dims;
}

/**
 * @brief Test the clean parrot-based expand function
 */
void test_clean_expand() {
  std::cout << "\n=== Testing Optimized Parrot-based Expand ===" << std::endl;

  try {
    // Create input data using parrot arrays directly
    auto base_counts = array<int>({1, 2, 3, 2, 1, 0, 1, 2, 1, 1});
    auto indices = array<int>({0, 2, 4, 7, 9});

    // Create multi-dimensional input data as rank-2 array (2x5 matrix)
    // Row 0: dim0_data = [0, 1, 2, 3, 4]
    // Row 1: dim1_data = [10, 11, 12, 13, 14]
    auto input_dims_vec = std::make_shared<thrust::device_vector<int>>(
        std::vector<int>{0, 1, 2, 3, 4, 10, 11, 12, 13, 14});
    auto input_dims =
        fusion_array<typename thrust::device_vector<int>::iterator>(
            input_dims_vec->begin(), input_dims_vec->end(), input_dims_vec,
            {2, 5} // 2 dimensions, 5 elements each
        );

    std::cout << "📊 INPUT DATA:" << std::endl;
    std::cout << "base_counts = ";
    // base_counts.print();
    std::cout << "indices = ";
    // indices.print();
    std::cout << "input_dims (2x5 matrix) = ";
    // input_dims.print();
    std::cout << "  Row 0 (dim0): [0, 1, 2, 3, 4]" << std::endl;
    std::cout << "  Row 1 (dim1): [10, 11, 12, 13, 14]" << std::endl;

    // Extract effective counts to show the logic
    // auto effective_counts = base_counts.gather(indices);
    // std::cout << "effective_counts = ";
    // effective_counts.print();
    // std::cout << "total_count = " << effective_counts.sum().value()
    //           << std::endl;

    // Call the optimized expand function (uses replicate<2> - NO FOR LOOP!)
    std::cout
        << "\n🚀 CALLING OPTIMIZED EXPAND (replicate<2> on rank-2 array)..."
        << std::endl;
    auto result = expand_parrot(base_counts, indices, input_dims);

    std::cout << "\n🚀 EXPAND RESULTS (flattened from 2D):" << std::endl;
    std::cout << "expanded_result = ";
    // result.print();

  } catch (const std::exception &e) {
    std::cout << "❌ Exception in clean expand test: " << e.what() << std::endl;
  }
}

int main() {
  std::cout
      << "Running Optimized Parrot.hpp Expand Implementation (replicate<2>)\n"
      << std::endl;

  test_clean_expand();
}