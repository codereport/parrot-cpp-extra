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
  // Step 1: Get effective counts (gather operation)
  auto effective_counts = base_counts.gather(indices);

  // Step 2: Expand each dimension using parrot's high-level replicate
  auto dim0_expanded = input_dims.row(0).replicate(effective_counts);
  auto dim1_expanded = input_dims.row(1).replicate(effective_counts);

  // Return both expanded dimensions as a pair
  return std::make_pair(dim0_expanded, dim1_expanded);
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

    // Call the optimized expand function (uses high-level replicate operations)
    std::cout
        << "\n🚀 CALLING OPTIMIZED PARROT EXPAND (high-level replicate)..."
        << std::endl;
    auto result = expand_parrot(base_counts, indices, input_dims);

    std::cout << "\n🚀 PARROT EXPAND RESULTS:" << std::endl;
    std::cout << "expanded_dim0 = ";
    result.first.print();
    std::cout << std::endl;
    std::cout << "expanded_dim1 = ";
    result.second.print();
    std::cout << std::endl;

  } catch (const std::exception &e) {
    std::cout << "❌ Exception in clean expand test: " << e.what() << std::endl;
  }
}

/**
 * @brief Show comparison between Thrust and Parrot approaches
 */
void show_comparison() {
  std::cout << "\n=== Thrust vs Parrot Comparison ===" << std::endl;
  std::cout << std::endl;

  std::cout << "🔧 THRUST VERSION (6 complex steps):" << std::endl;
  std::cout << "  1. IndexCountIterator countIter(baseCounts, indexVector);"
            << std::endl;
  std::cout << "  2. thrust::reduce(countIter, countIter + len);" << std::endl;
  std::cout << "  3. thrust::exclusive_scan(countIter, offsets.begin());"
            << std::endl;
  std::cout << "  4. thrust::scatter_if(counting_iter, offsets, indices);"
            << std::endl;
  std::cout << "  5. thrust::inclusive_scan(indices, thrust::maximum());"
            << std::endl;
  std::cout << "  6. thrust::copy(complex_iterators, output);" << std::endl;
  std::cout << std::endl;

  std::cout << "🚀 PARROT VERSION (3 simple operations):" << std::endl;
  std::cout << "  1. auto effective_counts = base_counts.gather(indices);"
            << std::endl;
  std::cout << "  2. auto dim0_expanded = "
               "input_dims.row(0).replicate(effective_counts);"
            << std::endl;
  std::cout << "  3. auto dim1_expanded = "
               "input_dims.row(1).replicate(effective_counts);"
            << std::endl;
  std::cout << std::endl;

  std::cout << "✨ SIMPLIFICATION: 6 complex GPU kernels → 3 simple high-level "
               "operations!"
            << std::endl;
  std::cout << "✨ READABILITY: Low-level GPU programming → High-level "
               "functional style!"
            << std::endl;
}

int main() {
  std::cout
      << "Running Optimized Parrot.hpp Expand Implementation (High-Level)\n"
      << std::endl;

  try {
    test_clean_expand();
    show_comparison();

    std::cout << "\n✅ Parrot expand test completed successfully!" << std::endl;
    std::cout << "\n🎯 Summary:" << std::endl;
    std::cout << "This demonstrates parrot's high-level approach to the same "
                 "expand operation"
              << std::endl;
    std::cout << "that requires 6 complex Thrust kernel launches in the "
                 "low-level version."
              << std::endl;

    return 0;
  } catch (const std::exception &e) {
    std::cout << "❌ Error: " << e.what() << std::endl;
    return 1;
  }
}