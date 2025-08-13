#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <iomanip>
#include <iostream>

// Functor to create outer sum: result[i,j] = arr1[i] + arr2[j]
struct outer_sum_functor {
    int size2;

    outer_sum_functor(int s) : size2(s) {}

    __host__ __device__ int operator()(int linear_idx) const {
        int i = linear_idx / size2;  // row index
        int j = linear_idx % size2;  // column index
        return i + j;                // outer sum operation
    }
};

// Functor to compute row indices for reduce_by_key
struct row_index_functor {
    int num_cols;

    row_index_functor(int cols) : num_cols(cols) {}

    __host__ __device__ int operator()(int linear_idx) const {
        return linear_idx / num_cols;
    }
};

int main() {
    const int N = 1000;

    // Create the outer sum matrix (N x N) where result[i,j] = i + j
    // We'll represent this as a 1D array of size N*N
    auto counting_iter  = thrust::make_counting_iterator(0);
    auto outer_sum_iter = thrust::make_transform_iterator(counting_iter,
                                                          outer_sum_functor(N));

    // Create row indices for each element [0,0,0,...,1,1,1,...,2,2,2,...]
    auto row_indices_iter = thrust::make_transform_iterator(
      counting_iter, row_index_functor(N));

    // Allocate space for the row sums (one sum per row)
    thrust::device_vector<int> row_sums(N);

    // Perform row-wise sum using reduce_by_key
    // This sums all elements in each row of the outer sum matrix
    thrust::reduce_by_key(row_indices_iter,            // keys (row indices)
                          row_indices_iter + (N * N),  // keys end
                          outer_sum_iter,  // values (outer sum elements)
                          thrust::make_discard_iterator(),  // output keys
                          row_sums.begin(),         // output values (row sums)
                          thrust::equal_to<int>(),  // key comparison
                          thrust::plus<int>()       // reduction operation
    );

    // Copy results to host for printing
    thrust::host_vector<int> host_results = row_sums;

    // Print the row sums
    std::cout << "Row sums:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "Row " << i << ": " << host_results[i] << std::endl;
    }

    return 0;
}