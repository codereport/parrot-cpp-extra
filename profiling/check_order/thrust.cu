#include <iostream>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

struct random_int_generator {
  int max_val;
  unsigned int seed;

  random_int_generator(int max, unsigned int s) : max_val(max), seed(s) {}

  __host__ __device__ int operator()(int idx) const {
    thrust::default_random_engine rng(seed);
    thrust::uniform_int_distribution<int> dist(0, max_val);
    rng.discard(idx);
    return dist(rng);
  }
};

thrust::device_vector<int> check_order(thrust::device_vector<int> &ints) {
  int n = ints.size();

  thrust::device_vector<int> sorted_ints = ints;
  thrust::sort(sorted_ints.begin(), sorted_ints.end());

  auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(sorted_ints.begin(), ints.begin()));
  auto zip_end = thrust::make_zip_iterator(
      thrust::make_tuple(sorted_ints.end(), ints.end()));

  auto differences_begin = thrust::make_transform_iterator(
      zip_begin, thrust::make_zip_function(thrust::not_equal_to<int>()));

  auto counting_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0),
      [] __host__ __device__(int x) { return x - 1; });

  thrust::device_vector<int> indices(n, thrust::default_init);
  auto indices_end = thrust::copy_if(
      counting_begin, counting_begin + n, differences_begin, indices.begin(),
      [] __host__ __device__(int x) { return x != 0; });

  indices.resize(indices_end - indices.begin());

  return indices;
}

auto main() -> int {
  const int N = 100000;

  // Create random integers using thrust::transform
  thrust::device_vector<int> ints(N);

  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(N), ints.begin(),
                    random_int_generator(N, 12345));

  // Check order
  auto result = check_order(ints);

  // Print results
  std::cout << "Out of order indices: ";
  thrust::copy(result.begin(), result.end(),
               std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;
  std::cout << "Total: " << result.size() << " elements out of order"
            << std::endl;

  return 0;
}