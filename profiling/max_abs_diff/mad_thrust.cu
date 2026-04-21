#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

struct rand_generator {
  unsigned int seed;
  rand_generator(unsigned int s) : seed(s) {}

  __host__ __device__ float operator()(int idx) const {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist(0.0f, 1000.0f);
    rng.discard(idx);
    return dist(rng);
  }
};

int main() {
  const int N = 10000;

  thrust::device_vector<float> a(N);
  thrust::transform(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(N), a.begin(),
                    rand_generator(42));

  auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(a.begin(), a.begin() + 1));

  auto abs_delta_begin = thrust::make_transform_iterator(
      zip_begin,
      [] __host__ __device__(thrust::tuple<float, float> t) {
        return fabsf(thrust::get<1>(t) - thrust::get<0>(t));
      });

  float result = thrust::reduce(abs_delta_begin, abs_delta_begin + N - 1,
                                0.0f, thrust::maximum<float>());

  std::cout << "max abs delta: " << result << std::endl;
  return 0;
}
