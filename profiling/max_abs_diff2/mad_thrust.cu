#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits.h>
#include <unistd.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>

int main() {
  const int N = 10000;

  char buf[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  buf[len] = '\0';
  auto data_path = std::filesystem::path(buf).parent_path().parent_path() / "mad_data.bin";

  std::vector<int> host(N);
  std::ifstream f(data_path, std::ios::binary);
  f.read(reinterpret_cast<char *>(host.data()), N * sizeof(int));

  thrust::device_vector<int> a(host.begin(), host.end());

  auto zip_begin = thrust::make_zip_iterator(
      thrust::make_tuple(a.begin(), a.begin() + 1));

  auto abs_delta_begin = thrust::make_transform_iterator(
      zip_begin,
      [] __host__ __device__(thrust::tuple<int, int> t) {
        return abs(thrust::get<1>(t) - thrust::get<0>(t));
      });

  int result = thrust::reduce(abs_delta_begin, abs_delta_begin + N - 1,
                              0, thrust::maximum<int>());

  std::cout << "max abs delta: " << result << std::endl;
  return 0;
}
