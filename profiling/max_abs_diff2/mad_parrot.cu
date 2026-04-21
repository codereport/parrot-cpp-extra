#include <parrot.hpp>
#include <filesystem>
#include <fstream>
#include <limits.h>
#include <unistd.h>
#include <vector>

int main() {
  const int N = 10000;

  char buf[PATH_MAX];
  ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  buf[len] = '\0';
  auto data_path = std::filesystem::path(buf).parent_path().parent_path() / "mad_data.bin";

  std::vector<int> host(N);
  std::ifstream f(data_path, std::ios::binary);
  f.read(reinterpret_cast<char *>(host.data()), N * sizeof(int));

  thrust::device_vector<int> dv(host.begin(), host.end());
  auto data = parrot::fusion_array(dv.begin(), dv.end());

  auto diff = data.deltas().abs().maxr().print();
  return 0;
}
