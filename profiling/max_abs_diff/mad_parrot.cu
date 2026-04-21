#include <parrot.hpp>

int main() {
  int N = 10000;
  auto data = parrot::scalar(1000).repeat(N).rand();
  auto diff = data.deltas().abs().maxr().print();
  return 0;
}
