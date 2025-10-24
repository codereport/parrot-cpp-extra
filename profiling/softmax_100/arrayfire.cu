#include <arrayfire.h>
#include <iomanip>
#include <iostream>

int main() {
  af::setBackend(AF_BACKEND_CUDA);

  // Create input: [0..9999] reshaped to 100x100 (column-major layout)
  auto m = af::moddims(af::iota(af::dim4(10000)), af::dim4(100, 100));
  
  // Softmax along columns: exp(x - max) / sum(exp(x - max))
  auto exp_shifted = af::exp(m - af::tile(af::max(m, 0), 100));
  auto result = exp_shifted / af::tile(af::sum(exp_shifted, 0), 100);

  // Print in row-major order
  std::cout << std::fixed << std::setprecision(6);
  float *h = result.host<float>();
  for (int i = 0; i < 100; ++i)
    for (int j = 0; j < 100; ++j)
      std::cout << h[j + i * 100] << (j < 99 ? " " : "\n");
  af::freeHost(h);
}
