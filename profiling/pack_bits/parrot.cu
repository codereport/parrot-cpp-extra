#include <cstdint>
#include <cstdlib>
#include <parrot.hpp>
#include <thrust/device_vector.h>
#include <thrust/functional.h>

#include "common.hpp"

using namespace parrot::literals;

int main(int argc, char *argv[]) {
  const int num_bytes = argc > 1 ? std::atoi(argv[1]) : (1 << 20);

  auto h_bits = pack_bits_common::make_input_bits(num_bytes);

  thrust::device_vector<uint8_t> dv(h_bits.begin(), h_bits.end());
  auto input = parrot::fusion_array(dv.begin(), dv.end());

  auto shift_bit = [] __host__ __device__(thrust::pair<uint8_t, int> p)
      -> uint8_t {
        int idx = p.second - 1;
        return p.first ? static_cast<uint8_t>(1u << (idx % 8)) : uint8_t{0};
      };

  auto h_output = input.enumerate()
                      .map(shift_bit)
                      .reshape({num_bytes, 8})
                      .reduce(uint8_t{0}, thrust::bit_or<uint8_t>{}, 2_ic)
                      .to_host();

  return pack_bits_common::verify_and_print("parrot", h_output, num_bytes);
}
