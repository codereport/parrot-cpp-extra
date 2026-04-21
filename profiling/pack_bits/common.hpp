#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace pack_bits_common {

// Deterministic test pattern: byte[i] = i % 256  =>  bit[8*i+j] = (i>>j)&1.
// This makes the expected packed output trivially verifiable.
inline std::vector<uint8_t> make_input_bits(int num_bytes) {
    const int num_bits = num_bytes * 8;
    std::vector<uint8_t> h_bits(num_bits);
    for (int i = 0; i < num_bytes; ++i) {
        for (int j = 0; j < 8; ++j) {
            h_bits[8 * i + j] = static_cast<uint8_t>((i >> j) & 1);
        }
    }
    return h_bits;
}

// Verify against the expected pattern and print a summary.
// Returns 0 if all bytes match, 1 otherwise (suitable for use as exit code).
inline int verify_and_print(const char *label,
                            const std::vector<uint8_t> &h_output,
                            int num_bytes) {
    uint64_t checksum = 0;
    int mismatches    = 0;
    for (int i = 0; i < num_bytes; ++i) {
        uint8_t expected = static_cast<uint8_t>(i % 256);
        if (h_output[i] != expected) ++mismatches;
        checksum += h_output[i];
    }

    std::printf("%s: num_bytes=%d\n", label, num_bytes);
    std::printf("first 16 bytes: ");
    for (int i = 0; i < 16 && i < num_bytes; ++i) {
        std::printf("%02x ", h_output[i]);
    }
    std::printf("\nchecksum=%llu mismatches=%d\n",
                static_cast<unsigned long long>(checksum),
                mismatches);

    return mismatches == 0 ? 0 : 1;
}

}  // namespace pack_bits_common
