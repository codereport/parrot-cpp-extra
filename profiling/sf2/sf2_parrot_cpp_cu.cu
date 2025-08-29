#include "parrot.hpp"
#include <iostream>

int main() {
  // Sushi For Two
  // This calculates the best segment of sushi plates for two people to eat
  // where plates are arranged by type (1 or 2) in a conveyor belt

  // Define our sushi arrangement (where numbers represent different sushi
  // types)

  int N = 1000000;

  // Create a range of 1000 elements
  auto sushi = parrot::scalar(2).repeat(N).rand();

  // Calculate the optimal segment size
  auto result =
      sushi
          .differ()               // Find where types change
          .where()                // 🔥 Get indices of changes
          .prepend(0)             // Add start boundary
          .append(sushi.size())   // Add end boundary
          .deltas()               // Calculate segment sizes
          .map_adj(parrot::min{}) // Get min of adjacent segments (CTAD)
          .dble()                 // Double (for two people)
          .maxr();                // 🔥 Find the maximum value

  std::cout << "Best sushi segment size: " << result.value() << std::endl;
  // Output: Best sushi segment size: 6 (maximum number of sushi pieces for
  // two people)

  return 0;
}