#include "parrot.hpp"

int main() {
  auto arr = parrot::range(1000).add(-1);
  auto result = arr.outer(arr, parrot::add{}).sum<2>();
  result.print();
  return 0;
}