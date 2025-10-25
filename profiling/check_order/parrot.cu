#include "parrot.hpp"

auto check_order(auto ints) { //
  return ints.sort().neq(ints).where() - 1;
}

auto main() -> int {
  auto ints = parrot::range(100000).rand();
  check_order(ints).print();
  return 0;
}