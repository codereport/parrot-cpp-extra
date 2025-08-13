#include "../../parrot.hpp"

int main() {
    auto arr    = pc::range(1000).add(-1);
    auto result = arr.outer(arr, pc::add{}).sum<2>();
    result.print();
    return 0;
}