#include "parrot.hpp"

int main() {
    auto data            = parrot::range(10000);
    auto running_average = data.sums().as<float>() / parrot::range(data.size());
    running_average.print();
}