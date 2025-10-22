#include "parrot.hpp"

auto main() -> int
{
    auto m = parrot::range(10000).as<float>().reshape({100, 100});

    // softmax
    auto cols = m.shape()[1];
    auto z = m - m.maxr<2>().replicate(cols);
    auto num = z.exp();
    auto den = num.sum<2>();
    (num / den.replicate(cols)).print();

    return 0;
}
