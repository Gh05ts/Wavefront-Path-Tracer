#pragma once

#include <cstdint>

enum class RngStrategy : uint32_t {
    XorShift32 = 0,
    Pcg32 = 1
};

struct RngState {
    uint64_t state = 0;
    RngStrategy strategy = RngStrategy::XorShift32;
};
