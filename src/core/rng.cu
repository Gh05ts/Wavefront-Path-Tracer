#include "core/rng.cuh"

namespace {
__device__
uint32_t xorshift32(uint32_t value) {
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    return value;
}

__device__
uint32_t pcg32(uint64_t& state) {
    constexpr uint64_t multiplier = 6364136223846793005ull;
    constexpr uint64_t increment = 1442695040888963407ull;

    uint64_t oldState = state;
    state = oldState * multiplier + increment;

    uint32_t xorshifted =
        static_cast<uint32_t>(((oldState >> 18u) ^ oldState) >> 27u);
    uint32_t rotation = static_cast<uint32_t>(oldState >> 59u);

    return (xorshifted >> rotation) |
           (xorshifted << ((32u - rotation) & 31u));
}
}

__device__
float randomFloat(RngState& state) {
    uint32_t value;

    if (state.strategy == RngStrategy::XorShift32) {
        uint32_t x = xorshift32(static_cast<uint32_t>(state.state));
        state.state = x;
        value = x;
    } else {
        value = pcg32(state.state);
    }

    return static_cast<float>(value >> 8) * 0x1.0p-24f;
}
