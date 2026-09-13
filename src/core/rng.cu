#include "core/rng.cuh"

__device__
float randomFloat(uint32_t& state) {
    // xorshift32: compact and sufficient for independent per-path sampling.
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    // Keep the upper 24 bits, which map exactly to the float mantissa range.
    return static_cast<float>(state >> 8) * 0x1.0p-24f;
}
