#pragma once

#include <cstdint>

#include "rng_types.hpp"

__host__ __device__
inline uint64_t splitMix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

__host__ __device__
inline RngState makeRngState(uint64_t key, RngStrategy strategy) {
    RngState result;
    result.strategy = strategy;

    if (strategy == RngStrategy::XorShift32) {
        uint32_t index = static_cast<uint32_t>(key);
        uint32_t seed = index * 747796405u + 2891336453u;
        result.state = seed == 0 ? 1u : seed;
    } else {
        result.state = splitMix64(key);
    }

    return result;
}

__host__ __device__
inline uint64_t pixelRngKey(
    uint32_t pixel,
    uint32_t sample,
    RngStrategy strategy) {
    if (strategy == RngStrategy::XorShift32)
        return static_cast<uint64_t>(pixel ^ (sample * 0x9e3779b9u));

    return (static_cast<uint64_t>(sample) << 32) | pixel;
}

__host__ __device__
inline uint64_t photonRngKey(
    uint32_t photonIndex,
    uint32_t seed,
    RngStrategy strategy) {
    if (strategy == RngStrategy::XorShift32)
        return static_cast<uint64_t>(seed ^ (photonIndex * 0x9e3779b9u));

    return (static_cast<uint64_t>(seed) << 32) | photonIndex;
}

__device__
float randomFloat(RngState& state);
