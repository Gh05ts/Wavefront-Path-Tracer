#pragma once

#include <cstdint>

__host__ __device__
inline uint32_t makeRngSeed(uint32_t index)
{
    uint32_t seed = index * 747796405u + 2891336453u;
    return seed == 0 ? 1u : seed;
}

__device__
float randomFloat(uint32_t& state);
