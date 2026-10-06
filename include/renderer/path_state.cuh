#pragma once

#include "../core/ray.cuh"
#include "../core/rng.cuh"

struct PathState {
    Ray ray;

    Vec3 throughput;
    Vec3 radiance;

    uint32_t pixelIndex;
    uint32_t depth;

    RngState rngState;
    uint32_t mediumMaterial = 0xffffffffu;
    float mediumDensity = 1.0f;

    float previousBsdfPdf;
    bool specularBounce;
    bool active;
};
