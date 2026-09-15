#pragma once

#include "../core/ray.cuh"

struct PathState {
    Ray ray;

    Vec3 throughput;
    Vec3 radiance;

    uint32_t pixelIndex;
    uint32_t depth;

    uint32_t rngState;

    float previousBsdfPdf;
    bool specularBounce;
    bool active;
};
