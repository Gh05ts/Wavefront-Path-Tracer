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
    float etaScale = 1.0f; // inverse accumulated radiance eta^2, for roulette only
    float indirectRadianceClamp = 0.0f;
};

// Clamp fully weighted indirect light contributions, before accumulation/SPP
// averaging. A common RGB scale preserves hue. This deliberately biases rare
// caustic tails, like Cycles' indirect sample clamp; it never alters the HDR
// texture, BSDF, throughput, PDF or subsequent path. Zero preserves raw transport.
// NEE/terminal splitting supplies the current surface depth. Emission and miss
// hits subtract one: a first-bounce BSDF escape competes with primary-surface NEE
// and must have the same direct/indirect classification.
__host__ __device__ inline void addPathRadiance(PathState& path, Vec3 contribution,
    bool emissionOrBackground = false) {
    bool indirect = path.depth > (emissionOrBackground ? 1u : 0u);
    if (indirect && path.indirectRadianceClamp > 0.0f) {
        float magnitude = (fabsf(contribution.x) / 3.0f + fabsf(contribution.y) / 3.0f
            + fabsf(contribution.z) / 3.0f);
        if (magnitude > path.indirectRadianceClamp)
            contribution *= path.indirectRadianceClamp / magnitude;
    }
    path.radiance += contribution;
}
