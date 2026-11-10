#pragma once

#include <vector>
#include "environment.cuh"

struct HostEnvironment {
    uint32_t width = 0, height = 0;
    std::vector<float4> pixels;
    std::vector<float> probabilities;
    std::vector<EnvironmentAlias> alias;
};

// Importance weights cover the bilinear texture footprint in adjacent cells.
// A further 1% uniform-solid-angle component preserves global support.
void buildEnvironmentDistribution(HostEnvironment& map);
HostEnvironment loadEnvironmentMap(const char* filename);
