#pragma once

#include <vector>
#include "environment.cuh"

struct HostEnvironment {
    uint32_t width = 0, height = 0;
    std::vector<float4> pixels;
    std::vector<float> probabilities;
    std::vector<EnvironmentAlias> alias;
};

// Sampling probabilities include a 1% uniform-solid-angle component, preserving
// support across bilinear reconstruction around otherwise black texels.
void buildEnvironmentDistribution(HostEnvironment& map);
HostEnvironment loadEnvironmentMap(const char* filename);
