#pragma once

#include <cstdint>
#include <vector>

struct EnvironmentImage {
    uint32_t width;
    uint32_t height;
    std::vector<float> pixels;
};

EnvironmentImage loadEnvironmentImage(const char* filename);
