#pragma once

#include <cstdint>
#include <vector>

#include "../core/vec3.cuh"

void writePpm(const char* filename, const std::vector<Vec3>& pixels, uint32_t width, uint32_t height);
