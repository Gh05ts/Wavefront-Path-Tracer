#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../core/vec3.cuh"

std::string sha256Framebuffer(const std::vector<Vec3>& pixels);
void writePpm(const char* filename, const std::vector<Vec3>& pixels, uint32_t width, uint32_t height);
