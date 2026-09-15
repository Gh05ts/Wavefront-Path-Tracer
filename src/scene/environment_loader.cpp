#include "scene/environment_loader.hpp"

#include "stb_image.h"

#include <cstdlib>
#include <iostream>

EnvironmentImage loadEnvironmentImage(const char* filename) {
    int width;
    int height;
    int channels;
    float* pixels = stbi_loadf(filename, &width, &height, &channels, STBI_rgb_alpha);

    if (pixels == nullptr) {
        std::cerr << "Failed to load HDR environment " << filename << ": " << stbi_failure_reason() << '\n';
        std::exit(1);
    }

    EnvironmentImage image{};
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.pixels.assign(pixels, pixels + 4 * width * height);
    stbi_image_free(pixels);
    return image;
}
