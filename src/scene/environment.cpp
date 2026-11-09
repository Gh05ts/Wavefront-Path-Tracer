#include "scene/environment.hpp"
#include "stb_image.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

void buildEnvironmentDistribution(HostEnvironment& map) {
    uint64_t count = uint64_t(map.width)*map.height;
    if (count == 0 || count > std::numeric_limits<uint32_t>::max() || map.pixels.size() != count)
        throw std::runtime_error("Invalid HDR environment dimensions");
    std::vector<double> weights(count);
    double total = 0.0;
    for (uint32_t i = 0; i < count; ++i) {
        const float4& p = map.pixels[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.x < 0 || p.y < 0 || p.z < 0)
            throw std::runtime_error("HDR environment contains negative or non-finite radiance");
        weights[i] = (0.2126*double(p.x)+0.7152*double(p.y)+0.0722*double(p.z)) *
            environmentSolidAngle(i/map.width, map.width, map.height);
        total += weights[i];
    }
    constexpr double fourPi = 12.5663706143591729538;
    map.probabilities.resize(count);
    map.alias.resize(count);
    double sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        double uniform = environmentSolidAngle(i/map.width, map.width, map.height)/fourPi;
        map.probabilities[i] = static_cast<float>(total > 0 ? .99*weights[i]/total + .01*uniform : uniform);
        sum += map.probabilities[i];
    }
    std::vector<uint32_t> small, large;
    for (uint32_t i = 0; i < count; ++i) {
        map.probabilities[i] /= sum;
        weights[i] = double(map.probabilities[i])*count;
        (weights[i] < 1.0 ? small : large).push_back(i);
    }
    while (!small.empty() && !large.empty()) {
        uint32_t low = small.back(); small.pop_back();
        uint32_t high = large.back(); large.pop_back();
        map.alias[low] = EnvironmentAlias{static_cast<float>(weights[low]),high};
        weights[high] += weights[low]-1;
        (weights[high] < 1.0 ? small : large).push_back(high);
    }
    for (uint32_t i : small) map.alias[i] = EnvironmentAlias{1,i};
    for (uint32_t i : large) map.alias[i] = EnvironmentAlias{1,i};
}

HostEnvironment loadEnvironmentMap(const char* filename) {
    if (!filename || !stbi_is_hdr(filename))
        throw std::runtime_error(std::string("Environment must be a readable Radiance .hdr file: ") + (filename ? filename : ""));
    int width = 0, height = 0, channels = 0;
    std::unique_ptr<float, void(*)(void*)> data(stbi_loadf(filename, &width, &height, &channels, 3), stbi_image_free);
    if (!data || width <= 0 || height <= 0)
        throw std::runtime_error(std::string("Could not decode HDR environment: ") + filename);
    if (uint64_t(width)*height > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("HDR environment exceeds texture-index range");
    HostEnvironment map;
    map.width = width; map.height = height;
    map.pixels.resize(size_t(width)*height);
    for (size_t i = 0; i < map.pixels.size(); ++i)
        map.pixels[i] = make_float4(data.get()[3*i], data.get()[3*i+1], data.get()[3*i+2], 1.0f);
    buildEnvironmentDistribution(map);
    return map;
}
