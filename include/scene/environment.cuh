#pragma once

#include <cstdint>
#include "../core/vec3.cuh"

struct EnvironmentAlias { float probability; uint32_t alias; };

struct EnvironmentView {
    cudaTextureObject_t texture = 0;
    const EnvironmentAlias* alias = nullptr;
    const float* probabilities = nullptr;
    uint32_t width = 0, height = 0;
    float intensity = 1.0f;
    float rotation = 0.0f; // right-handed rotation around +Y, radians
};

__host__ __device__ inline Vec3 rotateEnvironment(const Vec3& direction, float angle) {
    float c = cosf(angle), s = sinf(angle);
    return Vec3(c*direction.x + s*direction.z, direction.y, -s*direction.x + c*direction.z);
}

// Equirectangular: top +Y, center +X, three-quarter column +Z.
__host__ __device__ inline Vec2 environmentUv(const Vec3& world, float rotation) {
    Vec3 direction = rotateEnvironment(world, -rotation);
    constexpr float pi = 3.14159265359f;
    float u = atan2f(direction.z, direction.x) / (2.0f*pi) + 0.5f;
    u -= floorf(u);
    float v = atan2f(sqrtf(direction.x*direction.x + direction.z*direction.z), direction.y) / pi;
    return Vec2{u, fminf(1.0f, fmaxf(0.0f, v))};
}

__host__ __device__ inline float environmentSolidAngle(uint32_t row, uint32_t width, uint32_t height) {
    constexpr double pi = 3.14159265358979323846;
    // Stable near poles: cos(theta0)-cos(theta1) without cancellation.
    return static_cast<float>((4.0*pi/width) * sin(pi*(row+0.5)/height) * sin(pi/(2.0*height)));
}

__host__ __device__ inline uint32_t environmentTexel(const EnvironmentView& map, const Vec3& direction) {
    Vec2 uv = environmentUv(direction, map.rotation);
    uint32_t x = static_cast<uint32_t>(uv.x * map.width);
    uint32_t y = static_cast<uint32_t>(uv.y * map.height);
    return (y < map.height ? y : map.height-1)*map.width + (x < map.width ? x : map.width-1);
}

__host__ __device__ inline float environmentPdf(const EnvironmentView& map, const Vec3& direction) {
    if (map.width == 0 || map.height == 0 || map.probabilities == nullptr) return 0.0f;
    uint32_t index = environmentTexel(map, direction);
    return map.probabilities[index] / environmentSolidAngle(index/map.width, map.width, map.height);
}

struct EnvironmentSample { Vec3 direction; float pdf; };

// Independent uniforms: alias candidate/coin, longitude, and cos(latitude).
// Candidate is double so maps larger than 2^24 texels do not lose index support.
__host__ __device__ inline EnvironmentSample sampleEnvironment(const EnvironmentView& map,
    double candidate, float coin, float longitude, float latitude) {
    uint32_t count = map.width*map.height;
    uint32_t index = static_cast<uint32_t>(candidate * count);
    if (index >= count) index = count-1;
    if (coin >= map.alias[index].probability) index = map.alias[index].alias;
    uint32_t x = index % map.width, y = index / map.width;
    constexpr double pi = 3.14159265358979323846;
    double u = (x + fminf(1.0f-1.0e-7f, fmaxf(1.0e-7f, longitude))) / map.width;
    double jitter = fminf(1.0f-1.0e-7f, fmaxf(1.0e-7f, latitude));
    double cosine = (1.0-jitter)*cos(pi*y/map.height) + jitter*cos(pi*(y+1.0)/map.height);
    double sine = sqrt(fmax(0.0, 1.0-cosine*cosine));
    double phi = (u-0.5)*2*pi;
    Vec3 local(static_cast<float>(sine*cos(phi)), static_cast<float>(cosine), static_cast<float>(sine*sin(phi)));
    return EnvironmentSample{rotateEnvironment(local, map.rotation),
        map.probabilities[index] / environmentSolidAngle(y, map.width, map.height)};
}
