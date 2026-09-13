#pragma once

#include <cstdint>

#include "../core/ray.cuh"

struct Camera {
    Vec3 position;

    Vec3 forward;
    Vec3 right;
    Vec3 up;

    float verticalFov;
    float aspectRatio;

    __host__ __device__
    Ray generateRay(float u, float v) const {
        float tanHalfFov = tanf(verticalFov * 0.5f);

        float x = (2.0f * u - 1.0f) * aspectRatio * tanHalfFov;
        float y = (1.0f - 2.0f * v) * tanHalfFov;

        Vec3 direction = normalize(forward + right * x + up * y);

        return Ray(position, direction);
    }
};

Camera createDemoCamera(uint32_t width, uint32_t height);
