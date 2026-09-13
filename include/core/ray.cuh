#pragma once

#include "vec3.cuh"

struct Ray {
    Vec3 origin;
    Vec3 direction;

    __host__ __device__
    Ray() = default;

    __host__ __device__
    Ray(const Vec3& origin, const Vec3& direction): origin(origin), direction(direction) {}

    __host__ __device__
    Vec3 at(float t) const {
        return origin + direction * t;
    }
};
