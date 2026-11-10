#pragma once

#include "vec3.cuh"

// The old 1 mm intersection cutoff plus a 1 mm origin shift could jump over
// both sides of thin automotive glass. Use a small traversal cutoff and an
// origin displacement scaled to floating-point precision in world space.
inline constexpr float rayMinDistance = 1.0e-6f;

__host__ __device__ inline Vec3 offsetRayOrigin(const Vec3& position,
    const Vec3& geometricNormal, const Vec3& direction, float hitDistance = 0.0f) {
    float magnitude = fmaxf(fmaxf(1.0f, fabsf(hitDistance)),
        fmaxf(fabsf(position.x), fmaxf(fabsf(position.y), fabsf(position.z))));
    float sign = dot(direction, geometricNormal) >= 0.0f ? 1.0f : -1.0f;
    Vec3 offset = geometricNormal * (sign * 32.0f * 1.1920928955078125e-7f * magnitude);
    Vec3 result = position + offset;
    if (offset.x != 0) result.x = nextafterf(result.x, offset.x > 0 ? INFINITY : -INFINITY);
    if (offset.y != 0) result.y = nextafterf(result.y, offset.y > 0 ? INFINITY : -INFINITY);
    if (offset.z != 0) result.z = nextafterf(result.z, offset.z > 0 ? INFINITY : -INFINITY);
    return result;
}

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
