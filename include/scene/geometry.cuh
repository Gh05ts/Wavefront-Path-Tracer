#pragma once

#include <cstdint>
#include "../core/ray.cuh"

struct Sphere {
    Vec3 center;
    float radius;

    uint32_t material;
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;

    uint32_t material;
};

struct Hit {
    float t;
    Vec3 position;
    Vec3 normal;

    uint32_t material;
};
