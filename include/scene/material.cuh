#pragma once

#include <cstdint>
#include "../core/vec3.cuh"

enum class MaterialType: uint32_t {
    Diffuse,
    Metal,
    Dielectric,
    Emissive
};

struct Material {
    MaterialType type;

    Vec3 albedo;
    Vec3 emission;

    float roughness;
    float ior;
};
