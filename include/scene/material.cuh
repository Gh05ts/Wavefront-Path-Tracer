#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "../core/vec3.cuh"

constexpr uint32_t invalidTextureIndex = 0xffffffffu;

struct Texture {
    cudaTextureObject_t object;
};

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
    uint32_t albedoTexture = invalidTextureIndex;
};
