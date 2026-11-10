#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "../core/vec3.cuh"

constexpr uint32_t invalidTextureIndex = 0xffffffffu;
constexpr uint32_t invalidMaterialIndex = 0xffffffffu;

struct Texture {
    cudaTextureObject_t object;
};

struct TextureBinding {
    uint32_t texture = invalidTextureIndex;
    uint32_t texCoord = 0;
    Vec2 offset{0.0f, 0.0f};
    Vec2 scale{1.0f, 1.0f};
    float rotation = 0.0f;
    bool hasTransform = false;
};

enum class MaterialType: uint32_t {
    Diffuse,
    Metal,
    Dielectric,
    Emissive,
    SpecularGlossiness
};

struct Material {
    MaterialType type;

    Vec3 albedo;
    Vec3 emission;

    float roughness;
    float ior;
    float dispersion = 0.0f;
    float metallic = 0.0f;
    float normalScale = 1.0f;
    float alphaCutoff = 0.5f;
    bool alphaMasked = false;
    Vec3 attenuationColor = Vec3(1.0f, 1.0f, 1.0f);
    float attenuationDistance = 0.0f;
    float volumeDensity = 1.0f;
    TextureBinding albedoTexture;
    TextureBinding metallicRoughnessTexture;
    TextureBinding emissiveTexture;
    TextureBinding normalTexture;
    TextureBinding thicknessTexture;
    Vec3 specular = Vec3(1.0f, 1.0f, 1.0f);
    float glossiness = 1.0f;
    float opacity = 1.0f;
    TextureBinding specularGlossinessTexture;
    bool specularGlossinessWorkflow = false;
    // Imported opaque glTF metallic/roughness, including its dielectric lobe.
    // Legacy hand-authored diffuse/metal materials keep their original model.
    bool gltfPbrWorkflow = false;
    float specularWeight = 1.0f;
    Vec3 specularColor = Vec3(1.0f, 1.0f, 1.0f);
    TextureBinding specularWeightTexture;
    TextureBinding specularColorTexture;
    float clearcoat = 0.0f;
    float clearcoatRoughness = 0.0f;
    float clearcoatNormalScale = 1.0f;
    TextureBinding clearcoatTexture;
    TextureBinding clearcoatRoughnessTexture;
    TextureBinding clearcoatNormalTexture;
};
