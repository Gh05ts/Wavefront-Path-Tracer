#pragma once

#include <cstdint>
#include <vector>

#include "geometry.cuh"
#include "bvh.cuh"
#include "material.cuh"

#include <NXB/BVH.h>

struct Blas {
    Triangle* triangles;
    NXB::BVH2::DeviceView bvh;
};

struct InstanceTransform {
    Vec3 localToWorldX;
    Vec3 localToWorldY;
    Vec3 localToWorldZ;
    Vec3 worldToLocalX;
    Vec3 worldToLocalY;
    Vec3 worldToLocalZ;
    Vec3 translation;
};

__host__ __device__
inline Vec3 transformVector(const InstanceTransform& transform, const Vec3& vector) {
    return transform.localToWorldX * vector.x + transform.localToWorldY * vector.y + transform.localToWorldZ * vector.z;
}

__host__ __device__
inline Vec3 inverseTransformVector(const InstanceTransform& transform, const Vec3& vector) {
    return transform.worldToLocalX * vector.x + transform.worldToLocalY * vector.y + transform.worldToLocalZ * vector.z;
}

__host__ __device__
inline Vec3 transformNormal(const InstanceTransform& transform, const Vec3& normal) {
    return normalize(Vec3(
        dot(transform.worldToLocalX, normal),
        dot(transform.worldToLocalY, normal),
        dot(transform.worldToLocalZ, normal)));
}

inline InstanceTransform makeScaledTransform(const Vec3& translation, float scale) {
    float inverseScale = 1.0f / scale;
    return InstanceTransform{
        Vec3(scale, 0.0f, 0.0f), Vec3(0.0f, scale, 0.0f), Vec3(0.0f, 0.0f, scale),
        Vec3(inverseScale, 0.0f, 0.0f), Vec3(0.0f, inverseScale, 0.0f), Vec3(0.0f, 0.0f, inverseScale),
        translation};
}

struct MeshInstance {
    uint32_t blasIndex;
    uint32_t lightOffset;
    InstanceTransform transform;
};

struct MeshAsset {
    std::vector<Triangle> triangles;
};

struct SceneInstance {
    uint32_t meshIndex;
    InstanceTransform transform;
};

struct Scene {
    Sphere* spheres;
    uint32_t sphereCount;

    Triangle* triangles;
    uint32_t triangleCount;

    Triangle* staticTriangles;
    uint32_t staticTriangleCount;

    BvhNode* bvhNodes;
    uint32_t* bvhTriangleIndices;
    uint32_t bvhNodeCount;

    NXB::BVH2::DeviceView nexusBvh;
    NXB::BVH8::DeviceView nexusBvh8;

    NXB::BVH2::DeviceView tlas;
    Blas* blases;
    uint32_t blasCount;
    MeshInstance* instances;
    uint32_t instanceCount;

    Material* materials;
    uint32_t materialCount;
    Texture* textures;
    uint32_t textureCount;

    TriangleLight* lights;
    LightAliasEntry* lightAlias;
    uint32_t lightCount;
    bool blackBackground;
    bool ignoreDirectLightOcclusion;
    bool useNormalMaps;
    float normalMapMinimumCosine;
};

struct DeviceScene {
    Scene scene;

    Sphere* spheres;
    Triangle* triangles;
    Triangle* staticTriangles;
    BvhNode* bvhNodes;
    uint32_t* bvhTriangleIndices;
    NXB::BVH2 nexusBvh;
    NXB::BVH8 nexusBvh8;
    NXB::BVH2 tlas;
    std::vector<NXB::BVH2> blasBvhs;
    std::vector<Triangle*> meshTriangles;
    Blas* blases;
    MeshInstance* instances;
    Material* materials;
    Texture* textures;
    TriangleLight* lights;
    LightAliasEntry* lightAlias;
    std::vector<TriangleLight> hostLights;
    std::vector<float> lightWeights;
    std::vector<cudaArray_t> textureArrays;
    std::vector<cudaTextureObject_t> textureObjects;
};

DeviceScene createDemoScene();
DeviceScene createObjScene(const char* filename);
DeviceScene createCornellScene(const char* filename, float objectScale, const Vec3& objectTranslation);
DeviceScene createGltfScene(const char* filename, float sceneScale, bool addTopLight);
void buildTlasBlas(DeviceScene& deviceScene, std::vector<MeshAsset>& meshes, const std::vector<SceneInstance>& instances, const std::vector<Material>& materials);
void addStaticTriangleLights(DeviceScene& deviceScene, std::vector<Triangle>& triangles, const std::vector<Material>& materials);
void uploadTriangleLights(DeviceScene& deviceScene);

void destroyDeviceScene(DeviceScene& deviceScene);
