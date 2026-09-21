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

inline Vec3 rotateVector(const Vec3& vector, const Vec3& axis, float angleRadians) {
    Vec3 unitAxis = normalize(axis);
    float cosine = cosf(angleRadians);
    float sine = sinf(angleRadians);
    return vector * cosine + cross(unitAxis, vector) * sine + unitAxis * (dot(unitAxis, vector) * (1.0f - cosine));
}

inline Vec3 transformBasis(const Vec3& basisX, const Vec3& basisY, const Vec3& basisZ, const Vec3& vector) {
    return basisX * vector.x + basisY * vector.y + basisZ * vector.z;
}

inline void rotateTransform(InstanceTransform& transform, const Vec3& axis, float degrees) {
    constexpr float degreesToRadians = 3.14159265359f / 180.0f;
    float angle = degrees * degreesToRadians;
    Vec3 oldWorldToLocalX = transform.worldToLocalX;
    Vec3 oldWorldToLocalY = transform.worldToLocalY;
    Vec3 oldWorldToLocalZ = transform.worldToLocalZ;

    transform.localToWorldX = rotateVector(transform.localToWorldX, axis, angle);
    transform.localToWorldY = rotateVector(transform.localToWorldY, axis, angle);
    transform.localToWorldZ = rotateVector(transform.localToWorldZ, axis, angle);

    Vec3 inverseRotationX = rotateVector(Vec3(1.0f, 0.0f, 0.0f), axis, -angle);
    Vec3 inverseRotationY = rotateVector(Vec3(0.0f, 1.0f, 0.0f), axis, -angle);
    Vec3 inverseRotationZ = rotateVector(Vec3(0.0f, 0.0f, 1.0f), axis, -angle);
    transform.worldToLocalX = transformBasis(oldWorldToLocalX, oldWorldToLocalY, oldWorldToLocalZ, inverseRotationX);
    transform.worldToLocalY = transformBasis(oldWorldToLocalX, oldWorldToLocalY, oldWorldToLocalZ, inverseRotationY);
    transform.worldToLocalZ = transformBasis(oldWorldToLocalX, oldWorldToLocalY, oldWorldToLocalZ, inverseRotationZ);
}

inline void rotateY180(InstanceTransform& transform) {
    rotateTransform(transform, Vec3(0.0f, 1.0f, 0.0f), 180.0f);
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

enum class CornellObjectSource {
    Obj,
    Gltf,
    ProceduralPrism
};

enum class CornellLightProfile {
    Standard,
    Prism,
    Crystal
};

struct CornellSceneOptions {
    const char* filename = nullptr;
    CornellObjectSource objectSource = CornellObjectSource::Obj;
    float objectScale = 1.0f;
    Vec3 objectTranslation = Vec3(0.0f, 0.0f, 0.0f);
    bool neutralRoom = false;
    bool removeBackdrop = false;
    bool convertObjectMaterialsToDielectric = false;
    int32_t objectMaterialOverride = -1;
    CornellLightProfile lightProfile = CornellLightProfile::Standard;
};

enum class ObjAccelerationPolicy {
    TlasBlasBvh2,
    NexusBvh2,
    NexusBvh8,
    MedianBvh,
    SpatialSplitBvh
};

struct ObjSceneOptions {
    const char* filename = nullptr;
    float objectScale = 8.0f;
    Vec3 objectTranslation = Vec3(0.13f, -0.764f, 0.5f);
    ObjAccelerationPolicy acceleration = ObjAccelerationPolicy::TlasBlasBvh2;
};

DeviceScene createDemoScene();
DeviceScene createObjScene(const ObjSceneOptions& options);
DeviceScene createCornellScene(const CornellSceneOptions& options);
DeviceScene createGltfScene(const char* filename, float sceneScale, bool addTopLight);

void destroyDeviceScene(DeviceScene& deviceScene);
