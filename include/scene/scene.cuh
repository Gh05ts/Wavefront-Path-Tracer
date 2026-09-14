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

struct MeshInstance {
    uint32_t blasIndex;
    Vec3 translation;
    float scale;
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

    AreaLight areaLight;
    bool hasAreaLight;
    bool blackBackground;
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
    Blas* blases;
    MeshInstance* instances;
    Material* materials;
};

DeviceScene createDemoScene();
DeviceScene createObjScene(const char* filename);
DeviceScene createCornellScene(const char* filename);

void destroyDeviceScene(DeviceScene& deviceScene);
