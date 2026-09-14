#pragma once

#include <cstdint>

#include "geometry.cuh"
#include "bvh.cuh"
#include "material.cuh"

#include <NXB/BVH.h>

struct Scene {
    Sphere* spheres;
    uint32_t sphereCount;

    Triangle* triangles;
    uint32_t triangleCount;

    BvhNode* bvhNodes;
    uint32_t* bvhTriangleIndices;
    uint32_t bvhNodeCount;

    NXB::BVH2::DeviceView nexusBvh;
    NXB::BVH8::DeviceView nexusBvh8;

    Material* materials;
    uint32_t materialCount;
};

struct DeviceScene {
    Scene scene;

    Sphere* spheres;
    Triangle* triangles;
    BvhNode* bvhNodes;
    uint32_t* bvhTriangleIndices;
    NXB::BVH2 nexusBvh;
    NXB::BVH8 nexusBvh8;
    Material* materials;
};

DeviceScene createDemoScene();
DeviceScene createObjScene(const char* filename);

void destroyDeviceScene(DeviceScene& deviceScene);
