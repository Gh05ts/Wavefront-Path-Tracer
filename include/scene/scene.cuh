#pragma once

#include <cstdint>

#include "geometry.cuh"
#include "material.cuh"

struct Scene {
    Sphere* spheres;
    uint32_t sphereCount;

    Triangle* triangles;
    uint32_t triangleCount;

    Material* materials;
    uint32_t materialCount;
};

struct DeviceScene {
    Scene scene;

    Sphere* spheres;
    Triangle* triangles;
    Material* materials;
};

DeviceScene createDemoScene();

void destroyDeviceScene(DeviceScene& deviceScene);
