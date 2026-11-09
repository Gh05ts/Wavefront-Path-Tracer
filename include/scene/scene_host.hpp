#pragma once

#include <vector>

#include "obj_loader.hpp"
#include "scene.cuh"

struct SceneSphere {
    Sphere geometry;
    AccelerationMode acceleration;

    SceneSphere(Sphere sphere, AccelerationMode mode = AccelerationMode::Bvh)
        : geometry(sphere), acceleration(mode) {}
};

// Host-owned scene data assembled from a preset and its source assets. This
// type contains no CUDA allocations or device BVH handles; scene.cu is the
// only boundary that turns it into a DeviceScene.
struct HostScene {
    std::vector<SceneSphere> spheres;
    std::vector<Triangle> triangles;
    std::vector<Triangle> staticTriangles;
    std::vector<MeshAsset> meshes;
    std::vector<SceneInstance> instances;
    std::vector<Material> materials;
    std::vector<ObjTexture> textures;
    std::vector<TriangleLight> lights;
    std::vector<float> lightWeights;
    std::vector<PunctualLight> punctualLights;

    ObjAccelerationPolicy acceleration = ObjAccelerationPolicy::TlasBlasBvh2;
    bool blackBackground = false;
};

DeviceScene uploadScene(const HostScene& hostScene);

HostScene assembleDemoScene();
HostScene assembleRtWeekScene(const char* filename);
HostScene assembleObjScene(const ObjSceneOptions& options);
HostScene assembleCornellScene(const CornellSceneOptions& options);
HostScene assembleGltfScene(const char* filename, float sceneScale, bool addTopLight);
