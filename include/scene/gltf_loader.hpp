#pragma once

#include <vector>

#include "obj_loader.hpp"
#include "scene.cuh"

struct GltfScene {
    std::vector<MeshAsset> meshes;
    std::vector<SceneInstance> instances;
    std::vector<Material> materials;
    std::vector<ObjTexture> textures;
};

GltfScene loadGltfScene(const char* filename);
