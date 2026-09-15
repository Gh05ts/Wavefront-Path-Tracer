#pragma once

#include <string>
#include <vector>

#include "geometry.cuh"
#include "material.cuh"

struct ObjTexture {
    uint32_t width;
    uint32_t height;
    std::vector<unsigned char> pixels;
};

struct ObjMesh {
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
    std::vector<ObjTexture> textures;
};

struct ObjScene {
    std::vector<ObjMesh> meshes;
    std::vector<Material> materials;
    std::vector<ObjTexture> textures;
};

ObjScene loadObjScene(const char* filename);
ObjMesh loadObjMesh(const char* filename);
