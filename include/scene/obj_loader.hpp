#pragma once

#include <vector>

#include "geometry.cuh"
#include "material.cuh"

struct ObjMesh {
    std::vector<Triangle> triangles;
    std::vector<Material> materials;
};

ObjMesh loadObjMesh(const char* filename);
