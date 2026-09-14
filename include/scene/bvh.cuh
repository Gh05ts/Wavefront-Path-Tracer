#pragma once

#include <cstdint>
#include <vector>

#include "geometry.cuh"

struct Aabb {
    Vec3 minimum;
    Vec3 maximum;
};

struct BvhNode {
    Aabb bounds;
    uint32_t leftChild;
    uint32_t rightChild;
    uint32_t firstTriangle;
    uint32_t triangleCount;
};

struct HostBvh {
    std::vector<BvhNode> nodes;
    std::vector<uint32_t> triangleIndices;
};

HostBvh buildTriangleBvh(const std::vector<Triangle>& triangles);
HostBvh buildSpatialSplitBvh(const std::vector<Triangle>& triangles);
