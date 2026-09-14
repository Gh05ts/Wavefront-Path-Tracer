#include "scene/bvh.cuh"

#include <algorithm>
#include <limits>

namespace
{
constexpr uint32_t maxTrianglesPerLeaf = 4;

Aabb emptyAabb() {
    float maximum = std::numeric_limits<float>::max();
    return Aabb{Vec3(maximum, maximum, maximum), Vec3(-maximum, -maximum, -maximum)};
}

void grow(Aabb& bounds, const Vec3& point) {
    bounds.minimum.x = std::min(bounds.minimum.x, point.x);
    bounds.minimum.y = std::min(bounds.minimum.y, point.y);
    bounds.minimum.z = std::min(bounds.minimum.z, point.z);
    bounds.maximum.x = std::max(bounds.maximum.x, point.x);
    bounds.maximum.y = std::max(bounds.maximum.y, point.y);
    bounds.maximum.z = std::max(bounds.maximum.z, point.z);
}

Aabb triangleBounds(const Triangle& triangle) {
    Aabb bounds = emptyAabb();
    grow(bounds, triangle.v0);
    grow(bounds, triangle.v1);
    grow(bounds, triangle.v2);
    return bounds;
}

Vec3 triangleCentroid(const Triangle& triangle) {
    return (triangle.v0 + triangle.v1 + triangle.v2) / 3.0f;
}

float component(const Vec3& value, uint32_t axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

uint32_t longestAxis(const Aabb& bounds) {
    Vec3 extent = bounds.maximum - bounds.minimum;

    if (extent.x >= extent.y && extent.x >= extent.z)
        return 0;

    return extent.y >= extent.z ? 1 : 2;
}

uint32_t buildNode(HostBvh& bvh, const std::vector<Triangle>& triangles, uint32_t firstTriangle, uint32_t triangleCount) {
    uint32_t nodeIndex = static_cast<uint32_t>(bvh.nodes.size());
    bvh.nodes.push_back(BvhNode{});

    Aabb bounds = emptyAabb();
    Aabb centroidBounds = emptyAabb();

    for (uint32_t i = 0; i < triangleCount; ++i) {
        const Triangle& triangle = triangles[bvh.triangleIndices[firstTriangle + i]];
        Aabb boundsForTriangle = triangleBounds(triangle);
        grow(bounds, boundsForTriangle.minimum);
        grow(bounds, boundsForTriangle.maximum);
        grow(centroidBounds, triangleCentroid(triangle));
    }

    BvhNode& node = bvh.nodes[nodeIndex];
    node.bounds = bounds;

    if (triangleCount <= maxTrianglesPerLeaf) {
        node.firstTriangle = firstTriangle;
        node.triangleCount = triangleCount;
        return nodeIndex;
    }

    uint32_t axis = longestAxis(centroidBounds);
    uint32_t middle = firstTriangle + triangleCount / 2;

    std::nth_element(
        bvh.triangleIndices.begin() + firstTriangle,
        bvh.triangleIndices.begin() + middle,
        bvh.triangleIndices.begin() + firstTriangle + triangleCount,
        [&triangles, axis](uint32_t left, uint32_t right) {
            return component(triangleCentroid(triangles[left]), axis) < component(triangleCentroid(triangles[right]), axis);
        });

    uint32_t leftChild = buildNode(bvh, triangles, firstTriangle, middle - firstTriangle);
    uint32_t rightChild = buildNode(bvh, triangles, middle, triangleCount - (middle - firstTriangle));

    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.firstTriangle = 0;
    node.triangleCount = 0;

    return nodeIndex;
}
} // namespace

HostBvh buildTriangleBvh(const std::vector<Triangle>& triangles) {
    HostBvh bvh;
    bvh.triangleIndices.resize(triangles.size());

    for (uint32_t i = 0; i < triangles.size(); ++i)
        bvh.triangleIndices[i] = i;

    if (!triangles.empty()) {
        bvh.nodes.reserve(triangles.size() * 2);
        buildNode(bvh, triangles, 0, static_cast<uint32_t>(triangles.size()));
    }

    return bvh;
}
