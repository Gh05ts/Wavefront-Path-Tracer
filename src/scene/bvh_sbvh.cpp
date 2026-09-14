#include "scene/bvh.cuh"

#include <algorithm>
#include <limits>

namespace
{
constexpr uint32_t maxTrianglesPerLeaf = 4;
constexpr uint32_t spatialSplitBins = 8;
constexpr uint32_t maxBvhDepth = 48;

struct PrimitiveReference {
    uint32_t triangleIndex;
    Aabb bounds;
};

struct SpatialSplit {
    uint32_t axis;
    float position;
    float cost;
};

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

void grow(Aabb& bounds, const Aabb& other) {
    grow(bounds, other.minimum);
    grow(bounds, other.maximum);
}

Aabb triangleBounds(const Triangle& triangle) {
    Aabb bounds = emptyAabb();
    grow(bounds, triangle.v0);
    grow(bounds, triangle.v1);
    grow(bounds, triangle.v2);
    return bounds;
}

float component(const Vec3& value, uint32_t axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

float surfaceArea(const Aabb& bounds) {
    Vec3 extent = bounds.maximum - bounds.minimum;
    return 2.0f * (extent.x * extent.y + extent.y * extent.z + extent.z * extent.x);
}

Vec3 centroid(const PrimitiveReference& reference) {
    return 0.5f * (reference.bounds.minimum + reference.bounds.maximum);
}

uint32_t longestAxis(const Aabb& bounds) {
    Vec3 extent = bounds.maximum - bounds.minimum;

    if (extent.x >= extent.y && extent.x >= extent.z)
        return 0;

    return extent.y >= extent.z ? 1 : 2;
}

Aabb clippedBounds(const Aabb& bounds, uint32_t axis, float position, bool left) {
    Aabb clipped = bounds;

    if (left) {
        if (axis == 0)
            clipped.maximum.x = position;
        else if (axis == 1)
            clipped.maximum.y = position;
        else
            clipped.maximum.z = position;
    } else {
        if (axis == 0)
            clipped.minimum.x = position;
        else if (axis == 1)
            clipped.minimum.y = position;
        else
            clipped.minimum.z = position;
    }

    return clipped;
}

SpatialSplit findBestSpatialSplit(const std::vector<PrimitiveReference>& references, const Aabb& bounds) {
    SpatialSplit best{0, 0.0f, std::numeric_limits<float>::max()};

    for (uint32_t axis = 0; axis < 3; ++axis) {
        float minimum = component(bounds.minimum, axis);
        float maximum = component(bounds.maximum, axis);
        float extent = maximum - minimum;

        if (extent <= 1e-6f)
            continue;

        for (uint32_t bin = 1; bin < spatialSplitBins; ++bin) {
            float position = minimum + extent * static_cast<float>(bin) / static_cast<float>(spatialSplitBins);
            Aabb leftBounds = emptyAabb();
            Aabb rightBounds = emptyAabb();
            uint32_t leftCount = 0;
            uint32_t rightCount = 0;

            for (const PrimitiveReference& reference : references) {
                if (component(reference.bounds.minimum, axis) < position) {
                    grow(leftBounds, clippedBounds(reference.bounds, axis, position, true));
                    ++leftCount;
                }

                if (component(reference.bounds.maximum, axis) > position) {
                    grow(rightBounds, clippedBounds(reference.bounds, axis, position, false));
                    ++rightCount;
                }
            }

            if (leftCount == 0 || rightCount == 0 || (leftCount == references.size() && rightCount == references.size()))
                continue;

            float cost = surfaceArea(leftBounds) * static_cast<float>(leftCount) + surfaceArea(rightBounds) * static_cast<float>(rightCount);

            if (cost < best.cost)
                best = SpatialSplit{axis, position, cost};
        }
    }

    return best;
}

void makeLeaf(HostBvh& bvh, uint32_t nodeIndex, const std::vector<PrimitiveReference>& references, const Aabb& bounds) {
    uint32_t firstTriangle = static_cast<uint32_t>(bvh.triangleIndices.size());

    for (const PrimitiveReference& reference : references)
        bvh.triangleIndices.push_back(reference.triangleIndex);

    BvhNode& node = bvh.nodes[nodeIndex];
    node.bounds = bounds;
    node.firstTriangle = firstTriangle;
    node.triangleCount = static_cast<uint32_t>(references.size());
}

uint32_t buildNode(HostBvh& bvh, std::vector<PrimitiveReference>&& references, uint32_t depth) {
    uint32_t nodeIndex = static_cast<uint32_t>(bvh.nodes.size());
    bvh.nodes.push_back(BvhNode{});

    Aabb bounds = emptyAabb();
    for (const PrimitiveReference& reference : references)
        grow(bounds, reference.bounds);

    if (references.size() <= maxTrianglesPerLeaf || depth == maxBvhDepth) {
        makeLeaf(bvh, nodeIndex, references, bounds);
        return nodeIndex;
    }

    SpatialSplit split = findBestSpatialSplit(references, bounds);
    std::vector<PrimitiveReference> leftReferences;
    std::vector<PrimitiveReference> rightReferences;

    if (split.cost < surfaceArea(bounds) * static_cast<float>(references.size())) {
        leftReferences.reserve(references.size());
        rightReferences.reserve(references.size());

        for (const PrimitiveReference& reference : references) {
            if (component(reference.bounds.minimum, split.axis) < split.position)
                leftReferences.push_back(PrimitiveReference{reference.triangleIndex, clippedBounds(reference.bounds, split.axis, split.position, true)});

            if (component(reference.bounds.maximum, split.axis) > split.position)
                rightReferences.push_back(PrimitiveReference{reference.triangleIndex, clippedBounds(reference.bounds, split.axis, split.position, false)});
        }
    }

    if (leftReferences.empty() || rightReferences.empty()) {
        uint32_t axis = longestAxis(bounds);
        std::sort(references.begin(), references.end(), [axis](const PrimitiveReference& left, const PrimitiveReference& right) {
            return component(centroid(left), axis) < component(centroid(right), axis);
        });

        uint32_t middle = static_cast<uint32_t>(references.size() / 2);
        leftReferences.assign(references.begin(), references.begin() + middle);
        rightReferences.assign(references.begin() + middle, references.end());
    }

    uint32_t leftChild = buildNode(bvh, std::move(leftReferences), depth + 1);
    uint32_t rightChild = buildNode(bvh, std::move(rightReferences), depth + 1);

    BvhNode& node = bvh.nodes[nodeIndex];
    node.bounds = bounds;
    node.leftChild = leftChild;
    node.rightChild = rightChild;
    node.firstTriangle = 0;
    node.triangleCount = 0;

    return nodeIndex;
}
} // namespace

HostBvh buildSpatialSplitBvh(const std::vector<Triangle>& triangles) {
    HostBvh bvh;
    bvh.nodes.reserve(triangles.size() * 2);
    bvh.triangleIndices.reserve(triangles.size() * 2);

    std::vector<PrimitiveReference> references;
    references.reserve(triangles.size());

    for (uint32_t i = 0; i < triangles.size(); ++i)
        references.push_back(PrimitiveReference{i, triangleBounds(triangles[i])});

    if (!references.empty())
        buildNode(bvh, std::move(references), 0);

    return bvh;
}
