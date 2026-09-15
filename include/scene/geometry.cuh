#pragma once

#include <cstdint>
#include "../core/ray.cuh"

struct Sphere {
    Vec3 center;
    float radius;

    uint32_t material;
};

struct Vec2 {
    float x;
    float y;
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;

    Vec3 n0;
    Vec3 n1;
    Vec3 n2;

    Vec2 uv0;
    Vec2 uv1;
    Vec2 uv2;

    uint32_t material;
    uint32_t lightIndex;
    bool hasVertexNormals;
    bool hasTexcoords;
};

constexpr uint32_t invalidLightIndex = 0xffffffffu;

struct TriangleLight {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
    Vec3 normal;
    float area;
    uint32_t material;
    float selectionPdf;
};

struct LightAliasEntry {
    float probability;
    uint32_t alias;
};

struct Hit {
    float t;
    Vec3 position;
    Vec3 normal;
    Vec2 uv;

    uint32_t material;
    uint32_t lightIndex;
};
