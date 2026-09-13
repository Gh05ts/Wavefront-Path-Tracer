#pragma once

#include <cstdint>

#include "../core/ray.cuh"
#include "../scene/geometry.cuh"

struct RayWorkItem {
    Ray ray;
    uint32_t pathIndex;
};

struct HitWorkItem {
    uint32_t pathIndex;

    Hit hit;
};

struct MissWorkItem {
    uint32_t pathIndex;
};

struct RayQueue {
    RayWorkItem* items;
    uint32_t* count;

    uint32_t capacity;
};

struct HitQueue {
    HitWorkItem* items;
    uint32_t* count;

    uint32_t capacity;
};

struct MissQueue {
    MissWorkItem* items;
    uint32_t* count;

    uint32_t capacity;
};
