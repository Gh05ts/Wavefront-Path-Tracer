#pragma once

#include <cstdint>

#include "../core/ray.cuh"
#include "../scene/geometry.cuh"

struct RayWorkItem {
    Ray ray;
    uint32_t pathIndex;
};

struct Photon {
    Vec3 position;
    Vec3 direction;
    Vec3 flux;
    uint32_t rngState;
    uint32_t specularBounces;
    uint32_t spectralChannel;
    bool valid;
};

struct PhotonGrid {
    Photon* photons;
    uint32_t* heads;
    uint32_t* next;
    uint32_t resolution;
    Vec3 minimum;
    Vec3 maximum;
};

struct PhotonQueue {
    Photon* items;
    uint32_t* count;
    uint32_t capacity;
};

struct HitWorkItem {
    uint32_t pathIndex;

    Hit hit;
};

struct MissWorkItem {
    uint32_t pathIndex;
};

struct IntersectionResult {
    Hit hit;
    bool didHit;
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
