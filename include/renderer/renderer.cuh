#pragma once

#include <cstdint>

#include "../scene/camera.cuh"
#include "../scene/scene.cuh"
#include "path_state.cuh"
#include "queues.cuh"

__global__
void generatePrimaryRays(
    RayQueue queue,
    PathState* pathStates,
    Camera camera,
    uint32_t width,
    uint32_t height,
    uint32_t sampleIndex
);

__global__
void intersectScene(
    RayQueue rays,
    HitWorkItem* hitCandidates,
    MissWorkItem* missCandidates,
    uint8_t* hitFlags,
    uint8_t* missFlags,
    Scene scene
);

__global__
void shadeMisses(
    MissQueue misses,
    PathState* pathStates,
    Vec3* framebuffer
);

__global__
void shadeHits(
    HitQueue hits,
    PathState* pathStates,
    Scene scene,
    uint32_t maxDepth,
    uint32_t russianRouletteStartDepth,
    Vec3* framebuffer
);

__global__
void prepareNextRays(
    const PathState* pathStates,
    uint32_t pathCount,
    RayWorkItem* candidates,
    uint8_t* activeFlags
);
