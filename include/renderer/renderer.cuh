#pragma once

#include <cstdint>

#include "../scene/camera.cuh"
#include "../scene/scene.cuh"
#include "path_state.cuh"
#include "queues.cuh"
#include "tile_scheduler.cuh"

__global__
void generatePrimaryRays(
    RayQueue queue,
    PathState* pathStates,
    Camera camera,
    uint32_t width,
    uint32_t height,
    const uint32_t* sampleIndex,
    const RenderTile* tile
);

__global__
void advanceSampleIndex(uint32_t* sampleIndex, const RenderTile* tile);

__global__
void intersectScene(
    RayQueue rays,
    IntersectionResult* results,
    Scene scene
);

__global__
void shadePaths(
    RayQueue rays,
    const IntersectionResult* results,
    RayQueue nextRays,
    PathState* pathStates,
    Scene scene,
    uint32_t maxDepth,
    uint32_t russianRouletteStartDepth,
    bool intersectionDebug,
    bool shadingNormalDebug,
    Vec3* framebuffer
);
