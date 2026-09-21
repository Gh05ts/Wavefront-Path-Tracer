#include "renderer/renderer.cuh"
#include "renderer/queue_compaction.cuh"
#include "core/rng.cuh"

#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;

__global__
void generatePrimaryRays(RayQueue queue, PathState* pathStates, Camera camera, uint32_t width, uint32_t height, const uint32_t* sampleIndex, const RenderTile* tile) {
    uint32_t localPixel = blockIdx.x * blockDim.x + threadIdx.x;
    RenderTile renderTile = *tile;
    uint32_t tilePixelCount = renderTile.width * renderTile.height;

    if (localPixel >= tilePixelCount)
        return;

    uint32_t x = renderTile.x + localPixel % renderTile.width;
    uint32_t y = renderTile.y + localPixel / renderTile.width;
    uint32_t pixel = y * width + x;

    uint32_t rngState = makeRngSeed(pixel ^ (*sampleIndex * 0x9e3779b9u));

    float u = (static_cast<float>(x) + randomFloat(rngState)) / static_cast<float>(width);
    float v = (static_cast<float>(y) + randomFloat(rngState)) / static_cast<float>(height);

    Ray ray = camera.generateRay(u, v);

    pathStates[localPixel] = PathState{ray, Vec3(1.0f, 1.0f, 1.0f), Vec3(0.0f, 0.0f, 0.0f), pixel, 0, rngState, 0xffffffffu, 1.0f, 0.0f, true, true};

    uint32_t outputIndex = atomicAdd(queue.count, 1);

    if (outputIndex >= queue.capacity)
        return;

    queue.items[outputIndex] = RayWorkItem{ray, localPixel};
}

__global__
void advanceSampleIndex(uint32_t* sampleIndex, const RenderTile* tile) {
    if (tile->advanceSample)
        ++*sampleIndex;
}

#include "renderer/device_sampling.cuh"
#include "renderer/device_traversal.cuh"
#include "renderer/device_path_shading.cuh"

__global__
void persistentWavefrontTrace(RayQueue initialRays, RayQueue secondaryRays, PathState* pathStates, uint32_t* queueCounts, Camera camera, uint32_t width, uint32_t height, const uint32_t* sampleIndex, const RenderTile* tile, Scene scene, uint32_t maxDepth, uint32_t russianRouletteStartDepth, bool intersectionDebug, bool shadingNormalDebug, Vec3* framebuffer, PhotonGrid photonGrid, float photonGatherRadius) {
    cg::grid_group grid = cg::this_grid();
    uint32_t workerIndex = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t workerCount = gridDim.x * blockDim.x;
    RenderTile renderTile = *tile;
    uint32_t tilePixelCount = renderTile.width * renderTile.height;

    // Initialize one path and one ray work item per tile pixel. The queue is
    // already compact by construction, so this first wave needs no atomics.
    for (uint32_t localPixel = workerIndex; localPixel < tilePixelCount; localPixel += workerCount) {
        uint32_t x = renderTile.x + localPixel % renderTile.width;
        uint32_t y = renderTile.y + localPixel / renderTile.width;
        uint32_t pixel = y * width + x;
        uint32_t rngState = makeRngSeed(pixel ^ (*sampleIndex * 0x9e3779b9u));
        float u = (static_cast<float>(x) + randomFloat(rngState)) / static_cast<float>(width);
        float v = (static_cast<float>(y) + randomFloat(rngState)) / static_cast<float>(height);
        Ray ray = camera.generateRay(u, v);
        pathStates[localPixel] = PathState{ray, Vec3(1.0f, 1.0f, 1.0f), Vec3(0.0f, 0.0f, 0.0f), pixel, 0, rngState, 0xffffffffu, 1.0f, 0.0f, true, true};
        initialRays.items[localPixel] = RayWorkItem{ray, localPixel};
    }

    if (workerIndex == 0)
        *initialRays.count = min(tilePixelCount, initialRays.capacity);
    grid.sync();

    RayQueue currentRays = initialRays;
    RayQueue nextRays = secondaryRays;

    for (uint32_t bounce = 0; bounce < maxDepth; ++bounce) {
        uint32_t currentCount = *currentRays.count;
        if (workerIndex == 0 && queueCounts != nullptr)
            queueCounts[bounce] = currentCount;
        if (currentCount == 0)
            break;

        if (workerIndex == 0)
            *nextRays.count = 0;
        grid.sync();

        // This is the device-side wavefront scheduler. Resident blocks drain
        // queue chunks, then compact active continuations with warp ballots.
        // There is one global reservation per warp/chunk instead of one per
        // surviving path, without a block-wide barrier for every chunk.
        uint32_t blockStride = gridDim.x * blockDim.x;

        for (uint32_t chunkStart = blockIdx.x * blockDim.x; chunkStart < currentCount; chunkStart += blockStride) {
            uint32_t index = chunkStart + threadIdx.x;
            bool active = false;
            RayWorkItem continuation{};
            if (index < currentCount) {
                RayWorkItem work = currentRays.items[index];
                PathState& path = pathStates[work.pathIndex];
                TraceResult result = traceRay(work.ray, scene);
                shadePersistentPath(path, result, scene, maxDepth, russianRouletteStartDepth, intersectionDebug, shadingNormalDebug, framebuffer, photonGrid, photonGatherRadius);
                active = path.active;
                continuation = RayWorkItem{path.ray, work.pathIndex};
            }

            appendWarpCompacted(active, continuation, nextRays.items, nextRays.count, nextRays.capacity);
        }

        grid.sync();
        RayQueue swap = currentRays;
        currentRays = nextRays;
        nextRays = swap;
    }
}
