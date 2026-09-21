#pragma once

#include <cstdint>
#include <vector>

struct DistributedTile {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
};

struct RenderTaskId {
    uint32_t tileOrdinal;
    uint32_t sampleStart;
    uint32_t sampleCount;

    bool operator==(const RenderTaskId& other) const {
        return tileOrdinal == other.tileOrdinal &&
            sampleStart == other.sampleStart &&
            sampleCount == other.sampleCount;
    }
};

struct RenderTask {
    uint64_t jobFingerprint;
    RenderTaskId id;
    DistributedTile tile;
};

// Host/network representation of an unnormalized RGB tile sum. It intentionally
// does not depend on CUDA's Vec3 type so the coordinator remains host-only.
struct DistributedRadiance {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    DistributedRadiance& operator+=(const DistributedRadiance& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

struct DistributedTaskResult {
    uint64_t jobFingerprint = 0;
    RenderTaskId taskId{};
    DistributedTile tile{};
    uint32_t sampleCount = 0;
    std::vector<DistributedRadiance> radiance;
};

// Produces a deterministic task table. Every image pixel appears once per
// sample index, and task ordering is stable across coordinator restarts.
std::vector<RenderTask> createRenderTasks(
    uint32_t width,
    uint32_t height,
    uint32_t tileWidth,
    uint32_t tileHeight,
    uint32_t samplesPerPixel,
    uint32_t samplesPerTask,
    uint64_t jobFingerprint);

// A stable key for persistence, duplicate-result rejection, and checkpoint
// completion bitmaps. The job fingerprint is included to prevent accidental
// reuse of a task key across different renders.
uint64_t computeRenderTaskKey(uint64_t jobFingerprint, const RenderTaskId& id);
