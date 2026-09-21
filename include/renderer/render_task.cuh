#pragma once

#include <cstdint>
#include <vector>

#include "../core/vec3.cuh"
#include "../distributed/task.hpp"

class RenderDriver;

struct RenderTaskResult {
    bool success = false;
    RenderTaskId taskId{};
    DistributedTile tile{};
    uint32_t sampleCount = 0;
    std::vector<Vec3> radiance;
};

// Local execution seam for future distributed workers. The returned values are
// unnormalized sums over the task's absolute sample range.
class RenderTaskRenderer {
public:
    explicit RenderTaskRenderer(RenderDriver& driver);

    RenderTaskResult render(const RenderTask& task);

private:
    RenderDriver& driver_;
};
