#include "distributed/task.hpp"

#include <cstdint>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

namespace
{
bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "distributed_task_tests: " << message << '\n';
    return condition;
}
} // namespace

int main() {
    bool valid = true;
    constexpr uint64_t fingerprint = 0x123456789abcdef0ull;
    std::vector<RenderTask> tasks = createRenderTasks(5, 4, 3, 2, 10, 3, fingerprint);

    // Four spatial tiles and four sample batches per tile.
    valid &= check(tasks.size() == 16, "unexpected task count");

    std::set<uint64_t> keys;
    std::set<std::tuple<uint32_t, uint32_t, uint32_t>> taskIds;
    std::vector<uint32_t> pixelCoverage(5 * 4 * 10, 0);

    for (const RenderTask& task : tasks) {
        valid &= check(task.jobFingerprint == fingerprint, "task fingerprint changed");
        valid &= check(task.tile.width > 0 && task.tile.height > 0, "task tile has zero extent");
        valid &= check(task.tile.x + task.tile.width <= 5 && task.tile.y + task.tile.height <= 4, "task tile exceeds image bounds");
        valid &= check(task.id.sampleStart + task.id.sampleCount <= 10, "task sample range exceeds target");
        valid &= check(task.id.sampleCount > 0 && task.id.sampleCount <= 3, "invalid task batch size");

        keys.insert(computeRenderTaskKey(task.jobFingerprint, task.id));
        taskIds.emplace(task.id.tileOrdinal, task.id.sampleStart, task.id.sampleCount);

        for (uint32_t sample = task.id.sampleStart; sample < task.id.sampleStart + task.id.sampleCount; ++sample) {
            for (uint32_t y = task.tile.y; y < task.tile.y + task.tile.height; ++y) {
                for (uint32_t x = task.tile.x; x < task.tile.x + task.tile.width; ++x)
                    ++pixelCoverage[sample * 5 * 4 + y * 5 + x];
            }
        }
    }

    valid &= check(keys.size() == tasks.size(), "task keys are not unique");
    valid &= check(taskIds.size() == tasks.size(), "task IDs are not unique");
    for (uint32_t coverage : pixelCoverage)
        valid &= check(coverage == 1, "pixel/sample coverage is not exactly once");

    valid &= check(computeRenderTaskKey(fingerprint, tasks.front().id) ==
        computeRenderTaskKey(fingerprint, tasks.front().id), "task key is not stable");
    valid &= check(computeRenderTaskKey(fingerprint, tasks.front().id) !=
        computeRenderTaskKey(fingerprint + 1, tasks.front().id), "job fingerprint does not affect task key");

    valid &= check(createRenderTasks(0, 4, 2, 2, 4, 2, fingerprint).empty(), "zero-width image produced tasks");
    valid &= check(createRenderTasks(4, 4, 2, 2, 4, 0, fingerprint).empty(), "zero task batch produced tasks");
    return valid ? 0 : 1;
}
