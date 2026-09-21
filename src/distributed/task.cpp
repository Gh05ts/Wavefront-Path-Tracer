#include "distributed/task.hpp"

#include <algorithm>

namespace
{
class TaskKeyBuilder {
public:
    void appendU32(uint32_t value) {
        for (uint32_t byte = 0; byte < sizeof(value); ++byte)
            appendByte(static_cast<uint8_t>((value >> (byte * 8)) & 0xffu));
    }

    void appendU64(uint64_t value) {
        for (uint32_t byte = 0; byte < sizeof(value); ++byte)
            appendByte(static_cast<uint8_t>((value >> (byte * 8)) & 0xffu));
    }

    uint64_t value() const {
        return value_;
    }

private:
    void appendByte(uint8_t byte) {
        value_ ^= byte;
        value_ *= 1099511628211ull;
    }

    uint64_t value_ = 1469598103934665603ull;
};
} // namespace

uint64_t computeRenderTaskKey(uint64_t jobFingerprint, const RenderTaskId& id) {
    TaskKeyBuilder builder;
    builder.appendU64(jobFingerprint);
    builder.appendU32(id.tileOrdinal);
    builder.appendU32(id.sampleStart);
    builder.appendU32(id.sampleCount);
    return builder.value();
}

std::vector<RenderTask> createRenderTasks(
    uint32_t width,
    uint32_t height,
    uint32_t tileWidth,
    uint32_t tileHeight,
    uint32_t samplesPerPixel,
    uint32_t samplesPerTask,
    uint64_t jobFingerprint) {
    std::vector<RenderTask> tasks;
    if (width == 0 || height == 0 || tileWidth == 0 || tileHeight == 0 ||
        samplesPerPixel == 0 || samplesPerTask == 0)
        return tasks;

    uint32_t tileOrdinal = 0;
    for (uint32_t y = 0; y < height; y += tileHeight) {
        for (uint32_t x = 0; x < width; x += tileWidth) {
            DistributedTile tile{
                x,
                y,
                std::min(tileWidth, width - x),
                std::min(tileHeight, height - y)};

            for (uint32_t sampleStart = 0; sampleStart < samplesPerPixel; sampleStart += samplesPerTask) {
                uint32_t sampleCount = std::min(samplesPerTask, samplesPerPixel - sampleStart);
                RenderTaskId id{tileOrdinal, sampleStart, sampleCount};
                tasks.push_back(RenderTask{jobFingerprint, id, tile});
            }

            ++tileOrdinal;
        }
    }

    return tasks;
}
