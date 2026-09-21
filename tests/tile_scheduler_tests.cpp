#include "renderer/tile_scheduler.cuh"

#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "tile_scheduler_tests: " << message << '\n';
    return condition;
}

bool checkCoverage(uint32_t width, uint32_t height, const std::vector<RenderTile>& tiles) {
    std::vector<uint32_t> coverage(width * height, 0);
    bool valid = true;

    for (const RenderTile& tile : tiles) {
        valid &= check(tile.x < width && tile.y < height, "tile origin is outside the image");
        valid &= check(tile.width > 0 && tile.height > 0, "tile has zero extent");
        valid &= check(tile.x + tile.width <= width && tile.y + tile.height <= height, "tile exceeds image bounds");
        valid &= check(tile.advanceSample == 0, "scheduler should not assign sample advancement");

        for (uint32_t y = tile.y; y < tile.y + tile.height; ++y) {
            for (uint32_t x = tile.x; x < tile.x + tile.width; ++x)
                ++coverage[y * width + x];
        }
    }

    for (uint32_t count : coverage)
        valid &= check(count == 1, "pixel coverage is not exactly once");
    return valid;
}
} // namespace

int main() {
    bool valid = true;

    const std::vector<RenderTile> partialTiles = createRenderTiles(5, 4, 3, 2);
    valid &= check(partialTiles.size() == 4, "5x4 image with 3x2 tiles should produce four tiles");
    valid &= checkCoverage(5, 4, partialTiles);

    const std::vector<RenderTile> fullFrame = createRenderTiles(1920, 1080, 1920, 1080);
    valid &= check(fullFrame.size() == 1, "full-frame tile size should produce one tile");
    valid &= checkCoverage(1920, 1080, fullFrame);

    return valid ? 0 : 1;
}
