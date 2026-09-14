#include "renderer/tile_scheduler.cuh"

#include <algorithm>

std::vector<RenderTile> createRenderTiles(uint32_t width, uint32_t height, uint32_t tileWidth, uint32_t tileHeight) {
    std::vector<RenderTile> tiles;

    for (uint32_t y = 0; y < height; y += tileHeight) {
        for (uint32_t x = 0; x < width; x += tileWidth) {
            RenderTile tile;
            tile.x = x;
            tile.y = y;
            tile.width = std::min(tileWidth, width - x);
            tile.height = std::min(tileHeight, height - y);
            tile.advanceSample = 0;
            tiles.push_back(tile);
        }
    }

    return tiles;
}
