#pragma once

#include <cstdint>
#include <vector>

struct RenderTile {
    uint32_t x;
    uint32_t y;
    uint32_t width;
    uint32_t height;
    uint32_t advanceSample;
};

std::vector<RenderTile> createRenderTiles(uint32_t width, uint32_t height, uint32_t tileWidth, uint32_t tileHeight);
