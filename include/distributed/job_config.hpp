#pragma once

#include "../config.hpp"

// Network-safe render settings selected by the coordinator. Worker-local
// connection, cache, checkpoint, and output settings intentionally stay local.
struct DistributedWorkerConfiguration {
    uint32_t width = 1920;
    uint32_t height = 1080;
    uint32_t tileWidth = 960;
    uint32_t tileHeight = 540;
    uint32_t maxDepth = 32;
    uint32_t russianRouletteStartDepth = 12;
    uint32_t samplesPerPixel = 512;
    uint32_t blockSize = 256;
    bool tiledRendering = true;
    bool intersectionDebug = false;
    bool shadingNormalDebug = false;
    bool persistentWavefront = false;
    bool profileQueues = false;
    bool enableCaustics = false;
    bool enableCausticGather = true;
    uint32_t causticPhotonCount = 1048576;
    uint32_t causticMaxDepth = 8;
    float causticGatherRadius = 0.06f;
    bool pushAssets = false;
    SceneConfig scene;
};

inline DistributedWorkerConfiguration makeDistributedWorkerConfiguration(
    const RenderConfig& render,
    const SceneConfig& scene) {
    DistributedWorkerConfiguration configuration;
    configuration.width = render.width;
    configuration.height = render.height;
    configuration.tileWidth = render.tileWidth;
    configuration.tileHeight = render.tileHeight;
    configuration.maxDepth = render.maxDepth;
    configuration.russianRouletteStartDepth = render.russianRouletteStartDepth;
    configuration.samplesPerPixel = render.samplesPerPixel;
    configuration.blockSize = render.blockSize;
    configuration.tiledRendering = render.tiledRendering;
    configuration.intersectionDebug = render.intersectionDebug;
    configuration.shadingNormalDebug = render.shadingNormalDebug;
    configuration.persistentWavefront = render.persistentWavefront;
    configuration.profileQueues = render.profileQueues;
    configuration.enableCaustics = render.enableCaustics;
    configuration.enableCausticGather = render.enableCausticGather;
    configuration.causticPhotonCount = render.causticPhotonCount;
    configuration.causticMaxDepth = render.causticMaxDepth;
    configuration.causticGatherRadius = render.causticGatherRadius;
    configuration.pushAssets = render.pushAssets;
    configuration.scene = scene;
    return configuration;
}

inline void applyDistributedWorkerConfiguration(
    const DistributedWorkerConfiguration& configuration,
    RenderConfig& render,
    SceneConfig& scene) {
    render.width = configuration.width;
    render.height = configuration.height;
    render.tileWidth = configuration.tileWidth;
    render.tileHeight = configuration.tileHeight;
    render.maxDepth = configuration.maxDepth;
    render.russianRouletteStartDepth = configuration.russianRouletteStartDepth;
    render.samplesPerPixel = configuration.samplesPerPixel;
    render.blockSize = configuration.blockSize;
    render.tiledRendering = configuration.tiledRendering;
    render.intersectionDebug = configuration.intersectionDebug;
    render.shadingNormalDebug = configuration.shadingNormalDebug;
    render.persistentWavefront = configuration.persistentWavefront;
    render.profileQueues = configuration.profileQueues;
    render.enableCaustics = configuration.enableCaustics;
    render.enableCausticGather = configuration.enableCausticGather;
    render.causticPhotonCount = configuration.causticPhotonCount;
    render.causticMaxDepth = configuration.causticMaxDepth;
    render.causticGatherRadius = configuration.causticGatherRadius;
    render.pushAssets = configuration.pushAssets;
    scene = configuration.scene;
}
