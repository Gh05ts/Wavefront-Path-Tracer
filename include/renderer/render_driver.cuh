#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "../config.hpp"
#include "../scene/camera.cuh"
#include "path_state.cuh"
#include "queues.cuh"
#include "render_session.cuh"
#include "tile_scheduler.cuh"

struct RenderResources {
    Photon* devicePhotons = nullptr;
    uint32_t* devicePhotonMaterialHitCounts = nullptr;
    PhotonQueue photonQueue{};
    PhotonGrid photonGrid{};
    PhotonGrid shadingPhotonGrid{};

    PathState* devicePathStates = nullptr;
    uint32_t* deviceQueueCounts = nullptr;

    RayWorkItem* deviceRays = nullptr;
    uint32_t* deviceRayCount = nullptr;
    RayWorkItem* deviceNextRays = nullptr;
    uint32_t* deviceNextRayCount = nullptr;
    IntersectionResult* deviceIntersectionResults = nullptr;
    RayQueue rayQueue{};
    RayQueue nextRayQueue{};

    RenderSession renderSession{};
    RenderTile* deviceRenderTile = nullptr;

    cudaStream_t traceStream = nullptr;
    AsyncCheckpointWriter* checkpointWriter = nullptr;
    cudaGraph_t traceGraph = nullptr;
    cudaGraphExec_t traceGraphExec = nullptr;
    cudaEvent_t traceStart = nullptr;
    cudaEvent_t traceEnd = nullptr;

    RenderResources() = default;
    ~RenderResources();

    RenderResources(const RenderResources&) = delete;
    RenderResources& operator=(const RenderResources&) = delete;

    void release();
};

class RenderDriver {
public:
    RenderDriver(const DeviceScene& deviceScene, Camera camera, RenderConfig config, SceneConfig sceneConfig);

    RenderDriver(const RenderDriver&) = delete;
    RenderDriver& operator=(const RenderDriver&) = delete;

    int run();

private:
    friend class RenderTaskRenderer;

    bool prepare();
    bool validateConfiguration() const;
    bool preparePersistentLaunch();
    void buildPhotonMap();
    void allocateRenderResources();
    void createWavefrontGraph();
    void renderSamples(const std::vector<RenderTile>& renderTiles, uint32_t completedSamples);
    void launchPersistentTile(const RenderTile& renderTile);
    void launchCapturedTile();
    void writeFinalImage();

    const DeviceScene& deviceScene_;
    Camera camera_;
    RenderConfig config_;
    SceneConfig sceneConfig_;
    RenderResources resources_;
    uint32_t width_ = 0;
    uint32_t height_ = 0;
    uint32_t pixelCount_ = 0;
    uint32_t renderQueueCapacity_ = 0;
    uint32_t blockCount_ = 0;
    uint32_t persistentBlockCount_ = 0;
    uint64_t renderFingerprint_ = 0;
    bool prepared_ = false;
};
