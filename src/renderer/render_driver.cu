#include "renderer/render_driver.cuh"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "renderer/image_output.hpp"
#include "renderer/render_fingerprint.hpp"
#include "renderer/renderer.cuh"

namespace
{
void checkCuda(cudaError_t error, const char* expression, const char* filename, int line) {
    if (error == cudaSuccess)
        return;

    std::cerr << "CUDA error: " << cudaGetErrorString(error)
              << " (" << expression << ", " << filename << ':' << line << ")\n";
    std::exit(1);
}

#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

template <typename T>
void freeDevice(T*& pointer) {
    if (pointer == nullptr)
        return;
    CUDA_CHECK(cudaFree(pointer));
    pointer = nullptr;
}

bool isCausticPreset(ScenePreset preset) {
    return preset == ScenePreset::Cornell || preset == ScenePreset::Hurricane ||
        preset == ScenePreset::Prism || preset == ScenePreset::Crystal;
}

float elapsedMilliseconds(std::chrono::steady_clock::time_point begin, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<float, std::milli>(end - begin).count();
}
} // namespace

void RenderResources::release() {
    if (traceStream != nullptr)
        CUDA_CHECK(cudaStreamSynchronize(traceStream));

    destroyAsyncCheckpointWriter(checkpointWriter);
    checkpointWriter = nullptr;

    if (traceGraphExec != nullptr) {
        CUDA_CHECK(cudaGraphExecDestroy(traceGraphExec));
        traceGraphExec = nullptr;
    }
    if (traceGraph != nullptr) {
        CUDA_CHECK(cudaGraphDestroy(traceGraph));
        traceGraph = nullptr;
    }

    if (traceStart != nullptr) {
        CUDA_CHECK(cudaEventDestroy(traceStart));
        traceStart = nullptr;
    }
    if (traceEnd != nullptr) {
        CUDA_CHECK(cudaEventDestroy(traceEnd));
        traceEnd = nullptr;
    }
    if (traceStream != nullptr) {
        CUDA_CHECK(cudaStreamDestroy(traceStream));
        traceStream = nullptr;
    }

    freeDevice(devicePhotons);
    freeDevice(devicePhotonMaterialHitCounts);
    freeDevice(photonGrid.heads);
    freeDevice(photonGrid.next);
    photonGrid = PhotonGrid{};
    shadingPhotonGrid = PhotonGrid{};
    photonQueue = PhotonQueue{};

    freeDevice(devicePathStates);
    freeDevice(deviceQueueCounts);
    freeDevice(deviceRays);
    freeDevice(deviceRayCount);
    freeDevice(deviceNextRays);
    freeDevice(deviceNextRayCount);
    freeDevice(deviceIntersectionResults);
    rayQueue = RayQueue{};
    nextRayQueue = RayQueue{};

    if (renderSession.deviceFramebuffer != nullptr || renderSession.deviceSampleIndex != nullptr)
        destroyRenderSession(renderSession);
    renderSession = RenderSession{};
    freeDevice(deviceRenderTile);
}

RenderResources::~RenderResources() {
    release();
}

RenderDriver::RenderDriver(const DeviceScene& deviceScene, Camera camera, RenderConfig config, SceneConfig sceneConfig)
    : deviceScene_(deviceScene), camera_(camera), config_(config), sceneConfig_(sceneConfig),
      renderFingerprint_(computeRenderFingerprint(config_, sceneConfig_)) {}

bool RenderDriver::validateConfiguration() const {
    if (config_.width == 0 || config_.height == 0) {
        std::cerr << "Render dimensions must be greater than zero\n";
        return false;
    }
    if (config_.samplesPerPixel == 0 || config_.maxDepth == 0) {
        std::cerr << "Samples per pixel and maximum depth must be greater than zero\n";
        return false;
    }
    if (config_.blockSize == 0 || config_.blockSize > 1024) {
        std::cerr << "CUDA block size must be between 1 and 1024\n";
        return false;
    }
    if (config_.tiledRendering && (config_.tileWidth == 0 || config_.tileHeight == 0)) {
        std::cerr << "Tile dimensions must be greater than zero\n";
        return false;
    }
    if (config_.checkpointBufferCount == 0 || config_.checkpointProgressPercent == 0) {
        std::cerr << "Checkpoint buffer count and progress interval must be greater than zero\n";
        return false;
    }
    if (config_.checkpointProgressPercent > 100) {
        std::cerr << "Checkpoint progress interval cannot exceed 100 percent\n";
        return false;
    }
    if (config_.enableCaustics && config_.causticPhotonCount == 0) {
        std::cerr << "Photon count must be greater than zero when caustics are enabled\n";
        return false;
    }
    if (config_.causticGatherRadius <= 0.0f) {
        std::cerr << "Caustic gather radius must be greater than zero\n";
        return false;
    }
    return true;
}

bool RenderDriver::preparePersistentLaunch() {
    persistentBlockCount_ = blockCount_;
    if (!config_.persistentWavefront)
        return true;

    int cooperativeLaunch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&cooperativeLaunch, cudaDevAttrCooperativeLaunch, 0));
    if (!cooperativeLaunch) {
        std::cerr << "The selected CUDA device does not support cooperative launches required by --persistent-wavefront\n";
        return false;
    }

    int activeBlocksPerMultiprocessor = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &activeBlocksPerMultiprocessor,
        persistentWavefrontTrace,
        static_cast<int>(config_.blockSize),
        0));
    if (activeBlocksPerMultiprocessor <= 0) {
        std::cerr << "The selected block size cannot launch persistentWavefrontTrace\n";
        return false;
    }

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));
    uint32_t residentBlockLimit = static_cast<uint32_t>(activeBlocksPerMultiprocessor * deviceProperties.multiProcessorCount);
    persistentBlockCount_ = std::max(1u, std::min(blockCount_, residentBlockLimit));
    std::cout << "Persistent wavefront workers: " << persistentBlockCount_ << " cooperative blocks\n";
    return true;
}

void RenderDriver::buildPhotonMap() {
    if (!config_.enableCaustics || !isCausticPreset(sceneConfig_.preset))
        return;

    auto photonStart = std::chrono::steady_clock::now();
    // The renderer is non-recursive; 16 KiB is ample for the bounded
    // nearest-photon gather and avoids reserving excessive device memory.
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));
    CUDA_CHECK(cudaMalloc(&resources_.devicePhotons, sizeof(Photon) * config_.causticPhotonCount));
    CUDA_CHECK(cudaMalloc(&resources_.devicePhotonMaterialHitCounts, sizeof(uint32_t) * 3));
    CUDA_CHECK(cudaMemset(resources_.devicePhotonMaterialHitCounts, 0, sizeof(uint32_t) * 3));

    resources_.photonQueue.items = resources_.devicePhotons;
    resources_.photonQueue.capacity = config_.causticPhotonCount;
    uint32_t photonBlockCount = (config_.causticPhotonCount + config_.blockSize - 1) / config_.blockSize;
    bool spectralSampling = sceneConfig_.preset == ScenePreset::Prism || sceneConfig_.preset == ScenePreset::Crystal;

    emitPhotons<<<photonBlockCount, config_.blockSize>>>(resources_.photonQueue, deviceScene_.scene, config_.causticPhotonCount, 0x13579bdfu, spectralSampling);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto photonEmitEnd = std::chrono::steady_clock::now();

    tracePhotons<<<photonBlockCount, config_.blockSize>>>(resources_.photonQueue, deviceScene_.scene, config_.causticPhotonCount, config_.causticMaxDepth, resources_.devicePhotonMaterialHitCounts);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto photonTraceEnd = std::chrono::steady_clock::now();

    uint32_t photonMaterialHitCounts[3]{};
    CUDA_CHECK(cudaMemcpy(photonMaterialHitCounts, resources_.devicePhotonMaterialHitCounts, sizeof(photonMaterialHitCounts), cudaMemcpyDeviceToHost));

    resources_.photonGrid.photons = resources_.devicePhotons;
    resources_.photonGrid.resolution = 64;
    resources_.photonGrid.minimum = Vec3(-2.5f, -1.0f, -2.5f);
    resources_.photonGrid.maximum = Vec3(2.5f, 3.0f, 2.5f);
    uint32_t photonCellCount = resources_.photonGrid.resolution * resources_.photonGrid.resolution * resources_.photonGrid.resolution;
    CUDA_CHECK(cudaMalloc(&resources_.photonGrid.heads, sizeof(uint32_t) * photonCellCount));
    CUDA_CHECK(cudaMalloc(&resources_.photonGrid.next, sizeof(uint32_t) * config_.causticPhotonCount));
    CUDA_CHECK(cudaMemset(resources_.photonGrid.heads, 0xff, sizeof(uint32_t) * photonCellCount));
    CUDA_CHECK(cudaMemset(resources_.photonGrid.next, 0xff, sizeof(uint32_t) * config_.causticPhotonCount));

    buildPhotonGrid<<<photonBlockCount, config_.blockSize>>>(resources_.photonQueue, resources_.photonGrid, config_.causticPhotonCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto photonGridEnd = std::chrono::steady_clock::now();

    resources_.shadingPhotonGrid = resources_.photonGrid;
    if (!config_.enableCausticGather)
        resources_.shadingPhotonGrid.heads = nullptr;

    std::cout << "Traced " << config_.causticPhotonCount << " photons; Cornell diffuse hits: neutral=" << photonMaterialHitCounts[0]
              << ", red=" << photonMaterialHitCounts[1] << ", green=" << photonMaterialHitCounts[2] << '\n';
    std::cout << "Photon timings: emit=" << elapsedMilliseconds(photonStart, photonEmitEnd)
              << " ms, trace=" << elapsedMilliseconds(photonEmitEnd, photonTraceEnd)
              << " ms, grid=" << elapsedMilliseconds(photonTraceEnd, photonGridEnd) << " ms\n";
}

void RenderDriver::allocateRenderResources() {
    CUDA_CHECK(cudaMalloc(&resources_.devicePathStates, sizeof(PathState) * renderQueueCapacity_));
    if (config_.persistentWavefront && config_.profileQueues)
        CUDA_CHECK(cudaMalloc(&resources_.deviceQueueCounts, sizeof(uint32_t) * config_.maxDepth));

    resources_.renderSession = createRenderSession(width_, height_);

    CUDA_CHECK(cudaMalloc(&resources_.deviceRays, sizeof(RayWorkItem) * renderQueueCapacity_));
    CUDA_CHECK(cudaMalloc(&resources_.deviceRayCount, sizeof(uint32_t)));
    resources_.rayQueue.items = resources_.deviceRays;
    resources_.rayQueue.count = resources_.deviceRayCount;
    resources_.rayQueue.capacity = renderQueueCapacity_;

    CUDA_CHECK(cudaMalloc(&resources_.deviceNextRays, sizeof(RayWorkItem) * renderQueueCapacity_));
    CUDA_CHECK(cudaMalloc(&resources_.deviceNextRayCount, sizeof(uint32_t)));
    resources_.nextRayQueue.items = resources_.deviceNextRays;
    resources_.nextRayQueue.count = resources_.deviceNextRayCount;
    resources_.nextRayQueue.capacity = renderQueueCapacity_;

    if (!config_.persistentWavefront)
        CUDA_CHECK(cudaMalloc(&resources_.deviceIntersectionResults, sizeof(IntersectionResult) * renderQueueCapacity_));

    CUDA_CHECK(cudaMalloc(&resources_.deviceRenderTile, sizeof(RenderTile)));
    CUDA_CHECK(cudaStreamCreateWithFlags(&resources_.traceStream, cudaStreamNonBlocking));
    if (config_.distributedRole == DistributedRole::Local)
        resources_.checkpointWriter = createAsyncCheckpointWriter(width_, height_, config_.checkpointFilename, config_.checkpointBufferCount, renderFingerprint_);
}

void RenderDriver::createWavefrontGraph() {
    if (config_.persistentWavefront)
        return;

    CUDA_CHECK(cudaStreamBeginCapture(resources_.traceStream, cudaStreamCaptureModeGlobal));

    CUDA_CHECK(cudaMemsetAsync(resources_.deviceRayCount, 0, sizeof(uint32_t), resources_.traceStream));
    generatePrimaryRays<<<blockCount_, config_.blockSize, 0, resources_.traceStream>>>(
        resources_.rayQueue, resources_.devicePathStates, camera_, width_, height_,
        resources_.renderSession.deviceSampleIndex, resources_.deviceRenderTile);

    RayQueue currentRayQueue = resources_.rayQueue;
    RayQueue nextRays = resources_.nextRayQueue;
    for (uint32_t bounce = 0; bounce < config_.maxDepth; ++bounce) {
        CUDA_CHECK(cudaMemsetAsync(nextRays.count, 0, sizeof(uint32_t), resources_.traceStream));

        intersectScene<<<blockCount_, config_.blockSize, 0, resources_.traceStream>>>(
            currentRayQueue, resources_.deviceIntersectionResults, deviceScene_.scene);
        shadePaths<<<blockCount_, config_.blockSize, 0, resources_.traceStream>>>(
            currentRayQueue, resources_.deviceIntersectionResults, nextRays,
            resources_.devicePathStates, deviceScene_.scene, config_.maxDepth,
            config_.russianRouletteStartDepth, config_.intersectionDebug,
            config_.shadingNormalDebug, resources_.renderSession.deviceFramebuffer,
            resources_.shadingPhotonGrid, config_.causticGatherRadius);

        std::swap(currentRayQueue, nextRays);
    }

    advanceSampleIndex<<<1, 1, 0, resources_.traceStream>>>(
        resources_.renderSession.deviceSampleIndex, resources_.deviceRenderTile);

    CUDA_CHECK(cudaStreamEndCapture(resources_.traceStream, &resources_.traceGraph));
    CUDA_CHECK(cudaGraphInstantiate(&resources_.traceGraphExec, resources_.traceGraph, nullptr, nullptr, 0));
}

void RenderDriver::launchPersistentTile(const RenderTile& renderTile) {
    uint32_t tilePixelCount = renderTile.width * renderTile.height;
    uint32_t tileBlockCount = (tilePixelCount + config_.blockSize - 1) / config_.blockSize;
    uint32_t launchBlockCount = std::max(1u, std::min(tileBlockCount, persistentBlockCount_));

    RayQueue initialRays = resources_.rayQueue;
    RayQueue secondaryRays = resources_.nextRayQueue;
    PathState* pathStates = resources_.devicePathStates;
    uint32_t* queueCounts = resources_.deviceQueueCounts;
    Camera renderCamera = camera_;
    uint32_t renderWidth = width_;
    uint32_t renderHeight = height_;
    const uint32_t* renderSampleIndex = resources_.renderSession.deviceSampleIndex;
    RenderTile* renderTilePointer = resources_.deviceRenderTile;
    Scene renderScene = deviceScene_.scene;
    uint32_t renderMaxDepth = config_.maxDepth;
    uint32_t renderRussianRouletteStartDepth = config_.russianRouletteStartDepth;
    bool renderIntersectionDebug = config_.intersectionDebug;
    bool renderShadingNormalDebug = config_.shadingNormalDebug;
    Vec3* renderFramebuffer = resources_.renderSession.deviceFramebuffer;
    PhotonGrid renderPhotonGrid = resources_.shadingPhotonGrid;
    float renderPhotonGatherRadius = config_.causticGatherRadius;
    void* kernelArguments[] = {
        &initialRays,
        &secondaryRays,
        &pathStates,
        &queueCounts,
        &renderCamera,
        &renderWidth,
        &renderHeight,
        &renderSampleIndex,
        &renderTilePointer,
        &renderScene,
        &renderMaxDepth,
        &renderRussianRouletteStartDepth,
        &renderIntersectionDebug,
        &renderShadingNormalDebug,
        &renderFramebuffer,
        &renderPhotonGrid,
        &renderPhotonGatherRadius};

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(persistentWavefrontTrace),
        dim3(launchBlockCount), dim3(config_.blockSize), kernelArguments, 0,
        resources_.traceStream));
}

void RenderDriver::launchCapturedTile() {
    CUDA_CHECK(cudaGraphLaunch(resources_.traceGraphExec, resources_.traceStream));
}

void RenderDriver::renderSamples(const std::vector<RenderTile>& renderTiles, uint32_t completedSamples) {
    CUDA_CHECK(cudaEventCreate(&resources_.traceStart));
    CUDA_CHECK(cudaEventCreate(&resources_.traceEnd));
    CUDA_CHECK(cudaEventRecord(resources_.traceStart, resources_.traceStream));

    uint32_t checkpointStepSamples = std::max(1u, (config_.samplesPerPixel * config_.checkpointProgressPercent + 99) / 100);
    for (uint32_t sample = completedSamples; sample < config_.samplesPerPixel; ++sample) {
        for (uint32_t tileIndex = 0; tileIndex < renderTiles.size(); ++tileIndex) {
            RenderTile renderTile = renderTiles[tileIndex];
            renderTile.advanceSample = tileIndex + 1 == renderTiles.size();

            CUDA_CHECK(cudaMemcpyAsync(resources_.deviceRenderTile, &renderTile, sizeof(RenderTile), cudaMemcpyHostToDevice, resources_.traceStream));
            if (config_.persistentWavefront)
                launchPersistentTile(renderTile);
            else
                launchCapturedTile();
        }

        if (config_.persistentWavefront)
            advanceSampleIndex<<<1, 1, 0, resources_.traceStream>>>(resources_.renderSession.deviceSampleIndex, resources_.deviceRenderTile);

        std::cout << "Completed sample " << sample + 1 << " of " << config_.samplesPerPixel << '\n';

        uint32_t newCompletedSamples = sample + 1;
        if (newCompletedSamples % checkpointStepSamples == 0 && newCompletedSamples < config_.samplesPerPixel) {
            if (enqueueRenderSessionCheckpoint(*resources_.checkpointWriter, resources_.renderSession, newCompletedSamples, resources_.traceStream))
                std::cout << "Queued checkpoint at " << newCompletedSamples << " of " << config_.samplesPerPixel << " samples\n";
            else
                std::cout << "Skipped checkpoint at " << newCompletedSamples << " of " << config_.samplesPerPixel << " samples because all checkpoint buffers are busy\n";
        }
    }

    CUDA_CHECK(cudaEventRecord(resources_.traceEnd, resources_.traceStream));
    CUDA_CHECK(cudaEventSynchronize(resources_.traceEnd));

    float traceMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&traceMilliseconds, resources_.traceStart, resources_.traceEnd));
    std::cout << "Trace time (" << (config_.tiledRendering ? "tiled" : "full frame") << "): " << traceMilliseconds << " ms\n";

    if (config_.persistentWavefront && config_.profileQueues) {
        std::vector<uint32_t> queueCounts(config_.maxDepth);
        CUDA_CHECK(cudaMemcpy(queueCounts.data(), resources_.deviceQueueCounts, sizeof(uint32_t) * config_.maxDepth, cudaMemcpyDeviceToHost));
        std::cout << "Persistent queue occupancy (last work tile):";
        for (uint32_t bounce = 0; bounce < config_.maxDepth; ++bounce)
            std::cout << ' ' << queueCounts[bounce];
        std::cout << '\n';
    }
}

void RenderDriver::writeFinalImage() {
    std::vector<Vec3> pixels(pixelCount_);
    CUDA_CHECK(cudaMemcpy(pixels.data(), resources_.renderSession.deviceFramebuffer, sizeof(Vec3) * pixelCount_, cudaMemcpyDeviceToHost));

    uint32_t finalSampleCount = getRenderSessionSampleIndex(resources_.renderSession);
    if (finalSampleCount == 0) {
        std::cerr << "Render completed without accumulating any samples\n";
        std::exit(1);
    }

    for (Vec3& pixel : pixels)
        pixel = pixel / static_cast<float>(finalSampleCount);

    writePpm(config_.outputFilename, pixels, width_, height_);
    std::cout << "Wrote " << config_.outputFilename << '\n';
}

bool RenderDriver::prepare() {
    if (prepared_)
        return true;

    if (!validateConfiguration())
        return false;

    width_ = config_.width;
    height_ = config_.height;
    uint64_t pixelCount = static_cast<uint64_t>(width_) * height_;
    uint64_t queueCapacity = config_.tiledRendering ?
        static_cast<uint64_t>(config_.tileWidth) * config_.tileHeight : pixelCount;
    if (pixelCount > std::numeric_limits<uint32_t>::max() || queueCapacity > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "Render dimensions exceed the supported queue index range\n";
        return false;
    }
    pixelCount_ = static_cast<uint32_t>(pixelCount);
    renderQueueCapacity_ = static_cast<uint32_t>(queueCapacity);
    blockCount_ = (renderQueueCapacity_ + config_.blockSize - 1) / config_.blockSize;

    if (!preparePersistentLaunch())
        return false;

    buildPhotonMap();
    allocateRenderResources();
    prepared_ = true;
    return true;
}

int RenderDriver::run() {
    if (!prepare())
        return 1;

    std::vector<RenderTile> renderTiles = config_.tiledRendering ?
        createRenderTiles(width_, height_, config_.tileWidth, config_.tileHeight) :
        createRenderTiles(width_, height_, width_, height_);

    std::cout << "Render mode: " << (config_.persistentWavefront ? "persistent device wavefront" : (config_.tiledRendering ? "tiled" : "full frame"))
              << ", " << renderTiles.size() << " work tiles";
    if (config_.tiledRendering)
        std::cout << " at " << config_.tileWidth << 'x' << config_.tileHeight;
    std::cout << '\n';

    uint32_t completedSamples = 0;
    if (config_.resumeFromCheckpoint) {
        completedSamples = loadRenderSessionCheckpoint(resources_.renderSession, config_.checkpointFilename, renderFingerprint_);
        if (completedSamples > config_.samplesPerPixel) {
            std::cerr << "Checkpoint has more samples than the current render target\n";
            return 1;
        }
        std::cout << "Resuming from " << completedSamples << " completed samples\n";
    } else {
        resetRenderSession(resources_.renderSession, resources_.traceStream);
    }

    CUDA_CHECK(cudaStreamSynchronize(resources_.traceStream));
    createWavefrontGraph();
    renderSamples(renderTiles, completedSamples);
    writeFinalImage();
    return 0;
}
