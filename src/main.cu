#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include "renderer/renderer.cuh"
#include "renderer/render_session.cuh"

#define CUDA_CHECK(call)                                      \
    do                                                        \
    {                                                         \
        cudaError_t error = (call);                           \
        if (error != cudaSuccess)                             \
        {                                                     \
            std::cerr                                          \
                << "CUDA error: "                            \
                << cudaGetErrorString(error)                 \
                << " (" << __FILE__ << ":"                  \
                << __LINE__ << ")" << '\n';                  \
            std::exit(1);                                     \
        }                                                     \
    } while (0)

namespace
{
uint8_t toByte(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    value = std::sqrt(value);
    return static_cast<uint8_t>(value * 255.999f);
}

void writePpm(const char* filename, const std::vector<Vec3>& pixels, uint32_t width, uint32_t height) {
    std::ofstream output(filename);

    if (!output) {
        std::cerr << "Could not open " << filename << " for writing\n";
        std::exit(1);
    }

    output << "P3\n" << width << ' ' << height << "\n255\n";

    for (const Vec3& pixel : pixels) {
        output << static_cast<uint32_t>(toByte(pixel.x)) << ' '
               << static_cast<uint32_t>(toByte(pixel.y)) << ' '
               << static_cast<uint32_t>(toByte(pixel.z)) << '\n';
    }
}
} // namespace

int main() {
    constexpr uint32_t width = 1920;
    constexpr uint32_t height = 1080;
    constexpr uint32_t pixelCount = width * height;

    std::cout << "Starting wavefront path tracer\n";

    Camera camera = createDemoCamera(width, height);
    constexpr bool useObjScene = true;
    DeviceScene deviceScene = useObjScene ? createObjScene("../assets/stanford-bunny.obj") : createDemoScene();

    // --------------------------------------------------------
    // Path states
    // --------------------------------------------------------

    PathState* devicePathStates = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePathStates, sizeof(PathState) * pixelCount));

    RenderSession renderSession = createRenderSession(width, height);

    // --------------------------------------------------------
    // Ray queue
    // --------------------------------------------------------

    RayWorkItem* deviceRays = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceRays, sizeof(RayWorkItem) * pixelCount));

    uint32_t* deviceRayCount = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceRayCount, sizeof(uint32_t)));

    RayQueue rayQueue;

    rayQueue.items = deviceRays;
    rayQueue.count = deviceRayCount;
    rayQueue.capacity = pixelCount;

    RayWorkItem* deviceNextRays = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceNextRays, sizeof(RayWorkItem) * pixelCount));

    uint32_t* deviceNextRayCount = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceNextRayCount, sizeof(uint32_t)));

    RayQueue nextRayQueue;

    nextRayQueue.items = deviceNextRays;
    nextRayQueue.count = deviceNextRayCount;
    nextRayQueue.capacity = pixelCount;

    IntersectionResult* deviceIntersectionResults = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceIntersectionResults, sizeof(IntersectionResult) * pixelCount));

    // --------------------------------------------------------
    // Generate primary rays
    // --------------------------------------------------------

    constexpr uint32_t blockSize = 256;
    uint32_t blockCount = (pixelCount + blockSize - 1) / blockSize;

    // --------------------------------------------------------
    // Trace samples and path bounces
    // --------------------------------------------------------

    constexpr uint32_t maxDepth = 8;
    constexpr uint32_t russianRouletteStartDepth = 3;
    constexpr uint32_t samplesPerPixel = 64;
    constexpr bool resumeFromCheckpoint = false;
    constexpr char checkpointFilename[] = "render.checkpoint";
    constexpr uint32_t checkpointProgressPercent = 25;
    constexpr uint32_t checkpointBufferCount = 3;

    cudaStream_t traceStream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&traceStream, cudaStreamNonBlocking));

    AsyncCheckpointWriter* checkpointWriter = createAsyncCheckpointWriter(width, height, checkpointFilename, checkpointBufferCount);

    uint32_t completedSamples = 0;

    if (resumeFromCheckpoint) {
        completedSamples = loadRenderSessionCheckpoint(renderSession, checkpointFilename);

        if (completedSamples > samplesPerPixel) {
            std::cerr << "Checkpoint has more samples than the current render target\n";
            return 1;
        }

        std::cout << "Resuming from " << completedSamples << " completed samples\n";
    } else {
        resetRenderSession(renderSession, traceStream);
    }

    CUDA_CHECK(cudaStreamSynchronize(traceStream));

    cudaGraph_t traceGraph = nullptr;
    cudaGraphExec_t traceGraphExec = nullptr;

    CUDA_CHECK(cudaStreamBeginCapture(traceStream, cudaStreamCaptureModeGlobal));

    CUDA_CHECK(cudaMemsetAsync(deviceRayCount, 0, sizeof(uint32_t), traceStream));
    generatePrimaryRays<<<blockCount, blockSize, 0, traceStream>>>(rayQueue,  devicePathStates,  camera,  width,  height,  renderSession.deviceSampleIndex);

    RayQueue currentRayQueue = rayQueue;
    RayQueue nextRays = nextRayQueue;

    for (uint32_t bounce = 0; bounce < maxDepth; ++bounce) {
        CUDA_CHECK(cudaMemsetAsync(nextRays.count, 0, sizeof(uint32_t), traceStream));

        intersectScene<<<blockCount, blockSize, 0, traceStream>>>(currentRayQueue,  deviceIntersectionResults,  deviceScene.scene);
        shadePaths<<<blockCount, blockSize, 0, traceStream>>>(currentRayQueue,  deviceIntersectionResults,  nextRays,  devicePathStates,  deviceScene.scene,  maxDepth,  russianRouletteStartDepth,  renderSession.deviceFramebuffer);

        std::swap(currentRayQueue, nextRays);
    }

    advanceSampleIndex<<<1, 1, 0, traceStream>>>(renderSession.deviceSampleIndex);

    CUDA_CHECK(cudaStreamEndCapture(traceStream, &traceGraph));
    CUDA_CHECK(cudaGraphInstantiate(&traceGraphExec, traceGraph, nullptr, nullptr, 0));

    cudaEvent_t traceStart;
    cudaEvent_t traceEnd;
    CUDA_CHECK(cudaEventCreate(&traceStart));
    CUDA_CHECK(cudaEventCreate(&traceEnd));
    CUDA_CHECK(cudaEventRecord(traceStart, traceStream));

    uint32_t checkpointStepSamples = std::max(1u, (samplesPerPixel * checkpointProgressPercent + 99) / 100);

    for (uint32_t sample = completedSamples; sample < samplesPerPixel; ++sample) {
        CUDA_CHECK(cudaGraphLaunch(traceGraphExec, traceStream));

        std::cout << "Completed sample " << sample + 1 << " of " << samplesPerPixel << '\n';

        uint32_t newCompletedSamples = sample + 1;
        if (newCompletedSamples % checkpointStepSamples == 0 && newCompletedSamples < samplesPerPixel) {
            if (enqueueRenderSessionCheckpoint(*checkpointWriter, renderSession, newCompletedSamples, traceStream))
                std::cout << "Queued checkpoint at " << newCompletedSamples << " of " << samplesPerPixel << " samples\n";
            else
                std::cout << "Skipped checkpoint at " << newCompletedSamples << " of " << samplesPerPixel << " samples because both checkpoint buffers are busy\n";
        }
    }

    CUDA_CHECK(cudaEventRecord(traceEnd, traceStream));
    CUDA_CHECK(cudaEventSynchronize(traceEnd));

    float traceMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&traceMilliseconds, traceStart, traceEnd));
    std::cout << "Trace time: " << traceMilliseconds << " ms\n";

    // --------------------------------------------------------
    // Resolve framebuffer and write image
    // --------------------------------------------------------

    std::vector<Vec3> pixels(pixelCount);
    CUDA_CHECK(cudaMemcpy(pixels.data(), renderSession.deviceFramebuffer, sizeof(Vec3) * pixelCount, cudaMemcpyDeviceToHost));

    uint32_t finalSampleCount = getRenderSessionSampleIndex(renderSession);

    for(Vec3& pixel: pixels) {
        pixel = pixel / static_cast<float>(finalSampleCount);
    }

    constexpr char outputFilename[] = "render.ppm";

    writePpm(outputFilename, pixels, width, height);
    std::cout << "Wrote " << outputFilename << '\n';

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    destroyAsyncCheckpointWriter(checkpointWriter);
    destroyDeviceScene(deviceScene);

    CUDA_CHECK(cudaGraphExecDestroy(traceGraphExec));
    CUDA_CHECK(cudaGraphDestroy(traceGraph));
    CUDA_CHECK(cudaStreamDestroy(traceStream));

    CUDA_CHECK(cudaFree(devicePathStates));
    destroyRenderSession(renderSession);

    CUDA_CHECK(cudaFree(deviceRays));
    CUDA_CHECK(cudaFree(deviceRayCount));

    CUDA_CHECK(cudaFree(deviceNextRays));
    CUDA_CHECK(cudaFree(deviceNextRayCount));

    CUDA_CHECK(cudaFree(deviceIntersectionResults));
    CUDA_CHECK(cudaEventDestroy(traceStart));
    CUDA_CHECK(cudaEventDestroy(traceEnd));

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
