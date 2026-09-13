#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include "renderer/renderer.cuh"

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
    DeviceScene deviceScene = createDemoScene();

    // --------------------------------------------------------
    // Path states
    // --------------------------------------------------------

    PathState* devicePathStates = nullptr;
    CUDA_CHECK(cudaMalloc(&devicePathStates, sizeof(PathState) * pixelCount));

    Vec3* deviceFramebuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceFramebuffer, sizeof(Vec3) * pixelCount));
    CUDA_CHECK(cudaMemset(deviceFramebuffer, 0, sizeof(Vec3) * pixelCount));

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

    RayWorkItem* deviceRayCandidates = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceRayCandidates, sizeof(RayWorkItem) * pixelCount));

    uint8_t* deviceActiveFlags = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceActiveFlags, sizeof(uint8_t) * pixelCount));

    void* deviceCompactionTempStorage = nullptr;
    size_t compactionTempStorageBytes = 0;

    IntersectionResult* deviceIntersectionResults = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceIntersectionResults, sizeof(IntersectionResult) * pixelCount));

    CUDA_CHECK(cub::DeviceSelect::Flagged(nullptr, compactionTempStorageBytes, deviceRayCandidates, deviceActiveFlags, deviceNextRays, deviceNextRayCount, pixelCount));

    CUDA_CHECK(cudaMalloc(&deviceCompactionTempStorage, compactionTempStorageBytes));

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

    for (uint32_t sample = 0; sample < samplesPerPixel; ++sample) {
        CUDA_CHECK(cudaMemset(deviceRayCount, 0, sizeof(uint32_t)));
        generatePrimaryRays<<<blockCount, blockSize>>>(rayQueue,  devicePathStates,  camera,  width,  height,  sample);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        RayQueue currentRayQueue = rayQueue;
        RayQueue nextRays = nextRayQueue;

        uint32_t currentRayCount = pixelCount;
        for (uint32_t bounce = 0; bounce < maxDepth; ++bounce) {
            intersectScene<<<blockCount, blockSize>>>(currentRayQueue,  deviceIntersectionResults,  deviceScene.scene);
            CUDA_CHECK(cudaGetLastError());

            shadePaths<<<blockCount, blockSize>>>(currentRayQueue,  deviceIntersectionResults,  devicePathStates,  deviceRayCandidates,  deviceActiveFlags,  deviceScene.scene,  maxDepth,  russianRouletteStartDepth,  deviceFramebuffer);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cub::DeviceSelect::Flagged(deviceCompactionTempStorage, compactionTempStorageBytes, deviceRayCandidates, deviceActiveFlags, nextRays.items, nextRays.count, currentRayCount));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            uint32_t nextRayCount = 0;
            CUDA_CHECK(cudaMemcpy(&nextRayCount, nextRays.count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            if (nextRayCount == 0)
                break;

            std::swap(currentRayQueue, nextRays);
            currentRayCount = nextRayCount;
        }

        std::cout << "Completed sample " << sample + 1 << " of " << samplesPerPixel << '\n';
    }

    // --------------------------------------------------------
    // Resolve framebuffer and write image
    // --------------------------------------------------------

    std::vector<Vec3> pixels(pixelCount);
    CUDA_CHECK(cudaMemcpy(pixels.data(), deviceFramebuffer, sizeof(Vec3) * pixelCount, cudaMemcpyDeviceToHost));

    for(Vec3& pixel: pixels) {
        pixel = pixel / static_cast<float>(samplesPerPixel);
    }

    constexpr char outputFilename[] = "render.ppm";

    writePpm(outputFilename, pixels, width, height);
    std::cout << "Wrote " << outputFilename << '\n';

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    destroyDeviceScene(deviceScene);

    CUDA_CHECK(cudaFree(devicePathStates));
    CUDA_CHECK(cudaFree(deviceFramebuffer));

    CUDA_CHECK(cudaFree(deviceRays));
    CUDA_CHECK(cudaFree(deviceRayCount));

    CUDA_CHECK(cudaFree(deviceNextRays));
    CUDA_CHECK(cudaFree(deviceNextRayCount));

    CUDA_CHECK(cudaFree(deviceRayCandidates));
    CUDA_CHECK(cudaFree(deviceActiveFlags));
    CUDA_CHECK(cudaFree(deviceIntersectionResults));
    CUDA_CHECK(cudaFree(deviceCompactionTempStorage));

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
