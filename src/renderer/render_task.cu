#include "renderer/render_task.cuh"

#include "renderer/render_driver.cuh"
#include "renderer/renderer.cuh"

#include <cstdlib>
#include <iostream>

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
} // namespace

RenderTaskRenderer::RenderTaskRenderer(RenderDriver& driver)
    : driver_(driver) {}

RenderTaskResult RenderTaskRenderer::render(const RenderTask& task) {
    RenderTaskResult result;
    result.taskId = task.id;
    result.tile = task.tile;
    result.sampleCount = task.id.sampleCount;

    if (!driver_.prepare())
        return result;

    if (task.jobFingerprint != driver_.renderFingerprint_) {
        std::cerr << "Render task fingerprint does not match the prepared renderer\n";
        return result;
    }

    if (task.tile.width == 0 || task.tile.height == 0 ||
        task.tile.x >= driver_.width_ || task.tile.y >= driver_.height_ ||
        task.tile.width > driver_.width_ - task.tile.x ||
        task.tile.height > driver_.height_ - task.tile.y ||
        task.tile.width * task.tile.height > driver_.renderQueueCapacity_) {
        std::cerr << "Render task tile is outside the prepared renderer bounds\n";
        return result;
    }

    if (task.id.sampleCount == 0 || task.id.sampleStart > driver_.config_.samplesPerPixel ||
        task.id.sampleCount > driver_.config_.samplesPerPixel - task.id.sampleStart) {
        std::cerr << "Render task sample range is outside the prepared renderer bounds\n";
        return result;
    }

    if (!driver_.config_.persistentWavefront && driver_.resources_.traceGraphExec == nullptr)
        driver_.createWavefrontGraph();

    RenderTile renderTile{
        task.tile.x,
        task.tile.y,
        task.tile.width,
        task.tile.height,
        1};

    Vec3* tileFramebuffer = driver_.resources_.renderSession.deviceFramebuffer +
        static_cast<size_t>(task.tile.y) * driver_.width_ + task.tile.x;
    size_t framebufferPitch = static_cast<size_t>(driver_.width_) * sizeof(Vec3);
    size_t tileRowBytes = static_cast<size_t>(task.tile.width) * sizeof(Vec3);

    CUDA_CHECK(cudaMemset2DAsync(
        tileFramebuffer,
        framebufferPitch,
        0,
        tileRowBytes,
        task.tile.height,
        driver_.resources_.traceStream));
    setRenderSessionSampleIndex(
        driver_.resources_.renderSession,
        task.id.sampleStart,
        driver_.resources_.traceStream);

    for (uint32_t sample = 0; sample < task.id.sampleCount; ++sample) {
        CUDA_CHECK(cudaMemcpyAsync(
            driver_.resources_.deviceRenderTile,
            &renderTile,
            sizeof(renderTile),
            cudaMemcpyHostToDevice,
            driver_.resources_.traceStream));

        if (driver_.config_.persistentWavefront) {
            driver_.launchPersistentTile(renderTile);
            advanceSampleIndex<<<1, 1, 0, driver_.resources_.traceStream>>>(
                driver_.resources_.renderSession.deviceSampleIndex,
                driver_.resources_.deviceRenderTile);
            CUDA_CHECK(cudaGetLastError());
        } else {
            driver_.launchCapturedTile();
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(driver_.resources_.traceStream));

    result.radiance.resize(static_cast<size_t>(task.tile.width) * task.tile.height);
    CUDA_CHECK(cudaMemcpy2D(
        result.radiance.data(),
        tileRowBytes,
        tileFramebuffer,
        framebufferPitch,
        tileRowBytes,
        task.tile.height,
        cudaMemcpyDeviceToHost));
    result.success = true;
    return result;
}
