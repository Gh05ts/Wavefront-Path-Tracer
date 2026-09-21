#include "renderer/render_session.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace
{
bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "checkpoint_tests: " << message << '\n';
    return condition;
}

bool samePixels(const std::vector<Vec3>& first, const std::vector<Vec3>& second) {
    if (first.size() != second.size())
        return false;
    for (size_t i = 0; i < first.size(); ++i) {
        if (first[i].x != second[i].x || first[i].y != second[i].y || first[i].z != second[i].z)
            return false;
    }
    return true;
}
} // namespace

int main() {
    int deviceCount = 0;
    cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus == cudaErrorNoDevice || deviceStatus == cudaErrorInsufficientDriver || deviceCount == 0) {
        std::cout << "checkpoint_tests: skipped (no CUDA device)\n";
        return 0;
    }
    if (deviceStatus != cudaSuccess) {
        std::cerr << "checkpoint_tests: CUDA device query failed: " << cudaGetErrorString(deviceStatus) << '\n';
        return 1;
    }

    constexpr uint32_t width = 3;
    constexpr uint32_t height = 2;
    const std::string filename = "checkpoint_tests.render";
    std::remove(filename.c_str());
    std::remove((filename + ".tmp").c_str());

    bool valid = true;
    RenderSession source = createRenderSession(width, height);
    RenderSession restored = createRenderSession(width, height);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    const std::vector<Vec3> expected = {
        Vec3(0.25f, 0.50f, 0.75f), Vec3(1.25f, 1.50f, 1.75f), Vec3(2.25f, 2.50f, 2.75f),
        Vec3(3.25f, 3.50f, 3.75f), Vec3(4.25f, 4.50f, 4.75f), Vec3(5.25f, 5.50f, 5.75f)
    };
    cudaMemcpy(source.deviceFramebuffer, expected.data(), sizeof(Vec3) * expected.size(), cudaMemcpyHostToDevice);
    uint32_t sourceSample = 7;
    cudaMemcpy(source.deviceSampleIndex, &sourceSample, sizeof(sourceSample), cudaMemcpyHostToDevice);

    constexpr uint64_t fingerprint = 0x9b1d4f27a6c35011ull;
    AsyncCheckpointWriter* writer = createAsyncCheckpointWriter(width, height, filename.c_str(), 2, fingerprint);
    valid &= check(enqueueRenderSessionCheckpoint(*writer, source, sourceSample, stream), "checkpoint enqueue was rejected");
    finishAsyncCheckpointWriter(*writer);
    destroyAsyncCheckpointWriter(writer);

    valid &= check(loadRenderSessionCheckpoint(restored, filename.c_str(), fingerprint) == sourceSample, "checkpoint sample index did not round-trip");
    std::vector<Vec3> actual(expected.size());
    cudaMemcpy(actual.data(), restored.deviceFramebuffer, sizeof(Vec3) * actual.size(), cudaMemcpyDeviceToHost);
    valid &= check(samePixels(expected, actual), "checkpoint framebuffer did not round-trip");

    resetRenderSession(restored, stream);
    cudaStreamSynchronize(stream);
    std::vector<Vec3> cleared(expected.size());
    cudaMemcpy(cleared.data(), restored.deviceFramebuffer, sizeof(Vec3) * cleared.size(), cudaMemcpyDeviceToHost);
    valid &= check(getRenderSessionSampleIndex(restored) == 0, "session reset did not clear sample index");
    valid &= check(samePixels(cleared, std::vector<Vec3>(expected.size())), "session reset did not clear framebuffer");

    destroyRenderSession(source);
    destroyRenderSession(restored);
    cudaStreamDestroy(stream);
    std::remove(filename.c_str());
    std::remove((filename + ".tmp").c_str());
    return valid ? 0 : 1;
}
