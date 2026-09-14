#include "renderer/render_session.cuh"

#include <cstdlib>
#include <iostream>

namespace
{
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}
} // namespace

RenderSession createRenderSession(uint32_t width, uint32_t height) {
    RenderSession session{};
    session.width = width;
    session.height = height;

    uint32_t pixelCount = width * height;
    checkCuda(cudaMalloc(&session.deviceFramebuffer, sizeof(Vec3) * pixelCount));
    checkCuda(cudaMalloc(&session.deviceSampleIndex, sizeof(uint32_t)));

    return session;
}

void resetRenderSession(RenderSession& session, cudaStream_t stream) {
    uint32_t pixelCount = session.width * session.height;
    checkCuda(cudaMemsetAsync(session.deviceFramebuffer, 0, sizeof(Vec3) * pixelCount, stream));
    checkCuda(cudaMemsetAsync(session.deviceSampleIndex, 0, sizeof(uint32_t), stream));
}

uint32_t getRenderSessionSampleIndex(const RenderSession& session) {
    uint32_t sampleIndex = 0;
    checkCuda(cudaMemcpy(&sampleIndex, session.deviceSampleIndex, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return sampleIndex;
}

void destroyRenderSession(RenderSession& session) {
    checkCuda(cudaFree(session.deviceFramebuffer));
    checkCuda(cudaFree(session.deviceSampleIndex));
    session = RenderSession{};
}
