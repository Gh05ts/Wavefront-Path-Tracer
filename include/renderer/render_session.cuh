#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "../core/vec3.cuh"

struct RenderSession {
    uint32_t width;
    uint32_t height;
    Vec3* deviceFramebuffer;
    uint32_t* deviceSampleIndex;
};

struct AsyncCheckpointWriter;

RenderSession createRenderSession(uint32_t width, uint32_t height);
void resetRenderSession(RenderSession& session, cudaStream_t stream);
uint32_t getRenderSessionSampleIndex(const RenderSession& session);
AsyncCheckpointWriter* createAsyncCheckpointWriter(uint32_t width, uint32_t height, const char* filename, uint32_t bufferCount);
bool enqueueRenderSessionCheckpoint(AsyncCheckpointWriter& writer, const RenderSession& session, uint32_t sampleIndex, cudaStream_t stream);
void finishAsyncCheckpointWriter(AsyncCheckpointWriter& writer);
void destroyAsyncCheckpointWriter(AsyncCheckpointWriter* writer);
uint32_t loadRenderSessionCheckpoint(RenderSession& session, const char* filename);
void destroyRenderSession(RenderSession& session);
