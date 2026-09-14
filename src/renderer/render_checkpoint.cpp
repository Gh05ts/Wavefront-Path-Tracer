#include "renderer/render_session.cuh"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace
{
constexpr char checkpointMagic[8] = {'P', 'T', 'C', 'H', 'K', 'P', 'T', '1'};
constexpr uint32_t checkpointVersion = 1;

struct CheckpointHeader {
    char magic[8];
    uint32_t version;
    uint32_t width;
    uint32_t height;
    uint32_t sampleIndex;
};

void fail(const std::string& message) {
    std::cerr << message << '\n';
    std::exit(1);
}

void checkCuda(cudaError_t error) {
    if (error != cudaSuccess)
        fail(std::string("CUDA error: ") + cudaGetErrorString(error));
}

void writeCheckpoint(const char* filename, const Vec3* pixels, uint32_t width, uint32_t height, uint32_t sampleIndex) {
    CheckpointHeader header{};
    std::memcpy(header.magic, checkpointMagic, sizeof(checkpointMagic));
    header.version = checkpointVersion;
    header.width = width;
    header.height = height;
    header.sampleIndex = sampleIndex;

    std::string temporaryFilename = std::string(filename) + ".tmp";
    std::ofstream output(temporaryFilename, std::ios::binary | std::ios::trunc);

    if (!output)
        fail(std::string("Could not open checkpoint for writing: ") + temporaryFilename);

    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(pixels), sizeof(Vec3) * width * height);
    output.close();

    if (!output)
        fail(std::string("Could not write checkpoint: ") + temporaryFilename);

    if (std::rename(temporaryFilename.c_str(), filename) != 0)
        fail(std::string("Could not replace checkpoint: ") + filename);
}
} // namespace

struct AsyncCheckpointWriter {
    struct Slot {
        Vec3* pixels = nullptr;
        cudaEvent_t copyComplete = nullptr;
        std::atomic<bool> busy = false;
        std::thread writer;
    };

    uint32_t width;
    uint32_t height;
    std::string filename;
    std::mutex writeMutex;
    uint32_t latestWrittenSample = 0;
    std::unique_ptr<Slot[]> slots;
    uint32_t slotCount;
};

AsyncCheckpointWriter* createAsyncCheckpointWriter(uint32_t width, uint32_t height, const char* filename, uint32_t bufferCount) {
    static_assert(sizeof(Vec3) == sizeof(float) * 3, "Checkpoint stores Vec3 as three floats");

    if (bufferCount == 0)
        fail("Checkpoint buffer count must be at least one");

    AsyncCheckpointWriter* writer = new AsyncCheckpointWriter{};
    writer->width = width;
    writer->height = height;
    writer->filename = filename;
    writer->slots = std::make_unique<AsyncCheckpointWriter::Slot[]>(bufferCount);
    writer->slotCount = bufferCount;

    uint32_t pixelCount = width * height;
    for (uint32_t i = 0; i < writer->slotCount; ++i) {
        AsyncCheckpointWriter::Slot& slot = writer->slots[i];
        checkCuda(cudaHostAlloc(&slot.pixels, sizeof(Vec3) * pixelCount, cudaHostAllocDefault));
        checkCuda(cudaEventCreateWithFlags(&slot.copyComplete, cudaEventDisableTiming));
    }

    return writer;
}

bool enqueueRenderSessionCheckpoint(AsyncCheckpointWriter& writer, const RenderSession& session, uint32_t sampleIndex, cudaStream_t stream) {
    for (uint32_t i = 0; i < writer.slotCount; ++i) {
        AsyncCheckpointWriter::Slot& slot = writer.slots[i];
        bool expected = false;
        if (!slot.busy.compare_exchange_strong(expected, true))
            continue;

        if (slot.writer.joinable())
            slot.writer.join();

        uint32_t pixelCount = session.width * session.height;
        checkCuda(cudaMemcpyAsync(slot.pixels, session.deviceFramebuffer, sizeof(Vec3) * pixelCount, cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaEventRecord(slot.copyComplete, stream));

        slot.writer = std::thread([&writer, &slot, sampleIndex] {
            checkCuda(cudaEventSynchronize(slot.copyComplete));
            {
                std::lock_guard<std::mutex> lock(writer.writeMutex);
                if (sampleIndex >= writer.latestWrittenSample) {
                    writeCheckpoint(writer.filename.c_str(), slot.pixels, writer.width, writer.height, sampleIndex);
                    writer.latestWrittenSample = sampleIndex;
                }
            }
            slot.busy.store(false);
        });

        return true;
    }

    return false;
}

void finishAsyncCheckpointWriter(AsyncCheckpointWriter& writer) {
    for (uint32_t i = 0; i < writer.slotCount; ++i) {
        AsyncCheckpointWriter::Slot& slot = writer.slots[i];
        if (slot.writer.joinable())
            slot.writer.join();
    }
}

void destroyAsyncCheckpointWriter(AsyncCheckpointWriter* writer) {
    if (writer == nullptr)
        return;

    finishAsyncCheckpointWriter(*writer);

    for (uint32_t i = 0; i < writer->slotCount; ++i) {
        AsyncCheckpointWriter::Slot& slot = writer->slots[i];
        checkCuda(cudaEventDestroy(slot.copyComplete));
        checkCuda(cudaFreeHost(slot.pixels));
    }

    delete writer;
}

uint32_t loadRenderSessionCheckpoint(RenderSession& session, const char* filename) {
    std::ifstream input(filename, std::ios::binary);

    if (!input)
        fail(std::string("Could not open checkpoint: ") + filename);

    CheckpointHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));

    if (!input)
        fail(std::string("Checkpoint is truncated or unreadable: ") + filename);

    if (std::memcmp(header.magic, checkpointMagic, sizeof(checkpointMagic)) != 0 || header.version != checkpointVersion)
        fail(std::string("Unsupported checkpoint format: ") + filename);

    if (header.width != session.width || header.height != session.height)
        fail(std::string("Checkpoint resolution does not match the current render: ") + filename);

    uint32_t pixelCount = session.width * session.height;
    std::vector<Vec3> pixels(pixelCount);
    input.read(reinterpret_cast<char*>(pixels.data()), sizeof(Vec3) * pixels.size());

    if (!input)
        fail(std::string("Checkpoint is truncated or unreadable: ") + filename);

    checkCuda(cudaMemcpy(session.deviceFramebuffer, pixels.data(), sizeof(Vec3) * pixelCount, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(session.deviceSampleIndex, &header.sampleIndex, sizeof(uint32_t), cudaMemcpyHostToDevice));

    return header.sampleIndex;
}
