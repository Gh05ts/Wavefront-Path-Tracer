#include "distributed/checkpoint.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

namespace
{
constexpr char checkpointMagic[8] = {'P', 'T', 'D', 'I', 'S', 'T', '1', '\0'};
constexpr uint32_t checkpointVersion = 1;

struct CheckpointHeader {
    char magic[8];
    uint32_t version;
    uint32_t headerSize;
    uint64_t jobFingerprint;
    uint32_t width;
    uint32_t height;
    uint32_t tileWidth;
    uint32_t tileHeight;
    uint32_t samplesPerPixel;
    uint32_t samplesPerTask;
    uint64_t sequence;
    uint64_t pixelCount;
    uint64_t taskCount;
    uint64_t payloadBytes;
    uint64_t payloadChecksum;
};

void setError(std::string* error, const std::string& message) {
    if (error != nullptr)
        *error = message;
}

uint64_t checksumBytes(uint64_t hash, const void* data, size_t byteCount) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < byteCount; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

uint64_t payloadChecksum(const DistributedCheckpoint& checkpoint) {
    uint64_t hash = 1469598103934665603ull;
    hash = checksumBytes(hash, checkpoint.accumulatedRadiance.data(),
        checkpoint.accumulatedRadiance.size() * sizeof(DistributedRadiance));
    hash = checksumBytes(hash, checkpoint.sampleCounts.data(),
        checkpoint.sampleCounts.size() * sizeof(uint32_t));
    hash = checksumBytes(hash, checkpoint.completedTasks.data(), checkpoint.completedTasks.size());
    return hash;
}

bool checkedProduct(uint64_t first, uint64_t second, uint64_t& result) {
    if (second != 0 && first > std::numeric_limits<uint64_t>::max() / second)
        return false;
    result = first * second;
    return true;
}

bool validShape(const DistributedCheckpoint& checkpoint, std::string* error) {
    uint64_t pixelCount = 0;
    if (!checkedProduct(checkpoint.job.width, checkpoint.job.height, pixelCount)) {
        setError(error, "distributed checkpoint dimensions overflow");
        return false;
    }

    if (checkpoint.accumulatedRadiance.size() != pixelCount ||
        checkpoint.sampleCounts.size() != pixelCount) {
        setError(error, "distributed checkpoint pixel buffers have inconsistent sizes");
        return false;
    }

    for (uint8_t completed : checkpoint.completedTasks) {
        if (completed > 1) {
            setError(error, "distributed checkpoint completion bitmap contains an invalid value");
            return false;
        }
    }

    return true;
}
} // namespace

bool writeDistributedCheckpoint(const char* filename, const DistributedCheckpoint& checkpoint, std::string* error) {
    if (filename == nullptr || filename[0] == '\0') {
        setError(error, "distributed checkpoint filename is empty");
        return false;
    }
    if (!validShape(checkpoint, error))
        return false;

    const uint64_t pixelCount = checkpoint.accumulatedRadiance.size();
    const uint64_t radianceBytes = pixelCount * sizeof(DistributedRadiance);
    const uint64_t sampleBytes = pixelCount * sizeof(uint32_t);
    const uint64_t taskBytes = checkpoint.completedTasks.size();
    const uint64_t payloadBytes = radianceBytes + sampleBytes + taskBytes;

    CheckpointHeader header{};
    std::memcpy(header.magic, checkpointMagic, sizeof(checkpointMagic));
    header.version = checkpointVersion;
    header.headerSize = sizeof(CheckpointHeader);
    header.jobFingerprint = checkpoint.job.jobFingerprint;
    header.width = checkpoint.job.width;
    header.height = checkpoint.job.height;
    header.tileWidth = checkpoint.job.tileWidth;
    header.tileHeight = checkpoint.job.tileHeight;
    header.samplesPerPixel = checkpoint.job.samplesPerPixel;
    header.samplesPerTask = checkpoint.job.samplesPerTask;
    header.sequence = checkpoint.sequence;
    header.pixelCount = pixelCount;
    header.taskCount = taskBytes;
    header.payloadBytes = payloadBytes;
    header.payloadChecksum = payloadChecksum(checkpoint);

    const std::string temporaryFilename = std::string(filename) + ".tmp";
    std::ofstream output(temporaryFilename, std::ios::binary | std::ios::trunc);
    if (!output) {
        setError(error, "could not open distributed checkpoint temporary file");
        return false;
    }

    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(checkpoint.accumulatedRadiance.data()), radianceBytes);
    output.write(reinterpret_cast<const char*>(checkpoint.sampleCounts.data()), sampleBytes);
    output.write(reinterpret_cast<const char*>(checkpoint.completedTasks.data()), taskBytes);
    output.flush();
    output.close();

    if (!output) {
        setError(error, "could not write distributed checkpoint temporary file");
        return false;
    }
    if (std::rename(temporaryFilename.c_str(), filename) != 0) {
        setError(error, "could not atomically replace distributed checkpoint");
        return false;
    }
    return true;
}

bool readDistributedCheckpoint(const char* filename, DistributedCheckpoint& checkpoint, std::string* error) {
    if (filename == nullptr || filename[0] == '\0') {
        setError(error, "distributed checkpoint filename is empty");
        return false;
    }

    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        setError(error, "could not open distributed checkpoint");
        return false;
    }

    CheckpointHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input || std::memcmp(header.magic, checkpointMagic, sizeof(checkpointMagic)) != 0 ||
        header.version != checkpointVersion || header.headerSize != sizeof(CheckpointHeader)) {
        setError(error, "unsupported or truncated distributed checkpoint header");
        return false;
    }

    uint64_t expectedPixelCount = 0;
    if (!checkedProduct(header.width, header.height, expectedPixelCount) ||
        expectedPixelCount != header.pixelCount) {
        setError(error, "distributed checkpoint dimensions are invalid");
        return false;
    }

    const uint64_t radianceBytes = expectedPixelCount * sizeof(DistributedRadiance);
    const uint64_t sampleBytes = expectedPixelCount * sizeof(uint32_t);
    const uint64_t expectedPayloadBytes = radianceBytes + sampleBytes + header.taskCount;
    if (header.payloadBytes != expectedPayloadBytes) {
        setError(error, "distributed checkpoint payload size is invalid");
        return false;
    }

    input.seekg(0, std::ios::end);
    const std::streamoff fileSize = input.tellg();
    if (fileSize < 0 || static_cast<uint64_t>(fileSize) != sizeof(CheckpointHeader) + header.payloadBytes) {
        setError(error, "distributed checkpoint file is truncated or has trailing data");
        return false;
    }
    input.seekg(sizeof(CheckpointHeader), std::ios::beg);

    DistributedCheckpoint loaded;
    loaded.job = DistributedJobConfig{
        header.jobFingerprint,
        header.width,
        header.height,
        header.tileWidth,
        header.tileHeight,
        header.samplesPerPixel,
        header.samplesPerTask};
    loaded.sequence = header.sequence;
    loaded.accumulatedRadiance.resize(static_cast<size_t>(expectedPixelCount));
    loaded.sampleCounts.resize(static_cast<size_t>(expectedPixelCount));
    loaded.completedTasks.resize(static_cast<size_t>(header.taskCount));

    input.read(reinterpret_cast<char*>(loaded.accumulatedRadiance.data()), radianceBytes);
    input.read(reinterpret_cast<char*>(loaded.sampleCounts.data()), sampleBytes);
    input.read(reinterpret_cast<char*>(loaded.completedTasks.data()), header.taskCount);
    if (!input) {
        setError(error, "distributed checkpoint payload is truncated");
        return false;
    }
    if (payloadChecksum(loaded) != header.payloadChecksum) {
        setError(error, "distributed checkpoint payload checksum does not match");
        return false;
    }
    if (!validShape(loaded, error))
        return false;

    checkpoint = std::move(loaded);
    return true;
}
