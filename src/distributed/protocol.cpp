#include "distributed/protocol.hpp"

#include <cstring>
#include <limits>
#include <type_traits>

namespace
{
constexpr uint32_t maxStringBytes = 1u * 1024u * 1024u;
constexpr uint32_t maxAssetFiles = 100000;

void setError(std::string* error, const char* message) {
    if (error != nullptr)
        *error = message;
}

class Writer {
public:
    explicit Writer(std::vector<uint8_t>& bytes) : bytes_(bytes) {}

    void u8(uint8_t value) {
        bytes_.push_back(value);
    }

    void u32(uint32_t value) {
        for (uint32_t byte = 0; byte < sizeof(value); ++byte)
            bytes_.push_back(static_cast<uint8_t>((value >> (byte * 8)) & 0xffu));
    }

    void u64(uint64_t value) {
        for (uint32_t byte = 0; byte < sizeof(value); ++byte)
            bytes_.push_back(static_cast<uint8_t>((value >> (byte * 8)) & 0xffu));
    }

    void float32(float value) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "float representation must be four bytes");
        std::memcpy(&bits, &value, sizeof(bits));
        u32(bits);
    }

    void string(const std::string& value) {
        if (value.size() > maxStringBytes || value.size() > std::numeric_limits<uint32_t>::max()) {
            valid_ = false;
            return;
        }
        u32(static_cast<uint32_t>(value.size()));
        bytes_.insert(bytes_.end(), value.begin(), value.end());
    }

    void raw(const void* data, size_t byteCount) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        bytes_.insert(bytes_.end(), bytes, bytes + byteCount);
    }

    void bytes(const std::vector<uint8_t>& value) {
        if (value.size() > std::numeric_limits<uint32_t>::max()) {
            valid_ = false;
            return;
        }
        u32(static_cast<uint32_t>(value.size()));
        raw(value.data(), value.size());
    }

    void fail() { valid_ = false; }
    bool valid() const { return valid_ && bytes_.size() <= distributedMaxMessageBytes; }

private:
    std::vector<uint8_t>& bytes_;
    bool valid_ = true;
};

class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& bytes)
        : bytes_(bytes.data()), remaining_(bytes.size()) {}

    bool u8(uint8_t& value) {
        if (remaining_ < 1)
            return fail();
        value = bytes_[0];
        bytes_ += 1;
        remaining_ -= 1;
        return true;
    }

    bool u32(uint32_t& value) {
        if (remaining_ < sizeof(value))
            return fail();
        value = 0;
        for (uint32_t byte = 0; byte < sizeof(value); ++byte)
            value |= static_cast<uint32_t>(bytes_[byte]) << (byte * 8);
        bytes_ += sizeof(value);
        remaining_ -= sizeof(value);
        return true;
    }

    bool u64(uint64_t& value) {
        if (remaining_ < sizeof(value))
            return fail();
        value = 0;
        for (uint32_t byte = 0; byte < sizeof(value); ++byte)
            value |= static_cast<uint64_t>(bytes_[byte]) << (byte * 8);
        bytes_ += sizeof(value);
        remaining_ -= sizeof(value);
        return true;
    }

    bool float32(float& value) {
        uint32_t bits = 0;
        if (!u32(bits))
            return false;
        std::memcpy(&value, &bits, sizeof(value));
        return true;
    }

    bool string(std::string& value) {
        uint32_t length = 0;
        if (!u32(length) || length > maxStringBytes || length > remaining_)
            return fail();
        value.assign(reinterpret_cast<const char*>(bytes_), length);
        bytes_ += length;
        remaining_ -= length;
        return true;
    }

    bool bytes(std::vector<uint8_t>& value, uint32_t maximumBytes) {
        uint32_t length = 0;
        if (!u32(length) || length > maximumBytes || length > remaining_)
            return fail();
        value.assign(bytes_, bytes_ + length);
        bytes_ += length;
        remaining_ -= length;
        return true;
    }

    size_t remaining() const { return remaining_; }

private:
    bool fail() {
        valid_ = false;
        return false;
    }

    const uint8_t* bytes_ = nullptr;
    size_t remaining_ = 0;
    bool valid_ = true;
};

void appendTile(Writer& writer, const DistributedTile& tile) {
    writer.u32(tile.x);
    writer.u32(tile.y);
    writer.u32(tile.width);
    writer.u32(tile.height);
}

bool readTile(Reader& reader, DistributedTile& tile) {
    return reader.u32(tile.x) && reader.u32(tile.y) &&
        reader.u32(tile.width) && reader.u32(tile.height);
}

void appendTaskId(Writer& writer, const RenderTaskId& id) {
    writer.u32(id.tileOrdinal);
    writer.u32(id.sampleStart);
    writer.u32(id.sampleCount);
}

bool readTaskId(Reader& reader, RenderTaskId& id) {
    return reader.u32(id.tileOrdinal) && reader.u32(id.sampleStart) && reader.u32(id.sampleCount);
}

void appendTask(Writer& writer, const RenderTask& task) {
    writer.u64(task.jobFingerprint);
    appendTaskId(writer, task.id);
    appendTile(writer, task.tile);
}

bool readTask(Reader& reader, RenderTask& task) {
    return reader.u64(task.jobFingerprint) && readTaskId(reader, task.id) && readTile(reader, task.tile);
}

void appendAssignment(Writer& writer, const TaskAssignmentMessage& assignment) {
    appendTask(writer, assignment.task);
    writer.u64(assignment.leaseExpiresAtMs);
}

bool readAssignment(Reader& reader, TaskAssignmentMessage& assignment) {
    return readTask(reader, assignment.task) && reader.u64(assignment.leaseExpiresAtMs);
}

void appendResult(Writer& writer, const DistributedTaskResult& result) {
    writer.u64(result.jobFingerprint);
    appendTaskId(writer, result.taskId);
    appendTile(writer, result.tile);
    writer.u32(result.sampleCount);

    if (result.radiance.size() > std::numeric_limits<uint32_t>::max()) {
        writer.fail();
        return;
    }
    writer.u32(static_cast<uint32_t>(result.radiance.size()));
    for (const DistributedRadiance& pixel : result.radiance) {
        writer.float32(pixel.x);
        writer.float32(pixel.y);
        writer.float32(pixel.z);
    }
}

bool readResult(Reader& reader, DistributedTaskResult& result) {
    if (!reader.u64(result.jobFingerprint) || !readTaskId(reader, result.taskId) ||
        !readTile(reader, result.tile) || !reader.u32(result.sampleCount))
        return false;

    uint32_t pixelCount = 0;
    if (!reader.u32(pixelCount) || pixelCount > distributedMaxMessageBytes / sizeof(DistributedRadiance))
        return false;

    result.radiance.resize(pixelCount);
    for (DistributedRadiance& pixel : result.radiance) {
        if (!reader.float32(pixel.x) || !reader.float32(pixel.y) || !reader.float32(pixel.z))
            return false;
    }
    return true;
}

bool validCommitStatus(uint32_t value) {
    return value <= static_cast<uint32_t>(DistributedCommitStatus::Rejected);
}
} // namespace

DistributedMessageType messageType(const DistributedMessage& message) {
    return std::visit([](const auto& value) {
        using Message = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Message, WorkerHelloMessage>)
            return DistributedMessageType::WorkerHello;
        else if constexpr (std::is_same_v<Message, WorkerHelloAcceptedMessage>)
            return DistributedMessageType::WorkerHelloAccepted;
        else if constexpr (std::is_same_v<Message, TaskRequestMessage>)
            return DistributedMessageType::RequestTask;
        else if constexpr (std::is_same_v<Message, TaskAssignmentMessage>)
            return DistributedMessageType::TaskAssignment;
        else if constexpr (std::is_same_v<Message, TaskResultMessage>)
            return DistributedMessageType::TaskResult;
        else if constexpr (std::is_same_v<Message, ResultAckMessage>)
            return DistributedMessageType::ResultAck;
        else if constexpr (std::is_same_v<Message, NoTaskMessage>)
            return DistributedMessageType::NoTask;
        else if constexpr (std::is_same_v<Message, HeartbeatMessage>)
            return DistributedMessageType::Heartbeat;
        else if constexpr (std::is_same_v<Message, HeartbeatAckMessage>)
            return DistributedMessageType::HeartbeatAck;
        else if constexpr (std::is_same_v<Message, AssetRequestMessage>)
            return DistributedMessageType::AssetRequest;
        else if constexpr (std::is_same_v<Message, AssetChunkMessage>)
            return DistributedMessageType::AssetChunk;
        else
            return DistributedMessageType::Error;
    }, message);
}

bool encodeDistributedMessage(const DistributedMessage& message, std::vector<uint8_t>& bytes, std::string* error) {
    bytes.clear();
    Writer writer(bytes);
    writer.u32(distributedProtocolVersion);
    writer.u32(static_cast<uint32_t>(messageType(message)));

    std::visit([&writer](const auto& value) {
        using Message = std::decay_t<decltype(value)>;
        if constexpr (std::is_same_v<Message, WorkerHelloMessage>) {
            writer.u64(value.jobFingerprint);
            writer.string(value.workerId);
        } else if constexpr (std::is_same_v<Message, WorkerHelloAcceptedMessage>) {
            writer.u8(value.accepted ? 1 : 0);
            writer.string(value.message);
            writer.u64(value.jobFingerprint);
            writer.string(value.primaryAssetPath);
            if (value.assets.size() > maxAssetFiles) {
                writer.fail();
                return;
            }
            writer.u32(static_cast<uint32_t>(value.assets.size()));
            for (const DistributedAssetDescriptor& asset : value.assets) {
                writer.string(asset.relativePath);
                writer.u64(asset.byteSize);
                writer.u64(asset.checksum);
            }
        } else if constexpr (std::is_same_v<Message, TaskRequestMessage>) {
            writer.string(value.workerId);
        } else if constexpr (std::is_same_v<Message, TaskAssignmentMessage>) {
            appendAssignment(writer, value);
        } else if constexpr (std::is_same_v<Message, TaskResultMessage>) {
            appendResult(writer, value.result);
        } else if constexpr (std::is_same_v<Message, ResultAckMessage>) {
            appendTaskId(writer, value.taskId);
            writer.u32(static_cast<uint32_t>(value.status));
            writer.u8(value.hasNextTask ? 1 : 0);
            if (value.hasNextTask)
                appendAssignment(writer, value.nextTask);
        } else if constexpr (std::is_same_v<Message, NoTaskMessage>) {
            writer.u32(value.retryAfterMs);
            writer.u8(value.jobComplete ? 1 : 0);
        } else if constexpr (std::is_same_v<Message, HeartbeatMessage>) {
            writer.string(value.workerId);
        } else if constexpr (std::is_same_v<Message, HeartbeatAckMessage>) {
            writer.u32(value.renewedLeaseCount);
        } else if constexpr (std::is_same_v<Message, AssetRequestMessage>) {
            writer.string(value.workerId);
            writer.string(value.relativePath);
            writer.u64(value.offset);
        } else if constexpr (std::is_same_v<Message, AssetChunkMessage>) {
            writer.string(value.relativePath);
            writer.u64(value.offset);
            writer.u64(value.totalSize);
            writer.u64(value.checksum);
            writer.u8(value.finalChunk ? 1 : 0);
            if (value.bytes.size() > distributedAssetChunkBytes) {
                writer.fail();
                return;
            }
            writer.bytes(value.bytes);
        } else if constexpr (std::is_same_v<Message, ErrorMessage>) {
            writer.string(value.message);
        }
    }, message);

    if (!writer.valid()) {
        setError(error, "distributed message exceeds the maximum frame size or contains an oversized field");
        bytes.clear();
        return false;
    }
    return true;
}

bool decodeDistributedMessage(const std::vector<uint8_t>& bytes, DistributedMessage& message, std::string* error) {
    if (bytes.size() > distributedMaxMessageBytes) {
        setError(error, "distributed message exceeds the maximum frame size");
        return false;
    }

    Reader reader(bytes);
    uint32_t version = 0;
    uint32_t typeValue = 0;
    if (!reader.u32(version) || !reader.u32(typeValue) || version != distributedProtocolVersion) {
        setError(error, "unsupported or truncated distributed message header");
        return false;
    }

    const auto type = static_cast<DistributedMessageType>(typeValue);
    switch (type) {
    case DistributedMessageType::WorkerHello: {
        WorkerHelloMessage value;
        if (!reader.u64(value.jobFingerprint) || !reader.string(value.workerId))
            break;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::WorkerHelloAccepted: {
        WorkerHelloAcceptedMessage value;
        uint8_t accepted = 0;
        uint32_t assetCount = 0;
        if (!reader.u8(accepted) || accepted > 1 || !reader.string(value.message) ||
            !reader.u64(value.jobFingerprint) || !reader.string(value.primaryAssetPath) ||
            !reader.u32(assetCount) || assetCount > maxAssetFiles)
            break;
        value.accepted = accepted != 0;
        value.assets.resize(assetCount);
        for (DistributedAssetDescriptor& asset : value.assets) {
            if (!reader.string(asset.relativePath) || !reader.u64(asset.byteSize) || !reader.u64(asset.checksum)) {
                value.assets.clear();
                break;
            }
        }
        if (value.assets.size() != assetCount)
            break;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::RequestTask: {
        TaskRequestMessage value;
        if (!reader.string(value.workerId))
            break;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::TaskAssignment: {
        TaskAssignmentMessage value;
        if (!readAssignment(reader, value))
            break;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::TaskResult: {
        TaskResultMessage value;
        if (!readResult(reader, value.result))
            break;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::ResultAck: {
        ResultAckMessage value;
        uint32_t status = 0;
        uint8_t hasNextTask = 0;
        if (!readTaskId(reader, value.taskId) || !reader.u32(status) || !validCommitStatus(status) ||
            !reader.u8(hasNextTask) || hasNextTask > 1)
            break;
        value.status = static_cast<DistributedCommitStatus>(status);
        value.hasNextTask = hasNextTask != 0;
        if (value.hasNextTask && !readAssignment(reader, value.nextTask))
            break;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::NoTask: {
        NoTaskMessage value;
        uint8_t jobComplete = 0;
        if (!reader.u32(value.retryAfterMs) || !reader.u8(jobComplete) || jobComplete > 1)
            break;
        value.jobComplete = jobComplete != 0;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::Heartbeat: {
        HeartbeatMessage value;
        if (!reader.string(value.workerId))
            break;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::HeartbeatAck: {
        HeartbeatAckMessage value;
        if (!reader.u32(value.renewedLeaseCount))
            break;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::AssetRequest: {
        AssetRequestMessage value;
        if (!reader.string(value.workerId) || !reader.string(value.relativePath) || !reader.u64(value.offset))
            break;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::AssetChunk: {
        AssetChunkMessage value;
        uint8_t finalChunk = 0;
        if (!reader.string(value.relativePath) || !reader.u64(value.offset) ||
            !reader.u64(value.totalSize) || !reader.u64(value.checksum) ||
            !reader.u8(finalChunk) || finalChunk > 1 ||
            !reader.bytes(value.bytes, distributedAssetChunkBytes))
            break;
        value.finalChunk = finalChunk != 0;
        message = std::move(value);
        goto decoded;
    }
    case DistributedMessageType::Error: {
        ErrorMessage value;
        if (!reader.string(value.message))
            break;
        message = std::move(value);
        goto decoded;
    }
    default:
        setError(error, "unknown distributed message type");
        return false;
    }

    setError(error, "truncated or malformed distributed message payload");
    return false;

decoded:
    if (reader.remaining() != 0) {
        setError(error, "distributed message contains trailing bytes");
        return false;
    }
    return true;
}
