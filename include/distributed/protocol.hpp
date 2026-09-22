#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "assets.hpp"
#include "coordinator.hpp"

constexpr uint32_t distributedProtocolVersion = 3;
constexpr uint32_t distributedMaxMessageBytes = 128u * 1024u * 1024u;
constexpr uint32_t distributedAssetChunkBytes = 4u * 1024u * 1024u;

enum class DistributedMessageType : uint32_t {
    WorkerHello = 1,
    WorkerHelloAccepted = 2,
    RequestTask = 3,
    TaskAssignment = 4,
    TaskResult = 5,
    ResultAck = 6,
    NoTask = 7,
    Heartbeat = 8,
    HeartbeatAck = 9,
    Error = 10,
    AssetRequest = 11,
    AssetChunk = 12
};

struct WorkerHelloMessage {
    uint64_t jobFingerprint = 0;
    std::string workerId;
};

struct WorkerHelloAcceptedMessage {
    bool accepted = false;
    std::string message;
    uint64_t jobFingerprint = 0;
    std::string primaryAssetPath;
    std::vector<DistributedAssetDescriptor> assets;
};

struct AssetRequestMessage {
    std::string workerId;
    std::string relativePath;
    uint64_t offset = 0;
};

struct AssetChunkMessage {
    std::string relativePath;
    uint64_t offset = 0;
    uint64_t totalSize = 0;
    uint64_t checksum = 0;
    bool finalChunk = false;
    std::vector<uint8_t> bytes;
};

struct TaskRequestMessage {
    std::string workerId;
};

struct TaskAssignmentMessage {
    RenderTask task{};
    uint64_t leaseExpiresAtMs = 0;
};

struct TaskResultMessage {
    DistributedTaskResult result;
};

struct ResultAckMessage {
    RenderTaskId taskId{};
    DistributedCommitStatus status = DistributedCommitStatus::Rejected;
    bool hasNextTask = false;
    TaskAssignmentMessage nextTask{};
};

struct NoTaskMessage {
    uint32_t retryAfterMs = 1000;
    bool jobComplete = false;
};

struct HeartbeatMessage {
    std::string workerId;
};

struct HeartbeatAckMessage {
    uint32_t renewedLeaseCount = 0;
};

struct ErrorMessage {
    std::string message;
};

using DistributedMessage = std::variant<
    WorkerHelloMessage,
    WorkerHelloAcceptedMessage,
    TaskRequestMessage,
    TaskAssignmentMessage,
    TaskResultMessage,
    ResultAckMessage,
    NoTaskMessage,
    HeartbeatMessage,
    HeartbeatAckMessage,
    AssetRequestMessage,
    AssetChunkMessage,
    ErrorMessage>;

DistributedMessageType messageType(const DistributedMessage& message);

bool encodeDistributedMessage(const DistributedMessage& message, std::vector<uint8_t>& bytes, std::string* error = nullptr);
bool decodeDistributedMessage(const std::vector<uint8_t>& bytes, DistributedMessage& message, std::string* error = nullptr);
