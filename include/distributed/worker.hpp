#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "transport.hpp"

struct SceneConfig;

struct CoordinatorSessionConfig {
    uint64_t leaseDurationMs = 30000;
    uint32_t noTaskRetryAfterMs = 1000;
    const DistributedAssetCatalog* assetCatalog = nullptr;
};

struct CoordinatorSessionResponse {
    bool keepAlive = false;
    DistributedMessage message = ErrorMessage{};
};

// Handles one registered worker connection. The session owns no CUDA state;
// it only validates protocol messages and delegates task state to the host
// coordinator.
class CoordinatorWorkerSession {
public:
    CoordinatorWorkerSession(
        DistributedCoordinator& coordinator,
        CoordinatorSessionConfig config = {});

    CoordinatorSessionResponse handleMessage(const DistributedMessage& message, uint64_t nowMs);
    bool run(TcpConnection& connection, std::string* error = nullptr);
    void disconnect();

    bool registered() const { return registered_; }
    const std::string& workerId() const { return workerId_; }

private:
    CoordinatorSessionResponse errorResponse(const std::string& message, bool keepAlive = false) const;
    bool validateWorkerId(const std::string& workerId) const;
    std::optional<TaskAssignmentMessage> nextAssignment(uint64_t nowMs);

    DistributedCoordinator& coordinator_;
    CoordinatorSessionConfig config_;
    bool registered_ = false;
    std::string workerId_;
};

// Thin client-side protocol wrapper. A future GPU worker loop can use this
// class to request work and submit RenderTaskRenderer results without knowing
// the socket framing details.
class DistributedWorkerClient {
public:
    DistributedWorkerClient() = default;

    DistributedWorkerClient(const DistributedWorkerClient&) = delete;
    DistributedWorkerClient& operator=(const DistributedWorkerClient&) = delete;
    DistributedWorkerClient(DistributedWorkerClient&&) noexcept = default;
    DistributedWorkerClient& operator=(DistributedWorkerClient&&) noexcept = default;

    static std::optional<DistributedWorkerClient> connectToCoordinator(
        const std::string& host,
        uint16_t port,
        const std::string& workerId,
        uint64_t jobFingerprint,
        std::string* error = nullptr);

    bool requestTask(
        TaskAssignmentMessage& assignment,
        bool& hasTask,
        uint32_t* retryAfterMs = nullptr,
        std::string* error = nullptr);
    bool submitResult(const DistributedTaskResult& result, ResultAckMessage& acknowledgment, std::string* error = nullptr);
    bool heartbeat(HeartbeatAckMessage& acknowledgment, std::string* error = nullptr);
    bool synchronizeAssets(
        SceneConfig& scene,
        const std::string& cacheDirectory,
        std::string* error = nullptr);
    uint64_t coordinatorJobFingerprint() const { return coordinatorJobFingerprint_; }
    bool valid() const { return connection_.valid(); }
    void close() { connection_.close(); }

private:
    DistributedWorkerClient(
        TcpConnection connection,
        std::string workerId,
        uint64_t coordinatorJobFingerprint,
        std::string primaryAssetPath,
        std::vector<DistributedAssetDescriptor> assets)
        : connection_(std::move(connection)), workerId_(std::move(workerId)),
          coordinatorJobFingerprint_(coordinatorJobFingerprint),
          primaryAssetPath_(std::move(primaryAssetPath)), assets_(std::move(assets)) {}

    TcpConnection connection_;
    std::string workerId_;
    uint64_t coordinatorJobFingerprint_ = 0;
    std::string primaryAssetPath_;
    std::vector<DistributedAssetDescriptor> assets_;
};
