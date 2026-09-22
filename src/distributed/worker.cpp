#include "distributed/worker.hpp"
#include "config.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>

namespace
{
constexpr uint64_t fnvOffset = 1469598103934665603ull;
constexpr uint64_t fnvPrime = 1099511628211ull;

void setError(std::string* error, const std::string& message) {
    if (error != nullptr)
        *error = message;
}

bool hashCachedFile(const std::filesystem::path& path, uint64_t& byteSize, uint64_t& checksum) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return false;
    byteSize = 0;
    checksum = fnvOffset;
    char buffer[64 * 1024];
    while (input.read(buffer, sizeof(buffer)) || input.gcount() > 0) {
        for (std::streamsize i = 0; i < input.gcount(); ++i) {
            checksum ^= static_cast<uint8_t>(static_cast<unsigned char>(buffer[i]));
            checksum *= fnvPrime;
        }
        byteSize += static_cast<uint64_t>(input.gcount());
    }
    return input.eof();
}

bool safeAssetPath(const std::string& relativePath) {
    const std::filesystem::path path(relativePath);
    if (path.empty() || path.is_absolute())
        return false;
    for (const auto& component : path) {
        if (component == "..")
            return false;
    }
    return true;
}

uint64_t monotonicMilliseconds() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

bool sameWorker(const std::string& expected, const std::string& actual) {
    return !expected.empty() && expected == actual;
}
} // namespace

CoordinatorWorkerSession::CoordinatorWorkerSession(
    DistributedCoordinator& coordinator,
    CoordinatorSessionConfig config)
    : coordinator_(coordinator), config_(config) {}

CoordinatorSessionResponse CoordinatorWorkerSession::handleMessage(
    const DistributedMessage& message,
    uint64_t nowMs) {
    coordinator_.requeueExpired(nowMs);
    if (!registered_) {
        const auto* hello = std::get_if<WorkerHelloMessage>(&message);
        if (hello == nullptr)
            return errorResponse("worker must send hello before any task message");
        if (hello->workerId.empty())
            return CoordinatorSessionResponse{
                false,
                WorkerHelloAcceptedMessage{false, "worker ID cannot be empty"}};
        const bool deferredAssetFingerprint = hello->jobFingerprint == 0 && config_.assetCatalog != nullptr;
        if (hello->jobFingerprint != coordinator_.job().jobFingerprint && !deferredAssetFingerprint)
            return CoordinatorSessionResponse{
                false,
                WorkerHelloAcceptedMessage{false, "worker job fingerprint does not match"}};

        workerId_ = hello->workerId;
        registered_ = true;
        WorkerHelloAcceptedMessage accepted{true, "worker registered"};
        accepted.jobFingerprint = coordinator_.job().jobFingerprint;
        if (config_.assetCatalog != nullptr) {
            accepted.primaryAssetPath = config_.assetCatalog->primaryRelativePath;
            accepted.assets.reserve(config_.assetCatalog->files.size());
            for (const DistributedAssetFile& file : config_.assetCatalog->files)
                accepted.assets.push_back(DistributedAssetDescriptor{file.relativePath, file.byteSize, file.checksum});
        }
        return CoordinatorSessionResponse{
            true,
            std::move(accepted)};
    }

    if (const auto* request = std::get_if<TaskRequestMessage>(&message)) {
        if (!validateWorkerId(request->workerId))
            return errorResponse("task request worker ID does not match registration");

        if (const auto assignment = nextAssignment(nowMs))
            return CoordinatorSessionResponse{true, *assignment};
        return CoordinatorSessionResponse{
            true,
            NoTaskMessage{config_.noTaskRetryAfterMs, coordinator_.complete()}};
    }

    if (const auto* result = std::get_if<TaskResultMessage>(&message)) {
        if (result->result.jobFingerprint != coordinator_.job().jobFingerprint)
            return errorResponse("task result job fingerprint does not match coordinator");

        ResultAckMessage acknowledgment;
        acknowledgment.taskId = result->result.taskId;
        acknowledgment.status = coordinator_.commitResult(result->result);
        if (acknowledgment.status != DistributedCommitStatus::Rejected && !coordinator_.complete()) {
            if (const auto assignment = nextAssignment(nowMs)) {
                acknowledgment.hasNextTask = true;
                acknowledgment.nextTask = *assignment;
            }
        }
        return CoordinatorSessionResponse{true, acknowledgment};
    }

    if (const auto* request = std::get_if<AssetRequestMessage>(&message)) {
        if (!validateWorkerId(request->workerId))
            return errorResponse("asset request worker ID does not match registration");
        if (config_.assetCatalog == nullptr)
            return errorResponse("coordinator did not enable asset transfer");
        const DistributedAssetFile* file = findDistributedAsset(*config_.assetCatalog, request->relativePath);
        if (file == nullptr)
            return errorResponse("requested asset is not part of the scene catalog");

        std::vector<uint8_t> bytes;
        bool finalChunk = false;
        std::string readError;
        if (!readDistributedAssetChunk(
                *file, request->offset, distributedAssetChunkBytes, bytes, finalChunk, &readError))
            return errorResponse(readError);
        return CoordinatorSessionResponse{
            true,
            AssetChunkMessage{
                file->relativePath,
                request->offset,
                file->byteSize,
                file->checksum,
                finalChunk,
                std::move(bytes)}};
    }

    if (const auto* heartbeat = std::get_if<HeartbeatMessage>(&message)) {
        if (!validateWorkerId(heartbeat->workerId))
            return errorResponse("heartbeat worker ID does not match registration");
        return CoordinatorSessionResponse{
            true,
            HeartbeatAckMessage{
                coordinator_.renewWorkerLeases(workerId_, nowMs, config_.leaseDurationMs)}};
    }

    return errorResponse("unexpected message for registered worker");
}

bool CoordinatorWorkerSession::run(TcpConnection& connection, std::string* error) {
    if (!connection.valid()) {
        if (error != nullptr)
            *error = "cannot run worker session on an invalid connection";
        return false;
    }

    while (connection.valid()) {
        DistributedMessage incoming;
        if (!connection.receive(incoming, error)) {
            disconnect();
            return false;
        }

        const CoordinatorSessionResponse response = handleMessage(incoming, monotonicMilliseconds());
        if (!connection.send(response.message, error)) {
            disconnect();
            return false;
        }
        if (!response.keepAlive) {
            disconnect();
            return false;
        }
    }

    disconnect();
    return true;
}

void CoordinatorWorkerSession::disconnect() {
    if (registered_)
        coordinator_.releaseWorkerLeases(workerId_);
    registered_ = false;
    workerId_.clear();
}

CoordinatorSessionResponse CoordinatorWorkerSession::errorResponse(
    const std::string& message,
    bool keepAlive) const {
    return CoordinatorSessionResponse{keepAlive, ErrorMessage{message}};
}

bool CoordinatorWorkerSession::validateWorkerId(const std::string& workerId) const {
    return registered_ && sameWorker(workerId_, workerId);
}

std::optional<TaskAssignmentMessage> CoordinatorWorkerSession::nextAssignment(uint64_t nowMs) {
    const auto lease = coordinator_.leaseNext(workerId_, nowMs, config_.leaseDurationMs);
    if (!lease)
        return std::nullopt;
    return TaskAssignmentMessage{lease->task, lease->expiresAtMs};
}

std::optional<DistributedWorkerClient> DistributedWorkerClient::connectToCoordinator(
    const std::string& host,
    uint16_t port,
    const std::string& workerId,
    uint64_t jobFingerprint,
    std::string* error) {
    if (workerId.empty()) {
        if (error != nullptr)
            *error = "worker ID cannot be empty";
        return std::nullopt;
    }

    auto connection = TcpConnection::connectTo(host, port, error);
    if (!connection)
        return std::nullopt;

    if (!connection->send(WorkerHelloMessage{jobFingerprint, workerId}, error))
        return std::nullopt;

    DistributedMessage response;
    if (!connection->receive(response, error))
        return std::nullopt;
    const auto* accepted = std::get_if<WorkerHelloAcceptedMessage>(&response);
    if (accepted == nullptr || !accepted->accepted) {
        if (error != nullptr)
            *error = accepted == nullptr ? "coordinator returned an unexpected registration response" : accepted->message;
        return std::nullopt;
    }
    return DistributedWorkerClient(
        std::move(*connection), workerId, accepted->jobFingerprint,
        accepted->primaryAssetPath, accepted->assets);
}

bool DistributedWorkerClient::synchronizeAssets(
    SceneConfig& scene,
    const std::string& cacheDirectory,
    std::string* error) {
    if (assets_.empty())
        return true;
    if (primaryAssetPath_.empty()) {
        setError(error, "coordinator asset catalog has no primary asset path");
        return false;
    }
    if (cacheDirectory.empty()) {
        setError(error, "asset cache directory is empty");
        return false;
    }

    try {
        const std::filesystem::path cacheRoot =
            std::filesystem::path(cacheDirectory) / std::to_string(coordinatorJobFingerprint_);
        std::filesystem::create_directories(cacheRoot);

        for (const DistributedAssetDescriptor& asset : assets_) {
            if (!safeAssetPath(asset.relativePath)) {
                setError(error, "coordinator sent an unsafe asset path: " + asset.relativePath);
                return false;
            }
            const std::filesystem::path target = (cacheRoot / asset.relativePath).lexically_normal();
            std::filesystem::create_directories(target.parent_path());

            uint64_t cachedSize = 0;
            uint64_t cachedChecksum = 0;
            if (hashCachedFile(target, cachedSize, cachedChecksum) &&
                cachedSize == asset.byteSize && cachedChecksum == asset.checksum)
                continue;

            std::ofstream output(target, std::ios::binary | std::ios::trunc);
            if (!output) {
                setError(error, "could not create asset cache file: " + target.string());
                return false;
            }

            uint64_t offset = 0;
            while (offset < asset.byteSize || (asset.byteSize == 0 && offset == 0)) {
                if (!connection_.send(AssetRequestMessage{workerId_, asset.relativePath, offset}, error))
                    return false;
                DistributedMessage response;
                if (!connection_.receive(response, error))
                    return false;
                const auto* chunk = std::get_if<AssetChunkMessage>(&response);
                if (chunk == nullptr || chunk->relativePath != asset.relativePath ||
                    chunk->offset != offset || chunk->totalSize != asset.byteSize ||
                    chunk->checksum != asset.checksum || chunk->bytes.size() > distributedAssetChunkBytes ||
                    (!chunk->finalChunk && chunk->bytes.empty())) {
                    setError(error, "coordinator returned an invalid asset chunk");
                    return false;
                }
                if (!chunk->bytes.empty())
                    output.write(reinterpret_cast<const char*>(chunk->bytes.data()), chunk->bytes.size());
                if (!output) {
                    setError(error, "could not write asset cache file: " + target.string());
                    return false;
                }
                offset += chunk->bytes.size();
                if (chunk->finalChunk)
                    break;
            }
            output.close();
            if (!hashCachedFile(target, cachedSize, cachedChecksum) ||
                cachedSize != asset.byteSize || cachedChecksum != asset.checksum) {
                setError(error, "asset cache checksum validation failed: " + asset.relativePath);
                return false;
            }
        }

        if (!safeAssetPath(primaryAssetPath_)) {
            setError(error, "coordinator sent an unsafe primary asset path");
            return false;
        }
        const std::string primary =
            (cacheRoot / primaryAssetPath_).lexically_normal().string();
        scene.objectFilename = primary;
        scene.gltfFilename = primary;
        return true;
    } catch (const std::exception& exception) {
        setError(error, std::string("asset cache setup failed: ") + exception.what());
        return false;
    }
}

bool DistributedWorkerClient::requestTask(
    TaskAssignmentMessage& assignment,
    bool& hasTask,
    uint32_t* retryAfterMs,
    std::string* error,
    bool* jobComplete) {
    if (jobComplete != nullptr)
        *jobComplete = false;
    if (!connection_.send(TaskRequestMessage{workerId_}, error))
        return false;

    DistributedMessage response;
    if (!connection_.receive(response, error))
        return false;
    if (const auto* next = std::get_if<TaskAssignmentMessage>(&response)) {
        assignment = *next;
        hasTask = true;
        return true;
    }
    if (const auto* noTask = std::get_if<NoTaskMessage>(&response)) {
        hasTask = false;
        if (retryAfterMs != nullptr)
            *retryAfterMs = noTask->retryAfterMs;
        if (jobComplete != nullptr)
            *jobComplete = noTask->jobComplete;
        return true;
    }
    if (const auto* failure = std::get_if<ErrorMessage>(&response)) {
        if (error != nullptr)
            *error = failure->message;
    } else if (error != nullptr) {
        *error = "coordinator returned an unexpected task response";
    }
    return false;
}

bool DistributedWorkerClient::submitResult(
    const DistributedTaskResult& result,
    ResultAckMessage& acknowledgment,
    std::string* error) {
    if (!connection_.send(TaskResultMessage{result}, error))
        return false;

    DistributedMessage response;
    if (!connection_.receive(response, error))
        return false;
    const auto* ack = std::get_if<ResultAckMessage>(&response);
    if (ack != nullptr) {
        acknowledgment = *ack;
        return true;
    }
    if (const auto* failure = std::get_if<ErrorMessage>(&response)) {
        if (error != nullptr)
            *error = failure->message;
    } else if (error != nullptr) {
        *error = "coordinator returned an unexpected result response";
    }
    return false;
}

bool DistributedWorkerClient::heartbeat(HeartbeatAckMessage& acknowledgment, std::string* error) {
    if (!connection_.send(HeartbeatMessage{workerId_}, error))
        return false;

    DistributedMessage response;
    if (!connection_.receive(response, error))
        return false;
    const auto* ack = std::get_if<HeartbeatAckMessage>(&response);
    if (ack != nullptr) {
        acknowledgment = *ack;
        return true;
    }
    if (const auto* failure = std::get_if<ErrorMessage>(&response)) {
        if (error != nullptr)
            *error = failure->message;
    } else if (error != nullptr) {
        *error = "coordinator returned an unexpected heartbeat response";
    }
    return false;
}
