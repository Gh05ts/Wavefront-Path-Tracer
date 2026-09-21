#include "distributed/protocol.hpp"
#include "distributed/transport.hpp"

#include <atomic>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace
{
bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "distributed_transport_tests: " << message << '\n';
    return condition;
}

TaskAssignmentMessage assignment() {
    TaskAssignmentMessage value;
    value.task.jobFingerprint = 0xfeed1234ull;
    value.task.id = RenderTaskId{3, 16, 4};
    value.task.tile = DistributedTile{32, 64, 2, 1};
    value.leaseExpiresAtMs = 987654321ull;
    return value;
}

TaskResultMessage result() {
    TaskResultMessage value;
    value.result.jobFingerprint = 0xfeed1234ull;
    value.result.taskId = RenderTaskId{3, 16, 4};
    value.result.tile = DistributedTile{32, 64, 2, 1};
    value.result.sampleCount = 4;
    value.result.radiance = {
        DistributedRadiance{1.0f, 2.0f, 3.0f},
        DistributedRadiance{4.0f, 5.0f, 6.0f}};
    return value;
}

bool roundTrip(const DistributedMessage& original, DistributedMessage& decoded) {
    std::vector<uint8_t> bytes;
    std::string error;
    if (!encodeDistributedMessage(original, bytes, &error)) {
        std::cerr << "distributed_transport_tests: encode failed: " << error << '\n';
        return false;
    }
    if (!decodeDistributedMessage(bytes, decoded, &error)) {
        std::cerr << "distributed_transport_tests: decode failed: " << error << '\n';
        return false;
    }
    return true;
}
} // namespace

int main() {
    bool valid = true;

    {
        WorkerHelloMessage original{0x12345678ull, "worker-a"};
        DistributedMessage decoded;
        valid &= check(roundTrip(original, decoded), "worker hello did not round-trip");
        const auto* hello = std::get_if<WorkerHelloMessage>(&decoded);
        valid &= check(hello != nullptr && hello->jobFingerprint == original.jobFingerprint &&
            hello->workerId == original.workerId, "worker hello payload changed");
    }

    {
        DistributedMessage original = assignment();
        DistributedMessage decoded;
        valid &= check(roundTrip(original, decoded), "task assignment did not round-trip");
        const auto* value = std::get_if<TaskAssignmentMessage>(&decoded);
        valid &= check(value != nullptr && value->task.id == assignment().task.id &&
            value->task.tile.width == 2 && value->leaseExpiresAtMs == 987654321ull,
            "task assignment payload changed");
    }

    {
        ResultAckMessage original;
        original.taskId = RenderTaskId{3, 16, 4};
        original.status = DistributedCommitStatus::Accepted;
        original.hasNextTask = true;
        original.nextTask = assignment();
        DistributedMessage decoded;
        valid &= check(roundTrip(original, decoded), "result acknowledgment did not round-trip");
        const auto* value = std::get_if<ResultAckMessage>(&decoded);
        valid &= check(value != nullptr && value->status == DistributedCommitStatus::Accepted &&
            value->hasNextTask && value->nextTask.task.id == original.nextTask.task.id,
            "result acknowledgment payload changed");
    }

    {
        DistributedMessage original = result();
        DistributedMessage decoded;
        valid &= check(roundTrip(original, decoded), "task result did not round-trip");
        const auto* value = std::get_if<TaskResultMessage>(&decoded);
        valid &= check(value != nullptr && value->result.radiance.size() == 2 &&
            value->result.radiance[1].z == 6.0f, "task result payload changed");
    }

    {
        WorkerHelloAcceptedMessage original{true, "registered"};
        original.jobFingerprint = 0xfeed1234ull;
        original.primaryAssetPath = "Sponza.gltf";
        original.assets = {
            DistributedAssetDescriptor{"Sponza.gltf", 100, 11},
            DistributedAssetDescriptor{"Sponza.bin", 200, 22}};
        DistributedMessage decoded;
        valid &= check(roundTrip(original, decoded), "asset manifest did not round-trip");
        const auto* value = std::get_if<WorkerHelloAcceptedMessage>(&decoded);
        valid &= check(value != nullptr && value->jobFingerprint == original.jobFingerprint &&
            value->primaryAssetPath == original.primaryAssetPath && value->assets.size() == 2 &&
            value->assets[1].relativePath == "Sponza.bin", "asset manifest payload changed");
    }

    {
        AssetChunkMessage original{"Sponza.bin", 4096, 8192, 1234, true, {1, 2, 3, 4}};
        DistributedMessage decoded;
        valid &= check(roundTrip(original, decoded), "asset chunk did not round-trip");
        const auto* value = std::get_if<AssetChunkMessage>(&decoded);
        valid &= check(value != nullptr && value->offset == 4096 && value->finalChunk &&
            value->bytes.size() == 4 && value->bytes[3] == 4, "asset chunk payload changed");
    }

    {
        std::vector<uint8_t> bytes;
        std::string error;
        DistributedMessage original = NoTaskMessage{};
        valid &= check(encodeDistributedMessage(original, bytes, &error), "no-task encoding failed");
        valid &= check(bytes.size() > 1, "encoded no-task message is unexpectedly short");
        if (bytes.size() > 1) {
            bytes.pop_back();
            DistributedMessage ignored;
            valid &= check(!decodeDistributedMessage(bytes, ignored, &error),
                "truncated message was accepted");
        }
    }

    std::string error;
    auto listener = TcpListener::listenOn("127.0.0.1", 0, &error);
    if (!listener.has_value()) {
        if (error.find("Operation not permitted") != std::string::npos ||
            error.find("Permission denied") != std::string::npos) {
            std::cout << "distributed_transport_tests: skipped loopback transport (network sockets unavailable)\n";
            return valid ? 0 : 1;
        }
        valid &= check(false, error.empty() ? "could not create loopback listener" : error.c_str());
        return 1;
    }
    valid &= check(listener->port() != 0, "loopback listener did not report a port");

    std::atomic<bool> serverValid = true;
    std::thread server([&] {
        std::string serverError;
        auto connection = listener->accept(&serverError);
        if (!connection.has_value()) {
            serverValid = false;
            return;
        }

        DistributedMessage message;
        if (!connection->receive(message, &serverError) || !std::holds_alternative<WorkerHelloMessage>(message)) {
            serverValid = false;
            return;
        }
        if (!connection->send(WorkerHelloAcceptedMessage{true, "ready"}, &serverError)) {
            serverValid = false;
            return;
        }
        if (!connection->receive(message, &serverError) || !std::holds_alternative<TaskRequestMessage>(message)) {
            serverValid = false;
            return;
        }
        if (!connection->send(assignment(), &serverError)) {
            serverValid = false;
            return;
        }
        if (!connection->receive(message, &serverError) || !std::holds_alternative<TaskResultMessage>(message)) {
            serverValid = false;
            return;
        }

        ResultAckMessage ack;
        ack.taskId = result().result.taskId;
        ack.status = DistributedCommitStatus::Accepted;
        ack.hasNextTask = false;
        if (!connection->send(ack, &serverError))
            serverValid = false;
    });

    auto connection = TcpConnection::connectTo("127.0.0.1", listener->port(), &error);
    valid &= check(connection.has_value(), error.empty() ? "could not connect to loopback listener" : error.c_str());
    if (connection.has_value()) {
        valid &= check(connection->send(WorkerHelloMessage{0x12345678ull, "worker-a"}, &error),
            "TCP hello send failed");

        DistributedMessage message;
        valid &= check(connection->receive(message, &error) &&
            std::holds_alternative<WorkerHelloAcceptedMessage>(message),
            "TCP hello response failed");
        valid &= check(connection->send(TaskRequestMessage{"worker-a"}, &error),
            "TCP task request send failed");
        valid &= check(connection->receive(message, &error) &&
            std::holds_alternative<TaskAssignmentMessage>(message),
            "TCP task assignment receive failed");
        valid &= check(connection->send(result(), &error), "TCP result send failed");
        valid &= check(connection->receive(message, &error) &&
            std::holds_alternative<ResultAckMessage>(message),
            "TCP result acknowledgment receive failed");
    }

    if (connection.has_value())
        connection->close();
    listener->close();
    server.join();
    valid &= check(serverValid.load(), "loopback server exchange failed");
    return valid ? 0 : 1;
}
