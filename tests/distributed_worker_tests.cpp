#include "distributed/worker.hpp"
#include "distributed/assets.hpp"
#include "config.hpp"

#include <iostream>
#include <vector>

namespace
{
bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "distributed_worker_tests: " << message << '\n';
    return condition;
}

DistributedTaskResult resultFor(const TaskAssignmentMessage& assignment) {
    DistributedTaskResult result;
    result.jobFingerprint = assignment.task.jobFingerprint;
    result.taskId = assignment.task.id;
    result.tile = assignment.task.tile;
    result.sampleCount = assignment.task.id.sampleCount;
    result.radiance.resize(static_cast<size_t>(result.tile.width) * result.tile.height);
    for (DistributedRadiance& pixel : result.radiance)
        pixel = DistributedRadiance{1.0f, 2.0f, 3.0f};
    return result;
}
} // namespace

int main() {
    bool valid = true;

    {
        SceneConfig scene;
        scene.preset = ScenePreset::Sponza;
        scene.gltfFilename = "../../assets/sponza/Sponza.gltf";
        DistributedAssetCatalog catalog;
        std::string assetError;
        valid &= check(buildDistributedAssetCatalog(scene, catalog, &assetError),
            assetError.empty() ? "Sponza asset catalog could not be built" : assetError.c_str());
        valid &= check(catalog.primaryRelativePath == "Sponza.gltf" && catalog.files.size() > 2,
            "Sponza asset dependencies were not cataloged");
        valid &= check(findDistributedAsset(catalog, "Sponza.bin") != nullptr,
            "Sponza binary dependency was not cataloged");
    }

    constexpr uint64_t fingerprint = 0xabcdef0123456789ull;
    DistributedJobConfig job{fingerprint, 2, 1, 1, 1, 2, 1};
    const std::vector<RenderTask> tasks = createRenderTasks(
        job.width,
        job.height,
        job.tileWidth,
        job.tileHeight,
        job.samplesPerPixel,
        job.samplesPerTask,
        job.jobFingerprint);

    DistributedCoordinator coordinator(job, tasks);
    CoordinatorWorkerSession session(coordinator, CoordinatorSessionConfig{100, 25});

    CoordinatorSessionResponse response = session.handleMessage(
        WorkerHelloMessage{fingerprint, "worker-a"}, 1000);
    const auto* hello = std::get_if<WorkerHelloAcceptedMessage>(&response.message);
    valid &= check(response.keepAlive && hello != nullptr && hello->accepted,
        "valid worker was not registered");

    response = session.handleMessage(TaskRequestMessage{"worker-a"}, 1000);
    const auto* firstAssignment = std::get_if<TaskAssignmentMessage>(&response.message);
    valid &= check(response.keepAlive && firstAssignment != nullptr,
        "registered worker did not receive a task");
    const TaskAssignmentMessage firstAssignmentCopy = firstAssignment != nullptr ? *firstAssignment : TaskAssignmentMessage{};

    response = session.handleMessage(HeartbeatMessage{"worker-a"}, 1050);
    const auto* heartbeat = std::get_if<HeartbeatAckMessage>(&response.message);
    valid &= check(heartbeat != nullptr && heartbeat->renewedLeaseCount == 1,
        "heartbeat did not renew the active lease");

    if (firstAssignment != nullptr) {
        response = session.handleMessage(TaskResultMessage{resultFor(firstAssignmentCopy)}, 1100);
        const auto* acknowledgment = std::get_if<ResultAckMessage>(&response.message);
        valid &= check(acknowledgment != nullptr &&
            acknowledgment->status == DistributedCommitStatus::Accepted &&
            acknowledgment->hasNextTask,
            "accepted result did not receive the next task inline");
        valid &= check(coordinator.completedTaskCount() == 1,
            "worker result was not committed to the coordinator");

        if (acknowledgment != nullptr && acknowledgment->hasNextTask) {
            response = session.handleMessage(
                TaskResultMessage{resultFor(acknowledgment->nextTask)}, 1200);
            const auto* secondAcknowledgment = std::get_if<ResultAckMessage>(&response.message);
            valid &= check(secondAcknowledgment != nullptr &&
                secondAcknowledgment->status == DistributedCommitStatus::Accepted,
                "inline next task result was not accepted");
        }
    }

    session.disconnect();
    valid &= check(!session.registered(), "worker session remained registered after disconnect");
    valid &= check(coordinator.pendingTaskCount() == 2,
        "disconnect did not release the remaining leased tasks");

    DistributedCoordinator invalidCoordinator(job, tasks);
    CoordinatorWorkerSession invalidSession(invalidCoordinator);
    response = invalidSession.handleMessage(WorkerHelloMessage{fingerprint + 1, "worker-b"}, 0);
    hello = std::get_if<WorkerHelloAcceptedMessage>(&response.message);
    valid &= check(!response.keepAlive && hello != nullptr && !hello->accepted,
        "mismatched worker fingerprint was accepted");

    return valid ? 0 : 1;
}
