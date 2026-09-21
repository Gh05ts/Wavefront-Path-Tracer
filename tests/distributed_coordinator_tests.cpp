#include "distributed/coordinator.hpp"

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace
{
bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "distributed_coordinator_tests: " << message << '\n';
    return condition;
}

DistributedTaskResult resultFor(const DistributedTaskLease& lease, float value) {
    DistributedTaskResult result;
    result.jobFingerprint = lease.task.jobFingerprint;
    result.taskId = lease.task.id;
    result.tile = lease.task.tile;
    result.sampleCount = lease.task.id.sampleCount;
    result.radiance.resize(static_cast<size_t>(result.tile.width) * result.tile.height);
    for (DistributedRadiance& pixel : result.radiance)
        pixel = DistributedRadiance{value, value + 1.0f, value + 2.0f};
    return result;
}
} // namespace

int main() {
    bool valid = true;
    constexpr uint64_t fingerprint = 0x13579bdf2468ace0ull;
    DistributedJobConfig job{fingerprint, 4, 2, 2, 2, 4, 2};
    const std::vector<RenderTask> tasks = createRenderTasks(
        job.width,
        job.height,
        job.tileWidth,
        job.tileHeight,
        job.samplesPerPixel,
        job.samplesPerTask,
        job.jobFingerprint);

    DistributedCoordinator coordinator(job, tasks);
    valid &= check(coordinator.taskCount() == 4, "unexpected coordinator task count");
    valid &= check(coordinator.pendingTaskCount() == 4, "initial pending task count is wrong");

    const auto firstLease = coordinator.leaseNext("worker-a", 0, 100);
    const auto secondLease = coordinator.leaseNext("worker-b", 0, 100);
    valid &= check(firstLease.has_value() && secondLease.has_value(), "workers did not receive leases");
    valid &= check(!(firstLease->task.id == secondLease->task.id), "two workers received the same task");

    DistributedTaskResult firstResult = resultFor(*firstLease, 4.0f);
    valid &= check(coordinator.commitResult(firstResult) == DistributedCommitStatus::Accepted,
        "valid task result was rejected");
    valid &= check(coordinator.commitResult(firstResult) == DistributedCommitStatus::Duplicate,
        "duplicate task result was merged twice");
    valid &= check(coordinator.completedTaskCount() == 1, "completed task count did not advance");
    valid &= check(coordinator.sampleCounts()[0] == 2, "sample count was not accumulated");
    valid &= check(coordinator.accumulatedRadiance()[0].x == 4.0f, "radiance sum was not accumulated");

    DistributedCheckpoint checkpoint = coordinator.snapshot(7);
    const std::string filename = "distributed_coordinator_tests.checkpoint";
    std::string error;
    valid &= check(writeDistributedCheckpoint(filename.c_str(), checkpoint, &error),
        error.empty() ? "distributed checkpoint write failed" : error.c_str());

    DistributedCheckpoint loaded;
    error.clear();
    valid &= check(readDistributedCheckpoint(filename.c_str(), loaded, &error),
        error.empty() ? "distributed checkpoint read failed" : error.c_str());
    valid &= check(loaded.sequence == 7, "checkpoint sequence did not round-trip");
    valid &= check(loaded.completedTasks.size() == 4 && loaded.completedTasks[0] == 1,
        "checkpoint completion bitmap did not round-trip");

    DistributedCoordinator restored(job, tasks);
    error.clear();
    valid &= check(restored.restore(loaded, &error),
        error.empty() ? "coordinator restore failed" : error.c_str());
    valid &= check(restored.taskState(firstLease->task.id) == DistributedTaskState::Completed,
        "restored coordinator forgot completed task");
    valid &= check(restored.taskState(secondLease->task.id) == DistributedTaskState::Pending,
        "leased task was not made pending after restore");
    valid &= check(restored.pendingTaskCount() == 3, "restored pending task count is wrong");
    valid &= check(restored.commitResult(firstResult) == DistributedCommitStatus::Duplicate,
        "restored coordinator accepted a duplicate result");

    valid &= check(coordinator.requeueExpired(99) == 0, "lease expired too early");
    valid &= check(coordinator.requeueExpired(100) == 1, "expired lease was not requeued");
    valid &= check(coordinator.pendingTaskCount() == 3, "requeued task did not return to pending queue");

    DistributedTaskResult invalid = resultFor(*secondLease, 1.0f);
    invalid.radiance.pop_back();
    valid &= check(coordinator.commitResult(invalid) == DistributedCommitStatus::Rejected,
        "malformed task result was accepted");

    std::remove(filename.c_str());
    std::remove((filename + ".tmp").c_str());
    return valid ? 0 : 1;
}
