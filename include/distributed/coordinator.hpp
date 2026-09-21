#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "checkpoint.hpp"

enum class DistributedTaskState : uint8_t {
    Pending,
    Leased,
    Completed
};

struct DistributedTaskLease {
    RenderTask task;
    std::string workerId;
    uint64_t expiresAtMs = 0;
};

enum class DistributedCommitStatus : uint8_t {
    Accepted,
    Duplicate,
    Rejected
};

// Host-only coordinator state. It is deliberately independent of CUDA and
// networking so task assignment, merging, and recovery can be tested locally.
class DistributedCoordinator {
public:
    DistributedCoordinator(DistributedJobConfig job, std::vector<RenderTask> tasks);

    std::optional<DistributedTaskLease> leaseNext(
        const std::string& workerId,
        uint64_t nowMs,
        uint64_t leaseDurationMs);

    DistributedCommitStatus commitResult(const DistributedTaskResult& result);

    uint32_t renewWorkerLeases(const std::string& workerId, uint64_t nowMs, uint64_t leaseDurationMs);
    uint32_t releaseWorkerLeases(const std::string& workerId);
    uint32_t requeueExpired(uint64_t nowMs);
    bool restore(const DistributedCheckpoint& checkpoint, std::string* error = nullptr);
    DistributedCheckpoint snapshot(uint64_t sequence) const;

    const DistributedJobConfig& job() const { return job_; }
    const std::vector<DistributedRadiance>& accumulatedRadiance() const { return accumulatedRadiance_; }
    const std::vector<uint32_t>& sampleCounts() const { return sampleCounts_; }

    size_t taskCount() const { return tasks_.size(); }
    size_t pendingTaskCount() const;
    size_t completedTaskCount() const;
    bool complete() const;
    DistributedTaskState taskState(const RenderTaskId& id) const;

private:
    struct TaskRecord {
        RenderTask task;
        DistributedTaskState state = DistributedTaskState::Pending;
        std::string workerId;
        uint64_t expiresAtMs = 0;
    };

    static bool sameTile(const DistributedTile& first, const DistributedTile& second);
    size_t findTask(const RenderTaskId& id) const;
    bool validateResult(const DistributedTaskResult& result, size_t taskIndex) const;
    void rebuildPendingQueue();

    DistributedJobConfig job_;
    std::vector<TaskRecord> tasks_;
    std::vector<DistributedRadiance> accumulatedRadiance_;
    std::vector<uint32_t> sampleCounts_;
    std::deque<size_t> pendingTasks_;
    size_t completedTaskCount_ = 0;
    mutable std::mutex mutex_;
};
