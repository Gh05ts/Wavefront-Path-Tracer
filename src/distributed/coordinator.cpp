#include "distributed/coordinator.hpp"

#include <algorithm>
#include <limits>

namespace
{
bool sameId(const RenderTaskId& first, const RenderTaskId& second) {
    return first == second;
}
} // namespace

DistributedCoordinator::DistributedCoordinator(DistributedJobConfig job, std::vector<RenderTask> tasks)
    : job_(job), accumulatedRadiance_(static_cast<size_t>(job.width) * job.height),
      sampleCounts_(static_cast<size_t>(job.width) * job.height, 0) {
    tasks_.reserve(tasks.size());
    for (const RenderTask& task : tasks) {
        if (task.jobFingerprint != job_.jobFingerprint)
            continue;
        tasks_.push_back(TaskRecord{task, DistributedTaskState::Pending, {}, 0});
    }
    rebuildPendingQueue();
}

std::optional<DistributedTaskLease> DistributedCoordinator::leaseNext(
    const std::string& workerId,
    uint64_t nowMs,
    uint64_t leaseDurationMs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (workerId.empty() || pendingTasks_.empty())
        return std::nullopt;

    const size_t taskIndex = pendingTasks_.front();
    pendingTasks_.pop_front();
    TaskRecord& record = tasks_[taskIndex];
    record.state = DistributedTaskState::Leased;
    record.workerId = workerId;
    record.expiresAtMs = nowMs + leaseDurationMs;
    return DistributedTaskLease{record.task, workerId, record.expiresAtMs};
}

DistributedCommitStatus DistributedCoordinator::commitResult(const DistributedTaskResult& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (result.jobFingerprint != job_.jobFingerprint)
        return DistributedCommitStatus::Rejected;

    const size_t taskIndex = findTask(result.taskId);
    if (taskIndex == tasks_.size())
        return DistributedCommitStatus::Rejected;

    TaskRecord& record = tasks_[taskIndex];
    if (record.state == DistributedTaskState::Completed)
        return DistributedCommitStatus::Duplicate;
    if (!validateResult(result, taskIndex))
        return DistributedCommitStatus::Rejected;

    for (uint32_t localY = 0; localY < record.task.tile.height; ++localY) {
        for (uint32_t localX = 0; localX < record.task.tile.width; ++localX) {
            const size_t globalIndex =
                static_cast<size_t>(record.task.tile.y + localY) * job_.width +
                record.task.tile.x + localX;
            if (sampleCounts_[globalIndex] > std::numeric_limits<uint32_t>::max() - result.sampleCount)
                return DistributedCommitStatus::Rejected;
        }
    }

    for (uint32_t localY = 0; localY < record.task.tile.height; ++localY) {
        for (uint32_t localX = 0; localX < record.task.tile.width; ++localX) {
            const size_t localIndex = static_cast<size_t>(localY) * record.task.tile.width + localX;
            const size_t globalIndex =
                static_cast<size_t>(record.task.tile.y + localY) * job_.width +
                record.task.tile.x + localX;
            accumulatedRadiance_[globalIndex] += result.radiance[localIndex];
            sampleCounts_[globalIndex] += result.sampleCount;
        }
    }

    if (record.state == DistributedTaskState::Pending) {
        const auto pending = std::find(pendingTasks_.begin(), pendingTasks_.end(), taskIndex);
        if (pending != pendingTasks_.end())
            pendingTasks_.erase(pending);
    }
    record.state = DistributedTaskState::Completed;
    record.workerId.clear();
    record.expiresAtMs = 0;
    ++completedTaskCount_;
    return DistributedCommitStatus::Accepted;
}

uint32_t DistributedCoordinator::renewWorkerLeases(
    const std::string& workerId,
    uint64_t nowMs,
    uint64_t leaseDurationMs) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (workerId.empty())
        return 0;

    uint32_t renewed = 0;
    for (TaskRecord& record : tasks_) {
        if (record.state == DistributedTaskState::Leased && record.workerId == workerId) {
            record.expiresAtMs = nowMs + leaseDurationMs;
            ++renewed;
        }
    }
    return renewed;
}

uint32_t DistributedCoordinator::releaseWorkerLeases(const std::string& workerId) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (workerId.empty())
        return 0;

    uint32_t released = 0;
    for (size_t i = 0; i < tasks_.size(); ++i) {
        TaskRecord& record = tasks_[i];
        if (record.state == DistributedTaskState::Leased && record.workerId == workerId) {
            record.state = DistributedTaskState::Pending;
            record.workerId.clear();
            record.expiresAtMs = 0;
            pendingTasks_.push_back(i);
            ++released;
        }
    }
    return released;
}

uint32_t DistributedCoordinator::requeueExpired(uint64_t nowMs) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t requeued = 0;
    for (size_t i = 0; i < tasks_.size(); ++i) {
        TaskRecord& record = tasks_[i];
        if (record.state == DistributedTaskState::Leased && record.expiresAtMs <= nowMs) {
            record.state = DistributedTaskState::Pending;
            record.workerId.clear();
            record.expiresAtMs = 0;
            pendingTasks_.push_back(i);
            ++requeued;
        }
    }
    return requeued;
}

bool DistributedCoordinator::restore(const DistributedCheckpoint& checkpoint, std::string* error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!(checkpoint.job == job_)) {
        if (error != nullptr)
            *error = "distributed checkpoint job configuration does not match coordinator";
        return false;
    }
    if (checkpoint.accumulatedRadiance.size() != accumulatedRadiance_.size() ||
        checkpoint.sampleCounts.size() != sampleCounts_.size() ||
        checkpoint.completedTasks.size() != tasks_.size()) {
        if (error != nullptr)
            *error = "distributed checkpoint dimensions or task count do not match coordinator";
        return false;
    }

    accumulatedRadiance_ = checkpoint.accumulatedRadiance;
    sampleCounts_ = checkpoint.sampleCounts;
    for (uint8_t completed : checkpoint.completedTasks) {
        if (completed > 1) {
            if (error != nullptr)
                *error = "distributed checkpoint completion bitmap is invalid";
            return false;
        }
    }

    completedTaskCount_ = 0;
    for (size_t i = 0; i < tasks_.size(); ++i)
        tasks_[i].state = checkpoint.completedTasks[i] != 0 ? DistributedTaskState::Completed : DistributedTaskState::Pending;
    rebuildPendingQueue();
    return true;
}

DistributedCheckpoint DistributedCoordinator::snapshot(uint64_t sequence) const {
    std::lock_guard<std::mutex> lock(mutex_);
    DistributedCheckpoint checkpoint;
    checkpoint.job = job_;
    checkpoint.sequence = sequence;
    checkpoint.accumulatedRadiance = accumulatedRadiance_;
    checkpoint.sampleCounts = sampleCounts_;
    checkpoint.completedTasks.resize(tasks_.size(), 0);
    for (size_t i = 0; i < tasks_.size(); ++i) {
        if (tasks_[i].state == DistributedTaskState::Completed) {
            checkpoint.completedTasks[i] = 1;
        }
    }
    return checkpoint;
}

DistributedTaskState DistributedCoordinator::taskState(const RenderTaskId& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t taskIndex = findTask(id);
    return taskIndex == tasks_.size() ? DistributedTaskState::Pending : tasks_[taskIndex].state;
}

size_t DistributedCoordinator::pendingTaskCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return pendingTasks_.size();
}

size_t DistributedCoordinator::completedTaskCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completedTaskCount_;
}

bool DistributedCoordinator::complete() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completedTaskCount_ == tasks_.size();
}

bool DistributedCoordinator::sameTile(const DistributedTile& first, const DistributedTile& second) {
    return first.x == second.x && first.y == second.y &&
        first.width == second.width && first.height == second.height;
}

size_t DistributedCoordinator::findTask(const RenderTaskId& id) const {
    for (size_t i = 0; i < tasks_.size(); ++i) {
        if (sameId(tasks_[i].task.id, id))
            return i;
    }
    return tasks_.size();
}

bool DistributedCoordinator::validateResult(const DistributedTaskResult& result, size_t taskIndex) const {
    const RenderTask& task = tasks_[taskIndex].task;
    const size_t expectedRadianceCount = static_cast<size_t>(task.tile.width) * task.tile.height;
    return sameTile(result.tile, task.tile) &&
        result.sampleCount == task.id.sampleCount &&
        result.radiance.size() == expectedRadianceCount;
}

void DistributedCoordinator::rebuildPendingQueue() {
    pendingTasks_.clear();
    completedTaskCount_ = 0;
    for (size_t i = 0; i < tasks_.size(); ++i) {
        if (tasks_[i].state == DistributedTaskState::Completed) {
            ++completedTaskCount_;
        } else {
            tasks_[i].state = DistributedTaskState::Pending;
            tasks_[i].workerId.clear();
            tasks_[i].expiresAtMs = 0;
            pendingTasks_.push_back(i);
        }
    }
}
