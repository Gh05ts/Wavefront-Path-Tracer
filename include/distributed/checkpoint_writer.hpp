#pragma once

#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "checkpoint.hpp"

// Writes complete distributed checkpoints on a background host thread. Only
// the newest snapshot waiting in the queue is retained: every snapshot is a
// full state image, so older pending snapshots cannot add recoverability.
class DistributedCheckpointWriter {
public:
    explicit DistributedCheckpointWriter(std::string filename);
    ~DistributedCheckpointWriter();

    DistributedCheckpointWriter(const DistributedCheckpointWriter&) = delete;
    DistributedCheckpointWriter& operator=(const DistributedCheckpointWriter&) = delete;

    bool enqueue(DistributedCheckpoint checkpoint, std::string* error = nullptr);
    bool flush(std::string* error = nullptr);
    bool failed(std::string* error = nullptr) const;
    void shutdown();

private:
    void run();

    std::string filename_;
    mutable std::mutex mutex_;
    std::condition_variable workAvailable_;
    std::condition_variable idle_;
    std::optional<DistributedCheckpoint> pending_;
    std::thread thread_;
    bool writing_ = false;
    bool stopping_ = false;
    bool failed_ = false;
    std::string error_;
};
