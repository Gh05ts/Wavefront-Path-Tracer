#include "distributed/checkpoint_writer.hpp"

#include <utility>

namespace
{
void setError(std::string* error, const std::string& message) {
    if (error != nullptr)
        *error = message;
}
} // namespace

DistributedCheckpointWriter::DistributedCheckpointWriter(std::string filename)
    : filename_(std::move(filename)), thread_(&DistributedCheckpointWriter::run, this) {}

DistributedCheckpointWriter::~DistributedCheckpointWriter() {
    shutdown();
}

bool DistributedCheckpointWriter::enqueue(DistributedCheckpoint checkpoint, std::string* error) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) {
            setError(error, error_);
            return false;
        }
        if (stopping_) {
            setError(error, "distributed checkpoint writer is stopped");
            return false;
        }

        // A checkpoint is a complete snapshot. If the writer is currently
        // busy, replacing a not-yet-written snapshot keeps coordinator memory
        // bounded without sacrificing the newest recoverable state.
        pending_ = std::move(checkpoint);
    }
    workAvailable_.notify_one();
    return true;
}

bool DistributedCheckpointWriter::flush(std::string* error) {
    std::unique_lock<std::mutex> lock(mutex_);
    idle_.wait(lock, [this] {
        return (!pending_.has_value() && !writing_) || failed_;
    });
    if (failed_) {
        setError(error, error_);
        return false;
    }
    return true;
}

bool DistributedCheckpointWriter::failed(std::string* error) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!failed_)
        return false;
    setError(error, error_);
    return true;
}

void DistributedCheckpointWriter::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!thread_.joinable())
            return;
        stopping_ = true;
    }
    workAvailable_.notify_one();
    thread_.join();
}

void DistributedCheckpointWriter::run() {
    for (;;) {
        DistributedCheckpoint checkpoint;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            workAvailable_.wait(lock, [this] {
                return stopping_ || pending_.has_value();
            });

            if (!pending_.has_value()) {
                if (stopping_)
                    break;
                continue;
            }

            checkpoint = std::move(*pending_);
            pending_.reset();
            writing_ = true;
        }

        std::string writeError;
        const bool success = writeDistributedCheckpoint(
            filename_.c_str(), checkpoint, &writeError);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            writing_ = false;
            if (!success) {
                failed_ = true;
                error_ = writeError.empty()
                    ? "distributed checkpoint write failed"
                    : std::move(writeError);
                pending_.reset();
                stopping_ = true;
            }
        }
        idle_.notify_all();

        if (!success)
            break;
    }

    idle_.notify_all();
}
