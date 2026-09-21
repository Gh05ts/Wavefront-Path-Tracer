#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "task.hpp"

struct DistributedJobConfig {
    uint64_t jobFingerprint = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tileWidth = 0;
    uint32_t tileHeight = 0;
    uint32_t samplesPerPixel = 0;
    uint32_t samplesPerTask = 0;

    bool operator==(const DistributedJobConfig& other) const {
        return jobFingerprint == other.jobFingerprint &&
            width == other.width &&
            height == other.height &&
            tileWidth == other.tileWidth &&
            tileHeight == other.tileHeight &&
            samplesPerPixel == other.samplesPerPixel &&
            samplesPerTask == other.samplesPerTask;
    }
};

struct DistributedCheckpoint {
    DistributedJobConfig job;
    uint64_t sequence = 0;
    std::vector<DistributedRadiance> accumulatedRadiance;
    std::vector<uint32_t> sampleCounts;

    // One byte per deterministic task: zero means unfinished, nonzero means
    // the task result has been merged into accumulatedRadiance.
    std::vector<uint8_t> completedTasks;
};

bool writeDistributedCheckpoint(const char* filename, const DistributedCheckpoint& checkpoint, std::string* error = nullptr);
bool readDistributedCheckpoint(const char* filename, DistributedCheckpoint& checkpoint, std::string* error = nullptr);
