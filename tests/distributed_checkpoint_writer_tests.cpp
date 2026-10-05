#include "distributed/checkpoint_writer.hpp"

#include <cstdio>
#include <iostream>
#include <string>

namespace
{
bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "distributed_checkpoint_writer_tests: " << message << '\n';
    return condition;
}

DistributedCheckpoint checkpoint(uint64_t sequence, float value) {
    DistributedCheckpoint result;
    result.job = DistributedJobConfig{0x1234u, 1, 1, 1, 1, 4, 1};
    result.sequence = sequence;
    result.accumulatedRadiance = {DistributedRadiance{value, value + 1.0f, value + 2.0f}};
    result.sampleCounts = {static_cast<uint32_t>(sequence)};
    result.completedTasks = {static_cast<uint8_t>(sequence == 0 ? 0 : 1)};
    return result;
}
} // namespace

int main() {
    const std::string filename = "distributed_checkpoint_writer_tests.checkpoint";
    bool valid = true;

    {
        DistributedCheckpointWriter writer(filename);
        valid &= check(writer.enqueue(checkpoint(1, 1.0f)), "first checkpoint was not queued");
        valid &= check(writer.enqueue(checkpoint(2, 2.0f)), "second checkpoint was not queued");

        std::string error;
        valid &= check(writer.flush(&error), error.empty() ? "flush failed" : error.c_str());

        DistributedCheckpoint loaded;
        error.clear();
        valid &= check(readDistributedCheckpoint(filename.c_str(), loaded, &error),
            error.empty() ? "queued checkpoint could not be read" : error.c_str());
        valid &= check(loaded.sequence == 2, "writer did not retain the newest pending snapshot");
        valid &= check(loaded.accumulatedRadiance[0].x == 2.0f,
            "newest checkpoint payload was not written");

        valid &= check(writer.enqueue(checkpoint(3, 3.0f)), "third checkpoint was not queued");
    }

    DistributedCheckpoint loaded;
    std::string error;
    valid &= check(readDistributedCheckpoint(filename.c_str(), loaded, &error),
        error.empty() ? "shutdown did not flush the final checkpoint" : error.c_str());
    valid &= check(loaded.sequence == 3, "shutdown lost the final queued checkpoint");

    std::remove(filename.c_str());
    std::remove((filename + ".tmp").c_str());
    return valid ? 0 : 1;
}
