#include "distributed/coordinator_app.hpp"

#include "distributed/checkpoint.hpp"
#include "distributed/assets.hpp"
#include "distributed/coordinator.hpp"
#include "distributed/task.hpp"
#include "distributed/transport.hpp"
#include "distributed/worker.hpp"
#include "renderer/image_output.hpp"
#include "renderer/render_fingerprint.hpp"

#include <chrono>
#include <atomic>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

namespace
{
void writeDistributedImage(
    const char* filename,
    const DistributedCheckpoint& checkpoint) {
    std::vector<Vec3> pixels(checkpoint.accumulatedRadiance.size());
    for (size_t i = 0; i < pixels.size(); ++i) {
        const uint32_t samples = checkpoint.sampleCounts[i];
        if (samples == 0)
            continue;
        const float inverseSamples = 1.0f / static_cast<float>(samples);
        pixels[i] = Vec3(
            checkpoint.accumulatedRadiance[i].x * inverseSamples,
            checkpoint.accumulatedRadiance[i].y * inverseSamples,
            checkpoint.accumulatedRadiance[i].z * inverseSamples);
    }
    writePpm(filename, pixels, checkpoint.job.width, checkpoint.job.height);
    std::cout << "Wrote " << filename << '\n';
}

bool writeDistributedSnapshot(
    const RenderConfig& config,
    const DistributedCoordinator& coordinator,
    uint64_t sequence) {
    std::string error;
    if (!writeDistributedCheckpoint(
            config.distributedCheckpointFilename,
            coordinator.snapshot(sequence),
            &error)) {
        std::cerr << "Could not write distributed checkpoint: " << error << '\n';
        return false;
    }
    return true;
}
} // namespace

int runDistributedCoordinator(const RenderConfig& config, const SceneConfig& scene) {
    const uint64_t fingerprint = computeRenderFingerprint(config, scene);
    const DistributedJobConfig job{
        fingerprint,
        config.width,
        config.height,
        config.tileWidth,
        config.tileHeight,
        config.samplesPerPixel,
        config.distributedSamplesPerTask};
    const std::vector<RenderTask> tasks = createRenderTasks(
        job.width,
        job.height,
        job.tileWidth,
        job.tileHeight,
        job.samplesPerPixel,
        job.samplesPerTask,
        job.jobFingerprint);
    if (tasks.empty()) {
        std::cerr << "Distributed coordinator produced no tasks; check render dimensions and sample settings\n";
        return 1;
    }

    DistributedAssetCatalog assetCatalog;
    if (config.pushAssets) {
        std::string assetError;
        if (!buildDistributedAssetCatalog(scene, assetCatalog, &assetError)) {
            std::cerr << "Could not build distributed asset catalog: " << assetError << '\n';
            return 1;
        }
        std::cout << "Distributed asset transfer: " << assetCatalog.files.size()
                  << " files" << '\n';
    }

    DistributedCoordinator coordinator(job, tasks);
    uint64_t checkpointSequence = 0;
    const uint64_t leaseDurationMs = config.distributedLeaseDurationMs;
    if (config.resumeFromCheckpoint) {
        DistributedCheckpoint checkpoint;
        std::string error;
        if (!readDistributedCheckpoint(config.distributedCheckpointFilename, checkpoint, &error) ||
            !coordinator.restore(checkpoint, &error)) {
            std::cerr << "Could not resume distributed checkpoint: " << error << '\n';
            return 1;
        }
        checkpointSequence = checkpoint.sequence;
        std::cout << "Resumed distributed checkpoint sequence " << checkpoint.sequence
                  << ": " << coordinator.completedTaskCount() << '/' << coordinator.taskCount()
                  << " tasks completed\n";
    }

    std::string error;
    auto listener = TcpListener::listenOn(config.listenHost, config.listenPort, &error);
    if (!listener) {
        std::cerr << "Could not start coordinator listener: " << error << '\n';
        return 1;
    }

    std::cout << "Distributed coordinator listening on " << config.listenHost
              << ':' << listener->port() << " for " << tasks.size() << " tasks\n";
    std::cout << "Distributed checkpoint: " << config.distributedCheckpointFilename << '\n';

    auto nextCheckpoint = std::chrono::steady_clock::now();
    bool finalImageWritten = false;
    std::atomic<uint32_t> activeSessions{0};
    const DistributedAssetCatalog* assetCatalogPtr = config.pushAssets ? &assetCatalog : nullptr;

    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (!finalImageWritten && now >= nextCheckpoint) {
            if (!writeDistributedSnapshot(config, coordinator, ++checkpointSequence))
                return 1;
            std::cout << "Checkpointed " << coordinator.completedTaskCount() << '/'
                      << coordinator.taskCount() << " distributed tasks\n";
            nextCheckpoint = now + std::chrono::seconds(config.distributedCheckpointIntervalSeconds);
        }

        if (!finalImageWritten && coordinator.complete()) {
            const DistributedCheckpoint checkpoint = coordinator.snapshot(++checkpointSequence);
            std::string checkpointError;
            if (!writeDistributedCheckpoint(config.distributedCheckpointFilename, checkpoint, &checkpointError)) {
                std::cerr << "Could not write final distributed checkpoint: " << checkpointError << '\n';
                return 1;
            }
            writeDistributedImage(config.outputFilename, checkpoint);
            finalImageWritten = true;
            std::cout << "Distributed render complete\n";
        }

        if (finalImageWritten) {
            // Existing workers receive the completion response on their next
            // request and then close their sessions. Do not accept new
            // workers or touch coordinator state after the final image; wait
            // until detached session threads have released their references.
            if (activeSessions.load() == 0)
                return 0;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::string acceptError;
        auto connection = listener->accept(&acceptError, 250);
        if (!connection) {
            if (acceptError == "accept timed out")
                continue;
            std::cerr << "Coordinator accept failed: " << acceptError << '\n';
            return 1;
        }

        TcpConnection workerConnection = std::move(*connection);
        activeSessions.fetch_add(1);
        std::thread([&coordinator, &activeSessions, leaseDurationMs, assetCatalogPtr, connection = std::move(workerConnection)]() mutable {
            CoordinatorWorkerSession session(
                coordinator,
                CoordinatorSessionConfig{leaseDurationMs, 1000, assetCatalogPtr});
            std::string sessionError;
            session.run(connection, &sessionError);
            if (!sessionError.empty())
                std::cerr << "Worker session ended: " << sessionError << '\n';
            activeSessions.fetch_sub(1);
        }).detach();
    }
}
