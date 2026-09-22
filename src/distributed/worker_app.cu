#include "distributed/worker_app.hpp"

#include "distributed/worker.hpp"
#include "renderer/render_driver.cuh"
#include "renderer/render_fingerprint.hpp"
#include "renderer/render_task.cuh"
#include "scene/scene_factory.hpp"

#include <algorithm>
#include <chrono>
#include <atomic>
#include <iostream>
#include <thread>

namespace
{
DistributedTaskResult convertResult(const RenderTaskResult& localResult, uint64_t fingerprint) {
    DistributedTaskResult result;
    result.jobFingerprint = fingerprint;
    result.taskId = localResult.taskId;
    result.tile = localResult.tile;
    result.sampleCount = localResult.sampleCount;
    result.radiance.resize(localResult.radiance.size());
    for (size_t i = 0; i < localResult.radiance.size(); ++i) {
        result.radiance[i] = DistributedRadiance{
            localResult.radiance[i].x,
            localResult.radiance[i].y,
            localResult.radiance[i].z};
    }
    return result;
}
} // namespace

int runDistributedWorker(const RenderConfig& config, const SceneConfig& scene) {
    SceneConfig workerScene = scene;
    // With asset transfer enabled, the worker may not have the primary asset
    // yet, so defer fingerprint calculation until the cache is populated.
    uint64_t fingerprint = config.pushAssets ? 0 : computeRenderFingerprint(config, workerScene);
    std::string error;
    auto client = DistributedWorkerClient::connectToCoordinator(
        config.coordinatorHost,
        config.coordinatorPort,
        config.workerId,
        fingerprint,
        &error);
    if (!client) {
        std::cerr << "Could not register worker: " << error << '\n';
        return 1;
    }

    std::cout << "Registered distributed worker: " << config.workerId << '\n';

    if (config.pushAssets) {
        if (!client->synchronizeAssets(workerScene, config.assetCacheDirectory, &error)) {
            std::cerr << "Could not synchronize distributed assets: " << error << '\n';
            client->close();
            return 1;
        }
        fingerprint = computeRenderFingerprint(config, workerScene);
        if (fingerprint != client->coordinatorJobFingerprint()) {
            std::cerr << "Transferred assets produced a different render fingerprint\n";
            client->close();
            return 1;
        }
        std::cout << "Synchronized " << config.assetCacheDirectory << " assets\n";
    }

    Camera camera = createSceneCamera(workerScene, config.width, config.height);
    DeviceScene deviceScene = createSceneFromConfig(workerScene);
    applySceneConfig(deviceScene, workerScene);
    std::cout << "Scene preset: " << workerScene.name << '\n';

    int resultCode = 0;
    {
        RenderDriver renderDriver(deviceScene, camera, config, workerScene);
        RenderTaskRenderer taskRenderer(renderDriver);

        bool hasTask = false;
        TaskAssignmentMessage assignment;
        while (client->valid()) {
            if (!hasTask) {
                uint32_t retryAfterMs = 1000;
                if (!client->requestTask(assignment, hasTask, &retryAfterMs, &error)) {
                    std::cerr << "Worker task request failed: " << error << '\n';
                    resultCode = 1;
                    break;
                }
                if (!hasTask) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(retryAfterMs));
                    continue;
                }
            }

            std::cout << "Rendering tile " << assignment.task.id.tileOrdinal
                      << " samples " << assignment.task.id.sampleStart << '-'
                      << assignment.task.id.sampleStart + assignment.task.id.sampleCount << '\n';

            std::atomic<bool> stopHeartbeat = false;
            std::atomic<bool> heartbeatFailed = false;
            std::string heartbeatError;
            const uint64_t heartbeatIntervalMs = std::max<uint64_t>(
                1000ull,
                config.distributedLeaseDurationMs / 3ull);
            std::thread heartbeatThread([&] {
                while (!stopHeartbeat.load()) {
                    const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(heartbeatIntervalMs);
                    while (!stopHeartbeat.load() && std::chrono::steady_clock::now() < deadline)
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (stopHeartbeat.load())
                        break;

                    HeartbeatAckMessage heartbeat;
                    if (!client->heartbeat(heartbeat, &heartbeatError)) {
                        heartbeatFailed = true;
                        break;
                    }
                }
            });
            const RenderTaskResult localResult = taskRenderer.render(assignment.task);
            stopHeartbeat = true;
            heartbeatThread.join();
            if (heartbeatFailed.load()) {
                std::cerr << "Worker heartbeat failed during task: " << heartbeatError << '\n';
                resultCode = 1;
                break;
            }
            if (!localResult.success) {
                std::cerr << "Local task rendering failed\n";
                resultCode = 1;
                break;
            }

            ResultAckMessage acknowledgment;
            if (!client->submitResult(convertResult(localResult, fingerprint), acknowledgment, &error)) {
                std::cerr << "Worker result submission failed: " << error << '\n';
                resultCode = 1;
                break;
            }
            if (acknowledgment.status == DistributedCommitStatus::Rejected) {
                std::cerr << "Coordinator rejected worker result\n";
                resultCode = 1;
                break;
            }

            if (acknowledgment.hasNextTask) {
                assignment = acknowledgment.nextTask;
                hasTask = true;
            } else {
                hasTask = false;
            }
        }
    }

    client->close();
    destroyDeviceScene(deviceScene);
    return resultCode;
}
