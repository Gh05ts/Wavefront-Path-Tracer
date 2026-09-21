#include <iostream>

#include "config.hpp"
#include "distributed/coordinator_app.hpp"
#include "distributed/worker_app.hpp"
#include "renderer/render_driver.cuh"
#include "scene/scene_factory.hpp"

int main(int argc, char** argv) {
    RenderConfig config;
    SceneConfig sceneConfig = scenePreset(ScenePreset::Sponza);
    if (!parseCommandLine(argc, argv, config, sceneConfig))
        return 0;

    if (config.distributedRole == DistributedRole::Coordinator)
        return runDistributedCoordinator(config, sceneConfig);
    if (config.distributedRole == DistributedRole::Worker)
        return runDistributedWorker(config, sceneConfig);

    std::cout << "Starting wavefront path tracer\n";

    Camera camera = createSceneCamera(sceneConfig, config.width, config.height);
    DeviceScene deviceScene = createSceneFromConfig(sceneConfig);
    applySceneConfig(deviceScene, sceneConfig);
    std::cout << "Scene preset: " << sceneConfig.name << '\n';

    int result = 0;
    {
        RenderDriver renderDriver(deviceScene, camera, config, sceneConfig);
        result = renderDriver.run();
    }

    destroyDeviceScene(deviceScene);
    return result;
}
