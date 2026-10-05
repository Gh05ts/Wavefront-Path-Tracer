#include <iostream>
#include <string>
#include <vector>

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

    const bool interactiveCoordinator = argc == 2 &&
        std::string(argv[1]) == "--coordinator";
    std::vector<std::string> coordinatorCommandArguments;
    if (interactiveCoordinator) {
        std::cout << "Enter scene/render options for distributed workers "
                     "(for example: --scene sponza --push-assets --caustics):\n> "
                  << std::flush;
        std::string command;
        if (!std::getline(std::cin, command) ||
            !parseCommandLineText(command, config, sceneConfig, coordinatorCommandArguments))
            return 1;
        config.pushConfigurationToWorkers = true;
        std::cout << "Coordinator configuration will be sent to workers before task requests\n";
    }

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
