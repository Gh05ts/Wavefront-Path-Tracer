#include "scene/scene_factory.hpp"

#include <cstdlib>
#include <iostream>

namespace
{
bool usesCornellCamera(ScenePreset preset) {
    return preset == ScenePreset::Cornell || preset == ScenePreset::Hurricane ||
        preset == ScenePreset::Prism || preset == ScenePreset::Crystal;
}
}

Camera createSceneCamera(const SceneConfig& config, uint32_t width, uint32_t height) {
    return usesCornellCamera(config.preset) ?
        createCornellCamera(width, height) : createDemoCamera(width, height);
}

DeviceScene createSceneFromConfig(const SceneConfig& config) {
    switch (config.preset) {
    case ScenePreset::Sponza:
        return createGltfScene(config.gltfFilename.c_str(), config.gltfScale, config.addSponzaTopLight);

    case ScenePreset::Cornell:
        return createCornellScene(CornellSceneOptions{
            config.objectFilename.empty() ? nullptr : config.objectFilename.c_str(),
            config.objectSource,
            config.objectScale,
            config.objectTranslation,
            config.neutralRoom,
            config.removeBackdrop,
            config.convertObjectMaterialsToDielectric,
            config.objectMaterialOverride,
            config.lightProfile});

    case ScenePreset::Hurricane:
        return createCornellScene(CornellSceneOptions{
            config.objectFilename.empty() ? nullptr : config.objectFilename.c_str(),
            config.objectSource,
            config.objectScale,
            config.objectTranslation,
            config.neutralRoom,
            config.removeBackdrop,
            config.convertObjectMaterialsToDielectric,
            config.objectMaterialOverride,
            config.lightProfile});

    case ScenePreset::Prism:
        return createCornellScene(CornellSceneOptions{
            config.objectFilename.empty() ? nullptr : config.objectFilename.c_str(),
            config.objectSource,
            config.objectScale,
            config.objectTranslation,
            config.neutralRoom,
            config.removeBackdrop,
            config.convertObjectMaterialsToDielectric,
            config.objectMaterialOverride,
            config.lightProfile});

    case ScenePreset::Crystal:
        return createCornellScene(CornellSceneOptions{
            config.objectFilename.empty() ? nullptr : config.objectFilename.c_str(),
            config.objectSource,
            config.objectScale,
            config.objectTranslation,
            config.neutralRoom,
            config.removeBackdrop,
            config.convertObjectMaterialsToDielectric,
            config.objectMaterialOverride,
            config.lightProfile});

    case ScenePreset::Deer: {
        ObjSceneOptions options;
        options.filename = config.objectFilename.empty() ? nullptr : config.objectFilename.c_str();
        options.objectScale = config.objectScale;
        options.objectTranslation = config.objectTranslation;
        options.acceleration = config.acceleration;
        return createObjScene(options);
    }

    case ScenePreset::Demo:
        return createDemoScene();
    }

    std::cerr << "Unsupported scene preset\n";
    std::exit(1);
}

void applySceneConfig(DeviceScene& deviceScene, const SceneConfig& config) {
    deviceScene.scene.ignoreDirectLightOcclusion =
        config.preset == ScenePreset::Sponza && config.ignoreSponzaLightOcclusion;
    deviceScene.scene.useNormalMaps = config.useNormalMaps;
    deviceScene.scene.normalMapMinimumCosine = config.normalMapMinimumCosine;
}
