#include "config.hpp"
#include "renderer/render_fingerprint.hpp"

#include <cstring>
#include <iostream>

namespace
{
bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "config_tests: " << message << '\n';
    return condition;
}
} // namespace

int main() {
    bool valid = true;
    const char* names[] = {"sponza", "cornell", "hurricane", "prism", "crystal", "deer", "demo"};

    for (const char* name : names) {
        ScenePreset preset = ScenePreset::Demo;
        valid &= check(parseScenePreset(name, preset), "known scene preset was rejected");
        SceneConfig config = scenePreset(preset);
        valid &= check(config.name == name, "scene preset name does not round-trip");
    }

    SceneConfig cornellConfig = scenePreset(ScenePreset::Cornell);
    valid &= check(cornellConfig.objectSource == CornellObjectSource::Gltf &&
        cornellConfig.removeBackdrop, "Cornell manifest options were not loaded");
    SceneConfig prismConfig = scenePreset(ScenePreset::Prism);
    valid &= check(prismConfig.objectSource == CornellObjectSource::ProceduralPrism &&
        prismConfig.lightProfile == CornellLightProfile::Prism,
        "prism manifest options were not loaded");
    SceneConfig crystalConfig = scenePreset(ScenePreset::Crystal);
    valid &= check(crystalConfig.convertObjectMaterialsToDielectric &&
        crystalConfig.objectSource == CornellObjectSource::Obj,
        "crystal manifest options were not loaded");

    ScenePreset unknownPreset = ScenePreset::Demo;
    valid &= check(!parseScenePreset("unknown", unknownPreset), "unknown scene preset was accepted");

    RenderConfig render;
    SceneConfig scene = scenePreset(ScenePreset::Sponza);
    char arg0[] = "pathtracer";
    char arg1[] = "--scene";
    char arg2[] = "crystal";
    char arg3[] = "--caustics";
    char arg4[] = "--no-caustic-gather";
    char arg5[] = "--persistent-wavefront";
    char* argv[] = {arg0, arg1, arg2, arg3, arg4, arg5};
    valid &= check(parseCommandLine(static_cast<int>(sizeof(argv) / sizeof(argv[0])), argv, render, scene), "valid command line was rejected");
    valid &= check(scene.preset == ScenePreset::Crystal, "command line scene did not update the preset");
    valid &= check(render.enableCaustics && !render.enableCausticGather && render.persistentWavefront, "command line flags did not update render configuration");

    RenderConfig coordinatorRender;
    SceneConfig coordinatorScene = scenePreset(ScenePreset::Demo);
    char coordinatorArg0[] = "pathtracer";
    char coordinatorArg1[] = "--coordinator";
    char coordinatorArg2[] = "--listen-host";
    char coordinatorArg3[] = "127.0.0.1";
    char coordinatorArg4[] = "--listen-port";
    char coordinatorArg5[] = "9010";
    char coordinatorArg6[] = "--samples-per-task";
    char coordinatorArg7[] = "8";
    char coordinatorArg8[] = "--distributed-lease-seconds";
    char coordinatorArg9[] = "45";
    char* coordinatorArgv[] = {
        coordinatorArg0, coordinatorArg1, coordinatorArg2, coordinatorArg3,
        coordinatorArg4, coordinatorArg5, coordinatorArg6, coordinatorArg7,
        coordinatorArg8, coordinatorArg9};
    valid &= check(parseCommandLine(
        static_cast<int>(sizeof(coordinatorArgv) / sizeof(coordinatorArgv[0])),
        coordinatorArgv,
        coordinatorRender,
        coordinatorScene), "coordinator command line was rejected");
    valid &= check(coordinatorRender.distributedRole == DistributedRole::Coordinator &&
        coordinatorRender.tiledRendering && coordinatorRender.listenPort == 9010 &&
        coordinatorRender.distributedSamplesPerTask == 8 &&
        coordinatorRender.distributedLeaseDurationMs == 45000,
        "coordinator options did not update distributed configuration");

    RenderConfig workerRender;
    SceneConfig workerScene = scenePreset(ScenePreset::Demo);
    char workerArg0[] = "pathtracer";
    char workerArg1[] = "--worker";
    char workerArg2[] = "--worker-id";
    char workerArg3[] = "gpu-0";
    char workerArg4[] = "--coordinator-host";
    char workerArg5[] = "192.168.1.10";
    char workerArg6[] = "--coordinator-port";
    char workerArg7[] = "9010";
    char* workerArgv[] = {
        workerArg0, workerArg1, workerArg2, workerArg3,
        workerArg4, workerArg5, workerArg6, workerArg7};
    valid &= check(parseCommandLine(
        static_cast<int>(sizeof(workerArgv) / sizeof(workerArgv[0])),
        workerArgv,
        workerRender,
        workerScene), "worker command line was rejected");
    valid &= check(workerRender.distributedRole == DistributedRole::Worker &&
        workerRender.tiledRendering && std::strcmp(workerRender.workerId, "gpu-0") == 0 &&
        std::strcmp(workerRender.coordinatorHost, "192.168.1.10") == 0 &&
        workerRender.coordinatorPort == 9010,
        "worker options did not update distributed configuration");

    RenderConfig fingerprintRender;
    SceneConfig fingerprintScene = scenePreset(ScenePreset::Demo);
    uint64_t firstFingerprint = computeRenderFingerprint(fingerprintRender, fingerprintScene);
    valid &= check(firstFingerprint == computeRenderFingerprint(fingerprintRender, fingerprintScene), "fingerprint is not stable");
    fingerprintRender.maxDepth += 1;
    valid &= check(firstFingerprint != computeRenderFingerprint(fingerprintRender, fingerprintScene), "render configuration change did not change fingerprint");

    return valid ? 0 : 1;
}
