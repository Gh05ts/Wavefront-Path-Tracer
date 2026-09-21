#pragma once
#include <cstdint>
#include <string>
#include "scene/scene.cuh"

enum class ScenePreset { Sponza, Cornell, Hurricane, Prism, Crystal, Deer, Demo };
enum class DistributedRole { Local, Coordinator, Worker };

struct RenderConfig {
    uint32_t width = 1920, height = 1080, tileWidth = 960, tileHeight = 540;
    uint32_t maxDepth = 32, russianRouletteStartDepth = 12, samplesPerPixel = 512, blockSize = 256;
    bool tiledRendering = false, intersectionDebug = false, shadingNormalDebug = false, resumeFromCheckpoint = false, persistentWavefront = false, profileQueues = false;
    const char* checkpointFilename = "render.checkpoint";
    uint32_t checkpointProgressPercent = 25, checkpointBufferCount = 3;
    const char* outputFilename = "render.ppm";
    bool enableCaustics = false;
    bool enableCausticGather = true;
    uint32_t causticPhotonCount = 1048576;
    uint32_t causticMaxDepth = 8;
    float causticGatherRadius = 0.06f;
    DistributedRole distributedRole = DistributedRole::Local;
    const char* listenHost = "0.0.0.0";
    uint16_t listenPort = 9000;
    const char* coordinatorHost = "127.0.0.1";
    uint16_t coordinatorPort = 9000;
    const char* workerId = nullptr;
    const char* distributedCheckpointFilename = "render.distributed.checkpoint";
    uint32_t distributedCheckpointIntervalSeconds = 30;
    uint32_t distributedSamplesPerTask = 16;
    uint64_t distributedLeaseDurationMs = 30000;
    bool pushAssets = false;
    const char* assetCacheDirectory = ".pathtracer-assets";
};

struct SceneConfig {
    ScenePreset preset = ScenePreset::Sponza;
    std::string name = "sponza";
    std::string objectFilename = "../assets/deer-obj.obj";
    std::string gltfFilename = "../assets/sponza/Sponza.gltf";
    float gltfScale = 1.0f, objectScale = 0.1f;
    Vec3 objectTranslation = Vec3(0.181f, -0.906f, -0.2f);
    CornellObjectSource objectSource = CornellObjectSource::Obj;
    CornellLightProfile lightProfile = CornellLightProfile::Standard;
    ObjAccelerationPolicy acceleration = ObjAccelerationPolicy::TlasBlasBvh2;
    bool neutralRoom = false, removeBackdrop = false, convertObjectMaterialsToDielectric = false;
    int32_t objectMaterialOverride = -1;
    bool addSponzaTopLight = true, ignoreSponzaLightOcclusion = false, useNormalMaps = true;
    float normalMapMinimumCosine = 0.0f;
};

SceneConfig scenePreset(ScenePreset preset);
bool parseScenePreset(const char* value, ScenePreset& preset);
bool parseCommandLine(int argc, char** argv, RenderConfig& render, SceneConfig& scene);
