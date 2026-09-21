#include "config.hpp"
#include "scene/scene_manifest.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

static const char* nameOf(ScenePreset preset) {
    switch (preset) { case ScenePreset::Sponza: return "sponza"; case ScenePreset::Cornell: return "cornell"; case ScenePreset::Hurricane: return "hurricane"; case ScenePreset::Prism: return "prism"; case ScenePreset::Crystal: return "crystal"; case ScenePreset::Deer: return "deer"; case ScenePreset::Demo: return "demo"; }
    return "unknown";
}

static void printHelp() {
    std::cout << "Usage: pathtracer [options]\n\n"
              << "Options:\n"
              << "  --help, -h              Show this help text\n"
              << "  --scene NAME             Select a scene preset\n"
              << "                           NAME: sponza, cornell, hurricane, prism, crystal, deer, demo\n"
              << "  --scene-file FILE        Load a scene manifest JSON file\n"
              << "  --list-scenes            List available scene presets\n"
              << "  --caustics               Enable photon-mapped caustics (Cornell/Hurricane/Prism/Crystal)\n"
              << "  --caustic-photons N      Set the photon count (default: 1048576)\n"
              << "  --caustic-radius R       Set the gather radius (default: 0.06)\n"
              << "  --no-caustics            Disable caustic photon tracing\n"
              << "  --no-caustic-gather      Trace photons but omit camera gathering\n"
              << "  --persistent-wavefront  Use experimental device-side persistent wavefront\n"
              << "  --profile-queues        Print persistent queue occupancy by bounce\n"
              << "  --resume                 Resume from the selected checkpoint\n"
              << "  --coordinator            Run as a distributed coordinator\n"
              << "  --listen-host HOST       Coordinator bind host (default: 0.0.0.0)\n"
              << "  --listen-port PORT       Coordinator bind port (default: 9000)\n"
              << "  --worker                 Run as a distributed worker\n"
              << "  --worker-id ID           Unique worker identifier\n"
              << "  --coordinator-host HOST  Worker coordinator host (default: 127.0.0.1)\n"
              << "  --coordinator-port PORT  Worker coordinator port (default: 9000)\n"
              << "  --distributed-checkpoint FILE\n"
              << "                           Distributed checkpoint path\n"
              << "  --distributed-checkpoint-seconds N\n"
              << "                           Coordinator checkpoint interval (default: 30)\n"
              << "  --samples-per-task N     Distributed sample batch size (default: 16)\n"
              << "  --distributed-lease-seconds N\n"
              << "                           Worker lease duration (default: 30)\n"
              << "  --push-assets            Transfer scene assets to distributed workers\n"
              << "  --asset-cache DIR        Worker asset cache directory\n";
}

static bool parsePort(const char* value, uint16_t& port) {
    if (value == nullptr || value[0] == '\0')
        return false;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (*end != '\0' || parsed == 0 || parsed > 65535)
        return false;
    port = static_cast<uint16_t>(parsed);
    return true;
}

static bool parsePositiveUint32(const char* value, uint32_t& output) {
    if (value == nullptr || value[0] == '\0')
        return false;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(value, &end, 10);
    if (*end != '\0' || parsed == 0 || parsed > 0xfffffffful)
        return false;
    output = static_cast<uint32_t>(parsed);
    return true;
}

SceneConfig scenePreset(ScenePreset preset) {
    SceneConfig result;
    result.preset = preset;
    result.name = nameOf(preset);
    const std::string manifestName = std::string(nameOf(preset)) + ".json";
    const std::string candidates[] = {
        "../assets/scenes/" + manifestName,
        "assets/scenes/" + manifestName,
        "../../assets/scenes/" + manifestName,
        "scenes/" + manifestName};

    std::string selectedManifest;
    for (const std::string& candidate : candidates) {
        std::ifstream input(candidate);
        if (input) {
            selectedManifest = candidate;
            break;
        }
    }
    if (selectedManifest.empty()) {
        std::cerr << "Could not find scene manifest for preset '" << result.name << "'\n";
        std::exit(1);
    }

    std::string error;
    if (!loadSceneManifest(selectedManifest.c_str(), result, &error)) {
        std::cerr << "Could not load scene manifest '" << selectedManifest << "': " << error << '\n';
        std::exit(1);
    }
    return result;
}

bool parseScenePreset(const char* value, ScenePreset& preset) {
    if (!std::strcmp(value, "sponza")) preset = ScenePreset::Sponza;
    else if (!std::strcmp(value, "cornell")) preset = ScenePreset::Cornell;
    else if (!std::strcmp(value, "hurricane")) preset = ScenePreset::Hurricane;
    else if (!std::strcmp(value, "prism")) preset = ScenePreset::Prism;
    else if (!std::strcmp(value, "crystal")) preset = ScenePreset::Crystal;
    else if (!std::strcmp(value, "deer")) preset = ScenePreset::Deer;
    else if (!std::strcmp(value, "demo")) preset = ScenePreset::Demo;
    else return false;
    return true;
}

bool parseCommandLine(int argc, char** argv, RenderConfig& render, SceneConfig& scene) {
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--list-scenes")) {
            std::cout << "Scene presets: sponza, cornell, hurricane, prism, crystal, deer, demo\n";
            return false;
        }
        if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
            printHelp();
            return false;
        }
        if (!std::strcmp(argv[i], "--no-caustics")) { render.enableCaustics = false; continue; }
        if (!std::strcmp(argv[i], "--caustics")) { render.enableCaustics = true; continue; }
        if (!std::strcmp(argv[i], "--caustic-photons") && i + 1 < argc) {
            render.causticPhotonCount = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
            if (render.causticPhotonCount == 0) { std::cerr << "Photon count must be greater than zero\n"; return false; }
            continue;
        }
        if (!std::strcmp(argv[i], "--caustic-radius") && i + 1 < argc) {
            render.causticGatherRadius = std::strtof(argv[++i], nullptr);
            if (render.causticGatherRadius <= 0.0f) { std::cerr << "Gather radius must be greater than zero\n"; return false; }
            continue;
        }
        if (!std::strcmp(argv[i], "--no-caustic-gather")) { render.enableCausticGather = false; continue; }
        if (!std::strcmp(argv[i], "--persistent-wavefront")) { render.persistentWavefront = true; continue; }
        if (!std::strcmp(argv[i], "--profile-queues")) { render.profileQueues = true; continue; }
        if (!std::strcmp(argv[i], "--resume")) { render.resumeFromCheckpoint = true; continue; }
        if (!std::strcmp(argv[i], "--coordinator")) {
            if (render.distributedRole == DistributedRole::Worker) { std::cerr << "Cannot combine --coordinator and --worker\n"; return false; }
            render.distributedRole = DistributedRole::Coordinator;
            continue;
        }
        if (!std::strcmp(argv[i], "--worker")) {
            if (render.distributedRole == DistributedRole::Coordinator) { std::cerr << "Cannot combine --coordinator and --worker\n"; return false; }
            render.distributedRole = DistributedRole::Worker;
            continue;
        }
        if (!std::strcmp(argv[i], "--listen-host") && i + 1 < argc) { render.listenHost = argv[++i]; continue; }
        if (!std::strcmp(argv[i], "--listen-port") && i + 1 < argc) {
            if (!parsePort(argv[++i], render.listenPort)) { std::cerr << "Invalid listen port\n"; return false; }
            continue;
        }
        if (!std::strcmp(argv[i], "--coordinator-host") && i + 1 < argc) { render.coordinatorHost = argv[++i]; continue; }
        if (!std::strcmp(argv[i], "--coordinator-port") && i + 1 < argc) {
            if (!parsePort(argv[++i], render.coordinatorPort)) { std::cerr << "Invalid coordinator port\n"; return false; }
            continue;
        }
        if (!std::strcmp(argv[i], "--worker-id") && i + 1 < argc) { render.workerId = argv[++i]; continue; }
        if (!std::strcmp(argv[i], "--distributed-checkpoint") && i + 1 < argc) {
            render.distributedCheckpointFilename = argv[++i];
            continue;
        }
        if (!std::strcmp(argv[i], "--distributed-checkpoint-seconds") && i + 1 < argc) {
            if (!parsePositiveUint32(argv[++i], render.distributedCheckpointIntervalSeconds)) {
                std::cerr << "Distributed checkpoint interval must be greater than zero\n";
                return false;
            }
            continue;
        }
        if (!std::strcmp(argv[i], "--samples-per-task") && i + 1 < argc) {
            if (!parsePositiveUint32(argv[++i], render.distributedSamplesPerTask)) {
                std::cerr << "Samples per task must be greater than zero\n";
                return false;
            }
            continue;
        }
        if (!std::strcmp(argv[i], "--distributed-lease-seconds") && i + 1 < argc) {
            uint32_t leaseSeconds = 0;
            if (!parsePositiveUint32(argv[++i], leaseSeconds)) {
                std::cerr << "Distributed lease duration must be greater than zero\n";
                return false;
            }
            render.distributedLeaseDurationMs = static_cast<uint64_t>(leaseSeconds) * 1000ull;
            continue;
        }
        if (!std::strcmp(argv[i], "--push-assets")) { render.pushAssets = true; continue; }
        if (!std::strcmp(argv[i], "--asset-cache") && i + 1 < argc) {
            render.assetCacheDirectory = argv[++i];
            continue;
        }
        if (!std::strcmp(argv[i], "--scene") && i + 1 < argc) {
            ScenePreset preset;
            if (!parseScenePreset(argv[++i], preset)) { std::cerr << "Unknown scene preset: " << argv[i] << '\n'; return false; }
            scene = scenePreset(preset);
        } else if (!std::strcmp(argv[i], "--scene-file") && i + 1 < argc) {
            const char* filename = argv[++i];
            std::string error;
            if (!loadSceneManifest(filename, scene, &error)) {
                std::cerr << "Could not load scene manifest '" << filename << "': " << error << '\n';
                return false;
            }
        } else { std::cerr << "Unknown option: " << argv[i] << "\n"; printHelp(); return false; }
    }
    if (render.distributedRole != DistributedRole::Local) {
        render.tiledRendering = true;
        if (render.distributedRole == DistributedRole::Worker && (render.workerId == nullptr || render.workerId[0] == '\0')) {
            std::cerr << "--worker requires --worker-id\n";
            return false;
        }
    }
    if (render.intersectionDebug || render.shadingNormalDebug) render.samplesPerPixel = 1;
    return true;
}
