#include "renderer/frame_export.hpp"
#include "renderer/capture_lighting.hpp"
#include "renderer/render_driver.cuh"
#include "renderer/render_fingerprint.hpp"
#include "renderer/render_task.cuh"
#include "scene/scene_factory.hpp"
#include "scene/scene_manifest.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace {
using Json = nlohmann::json;

void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { cudaCheck(cudaMalloc(&data, sizeof(T) * count)); }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct SceneOwner {
    DeviceScene scene{};
    bool ready = false;
    ~SceneOwner() { if (ready) destroyDeviceScene(scene); }
};

uint32_t integer(const Json& root, const char* key, uint32_t fallback, bool allowZero = false) {
    if (!root.contains(key))
        return fallback;
    if (!root.at(key).is_number_integer())
        throw std::invalid_argument(std::string(key) + " must be an integer");
    uint64_t value = root.at(key).get<uint64_t>();
    if ((!allowZero && value == 0) || value > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument(std::string(key) + " is outside the supported integer range");
    return static_cast<uint32_t>(value);
}

Vec3 vector(const Json& value) {
    if (!value.is_array() || value.size() != 3)
        throw std::invalid_argument("Camera vectors must have three components");
    for (const auto& component : value)
        if (!component.is_number() || !std::isfinite(component.get<float>()))
            throw std::invalid_argument("Camera vectors must contain finite numbers");
    return Vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
}

Camera cameraForFrame(const Json& frame, Camera camera) {
    if (!frame.is_object())
        throw std::invalid_argument("Each frame must be a camera object");
    if (frame.contains("position"))
        camera.position = vector(frame.at("position"));
    Vec3 forward = frame.contains("target") ? vector(frame.at("target")) - camera.position : camera.forward;
    Vec3 up = frame.contains("up") ? vector(frame.at("up")) : Vec3(0.0f, 1.0f, 0.0f);
    if (!(lengthSquared(forward) > 1e-12f) || !std::isfinite(lengthSquared(forward)))
        throw std::invalid_argument("Camera target must differ from position");
    camera.forward = normalize(forward);
    Vec3 right = cross(camera.forward, up);
    if (!(lengthSquared(right) > 1e-12f) || !std::isfinite(lengthSquared(right)))
        throw std::invalid_argument("Camera up must not be parallel to its direction");
    camera.right = normalize(right);
    camera.up = cross(camera.right, camera.forward);
    if (frame.contains("vertical_fov_degrees")) {
        if (!frame.at("vertical_fov_degrees").is_number())
            throw std::invalid_argument("Camera vertical FOV must be numeric");
        float degrees = frame.at("vertical_fov_degrees").get<float>();
        if (!std::isfinite(degrees) || degrees <= 0.0f || degrees >= 179.0f)
            throw std::invalid_argument("Camera vertical FOV must be between 0 and 179 degrees");
        camera.verticalFov = degrees * 3.14159265359f / 180.0f;
    }
    return camera;
}

std::vector<Vec3> renderRange(RenderTaskRenderer& renderer, const std::vector<RenderTile>& tiles,
    uint64_t fingerprint, uint32_t width, uint32_t height, uint32_t start, uint32_t count) {
    std::vector<Vec3> pixels(static_cast<size_t>(width) * height);
    for (size_t ordinal = 0; ordinal < tiles.size(); ++ordinal) {
        const RenderTile& tile = tiles[ordinal];
        RenderTask task{fingerprint, {static_cast<uint32_t>(ordinal), start, count},
            {tile.x, tile.y, tile.width, tile.height}};
        RenderTaskResult result = renderer.render(task);
        if (!result.success)
            throw std::runtime_error("Tile rendering failed");
        for (uint32_t y = 0; y < tile.height; ++y)
            for (uint32_t x = 0; x < tile.width; ++x)
                pixels[static_cast<size_t>(tile.y + y) * width + tile.x + x] =
                    result.radiance[static_cast<size_t>(y) * tile.width + x] / static_cast<float>(count);
    }
    return pixels;
}

std::vector<FrameGuide> captureGuides(Scene scene, Camera camera, Camera previousCamera,
    bool hasPreviousCamera, const RenderConfig& config, const std::vector<RenderTile>& tiles) {
    size_t capacity = static_cast<size_t>(config.tileWidth) * config.tileHeight;
    DeviceBuffer<FrameGuide> device(capacity);
    std::vector<FrameGuide> tileGuides(capacity);
    std::vector<FrameGuide> guides(static_cast<size_t>(config.width) * config.height);
    for (const RenderTile& tile : tiles) {
        uint32_t count = tile.width * tile.height;
        generateFrameGuides<<<(count + 255) / 256, 256>>>(device.data, scene, camera, previousCamera,
            hasPreviousCamera, config.width, config.height, tile);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(tileGuides.data(), device.data, sizeof(FrameGuide) * count, cudaMemcpyDeviceToHost));
        for (uint32_t y = 0; y < tile.height; ++y)
            std::copy_n(tileGuides.data() + static_cast<size_t>(y) * tile.width, tile.width,
                guides.data() + static_cast<size_t>(tile.y + y) * config.width + tile.x);
    }
    return guides;
}

int capture(const std::filesystem::path& configFile, const std::filesystem::path& output) {
    std::ifstream input(configFile);
    if (!input)
        throw std::runtime_error("Cannot open capture configuration");
    Json root;
    input >> root;
    if (!root.is_object() || !root.contains("frames") || !root.at("frames").is_array() || root.at("frames").empty())
        throw std::invalid_argument("Configuration must contain a nonempty frames array");
    std::string sequence = root.at("sequence_id").get<std::string>();
    if (sequence.empty())
        throw std::invalid_argument("sequence_id must not be empty");
    RenderConfig config;
    config.width = integer(root, "width", 640);
    config.height = integer(root, "height", 360);
    config.tileWidth = std::min(config.width, integer(root, "tile_width", 256));
    config.tileHeight = std::min(config.height, integer(root, "tile_height", 256));
    config.maxDepth = integer(root, "max_depth", 16);
    config.russianRouletteStartDepth = integer(root, "russian_roulette_start_depth", 5);
    config.persistentWavefront = root.value("persistent_wavefront", false);
    config.tiledRendering = true;
    config.rngStrategy = RngStrategy::Pcg32;
    config.enableCaustics = root.value("caustics", false);
    config.causticPhotonCount = integer(root, "caustic_photons", config.causticPhotonCount);
    config.causticGatherRadius = root.value("caustic_radius", config.causticGatherRadius);
    if (!std::isfinite(config.causticGatherRadius) || config.causticGatherRadius <= 0.0f)
        throw std::invalid_argument("caustic_radius must be finite and positive");
    uint32_t samples = integer(root, "samples", 4);
    uint32_t referenceSamples = integer(root, "reference_samples", 256);
    uint32_t sampleStart = integer(root, "sample_start", 0, true);
    uint64_t stride = static_cast<uint64_t>(samples) + referenceSamples;
    uint64_t frameCount = root.at("frames").size();
    if (frameCount > (std::numeric_limits<uint32_t>::max() - sampleStart) / stride ||
        config.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        config.height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        static_cast<uint64_t>(config.width) * config.height > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Capture exceeds the pixel or sample index range");
    config.samplesPerPixel = static_cast<uint32_t>(sampleStart + frameCount * stride);

    SceneConfig sceneConfig;
    if (root.contains("scene_file")) {
        std::filesystem::path filename = root.at("scene_file").get<std::string>();
        if (filename.is_relative()) filename = configFile.parent_path() / filename;
        std::string error;
        if (!loadSceneManifest(filename.c_str(), sceneConfig, &error))
            throw std::invalid_argument(error);
    } else {
        std::string name = root.value("scene", "demo");
        ScenePreset preset;
        if (!parseScenePreset(name.c_str(), preset))
            throw std::invalid_argument("Unknown scene preset: " + name);
        sceneConfig = scenePreset(preset);
    }
    if (config.enableCaustics && sceneConfig.preset != ScenePreset::Cornell &&
        sceneConfig.preset != ScenePreset::Hurricane && sceneConfig.preset != ScenePreset::Prism &&
        sceneConfig.preset != ScenePreset::Crystal)
        throw std::invalid_argument("This scene preset does not support photon-mapped caustics");
    std::optional<CaptureLighting> lighting;
    if (root.contains("capture_lighting")) {
        if (sceneConfig.preset != ScenePreset::Demo)
            throw std::invalid_argument("capture_lighting currently supports only the demo preset");
        lighting = parseCaptureLighting(root.at("capture_lighting"));
    }
    std::vector<Camera> cameras;
    std::vector<std::filesystem::path> directories;
    Camera defaultCamera = createSceneCamera(sceneConfig, config.width, config.height);
    for (size_t i = 0; i < frameCount; ++i) {
        cameras.push_back(cameraForFrame(root.at("frames").at(i), defaultCamera));
        std::ostringstream name;
        name << "frame_" << std::setw(6) << std::setfill('0') << i;
        directories.push_back(output / name.str());
        if (std::filesystem::exists(directories.back()) || std::filesystem::exists(directories.back().string() + ".partial"))
            throw std::invalid_argument("Capture output already exists: " + directories.back().string());
    }
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
        throw std::runtime_error("A CUDA device is required for dataset capture");
    SceneOwner scene;
    if (lighting) {
        HostScene host = assembleDemoScene();
        applyCaptureLighting(host, *lighting);
        scene.scene = uploadScene(host);
    } else {
        scene.scene = createSceneFromConfig(sceneConfig);
    }
    scene.ready = true;
    applySceneConfig(scene.scene, sceneConfig);
    // The task key must match RenderDriver's prepared key. Capture metadata
    // additionally identifies the lighting intervention applied to the scene.
    const uint64_t taskFingerprint = computeRenderFingerprint(config, sceneConfig);
    uint64_t fingerprint = taskFingerprint;
    uint64_t sceneIdentity = computeSceneFingerprint(sceneConfig);
    if (lighting) {
        fingerprint = captureLightingFingerprint(fingerprint, *lighting);
        sceneIdentity = captureLightingFingerprint(sceneIdentity, *lighting);
    }
    auto tiles = createRenderTiles(config.width, config.height, config.tileWidth, config.tileHeight);
    for (size_t i = 0; i < cameras.size(); ++i) {
        uint32_t start = static_cast<uint32_t>(sampleStart + i * stride);
        std::cout << "Capturing frame " << i << ": " << samples << " input + " << referenceSamples << " reference SPP\n";
        RenderDriver driver(scene.scene, cameras[i], config, sceneConfig);
        RenderTaskRenderer renderer(driver);
        CapturedFrame frame;
        frame.width = config.width;
        frame.height = config.height;
        frame.samples = samples;
        frame.referenceSamples = referenceSamples;
        frame.radiance = renderRange(renderer, tiles, taskFingerprint, config.width, config.height, start, samples);
        frame.reference = renderRange(renderer, tiles, taskFingerprint, config.width, config.height, start + samples, referenceSamples);
        frame.guides = captureGuides(scene.scene.scene, cameras[i], cameras[i == 0 ? 0 : i - 1], i > 0, config, tiles);
        std::ostringstream fingerprintText;
        fingerprintText << std::hex << fingerprint;
        std::ostringstream sceneFingerprintText;
        sceneFingerprintText << std::hex << sceneIdentity;
        Json metadata = {
            {"sequence_id", sequence}, {"frame_index", i},
            {"previous_frame_index", i == 0 ? Json(nullptr) : Json(i - 1)},
            {"camera", frameCameraMetadata(cameras[i])},
            {"previous_camera", i == 0 ? Json(nullptr) : frameCameraMetadata(cameras[i - 1])},
            {"scene", sceneConfig.name}, {"render_fingerprint", fingerprintText.str()},
            {"scene_fingerprint", sceneFingerprintText.str()},
            {"render", {{"rng", "pcg32"}, {"rng_key", "pixel_sample_v1"},
                {"sample_start", start}, {"samples", samples},
                {"reference_sample_start", start + samples}, {"reference_samples", referenceSamples},
                {"max_depth", config.maxDepth}, {"russian_roulette_start_depth", config.russianRouletteStartDepth},
                {"persistent_wavefront", config.persistentWavefront},
                {"tile_width", config.tileWidth}, {"tile_height", config.tileHeight},
                {"caustics", config.enableCaustics}, {"caustic_photons", config.causticPhotonCount},
                {"caustic_radius", config.causticGatherRadius}, {"caustic_seed", 0x13579bdfu},
                {"reference_estimator", "matched_renderer_settings"}}}};
        if (lighting) {
            std::ostringstream geometryIdentity;
            geometryIdentity << std::hex << computeSceneFingerprint(sceneConfig);
            metadata["geometry_fingerprint"] = geometryIdentity.str();
            metadata["capture_lighting"] = lighting->metadata();
        }
        writeCapturedFrame(directories[i], frame, std::move(metadata));
        std::cout << "Wrote " << directories[i] << '\n';
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3 || std::string(argv[1]) == "--help") {
        std::cout << "Usage: dataset_capture CONFIG.json OUTPUT_DIRECTORY\n"
                  << "Run from the renderer root or build directory. Output directories must be new.\n";
        return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 1;
    }
    try {
        return capture(std::filesystem::absolute(argv[1]), std::filesystem::absolute(argv[2]));
    } catch (const std::exception& exception) {
        std::cerr << "dataset_capture: " << exception.what() << '\n';
        return 1;
    }
}
