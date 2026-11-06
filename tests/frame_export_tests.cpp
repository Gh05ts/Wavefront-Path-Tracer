#include "renderer/frame_export.hpp"

#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfInputFile.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <chrono>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

CapturedFrame fixture() {
    CapturedFrame frame;
    frame.width = 3;
    frame.height = 2;
    frame.samples = 4;
    frame.referenceSamples = 16;
    frame.radiance.assign(6, Vec3(0.125f, 4.0f, 32.0f));
    frame.reference.assign(6, Vec3(0.25f, 5.0f, 33.0f));
    frame.guides.resize(6);
    for (FrameGuide& guide : frame.guides) {
        guide.albedo = Vec3(0.2f, 0.5f, 0.8f);
        guide.normal = guide.geometricNormal = Vec3(0.0f, 0.0f, -1.0f);
        guide.depth = 2.0f;
        guide.valid = 1.0f;
    }
    // An asymmetric last pixel catches transposition and row/stride mistakes.
    frame.radiance.back() = Vec3(1.0f, 2.0f, 3.0f);
    return frame;
}

Camera testCamera() {
    return Camera{Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(-1, 0, 0), Vec3(0, 1, 0),
        3.14159265359f / 2.0f, 1.5f};
}

nlohmann::json metadata() {
    return {{"sequence_id", "fixture"}, {"frame_index", 0}, {"previous_frame_index", nullptr},
        {"scene", "fixture"}, {"camera", frameCameraMetadata(testCamera())}, {"previous_camera", nullptr},
        {"render_fingerprint", "fixture"},
        {"render", {{"rng", "pcg32"}, {"rng_key", "pixel_sample_v1"}, {"sample_start", 0}, {"samples", 4},
            {"reference_sample_start", 4}, {"reference_samples", 16}, {"reference_estimator", "matched_renderer_settings"}}}};
}

void projectionTests() {
    Camera camera = testCamera();
    camera.aspectRatio = 8.0f / 4.0f;
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 8; ++x) {
            Ray ray = camera.generateRay((x + 0.5f) / 8, (y + 0.5f) / 4);
            Vec2 pixel{};
            bool inside = projectFramePoint(camera, ray.at(5.0f), 8, 4, pixel);
            require(inside, "Primary point did not project into image");
            require(std::fabs(pixel.x - x) < 1e-5f && std::fabs(pixel.y - y) < 1e-5f,
                "Projection does not invert ray generation at integer pixel centers");
        }
    }
    Vec2 pixel{};
    require(!projectFramePoint(camera, Vec3(0, 0, -1), 8, 4, pixel), "Behind-camera projection was accepted");
    require(!projectFramePoint(camera, Vec3(100, 0, 1), 8, 4, pixel), "Offscreen projection was accepted");
    Camera previous = camera;
    previous.position = Vec3(0.5f, 0, 0);
    require(projectFramePoint(previous, Vec3(0, 0, 2), 8, 4, pixel), "Translated-camera projection failed");
    require(pixel.x > 3.5f, "Backward motion has the wrong sign for the camera basis");
}

void exportTests(const std::filesystem::path& directory) {
    projectionTests();
    CapturedFrame frame = fixture();
    writeCapturedFrame(directory, frame, metadata());
    nlohmann::json manifest;
    std::ifstream(directory / "manifest.json") >> manifest;
    require(manifest.at("version") == 1 && manifest.at("image_sha256").get<std::string>().size() == 64,
        "Missing version or image digest");
    namespace Exr = OPENEXR_IMF_NAMESPACE;
    Exr::InputFile input((directory / "frame.exr").c_str());
    require(input.header().channels().findChannel("radiance.G")->type == Exr::FLOAT,
        "HDR radiance was not stored at float32 precision");
    require(input.header().channels().findChannel("sample_count.N")->type == Exr::UINT,
        "Sample counts were not stored as unsigned integers");
    std::vector<float> green(6);
    std::vector<uint32_t> counts(6);
    Exr::FrameBuffer buffer;
    buffer.insert("radiance.G", Exr::Slice(Exr::FLOAT, reinterpret_cast<char*>(green.data()), sizeof(float), 3 * sizeof(float)));
    buffer.insert("sample_count.N", Exr::Slice(Exr::UINT, reinterpret_cast<char*>(counts.data()), sizeof(uint32_t), 3 * sizeof(uint32_t)));
    input.setFrameBuffer(buffer);
    input.readPixels(0, 1);
    require(green[0] == 4.0f && green[5] == 2.0f, "HDR values or image layout changed during export");
    require(counts[0] == 4 && counts[5] == 4, "Sample count round-trip failed");
    bool rejected = false;
    try { writeCapturedFrame(directory, frame, metadata()); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Existing captures must never be overwritten");
    frame.radiance[0].x = std::numeric_limits<float>::quiet_NaN();
    auto bad = directory.string() + "-invalid";
    rejected = false;
    try { writeCapturedFrame(bad, frame, metadata()); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && !std::filesystem::exists(bad), "Non-finite capture was published");
}
} // namespace

int main(int argc, char** argv) {
    bool keep = argc == 2;
    std::filesystem::path directory = keep ? std::filesystem::path(argv[1]) :
        std::filesystem::temp_directory_path() / ("frame-export-test-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        exportTests(directory);
        if (!keep) std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "frame_export_tests: " << error.what() << '\n';
        if (!keep) std::filesystem::remove_all(directory);
        return 1;
    }
}
