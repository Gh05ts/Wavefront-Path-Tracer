#include "renderer/frame_export.hpp"

#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfHeader.h>
#include <ImfOutputFile.h>
#include <openssl/evp.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace {
using Json = nlohmann::json;

std::string fileDigest(const std::filesystem::path& filename) {
    std::ifstream input(filename, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot read exported image for hashing");
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(EVP_MD_CTX_new(), EVP_MD_CTX_free);
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Could not initialize image digest");
    std::array<char, 65536> buffer{};
    while (input.read(buffer.data(), buffer.size()) || input.gcount() > 0)
        if (EVP_DigestUpdate(context.get(), buffer.data(), static_cast<size_t>(input.gcount())) != 1)
            throw std::runtime_error("Could not update image digest");
    if (!input.eof())
        throw std::runtime_error("Failed while hashing exported image");
    std::array<unsigned char, 32> digest{};
    unsigned int digestSize = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &digestSize) != 1 || digestSize != digest.size())
        throw std::runtime_error("Could not finish image digest");
    std::ostringstream hex;
    for (unsigned char byte : digest)
        hex << std::hex << std::setw(2) << std::setfill('0') << static_cast<unsigned>(byte);
    return hex.str();
}

bool finite(const Vec3& value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void validate(const CapturedFrame& frame) {
    size_t count = static_cast<size_t>(frame.width) * frame.height;
    if (frame.width == 0 || frame.height == 0 ||
        frame.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        frame.height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
        frame.samples == 0 || frame.referenceSamples == 0 ||
        frame.radiance.size() != count || frame.reference.size() != count || frame.guides.size() != count)
        throw std::invalid_argument("Invalid capture dimensions, counts, or buffer sizes");
    for (size_t i = 0; i < count; ++i) {
        const FrameGuide& guide = frame.guides[i];
        if (!finite(frame.radiance[i]) || !finite(frame.reference[i]) ||
            !finite(guide.albedo) || !finite(guide.normal) || !finite(guide.geometricNormal) ||
            !std::isfinite(guide.depth) || guide.depth < 0.0f ||
            !std::isfinite(guide.motion.x) || !std::isfinite(guide.motion.y) ||
            (guide.valid != 0.0f && guide.valid != 1.0f) ||
            (guide.historyValid != 0.0f && guide.historyValid != 1.0f) || guide.historyValid > guide.valid)
            throw std::invalid_argument("Invalid or non-finite capture pixel");
    }
}

void writeExr(const std::filesystem::path& filename, const CapturedFrame& frame) {
    namespace Exr = OPENEXR_IMF_NAMESPACE;
    Exr::Header header(static_cast<int>(frame.width), static_cast<int>(frame.height));
    Exr::FrameBuffer buffer;
    auto channel = [&](const char* name, Exr::PixelType type, const void* base, size_t stride) {
        header.channels().insert(name, Exr::Channel(type));
        buffer.insert(name, Exr::Slice(type, const_cast<char*>(static_cast<const char*>(base)),
            stride, stride * frame.width));
    };
    auto rgb = [&](const char* prefix, const std::vector<Vec3>& pixels) {
        channel((std::string(prefix) + ".R").c_str(), Exr::FLOAT, &pixels[0].x, sizeof(Vec3));
        channel((std::string(prefix) + ".G").c_str(), Exr::FLOAT, &pixels[0].y, sizeof(Vec3));
        channel((std::string(prefix) + ".B").c_str(), Exr::FLOAT, &pixels[0].z, sizeof(Vec3));
    };
    rgb("radiance", frame.radiance);
    rgb("reference", frame.reference);
    const FrameGuide& guide = frame.guides[0];
    channel("albedo.R", Exr::FLOAT, &guide.albedo.x, sizeof(FrameGuide));
    channel("albedo.G", Exr::FLOAT, &guide.albedo.y, sizeof(FrameGuide));
    channel("albedo.B", Exr::FLOAT, &guide.albedo.z, sizeof(FrameGuide));
    channel("normal.X", Exr::FLOAT, &guide.normal.x, sizeof(FrameGuide));
    channel("normal.Y", Exr::FLOAT, &guide.normal.y, sizeof(FrameGuide));
    channel("normal.Z", Exr::FLOAT, &guide.normal.z, sizeof(FrameGuide));
    channel("geometric_normal.X", Exr::FLOAT, &guide.geometricNormal.x, sizeof(FrameGuide));
    channel("geometric_normal.Y", Exr::FLOAT, &guide.geometricNormal.y, sizeof(FrameGuide));
    channel("geometric_normal.Z", Exr::FLOAT, &guide.geometricNormal.z, sizeof(FrameGuide));
    channel("depth.Z", Exr::FLOAT, &guide.depth, sizeof(FrameGuide));
    channel("motion.X", Exr::FLOAT, &guide.motion.x, sizeof(FrameGuide));
    channel("motion.Y", Exr::FLOAT, &guide.motion.y, sizeof(FrameGuide));
    channel("guide_valid.Y", Exr::FLOAT, &guide.valid, sizeof(FrameGuide));
    channel("history_valid.Y", Exr::FLOAT, &guide.historyValid, sizeof(FrameGuide));
    std::vector<uint32_t> counts(frame.radiance.size(), frame.samples);
    std::vector<uint32_t> referenceCounts(frame.radiance.size(), frame.referenceSamples);
    channel("sample_count.N", Exr::UINT, counts.data(), sizeof(uint32_t));
    channel("reference_count.N", Exr::UINT, referenceCounts.data(), sizeof(uint32_t));
    Exr::OutputFile output(filename.c_str(), header);
    output.setFrameBuffer(buffer);
    output.writePixels(static_cast<int>(frame.height));
}
} // namespace

nlohmann::json frameCameraMetadata(const Camera& camera) {
    auto vector = [](const Vec3& value) { return Json::array({value.x, value.y, value.z}); };
    return {{"position", vector(camera.position)}, {"forward", vector(camera.forward)},
        {"right", vector(camera.right)}, {"up", vector(camera.up)},
        {"vertical_fov_radians", camera.verticalFov}, {"aspect_ratio", camera.aspectRatio}};
}

nlohmann::json frameConventions() {
    return {{"radiance", "linear_scene_referred_rgb"}, {"radiance_normalization", "mean"},
        {"exposure", 1.0}, {"image_origin", "top_left"}, {"normal_space", "world"},
        {"depth", "camera_forward_distance"}, {"motion", "previous_pixel_minus_current_pixel"},
        {"motion_units", "pixels"}, {"pixel_coordinates", "integer_centers"},
        {"guides", "unjittered_pixel_centers"},
        {"radiance_sampling", "independent_uniform_subpixel_jitter"},
        {"scene_motion", "static"}, {"history_valid", "previous_frustum_and_visibility"}};
}

void writeCapturedFrame(const std::filesystem::path& directory,
    const CapturedFrame& frame, nlohmann::json metadata) {
    validate(frame);
    if (!metadata.is_object())
        throw std::invalid_argument("Capture metadata must be an object");
    std::filesystem::path staging = directory;
    staging += ".partial";
    if (std::filesystem::exists(directory) || std::filesystem::exists(staging))
        throw std::runtime_error("Capture output or staging directory already exists: " + directory.string());
    if (!directory.parent_path().empty())
        std::filesystem::create_directories(directory.parent_path());
    if (!std::filesystem::create_directory(staging))
        throw std::runtime_error("Could not create capture staging directory");
    try {
        writeExr(staging / "frame.exr", frame);
        metadata["schema"] = "cuda-wavefront-frame";
        metadata["version"] = 1;
        metadata["width"] = frame.width;
        metadata["height"] = frame.height;
        metadata["image"] = "frame.exr";
        metadata["image_sha256"] = fileDigest(staging / "frame.exr");
        metadata["conventions"] = frameConventions();
        std::ofstream manifest(staging / "manifest.json");
        manifest.exceptions(std::ios::failbit | std::ios::badbit);
        manifest << metadata.dump(2) << '\n';
        manifest.close();
        std::filesystem::rename(staging, directory);
    } catch (...) {
        std::filesystem::remove_all(staging);
        throw;
    }
}
