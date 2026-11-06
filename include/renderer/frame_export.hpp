#pragma once

#include <filesystem>
#include <vector>

#include "frame_guides.cuh"
#include "json.hpp"

struct CapturedFrame {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t samples = 0;
    uint32_t referenceSamples = 0;
    std::vector<Vec3> radiance;  // Linear HDR means, never display transformed.
    std::vector<Vec3> reference;
    std::vector<FrameGuide> guides;
};

nlohmann::json frameCameraMetadata(const Camera& camera);
nlohmann::json frameConventions();

// Publishes frame.exr and manifest.json together by renaming a staging directory.
// Existing output directories are rejected; failure never publishes a manifest.
void writeCapturedFrame(const std::filesystem::path& directory,
    const CapturedFrame& frame, nlohmann::json metadata);
