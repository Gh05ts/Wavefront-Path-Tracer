#pragma once

#include "../scene/camera.cuh"
#include "tile_scheduler.cuh"

struct Scene;

// Guides use a separate, unjittered primary ray through each pixel center.
// Surface IDs and object motion are deliberately absent: v1 captures static scenes.
struct FrameGuide {
    Vec3 albedo{};
    Vec3 normal{};
    Vec3 geometricNormal{};
    float depth = 0.0f;
    Vec2 motion{0.0f, 0.0f};
    float valid = 0.0f;
    float historyValid = 0.0f;
};

// Pixel coordinates have integer centers; motion is previous minus current.
// A false result means behind the camera or outside the previous image.
__host__ __device__ inline bool projectFramePoint(
    const Camera& camera, const Vec3& point, uint32_t width, uint32_t height,
    Vec2& pixel) {
    Vec3 relative = point - camera.position;
    float depth = dot(relative, camera.forward);
    if (!(depth > 0.0f))
        return false;
    float halfHeight = depth * tanf(camera.verticalFov * 0.5f);
    float u = 0.5f * (dot(relative, camera.right) / (halfHeight * camera.aspectRatio) + 1.0f);
    float v = 0.5f * (1.0f - dot(relative, camera.up) / halfHeight);
    pixel = Vec2{u * width - 0.5f, v * height - 0.5f};
    // Bilinear reuse requires a coordinate inside the previous center domain.
    constexpr float epsilon = 0.0001f;
    bool inside = pixel.x >= -epsilon && pixel.y >= -epsilon &&
        pixel.x <= static_cast<float>(width - 1) + epsilon &&
        pixel.y <= static_cast<float>(height - 1) + epsilon;
    if (inside) {
        pixel.x = fminf(fmaxf(pixel.x, 0.0f), static_cast<float>(width - 1));
        pixel.y = fminf(fmaxf(pixel.y, 0.0f), static_cast<float>(height - 1));
    }
    return inside;
}

// Output is tile-local. No path queues or radiance accumulation are modified.
__global__ void generateFrameGuides(
    FrameGuide* output, Scene scene, Camera camera, Camera previousCamera,
    bool hasPreviousCamera, uint32_t width, uint32_t height, RenderTile tile);
