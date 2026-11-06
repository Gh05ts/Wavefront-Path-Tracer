#pragma once

// Included by renderer.cu after traversal and material lookup helpers.
#include "frame_guides.cuh"

__global__ void generateFrameGuides(
    FrameGuide* output, Scene scene, Camera camera, Camera previousCamera,
    bool hasPreviousCamera, uint32_t width, uint32_t height, RenderTile tile) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tile.width * tile.height)
        return;
    uint32_t x = tile.x + index % tile.width;
    uint32_t y = tile.y + index / tile.width;
    Ray ray = camera.generateRay((x + 0.5f) / width, (y + 0.5f) / height);
    TraceResult result = traceRay(ray, scene);
    FrameGuide guide{};
    if (result.didHit) {
        const Hit& hit = result.hit;
        const Material& material = scene.materials[hit.material];
        guide.albedo = getMaterialAlbedo(material, hit, scene);
        guide.normal = getMaterialNormal(material, hit, scene);
        guide.geometricNormal = hit.geometricNormal;
        if (dot(guide.normal, guide.geometricNormal) < 0.0f)
            guide.normal = -guide.normal;
        guide.depth = dot(hit.position - camera.position, camera.forward);
        guide.valid = 1.0f;
        Vec2 previousPixel{};
        if (hasPreviousCamera && projectFramePoint(previousCamera, hit.position, width, height, previousPixel)) {
            guide.motion = Vec2{previousPixel.x - x, previousPixel.y - y};
            // Camera-only motion: check that this surface was visible from the
            // previous camera, using a continuous ray through its projection.
            Ray previousRay = previousCamera.generateRay(
                (previousPixel.x + 0.5f) / width, (previousPixel.y + 0.5f) / height);
            TraceResult previousHit = traceRay(previousRay, scene);
            float tolerance = fmaxf(0.001f, length(hit.position - previousCamera.position) * 0.0001f);
            if (previousHit.didHit && lengthSquared(previousHit.hit.position - hit.position) <= tolerance * tolerance)
                guide.historyValid = 1.0f;
        }
    }
    output[index] = guide;
}
