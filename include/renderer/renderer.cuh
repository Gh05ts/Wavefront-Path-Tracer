#pragma once

#include <cstdint>

#include "../scene/camera.cuh"
#include "../scene/scene.cuh"
#include "path_state.cuh"
#include "queues.cuh"
#include "tile_scheduler.cuh"

struct TraceResult {
    Hit hit;
    bool didHit;
};

__device__ TraceResult traceRay(const Ray& ray, Scene scene);
__device__ TraceResult traceRayExternal(const Ray& ray, Scene scene);
__device__ Vec3 sampleCosineHemisphere(const Vec3& normal, uint32_t& rngState);
__device__ Vec3 reflect(const Vec3& incident, const Vec3& normal);
__device__ Vec3 refract(const Vec3& incident, const Vec3& normal, float eta);
__device__ float schlickReflectance(float cosine, float refractionRatio);
__device__ Vec3 gatherCaustics(const Hit& hit, PhotonGrid grid, float radius);

__global__
void emitPhotons(PhotonQueue photons, Scene scene, uint32_t photonCount, uint32_t seed, bool spectralSampling);

__global__
void tracePhotons(PhotonQueue photons, Scene scene, uint32_t photonCount, uint32_t maxDepth, uint32_t* materialHitCounts);

__global__
void buildPhotonGrid(PhotonQueue photons, PhotonGrid grid, uint32_t photonCount);

__global__
void generatePrimaryRays(
    RayQueue queue,
    PathState* pathStates,
    Camera camera,
    uint32_t width,
    uint32_t height,
    const uint32_t* sampleIndex,
    const RenderTile* tile
);

__global__
void advanceSampleIndex(uint32_t* sampleIndex, const RenderTile* tile);

__global__
void intersectScene(
    RayQueue rays,
    IntersectionResult* results,
    Scene scene
);

__global__
void shadePaths(
    RayQueue rays,
    const IntersectionResult* results,
    RayQueue nextRays,
    PathState* pathStates,
    Scene scene,
    uint32_t maxDepth,
    uint32_t russianRouletteStartDepth,
    bool intersectionDebug,
    bool shadingNormalDebug,
    Vec3* framebuffer,
    PhotonGrid photonGrid,
    float photonGatherRadius
);

__global__
void persistentWavefrontTrace(
    RayQueue initialRays,
    RayQueue secondaryRays,
    PathState* pathStates,
    uint32_t* queueCounts,
    Camera camera,
    uint32_t width,
    uint32_t height,
    const uint32_t* sampleIndex,
    const RenderTile* tile,
    Scene scene,
    uint32_t maxDepth,
    uint32_t russianRouletteStartDepth,
    bool intersectionDebug,
    bool shadingNormalDebug,
    Vec3* framebuffer,
    PhotonGrid photonGrid,
    float photonGatherRadius
);
