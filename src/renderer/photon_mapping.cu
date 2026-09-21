#include "renderer/renderer.cuh"
#include "core/rng.cuh"

__global__
void emitPhotons(PhotonQueue photons, Scene scene, uint32_t photonCount, uint32_t seed, bool spectralSampling) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= photonCount || scene.lightCount == 0)
        return;
    uint32_t rngState = makeRngSeed(seed ^ (index * 0x9e3779b9u));
    uint32_t candidate = min(static_cast<uint32_t>(randomFloat(rngState) * scene.lightCount), scene.lightCount - 1);
    uint32_t lightIndex = randomFloat(rngState) < scene.lightAlias[candidate].probability ? candidate : scene.lightAlias[candidate].alias;
    const TriangleLight& light = scene.lights[lightIndex];
    float rootU = sqrtf(randomFloat(rngState));
    float v = randomFloat(rngState);
    Vec3 point = (1.0f - rootU) * light.v0 + rootU * ((1.0f - v) * light.v1 + v * light.v2);
    Vec3 direction = sampleCosineHemisphere(light.normal, rngState);
    const Material& material = scene.materials[light.material];
    uint32_t spectralChannel = 3;
    Vec3 flux = material.emission * (light.area * 3.14159265359f / static_cast<float>(photonCount));
    if (spectralSampling) {
        spectralChannel = min(static_cast<uint32_t>(randomFloat(rngState) * 3.0f), 2u);
        flux = Vec3(0.0f, 0.0f, 0.0f);
        float channelFlux = (spectralChannel == 0 ? material.emission.x : spectralChannel == 1 ? material.emission.y : material.emission.z) *
            (light.area * 3.14159265359f * 3.0f / static_cast<float>(photonCount));
        if (spectralChannel == 0) flux.x = channelFlux;
        else if (spectralChannel == 1) flux.y = channelFlux;
        else flux.z = channelFlux;
    }
    photons.items[index] = Photon{point + light.normal * 0.001f, direction, flux, rngState, 0, spectralChannel, false};
}

__global__
void tracePhotons(PhotonQueue photons, Scene scene, uint32_t photonCount, uint32_t maxDepth, uint32_t* materialHitCounts) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= photonCount) return;
    Photon photon = photons.items[index];
    Ray ray(photon.position, photon.direction);
    for (uint32_t depth = 0; depth < maxDepth; ++depth) {
        TraceResult result = traceRayExternal(ray, scene);
        if (!result.didHit) return;
        const Hit& hit = result.hit;
        if (hit.material >= scene.materialCount) return;
        const Material& material = scene.materials[hit.material];
        Vec3 normal = dot(hit.normal, -ray.direction) >= 0.0f ? hit.normal : -hit.normal;
        if (material.type == MaterialType::Diffuse) {
            if (photon.specularBounces == 0) return;
            photon.position = hit.position;
            photon.direction = -ray.direction;
            photon.valid = true;
            if (materialHitCounts != nullptr && hit.material < 3) atomicAdd(&materialHitCounts[hit.material], 1);
            photons.items[index] = photon;
            return;
        }
        Vec3 direction;
        if (material.type == MaterialType::Metal) {
            direction = normalize(reflect(normalize(ray.direction), normal));
        } else if (material.type == MaterialType::Dielectric) {
            bool frontFace = dot(ray.direction, normal) < 0.0f;
            Vec3 interfaceNormal = frontFace ? normal : -normal;
            float ior = material.ior;
            if (material.dispersion > 0.0f && photon.spectralChannel < 3) {
                float channelOffset = static_cast<float>(photon.spectralChannel) - 1.0f;
                ior += channelOffset * material.dispersion;
            }
            float refractionRatio = frontFace ? 1.0f / ior : ior;
            float cosTheta = fminf(dot(-ray.direction, interfaceNormal), 1.0f);
            float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
            bool reflected = refractionRatio * sinTheta > 1.0f || schlickReflectance(cosTheta, refractionRatio) > randomFloat(photon.rngState);
            direction = reflected ? reflect(ray.direction, interfaceNormal) : refract(ray.direction, interfaceNormal, refractionRatio);
        } else {
            direction = sampleCosineHemisphere(normal, photon.rngState);
        }
        if (material.type == MaterialType::Metal || material.type == MaterialType::Dielectric) ++photon.specularBounces;
        photon.flux = hadamard(photon.flux, material.albedo);
        ray = Ray(hit.position + (dot(direction, hit.geometricNormal) >= 0.0f ? 1.0f : -1.0f) * hit.geometricNormal * 0.001f, direction);
    }
}

__global__
void buildPhotonGrid(PhotonQueue photons, PhotonGrid grid, uint32_t photonCount) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= photonCount || !photons.items[index].valid) return;
    Vec3 extent = grid.maximum - grid.minimum;
    Vec3 offset = photons.items[index].position - grid.minimum;
    Vec3 normalized(offset.x / extent.x, offset.y / extent.y, offset.z / extent.z);
    int x = min(max(static_cast<int>(normalized.x * grid.resolution), 0), static_cast<int>(grid.resolution) - 1);
    int y = min(max(static_cast<int>(normalized.y * grid.resolution), 0), static_cast<int>(grid.resolution) - 1);
    int z = min(max(static_cast<int>(normalized.z * grid.resolution), 0), static_cast<int>(grid.resolution) - 1);
    if (normalized.x < 0.0f || normalized.y < 0.0f || normalized.z < 0.0f || normalized.x > 1.0f || normalized.y > 1.0f || normalized.z > 1.0f) return;
    uint32_t cell = (static_cast<uint32_t>(z) * grid.resolution + static_cast<uint32_t>(y)) * grid.resolution + static_cast<uint32_t>(x);
    grid.next[index] = atomicExch(&grid.heads[cell], index);
}

__device__ Vec3 gatherCaustics(const Hit& hit, PhotonGrid grid, float radius) {
    Vec3 extent = grid.maximum - grid.minimum;
    Vec3 offset = hit.position - grid.minimum;
    Vec3 normalized(offset.x / extent.x, offset.y / extent.y, offset.z / extent.z);
    int centerX = min(max(static_cast<int>(normalized.x * grid.resolution), 0), static_cast<int>(grid.resolution) - 1);
    int centerY = min(max(static_cast<int>(normalized.y * grid.resolution), 0), static_cast<int>(grid.resolution) - 1);
    int centerZ = min(max(static_cast<int>(normalized.z * grid.resolution), 0), static_cast<int>(grid.resolution) - 1);
    constexpr uint32_t maxNearestPhotons = 64;
    constexpr uint32_t minimumPhotons = 16;
    float searchRadius = fmaxf(radius * 4.0f, 0.1f);
    int cellRadius = max(1, static_cast<int>(ceilf(searchRadius * grid.resolution / fmaxf(extent.x, 0.0001f))));
    float searchRadiusSquared = searchRadius * searchRadius;
    float nearestDistances[maxNearestPhotons];
    uint32_t nearestIndices[maxNearestPhotons];
    uint32_t nearestCount = 0;
    uint32_t farthestIndex = 0;
    float farthestDistanceSquared = 0.0f;
    for (uint32_t i = 0; i < maxNearestPhotons; ++i) {
        nearestDistances[i] = 0.0f;
        nearestIndices[i] = 0xffffffffu;
    }
    for (int z = centerZ - cellRadius; z <= centerZ + cellRadius; ++z)
        for (int y = centerY - cellRadius; y <= centerY + cellRadius; ++y)
            for (int x = centerX - cellRadius; x <= centerX + cellRadius; ++x) {
                if (x < 0 || y < 0 || z < 0 || x >= static_cast<int>(grid.resolution) || y >= static_cast<int>(grid.resolution) || z >= static_cast<int>(grid.resolution)) continue;
                uint32_t cell = (static_cast<uint32_t>(z) * grid.resolution + static_cast<uint32_t>(y)) * grid.resolution + static_cast<uint32_t>(x);
                for (uint32_t photonIndex = grid.heads[cell]; photonIndex != 0xffffffffu; photonIndex = grid.next[photonIndex]) {
                    const Photon& photon = grid.photons[photonIndex];
                    Vec3 photonOffset = photon.position - hit.position;
                    float distanceSquared = dot(photonOffset, photonOffset);
                    float cosine = dot(hit.normal, photon.direction);
                    if (distanceSquared > searchRadiusSquared || cosine <= 0.0f) continue;
                    if (nearestCount < maxNearestPhotons) {
                        nearestDistances[nearestCount] = distanceSquared;
                        nearestIndices[nearestCount] = photonIndex;
                        if (distanceSquared > farthestDistanceSquared) {
                            farthestDistanceSquared = distanceSquared;
                            farthestIndex = nearestCount;
                        }
                        ++nearestCount;
                    } else if (distanceSquared < farthestDistanceSquared) {
                        nearestDistances[farthestIndex] = distanceSquared;
                        nearestIndices[farthestIndex] = photonIndex;
                        farthestIndex = 0;
                        farthestDistanceSquared = nearestDistances[0];
                        for (uint32_t i = 1; i < maxNearestPhotons; ++i) {
                            if (nearestDistances[i] > farthestDistanceSquared) {
                                farthestDistanceSquared = nearestDistances[i];
                                farthestIndex = i;
                            }
                        }
                    }
                }
            }
    if (nearestCount == 0) return Vec3(0.0f, 0.0f, 0.0f);

    float gatherRadiusSquared = searchRadiusSquared;
    if (nearestCount >= minimumPhotons) {
        gatherRadiusSquared = 0.0f;
        for (uint32_t i = 0; i < nearestCount; ++i)
            gatherRadiusSquared = fmaxf(gatherRadiusSquared, nearestDistances[i]);
        gatherRadiusSquared = fmaxf(gatherRadiusSquared, radius * radius);
    }
    float gatherRadius = sqrtf(gatherRadiusSquared);
    Vec3 result(0.0f, 0.0f, 0.0f);
    for (uint32_t i = 0; i < nearestCount; ++i) {
        const Photon& photon = grid.photons[nearestIndices[i]];
        float distance = sqrtf(nearestDistances[i]);
        float coneWeight = fmaxf(0.0f, 1.0f - distance / gatherRadius);
        result += photon.flux * (coneWeight * dot(hit.normal, photon.direction));
    }
    return result * (3.0f / (3.14159265359f * gatherRadiusSquared));
}
