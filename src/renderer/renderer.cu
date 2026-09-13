#include "renderer/renderer.cuh"
#include "core/rng.cuh"

#include <cmath>

__global__
void generatePrimaryRays(RayQueue queue, PathState* pathStates, Camera camera, uint32_t width, uint32_t height, uint32_t sampleIndex) {
    uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pixelCount = width * height;

    if (pixel >= pixelCount)
        return;

    uint32_t x = pixel % width;
    uint32_t y = pixel / width;

    uint32_t rngState = makeRngSeed(pixel ^ (sampleIndex * 0x9e3779b9u));

    float u = (static_cast<float>(x) + randomFloat(rngState)) / static_cast<float>(width);
    float v = (static_cast<float>(y) + randomFloat(rngState)) / static_cast<float>(height);

    Ray ray = camera.generateRay(u, v);

    pathStates[pixel] = PathState{ray, Vec3(1.0f, 1.0f, 1.0f), Vec3(0.0f, 0.0f, 0.0f), pixel, 0, rngState, true};

    uint32_t outputIndex = atomicAdd(queue.count, 1);

    if (outputIndex >= queue.capacity)
        return;

    queue.items[outputIndex] = RayWorkItem{ray, pixel};
}

__device__
Vec3 sampleCosineHemisphere(const Vec3& normal, uint32_t& rngState) {
    constexpr float twoPi = 6.28318530718f;

    float u1 = randomFloat(rngState);
    float u2 = randomFloat(rngState);

    float radius = sqrtf(u1);
    float phi = twoPi * u2;

    float x = radius * cosf(phi);
    float y = radius * sinf(phi);
    float z = sqrtf(1.0f - u1);

    Vec3 helper = fabsf(normal.x) > 0.1f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 tangent = normalize(cross(helper, normal));
    Vec3 bitangent = cross(normal, tangent);

    return normalize(tangent * x + bitangent * y + normal * z);
}

__device__
Vec3 randomInUnitSphere(uint32_t& rngState) {
    while (true) {
        Vec3 point(2.0f * randomFloat(rngState) - 1.0f, 2.0f * randomFloat(rngState) - 1.0f, 2.0f * randomFloat(rngState) - 1.0f);

        if (lengthSquared(point) < 1.0f)
            return point;
    }
}

__device__
Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - 2.0f * dot(incident, normal) * normal;
}

__device__
Vec3 refract(const Vec3& incident, const Vec3& normal, float eta) {
    float cosTheta = fminf(dot(-incident, normal), 1.0f);
    Vec3 perpendicular = eta * (incident + cosTheta * normal);
    Vec3 parallel = -sqrtf(fmaxf(0.0f, 1.0f - lengthSquared(perpendicular))) * normal;
    return perpendicular + parallel;
}

__device__
float schlickReflectance(float cosine, float refractionRatio) {
    float r0 = (1.0f - refractionRatio) / (1.0f + refractionRatio);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

__device__
bool intersectSphere(const Ray& ray, const Sphere& sphere, float& t) {
    Vec3 oc = ray.origin - sphere.center;

    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0.0f)
        return false;

    float sqrtD = sqrtf(discriminant);
    float t0 = (-b - sqrtD) / (2.0f * a);
    float t1 = (-b + sqrtD) / (2.0f * a);

    constexpr float tMin = 0.001f;

    if (t0 > tMin) {
        t = t0;
        return true;
    }

    if (t1 > tMin) {
        t = t1;
        return true;
    }

    return false;
}

__device__
bool intersectTriangle(const Ray& ray, const Triangle& triangle, float& t) {
    constexpr float epsilon = 1e-8f;
    constexpr float tMin = 0.001f;

    Vec3 edge1 = triangle.v1 - triangle.v0;
    Vec3 edge2 = triangle.v2 - triangle.v0;
    Vec3 p = cross(ray.direction, edge2);

    float determinant = dot(edge1, p);

    if (fabsf(determinant) < epsilon)
        return false;

    float inverseDeterminant = 1.0f / determinant;
    Vec3 originToVertex = ray.origin - triangle.v0;

    float u = dot(originToVertex, p) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f)
        return false;

    Vec3 q = cross(originToVertex, edge1);

    float v = dot(ray.direction, q) * inverseDeterminant;
    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = dot(edge2, q) * inverseDeterminant;

    return t > tMin;
}

__global__
void intersectScene(RayQueue rays, IntersectionResult* results, Scene scene) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rayCount = *rays.count;

    if (index >= rayCount)
        return;

    const RayWorkItem& work = rays.items[index];

    float closestT = 1e30f;
    int closestMaterial = -1;
    Vec3 closestNormal;

    for (uint32_t i = 0; i < scene.sphereCount; ++i) {
        float t;

        if (intersectSphere(work.ray, scene.spheres[i], t)) {
            if (t < closestT) {
                closestT = t;
                closestMaterial = static_cast<int>(scene.spheres[i].material);

                Vec3 position = work.ray.at(t);

                closestNormal = normalize(position - scene.spheres[i].center);
            }
        }
    }

    for (uint32_t i = 0; i < scene.triangleCount; ++i) {
        float t;

        if (intersectTriangle(work.ray, scene.triangles[i], t) && t < closestT) {
            closestT = t;
            closestMaterial = static_cast<int>(scene.triangles[i].material);

            Vec3 edge1 = scene.triangles[i].v1 - scene.triangles[i].v0;
            Vec3 edge2 = scene.triangles[i].v2 - scene.triangles[i].v0;
            closestNormal = normalize(cross(edge1, edge2));

            if (dot(closestNormal, work.ray.direction) > 0.0f)
                closestNormal = -closestNormal;
        }
    }

    if (closestMaterial < 0) {
        results[index].didHit = false;
        return;
    }

    Vec3 position = work.ray.at(closestT);

    Hit hit;

    hit.t = closestT;
    hit.position = position;
    hit.normal = closestNormal;
    hit.material = static_cast<uint32_t>(closestMaterial);

    results[index] = IntersectionResult{hit, true};
}

__global__
void shadePaths(RayQueue rays, const IntersectionResult* results, RayQueue nextRays, PathState* pathStates, Scene scene, uint32_t maxDepth, uint32_t russianRouletteStartDepth, Vec3* framebuffer) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rayCount = *rays.count;

    bool active = false;
    RayWorkItem continuation;

    if (index < rayCount) {
        const RayWorkItem& work = rays.items[index];
        PathState& path = pathStates[work.pathIndex];

        if (!results[index].didHit) {
            float t = 0.5f * (path.ray.direction.y + 1.0f);
            Vec3 sky = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
            path.radiance += hadamard(path.throughput, sky);
            path.active = false;
            framebuffer[path.pixelIndex] += path.radiance;
        } else {
            const Hit& hit = results[index].hit;
            const Material& material = scene.materials[hit.material];

            if (material.type == MaterialType::Emissive) {
                path.radiance += hadamard(path.throughput, material.emission);
                path.active = false;
                framebuffer[path.pixelIndex] += path.radiance;
            } else {
                Vec3 direction;

                switch (material.type) {
                case MaterialType::Diffuse:
                    path.throughput = hadamard(path.throughput, material.albedo);
                    direction = sampleCosineHemisphere(hit.normal, path.rngState);
                    break;

                case MaterialType::Metal: {
                    path.throughput = hadamard(path.throughput, material.albedo);
                    Vec3 reflected = reflect(normalize(path.ray.direction), hit.normal);
                    direction = normalize(reflected + material.roughness * randomInUnitSphere(path.rngState));

                    if (dot(direction, hit.normal) <= 0.0f)
                        path.active = false;

                    break;
                }

                case MaterialType::Dielectric: {
                    path.throughput = hadamard(path.throughput, material.albedo);
                    Vec3 incident = normalize(path.ray.direction);
                    bool frontFace = dot(incident, hit.normal) < 0.0f;
                    Vec3 normal = frontFace ? hit.normal : -hit.normal;
                    float refractionRatio = frontFace ? 1.0f / material.ior : material.ior;
                    float cosTheta = fminf(dot(-incident, normal), 1.0f);
                    float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

                    if (refractionRatio * sinTheta > 1.0f || schlickReflectance(cosTheta, refractionRatio) > randomFloat(path.rngState))
                        direction = reflect(incident, normal);
                    else
                        direction = refract(incident, normal, refractionRatio);

                    break;
                }

                case MaterialType::Emissive:
                    break;
                }

                if (path.active) {
                    path.ray.origin = hit.position + direction * 0.001f;
                    path.ray.direction = direction;
                    ++path.depth;

                    if (path.depth >= maxDepth) {
                        path.active = false;
                        framebuffer[path.pixelIndex] += path.radiance;
                    } else if (path.depth >= russianRouletteStartDepth) {
                        float survivalProbability = fminf(0.95f, fmaxf(0.05f, fmaxf(path.throughput.x, fmaxf(path.throughput.y, path.throughput.z))));

                        if (randomFloat(path.rngState) > survivalProbability) {
                            path.active = false;
                            framebuffer[path.pixelIndex] += path.radiance;
                        } else {
                            path.throughput *= 1.0f / survivalProbability;
                        }
                    }
                } else {
                    framebuffer[path.pixelIndex] += path.radiance;
                }
            }
        }

        active = path.active;
        continuation = RayWorkItem{path.ray, work.pathIndex};
    }

    constexpr uint32_t warpSize = 32;
    constexpr uint32_t warpCount = 8;
    __shared__ uint32_t warpOffsets[warpCount];
    __shared__ uint32_t blockBase;

    uint32_t lane = threadIdx.x % warpSize;
    uint32_t warp = threadIdx.x / warpSize;
    unsigned int activeMask = __ballot_sync(0xffffffff, active);
    uint32_t rank = __popc(activeMask & (lane == 0 ? 0u : ((1u << lane) - 1u)));

    if (lane == 0)
        warpOffsets[warp] = __popc(activeMask);

    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t blockCount = 0;

        for (uint32_t i = 0; i < warpCount; ++i) {
            uint32_t warpCount = warpOffsets[i];
            warpOffsets[i] = blockCount;
            blockCount += warpCount;
        }

        blockBase = blockCount > 0 ? atomicAdd(nextRays.count, blockCount) : 0;
    }

    __syncthreads();

    if (active)
        nextRays.items[blockBase + warpOffsets[warp] + rank] = continuation;
}
