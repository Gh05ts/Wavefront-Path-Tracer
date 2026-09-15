#include "renderer/renderer.cuh"
#include "core/rng.cuh"

#include <cmath>

__global__
void generatePrimaryRays(RayQueue queue, PathState* pathStates, Camera camera, uint32_t width, uint32_t height, const uint32_t* sampleIndex, const RenderTile* tile) {
    uint32_t localPixel = blockIdx.x * blockDim.x + threadIdx.x;
    RenderTile renderTile = *tile;
    uint32_t tilePixelCount = renderTile.width * renderTile.height;

    if (localPixel >= tilePixelCount)
        return;

    uint32_t x = renderTile.x + localPixel % renderTile.width;
    uint32_t y = renderTile.y + localPixel / renderTile.width;
    uint32_t pixel = y * width + x;

    uint32_t rngState = makeRngSeed(pixel ^ (*sampleIndex * 0x9e3779b9u));

    float u = (static_cast<float>(x) + randomFloat(rngState)) / static_cast<float>(width);
    float v = (static_cast<float>(y) + randomFloat(rngState)) / static_cast<float>(height);

    Ray ray = camera.generateRay(u, v);

    pathStates[localPixel] = PathState{ray, Vec3(1.0f, 1.0f, 1.0f), Vec3(0.0f, 0.0f, 0.0f), pixel, 0, rngState, 0.0f, true, true};

    uint32_t outputIndex = atomicAdd(queue.count, 1);

    if (outputIndex >= queue.capacity)
        return;

    queue.items[outputIndex] = RayWorkItem{ray, localPixel};
}

__global__
void advanceSampleIndex(uint32_t* sampleIndex, const RenderTile* tile) {
    if (tile->advanceSample)
        ++*sampleIndex;
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

__device__
bool intersectAabb(const Ray& ray, const Vec3& inverseDirection, float minimumX, float minimumY, float minimumZ, float maximumX, float maximumY, float maximumZ, float closestT, float& nearT) {
    float tx0 = (minimumX - ray.origin.x) * inverseDirection.x;
    float tx1 = (maximumX - ray.origin.x) * inverseDirection.x;
    float ty0 = (minimumY - ray.origin.y) * inverseDirection.y;
    float ty1 = (maximumY - ray.origin.y) * inverseDirection.y;
    float tz0 = (minimumZ - ray.origin.z) * inverseDirection.z;
    float tz1 = (maximumZ - ray.origin.z) * inverseDirection.z;

    float tmin = fmaxf(fminf(tx0, tx1), fmaxf(fminf(ty0, ty1), fminf(tz0, tz1)));
    float tmax = fminf(fmaxf(tx0, tx1), fminf(fmaxf(ty0, ty1), fmaxf(tz0, tz1)));
    nearT = tmin;

    return tmax >= fmaxf(tmin, 0.001f) && tmin < closestT;
}

__device__
bool intersectAabb(const Ray& ray, const Aabb& bounds, float closestT, float& nearT) {
    Vec3 inverseDirection(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    return intersectAabb(ray, inverseDirection,
        bounds.minimum.x, bounds.minimum.y, bounds.minimum.z,
        bounds.maximum.x, bounds.maximum.y, bounds.maximum.z,
        closestT, nearT);
}

__device__
bool intersectNexusAabb(const Ray& ray, const NXB::AABB& bounds, float closestT, float& nearT) {
    Vec3 inverseDirection(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    return intersectAabb(ray, inverseDirection,
        bounds.bMin.x, bounds.bMin.y, bounds.bMin.z,
        bounds.bMax.x, bounds.bMax.y, bounds.bMax.z,
        closestT, nearT);
}

__device__
bool intersectNexusBvh8ChildAabb(const Ray& ray, const Vec3& inverseDirection, const NXB::BVH8::NodeExplicit& node, const Vec3& cellSize, uint32_t slot, float closestT, float& nearT) {
    float minimumX = node.p.x + static_cast<float>(node.qlox[slot]) * cellSize.x;
    float minimumY = node.p.y + static_cast<float>(node.qloy[slot]) * cellSize.y;
    float minimumZ = node.p.z + static_cast<float>(node.qloz[slot]) * cellSize.z;
    float maximumX = node.p.x + static_cast<float>(node.qhix[slot]) * cellSize.x;
    float maximumY = node.p.y + static_cast<float>(node.qhiy[slot]) * cellSize.y;
    float maximumZ = node.p.z + static_cast<float>(node.qhiz[slot]) * cellSize.z;

    return intersectAabb(ray, inverseDirection,
        minimumX, minimumY, minimumZ,
        maximumX, maximumY, maximumZ,
        closestT, nearT);
}

__device__
void intersectNexusBvh2(const Ray& ray, const Triangle* triangles, const NXB::BVH2::DeviceView& bvh, float& closestT, int& closestMaterial, Vec3& closestNormal) {
    constexpr uint32_t maxBvhStackSize = 64;
    uint32_t stack[maxBvhStackSize];
    uint32_t stackSize = 1;
    stack[0] = bvh.nodeCount - 1;

    while (stackSize > 0) {
        const NXB::BVH2::Node& node = bvh.nodes[stack[--stackSize]];
        float nearT;

        if (!intersectNexusAabb(ray, node.bounds, closestT, nearT))
            continue;

        if (node.leftChild == NXB::InvalidIdx) {
            const Triangle& triangle = triangles[node.rightChild];
            float t;

            if (intersectTriangle(ray, triangle, t) && t < closestT) {
                closestT = t;
                closestMaterial = static_cast<int>(triangle.material);

                Vec3 edge1 = triangle.v1 - triangle.v0;
                Vec3 edge2 = triangle.v2 - triangle.v0;
                closestNormal = normalize(cross(edge1, edge2));

                if (dot(closestNormal, ray.direction) > 0.0f)
                    closestNormal = -closestNormal;
            }
        } else {
            float leftNearT;
            float rightNearT;
            bool hitLeft = intersectNexusAabb(ray, bvh.nodes[node.leftChild].bounds, closestT, leftNearT);
            bool hitRight = intersectNexusAabb(ray, bvh.nodes[node.rightChild].bounds, closestT, rightNearT);

            if (hitLeft && hitRight) {
                uint32_t nearChild = leftNearT < rightNearT ? node.leftChild : node.rightChild;
                uint32_t farChild = leftNearT < rightNearT ? node.rightChild : node.leftChild;
                stack[stackSize++] = farChild;
                stack[stackSize++] = nearChild;
            } else if (hitLeft) {
                stack[stackSize++] = node.leftChild;
            } else if (hitRight) {
                stack[stackSize++] = node.rightChild;
            }
        }
    }
}

__device__
bool isOccluded(const Ray& ray, float maximumDistance, Scene scene) {
    for (uint32_t i = 0; i < scene.sphereCount; ++i) {
        float t;

        if (intersectSphere(ray, scene.spheres[i], t) && t < maximumDistance)
            return true;
    }

    for (uint32_t i = 0; i < scene.staticTriangleCount; ++i) {
        float t;

        if (intersectTriangle(ray, scene.staticTriangles[i], t) && t < maximumDistance)
            return true;
    }

    for (uint32_t i = 0; i < scene.instanceCount; ++i) {
        const MeshInstance& instance = scene.instances[i];
        const Blas& blas = scene.blases[instance.blasIndex];
        Ray localRay;
        localRay.origin = (ray.origin - instance.translation) / instance.scale;
        localRay.direction = ray.direction / instance.scale;

        float closestT = maximumDistance;
        int closestMaterial = -1;
        Vec3 closestNormal;
        intersectNexusBvh2(localRay, blas.triangles, blas.bvh, closestT, closestMaterial, closestNormal);

        if (closestMaterial >= 0)
            return true;
    }

    return false;
}

__device__
float powerHeuristic(float firstPdf, float secondPdf) {
    float firstSquared = firstPdf * firstPdf;
    float secondSquared = secondPdf * secondPdf;
    return firstSquared / (firstSquared + secondSquared);
}

__device__
void sampleDirectLight(PathState& path, const Hit& hit, const Material& material, Scene scene) {
    if (!scene.hasAreaLight)
        return;

    const AreaLight& light = scene.areaLight;
    Vec3 lightPosition = light.corner + randomFloat(path.rngState) * light.edgeU + randomFloat(path.rngState) * light.edgeV;
    Vec3 toLight = lightPosition - hit.position;
    float distanceSquared = lengthSquared(toLight);
    float distance = sqrtf(distanceSquared);
    Vec3 direction = toLight / distance;
    float surfaceCosine = fmaxf(0.0f, dot(hit.normal, direction));
    float lightCosine = fmaxf(0.0f, dot(light.normal, -direction));

    if (surfaceCosine == 0.0f || lightCosine == 0.0f)
        return;

    Ray shadowRay(hit.position + direction * 0.001f, direction);

    if (isOccluded(shadowRay, distance - 0.002f, scene))
        return;

    const Material& lightMaterial = scene.materials[light.material];
    constexpr float inversePi = 0.31830988618f;
    float geometryTerm = surfaceCosine * lightCosine * light.area / distanceSquared;
    float lightPdf = distanceSquared / (lightCosine * light.area);
    float bsdfPdf = surfaceCosine * inversePi;
    float misWeight = powerHeuristic(lightPdf, bsdfPdf);
    Vec3 directLighting = hadamard(material.albedo, lightMaterial.emission) * (inversePi * geometryTerm * misWeight);
    path.radiance += hadamard(path.throughput, directLighting);
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

    for (uint32_t i = 0; i < scene.staticTriangleCount; ++i) {
        const Triangle& triangle = scene.staticTriangles[i];
        float t;

        if (intersectTriangle(work.ray, triangle, t) && t < closestT) {
            closestT = t;
            closestMaterial = static_cast<int>(triangle.material);

            Vec3 edge1 = triangle.v1 - triangle.v0;
            Vec3 edge2 = triangle.v2 - triangle.v0;
            closestNormal = normalize(cross(edge1, edge2));

            if (dot(closestNormal, work.ray.direction) > 0.0f)
                closestNormal = -closestNormal;
        }
    }

    if (scene.tlas.nodes != nullptr) {
        constexpr uint32_t maxTlasStackSize = 64;
        uint32_t stack[maxTlasStackSize];
        uint32_t stackSize = 1;
        stack[0] = scene.tlas.nodeCount - 1;

        while (stackSize > 0) {
            const NXB::BVH2::Node& node = scene.tlas.nodes[stack[--stackSize]];
            float nearT;

            if (!intersectNexusAabb(work.ray, node.bounds, closestT, nearT))
                continue;

            if (node.leftChild == NXB::InvalidIdx) {
                const MeshInstance& instance = scene.instances[node.rightChild];
                const Blas& blas = scene.blases[instance.blasIndex];
                Ray localRay;
                localRay.origin = (work.ray.origin - instance.translation) / instance.scale;
                localRay.direction = work.ray.direction / instance.scale;
                intersectNexusBvh2(localRay, blas.triangles, blas.bvh, closestT, closestMaterial, closestNormal);
            } else {
                float leftNearT;
                float rightNearT;
                bool hitLeft = intersectNexusAabb(work.ray, scene.tlas.nodes[node.leftChild].bounds, closestT, leftNearT);
                bool hitRight = intersectNexusAabb(work.ray, scene.tlas.nodes[node.rightChild].bounds, closestT, rightNearT);

                if (hitLeft && hitRight) {
                    uint32_t nearChild = leftNearT < rightNearT ? node.leftChild : node.rightChild;
                    uint32_t farChild = leftNearT < rightNearT ? node.rightChild : node.leftChild;
                    stack[stackSize++] = farChild;
                    stack[stackSize++] = nearChild;
                } else if (hitLeft) {
                    stack[stackSize++] = node.leftChild;
                } else if (hitRight) {
                    stack[stackSize++] = node.rightChild;
                }
            }
        }
    } else if (scene.nexusBvh8.nodes != nullptr) {
        constexpr uint32_t maxBvhStackSize = 96;
        struct StackEntry {
            uint32_t index;
            uint32_t triangleCount;
        };
        StackEntry stack[maxBvhStackSize];
        uint32_t stackSize = 1;
        stack[0] = StackEntry{0, 0};
        Vec3 inverseDirection(1.0f / work.ray.direction.x, 1.0f / work.ray.direction.y, 1.0f / work.ray.direction.z);

        while (stackSize > 0) {
            StackEntry entry = stack[--stackSize];

            if (entry.triangleCount > 0) {
                for (uint32_t i = 0; i < entry.triangleCount; ++i) {
                    const Triangle& triangle = scene.triangles[scene.nexusBvh8.primIdx[entry.index + i]];
                    float t;

                    if (intersectTriangle(work.ray, triangle, t) && t < closestT) {
                        closestT = t;
                        closestMaterial = static_cast<int>(triangle.material);

                        Vec3 edge1 = triangle.v1 - triangle.v0;
                        Vec3 edge2 = triangle.v2 - triangle.v0;
                        closestNormal = normalize(cross(edge1, edge2));

                        if (dot(closestNormal, work.ray.direction) > 0.0f)
                            closestNormal = -closestNormal;
                    }
                }

                continue;
            }

            const NXB::BVH8::NodeExplicit& node = reinterpret_cast<const NXB::BVH8::NodeExplicit&>(scene.nexusBvh8.nodes[entry.index]);
            Vec3 cellSize(
                ldexpf(1.0f, static_cast<int>(node.e[0]) - 127),
                ldexpf(1.0f, static_cast<int>(node.e[1]) - 127),
                ldexpf(1.0f, static_cast<int>(node.e[2]) - 127));
            StackEntry children[8];
            float childNearT[8];
            uint32_t childCount = 0;

            for (uint32_t slot = 0; slot < 8; ++slot) {
                uint8_t meta = node.meta[slot];
                if (meta == 0)
                    continue;

                float nearT;
                if (!intersectNexusBvh8ChildAabb(work.ray, inverseDirection, node, cellSize, slot, closestT, nearT))
                    continue;

                StackEntry child;
                if ((node.imask >> slot) & 1) {
                    child.index = node.childBaseIdx + __popc(node.imask & ((1u << slot) - 1));
                    child.triangleCount = 0;
                } else {
                    child.index = node.primBaseIdx + (meta & 0x1f);
                    child.triangleCount = __popc(meta >> 5);
                }

                uint32_t insertIndex = childCount++;
                while (insertIndex > 0 && nearT < childNearT[insertIndex - 1]) {
                    children[insertIndex] = children[insertIndex - 1];
                    childNearT[insertIndex] = childNearT[insertIndex - 1];
                    --insertIndex;
                }
                children[insertIndex] = child;
                childNearT[insertIndex] = nearT;
            }

            for (uint32_t i = childCount; i > 0; --i) {
                if (stackSize < maxBvhStackSize)
                    stack[stackSize++] = children[i - 1];
            }
        }
    } else if (scene.nexusBvh.nodes != nullptr) {
        constexpr uint32_t maxBvhStackSize = 64;
        uint32_t stack[maxBvhStackSize];
        uint32_t stackSize = 1;
        stack[0] = scene.nexusBvh.nodeCount - 1;

        while (stackSize > 0) {
            const NXB::BVH2::Node& node = scene.nexusBvh.nodes[stack[--stackSize]];
            float nearT;

            if (!intersectNexusAabb(work.ray, node.bounds, closestT, nearT))
                continue;

            if (node.leftChild == NXB::InvalidIdx) {
                const Triangle& triangle = scene.triangles[node.rightChild];
                float t;

                if (intersectTriangle(work.ray, triangle, t) && t < closestT) {
                    closestT = t;
                    closestMaterial = static_cast<int>(triangle.material);

                    Vec3 edge1 = triangle.v1 - triangle.v0;
                    Vec3 edge2 = triangle.v2 - triangle.v0;
                    closestNormal = normalize(cross(edge1, edge2));

                    if (dot(closestNormal, work.ray.direction) > 0.0f)
                        closestNormal = -closestNormal;
                }
            } else {
                float leftNearT;
                float rightNearT;
                bool hitLeft = intersectNexusAabb(work.ray, scene.nexusBvh.nodes[node.leftChild].bounds, closestT, leftNearT);
                bool hitRight = intersectNexusAabb(work.ray, scene.nexusBvh.nodes[node.rightChild].bounds, closestT, rightNearT);

                if (hitLeft && hitRight) {
                    uint32_t nearChild = leftNearT < rightNearT ? node.leftChild : node.rightChild;
                    uint32_t farChild = leftNearT < rightNearT ? node.rightChild : node.leftChild;
                    stack[stackSize++] = farChild;
                    stack[stackSize++] = nearChild;
                } else if (hitLeft) {
                    stack[stackSize++] = node.leftChild;
                } else if (hitRight) {
                    stack[stackSize++] = node.rightChild;
                }
            }
        }
    } else if (scene.bvhNodeCount > 0) {
        constexpr uint32_t maxBvhStackSize = 64;
        uint32_t stack[maxBvhStackSize];
        uint32_t stackSize = 1;
        stack[0] = 0;

        while (stackSize > 0) {
            const BvhNode& node = scene.bvhNodes[stack[--stackSize]];
            float nearT;

            if (!intersectAabb(work.ray, node.bounds, closestT, nearT))
                continue;

            if (node.triangleCount > 0) {
                for (uint32_t i = 0; i < node.triangleCount; ++i) {
                    const Triangle& triangle = scene.triangles[scene.bvhTriangleIndices[node.firstTriangle + i]];
                    float t;

                    if (intersectTriangle(work.ray, triangle, t) && t < closestT) {
                        closestT = t;
                        closestMaterial = static_cast<int>(triangle.material);

                        Vec3 edge1 = triangle.v1 - triangle.v0;
                        Vec3 edge2 = triangle.v2 - triangle.v0;
                        closestNormal = normalize(cross(edge1, edge2));

                        if (dot(closestNormal, work.ray.direction) > 0.0f)
                            closestNormal = -closestNormal;
                    }
                }
            } else {
                float leftNearT;
                float rightNearT;
                bool hitLeft = intersectAabb(work.ray, scene.bvhNodes[node.leftChild].bounds, closestT, leftNearT);
                bool hitRight = intersectAabb(work.ray, scene.bvhNodes[node.rightChild].bounds, closestT, rightNearT);

                if (hitLeft && hitRight) {
                    uint32_t nearChild = leftNearT < rightNearT ? node.leftChild : node.rightChild;
                    uint32_t farChild = leftNearT < rightNearT ? node.rightChild : node.leftChild;
                    stack[stackSize++] = farChild;
                    stack[stackSize++] = nearChild;
                } else if (hitLeft) {
                    stack[stackSize++] = node.leftChild;
                } else if (hitRight) {
                    stack[stackSize++] = node.rightChild;
                }
            }
        }
    } else {
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
            if (!scene.blackBackground) {
                float t = 0.5f * (path.ray.direction.y + 1.0f);
                Vec3 sky = (1.0f - t) * Vec3(1.0f, 1.0f, 1.0f) + t * Vec3(0.5f, 0.7f, 1.0f);
                path.radiance += hadamard(path.throughput, sky);
            }
            path.active = false;
            framebuffer[path.pixelIndex] += path.radiance;
        } else {
            const Hit& hit = results[index].hit;
            const Material& material = scene.materials[hit.material];

            if (material.type == MaterialType::Emissive) {
                float misWeight = 1.0f;

                if (scene.hasAreaLight && hit.material == scene.areaLight.material) {
                    float lightCosine = fmaxf(0.0f, dot(scene.areaLight.normal, -path.ray.direction));

                    if (lightCosine == 0.0f)
                        misWeight = 0.0f;
                    else if (path.depth > 0 && !path.specularBounce) {
                        float lightPdf = hit.t * hit.t / (lightCosine * scene.areaLight.area);
                        misWeight = powerHeuristic(path.previousBsdfPdf, lightPdf);
                    }
                }

                path.radiance += hadamard(path.throughput, material.emission) * misWeight;
                path.active = false;
                framebuffer[path.pixelIndex] += path.radiance;
            } else {
                Vec3 direction;

                switch (material.type) {
                case MaterialType::Diffuse:
                    sampleDirectLight(path, hit, material, scene);
                    path.throughput = hadamard(path.throughput, material.albedo);
                    direction = sampleCosineHemisphere(hit.normal, path.rngState);
                    path.previousBsdfPdf = fmaxf(0.0f, dot(hit.normal, direction)) * 0.31830988618f;
                    path.specularBounce = false;
                    break;

                case MaterialType::Metal: {
                    path.throughput = hadamard(path.throughput, material.albedo);
                    Vec3 reflected = reflect(normalize(path.ray.direction), hit.normal);
                    direction = normalize(reflected + material.roughness * randomInUnitSphere(path.rngState));

                    if (dot(direction, hit.normal) <= 0.0f)
                        path.active = false;

                    path.specularBounce = true;
                    path.previousBsdfPdf = 0.0f;

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

                    path.specularBounce = true;
                    path.previousBsdfPdf = 0.0f;

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
