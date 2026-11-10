#pragma once
#include "specular_glossiness.cuh"
#include "gltf_pbr.cuh"

// Header-backed device traversal and material lookup implementation. It is
// included by renderer.cu to preserve the existing kernel compilation unit.

struct TraceHitState {
    float closestT = 1e30f;
    int closestMaterial = -1;
    Vec3 closestGeometricNormal;
    Vec3 closestNormal;
    Vec2 closestUv{0.0f, 0.0f};
    Vec2 closestUv1{0.0f, 0.0f};
    Vec3 closestTangent;
    Vec3 closestBitangent;
    Vec3 closestTangent1;
    Vec3 closestBitangent1;
    uint32_t closestLightIndex = invalidLightIndex;
    uint32_t closestSphereIndex = 0xffffffffu;
};

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

    constexpr float tMin = rayMinDistance;

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
bool intersectTriangle(const Ray& ray, const Triangle& triangle, float& t, float& u, float& v) {
    constexpr float epsilon = 1e-8f;
    constexpr float tMin = rayMinDistance;

    Vec3 edge1 = triangle.v1 - triangle.v0;
    Vec3 edge2 = triangle.v2 - triangle.v0;
    Vec3 p = cross(ray.direction, edge2);

    float determinant = dot(edge1, p);

    if (fabsf(determinant) < epsilon)
        return false;

    float inverseDeterminant = 1.0f / determinant;
    Vec3 originToVertex = ray.origin - triangle.v0;

    u = dot(originToVertex, p) * inverseDeterminant;
    if (u < 0.0f || u > 1.0f)
        return false;

    Vec3 q = cross(originToVertex, edge1);

    v = dot(ray.direction, q) * inverseDeterminant;
    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = dot(edge2, q) * inverseDeterminant;

    return t > tMin;
}

__device__
bool intersectTriangle(const Ray& ray, const Triangle& triangle, float& t) {
    float u;
    float v;
    return intersectTriangle(ray, triangle, t, u, v);
}

__device__
Vec3 getTriangleNormal(const Triangle& triangle, const Ray& ray, float u, float v, bool faceForward = true) {
    Vec3 normal;

    if (triangle.hasVertexNormals)
        normal = normalize((1.0f - u - v) * triangle.n0 + u * triangle.n1 + v * triangle.n2);
    else
        normal = normalize(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));

    // Dielectrics retain their boundary orientation to distinguish entry/exit.
    return faceForward && dot(normal, ray.direction) > 0.0f ? -normal : normal;
}

__device__
Vec3 getTriangleGeometricNormal(const Triangle& triangle, const Ray& ray, bool faceForward = true) {
    Vec3 normal = normalize(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
    // Dielectrics retain their boundary orientation to distinguish entry/exit.
    return faceForward && dot(normal, ray.direction) > 0.0f ? -normal : normal;
}

__device__
Vec2 getTriangleUv(const Triangle& triangle, float u, float v, bool secondSet) {
    if ((!secondSet && !triangle.hasTexcoords) || (secondSet && !triangle.hasTexcoords1))
        return Vec2{0.0f, 0.0f};

    float w = 1.0f - u - v;
    Vec2 uv0 = secondSet ? triangle.uv1_0 : triangle.uv0;
    Vec2 uv1 = secondSet ? triangle.uv1_1 : triangle.uv1;
    Vec2 uv2 = secondSet ? triangle.uv1_2 : triangle.uv2;
    return Vec2{
        w * uv0.x + u * uv1.x + v * uv2.x,
        w * uv0.y + u * uv1.y + v * uv2.y};
}

__device__
float getMaterialOpacity(const Material& material, const Vec2& uv, const Vec2& uv1, Scene scene);

__device__
bool intersectTriangleHit(const Ray& ray, const Triangle& triangle, Scene scene, float& closestT, int& closestMaterial, Vec3& closestGeometricNormal, Vec3& closestNormal, Vec2& closestUv, Vec2& closestUv1, Vec3& closestTangent, Vec3& closestBitangent, Vec3& closestTangent1, Vec3& closestBitangent1, uint32_t& closestLightIndex, uint32_t materialOverride = invalidMaterialIndex) {
    float t;
    float u;
    float v;

    if (!intersectTriangle(ray, triangle, t, u, v) || t >= closestT)
        return false;

    Vec2 uv = getTriangleUv(triangle, u, v, false);
    Vec2 uv1 = getTriangleUv(triangle, u, v, true);

    uint32_t materialIndex = materialOverride == invalidMaterialIndex ? triangle.material : materialOverride;
    if (scene.materials[materialIndex].alphaMasked && getMaterialOpacity(scene.materials[materialIndex], uv, uv1, scene) < scene.materials[materialIndex].alphaCutoff)
        return false;

    closestT = t;
    closestMaterial = static_cast<int>(materialIndex);
    closestGeometricNormal = getTriangleGeometricNormal(triangle, ray, scene.materials[materialIndex].type != MaterialType::Dielectric);
    closestNormal = getTriangleNormal(triangle, ray, u, v, scene.materials[materialIndex].type != MaterialType::Dielectric);
    closestUv = uv;
    closestUv1 = uv1;
    if (triangle.hasVertexTangents) {
        float w = 1.0f - u - v;
        Vec3 tangent = w * triangle.vertexTangent0 + u * triangle.vertexTangent1 + v * triangle.vertexTangent2;
        tangent = tangent - closestNormal * dot(closestNormal, tangent);

        if (lengthSquared(tangent) > 0.00000001f) {
            closestTangent = normalize(tangent);
            float sign = w * triangle.vertexTangentSign0 + u * triangle.vertexTangentSign1 + v * triangle.vertexTangentSign2;
            closestBitangent = (sign < 0.0f ? -1.0f : 1.0f) * normalize(cross(closestNormal, closestTangent));
        } else {
            closestTangent = triangle.tangent;
            closestBitangent = triangle.bitangent;
        }
    } else {
        closestTangent = triangle.tangent;
        closestBitangent = triangle.bitangent;
    }
    closestTangent1 = triangle.tangent1;
    closestBitangent1 = triangle.bitangent1;
    closestLightIndex = materialOverride == invalidMaterialIndex ? triangle.lightIndex : invalidLightIndex;
    return true;
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

    return tmax >= fmaxf(tmin, rayMinDistance) && tmin < closestT;
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

__device__ __forceinline__
bool intersectNexusAabb(const Ray& ray, const Vec3& inverseDirection, const NXB::AABB& bounds, float closestT, float& nearT) {
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
void intersectNexusBvh2(const Ray& ray, const Triangle* triangles, const NXB::BVH2::DeviceView& bvh, Scene scene, float& closestT, int& closestMaterial, Vec3& closestGeometricNormal, Vec3& closestNormal, Vec2& closestUv, Vec2& closestUv1, Vec3& closestTangent, Vec3& closestBitangent, Vec3& closestTangent1, Vec3& closestBitangent1, uint32_t& closestLightIndex, uint32_t materialOverride = invalidMaterialIndex) {
    Vec3 inverseDirection(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    constexpr uint32_t maxBvhStackSize = 64;
    uint32_t stack[maxBvhStackSize];
    uint32_t stackSize = 1;
    stack[0] = bvh.nodeCount - 1;

    while (stackSize > 0) {
        const NXB::BVH2::Node& node = bvh.nodes[stack[--stackSize]];
        float nearT;

        if (!intersectNexusAabb(ray, inverseDirection, node.bounds, closestT, nearT))
            continue;

        if (node.leftChild == NXB::InvalidIdx) {
            const Triangle& triangle = triangles[node.rightChild];
            float t;
            float u;
            float v;

            if (intersectTriangle(ray, triangle, t, u, v) && t < closestT) {
                Vec2 uv = getTriangleUv(triangle, u, v, false);
                Vec2 uv1 = getTriangleUv(triangle, u, v, true);
                uint32_t materialIndex = materialOverride == invalidMaterialIndex ? triangle.material : materialOverride;

                if (scene.materials[materialIndex].alphaMasked && getMaterialOpacity(scene.materials[materialIndex], uv, uv1, scene) < scene.materials[materialIndex].alphaCutoff)
                    continue;

                closestT = t;
                closestMaterial = static_cast<int>(materialIndex);
                closestGeometricNormal = getTriangleGeometricNormal(triangle, ray, scene.materials[materialIndex].type != MaterialType::Dielectric);
                closestNormal = getTriangleNormal(triangle, ray, u, v, scene.materials[materialIndex].type != MaterialType::Dielectric);
                closestUv = uv;
                closestUv1 = uv1;
                if (triangle.hasVertexTangents) {
                    float w = 1.0f - u - v;
                    Vec3 tangent = w * triangle.vertexTangent0 + u * triangle.vertexTangent1 + v * triangle.vertexTangent2;
                    tangent = tangent - closestNormal * dot(closestNormal, tangent);

                    if (lengthSquared(tangent) > 0.00000001f) {
                        closestTangent = normalize(tangent);
                        float sign = w * triangle.vertexTangentSign0 + u * triangle.vertexTangentSign1 + v * triangle.vertexTangentSign2;
                        closestBitangent = (sign < 0.0f ? -1.0f : 1.0f) * normalize(cross(closestNormal, closestTangent));
                    } else {
                        closestTangent = triangle.tangent;
                        closestBitangent = triangle.bitangent;
                    }
                } else {
                    closestTangent = triangle.tangent;
                    closestBitangent = triangle.bitangent;
                }
                closestTangent1 = triangle.tangent1;
                closestBitangent1 = triangle.bitangent1;
                closestLightIndex = materialOverride == invalidMaterialIndex ? triangle.lightIndex : invalidLightIndex;
            }
        } else {
            float leftNearT;
            float rightNearT;
            bool hitLeft = intersectNexusAabb(ray, inverseDirection, bvh.nodes[node.leftChild].bounds, closestT, leftNearT);
            bool hitRight = intersectNexusAabb(ray, inverseDirection, bvh.nodes[node.rightChild].bounds, closestT, rightNearT);

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
bool isOccluded(const Ray& ray, float maximumDistance, Scene scene);

__device__
float powerHeuristic(float firstPdf, float secondPdf) {
    float firstSquared = firstPdf * firstPdf;
    float secondSquared = secondPdf * secondPdf;
    return firstSquared / (firstSquared + secondSquared);
}

__device__
Vec2 getTextureUv(const TextureBinding& binding, const Hit& hit) {
    Vec2 uv = binding.texCoord == 1 ? hit.uv1 : hit.uv;

    if (!binding.hasTransform)
        return uv;

    uv.x *= binding.scale.x;
    uv.y *= binding.scale.y;
    float sine;
    float cosine;
    sincosf(binding.rotation, &sine, &cosine);
    return Vec2{cosine * uv.x - sine * uv.y + binding.offset.x, sine * uv.x + cosine * uv.y + binding.offset.y};
}

__device__
float4 sampleTexture(const TextureBinding& binding, const Hit& hit, Scene scene, float4 fallback) {
    if (binding.texture == invalidTextureIndex || binding.texture >= scene.textureCount)
        return fallback;

    Vec2 uv = getTextureUv(binding, hit);
    return tex2D<float4>(scene.textures[binding.texture].object, uv.x, 1.0f - uv.y);
}

__device__
Vec3 srgbToLinear(const float4& texel) {
    Vec3 srgb(texel.x, texel.y, texel.z);
    return Vec3(powf(srgb.x, 2.2f), powf(srgb.y, 2.2f), powf(srgb.z, 2.2f));
}

__host__ __device__ inline float gltfSrgbChannel(float value) {
    return value <= 0.04045f ? value / 12.92f : powf((value + 0.055f) / 1.055f, 2.4f);
}

__device__ Vec3 gltfSrgbToLinear(const float4& texel) {
    return Vec3(gltfSrgbChannel(texel.x), gltfSrgbChannel(texel.y), gltfSrgbChannel(texel.z));
}

__device__
Vec3 getMaterialAlbedo(const Material& material, const Hit& hit, Scene scene) {
    float4 texel = sampleTexture(material.albedoTexture, hit, scene, make_float4(1.0f, 1.0f, 1.0f, 1.0f));
    return hadamard(material.albedo, material.specularGlossinessWorkflow ? gltfSrgbToLinear(texel) : srgbToLinear(texel));
}

__device__
float getMaterialOpacity(const Material& material, const Vec2& uv, const Vec2& uv1, Scene scene) {
    if (material.albedoTexture.texture == invalidTextureIndex || material.albedoTexture.texture >= scene.textureCount)
        return material.opacity;

    Hit hit{};
    hit.uv = uv;
    hit.uv1 = uv1;
    return material.opacity * sampleTexture(material.albedoTexture, hit, scene, make_float4(1.0f, 1.0f, 1.0f, 1.0f)).w;
}

__device__
Vec3 getMaterialEmission(const Material& material, const Hit& hit, Scene scene) {
    return hadamard(material.emission, srgbToLinear(sampleTexture(material.emissiveTexture, hit, scene, make_float4(1.0f, 1.0f, 1.0f, 1.0f))));
}

__device__
float getMaterialMetallic(const Material& material, const Hit& hit, Scene scene) {
    return material.metallic * sampleTexture(material.metallicRoughnessTexture, hit, scene, make_float4(1.0f, 1.0f, 1.0f, 1.0f)).z;
}

__device__
float getMaterialRoughness(const Material& material, const Hit& hit, Scene scene) {
    if (material.specularGlossinessWorkflow)
        return 1.0f - material.glossiness * sampleTexture(material.specularGlossinessTexture, hit, scene, make_float4(1.0f, 1.0f, 1.0f, 1.0f)).w;
    return material.roughness * sampleTexture(material.metallicRoughnessTexture, hit, scene, make_float4(1.0f, 1.0f, 1.0f, 1.0f)).y;
}

__device__ SpecularGlossinessSurface getSpecGlossSurface(const Material& material, const Hit& hit, Scene scene) {
    float4 texel = sampleTexture(material.specularGlossinessTexture, hit, scene, make_float4(1.0f, 1.0f, 1.0f, 1.0f));
    return SpecularGlossinessSurface{getMaterialAlbedo(material, hit, scene),
        hadamard(material.specular, gltfSrgbToLinear(texel)), getMaterialRoughness(material, hit, scene)};
}

__device__ float diffuseEnvironmentProbability(Scene scene) {
    const auto& map = scene.environment;
    return map.texture && map.intensity > 0.0f && map.width && map.height &&
        map.probabilities && map.alias ? 0.5f : 0.0f;
}

// A continuation is an actual traced path, so an environment proposal can
// discover sun connections through specular geometry that blocks ordinary NEE.
// Keep a cosine component for full diffuse support and account for the complete
// mixture both in throughput and in the competing direct-light MIS estimator.
__device__ float diffuseContinuationPdf(Scene scene, const Vec3& normal, const Vec3& direction) {
    float probability = diffuseEnvironmentProbability(scene);
    float cosinePdf = fmaxf(0.0f, dot(normal, direction)) * 0.31830988618f;
    return probability > 0.0f ? (1.0f - probability) * cosinePdf +
        probability * environmentPdf(scene.environment, direction) : cosinePdf;
}

__device__ EnvironmentSample samplePathEnvironment(Scene scene, RngState& rng) {
    float coarse = randomFloat(rng), fine = randomFloat(rng);
    double candidate = double(coarse) + double(fine) * (1.0 / 16777216.0);
    float coin = randomFloat(rng), u = randomFloat(rng), v = randomFloat(rng);
    return sampleEnvironment(scene.environment, candidate, coin, u, v);
}

__device__ Vec3 evaluateSurface(const Material& material, const Hit& hit, Scene scene,
    const Vec3& outgoing, const Vec3& incoming, float& pdf) {
    if (material.type == MaterialType::SpecularGlossiness)
        return evaluateSpecGloss(getSpecGlossSurface(material, hit, scene), hit.normal, outgoing, incoming, pdf);
    MaterialType effective = material.type == MaterialType::Diffuse &&
        getMaterialMetallic(material, hit, scene) > 0.5f ? MaterialType::Metal : material.type;
    if (effective == MaterialType::Metal) {
        if (getMaterialRoughness(material, hit, scene) <= 0.0f) { pdf = 0.0f; return Vec3(); }
        SpecularGlossinessSurface surface{Vec3(), getMaterialAlbedo(material, hit, scene),
            getMaterialRoughness(material, hit, scene)};
        return evaluateSpecGloss(surface, hit.normal, outgoing, incoming, pdf);
    }
    pdf = diffuseContinuationPdf(scene, hit.normal, incoming);
    return getMaterialAlbedo(material, hit, scene) * 0.31830988618f;
}

__device__
float getMaterialThickness(const Material& material, const Hit& hit, Scene scene) {
    if (material.thicknessTexture.texture == invalidTextureIndex || material.thicknessTexture.texture >= scene.textureCount)
        return 1.0f;

    return sampleTexture(material.thicknessTexture, hit, scene, make_float4(1.0f, 1.0f, 1.0f, 1.0f)).x;
}

__device__
Vec3 getMaterialNormal(const Material& material, const Hit& hit, Scene scene) {
    if (!scene.useNormalMaps || material.normalTexture.texture == invalidTextureIndex || material.normalTexture.texture >= scene.textureCount)
        return hit.normal;

    Vec3 tangent = material.normalTexture.texCoord == 1 ? hit.tangent1 : hit.tangent;
    Vec3 bitangent = material.normalTexture.texCoord == 1 ? hit.bitangent1 : hit.bitangent;

    if (!isfinite(tangent.x) || !isfinite(tangent.y) || !isfinite(tangent.z) ||
        !isfinite(bitangent.x) || !isfinite(bitangent.y) || !isfinite(bitangent.z) ||
        lengthSquared(tangent) < 0.00000001f || lengthSquared(bitangent) < 0.00000001f)
        return hit.normal;

    tangent = normalize(tangent - hit.normal * dot(hit.normal, tangent));
    bitangent = normalize(cross(hit.normal, tangent));

    if (!isfinite(tangent.x) || !isfinite(tangent.y) || !isfinite(tangent.z) ||
        !isfinite(bitangent.x) || !isfinite(bitangent.y) || !isfinite(bitangent.z))
        return hit.normal;

    if (dot(bitangent, material.normalTexture.texCoord == 1 ? hit.bitangent1 : hit.bitangent) < 0.0f)
        bitangent = -bitangent;

    float4 texel = sampleTexture(material.normalTexture, hit, scene, make_float4(0.5f, 0.5f, 1.0f, 1.0f));
    Vec3 tangentNormal((texel.x * 2.0f - 1.0f) * material.normalScale, (texel.y * 2.0f - 1.0f) * material.normalScale, texel.z * 2.0f - 1.0f);
    tangentNormal = normalize(tangentNormal);

    if (scene.normalMapMinimumCosine > 0.0f && tangentNormal.z < scene.normalMapMinimumCosine) {
        float minimumCosine = fminf(scene.normalMapMinimumCosine, 1.0f);
        float tangentLength = sqrtf(fmaxf(0.0f, tangentNormal.x * tangentNormal.x + tangentNormal.y * tangentNormal.y));

        if (tangentLength > 0.0f) {
            float scale = sqrtf(fmaxf(0.0f, 1.0f - minimumCosine * minimumCosine)) / tangentLength;
            tangentNormal.x *= scale;
            tangentNormal.y *= scale;
        } else {
            tangentNormal.x = 0.0f;
            tangentNormal.y = 0.0f;
        }

        tangentNormal.z = minimumCosine;
    }

    Vec3 normal = tangent * tangentNormal.x + bitangent * tangentNormal.y + hit.normal * tangentNormal.z;

    if (!isfinite(normal.x) || !isfinite(normal.y) || !isfinite(normal.z) || lengthSquared(normal) < 0.00000001f)
        return hit.normal;

    return normalize(normal);
}

__device__ GltfPbrSurface getGltfPbrSurface(const Material& material, const Hit& hit,
    const Hit& shadedHit, Scene scene) {
    float weight=material.specularWeight*sampleTexture(material.specularWeightTexture,hit,scene,make_float4(1,1,1,1)).w;
    Vec3 color=hadamard(material.specularColor,gltfSrgbToLinear(sampleTexture(material.specularColorTexture,hit,scene,make_float4(1,1,1,1))));
    float f0=(material.ior-1.f)/(material.ior+1.f); f0*=f0;
    Vec3 dielectricF0(fminf(1.f,f0*color.x)*weight,fminf(1.f,f0*color.y)*weight,fminf(1.f,f0*color.z)*weight);
    Material coatMaterial=material;
    coatMaterial.normalTexture=material.clearcoatNormalTexture;
    coatMaterial.normalScale=material.clearcoatNormalScale;
    Vec3 coatNormal=getMaterialNormal(coatMaterial,hit,scene);
    if (dot(coatNormal,hit.geometricNormal)<0) coatNormal=-coatNormal;
    return {getMaterialAlbedo(material,hit,scene),shadedHit.normal,coatNormal,
        getMaterialMetallic(material,hit,scene),getMaterialRoughness(material,hit,scene),dielectricF0,weight,
        material.clearcoat*sampleTexture(material.clearcoatTexture,hit,scene,make_float4(1,1,1,1)).x,
        material.clearcoatRoughness*sampleTexture(material.clearcoatRoughnessTexture,hit,scene,make_float4(1,1,1,1)).y};
}

__device__ Vec3 evaluateSurfaceCos(const Material& material, const Hit& hit, const Hit& shadedHit,
    Scene scene, Vec3 view, Vec3 light, float& pdf) {
    if (material.gltfPbrWorkflow) {
        auto surface=getGltfPbrSurface(material,hit,shadedHit,scene);
        correctGltfReflectionNormals(surface,hit.geometricNormal,view);
        return evaluateGltfPbrCos(surface,view,light,diffuseContinuationPdf(scene,surface.normal,light),pdf);
    }
    return evaluateSurface(material,shadedHit,scene,view,light,pdf)*fmaxf(0.f,dot(shadedHit.normal,light));
}

__device__
void sampleDirectLight(PathState& path, const Hit& hit, const Hit& shadedHit, const Material& material, Scene scene) {
    if (scene.lightCount == 0)
        return;

    uint32_t candidate = min(static_cast<uint32_t>(randomFloat(path.rngState) * scene.lightCount), scene.lightCount - 1);
    uint32_t lightIndex = randomFloat(path.rngState) < scene.lightAlias[candidate].probability ? candidate : scene.lightAlias[candidate].alias;
    const TriangleLight& light = scene.lights[lightIndex];
    float rootU = sqrtf(randomFloat(path.rngState));
    float v = randomFloat(path.rngState);
    Vec3 lightPosition = (1.0f - rootU) * light.v0 + rootU * (1.0f - v) * light.v1 + rootU * v * light.v2;
    Vec3 toLight = lightPosition - hit.position;
    float distanceSquared = lengthSquared(toLight);
    float distance = sqrtf(distanceSquared);
    Vec3 direction = toLight / distance;
    float geometricCosine = fmaxf(0.0f, dot(hit.geometricNormal, direction));
    float shadingCosine = fmaxf(0.0f, dot(shadedHit.normal, direction));
    float lightCosine = fmaxf(0.0f, dot(light.normal, -direction));

    if (geometricCosine == 0.0f || (!material.gltfPbrWorkflow && shadingCosine == 0.0f) || lightCosine == 0.0f)
        return;

    Vec3 shadowOrigin = offsetRayOrigin(hit.position, hit.geometricNormal, direction, hit.t);
    Vec3 shadowTarget = offsetRayOrigin(lightPosition, light.normal, -direction, distance);
    Vec3 shadowSegment = shadowTarget - shadowOrigin;
    float shadowDistance = length(shadowSegment);
    if (!(shadowDistance > rayMinDistance)) return;
    Ray shadowRay(shadowOrigin, shadowSegment / shadowDistance);

    if (!scene.ignoreDirectLightOcclusion && isOccluded(shadowRay, shadowDistance - rayMinDistance, scene))
        return;

    const Material& lightMaterial = scene.materials[light.material];
    float geometryTerm = lightCosine * light.area / (distanceSquared * light.selectionPdf);
    float lightPdf = light.selectionPdf * distanceSquared / (lightCosine * light.area);
    float bsdfPdf;
    Vec3 bsdf = evaluateSurfaceCos(material, hit, shadedHit, scene, -path.ray.direction, direction, bsdfPdf);
    float misWeight = powerHeuristic(lightPdf, bsdfPdf);
    Vec3 directLighting = hadamard(bsdf, lightMaterial.emission) * (geometryTerm * misWeight);
    addPathRadiance(path, hadamard(path.throughput, directLighting));
}

__device__ Vec3 environmentRadiance(const EnvironmentView& map, const Vec3& direction) {
    Vec2 uv = environmentUv(direction, map.rotation);
    float4 texel = tex2D<float4>(map.texture, uv.x, uv.y);
    return Vec3(texel.x, texel.y, texel.z) * map.intensity;
}

__device__ void addEnvironmentOrSky(PathState& path, Scene scene) {
    if (scene.environment.texture) {
        float weight = path.depth > 0 && !path.specularBounce ?
            powerHeuristic(path.previousBsdfPdf, environmentPdf(scene.environment, path.ray.direction)) : 1.0f;
        addPathRadiance(path, hadamard(path.throughput, environmentRadiance(scene.environment, path.ray.direction)) * weight, true);
    } else if (!scene.blackBackground) {
        float t = 0.5f * (path.ray.direction.y + 1.0f);
        Vec3 sky = (1.0f-t)*Vec3(1.0f,1.0f,1.0f) + t*Vec3(0.5f,0.7f,1.0f);
        addPathRadiance(path, hadamard(path.throughput, sky), true);
    }
}

__device__ Vec3 continueDielectricSurface(PathState& path, const Hit& hit,
    const Hit& shadedHit, const Material& material, Scene scene, uint32_t maxDepth) {
    path.throughput = hadamard(path.throughput, getMaterialAlbedo(material, shadedHit, scene));
    auto boundary = dielectricInterface(normalize(path.ray.direction), hit.geometricNormal,
        shadedHit.normal, material.ior);
    float etaSquared = boundary.eta * boundary.eta;
    bool reflectionEscapes = false, transmissionEscapes = false;
    // Integrate terminal HDR branches instead of assigning the full sun value
    // to a rare Fresnel coin flip. Other geometry (including more glass) still
    // occludes each connection. Respect the existing bounce cap and do not add
    // the same terminal branch again when the sampled continuation escapes.
    if (scene.environment.texture && path.depth + 1 < maxDepth) {
        if (boundary.reflectance > 0.0f) {
            Vec3 direction = boundary.reflection;
            Ray ray(offsetRayOrigin(hit.position, hit.geometricNormal, direction, hit.t), direction);
            reflectionEscapes = !isOccluded(ray, 1.0e30f, scene);
            if (reflectionEscapes)
                addPathRadiance(path, hadamard(path.throughput, environmentRadiance(scene.environment, direction)) * boundary.reflectance);
        }
        if (boundary.reflectance < 1.0f) {
            Vec3 direction = boundary.transmission;
            Ray ray(offsetRayOrigin(hit.position, hit.geometricNormal, direction, hit.t), direction);
            transmissionEscapes = !isOccluded(ray, 1.0e30f, scene);
            if (transmissionEscapes)
                addPathRadiance(path, hadamard(path.throughput, environmentRadiance(scene.environment, direction)) *
                    ((1.0f - boundary.reflectance) * etaSquared));
        }
    }
    bool reflected = boundary.reflectance >= 1.0f || boundary.reflectance > randomFloat(path.rngState);
    if (!reflected) {
        // Camera paths transport radiance; paired air/glass boundaries cancel
        // this factor. Its inverse keeps the roulette heuristic eta-neutral.
        path.throughput *= etaSquared;
        path.etaScale /= etaSquared;
        if (boundary.frontFace) {
            path.mediumMaterial = hit.material;
            path.mediumDensity = material.volumeDensity * getMaterialThickness(material, hit, scene);
        } else {
            path.mediumMaterial = 0xffffffffu;
            path.mediumDensity = 1.0f;
        }
    }
    if (reflected ? reflectionEscapes : transmissionEscapes) path.active = false;
    path.specularBounce = true;
    path.previousBsdfPdf = 0.0f;
    return reflected ? boundary.reflection : boundary.transmission;
}

__device__ void sampleDirectEnvironment(PathState& path, const Hit& hit, const Hit& shadedHit,
    const Material& material, Scene scene) {
    const EnvironmentView& map = scene.environment;
    if (!map.texture || map.intensity == 0.0f) return;
    EnvironmentSample light = samplePathEnvironment(scene, path.rngState);
    float cosine = dot(shadedHit.normal, light.direction);
    if ((!material.gltfPbrWorkflow && cosine <= 0.0f) || dot(hit.geometricNormal, light.direction) <= 0.0f || !(light.pdf > 0.0f)) return;
    Ray shadow(offsetRayOrigin(hit.position, hit.geometricNormal, light.direction, hit.t), light.direction);
    if (!scene.ignoreDirectLightOcclusion && isOccluded(shadow, 1.0e30f, scene)) return;
    float bsdfPdf;
    Vec3 bsdf = evaluateSurfaceCos(material, hit, shadedHit, scene, -path.ray.direction, light.direction, bsdfPdf);
    float weight = powerHeuristic(light.pdf, bsdfPdf);
    Vec3 direct = hadamard(bsdf, environmentRadiance(map, light.direction)) * (weight / light.pdf);
    addPathRadiance(path, hadamard(path.throughput, direct));
}

__device__ void addPunctualLight(PathState& path, const Hit& hit, const Hit& shadedHit,
    const Material& material, Scene scene, const PunctualLight& source, float inverseSelectionPdf) {
    PunctualLightSample light = samplePunctualLight(source, hit.position);
    float cosine = dot(shadedHit.normal, light.direction);
    if ((!material.gltfPbrWorkflow && cosine <= 0.0f) || dot(hit.geometricNormal, light.direction) <= 0.0f ||
        lengthSquared(light.incident) == 0.0f) return;
    Ray shadow(offsetRayOrigin(hit.position, hit.geometricNormal, light.direction, hit.t), light.direction);
    float shadowDistance = light.distance;
    if (source.type != PunctualLightType::Directional) {
        Vec3 segment = source.position - shadow.origin;
        shadowDistance = length(segment);
        if (!(shadowDistance > rayMinDistance)) return;
        shadow.direction = segment / shadowDistance;
    }
    if (!scene.ignoreDirectLightOcclusion && isOccluded(shadow, shadowDistance - rayMinDistance, scene)) return;
    float ignoredPdf;
    Vec3 bsdf = evaluateSurfaceCos(material, hit, shadedHit, scene, -path.ray.direction, light.direction, ignoredPdf);
    // Delta lights cannot be reached by BSDF direction sampling: no MIS weight.
    Vec3 direct = hadamard(bsdf, light.incident) * inverseSelectionPdf;
    addPathRadiance(path, hadamard(path.throughput, direct));
}

__device__ void sampleDirectPunctualLight(PathState& path, const Hit& hit, const Hit& shadedHit,
    const Material& material, Scene scene) {
    for (uint32_t i = 0; i < scene.directionalLightCount; ++i)
        addPunctualLight(path, hit, shadedHit, material, scene, scene.punctualLights[i], 1.0f);
    uint32_t count = scene.punctualLightCount - scene.directionalLightCount;
    if (count == 0) return;
    uint32_t index = scene.directionalLightCount + min(static_cast<uint32_t>(randomFloat(path.rngState) * count), count - 1);
    addPunctualLight(path, hit, shadedHit, material, scene, scene.punctualLights[index], static_cast<float>(count));
}

__device__ Vec3 continueDiffuseSurface(PathState& path, const Hit& hit, const Hit& shadedHit,
    const Material& material, Scene scene) {
    float probability = diffuseEnvironmentProbability(scene);
    Vec3 direction = probability > 0.0f && randomFloat(path.rngState) < probability ?
        samplePathEnvironment(scene, path.rngState).direction :
        sampleCosineHemisphere(shadedHit.normal, path.rngState);
    float cosine = dot(shadedHit.normal, direction);
    float pdf = diffuseContinuationPdf(scene, shadedHit.normal, direction);
    if (!(pdf > 0.0f) || cosine <= 0.0f || dot(direction, hit.geometricNormal) <= 0.0f) {
        path.active = false;
        return direction;
    }
    // Reject below-geometry samples. Resampling them about another normal and
    // reporting only its cosine density does not match the sampled distribution.
    path.throughput = hadamard(path.throughput, getMaterialAlbedo(material, shadedHit, scene)) *
        (cosine * 0.31830988618f / pdf);
    path.previousBsdfPdf = pdf;
    path.specularBounce = false;
    return direction;
}

__device__ Vec3 continueGlossySurface(PathState& path, const Hit& hit, const Hit& shadedHit,
    const Material& material, Scene scene) {
    SpecularGlossinessSurface surface = material.type == MaterialType::SpecularGlossiness ?
        getSpecGlossSurface(material, shadedHit, scene) :
        SpecularGlossinessSurface{Vec3(), getMaterialAlbedo(material, shadedHit, scene),
            getMaterialRoughness(material, shadedHit, scene)};
    Vec3 outgoing = -path.ray.direction;
    float choice = randomFloat(path.rngState), u = randomFloat(path.rngState), v = randomFloat(path.rngState);
    Vec3 direction = sampleSpecGloss(surface, shadedHit.normal, outgoing, choice, u, v);
    float pdf;
    Vec3 bsdf = evaluateSpecGloss(surface, shadedHit.normal, outgoing, direction, pdf);
    if (!(pdf > 0.0f) || dot(direction, hit.geometricNormal) <= 0.0f) {
        path.active = false;
        return direction;
    }
    path.throughput = hadamard(path.throughput, bsdf) * (fmaxf(0.0f, dot(shadedHit.normal, direction)) / pdf);
    path.previousBsdfPdf = pdf;
    path.specularBounce = false; // finite GGX, participates in area-light MIS
    return direction;
}

__device__ Vec3 continueGltfPbrSurface(PathState& path, const Hit& hit, const Hit& shadedHit,
    const Material& material, Scene scene, uint32_t maxDepth) {
    auto surface=getGltfPbrSurface(material,hit,shadedHit,scene);
    Vec3 view=-path.ray.direction;
    correctGltfReflectionNormals(surface,hit.geometricNormal,view);
    auto weights=pbrWeights(surface,view);
    float choice=randomFloat(path.rngState);
    bool coat=choice<weights.coat;
    bool specular=!coat && choice<weights.coat+weights.specular;
    bool delta=(coat && surface.coatRoughness<=0) || (specular && surface.roughness<=0);
    bool baseEscapes=false,coatEscapes=false;
    Vec3 baseCoefficient=pbrBaseFresnel(surface,dot(pbrSpecularNormal(surface),view))*(1.f-weights.coat);
    // Integrate visible delta/HDR connections exactly, as for smooth glass.
    // Opaque geometry still blocks each connection; no transparent shadow hack.
    if (scene.environment.texture && path.depth+1<maxDepth) {
        for (int lobe=0;lobe<2;++lobe) {
            bool isCoat=lobe==1;
            float mass=isCoat?weights.coat:weights.specular;
            float roughness=isCoat?surface.coatRoughness:surface.roughness;
            if (!(mass>0) || roughness>0) continue;
            Vec3 normal=isCoat?surface.coatNormal:pbrSpecularNormal(surface);
            Vec3 direction=pbrSampleReflection(normal,view,0,0,0);
            if (dot(normal,view)<=0 || dot(direction,hit.geometricNormal)<=0) continue;
            Ray ray(offsetRayOrigin(hit.position,hit.geometricNormal,direction,hit.t),direction);
            bool escapes=!isOccluded(ray,1.e30f,scene);
            if (isCoat) coatEscapes=escapes; else baseEscapes=escapes;
            if (escapes) {
                Vec3 coefficient=isCoat?Vec3(weights.coat,weights.coat,weights.coat):baseCoefficient;
                addPathRadiance(path,hadamard(path.throughput,hadamard(coefficient,environmentRadiance(scene.environment,direction))));
            }
        }
    }
    Vec3 direction;
    if (coat || specular) {
        float u=randomFloat(path.rngState),v=randomFloat(path.rngState);
        Vec3 normal=coat?surface.coatNormal:pbrSpecularNormal(surface);
        direction=pbrSampleReflection(normal,view,coat?surface.coatRoughness:surface.roughness,u,v);
        if (dot(normal,view)<=0) { path.active=false; return direction; }
    } else {
        float probability=diffuseEnvironmentProbability(scene);
        direction=probability>0 && randomFloat(path.rngState)<probability ?
            samplePathEnvironment(scene,path.rngState).direction : sampleCosineHemisphere(surface.normal,path.rngState);
    }
    if (dot(direction,hit.geometricNormal)<=0) { path.active=false; return direction; }
    if (delta) {
        float mass=coat?weights.coat:weights.specular;
        Vec3 coefficient=coat?Vec3(weights.coat,weights.coat,weights.coat):baseCoefficient;
        path.throughput=hadamard(path.throughput,coefficient)/mass;
        path.specularBounce=true;
        path.previousBsdfPdf=0;
        if (coat?coatEscapes:baseEscapes) path.active=false;
    } else {
        float pdf;
        Vec3 value=evaluateGltfPbrCos(surface,view,direction,diffuseContinuationPdf(scene,surface.normal,direction),pdf);
        if (!(pdf>0)) { path.active=false; return direction; }
        path.throughput=hadamard(path.throughput,value)/pdf;
        path.previousBsdfPdf=pdf;
        path.specularBounce=false;
    }
    return direction;
}

__device__ __forceinline__
void intersectSphereHit(const Ray& ray, const Sphere& sphere, uint32_t sphereIndex, TraceHitState& hitState) {
    float t;
    if (!intersectSphere(ray, sphere, t) || t > hitState.closestT)
        return;
    // Preserve the old sphere-array ordering for exactly coincident hits.
    if (t == hitState.closestT && sphereIndex >= hitState.closestSphereIndex)
        return;
    hitState.closestT = t;
    hitState.closestMaterial = static_cast<int>(sphere.material);
    hitState.closestSphereIndex = sphereIndex;
}

__device__ __forceinline__
void intersectMeshInstance(const Ray& ray, uint32_t instanceIndex, Scene scene, TraceHitState& hitState, bool useBvh) {
    const MeshInstance& instance = scene.instances[instanceIndex];
    const Blas& blas = scene.blases[instance.blasIndex];
    Ray localRay;
    localRay.origin = inverseTransformVector(instance.transform, ray.origin - instance.transform.translation);
    localRay.direction = inverseTransformVector(instance.transform, ray.direction);
    float previousClosestT = hitState.closestT;
    if (useBvh && blas.bvh.nodes != nullptr) {
        intersectNexusBvh2(localRay, blas.triangles, blas.bvh, scene, hitState.closestT,
            hitState.closestMaterial, hitState.closestGeometricNormal, hitState.closestNormal,
            hitState.closestUv, hitState.closestUv1, hitState.closestTangent, hitState.closestBitangent,
            hitState.closestTangent1, hitState.closestBitangent1, hitState.closestLightIndex, instance.materialOverride);
    } else {
        for (uint32_t i = 0; i < blas.triangleCount; ++i) {
            intersectTriangleHit(localRay, blas.triangles[i], scene, hitState.closestT,
                hitState.closestMaterial, hitState.closestGeometricNormal, hitState.closestNormal,
                hitState.closestUv, hitState.closestUv1, hitState.closestTangent, hitState.closestBitangent,
                hitState.closestTangent1, hitState.closestBitangent1, hitState.closestLightIndex, instance.materialOverride);
        }
    }
    if (hitState.closestT >= previousClosestT)
        return;
    hitState.closestGeometricNormal = transformNormal(instance.transform, hitState.closestGeometricNormal);
    hitState.closestNormal = transformNormal(instance.transform, hitState.closestNormal);
    hitState.closestTangent = normalize(transformVector(instance.transform, hitState.closestTangent));
    hitState.closestBitangent = normalize(transformVector(instance.transform, hitState.closestBitangent));
    hitState.closestTangent1 = normalize(transformVector(instance.transform, hitState.closestTangent1));
    hitState.closestBitangent1 = normalize(transformVector(instance.transform, hitState.closestBitangent1));
    if (hitState.closestLightIndex != invalidLightIndex)
        hitState.closestLightIndex += instance.lightOffset;
    hitState.closestSphereIndex = 0xffffffffu;
}

__device__ __forceinline__
void intersectSceneTlas(const Ray& ray, Scene scene, TraceHitState& hitState) {
    Vec3 inverseDirection(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
    float& closestT = hitState.closestT;
    if (scene.tlas.nodes != nullptr) {
        constexpr uint32_t maxTlasStackSize = 64;
        uint32_t stack[maxTlasStackSize];
        uint32_t stackSize = 1;
        stack[0] = scene.tlas.nodeCount - 1;

        while (stackSize > 0) {
            const NXB::BVH2::Node& node = scene.tlas.nodes[stack[--stackSize]];
            float nearT;

            if (!intersectNexusAabb(ray, inverseDirection, node.bounds, closestT, nearT))
                continue;

            if (node.leftChild == NXB::InvalidIdx) {
                const SceneLeaf& leaf = scene.sceneLeaves[node.rightChild];
                if (leaf.type == SceneLeafType::Sphere)
                    intersectSphereHit(ray, scene.spheres[leaf.index], leaf.index, hitState);
                else
                    intersectMeshInstance(ray, leaf.index, scene, hitState, true);
            } else {
                float leftNearT;
                float rightNearT;
                bool hitLeft = intersectNexusAabb(ray, inverseDirection, scene.tlas.nodes[node.leftChild].bounds, closestT, leftNearT);
                bool hitRight = intersectNexusAabb(ray, inverseDirection, scene.tlas.nodes[node.rightChild].bounds, closestT, rightNearT);

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
}

__device__ __forceinline__
void intersectDirectTriangles(const Ray& ray, Scene scene, TraceHitState& hitState) {
    float& closestT = hitState.closestT;
    int& closestMaterial = hitState.closestMaterial;
    Vec3& closestGeometricNormal = hitState.closestGeometricNormal;
    Vec3& closestNormal = hitState.closestNormal;
    Vec2& closestUv = hitState.closestUv;
    Vec2& closestUv1 = hitState.closestUv1;
    Vec3& closestTangent = hitState.closestTangent;
    Vec3& closestBitangent = hitState.closestBitangent;
    Vec3& closestTangent1 = hitState.closestTangent1;
    Vec3& closestBitangent1 = hitState.closestBitangent1;
    uint32_t& closestLightIndex = hitState.closestLightIndex;
    if (scene.nexusBvh8.nodes != nullptr) {
        constexpr uint32_t maxBvhStackSize = 96;
        struct StackEntry {
            uint32_t index;
            uint32_t triangleCount;
        };
        StackEntry stack[maxBvhStackSize];
        uint32_t stackSize = 1;
        stack[0] = StackEntry{0, 0};
        Vec3 inverseDirection(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);

        while (stackSize > 0) {
            StackEntry entry = stack[--stackSize];

            if (entry.triangleCount > 0) {
                for (uint32_t i = 0; i < entry.triangleCount; ++i) {
                    const Triangle& triangle = scene.triangles[scene.nexusBvh8.primIdx[entry.index + i]];
                    intersectTriangleHit(ray, triangle, scene, closestT, closestMaterial, closestGeometricNormal, closestNormal, closestUv, closestUv1, closestTangent, closestBitangent, closestTangent1, closestBitangent1, closestLightIndex);
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
                if (!intersectNexusBvh8ChildAabb(ray, inverseDirection, node, cellSize, slot, closestT, nearT))
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

            if (!intersectNexusAabb(ray, node.bounds, closestT, nearT))
                continue;

            if (node.leftChild == NXB::InvalidIdx) {
                const Triangle& triangle = scene.triangles[node.rightChild];
                intersectTriangleHit(ray, triangle, scene, closestT, closestMaterial, closestGeometricNormal, closestNormal, closestUv, closestUv1, closestTangent, closestBitangent, closestTangent1, closestBitangent1, closestLightIndex);
            } else {
                float leftNearT;
                float rightNearT;
                bool hitLeft = intersectNexusAabb(ray, scene.nexusBvh.nodes[node.leftChild].bounds, closestT, leftNearT);
                bool hitRight = intersectNexusAabb(ray, scene.nexusBvh.nodes[node.rightChild].bounds, closestT, rightNearT);

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

            if (!intersectAabb(ray, node.bounds, closestT, nearT))
                continue;

            if (node.triangleCount > 0) {
                for (uint32_t i = 0; i < node.triangleCount; ++i) {
                    const Triangle& triangle = scene.triangles[scene.bvhTriangleIndices[node.firstTriangle + i]];
                    intersectTriangleHit(ray, triangle, scene, closestT, closestMaterial, closestGeometricNormal, closestNormal, closestUv, closestUv1, closestTangent, closestBitangent, closestTangent1, closestBitangent1, closestLightIndex);
                }
            } else {
                float leftNearT;
                float rightNearT;
                bool hitLeft = intersectAabb(ray, scene.bvhNodes[node.leftChild].bounds, closestT, leftNearT);
                bool hitRight = intersectAabb(ray, scene.bvhNodes[node.rightChild].bounds, closestT, rightNearT);

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
            intersectTriangleHit(ray, scene.triangles[i], scene, closestT, closestMaterial, closestGeometricNormal, closestNormal, closestUv, closestUv1, closestTangent, closestBitangent, closestTangent1, closestBitangent1, closestLightIndex);
        }
    }
}

__device__
bool isOccluded(const Ray& ray, float maximumDistance, Scene scene) {
    for (uint32_t i = 0; i < scene.directSphereCount; ++i) {
        float t;
        if (intersectSphere(ray, scene.spheres[scene.directSphereIndices[i]], t) && t < maximumDistance)
            return true;
    }
    for (uint32_t i = 0; i < scene.directInstanceCount; ++i) {
        TraceHitState instanceHit;
        instanceHit.closestT = maximumDistance;
        intersectMeshInstance(ray, scene.directInstanceIndices[i], scene, instanceHit, false);
        if (instanceHit.closestMaterial >= 0)
            return true;
    }
    for (uint32_t i = 0; i < scene.staticTriangleCount; ++i) {
        const Triangle& triangle = scene.staticTriangles[i];
        float t, u, v;
        if (intersectTriangle(ray, triangle, t, u, v) && t < maximumDistance) {
            const Material& material = scene.materials[triangle.material];
            if (!material.alphaMasked || getMaterialOpacity(material,
                getTriangleUv(triangle, u, v, false), getTriangleUv(triangle, u, v, true), scene) >= material.alphaCutoff)
                return true;
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

            if (!intersectNexusAabb(ray, node.bounds, maximumDistance, nearT))
                continue;

            if (node.leftChild == NXB::InvalidIdx) {
                const SceneLeaf& leaf = scene.sceneLeaves[node.rightChild];
                if (leaf.type == SceneLeafType::Sphere) {
                    float t;
                    if (intersectSphere(ray, scene.spheres[leaf.index], t) && t < maximumDistance)
                        return true;
                } else {
                    TraceHitState instanceHit;
                    instanceHit.closestT = maximumDistance;
                    intersectMeshInstance(ray, leaf.index, scene, instanceHit, true);
                    if (instanceHit.closestMaterial >= 0)
                        return true;
                }
            } else {
                float leftNearT;
                float rightNearT;
                bool hitLeft = intersectNexusAabb(ray, scene.tlas.nodes[node.leftChild].bounds, maximumDistance, leftNearT);
                bool hitRight = intersectNexusAabb(ray, scene.tlas.nodes[node.rightChild].bounds, maximumDistance, rightNearT);

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

    if (scene.triangleCount > 0) {
        TraceHitState triangleHit;
        triangleHit.closestT = maximumDistance;
        intersectDirectTriangles(ray, scene, triangleHit);
        if (triangleHit.closestMaterial >= 0)
            return true;
    }
    return false;
}

__device__ __forceinline__ TraceResult traceRay(const Ray& ray, Scene scene) {
    RayWorkItem work{ray, 0};
    TraceHitState hitState;
    float& closestT = hitState.closestT;
    int& closestMaterial = hitState.closestMaterial;
    Vec3& closestGeometricNormal = hitState.closestGeometricNormal;
    Vec3& closestNormal = hitState.closestNormal;
    Vec2& closestUv = hitState.closestUv;
    Vec2& closestUv1 = hitState.closestUv1;
    Vec3& closestTangent = hitState.closestTangent;
    Vec3& closestBitangent = hitState.closestBitangent;
    Vec3& closestTangent1 = hitState.closestTangent1;
    Vec3& closestBitangent1 = hitState.closestBitangent1;
    uint32_t& closestLightIndex = hitState.closestLightIndex;
    for (uint32_t i = 0; i < scene.directSphereCount; ++i) {
        uint32_t sphereIndex = scene.directSphereIndices[i];
        intersectSphereHit(ray, scene.spheres[sphereIndex], sphereIndex, hitState);
    }
    for (uint32_t i = 0; i < scene.directInstanceCount; ++i)
        intersectMeshInstance(ray, scene.directInstanceIndices[i], scene, hitState, false);
    for (uint32_t i = 0; i < scene.staticTriangleCount; ++i) {
        const Triangle& triangle = scene.staticTriangles[i];
        if (intersectTriangleHit(ray, triangle, scene, closestT, closestMaterial, closestGeometricNormal,
            closestNormal, closestUv, closestUv1, closestTangent, closestBitangent,
            closestTangent1, closestBitangent1, closestLightIndex))
            hitState.closestSphereIndex = 0xffffffffu;
    }
    intersectSceneTlas(ray, scene, hitState);
    // A sphere TLAS may coexist with any of the direct triangle accelerators.
    if (scene.triangleCount > 0) {
        float previousClosestT = closestT;
        intersectDirectTriangles(ray, scene, hitState);
        if (closestT < previousClosestT)
            hitState.closestSphereIndex = 0xffffffffu;
    }
    // Sphere candidates need only a distance and index during traversal.
    // Compute shading attributes once, if a sphere is the final winner.
    if (hitState.closestSphereIndex != 0xffffffffu) {
        const Sphere& sphere = scene.spheres[hitState.closestSphereIndex];
        closestNormal = normalize(ray.at(closestT) - sphere.center);
        closestGeometricNormal = closestNormal;
        closestUv = closestUv1 = Vec2{0.0f, 0.0f};
        closestTangent = closestBitangent = Vec3(0.0f, 0.0f, 0.0f);
        closestTangent1 = closestBitangent1 = Vec3(0.0f, 0.0f, 0.0f);
        closestLightIndex = invalidLightIndex;
    }

    if (closestMaterial < 0) {
        return TraceResult{Hit{}, false};
    }

    Vec3 position = work.ray.at(closestT);

    Hit hit;

    hit.t = closestT;
    hit.position = position;
    hit.geometricNormal = closestGeometricNormal;
    hit.normal = closestNormal;
    hit.uv = closestUv;
    hit.uv1 = closestUv1;
    hit.tangent = closestTangent;
    hit.bitangent = closestBitangent;
    hit.tangent1 = closestTangent1;
    hit.bitangent1 = closestBitangent1;
    hit.material = static_cast<uint32_t>(closestMaterial);
    hit.lightIndex = closestLightIndex;

    return TraceResult{hit, true};
}

__global__
void intersectScene(RayQueue rays, IntersectionResult* results, Scene scene) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t rayCount = *rays.count;
    if (index >= rayCount)
        return;
    TraceResult result = traceRay(rays.items[index].ray, scene);
    results[index] = IntersectionResult{result.hit, result.didHit};
}

__device__ TraceResult traceRayExternal(const Ray& ray, Scene scene) {
    return traceRay(ray, scene);
}
