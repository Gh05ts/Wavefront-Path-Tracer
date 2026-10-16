#include "renderer/renderer.cuh"
#include "scene/scene_host.hpp"
#include "core/rng.cuh"
#include "renderer/device_sampling.cuh"
#include "renderer/device_traversal.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace {
void cudaCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "scene_acceleration_tests: " << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

bool check(bool condition, const char* message) {
    if (!condition)
        std::cerr << "scene_acceleration_tests: " << message << '\n';
    return condition;
}

template <typename T>
std::vector<T> download(const T* pointer, size_t count) {
    std::vector<T> values(count);
    if (count != 0)
        cudaCheck(cudaMemcpy(values.data(), pointer, sizeof(T) * count, cudaMemcpyDeviceToHost));
    return values;
}

struct ExpectedHit {
    double t = 1e30;
    uint32_t material = invalidMaterialIndex;
    Vec3 normal;
    bool sphere = false;
};

// Independent host reference: scan all geometry, ignoring every placement policy.
void referenceSphere(const Ray& ray, const Sphere& sphere, ExpectedHit& result) {
    double x = double(ray.origin.x) - sphere.center.x;
    double y = double(ray.origin.y) - sphere.center.y;
    double z = double(ray.origin.z) - sphere.center.z;
    double a = double(ray.direction.x) * ray.direction.x + double(ray.direction.y) * ray.direction.y + double(ray.direction.z) * ray.direction.z;
    double b = x * ray.direction.x + y * ray.direction.y + z * ray.direction.z;
    double c = x*x + y*y + z*z - double(sphere.radius) * sphere.radius;
    double discriminant = b*b - a*c;
    if (discriminant < 0)
        return;
    double root = std::sqrt(discriminant);
    double t = (-b - root) / a;
    if (t <= 0.001)
        t = (-b + root) / a;
    if (t <= 0.001 || t >= result.t)
        return;
    result.t = t;
    result.material = sphere.material;
    result.normal = normalize(ray.at(static_cast<float>(t)) - sphere.center);
    result.sphere = true;
}

void referenceTriangle(const Ray& ray, const Triangle& triangle, const HostScene& scene,
    ExpectedHit& result, uint32_t overrideMaterial = invalidMaterialIndex,
    const InstanceTransform* transform = nullptr) {
    Ray localRay = ray;
    if (transform != nullptr) {
        localRay.origin = inverseTransformVector(*transform, ray.origin - transform->translation);
        localRay.direction = inverseTransformVector(*transform, ray.direction);
    }
    Vec3 e1 = triangle.v1 - triangle.v0;
    Vec3 e2 = triangle.v2 - triangle.v0;
    Vec3 p = cross(localRay.direction, e2);
    float determinant = dot(e1, p);
    if (std::fabs(determinant) < 1e-8f)
        return;
    Vec3 offset = localRay.origin - triangle.v0;
    float u = dot(offset, p) / determinant;
    Vec3 q = cross(offset, e1);
    float v = dot(localRay.direction, q) / determinant;
    float t = dot(e2, q) / determinant;
    if (u < 0 || v < 0 || u + v > 1 || t <= 0.001f || t >= result.t)
        return;
    uint32_t materialIndex = overrideMaterial == invalidMaterialIndex ? triangle.material : overrideMaterial;
    const Material& material = scene.materials[materialIndex];
    if (material.alphaMasked) {
        float alpha = 1;
        if (material.albedoTexture.texture != invalidTextureIndex)
            alpha = scene.textures[material.albedoTexture.texture].pixels[3] / 255.0f;
        if (alpha < material.alphaCutoff)
            return;
    }
    Vec3 normal = triangle.hasVertexNormals
        ? normalize(triangle.n0 * (1-u-v) + triangle.n1*u + triangle.n2*v)
        : normalize(cross(e1, e2));
    if (dot(normal, localRay.direction) > 0)
        normal = -normal;
    result.t = t;
    result.material = materialIndex;
    result.normal = transform == nullptr ? normal : transformNormal(*transform, normal);
    result.sphere = false;
}

ExpectedHit reference(const HostScene& scene, const Ray& ray) {
    ExpectedHit result;
    for (const SceneSphere& sphere : scene.spheres)
        referenceSphere(ray, sphere.geometry, result);
    for (const Triangle& triangle : scene.staticTriangles)
        referenceTriangle(ray, triangle, scene, result);
    for (const Triangle& triangle : scene.triangles)
        referenceTriangle(ray, triangle, scene, result);
    for (const SceneInstance& instance : scene.instances) {
        for (const Triangle& triangle : scene.meshes[instance.meshIndex].triangles)
            referenceTriangle(ray, triangle, scene, result, instance.materialOverride, &instance.transform);
    }
    return result;
}

struct Probe {
    TraceResult result;
    bool blockedBefore;
    bool blockedAfter;
};

__global__ void probeScene(Scene scene, const Ray* rays, const float* limits, Probe* output, uint32_t count) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    output[index].result = traceRay(rays[index], scene);
    output[index].blockedBefore = isOccluded(rays[index], limits[index] - 0.02f, scene);
    output[index].blockedAfter = isOccluded(rays[index], limits[index] + 0.02f, scene);
}

bool checkPartition(const HostScene& host, const DeviceScene& device) {
    std::vector<uint32_t> sphereUses(host.spheres.size());
    std::vector<uint32_t> instanceUses(host.instances.size());
    bool valid = true;
    for (uint32_t index : download(device.directSphereIndices, device.scene.directSphereCount)) {
        if (!check(index < host.spheres.size(), "invalid direct sphere index")) return false;
        ++sphereUses[index];
        valid &= check(host.spheres[index].acceleration == AccelerationMode::Direct, "accelerated sphere was tested directly");
    }
    for (uint32_t index : download(device.directInstanceIndices, device.scene.directInstanceCount)) {
        if (!check(index < host.instances.size(), "invalid direct instance index")) return false;
        ++instanceUses[index];
        valid &= check(host.instances[index].acceleration == AccelerationMode::Direct, "accelerated instance was tested directly");
    }
    for (const SceneLeaf& leaf : download(device.sceneLeaves, device.scene.tlas.primCount)) {
        if (leaf.type == SceneLeafType::Sphere) {
            if (!check(leaf.index < host.spheres.size(), "invalid sphere leaf")) return false;
            ++sphereUses[leaf.index];
            valid &= check(host.spheres[leaf.index].acceleration == AccelerationMode::Bvh, "direct sphere entered TLAS");
        } else {
            if (!check(leaf.index < host.instances.size(), "invalid instance leaf")) return false;
            ++instanceUses[leaf.index];
            valid &= check(host.instances[leaf.index].acceleration == AccelerationMode::Bvh, "direct instance entered TLAS");
        }
    }
    for (uint32_t count : sphereUses)
        valid &= check(count == 1, "sphere must appear in exactly one traversal list");
    for (size_t index = 0; index < instanceUses.size(); ++index) {
        const SceneInstance& instance = host.instances[index];
        uint32_t expected = host.meshes[instance.meshIndex].triangles.empty() &&
            instance.acceleration == AccelerationMode::Bvh ? 0u : 1u;
        valid &= check(instanceUses[index] == expected, "nonempty instance must appear in exactly one traversal list");
    }
    for (size_t meshIndex = 0; meshIndex < host.meshes.size(); ++meshIndex) {
        bool needsBvh = false;
        for (const SceneInstance& instance : host.instances)
            needsBvh |= instance.meshIndex == meshIndex && instance.acceleration == AccelerationMode::Bvh &&
                !host.meshes[meshIndex].triangles.empty();
        valid &= check((device.blasBvhs[meshIndex].View().nodes != nullptr) == needsBvh,
            "BLAS must be built only for meshes with accelerated instances");
    }
    return valid;
}

bool runCase(const char* name, const HostScene& host) {
    DeviceScene device = uploadScene(host);
    bool valid = checkPartition(host, device);
    std::vector<Ray> rays = {
        Ray(Vec3(0,0,0), Vec3(0,0,1)),
        Ray(Vec3(1,0,0), Vec3(0,0,1)), // tangent to the unit sphere
        Ray(Vec3(0,0,3), Vec3(1,0,0)), // inside the unit sphere
        Ray(Vec3(0,0,0), Vec3(0,-1,0)),
        Ray(Vec3(0,0,0), Vec3(0,0,-1)),
        Ray(Vec3(-3,0,0), Vec3(0,0,1)),
        Ray(Vec3(3,0,0), Vec3(0,0,1)),
        Ray(Vec3(0,0,8), Vec3(0,0,-1))
    };
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> position(-4, 4);
    std::uniform_real_distribution<float> direction(-1, 1);
    for (uint32_t i = 0; i < 2048; ++i) {
        Vec3 origin(position(rng), position(rng), position(rng));
        Vec3 vector(direction(rng), direction(rng), direction(rng));
        rays.emplace_back(origin, normalize(vector));
    }
    std::vector<ExpectedHit> expected;
    std::vector<float> limits;
    for (const Ray& ray : rays) {
        expected.push_back(reference(host, ray));
        limits.push_back(expected.back().material == invalidMaterialIndex ? 1000.0f : static_cast<float>(expected.back().t));
    }
    Ray* deviceRays = nullptr;
    float* deviceLimits = nullptr;
    Probe* deviceOutput = nullptr;
    cudaCheck(cudaMalloc(&deviceRays, sizeof(Ray) * rays.size()));
    cudaCheck(cudaMalloc(&deviceLimits, sizeof(float) * rays.size()));
    cudaCheck(cudaMalloc(&deviceOutput, sizeof(Probe) * rays.size()));
    cudaCheck(cudaMemcpy(deviceRays, rays.data(), sizeof(Ray) * rays.size(), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(deviceLimits, limits.data(), sizeof(float) * rays.size(), cudaMemcpyHostToDevice));
    probeScene<<<(rays.size()+127)/128, 128>>>(device.scene, deviceRays, deviceLimits, deviceOutput, rays.size());
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    auto output = download(deviceOutput, rays.size());
    for (size_t i = 0; i < rays.size(); ++i) {
        bool hit = expected[i].material != invalidMaterialIndex;
        bool correct = output[i].result.didHit == hit && !output[i].blockedBefore && output[i].blockedAfter == hit;
        if (hit && output[i].result.didHit) {
            const Hit& actual = output[i].result.hit;
            correct &= actual.material == expected[i].material;
            correct &= std::fabs(actual.t - expected[i].t) < 0.0003 * std::max(1.0, expected[i].t);
            correct &= length(actual.normal - expected[i].normal) < 0.003f;
            if (expected[i].sphere) {
                correct &= actual.uv.x == 0 && actual.uv.y == 0 && actual.uv1.x == 0 && actual.uv1.y == 0;
                correct &= lengthSquared(actual.tangent) == 0 && lengthSquared(actual.bitangent) == 0;
                correct &= lengthSquared(actual.tangent1) == 0 && lengthSquared(actual.bitangent1) == 0;
                correct &= actual.lightIndex == invalidLightIndex;
            }
        }
        if (!correct) {
            std::cerr << name << ": ray " << i << " disagrees with reference; expected material "
                      << expected[i].material << " at " << expected[i].t << ", actual hit "
                      << output[i].result.didHit << " material " << output[i].result.hit.material
                      << " at " << output[i].result.hit.t << ", shadows "
                      << output[i].blockedBefore << '/' << output[i].blockedAfter << '\n';
            valid = false;
            break;
        }
    }
    cudaCheck(cudaFree(deviceRays));
    cudaCheck(cudaFree(deviceLimits));
    cudaCheck(cudaFree(deviceOutput));
    destroyDeviceScene(device);
    std::cout << name << ": " << (valid ? "passed" : "FAILED") << '\n';
    return valid;
}

Triangle triangleAt(float z, uint32_t material) {
    Triangle triangle{};
    triangle.v0 = Vec3(-2,-1,z);
    triangle.v1 = Vec3(2,-1,z);
    triangle.v2 = Vec3(0,2,z);
    triangle.material = material;
    triangle.lightIndex = invalidLightIndex;
    triangle.tangent = triangle.tangent1 = Vec3(1,0,0);
    triangle.bitangent = triangle.bitangent1 = Vec3(0,1,0);
    triangle.hasTexcoords = true;
    triangle.uv0 = Vec2{0,0};
    triangle.uv1 = Vec2{1,0};
    triangle.uv2 = Vec2{0.5f,1};
    return triangle;
}

HostScene baseScene() {
    HostScene scene;
    for (uint32_t i = 0; i < 4; ++i) {
        Material material{};
        material.type = MaterialType::Diffuse;
        material.albedo = Vec3(0.5f,0.5f,0.5f);
        scene.materials.push_back(material);
    }
    scene.textures.push_back(ObjTexture{1, 1, {255,255,255,0}});
    scene.materials[3].alphaMasked = true;
    scene.materials[3].albedoTexture.texture = 0;
    return scene;
}
}

int main() {
    int deviceCount = 0;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || deviceCount == 0) {
        std::cout << "scene_acceleration_tests: skipped (no CUDA device)\n";
        return 77;
    }
    cudaCheck(status);
    bool valid = runCase("empty scene", baseScene());
    HostScene spheres = baseScene();
    spheres.spheres = {Sphere{Vec3(0,0,3),1,0},
        SceneSphere{Sphere{Vec3(-3,0,3),0.5f,1}, AccelerationMode::Direct},
        SceneSphere{Sphere{Vec3(0,-21,0),20,2}, AccelerationMode::Direct}};
    valid &= runCase("sphere partitions", spheres);
    HostScene allDirect = spheres;
    for (SceneSphere& sphere : allDirect.spheres)
        sphere.acceleration = AccelerationMode::Direct;
    valid &= runCase("all direct spheres", allDirect);
    HostScene allBvh = spheres;
    for (SceneSphere& sphere : allBvh.spheres)
        sphere.acceleration = AccelerationMode::Bvh;
    valid &= runCase("all accelerated spheres", allBvh);
    spheres.spheres.emplace_back(Sphere{Vec3(0,0,3),1,1}, AccelerationMode::Direct);
    valid &= runCase("coincident spheres across partitions", spheres);

    HostScene field = baseScene();
    field.spheres.emplace_back(Sphere{Vec3(0,-21,0),20,2}, AccelerationMode::Direct);
    for (uint32_t x = 0; x < 12; ++x)
        for (uint32_t z = 0; z < 12; ++z)
            for (uint32_t layer = 0; layer < 2; ++layer)
                field.spheres.emplace_back(Sphere{
                    Vec3(-3.3f + 0.6f*x, -0.3f + 0.6f*layer, 1.0f + 0.6f*z),
                    0.15f, (x+z+layer)%3});
    valid &= runCase("large sphere hierarchy", field);

    HostScene mixed = spheres;
    mixed.meshes.push_back(MeshAsset{{triangleAt(1,3), triangleAt(5,2)}});
    mixed.instances.push_back(SceneInstance{0, makeScaledTransform(Vec3(0,0,0),1)});
    mixed.instances.push_back(SceneInstance{0, makeScaledTransform(Vec3(4,0,1),0.5f), 1, AccelerationMode::Direct});
    mixed.staticTriangles.push_back(triangleAt(0.5f,3));
    valid &= runCase("mixed geometry and shared mesh", mixed);
    mixed.triangles.push_back(triangleAt(7,2));
    valid &= runCase("mesh TLAS with additional flat triangles", mixed);

    HostScene directMesh = baseScene();
    directMesh.meshes = mixed.meshes;
    directMesh.instances.push_back(SceneInstance{0, makeScaledTransform(Vec3(0,0,0),1), 1, AccelerationMode::Direct});
    valid &= runCase("direct mesh without BLAS or TLAS", directMesh);
    directMesh.instances[0].acceleration = AccelerationMode::Bvh;
    valid &= runCase("accelerated mesh without spheres", directMesh);
    HostScene emptyMesh = baseScene();
    emptyMesh.meshes.push_back(MeshAsset{});
    emptyMesh.instances.push_back(SceneInstance{0, makeScaledTransform(Vec3(0,0,0),1)});
    valid &= runCase("empty accelerated mesh", emptyMesh);
    emptyMesh.instances[0].acceleration = AccelerationMode::Direct;
    valid &= runCase("empty direct mesh", emptyMesh);

    HostScene resetAttributes = baseScene();
    resetAttributes.spheres.emplace_back(Sphere{Vec3(0,0,3),1,0});
    resetAttributes.meshes.push_back(MeshAsset{{triangleAt(5,2)}});
    resetAttributes.instances.push_back(SceneInstance{0, makeScaledTransform(Vec3(0,0,0),1), invalidMaterialIndex, AccelerationMode::Direct});
    valid &= runCase("sphere replaces direct mesh attributes", resetAttributes);

    for (ObjAccelerationPolicy policy : {ObjAccelerationPolicy::NexusBvh2, ObjAccelerationPolicy::NexusBvh8,
            ObjAccelerationPolicy::MedianBvh, ObjAccelerationPolicy::SpatialSplitBvh}) {
        HostScene directTriangles = spheres;
        directTriangles.acceleration = policy;
        directTriangles.triangles = {triangleAt(1,3), triangleAt(5,2)};
        valid &= runCase("sphere TLAS with alternate triangle BVH", directTriangles);
    }
    return valid ? 0 : 1;
}
