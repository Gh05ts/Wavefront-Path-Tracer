#include "scene/scene.cuh"
#include "scene/scene_host.hpp"
#include "scene/bvh.cuh"

#include <NXB/BVHBuilder.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace
{
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

template <typename T>
void freeDevice(T*& pointer) {
    if (pointer == nullptr)
        return;
    checkCuda(cudaFree(pointer));
    pointer = nullptr;
}

template <typename T>
void uploadVector(T*& devicePointer, const std::vector<T>& values) {
    if (values.empty())
        return;
    checkCuda(cudaMalloc(&devicePointer, sizeof(T) * values.size()));
    checkCuda(cudaMemcpy(devicePointer, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice));
}

void uploadTextures(DeviceScene& deviceScene, const std::vector<ObjTexture>& hostTextures) {
    if (hostTextures.empty())
        return;

    std::vector<Texture> hostTextureViews;
    hostTextureViews.reserve(hostTextures.size());
    deviceScene.textureArrays.reserve(hostTextures.size());
    deviceScene.textureObjects.reserve(hostTextures.size());

    for (const ObjTexture& hostTexture : hostTextures) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray_t array = nullptr;
        checkCuda(cudaMallocArray(&array, &channelDesc, hostTexture.width, hostTexture.height));
        checkCuda(cudaMemcpy2DToArray(array, 0, 0, hostTexture.pixels.data(), hostTexture.width * sizeof(uchar4), hostTexture.width * sizeof(uchar4), hostTexture.height, cudaMemcpyHostToDevice));

        cudaResourceDesc resourceDesc{};
        resourceDesc.resType = cudaResourceTypeArray;
        resourceDesc.res.array.array = array;

        cudaTextureDesc textureDesc{};
        textureDesc.addressMode[0] = cudaAddressModeWrap;
        textureDesc.addressMode[1] = cudaAddressModeWrap;
        textureDesc.filterMode = cudaFilterModeLinear;
        textureDesc.readMode = cudaReadModeNormalizedFloat;
        textureDesc.normalizedCoords = 1;

        cudaTextureObject_t textureObject = 0;
        checkCuda(cudaCreateTextureObject(&textureObject, &resourceDesc, &textureDesc, nullptr));
        deviceScene.textureArrays.push_back(array);
        deviceScene.textureObjects.push_back(textureObject);
        hostTextureViews.push_back(Texture{textureObject});
    }

    uploadVector(deviceScene.textures, hostTextureViews);
    deviceScene.scene.textures = deviceScene.textures;
    deviceScene.scene.textureCount = static_cast<uint32_t>(hostTextureViews.size());
}

void uploadTriangleLights(DeviceScene& deviceScene) {
    if (deviceScene.hostLights.empty())
        return;

    float totalWeight = 0.0f;
    for (float weight : deviceScene.lightWeights)
        totalWeight += weight;

    std::vector<LightAliasEntry> alias(deviceScene.hostLights.size());
    std::vector<float> scaled(deviceScene.hostLights.size());
    std::vector<uint32_t> small;
    std::vector<uint32_t> large;

    for (uint32_t i = 0; i < deviceScene.hostLights.size(); ++i) {
        deviceScene.hostLights[i].selectionPdf = deviceScene.lightWeights[i] / totalWeight;
        scaled[i] = deviceScene.hostLights[i].selectionPdf * deviceScene.hostLights.size();
        (scaled[i] < 1.0f ? small : large).push_back(i);
    }

    while (!small.empty() && !large.empty()) {
        uint32_t low = small.back();
        small.pop_back();
        uint32_t high = large.back();
        large.pop_back();
        alias[low] = LightAliasEntry{scaled[low], high};
        scaled[high] = scaled[high] + scaled[low] - 1.0f;
        (scaled[high] < 1.0f ? small : large).push_back(high);
    }

    for (uint32_t index : small)
        alias[index] = LightAliasEntry{1.0f, index};
    for (uint32_t index : large)
        alias[index] = LightAliasEntry{1.0f, index};

    uploadVector(deviceScene.lights, deviceScene.hostLights);
    uploadVector(deviceScene.lightAlias, alias);
    deviceScene.scene.lights = deviceScene.lights;
    deviceScene.scene.lightAlias = deviceScene.lightAlias;
    deviceScene.scene.lightCount = static_cast<uint32_t>(deviceScene.hostLights.size());
}

void uploadMeshInstances(DeviceScene& deviceScene, const HostScene& hostScene) {
    std::vector<Blas> hostBlases(hostScene.meshes.size());
    deviceScene.meshTriangles.reserve(hostScene.meshes.size());
    deviceScene.blasBvhs.reserve(hostScene.meshes.size());

    for (uint32_t meshIndex = 0; meshIndex < hostScene.meshes.size(); ++meshIndex) {
        const MeshAsset& mesh = hostScene.meshes[meshIndex];
        Triangle* deviceTriangles = nullptr;
        uploadVector(deviceTriangles, mesh.triangles);
        deviceScene.meshTriangles.push_back(deviceTriangles);

        std::vector<NXB::Triangle> nexusTriangles;
        nexusTriangles.reserve(mesh.triangles.size());
        for (const Triangle& triangle : mesh.triangles) {
            nexusTriangles.emplace_back(
                make_float3(triangle.v0.x, triangle.v0.y, triangle.v0.z),
                make_float3(triangle.v1.x, triangle.v1.y, triangle.v1.z),
                make_float3(triangle.v2.x, triangle.v2.y, triangle.v2.z));
        }

        NXB::DeviceBuffer<NXB::Triangle> deviceNexusTriangles(nexusTriangles);
        deviceScene.blasBvhs.push_back(NXB::BuildBVH2(deviceNexusTriangles.Get(), static_cast<uint32_t>(nexusTriangles.size())));
        hostBlases[meshIndex] = Blas{deviceTriangles, deviceScene.blasBvhs.back().View()};
    }

    std::vector<NXB::AABB> instanceBounds;
    std::vector<MeshInstance> hostInstances;
    instanceBounds.reserve(hostScene.instances.size());
    hostInstances.reserve(hostScene.instances.size());

    uint32_t lightOffset = 0;
    for (const SceneInstance& instance : hostScene.instances) {
        const NXB::AABB& localBounds = deviceScene.blasBvhs[instance.meshIndex].Bounds();
        NXB::AABB worldBounds;
        worldBounds.bMin = make_float3(1e30f, 1e30f, 1e30f);
        worldBounds.bMax = make_float3(-1e30f, -1e30f, -1e30f);

        for (uint32_t corner = 0; corner < 8; ++corner) {
            Vec3 localPoint(
                (corner & 1) ? localBounds.bMax.x : localBounds.bMin.x,
                (corner & 2) ? localBounds.bMax.y : localBounds.bMin.y,
                (corner & 4) ? localBounds.bMax.z : localBounds.bMin.z);
            Vec3 worldPoint = transformVector(instance.transform, localPoint) + instance.transform.translation;
            worldBounds.bMin.x = fminf(worldBounds.bMin.x, worldPoint.x);
            worldBounds.bMin.y = fminf(worldBounds.bMin.y, worldPoint.y);
            worldBounds.bMin.z = fminf(worldBounds.bMin.z, worldPoint.z);
            worldBounds.bMax.x = fmaxf(worldBounds.bMax.x, worldPoint.x);
            worldBounds.bMax.y = fmaxf(worldBounds.bMax.y, worldPoint.y);
            worldBounds.bMax.z = fmaxf(worldBounds.bMax.z, worldPoint.z);
        }

        hostInstances.push_back(MeshInstance{instance.meshIndex, lightOffset, instance.transform});
        for (const Triangle& triangle : hostScene.meshes[instance.meshIndex].triangles) {
            if (triangle.lightIndex != invalidLightIndex)
                ++lightOffset;
        }
        instanceBounds.push_back(worldBounds);
    }

    NXB::DeviceBuffer<NXB::AABB> deviceInstanceBounds(instanceBounds);
    deviceScene.tlas = NXB::BuildBVH2(deviceInstanceBounds.Get(), static_cast<uint32_t>(instanceBounds.size()));

    uploadVector(deviceScene.blases, hostBlases);
    uploadVector(deviceScene.instances, hostInstances);
    deviceScene.scene.tlas = deviceScene.tlas.View();
    deviceScene.scene.blases = deviceScene.blases;
    deviceScene.scene.blasCount = static_cast<uint32_t>(hostBlases.size());
    deviceScene.scene.instances = deviceScene.instances;
    deviceScene.scene.instanceCount = static_cast<uint32_t>(hostInstances.size());
}

void uploadDirectAcceleration(DeviceScene& deviceScene, const HostScene& hostScene) {
    auto bvhBuildStart = std::chrono::steady_clock::now();
    HostBvh hostBvh;
    NXB::BVHBuildMetrics nexusMetrics{};

    const bool useNexusBvh = hostScene.acceleration == ObjAccelerationPolicy::NexusBvh2 ||
        hostScene.acceleration == ObjAccelerationPolicy::NexusBvh8;
    const bool useNexusBvh8 = hostScene.acceleration == ObjAccelerationPolicy::NexusBvh8;

    if (useNexusBvh) {
        std::vector<NXB::Triangle> nexusTriangles;
        nexusTriangles.reserve(hostScene.triangles.size());
        for (const Triangle& triangle : hostScene.triangles) {
            nexusTriangles.emplace_back(
                make_float3(triangle.v0.x, triangle.v0.y, triangle.v0.z),
                make_float3(triangle.v1.x, triangle.v1.y, triangle.v1.z),
                make_float3(triangle.v2.x, triangle.v2.y, triangle.v2.z));
        }

        NXB::DeviceBuffer<NXB::Triangle> deviceNexusTriangles(nexusTriangles);
        if (useNexusBvh8)
            deviceScene.nexusBvh8 = NXB::BuildBVH8(deviceNexusTriangles.Get(), static_cast<uint32_t>(nexusTriangles.size()), NXB::BuildConfig{}, &nexusMetrics);
        else
            deviceScene.nexusBvh = NXB::BuildBVH2(deviceNexusTriangles.Get(), static_cast<uint32_t>(nexusTriangles.size()), NXB::BuildConfig{}, &nexusMetrics);
    } else {
        hostBvh = hostScene.acceleration == ObjAccelerationPolicy::SpatialSplitBvh ?
            buildSpatialSplitBvh(hostScene.triangles) : buildTriangleBvh(hostScene.triangles);
    }

    auto bvhBuildEnd = std::chrono::steady_clock::now();
    float bvhBuildMilliseconds = std::chrono::duration<float, std::milli>(bvhBuildEnd - bvhBuildStart).count();

    if (!useNexusBvh) {
        uploadVector(deviceScene.bvhNodes, hostBvh.nodes);
        uploadVector(deviceScene.bvhTriangleIndices, hostBvh.triangleIndices);
        deviceScene.scene.bvhNodeCount = static_cast<uint32_t>(hostBvh.nodes.size());
    }

    deviceScene.scene.nexusBvh = deviceScene.nexusBvh.View();
    deviceScene.scene.nexusBvh8 = deviceScene.nexusBvh8.View();

    if (hostScene.acceleration == ObjAccelerationPolicy::NexusBvh2 || hostScene.acceleration == ObjAccelerationPolicy::NexusBvh8) {
        if (useNexusBvh8)
            std::cout << "Built NexusBVH H-PLOC BVH8 with " << deviceScene.nexusBvh8.NodeCount() << " nodes (average " << deviceScene.nexusBvh8.AverageChildPerNode() << " children) for " << hostScene.triangles.size() << " triangles in " << bvhBuildMilliseconds << " ms\n";
        else
            std::cout << "Built NexusBVH H-PLOC BVH2 with " << deviceScene.nexusBvh.NodeCount() << " nodes for " << hostScene.triangles.size() << " triangles in " << bvhBuildMilliseconds << " ms\n";
        std::cout << "  bounds " << nexusMetrics.computeSceneBoundsTime << " ms, morton " << nexusMetrics.computeMortonCodesTime << " ms, sort " << nexusMetrics.radixSortTime << " ms, build " << nexusMetrics.bvhBuildTime << " ms\n";
    } else {
        bool useSpatialSplit = hostScene.acceleration == ObjAccelerationPolicy::SpatialSplitBvh;
        std::cout << "Built " << (useSpatialSplit ? "SBVH" : "median BVH") << " with " << hostBvh.nodes.size() << " nodes for " << hostScene.triangles.size() << " triangles in " << bvhBuildMilliseconds << " ms\n";
    }
}

DeviceScene uploadScene(const HostScene& hostScene) {
    DeviceScene deviceScene{};
    deviceScene.hostLights = hostScene.lights;
    deviceScene.lightWeights = hostScene.lightWeights;

    uploadVector(deviceScene.spheres, hostScene.spheres);
    uploadVector(deviceScene.triangles, hostScene.triangles);
    uploadVector(deviceScene.staticTriangles, hostScene.staticTriangles);
    uploadVector(deviceScene.materials, hostScene.materials);
    uploadTextures(deviceScene, hostScene.textures);

    if (!hostScene.meshes.empty()) {
        auto bvhBuildStart = std::chrono::steady_clock::now();
        uploadMeshInstances(deviceScene, hostScene);
        auto bvhBuildEnd = std::chrono::steady_clock::now();
        float bvhBuildMilliseconds = std::chrono::duration<float, std::milli>(bvhBuildEnd - bvhBuildStart).count();
        uint32_t objectTriangleCount = 0;
        for (const MeshAsset& mesh : hostScene.meshes)
            objectTriangleCount += static_cast<uint32_t>(mesh.triangles.size());
        std::cout << "Built mesh-instance TLAS/BLAS BVH2 with " << deviceScene.blasBvhs.size() << " BLASes and " << deviceScene.tlas.NodeCount() << " TLAS nodes for " << objectTriangleCount << " object triangles in " << bvhBuildMilliseconds << " ms\n";
    } else if (!hostScene.triangles.empty()) {
        uploadDirectAcceleration(deviceScene, hostScene);
    }

    uploadTriangleLights(deviceScene);

    deviceScene.scene.spheres = deviceScene.spheres;
    deviceScene.scene.sphereCount = static_cast<uint32_t>(hostScene.spheres.size());
    deviceScene.scene.triangles = deviceScene.triangles;
    deviceScene.scene.triangleCount = static_cast<uint32_t>(hostScene.triangles.size());
    deviceScene.scene.staticTriangles = deviceScene.staticTriangles;
    deviceScene.scene.staticTriangleCount = static_cast<uint32_t>(hostScene.staticTriangles.size());
    deviceScene.scene.materials = deviceScene.materials;
    deviceScene.scene.materialCount = static_cast<uint32_t>(hostScene.materials.size());
    deviceScene.scene.blackBackground = hostScene.blackBackground;
    return deviceScene;
}
} // namespace

DeviceScene createDemoScene() {
    return uploadScene(assembleDemoScene());
}

DeviceScene createObjScene(const ObjSceneOptions& options) {
    return uploadScene(assembleObjScene(options));
}

DeviceScene createCornellScene(const CornellSceneOptions& options) {
    return uploadScene(assembleCornellScene(options));
}

DeviceScene createGltfScene(const char* filename, float sceneScale, bool addTopLight) {
    return uploadScene(assembleGltfScene(filename, sceneScale, addTopLight));
}

void destroyDeviceScene(DeviceScene& deviceScene) {
    freeDevice(deviceScene.spheres);
    freeDevice(deviceScene.triangles);
    freeDevice(deviceScene.staticTriangles);
    for (Triangle* triangles : deviceScene.meshTriangles)
        freeDevice(triangles);
    freeDevice(deviceScene.bvhNodes);
    freeDevice(deviceScene.bvhTriangleIndices);
    deviceScene.nexusBvh = NXB::BVH2{};
    deviceScene.nexusBvh8 = NXB::BVH8{};
    deviceScene.tlas = NXB::BVH2{};
    deviceScene.blasBvhs.clear();
    freeDevice(deviceScene.blases);
    freeDevice(deviceScene.instances);
    freeDevice(deviceScene.materials);
    freeDevice(deviceScene.lights);
    freeDevice(deviceScene.lightAlias);
    for (cudaTextureObject_t textureObject : deviceScene.textureObjects)
        checkCuda(cudaDestroyTextureObject(textureObject));
    for (cudaArray_t textureArray : deviceScene.textureArrays)
        checkCuda(cudaFreeArray(textureArray));
    freeDevice(deviceScene.textures);

    deviceScene = DeviceScene{};
}
