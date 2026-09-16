#include "scene/scene.cuh"
#include "scene/bvh.cuh"
#include "scene/gltf_loader.hpp"
#include "scene/obj_loader.hpp"

#include <NXB/BVHBuilder.h>

#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

namespace
{
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

Triangle makeTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2, uint32_t material) {
    Triangle triangle{};
    triangle.v0 = v0;
    triangle.v1 = v1;
    triangle.v2 = v2;
    triangle.material = material;
    triangle.lightIndex = invalidLightIndex;
    return triangle;
}

void addQuad(std::vector<Triangle>& triangles, const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3, uint32_t material) {
    triangles.push_back(makeTriangle(v0, v1, v2, material));
    triangles.push_back(makeTriangle(v0, v2, v3, material));
}

Material makeMaterial(MaterialType type, const Vec3& albedo, const Vec3& emission = Vec3(0.0f, 0.0f, 0.0f)) {
    Material material{};
    material.type = type;
    material.albedo = albedo;
    material.emission = emission;
    material.roughness = 0.0f;
    material.ior = 1.0f;
    return material;
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

    checkCuda(cudaMalloc(&deviceScene.textures, sizeof(Texture) * hostTextureViews.size()));
    checkCuda(cudaMemcpy(deviceScene.textures, hostTextureViews.data(), sizeof(Texture) * hostTextureViews.size(), cudaMemcpyHostToDevice));
    deviceScene.scene.textures = deviceScene.textures;
    deviceScene.scene.textureCount = static_cast<uint32_t>(hostTextureViews.size());
}

} // namespace

DeviceScene createDemoScene() {
    Sphere hostSpheres[2];

    hostSpheres[0].center = Vec3(0.0f, -100.5f, 0.0f);
    hostSpheres[0].radius = 100.0f;
    hostSpheres[0].material = 3;

    hostSpheres[1].center = Vec3(1.35f, 0.08f, 0.0f);
    hostSpheres[1].radius = 0.5f;
    hostSpheres[1].material = 2;

    Triangle hostTriangles[18];

    const Vec3 cube0(-1.85f, -0.42f, -0.5f);
    const Vec3 cube1(-0.85f, -0.42f, -0.5f);
    const Vec3 cube2(-0.85f,  0.58f, -0.5f);
    const Vec3 cube3(-1.85f,  0.58f, -0.5f);
    const Vec3 cube4(-1.85f, -0.42f,  0.5f);
    const Vec3 cube5(-0.85f, -0.42f,  0.5f);
    const Vec3 cube6(-0.85f,  0.58f,  0.5f);
    const Vec3 cube7(-1.85f,  0.58f,  0.5f);

    hostTriangles[0] = makeTriangle(cube4, cube5, cube6, 0);
    hostTriangles[1] = makeTriangle(cube4, cube6, cube7, 0);
    hostTriangles[2] = makeTriangle(cube0, cube2, cube1, 0);
    hostTriangles[3] = makeTriangle(cube0, cube3, cube2, 0);
    hostTriangles[4] = makeTriangle(cube0, cube4, cube7, 0);
    hostTriangles[5] = makeTriangle(cube0, cube7, cube3, 0);
    hostTriangles[6] = makeTriangle(cube1, cube2, cube6, 0);
    hostTriangles[7] = makeTriangle(cube1, cube6, cube5, 0);
    hostTriangles[8] = makeTriangle(cube3, cube7, cube6, 0);
    hostTriangles[9] = makeTriangle(cube3, cube6, cube2, 0);
    hostTriangles[10] = makeTriangle(cube0, cube1, cube5, 0);
    hostTriangles[11] = makeTriangle(cube0, cube5, cube4, 0);

    const Vec3 pyramid0(-0.183f, -0.42f, -0.683f);
    const Vec3 pyramid1( 0.683f, -0.42f, -0.183f);
    const Vec3 pyramid2( 0.183f, -0.42f,  0.683f);
    const Vec3 pyramid3(-0.683f, -0.42f,  0.183f);
    const Vec3 pyramidTop(0.0f, 0.73f, 0.0f);

    hostTriangles[12] = makeTriangle(pyramid0, pyramid1, pyramidTop, 1);
    hostTriangles[13] = makeTriangle(pyramid1, pyramid2, pyramidTop, 1);
    hostTriangles[14] = makeTriangle(pyramid2, pyramid3, pyramidTop, 1);
    hostTriangles[15] = makeTriangle(pyramid3, pyramid0, pyramidTop, 1);
    hostTriangles[16] = makeTriangle(pyramid0, pyramid2, pyramid1, 1);
    hostTriangles[17] = makeTriangle(pyramid0, pyramid3, pyramid2, 1);

    Material hostMaterials[4];

    hostMaterials[0].type = MaterialType::Diffuse;
    hostMaterials[0].albedo = Vec3(0.7f, 0.2f, 0.15f);
    hostMaterials[0].emission = Vec3(0.0f, 0.0f, 0.0f);
    hostMaterials[0].roughness = 0.0f;
    hostMaterials[0].ior = 1.0f;

    hostMaterials[1].type = MaterialType::Metal;
    hostMaterials[1].albedo = Vec3(0.85f, 0.65f, 0.25f);
    hostMaterials[1].emission = Vec3(0.0f, 0.0f, 0.0f);
    hostMaterials[1].roughness = 0.08f;
    hostMaterials[1].ior = 1.0f;

    hostMaterials[2].type = MaterialType::Dielectric;
    hostMaterials[2].albedo = Vec3(1.0f, 1.0f, 1.0f);
    hostMaterials[2].emission = Vec3(0.0f, 0.0f, 0.0f);
    hostMaterials[2].roughness = 0.0f;
    hostMaterials[2].ior = 1.5f;

    hostMaterials[3].type = MaterialType::Diffuse;
    hostMaterials[3].albedo = Vec3(0.75f, 0.75f, 0.75f);
    hostMaterials[3].emission = Vec3(0.0f, 0.0f, 0.0f);
    hostMaterials[3].roughness = 0.0f;
    hostMaterials[3].ior = 1.0f;

    DeviceScene deviceScene{};

    checkCuda(cudaMalloc(&deviceScene.spheres, sizeof(hostSpheres)));
    checkCuda(cudaMemcpy(deviceScene.spheres, hostSpheres, sizeof(hostSpheres), cudaMemcpyHostToDevice));

    checkCuda(cudaMalloc(&deviceScene.triangles, sizeof(hostTriangles)));
    checkCuda(cudaMemcpy(deviceScene.triangles, hostTriangles, sizeof(hostTriangles), cudaMemcpyHostToDevice));

    checkCuda(cudaMalloc(&deviceScene.materials, sizeof(hostMaterials)));
    checkCuda(cudaMemcpy(deviceScene.materials, hostMaterials, sizeof(hostMaterials), cudaMemcpyHostToDevice));

    deviceScene.scene.spheres = deviceScene.spheres;
    deviceScene.scene.sphereCount = 2;
    deviceScene.scene.triangles = deviceScene.triangles;
    deviceScene.scene.triangleCount = 18;
    deviceScene.scene.materials = deviceScene.materials;
    deviceScene.scene.materialCount = 4;

    return deviceScene;
}

void buildTlasBlas(DeviceScene& deviceScene, std::vector<MeshAsset>& meshes, const std::vector<SceneInstance>& instances, const std::vector<Material>& materials) {
    std::vector<Blas> hostBlases;
    hostBlases.resize(meshes.size());
    deviceScene.meshTriangles.reserve(meshes.size());
    deviceScene.blasBvhs.reserve(meshes.size());

    for (uint32_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex) {
        MeshAsset& mesh = meshes[meshIndex];
        uint32_t meshLightCount = 0;

        for (Triangle& triangle : mesh.triangles) {
            if (materials[triangle.material].type == MaterialType::Emissive)
                triangle.lightIndex = meshLightCount++;
            else
                triangle.lightIndex = invalidLightIndex;
        }
        Triangle* deviceTriangles = nullptr;
        checkCuda(cudaMalloc(&deviceTriangles, sizeof(Triangle) * mesh.triangles.size()));
        checkCuda(cudaMemcpy(deviceTriangles, mesh.triangles.data(), sizeof(Triangle) * mesh.triangles.size(), cudaMemcpyHostToDevice));
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
    instanceBounds.reserve(instances.size());
    hostInstances.reserve(instances.size());

    for (const SceneInstance& instance : instances) {
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

        uint32_t lightOffset = static_cast<uint32_t>(deviceScene.hostLights.size());

        for (const Triangle& triangle : meshes[instance.meshIndex].triangles) {
            if (triangle.lightIndex == invalidLightIndex)
                continue;

            Vec3 v0 = transformVector(instance.transform, triangle.v0) + instance.transform.translation;
            Vec3 v1 = transformVector(instance.transform, triangle.v1) + instance.transform.translation;
            Vec3 v2 = transformVector(instance.transform, triangle.v2) + instance.transform.translation;
            Vec3 normal = normalize(cross(v1 - v0, v2 - v0));
            float area = 0.5f * length(cross(v1 - v0, v2 - v0));
            const Material& material = materials[triangle.material];
            float power = area * (0.2126f * material.emission.x + 0.7152f * material.emission.y + 0.0722f * material.emission.z);

            deviceScene.hostLights.push_back(TriangleLight{v0, v1, v2, normal, area, triangle.material, 0.0f});
            deviceScene.lightWeights.push_back(power);
        }

        instanceBounds.push_back(worldBounds);
        hostInstances.push_back(MeshInstance{instance.meshIndex, lightOffset, instance.transform});
    }

    NXB::DeviceBuffer<NXB::AABB> deviceInstanceBounds(instanceBounds);
    deviceScene.tlas = NXB::BuildBVH2(deviceInstanceBounds.Get(), static_cast<uint32_t>(instanceBounds.size()));

    checkCuda(cudaMalloc(&deviceScene.blases, sizeof(Blas) * hostBlases.size()));
    checkCuda(cudaMemcpy(deviceScene.blases, hostBlases.data(), sizeof(Blas) * hostBlases.size(), cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc(&deviceScene.instances, sizeof(MeshInstance) * hostInstances.size()));
    checkCuda(cudaMemcpy(deviceScene.instances, hostInstances.data(), sizeof(MeshInstance) * hostInstances.size(), cudaMemcpyHostToDevice));

    deviceScene.scene.tlas = deviceScene.tlas.View();
    deviceScene.scene.blases = deviceScene.blases;
    deviceScene.scene.blasCount = static_cast<uint32_t>(hostBlases.size());
    deviceScene.scene.instances = deviceScene.instances;
    deviceScene.scene.instanceCount = static_cast<uint32_t>(hostInstances.size());
}

void addStaticTriangleLights(DeviceScene& deviceScene, std::vector<Triangle>& triangles, const std::vector<Material>& materials) {
    for (Triangle& triangle : triangles) {
        if (materials[triangle.material].type != MaterialType::Emissive) {
            triangle.lightIndex = invalidLightIndex;
            continue;
        }

        Vec3 normal = normalize(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
        float area = 0.5f * length(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
        const Material& material = materials[triangle.material];
        float power = area * (0.2126f * material.emission.x + 0.7152f * material.emission.y + 0.0722f * material.emission.z);
        triangle.lightIndex = static_cast<uint32_t>(deviceScene.hostLights.size());
        deviceScene.hostLights.push_back(TriangleLight{triangle.v0, triangle.v1, triangle.v2, normal, area, triangle.material, 0.0f});
        deviceScene.lightWeights.push_back(power);
    }
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

    checkCuda(cudaMalloc(&deviceScene.lights, sizeof(TriangleLight) * deviceScene.hostLights.size()));
    checkCuda(cudaMemcpy(deviceScene.lights, deviceScene.hostLights.data(), sizeof(TriangleLight) * deviceScene.hostLights.size(), cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc(&deviceScene.lightAlias, sizeof(LightAliasEntry) * alias.size()));
    checkCuda(cudaMemcpy(deviceScene.lightAlias, alias.data(), sizeof(LightAliasEntry) * alias.size(), cudaMemcpyHostToDevice));
    deviceScene.scene.lights = deviceScene.lights;
    deviceScene.scene.lightAlias = deviceScene.lightAlias;
    deviceScene.scene.lightCount = static_cast<uint32_t>(deviceScene.hostLights.size());
}

DeviceScene createCornellScene(const char* filename, float objectScale, const Vec3& objectTranslation) {
    ObjScene objScene = loadObjScene(filename);

    uint32_t objectTriangleCount = 0;
    constexpr uint32_t objectMaterialOffset = 5;

    for (ObjMesh& mesh : objScene.meshes) {
        objectTriangleCount += static_cast<uint32_t>(mesh.triangles.size());

        for (Triangle& triangle : mesh.triangles)
            triangle.material += objectMaterialOffset;
    }

    DeviceScene deviceScene{};
    auto bvhBuildStart = std::chrono::steady_clock::now();
    std::vector<MeshAsset> meshes;
    meshes.reserve(objScene.meshes.size());

    for (ObjMesh& mesh : objScene.meshes)
        meshes.push_back(MeshAsset{std::move(mesh.triangles)});

    std::vector<SceneInstance> instances;
    instances.reserve(meshes.size());

    for (uint32_t meshIndex = 0; meshIndex < meshes.size(); ++meshIndex)
        instances.push_back(SceneInstance{meshIndex, makeScaledTransform(objectTranslation, objectScale)});

    std::vector<Triangle> hostStaticTriangles;
    hostStaticTriangles.reserve(12);

    constexpr float roomHalfWidth = 2.5f;
    constexpr float floorY = -1.0f;
    constexpr float ceilingY = 3.0f;
    constexpr float backZ = -2.5f;
    constexpr float frontZ = 2.5f;

    addQuad(hostStaticTriangles,
        Vec3(-roomHalfWidth, floorY, frontZ), Vec3(roomHalfWidth, floorY, frontZ),
        Vec3(roomHalfWidth, floorY, backZ), Vec3(-roomHalfWidth, floorY, backZ), 0);
    addQuad(hostStaticTriangles,
        Vec3(-roomHalfWidth, ceilingY, backZ), Vec3(roomHalfWidth, ceilingY, backZ),
        Vec3(roomHalfWidth, ceilingY, frontZ), Vec3(-roomHalfWidth, ceilingY, frontZ), 0);
    addQuad(hostStaticTriangles,
        Vec3(-roomHalfWidth, floorY, backZ), Vec3(roomHalfWidth, floorY, backZ),
        Vec3(roomHalfWidth, ceilingY, backZ), Vec3(-roomHalfWidth, ceilingY, backZ), 0);
    addQuad(hostStaticTriangles,
        Vec3(-roomHalfWidth, floorY, frontZ), Vec3(-roomHalfWidth, floorY, backZ),
        Vec3(-roomHalfWidth, ceilingY, backZ), Vec3(-roomHalfWidth, ceilingY, frontZ), 1);
    addQuad(hostStaticTriangles,
        Vec3(roomHalfWidth, floorY, backZ), Vec3(roomHalfWidth, floorY, frontZ),
        Vec3(roomHalfWidth, ceilingY, frontZ), Vec3(roomHalfWidth, ceilingY, backZ), 2);

    const Vec3 lightCorner(-0.75f, 2.95f, -0.75f);
    const Vec3 lightEdgeU(1.5f, 0.0f, 0.0f);
    const Vec3 lightEdgeV(0.0f, 0.0f, 1.5f);
    addQuad(hostStaticTriangles,
        lightCorner, lightCorner + lightEdgeU, lightCorner + lightEdgeU + lightEdgeV, lightCorner + lightEdgeV, 4);

    std::vector<Material> hostMaterials;
    hostMaterials.reserve(objectMaterialOffset + objScene.materials.size());
    hostMaterials.push_back(makeMaterial(MaterialType::Diffuse, Vec3(0.73f, 0.73f, 0.73f)));
    hostMaterials.push_back(makeMaterial(MaterialType::Diffuse, Vec3(0.65f, 0.05f, 0.05f)));
    hostMaterials.push_back(makeMaterial(MaterialType::Diffuse, Vec3(0.12f, 0.45f, 0.15f)));
    hostMaterials.push_back(makeMaterial(MaterialType::Diffuse, Vec3(0.72f, 0.72f, 0.72f)));
    hostMaterials.push_back(makeMaterial(MaterialType::Emissive, Vec3(1.0f, 1.0f, 1.0f), Vec3(16.0f, 16.0f, 16.0f)));
    hostMaterials.insert(hostMaterials.end(), objScene.materials.begin(), objScene.materials.end());

    buildTlasBlas(deviceScene, meshes, instances, hostMaterials);
    addStaticTriangleLights(deviceScene, hostStaticTriangles, hostMaterials);
    uploadTriangleLights(deviceScene);
    auto bvhBuildEnd = std::chrono::steady_clock::now();

    checkCuda(cudaMalloc(&deviceScene.staticTriangles, sizeof(Triangle) * hostStaticTriangles.size()));
    checkCuda(cudaMemcpy(deviceScene.staticTriangles, hostStaticTriangles.data(), sizeof(Triangle) * hostStaticTriangles.size(), cudaMemcpyHostToDevice));
    checkCuda(cudaMalloc(&deviceScene.materials, sizeof(Material) * hostMaterials.size()));
    checkCuda(cudaMemcpy(deviceScene.materials, hostMaterials.data(), sizeof(Material) * hostMaterials.size(), cudaMemcpyHostToDevice));
    uploadTextures(deviceScene, objScene.textures);

    deviceScene.scene.staticTriangles = deviceScene.staticTriangles;
    deviceScene.scene.staticTriangleCount = static_cast<uint32_t>(hostStaticTriangles.size());
    deviceScene.scene.materials = deviceScene.materials;
    deviceScene.scene.materialCount = static_cast<uint32_t>(hostMaterials.size());
    deviceScene.scene.blackBackground = true;

    float bvhBuildMilliseconds = std::chrono::duration<float, std::milli>(bvhBuildEnd - bvhBuildStart).count();
    std::cout << "Built Cornell TLAS/BLAS BVH2 with " << deviceScene.blasBvhs.size() << " BLASes and " << deviceScene.tlas.NodeCount() << " TLAS nodes for " << objectTriangleCount << " object triangles in " << bvhBuildMilliseconds << " ms\n";

    return deviceScene;
}

DeviceScene createGltfScene(const char* filename, float sceneScale, bool addTopLight) {
    GltfScene gltfScene = loadGltfScene(filename);

    if (gltfScene.meshes.empty() || gltfScene.instances.empty()) {
        std::cerr << "glTF scene contains no renderable mesh instances\n";
        std::exit(1);
    }

    for (SceneInstance& instance : gltfScene.instances) {
        instance.transform.localToWorldX = sceneScale * instance.transform.localToWorldX;
        instance.transform.localToWorldY = sceneScale * instance.transform.localToWorldY;
        instance.transform.localToWorldZ = sceneScale * instance.transform.localToWorldZ;
        instance.transform.worldToLocalX = instance.transform.worldToLocalX / sceneScale;
        instance.transform.worldToLocalY = instance.transform.worldToLocalY / sceneScale;
        instance.transform.worldToLocalZ = instance.transform.worldToLocalZ / sceneScale;
        instance.transform.translation = sceneScale * instance.transform.translation;
    }

    std::vector<Triangle> hostStaticTriangles;
    if (addTopLight) {
        uint32_t lightMaterial = static_cast<uint32_t>(gltfScene.materials.size());
        gltfScene.materials.push_back(makeMaterial(MaterialType::Emissive, Vec3(1.0f, 1.0f, 1.0f), Vec3(80.0f, 80.0f, 80.0f)));

        const Vec3 lightCorner(-4.0f, 9.2f, -4.0f);
        const Vec3 lightEdgeU(8.0f, 0.0f, 0.0f);
        const Vec3 lightEdgeV(0.0f, 0.0f, 8.0f);
        addQuad(hostStaticTriangles,
            lightCorner, lightCorner + lightEdgeU, lightCorner + lightEdgeU + lightEdgeV, lightCorner + lightEdgeV, lightMaterial);
    }

    DeviceScene deviceScene{};
    auto bvhBuildStart = std::chrono::steady_clock::now();
    buildTlasBlas(deviceScene, gltfScene.meshes, gltfScene.instances, gltfScene.materials);
    addStaticTriangleLights(deviceScene, hostStaticTriangles, gltfScene.materials);
    uploadTriangleLights(deviceScene);
    auto bvhBuildEnd = std::chrono::steady_clock::now();

    checkCuda(cudaMalloc(&deviceScene.materials, sizeof(Material) * gltfScene.materials.size()));
    checkCuda(cudaMemcpy(deviceScene.materials, gltfScene.materials.data(), sizeof(Material) * gltfScene.materials.size(), cudaMemcpyHostToDevice));
    uploadTextures(deviceScene, gltfScene.textures);

    if (!hostStaticTriangles.empty()) {
        checkCuda(cudaMalloc(&deviceScene.staticTriangles, sizeof(Triangle) * hostStaticTriangles.size()));
        checkCuda(cudaMemcpy(deviceScene.staticTriangles, hostStaticTriangles.data(), sizeof(Triangle) * hostStaticTriangles.size(), cudaMemcpyHostToDevice));
    }

    deviceScene.scene.staticTriangles = deviceScene.staticTriangles;
    deviceScene.scene.staticTriangleCount = static_cast<uint32_t>(hostStaticTriangles.size());
    deviceScene.scene.materials = deviceScene.materials;
    deviceScene.scene.materialCount = static_cast<uint32_t>(gltfScene.materials.size());
    deviceScene.scene.blackBackground = false;

    float bvhBuildMilliseconds = std::chrono::duration<float, std::milli>(bvhBuildEnd - bvhBuildStart).count();
    std::cout << "Built glTF TLAS/BLAS BVH2 with " << deviceScene.blasBvhs.size() << " BLASes, " << deviceScene.tlas.NodeCount() << " TLAS nodes, " << deviceScene.scene.lightCount << " triangle lights, and " << deviceScene.scene.instanceCount << " instances in " << bvhBuildMilliseconds << " ms\n";

    return deviceScene;
}

DeviceScene createObjScene(const char* filename) {
    ObjMesh mesh = loadObjMesh(filename);

    constexpr float objectScale = 8.0f;
    const Vec3 objectTranslation(0.13f, -0.764f, 0.5f);

    constexpr bool useTlasBlas = true;
    constexpr bool useSpatialSplitBvh = false;
    constexpr bool useNexusBvh = true;
    constexpr bool useNexusBvh8 = false;

    if (!useTlasBlas) {
        for (Triangle& triangle : mesh.triangles) {
            triangle.v0 = objectScale * triangle.v0 + objectTranslation;
            triangle.v1 = objectScale * triangle.v1 + objectTranslation;
            triangle.v2 = objectScale * triangle.v2 + objectTranslation;
        }
    }

    DeviceScene deviceScene{};
    auto bvhBuildStart = std::chrono::steady_clock::now();
    HostBvh bvh;
    NXB::BVHBuildMetrics nexusMetrics{};
    NXB::BVHBuildMetrics tlasMetrics{};

    if (useTlasBlas) {
        std::vector<NXB::Triangle> nexusTriangles;
        nexusTriangles.reserve(mesh.triangles.size());

        for (const Triangle& triangle : mesh.triangles) {
            nexusTriangles.emplace_back(
                make_float3(triangle.v0.x, triangle.v0.y, triangle.v0.z),
                make_float3(triangle.v1.x, triangle.v1.y, triangle.v1.z),
                make_float3(triangle.v2.x, triangle.v2.y, triangle.v2.z));
        }

        NXB::DeviceBuffer<NXB::Triangle> deviceNexusTriangles(nexusTriangles);
        deviceScene.blasBvhs.push_back(NXB::BuildBVH2(deviceNexusTriangles.Get(), static_cast<uint32_t>(nexusTriangles.size()), NXB::BuildConfig{}, &nexusMetrics));

        const NXB::AABB& localBounds = deviceScene.blasBvhs[0].Bounds();
        NXB::AABB instanceBounds(
            make_float3(objectScale * localBounds.bMin.x + objectTranslation.x, objectScale * localBounds.bMin.y + objectTranslation.y, objectScale * localBounds.bMin.z + objectTranslation.z),
            make_float3(objectScale * localBounds.bMax.x + objectTranslation.x, objectScale * localBounds.bMax.y + objectTranslation.y, objectScale * localBounds.bMax.z + objectTranslation.z));
        NXB::DeviceBuffer<NXB::AABB> deviceInstanceBounds(std::vector<NXB::AABB>{instanceBounds});
        deviceScene.tlas = NXB::BuildBVH2(deviceInstanceBounds.Get(), 1, NXB::BuildConfig{}, &tlasMetrics);
    } else if (useNexusBvh) {
        std::vector<NXB::Triangle> nexusTriangles;
        nexusTriangles.reserve(mesh.triangles.size());

        for (const Triangle& triangle : mesh.triangles) {
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
        bvh = useSpatialSplitBvh ? buildSpatialSplitBvh(mesh.triangles) : buildTriangleBvh(mesh.triangles);
    }

    auto bvhBuildEnd = std::chrono::steady_clock::now();
    float bvhBuildMilliseconds = std::chrono::duration<float, std::milli>(bvhBuildEnd - bvhBuildStart).count();

    Sphere hostSpheres[1];
    hostSpheres[0].center = Vec3(0.0f, -100.5f, 0.0f);
    hostSpheres[0].radius = 100.0f;
    hostSpheres[0].material = 0;

    Material groundMaterial{};
    groundMaterial.type = MaterialType::Diffuse;
    groundMaterial.albedo = Vec3(0.75f, 0.75f, 0.75f);
    groundMaterial.emission = Vec3(0.0f, 0.0f, 0.0f);
    groundMaterial.roughness = 0.0f;
    groundMaterial.ior = 1.0f;

    std::vector<Material> hostMaterials;
    hostMaterials.reserve(mesh.materials.size() + 1);
    hostMaterials.push_back(groundMaterial);
    hostMaterials.insert(hostMaterials.end(), mesh.materials.begin(), mesh.materials.end());

    for (Triangle& triangle : mesh.triangles)
        triangle.material += 1;

    checkCuda(cudaMalloc(&deviceScene.spheres, sizeof(hostSpheres)));
    checkCuda(cudaMemcpy(deviceScene.spheres, hostSpheres, sizeof(hostSpheres), cudaMemcpyHostToDevice));

    checkCuda(cudaMalloc(&deviceScene.triangles, sizeof(Triangle) * mesh.triangles.size()));
    checkCuda(cudaMemcpy(deviceScene.triangles, mesh.triangles.data(), sizeof(Triangle) * mesh.triangles.size(), cudaMemcpyHostToDevice));

    if (useTlasBlas) {
        Blas hostBlas{};
        hostBlas.triangles = deviceScene.triangles;
        hostBlas.bvh = deviceScene.blasBvhs[0].View();

        MeshInstance hostInstance{};
        hostInstance.blasIndex = 0;
        hostInstance.lightOffset = invalidLightIndex;
        hostInstance.transform = makeScaledTransform(objectTranslation, objectScale);

        checkCuda(cudaMalloc(&deviceScene.blases, sizeof(hostBlas)));
        checkCuda(cudaMemcpy(deviceScene.blases, &hostBlas, sizeof(hostBlas), cudaMemcpyHostToDevice));
        checkCuda(cudaMalloc(&deviceScene.instances, sizeof(hostInstance)));
        checkCuda(cudaMemcpy(deviceScene.instances, &hostInstance, sizeof(hostInstance), cudaMemcpyHostToDevice));
    }

    if (!useNexusBvh) {
        checkCuda(cudaMalloc(&deviceScene.bvhNodes, sizeof(BvhNode) * bvh.nodes.size()));
        checkCuda(cudaMemcpy(deviceScene.bvhNodes, bvh.nodes.data(), sizeof(BvhNode) * bvh.nodes.size(), cudaMemcpyHostToDevice));

        checkCuda(cudaMalloc(&deviceScene.bvhTriangleIndices, sizeof(uint32_t) * bvh.triangleIndices.size()));
        checkCuda(cudaMemcpy(deviceScene.bvhTriangleIndices, bvh.triangleIndices.data(), sizeof(uint32_t) * bvh.triangleIndices.size(), cudaMemcpyHostToDevice));
    }

    checkCuda(cudaMalloc(&deviceScene.materials, sizeof(Material) * hostMaterials.size()));
    checkCuda(cudaMemcpy(deviceScene.materials, hostMaterials.data(), sizeof(Material) * hostMaterials.size(), cudaMemcpyHostToDevice));
    uploadTextures(deviceScene, mesh.textures);

    deviceScene.scene.spheres = deviceScene.spheres;
    deviceScene.scene.sphereCount = 1;
    deviceScene.scene.triangles = deviceScene.triangles;
    deviceScene.scene.triangleCount = static_cast<uint32_t>(mesh.triangles.size());
    deviceScene.scene.bvhNodes = deviceScene.bvhNodes;
    deviceScene.scene.bvhTriangleIndices = deviceScene.bvhTriangleIndices;
    deviceScene.scene.bvhNodeCount = static_cast<uint32_t>(bvh.nodes.size());
    deviceScene.scene.nexusBvh = deviceScene.nexusBvh.View();
    deviceScene.scene.nexusBvh8 = deviceScene.nexusBvh8.View();
    deviceScene.scene.tlas = deviceScene.tlas.View();
    deviceScene.scene.blases = deviceScene.blases;
    deviceScene.scene.blasCount = static_cast<uint32_t>(deviceScene.blasBvhs.size());
    deviceScene.scene.instances = deviceScene.instances;
    deviceScene.scene.instanceCount = useTlasBlas ? 1 : 0;
    deviceScene.scene.materials = deviceScene.materials;
    deviceScene.scene.materialCount = static_cast<uint32_t>(hostMaterials.size());

    if (useTlasBlas) {
        std::cout << "Built NexusBVH TLAS/BLAS BVH2 with " << deviceScene.blasBvhs[0].NodeCount() << " BLAS nodes and " << deviceScene.tlas.NodeCount() << " TLAS nodes for " << mesh.triangles.size() << " triangles and 1 instance in " << bvhBuildMilliseconds << " ms\n";
        std::cout << "  BLAS bounds " << nexusMetrics.computeSceneBoundsTime << " ms, morton " << nexusMetrics.computeMortonCodesTime << " ms, sort " << nexusMetrics.radixSortTime << " ms, build " << nexusMetrics.bvhBuildTime << " ms\n";
        std::cout << "  TLAS bounds " << tlasMetrics.computeSceneBoundsTime << " ms, morton " << tlasMetrics.computeMortonCodesTime << " ms, sort " << tlasMetrics.radixSortTime << " ms, build " << tlasMetrics.bvhBuildTime << " ms\n";
    } else if (useNexusBvh) {
        if (useNexusBvh8)
            std::cout << "Built NexusBVH H-PLOC BVH8 with " << deviceScene.nexusBvh8.NodeCount() << " nodes (average " << deviceScene.nexusBvh8.AverageChildPerNode() << " children) for " << mesh.triangles.size() << " triangles in " << bvhBuildMilliseconds << " ms\n";
        else
            std::cout << "Built NexusBVH H-PLOC BVH2 with " << deviceScene.nexusBvh.NodeCount() << " nodes for " << mesh.triangles.size() << " triangles in " << bvhBuildMilliseconds << " ms\n";
        std::cout << "  bounds " << nexusMetrics.computeSceneBoundsTime << " ms, morton " << nexusMetrics.computeMortonCodesTime << " ms, sort " << nexusMetrics.radixSortTime << " ms, build " << nexusMetrics.bvhBuildTime << " ms\n";
    } else {
        std::cout << "Built " << (useSpatialSplitBvh ? "SBVH" : "median BVH") << " with " << bvh.nodes.size() << " nodes for " << mesh.triangles.size() << " triangles in " << bvhBuildMilliseconds << " ms\n";
    }

    return deviceScene;
}

void destroyDeviceScene(DeviceScene& deviceScene) {
    checkCuda(cudaFree(deviceScene.spheres));
    checkCuda(cudaFree(deviceScene.triangles));
    checkCuda(cudaFree(deviceScene.staticTriangles));
    for (Triangle* triangles : deviceScene.meshTriangles)
        checkCuda(cudaFree(triangles));
    checkCuda(cudaFree(deviceScene.bvhNodes));
    checkCuda(cudaFree(deviceScene.bvhTriangleIndices));
    deviceScene.nexusBvh = NXB::BVH2{};
    deviceScene.nexusBvh8 = NXB::BVH8{};
    deviceScene.tlas = NXB::BVH2{};
    deviceScene.blasBvhs.clear();
    checkCuda(cudaFree(deviceScene.blases));
    checkCuda(cudaFree(deviceScene.instances));
    checkCuda(cudaFree(deviceScene.materials));
    checkCuda(cudaFree(deviceScene.lights));
    checkCuda(cudaFree(deviceScene.lightAlias));
    for (cudaTextureObject_t textureObject : deviceScene.textureObjects)
        checkCuda(cudaDestroyTextureObject(textureObject));
    for (cudaArray_t textureArray : deviceScene.textureArrays)
        checkCuda(cudaFreeArray(textureArray));
    checkCuda(cudaFree(deviceScene.textures));

    deviceScene = DeviceScene{};
}
