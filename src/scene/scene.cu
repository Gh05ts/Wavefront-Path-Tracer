#include "scene/scene.cuh"
#include "scene/obj_loader.hpp"

#include <cuda_runtime.h>

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

Triangle makeTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2, uint32_t material) {
    return Triangle{v0, v1, v2, material};
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

DeviceScene createObjScene(const char* filename) {
    ObjMesh mesh = loadObjMesh(filename);

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

    DeviceScene deviceScene{};

    checkCuda(cudaMalloc(&deviceScene.spheres, sizeof(hostSpheres)));
    checkCuda(cudaMemcpy(deviceScene.spheres, hostSpheres, sizeof(hostSpheres), cudaMemcpyHostToDevice));

    checkCuda(cudaMalloc(&deviceScene.triangles, sizeof(Triangle) * mesh.triangles.size()));
    checkCuda(cudaMemcpy(deviceScene.triangles, mesh.triangles.data(), sizeof(Triangle) * mesh.triangles.size(), cudaMemcpyHostToDevice));

    checkCuda(cudaMalloc(&deviceScene.materials, sizeof(Material) * hostMaterials.size()));
    checkCuda(cudaMemcpy(deviceScene.materials, hostMaterials.data(), sizeof(Material) * hostMaterials.size(), cudaMemcpyHostToDevice));

    deviceScene.scene.spheres = deviceScene.spheres;
    deviceScene.scene.sphereCount = 1;
    deviceScene.scene.triangles = deviceScene.triangles;
    deviceScene.scene.triangleCount = static_cast<uint32_t>(mesh.triangles.size());
    deviceScene.scene.materials = deviceScene.materials;
    deviceScene.scene.materialCount = static_cast<uint32_t>(hostMaterials.size());

    return deviceScene;
}

void destroyDeviceScene(DeviceScene& deviceScene) {
    checkCuda(cudaFree(deviceScene.spheres));
    checkCuda(cudaFree(deviceScene.triangles));
    checkCuda(cudaFree(deviceScene.materials));

    deviceScene = DeviceScene{};
}
