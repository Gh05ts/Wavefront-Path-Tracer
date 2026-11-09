#include "renderer/renderer.cuh"
#include "scene/scene_host.hpp"
#include "core/rng.cuh"
#include "renderer/device_sampling.cuh"
#include "renderer/device_traversal.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>

void checked(cudaError_t value) {
    if (value != cudaSuccess) throw std::runtime_error(cudaGetErrorString(value));
}
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

__global__ void measure(Scene scene, Vec3* results) {
    Hit hit{};
    hit.position = Vec3(); hit.normal = hit.geometricNormal = Vec3(0,0,1);
    PathState path{};
    path.ray.direction = Vec3(0,0,-1);
    path.throughput = Vec3(1,1,1);
    path.rngState = makeRngState(1, RngStrategy::Pcg32);
    sampleDirectPunctualLight(path, hit, hit, scene.materials[0], scene);
    results[0] = path.radiance;
    SpecularGlossinessSurface surface = getSpecGlossSurface(scene.materials[1], hit, scene);
    results[1] = surface.f0;
    results[2] = Vec3(surface.roughness, getMaterialOpacity(scene.materials[1], Vec2{}, Vec2{}, scene), 0);
    results[3] = surface.diffuse;
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try {
        HostScene host;
        Material diffuse{}; diffuse.type = MaterialType::Diffuse; diffuse.albedo = Vec3(.5f,.5f,.5f);
        Material glossy{}; glossy.type = MaterialType::SpecularGlossiness;
        glossy.specularGlossinessWorkflow = true;
        glossy.albedo = Vec3(.8f,.7f,.6f); glossy.specular = Vec3(.5f,.4f,.3f);
        glossy.glossiness = .8f; glossy.opacity = .6f;
        glossy.albedoTexture.texture = glossy.specularGlossinessTexture.texture = 0;
        host.materials = {diffuse, glossy};
        host.textures.push_back(ObjTexture{1,1,{128,64,32,128}});
        PunctualLight light; light.position = Vec3(0,0,2); light.intensity = Vec3(8,8,8);
        host.punctualLights.push_back(light);
        Vec3* deviceResult = nullptr;
        checked(cudaMalloc(&deviceResult, 4*sizeof(Vec3)));
        auto run = [&]() {
            DeviceScene device = uploadScene(host);
            measure<<<1,1>>>(device.scene, deviceResult);
            checked(cudaGetLastError());
            Vec3 result[4]; checked(cudaMemcpy(result, deviceResult, sizeof result, cudaMemcpyDeviceToHost));
            destroyDeviceScene(device);
            return std::vector<Vec3>(result,result+4);
        };
        auto visible = run();
        require(std::abs(visible[0].x - 1.0f/3.14159265359f) < 1e-5f, "Analytic point-light irradiance");
        float srgb = gltfSrgbChannel(128.0f/255);
        require(std::abs(visible[1].x - .5f*srgb) < 1e-6f, "Specular RGB decoded as sRGB");
        require(std::abs(visible[2].x - (1-.8f*128/255)) < 1e-6f, "Glossiness alpha remains linear");
        require(std::abs(visible[2].y - .6f*128/255) < 1e-6f, "Opacity alpha remains linear");
        require(std::abs(visible[3].x - .8f*srgb) < 1e-6f, "Diffuse factor times linear texture");
        host.spheres.emplace_back(Sphere{Vec3(0,0,1),.2f,0});
        require(lengthSquared(run()[0]) == 0, "Punctual shadow ray is occluded");
        // A blocker beyond the emitter must not shadow a point light.
        host.spheres[0].geometry.center = Vec3(0,0,3);
        require(std::abs(run()[0].x - visible[0].x) < 1e-6f, "Finite point-light shadow segment");
        host.spheres.clear();
        PunctualLight sun; sun.type = PunctualLightType::Directional;
        sun.intensity = Vec3(2,2,2);
        host.punctualLights.push_back(sun); // upload sorts the directional prefix
        require(std::abs(run()[0].x - 2*visible[0].x) < 1e-6f, "Directional plus positional light sum");
        checked(cudaFree(deviceResult));
        std::cout << "GPU light/shadow and material texture-channel checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
