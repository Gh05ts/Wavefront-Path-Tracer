#include "renderer/renderer.cuh"
#include "scene/scene_host.hpp"
#include "scene/environment.hpp"
#include "core/rng.cuh"
#include "renderer/device_sampling.cuh"
#include "renderer/device_traversal.cuh"

#include <iostream>
#include <stdexcept>

void checked(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}

__global__ void energy(Scene scene,Vec3* values,Vec3* direct,int count) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    Hit hit{};hit.position=Vec3();hit.normal=hit.geometricNormal=Vec3(0,1,0);
    PathState path{};path.rngState=makeRngState(index+1,RngStrategy::Pcg32);
    path.ray.direction=Vec3(0,-1,0);path.throughput=Vec3(1,1,1);
    sampleDirectEnvironment(path,hit,hit,scene.materials[0],scene);
    direct[index]=path.radiance;
    // One BSDF sample complements NEE; a constant map gives a known analytic integral.
    path.ray.direction=sampleCosineHemisphere(hit.normal,path.rngState);
    path.throughput=scene.materials[0].albedo;
    path.depth=1;path.specularBounce=false;
    path.previousBsdfPdf=dot(hit.normal,path.ray.direction)*0.31830988618f;
    addEnvironmentOrSky(path,scene);values[index]=path.radiance;
}

__global__ void lookup(Scene scene,Vec3 direction,Vec3* result) {
    PathState path{};path.throughput=Vec3(1,1,1);path.ray.direction=direction;path.specularBounce=true;
    addEnvironmentOrSky(path,scene);result[0]=path.radiance;
}

int main(){
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
    try {
        HostScene host;host.blackBackground=true;
        Material material{};material.type=MaterialType::Diffuse;material.albedo=Vec3(.2f,.4f,.6f);
        host.materials.push_back(material);DeviceScene scene=uploadScene(host);
        HostEnvironment map;map.width=16;map.height=8;map.pixels.assign(128,make_float4(4,4,4,1));
        buildEnvironmentDistribution(map);uploadEnvironmentMap(scene,map,2.5f,0);
        constexpr int samples=65536;Vec3 *values=nullptr,*direct=nullptr;
        checked(cudaMalloc(&values,samples*sizeof(Vec3)));checked(cudaMalloc(&direct,samples*sizeof(Vec3)));
        energy<<<samples/256,256>>>(scene.scene,values,direct,samples);checked(cudaGetLastError());
        std::vector<Vec3> result(samples);checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        Vec3 mean;for(const auto& value:result){require(std::isfinite(value.x),"Non-finite environment estimate");mean+=value;}mean*=1.0f/samples;
        require(std::abs(mean.x-2)<.02f&&std::abs(mean.y-4)<.04f&&std::abs(mean.z-6)<.06f,"MIS energy: environment contribution missing or double counted");
        // Surround the receiving point with opaque geometry: all environment NEE is blocked.
        host.spheres.emplace_back(Sphere{Vec3(),10,0});DeviceScene blocked=uploadScene(host);
        uploadEnvironmentMap(blocked,map,1,0);
        energy<<<samples/256,256>>>(blocked.scene,values,direct,samples);checked(cudaGetLastError());
        checked(cudaMemcpy(result.data(),direct,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        for(const auto& value:result)require(lengthSquared(value)==0,"Environment shadow occlusion failed");
        destroyDeviceScene(blocked);
        // Pixel-center lookup must retain HDR values, without an LDR/gamma conversion.
        map.width=4;map.height=2;map.pixels.assign(8,make_float4(0,0,0,1));map.pixels[0]=make_float4(16,8,4,1);
        buildEnvironmentDistribution(map);uploadEnvironmentMap(scene,map,2,0);
        float theta=3.14159265359f*.25f,phi=-3.14159265359f*.75f;
        Vec3 direction(sinf(theta)*cosf(phi),cosf(theta),sinf(theta)*sinf(phi));
        lookup<<<1,1>>>(scene.scene,direction,values);Vec3 original;
        checked(cudaMemcpy(&original,values,sizeof original,cudaMemcpyDeviceToHost));
        require(std::abs(original.x-32)<.01f&&std::abs(original.y-16)<.01f,"HDR texture lookup/intensity");
        uploadEnvironmentMap(scene,map,2,90);
        lookup<<<1,1>>>(scene.scene,rotateEnvironment(direction,3.14159265359f/2),values);Vec3 rotated;
        checked(cudaMemcpy(&rotated,values,sizeof rotated,cudaMemcpyDeviceToHost));
        require(length(original-rotated)<.01f,"HDR texture rotation");
        uploadEnvironmentMap(scene,map,0,0);
        lookup<<<1,1>>>(scene.scene,direction,values);Vec3 black;
        checked(cudaMemcpy(&black,values,sizeof black,cudaMemcpyDeviceToHost));require(lengthSquared(black)==0,"Zero environment intensity");
        clearEnvironmentMap(scene);require(scene.scene.environment.texture==0&&scene.environmentArray==nullptr,"Environment release/reset");
        scene.scene.blackBackground=false;
        lookup<<<1,1>>>(scene.scene,Vec3(0,1,0),values);Vec3 sky;
        checked(cudaMemcpy(&sky,values,sizeof sky,cudaMemcpyDeviceToHost));
        require(length(sky-Vec3(.5f,.7f,1))<1e-6f,"Procedural sky fallback changed without an HDR map");
        checked(cudaFree(values));checked(cudaFree(direct));destroyDeviceScene(scene);
        std::cout<<"GPU environment MIS energy, shadows, HDR lookup, intensity and rotation checks passed\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
