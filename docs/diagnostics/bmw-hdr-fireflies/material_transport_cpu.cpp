#include <cuda_runtime.h>
#include <algorithm>
using std::min;using std::max;
struct FakeIndex{unsigned x=0;} blockIdx,blockDim,threadIdx;
int __popc(unsigned x){return __builtin_popcount(x);}
template<class T>T tex2D(cudaTextureObject_t,float,float){return T{};}
#include "renderer/renderer.cuh"
#include "core/rng.cuh"
#include "renderer/device_sampling.cuh"
#include "renderer/device_traversal.cuh"
#include <iostream>

// No loaded assets: exercise both sides of actual triangle intersections and
// test refraction through a parallel glass slab against Snell's law.
__global__ void checkMaterialTransport(int* failure) {
    *failure = 0;
    Material materials[2]{};
    materials[0].type = MaterialType::Dielectric;
    materials[0].ior = 1.5f;
    materials[0].albedo = Vec3(1,1,1);
    materials[1].type = MaterialType::Diffuse; // glTF metals resolve at shading
    materials[1].metallic = 1;
    materials[1].roughness = .47f;
    materials[1].albedo = Vec3(.9f,.8f,.7f);
    Triangle triangles[2]{};
    triangles[0].v0 = Vec3(-2,-2,0);
    triangles[0].v1 = Vec3(2,-2,0);
    triangles[0].v2 = Vec3(0,2,0);
    triangles[1].v0 = Vec3(-2,-2,-1);
    triangles[1].v1 = Vec3(0,2,-1);
    triangles[1].v2 = Vec3(2,-2,-1);
    Scene scene{};
    scene.triangles = triangles; scene.triangleCount = 2;
    scene.materials = materials;
    Vec3 incident(.3f,0,-sqrtf(.91f));
    auto entry = traceRay(Ray(Vec3(0,0,1),incident),scene);
    if (!entry.didHit || dot(incident,entry.hit.normal) >= 0) { *failure=1; return; }
    auto inside = refract(incident,entry.hit.normal,1/materials[0].ior);
    auto exit = traceRay(Ray(entry.hit.position+inside*.01f,inside),scene);
    if (!exit.didHit || dot(inside,exit.hit.normal) <= 0) { *failure=2; return; }
    auto outside = refract(inside,-exit.hit.normal,materials[0].ior);
    if (length(outside-incident) > 1e-5f) { *failure=3; return; }
    // Preserve the existing two-sided shading convention for opaque surfaces.
    materials[0].type = MaterialType::Diffuse;
    exit = traceRay(Ray(Vec3(0,0,-.5f),Vec3(0,0,1)),scene);
    if (!exit.didHit || dot(exit.hit.normal,Vec3(0,0,1)) >= 0) { *failure=4; return; }
    Hit hit{}; hit.normal = hit.geometricNormal = Vec3(0,0,1);
    SpecularGlossinessSurface metal{Vec3(),materials[1].albedo,materials[1].roughness};
    float pdf,expectedPdf;
    Vec3 actual = evaluateSurface(materials[1],hit,scene,hit.normal,hit.normal,pdf);
    Vec3 expected = evaluateSpecGloss(metal,hit.normal,hit.normal,hit.normal,expectedPdf);
    if (fabsf(pdf-expectedPdf)>1e-5f || length(actual-expected)>1e-5f) { *failure=5; return; }
    PathState path{}; path.active=true; path.specularBounce=true;
    path.ray.direction=-hit.normal; path.throughput=Vec3(1,1,1);
    path.rngState=makeRngState(42,RngStrategy::Pcg32);
    continueGlossySurface(path,hit,hit,materials[1],scene);
    if (!path.active || path.specularBounce || !(path.previousBsdfPdf>0)) { *failure=6; return; }
    // Zero-roughness metal remains a delta, with no finite-density NEE lobe.
    materials[1].roughness=0;
    actual=evaluateSurface(materials[1],hit,scene,hit.normal,hit.normal,pdf);
    if (pdf!=0 || lengthSquared(actual)!=0) { *failure=7; return; }
}

int main(){int failure=-1;checkMaterialTransport(&failure);std::cout<<"failure="<<failure<<"\n";return failure;}
