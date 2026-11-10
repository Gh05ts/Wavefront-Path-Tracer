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
    // BMW uses BVH2 traversal, whose leaf intersection has a separate normal path.
    NXB::BVH2::Node nodes[3]{};
    nodes[0].bounds = NXB::AABB(make_float3(-2,-2,-.0001f),make_float3(2,2,.0001f));
    nodes[1].bounds = NXB::AABB(make_float3(-2,-2,-1.0001f),make_float3(2,2,-.9999f));
    nodes[0].leftChild = nodes[1].leftChild = NXB::InvalidIdx;
    nodes[0].rightChild = 0; nodes[1].rightChild = 1;
    nodes[2].bounds = NXB::AABB(make_float3(-2,-2,-1.0001f),make_float3(2,2,.0001f));
    nodes[2].leftChild = 0; nodes[2].rightChild = 1;
    scene.nexusBvh = NXB::BVH2::DeviceView{nodes,3,2,nodes[2].bounds};
    entry = traceRay(Ray(Vec3(0,0,1),incident),scene);
    if (!entry.didHit || dot(incident,entry.hit.normal) >= 0) { *failure=8; return; }
    exit = traceRay(Ray(entry.hit.position+inside*.01f,inside),scene);
    if (!exit.didHit || dot(inside,exit.hit.normal) <= 0) { *failure=9; return; }
    // Preserve the existing two-sided shading convention for opaque surfaces.
    materials[0].type = MaterialType::Diffuse;
    exit = traceRay(Ray(Vec3(0,0,-.5f),Vec3(0,0,1)),scene);
    if (!exit.didHit || dot(exit.hit.normal,Vec3(0,0,1)) >= 0) { *failure=4; return; }
    Hit hit{}; hit.normal = hit.geometricNormal = Vec3(0,0,1);
    // At normal incidence GGX has f = F0/(4*pi*roughness^4), G=1.
    // The existing sampler mixes 95% GGX and 5% cosine for this pure metal.
    float roughnessSquared = materials[1].roughness * materials[1].roughness;
    float normalLobe = 1 / (4 * 3.14159265359f * roughnessSquared * roughnessSquared);
    float pdf, expectedPdf = .95f * normalLobe + .05f / 3.14159265359f;
    Vec3 actual = evaluateSurface(materials[1],hit,scene,hit.normal,hit.normal,pdf);
    Vec3 expected = materials[1].albedo * normalLobe;
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
    // Independent Fresnel values, reciprocity and the critical-angle limit.
    if (fabsf(dielectricReflectance(1,1/1.5f)-.04f)>1e-6f ||
        fabsf(dielectricReflectance(.767f,1.5f)-.23811942f)>2e-6f ||
        dielectricReflectance(.7f,1.5f)!=1 || dielectricReflectance(0,1)!=0) {
        *failure=10; return;
    }
    float cosine=.8f, eta=1/1.5f;
    float transmittedCosine=sqrtf(1-eta*eta*(1-cosine*cosine));
    if (fabsf(dielectricReflectance(cosine,eta)-dielectricReflectance(transmittedCosine,1/eta))>1e-6f) {
        *failure=11; return;
    }
    // Smooth normals cannot determine medium membership or send a specular
    // branch to the wrong geometric hemisphere. Cover entry and exit.
    for (int side=0;side<2;++side) {
        Vec3 geometric(0,0,side?-1:1), incoming(0,0,-1);
        for (int tilt=0;tilt<2;++tilt) {
            Vec3 smooth=normalize(Vec3(.99f,0,tilt?-.14f:.14f));
            auto boundary=dielectricInterface(incoming,geometric,smooth,1.5f);
            Vec3 facing=side?-geometric:geometric;
            if (boundary.frontFace==bool(side) || dot(boundary.reflection,facing)<=0 ||
                (boundary.reflectance<1 && dot(boundary.transmission,facing)>=0)) {
                *failure=12; return;
            }
        }
    }
    // Exercise the shared camera continuation, not just Snell's helper.
    materials[0].type=MaterialType::Dielectric;
    // A 1.5 mm pane was skipped by the combined old 1 mm shift + cutoff.
    triangles[1].v0.z=triangles[1].v1.z=triangles[1].v2.z=-.0015f;
    scene.nexusBvh={};
    entry=traceRay(Ray(Vec3(0,0,1),Vec3(0,0,-1)),scene);
    Vec3 thinDirection(0,0,-1);
    Ray thinRay(offsetRayOrigin(entry.hit.position,entry.hit.geometricNormal,thinDirection,entry.hit.t),thinDirection);
    exit=traceRay(thinRay,scene);
    if (!exit.didHit || dot(thinDirection,exit.hit.geometricNormal)<=0 || exit.hit.t>.0015f) {
        *failure=15;return;
    }
    nodes[1].bounds=NXB::AABB(make_float3(-2,-2,-.0016f),make_float3(2,2,-.0014f));
    scene.nexusBvh=NXB::BVH2::DeviceView{nodes,3,2,nodes[2].bounds};
    exit=traceRay(thinRay,scene);
    if (!exit.didHit || dot(thinDirection,exit.hit.geometricNormal)<=0) { *failure=16;return; }
    scene.nexusBvh={};scene.triangleCount=0;
    Hit boundaryHit{};boundaryHit.normal=boundaryHit.geometricNormal=Vec3(0,0,1);
    PathState entered{};bool foundEntry=false,foundExit=false;
    for (int seed=0;seed<64 && !foundEntry;++seed) {
        entered={};entered.active=true;entered.throughput=Vec3(1,1,1);
        entered.ray.direction=Vec3(0,0,-1);entered.rngState=makeRngState(seed,RngStrategy::Pcg32);
        continueDielectricSurface(entered,boundaryHit,boundaryHit,materials[0],scene,12);
        foundEntry=entered.mediumMaterial==0;
    }
    if (!foundEntry || fabsf(entered.throughput.x-4/9.f)>1e-6f || fabsf(entered.etaScale-2.25f)>1e-6f) {
        *failure=13;return;
    }
    boundaryHit.normal=boundaryHit.geometricNormal=Vec3(0,0,-1);
    for (int seed=0;seed<64 && !foundExit;++seed) {
        path=entered;path.rngState=makeRngState(seed,RngStrategy::Pcg32);
        continueDielectricSurface(path,boundaryHit,boundaryHit,materials[0],scene,12);
        foundExit=path.mediumMaterial==invalidMaterialIndex;
    }
    if (!foundExit || fabsf(path.throughput.x-1)>1e-6f || fabsf(path.etaScale-1)>1e-6f) {
        *failure=14;return;
    }
    // Precision-scaled shadow origins must not self-occlude a tilted emitter.
    // Compare against identical samples with occlusion explicitly disabled.
    materials[0].type=MaterialType::Diffuse;materials[0].albedo=Vec3(1,1,1);materials[0].metallic=0;
    materials[1].type=MaterialType::Emissive;materials[1].emission=Vec3(1,1,1);
    triangles[0]={};triangles[0].material=1;
    triangles[0].v0=Vec3(-10,-10,1);triangles[0].v1=Vec3(0,10,3);triangles[0].v2=Vec3(10,-10,2);
    TriangleLight light{};light.v0=triangles[0].v0;light.v1=triangles[0].v1;light.v2=triangles[0].v2;
    light.normal=normalize(cross(light.v1-light.v0,light.v2-light.v0));
    light.area=length(cross(light.v1-light.v0,light.v2-light.v0))*.5f;
    light.selectionPdf=1;light.material=1;
    LightAliasEntry alias{};alias.probability=1;alias.alias=0;
    scene.triangleCount=1;scene.lights=&light;scene.lightAlias=&alias;scene.lightCount=1;
    boundaryHit.normal=boundaryHit.geometricNormal=Vec3(0,0,1);boundaryHit.t=100;
    float energy=0;
    for (int seed=0;seed<64;++seed) {
        PathState visible{};visible.throughput=Vec3(1,1,1);visible.ray.direction=Vec3(0,0,-1);
        visible.rngState=makeRngState(seed,RngStrategy::Pcg32);PathState unobstructed=visible;
        scene.ignoreDirectLightOcclusion=false;
        sampleDirectLight(visible,boundaryHit,boundaryHit,materials[0],scene);
        scene.ignoreDirectLightOcclusion=true;
        sampleDirectLight(unobstructed,boundaryHit,boundaryHit,materials[0],scene);
        if (length(visible.radiance-unobstructed.radiance)>1e-6f) { *failure=17;return; }
        energy+=visible.radiance.x;
    }
    if (!(energy>0)) { *failure=18;return; }
    triangles[1]=triangles[0];triangles[1].material=0;
    triangles[1].v0*=.5f;triangles[1].v1*=.5f;triangles[1].v2*=.5f;
    scene.triangleCount=2;scene.ignoreDirectLightOcclusion=false;
    path={};path.throughput=Vec3(1,1,1);path.ray.direction=Vec3(0,0,-1);
    path.rngState=makeRngState(42,RngStrategy::Pcg32);
    sampleDirectLight(path,boundaryHit,boundaryHit,materials[0],scene);
    if (lengthSquared(path.radiance)>0) { *failure=19;return; }

    // High-intensity punctual NEE: direct highlights stay HDR; indirect
    // contributions are limited *after* throughput, with RGB ratios preserved.
    scene={}; scene.materials=materials;
    PunctualLight source{};source.type=PunctualLightType::Directional;
    source.direction=Vec3(0,0,-1);source.intensity=Vec3(1000,500,250);
    path={};path.throughput=Vec3(1,1,1);path.ray.direction=Vec3(0,0,-1);
    path.indirectRadianceClamp=10;
    PathState indirect=path;indirect.depth=1;
    addPunctualLight(path,boundaryHit,boundaryHit,materials[0],scene,source,1);
    addPunctualLight(indirect,boundaryHit,boundaryHit,materials[0],scene,source,1);
    if (path.radiance.x<300 || length(indirect.radiance-Vec3(120/7.f,60/7.f,30/7.f))>1e-4f) {
        *failure=20;return;
    }
    PathState dim=indirect;dim.radiance={};dim.throughput=Vec3(.001f,.001f,.001f);
    addPunctualLight(dim,boundaryHit,boundaryHit,materials[0],scene,source,1);
    if (length(dim.radiance-path.radiance*.001f)>1e-5f) { *failure=21;return; }
    PathState raw=indirect;raw.radiance={};raw.indirectRadianceClamp=0;
    addPunctualLight(raw,boundaryHit,boundaryHit,materials[0],scene,source,1);
    if (length(raw.radiance-path.radiance)>1e-5f || length(raw.throughput-indirect.throughput)>0) {
        *failure=22;return;
    }
}

int main() {
    int count=0;
    if (cudaGetDeviceCount(&count)!=cudaSuccess || count==0) return 77;
    int* failure=nullptr;
    if (cudaMalloc(&failure,sizeof(int))!=cudaSuccess) return 1;
    checkMaterialTransport<<<1,1>>>(failure);
    int result=-1;
    auto status=cudaMemcpy(&result,failure,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(failure);
    if (status!=cudaSuccess || result!=0) {
        std::cerr<<"Material transport regression failed: "<<result<<'\n'; return 1;
    }
    std::cout<<"Glass boundaries, exact Fresnel, normal hemispheres, radiance eta, Snell slab and rough-metal checks passed\n";
    return 0;
}
