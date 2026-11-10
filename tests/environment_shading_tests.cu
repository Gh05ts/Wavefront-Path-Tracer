#include "renderer/renderer.cuh"
#include "scene/scene_host.hpp"
#include "scene/environment.hpp"
#include "core/rng.cuh"
#include "renderer/device_sampling.cuh"
#include "renderer/device_traversal.cuh"

#include <algorithm>
#include <iostream>
#include <stdexcept>

void checked(cudaError_t value){if(value!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(value));}
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}

__global__ void energy(Scene scene,Vec3* values,Vec3* direct,int count,Vec3 normal) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    Hit hit{};hit.position=Vec3();hit.normal=normal;hit.geometricNormal=Vec3(0,1,0);
    PathState path{};path.rngState=makeRngState(index+1,RngStrategy::Pcg32);
    path.ray.direction=Vec3(0,-1,0);path.throughput=Vec3(1,1,1);path.active=true;
    sampleDirectEnvironment(path,hit,hit,scene.materials[0],scene);
    direct[index]=path.radiance;
    // One BSDF sample complements NEE; a constant map gives a known analytic integral.
    path.ray.direction=scene.materials[0].gltfPbrWorkflow ? continueGltfPbrSurface(path,hit,hit,scene.materials[0],scene,12) :
        scene.materials[0].metallic>.5f ? continueGlossySurface(path,hit,hit,scene.materials[0],scene) :
        continueDiffuseSurface(path,hit,hit,scene.materials[0],scene);
    path.depth=1;
    if(path.active)addEnvironmentOrSky(path,scene);values[index]=path.radiance;
}

__global__ void lookup(Scene scene,Vec3 direction,Vec3* result) {
    PathState path{};path.throughput=Vec3(1,1,1);path.ray.direction=direction;path.specularBounce=true;
    addEnvironmentOrSky(path,scene);result[0]=path.radiance;
}

__global__ void glassTerminal(Scene scene, Vec3* values, int count, uint32_t maxDepth) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    Hit hit{};hit.normal=hit.geometricNormal=Vec3(0,1,0);
    PathState path{};path.active=true;path.throughput=Vec3(1,1,1);
    path.ray.direction=Vec3(0,-1,0);path.rngState=makeRngState(index+1,RngStrategy::Pcg32);
    continueDielectricSurface(path,hit,hit,scene.materials[0],scene,maxDepth);
    values[index]=path.radiance;
    // Both branches escape in the empty scene, so none may be counted again
    // on a later miss. The enclosing sphere blocks both in the occlusion test.
    if (scene.sphereCount==0 && maxDepth>1 && path.active) values[index]=Vec3(-1,-1,-1);
}

// Exercise real HDR miss and deterministic glass-terminal accumulation, not
// a display filter. A first-bounce escape is direct, matching primary NEE;
// a terminal connection at the second surface is indirect even with splitting.
__global__ void clampConnections(Scene scene, Vec3* values) {
    PathState path{};path.active=true;path.throughput=Vec3(1,1,1);
    path.ray.direction=Vec3(0,-1,0);path.specularBounce=true;
    path.rngState=makeRngState(42,RngStrategy::Pcg32);
    path.indirectRadianceClamp=10;
    path.depth=1;
    addEnvironmentOrSky(path,scene);values[0]=path.radiance;
    path.depth=2;path.radiance={};
    addEnvironmentOrSky(path,scene);values[1]=path.radiance;
    path.indirectRadianceClamp=0;path.radiance={};
    addEnvironmentOrSky(path,scene);values[2]=path.radiance;
    Hit hit{};hit.normal=hit.geometricNormal=Vec3(0,1,0);
    path.indirectRadianceClamp=10;path.radiance={};path.depth=0;
    PathState indirect=path;indirect.depth=1;
    continueDielectricSurface(path,hit,hit,scene.materials[0],scene,12);
    continueDielectricSurface(indirect,hit,hit,scene.materials[0],scene,12);
    values[3]=path.radiance;values[4]=indirect.radiance;
    values[5]=path.throughput-indirect.throughput;
}

// A bright compact source behind a parallel slab has an independently
// integrated target. Ordinary NEE must remain blocked by the glass; the traced
// continuation is responsible for discovering the specular connection.
__global__ void slabSun(Scene scene, Vec3* values, int count) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    Hit hit{};hit.normal=hit.geometricNormal=Vec3(0,1,0);
    PathState path{};path.active=true;path.throughput=Vec3(1,1,1);
    path.ray.direction=Vec3(0,-1,0);path.rngState=makeRngState(index+987,RngStrategy::Pcg32);
    sampleDirectEnvironment(path,hit,hit,scene.materials[0],scene);
    Vec3 direction=scene.materials[0].metallic>.5f ? continueGlossySurface(path,hit,hit,scene.materials[0],scene) :
        continueDiffuseSurface(path,hit,hit,scene.materials[0],scene);
    path.ray=Ray(offsetRayOrigin(hit.position,hit.geometricNormal,direction),direction);
    path.depth=1;
    while(path.active && path.depth<12) {
        auto next=traceRay(path.ray,scene);
        if(!next.didHit){addEnvironmentOrSky(path,scene);break;}
        if(scene.materials[next.hit.material].type!=MaterialType::Dielectric)break;
        direction=continueDielectricSurface(path,next.hit,next.hit,scene.materials[next.hit.material],scene,12);
        path.ray=Ray(offsetRayOrigin(next.hit.position,next.hit.geometricNormal,direction,next.hit.t),direction);
        ++path.depth;
    }
    values[index]=path.radiance;
}

double integratedSun(uint32_t x,uint32_t y,uint32_t width,uint32_t height,double radiance,bool slab,bool metal=false) {
    constexpr double pi=3.14159265358979323846;
    constexpr int subdivisions=64;
    double integral=0;
    // Integrate the entire bilinear reconstruction footprint, including the
    // neighboring dark texel cells. This does not use the sampler's PDF/table.
    for(int row=int(y)-1;row<=int(y)+1;++row)for(int col=int(x)-1;col<=int(x)+1;++col)
        for(int j=0;j<subdivisions;++j)for(int i=0;i<subdivisions;++i) {
            double u=col+(i+.5)/subdivisions,v=row+(j+.5)/subdivisions;
            double tent=std::max(0.,1-std::abs(u-.5-x))*std::max(0.,1-std::abs(v-.5-y));
            double cosine=std::cos(pi*v/height);
            double omega=(2*pi/(width*subdivisions))*(std::cos(pi*(row+double(j)/subdivisions)/height)-
                std::cos(pi*(row+double(j+1)/subdivisions)/height));
            double transmission=1;
            if(slab) {
                double cosT=std::sqrt(1-(1-cosine*cosine)/2.25);
                double rs=(cosine-1.5*cosT)/(cosine+1.5*cosT);
                double rp=(1.5*cosine-cosT)/(1.5*cosine+cosT);
                double f=(rs*rs+rp*rp)/2;
                // Upward exits at path depths 2,4,6,8,10; the fixed cap is 12.
                transmission=(1-f)/(1+f)*(1-std::pow(f,10));
            }
            double bsdf=1/pi;
            if(metal) {
                // Independent isotropic GGX for normal view, F0=.8 and
                // perceptual roughness=.5 (alpha=.25), no diffuse lobe.
                double nh=std::sqrt((1+cosine)/2),a2=.0625;
                double denominator=1-nh*nh+a2*nh*nh;
                double d=a2/(pi*denominator*denominator);
                double g=2*cosine/(cosine+std::sqrt(a2+(1-a2)*cosine*cosine));
                double f=.8+.2*std::pow(1-nh,5);
                bsdf=f*d*g/(4*cosine);
            }
            integral+=radiance*tent*std::max(0.,cosine)*transmission*omega*bsdf;
        }
    return integral;
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
        energy<<<samples/256,256>>>(scene.scene,values,direct,samples,Vec3(0,1,0));checked(cudaGetLastError());
        std::vector<Vec3> result(samples);checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        Vec3 mean;for(const auto& value:result){require(std::isfinite(value.x),"Non-finite environment estimate");mean+=value;}mean*=1.0f/samples;
        require(std::abs(mean.x-2)<.02f&&std::abs(mean.y-4)<.04f&&std::abs(mean.z-6)<.06f,"MIS energy: environment contribution missing or double counted");
        // Independently known projected-area overlap: for a tilted diffuse
        // normal clipped by the geometric hemisphere, integral = (1+n.y)/2.
        energy<<<samples/256,256>>>(scene.scene,values,direct,samples,Vec3(sqrtf(.75f),.5f,0));
        checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        mean=Vec3();for(const auto& value:result)mean+=value;mean*=1.0f/samples;
        require(std::abs(mean.x-1.5f)<.025f&&std::abs(mean.y-3)<.05f&&std::abs(mean.z-4.5f)<.075f,"Tilted diffuse continuation PDF or geometric rejection");
        // BMW paint: specularFactor=0, smooth coat=1. Analytic constant-map
        // response is .04 + .96*baseColor. Exercise actual NEE + continuation,
        // including delta splitting, MIS, and the later environment miss.
        host.materials[0].gltfPbrWorkflow=true;host.materials[0].ior=1.5f;
        host.materials[0].specularWeight=0;host.materials[0].clearcoat=1;
        DeviceScene paint=uploadScene(host);uploadEnvironmentMap(paint,map,2.5f,0);
        energy<<<samples/256,256>>>(paint.scene,values,direct,samples,Vec3(0,1,0));
        checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        mean=Vec3();for(const auto& value:result)mean+=value;mean*=1.f/samples;
        require(length(mean-Vec3(2.32f,4.24f,6.16f))<.07f,"glTF smooth coat energy or delta double counting");
        destroyDeviceScene(paint);
        // Perfect imported metal has no finite-density direct term.
        host.materials[0].metallic=1;
        paint=uploadScene(host);uploadEnvironmentMap(paint,map,2.5f,0);
        energy<<<samples/256,256>>>(paint.scene,values,direct,samples,Vec3(0,1,0));
        checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        for(const auto& value:result)require(length(value-Vec3(2.32f,4.24f,6.16f))<1.e-5f,"glTF coincident delta coat/metal lobes");
        destroyDeviceScene(paint);
        host.materials[0].gltfPbrWorkflow=false;host.materials[0].metallic=0;host.materials[0].clearcoat=0;
        // Surround the receiving point with opaque geometry: all environment NEE is blocked.
        host.spheres.emplace_back(Sphere{Vec3(),10,0});DeviceScene blocked=uploadScene(host);
        uploadEnvironmentMap(blocked,map,1,0);
        energy<<<samples/256,256>>>(blocked.scene,values,direct,samples,Vec3(0,1,0));checked(cudaGetLastError());
        checked(cudaMemcpy(result.data(),direct,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        for(const auto& value:result)require(lengthSquared(value)==0,"Environment shadow occlusion failed");
        destroyDeviceScene(blocked);
        // A narrow high-dynamic-range source exposes failures hidden by the
        // constant-map energy test. Paired boundaries preserve radiance eta.
        host.spheres.clear();
        Material pane{};pane.type=MaterialType::Dielectric;pane.albedo=Vec3(1,1,1);pane.ior=1.5f;
        host.materials.push_back(pane);
        map.width=512;map.height=256;map.pixels.assign(map.width*map.height,make_float4(0,0,0,1));
        constexpr uint32_t sunX=170,sunY=64;
        map.pixels[sunY*map.width+sunX]=make_float4(50000,50000,50000,1);buildEnvironmentDistribution(map);
        for(int layer=0;layer<2;++layer)for(int half=0;half<2;++half) {
            Triangle triangle{};float y=1+layer*.01f;
            triangle.v0=Vec3(-1000,y,-1000);
            triangle.v1=half?Vec3(1000,y,1000):Vec3(1000,y,-1000);
            triangle.v2=half?Vec3(-1000,y,1000):Vec3(1000,y,1000);
            if(layer)std::swap(triangle.v1,triangle.v2);
            triangle.material=1;host.triangles.push_back(triangle);
        }
        DeviceScene slab=uploadScene(host);uploadEnvironmentMap(slab,map,1,0);
        slabSun<<<samples/256,256>>>(slab.scene,values,samples);
        checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        double sum=0,squared=0;for(const auto& value:result){sum+=value.x;squared+=double(value.x)*value.x;}
        double expectedSun=integratedSun(sunX,sunY,map.width,map.height,50000,true)*host.materials[0].albedo.x;
        double sunMean=sum/samples,sunError=std::sqrt(std::max(0.,squared/samples-sunMean*sunMean)/samples);
        std::cout<<"Glass/HDR-sun mean="<<sunMean<<" independent target="<<expectedSun<<" stderr="<<sunError<<'\n';
        require(std::abs(sunMean-expectedSun)<.03*expectedSun,"Glass/HDR-sun energy disagrees with independent slab integral");
        require(sunError<.01*expectedSun,"Glass/HDR-sun continuation variance remains excessive");
        destroyDeviceScene(slab);
        // Finite GGX still uses its BSDF continuation proposal. Its direct
        // HDR lighting must retain the independently integrated MIS energy.
        host.materials[0].metallic=1;host.materials[0].roughness=.5f;host.materials[0].albedo=Vec3(.8f,.8f,.8f);
        host.triangles.clear();host.materials.pop_back();
        DeviceScene metalSun=uploadScene(host);uploadEnvironmentMap(metalSun,map,1,0);
        energy<<<samples/256,256>>>(metalSun.scene,values,direct,samples,Vec3(0,1,0));
        checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        sum=0;for(const auto& value:result)sum+=value.x;
        expectedSun=integratedSun(sunX,sunY,map.width,map.height,50000,false,true);
        require(std::abs(sum/samples-expectedSun)<.03*expectedSun,"HDR-sun rough-metal direct MIS energy mismatch");
        destroyDeviceScene(metalSun);
        host.materials[0].metallic=0;host.materials[0].roughness=0;host.materials[0].albedo=Vec3(.2f,.4f,.6f);
        // Direct illumination uses the same mixed continuation PDF in MIS.
        DeviceScene directSun=uploadScene(host);uploadEnvironmentMap(directSun,map,1,0);
        energy<<<samples/256,256>>>(directSun.scene,values,direct,samples,Vec3(0,1,0));
        checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        sum=0;for(const auto& value:result)sum+=value.x;
        expectedSun=integratedSun(sunX,sunY,map.width,map.height,50000,false)*host.materials[0].albedo.x;
        require(std::abs(sum/samples-expectedSun)<.03*expectedSun,"HDR-sun direct MIS energy mismatch");
        destroyDeviceScene(directSun);
        map.width=16;map.height=8;map.pixels.assign(128,make_float4(4,4,4,1));buildEnvironmentDistribution(map);
        // Glass terminal splitting integrates R + (1-R)*eta^2 exactly for a
        // constant HDR half-space, independent of the random Fresnel choice.
        host.spheres.clear();host.materials[0].type=MaterialType::Dielectric;host.materials[0].ior=1.5f;
        DeviceScene glass=uploadScene(host);uploadEnvironmentMap(glass,map,2.5f,0);
        glassTerminal<<<samples/256,256>>>(glass.scene,values,samples,12);
        checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        Vec3 expected=host.materials[0].albedo*(10*(.04f+.96f*4/9.f));
        for(const auto& value:result)require(length(value-expected)<1e-5f,"Glass terminal Fresnel energy or double counting");
        glassTerminal<<<samples/256,256>>>(glass.scene,values,samples,1);
        checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        for(const auto& value:result)require(lengthSquared(value)==0,"Glass terminal split bypassed depth cap");
        destroyDeviceScene(glass);
        map.pixels.assign(128,make_float4(50000,50000,50000,1));buildEnvironmentDistribution(map);
        glass=uploadScene(host);uploadEnvironmentMap(glass,map,1,0);
        clampConnections<<<1,1>>>(glass.scene,values);
        checked(cudaMemcpy(result.data(),values,6*sizeof(Vec3),cudaMemcpyDeviceToHost));
        require(length(result[0]-Vec3(50000,50000,50000))<.1f,"Clamp dimmed a first-bounce HDR escape");
        require(length(result[1]-Vec3(10,10,10))<1e-4f,"Indirect HDR miss bypassed clamp");
        require(length(result[2]-result[0])<.1f,"Zero clamp changed raw HDR transport");
        expected=host.materials[0].albedo*(50000*(.04f+.96f*4/9.f));
        require(length(result[3]-expected)<.1f,"Clamp dimmed primary glass terminal branches");
        // Each terminal branch is a separate light contribution, as in Cycles.
        require(length(result[4]-Vec3(10,20,30))<1e-3f,"Split indirect glass/HDR branches bypassed clamp or changed hue");
        require(lengthSquared(result[5])==0,"Contribution clamp changed path throughput");
        destroyDeviceScene(glass);
        map.pixels.assign(128,make_float4(4,4,4,1));buildEnvironmentDistribution(map);
        host.spheres.emplace_back(Sphere{Vec3(),10,0});glass=uploadScene(host);uploadEnvironmentMap(glass,map,2.5f,0);
        glassTerminal<<<samples/256,256>>>(glass.scene,values,samples,12);
        checked(cudaMemcpy(result.data(),values,samples*sizeof(Vec3),cudaMemcpyDeviceToHost));
        for(const auto& value:result)require(lengthSquared(value)==0,"Glass terminal split ignored geometry occlusion");
        destroyDeviceScene(glass);
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
