// CPU diagnostic only: replay existing device helper code, no CUDA allocations.
#include "renderer/renderer.cuh"
#include "scene/gltf_loader.hpp"
#include "scene/environment.hpp"
#include "json.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
using std::min;using std::max;
struct FakeIndex{unsigned x=0;} blockIdx,blockDim,threadIdx;
int __popc(unsigned x){return __builtin_popcount(x);}
HostEnvironment panorama;
template<class T>T tex2D(cudaTextureObject_t,float u,float v){
    float tx=u*panorama.width-.5f,ty=v*panorama.height-.5f;
    int x=int(floorf(tx)),y=int(floorf(ty));float fx=tx-x,fy=ty-y;
    float4 result{};
    for(int j=0;j<2;++j)for(int i=0;i<2;++i){
        int xx=(x+i+int(panorama.width))%panorama.width,yy=std::clamp(y+j,0,int(panorama.height)-1);
        auto p=panorama.pixels[yy*panorama.width+xx];float w=(i?fx:1-fx)*(j?fy:1-fy);
        result.x+=w*p.x;result.y+=w*p.y;result.z+=w*p.z;
    }return result;
}
template<class...Args>void appendWarpCompacted(Args&&...){}
Vec3 gatherCaustics(const Hit&,PhotonGrid,float){return Vec3();}
#ifdef WAVEFRONT_REPLAY_GLASS_MODE
#define dielectricInterface replaySolidDielectricInterface
#endif
#include "renderer/device_sampling.cuh"
#ifdef WAVEFRONT_REPLAY_GLASS_MODE
#undef dielectricInterface
bool diagnosticThinGlass=false;
DielectricInterface dielectricInterface(const Vec3& incident,const Vec3& geometric,Vec3 shading,float ior){
    if(!diagnosticThinGlass)return replaySolidDielectricInterface(incident,geometric,shading,ior);
    bool front=dot(incident,geometric)<0;
    auto boundary=replaySolidDielectricInterface(incident,front?geometric:-geometric,front?shading:-shading,ior);
    boundary.frontFace=front;boundary.transmission=incident;boundary.eta=1;
    return boundary;
}
#endif
#ifdef WAVEFRONT_HAS_DIELECTRIC_INTERFACE
#define continueDielectricSurface integratedDielectricSurface
#include "renderer/device_traversal.cuh"
#undef continueDielectricSurface
bool diagnosticNoTerminalSplit=false;
Vec3 continueDielectricSurface(PathState& path,const Hit& hit,const Hit& shadedHit,
    const Material& material,Scene scene,uint32_t maxDepth){
    // Diagnostic ablation only: preserve the stochastic branch choice and all
    // geometry; later misses still use the original environment in the shader.
    if(diagnosticNoTerminalSplit)scene.environment.texture=0;
    auto direction=integratedDielectricSurface(path,hit,shadedHit,material,scene,maxDepth);
#ifdef WAVEFRONT_REPLAY_GLASS_MODE
    if(diagnosticThinGlass){path.mediumMaterial=invalidMaterialIndex;path.mediumDensity=1;}
#endif
    return direction;
}
#else
#include "renderer/device_traversal.cuh"
bool diagnosticNoTerminalSplit=false;
#endif
#include "renderer/device_path_shading.cuh"
Vec3 vec(const nlohmann::json& j){return Vec3(j[0],j[1],j[2]);}
// Probe sub-millimeter boundaries without the production intersection cutoff.
void probeThin(const std::vector<Triangle>& triangles,Vec3 origin,Vec3 direction,std::ostream& out){
    std::vector<std::pair<float,std::pair<unsigned,float>>> hits;
    for(const auto& triangle:triangles){
        Vec3 e1=triangle.v1-triangle.v0,e2=triangle.v2-triangle.v0,p=cross(direction,e2);
        float det=dot(e1,p);if(fabsf(det)<1e-8f)continue;
        Vec3 offset=origin-triangle.v0;float u=dot(offset,p)/det;if(u<0 || u>1)continue;
        Vec3 q=cross(offset,e1);float v=dot(direction,q)/det;if(v<0 || u+v>1)continue;
        float t=dot(e2,q)/det;if(t<1e-5f)continue;
        hits.push_back({t,{triangle.material,dot(direction,normalize(cross(e1,e2)))}});
    }
    std::sort(hits.begin(),hits.end());
    out<<" thin_probe=";
    for(size_t i=0;i<std::min(size_t(3),hits.size());++i)
        out<<'['<<hits[i].first<<','<<hits[i].second.first<<','<<hits[i].second.second<<']';
}
int main(int argc,char**argv){
    if(argc<4){std::cerr<<"usage: cpu_replay scene hdr manifest [count [start [x y [roulette-start [original|no-sun|constant|no-split]]]]]\n";return 2;}
    const int rouletteStart=argc>8?std::stoi(argv[8]):5;
    const std::string mode=argc>9?argv[9]:"original";
    if(mode!="original" && mode!="no-sun" && mode!="constant" && mode!="no-split")return 2;
    diagnosticNoTerminalSplit=mode=="no-split";
    auto source=loadGltfScene(argv[1]);
    if(!source.textures.empty()){std::cerr<<"CPU diagnostic supports texture-free glTF only\n";return 1;}
    std::vector<Triangle> triangles;
    for(auto& instance:source.instances)for(auto t:source.meshes[instance.meshIndex].triangles){
        auto& f=instance.transform;
        t.v0=transformVector(f,t.v0)+f.translation;t.v1=transformVector(f,t.v1)+f.translation;t.v2=transformVector(f,t.v2)+f.translation;
        t.n0=transformNormal(f,t.n0);t.n1=transformNormal(f,t.n1);t.n2=transformNormal(f,t.n2);
        if(instance.materialOverride!=invalidMaterialIndex)t.material=instance.materialOverride;
        triangles.push_back(t);
    }
    auto bvh=buildTriangleBvh(triangles);Scene scene{};
    scene.triangles=triangles.data();scene.triangleCount=triangles.size();
    scene.bvhNodes=bvh.nodes.data();scene.bvhNodeCount=bvh.nodes.size();scene.bvhTriangleIndices=bvh.triangleIndices.data();
    scene.materials=source.materials.data();scene.useNormalMaps=true;
    panorama=loadEnvironmentMap(argv[2]);
    // Keep the original alias table and RNG draws so these radiance-only
    // ablations follow exactly the same sampled geometry. No asset is edited.
    for(auto& p:panorama.pixels){
        if(mode=="constant")p=make_float4(1,1,1,1);
        else if(mode=="no-sun" && std::max({p.x,p.y,p.z})>100)p=make_float4(0,0,0,1);
    }
    scene.environment=EnvironmentView{1,panorama.alias.data(),panorama.probabilities.data(),panorama.width,panorama.height,1,0};
    nlohmann::json meta;std::ifstream(argv[3])>>meta;auto c=meta["camera"];
    Camera camera{vec(c["position"]),vec(c["forward"]),vec(c["right"]),vec(c["up"]),c["vertical_fov_radians"],c["aspect_ratio"]};
    uint32_t width=meta["width"],height=meta["height"];
    std::vector<std::pair<int,int>> pixels{{1790,922},{1727,911},{1359,826},{1839,861},{1428,820},{1600,950}};
    if(argc>7)pixels={{std::stoi(argv[6]),std::stoi(argv[7])}};
    for(auto xy:pixels){
        uint32_t x=xy.first,y=xy.second,pixel=y*width+x,start=meta["render"]["reference_sample_start"];
        uint32_t count=argc>4?std::stoul(argv[4]):uint32_t(meta["render"]["reference_samples"]);
        if(argc>5)start=std::stoul(argv[5]);
        Vec3 total;double redSum=0,redSquared=0;unsigned bright=0,sideMismatch=0,wrongSide=0,unpaired=0;float peak=0,unpairedPeak=-1;std::string worst,unpairedHistory;uint32_t worstSample=0,unpairedSample=0;
        for(uint32_t sample=start;sample<start+count;++sample){
            auto rng=makeRngState(pixelRngKey(pixel,sample,RngStrategy::Pcg32),RngStrategy::Pcg32);
            float u=(x+randomFloat(rng))/float(width),v=(y+randomFloat(rng))/float(height);
            PathState path{camera.generateRay(u,v),Vec3(1,1,1),Vec3(),0,0,rng,0xffffffffu,1,0,true,true};
            Vec3 framebuffer;std::ostringstream history;bool hadUnpaired=false;
            while(path.active){
                auto hit=traceRay(path.ray,scene);
                history<<"depth="<<path.depth<<" material="<<(hit.didHit?int(hit.hit.material):-1)
                    <<" type="<<(hit.didHit?int(scene.materials[hit.hit.material].type):-1)<<" T="<<path.throughput.x;
                if(hit.didHit)history<<" faceS="<<dot(path.ray.direction,hit.hit.normal)<<" faceG="<<dot(path.ray.direction,hit.hit.geometricNormal)
                    <<" t="<<hit.hit.t<<" P="<<hit.hit.position.x<<','<<hit.hit.position.y<<','<<hit.hit.position.z
                    <<" medium="<<path.mediumMaterial<<" roughness="<<scene.materials[hit.hit.material].roughness
                    <<" metallic="<<scene.materials[hit.hit.material].metallic;
                if(!hit.didHit)history<<" env="<<environmentRadiance(scene.environment,path.ray.direction).x
                    <<" envPdf="<<environmentPdf(scene.environment,path.ray.direction)
                    <<" previousPdf="<<path.previousBsdfPdf<<" specular="<<path.specularBounce;
                const Vec3 incoming=path.ray.direction;
                bool glass=hit.didHit && scene.materials[hit.hit.material].type==MaterialType::Dielectric;
                Vec3 outgoing,scatteringNormal=hit.didHit?hit.hit.normal:Vec3();
#ifdef WAVEFRONT_HAS_DIELECTRIC_INTERFACE
                if(glass){
                    auto boundary=dielectricInterface(incoming,hit.hit.geometricNormal,hit.hit.normal,scene.materials[hit.hit.material].ior);
                    auto copiedRng=path.rngState;
                    bool reflected=boundary.reflectance>=1 || boundary.reflectance>randomFloat(copiedRng);
                    outgoing=reflected?boundary.reflection:boundary.transmission;
                    scatteringNormal=boundary.normal;
                    history<<" fresnel="<<boundary.reflectance<<" eta="<<boundary.eta;
                    if(count==1){
                        for(int branch=0;branch<2;++branch){
                            Vec3 d=branch?boundary.transmission:boundary.reflection;
                            Ray branchRay(offsetRayOrigin(hit.hit.position,hit.hit.geometricNormal,d,hit.hit.t),d);
                            auto next=traceRay(branchRay,scene);
                            history<<(branch?" transmit":" reflect")<<"_env="<<environmentRadiance(scene.environment,d).x
                                <<" next="<<(next.didHit?int(next.hit.material):-1);
                        }
                    }
                }
#endif
                if(glass){
                    bool entering=dot(incoming,hit.hit.geometricNormal)<0;
                    sideMismatch+=entering!=(dot(incoming,hit.hit.normal)<0);
                    bool missing=(entering && path.mediumMaterial!=0xffffffffu)||(!entering && path.mediumMaterial==0xffffffffu);
                    unpaired+=missing;hadUnpaired|=missing;
                }
                float before=path.radiance.x;
                shadePersistentPath(path,hit,scene,12,rouletteStart,false,false,&framebuffer,PhotonGrid{},.06f);
#ifdef WAVEFRONT_HAS_DIELECTRIC_INTERFACE
                if(!glass)outgoing=path.ray.direction;
#else
                outgoing=path.ray.direction;
#endif
                history<<" delta="<<path.radiance.x-before<<" nextT="<<path.throughput.x<<" nextPdf="<<path.previousBsdfPdf;
                if(hit.didHit)history<<" outG="<<dot(outgoing,hit.hit.geometricNormal)
                    <<" turn="<<dot(incoming,outgoing);
                if(count==1 && glass)probeThin(triangles,hit.hit.position,outgoing,history);
                if(glass){
                    bool entering=dot(incoming,hit.hit.geometricNormal)<0;
                    bool transmitted=(dot(incoming,scatteringNormal)*dot(outgoing,scatteringNormal))>0;
                    wrongSide+=transmitted ? ((dot(outgoing,hit.hit.geometricNormal)<0)!=entering)
                                          : ((dot(outgoing,hit.hit.geometricNormal)<0)==entering);
                }
                history<<"; ";
            }
            total+=framebuffer;
            redSum+=framebuffer.x;redSquared+=double(framebuffer.x)*framebuffer.x;
            bright+=framebuffer.x>100;
            if(framebuffer.x>peak){peak=framebuffer.x;worst=history.str();worstSample=sample;}
            if(hadUnpaired && framebuffer.x>unpairedPeak){unpairedPeak=framebuffer.x;unpairedHistory=history.str();unpairedSample=sample;}
        }
        std::cout<<std::setprecision(9)<<"pixel "<<x<<','<<y<<" mean="<<total.x/count<<','<<total.y/count<<','<<total.z/count
            <<" red_mean64="<<redSum/count<<" red_stderr="<<sqrt(std::max(0.0,(redSquared-redSum*redSum/count)/std::max(1u,count-1))/count)
            <<" red_gt100="<<bright<<" roulette_start="<<rouletteStart<<" mode="<<mode
            <<" side_mismatch="<<sideMismatch<<" wrong_side="<<wrongSide<<" unpaired="<<unpaired
            <<" max="<<peak<<" sample="<<worstSample<<" path: "<<worst<<'\n'<<std::flush;
        if(unpairedPeak>=0)std::cout<<"unpaired_max="<<unpairedPeak<<" sample="<<unpairedSample<<" path: "<<unpairedHistory<<'\n';
    }
}
