// CPU-only material-model ablation. No GPU allocation or production shader edit.
#define WAVEFRONT_REPLAY_GLASS_MODE
#define main legacyPixelReplayMain
#include "cpu_replay.cpp"
#undef main

int main(int argc,char**argv){
    if(argc!=11){std::cerr<<"usage: appearance_tiles scene hdr manifest solid|thin spp x y width height output-prefix\n";return 2;}
    std::string mode=argv[4];if(mode!="solid"&&mode!="thin")return 2;
    diagnosticThinGlass=mode=="thin";
    unsigned count=std::stoul(argv[5]),x0=std::stoul(argv[6]),y0=std::stoul(argv[7]),w=std::stoul(argv[8]),h=std::stoul(argv[9]);
    auto source=loadGltfScene(argv[1]);
    if(!source.textures.empty())return 2;
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
    scene.environment={1,panorama.alias.data(),panorama.probabilities.data(),panorama.width,panorama.height,1,0};
    nlohmann::json meta;std::ifstream(argv[3])>>meta;auto c=meta["camera"];
    Camera camera{vec(c["position"]),vec(c["forward"]),vec(c["right"]),vec(c["up"]),c["vertical_fov_radians"],c["aspect_ratio"]};
    unsigned width=meta["width"],height=meta["height"],start=meta["render"]["reference_sample_start"];
    if(x0+w>width||y0+h>height)return 2;
    std::vector<Vec3> result(w*h);
    for(unsigned row=0;row<h;++row)for(unsigned col=0;col<w;++col){
        unsigned x=x0+col,y=y0+row,pixel=y*width+x;
        Vec3 sum;
        for(unsigned sample=start;sample<start+count;++sample){
            auto rng=makeRngState(pixelRngKey(pixel,sample,RngStrategy::Pcg32),RngStrategy::Pcg32);
            float u=(x+randomFloat(rng))/width,v=(y+randomFloat(rng))/height;
            PathState path{camera.generateRay(u,v),Vec3(1,1,1),Vec3(),0,0,rng,0xffffffffu,1,0,true,true};
            Vec3 frame;
            while(path.active){auto hit=traceRay(path.ray,scene);shadePersistentPath(path,hit,scene,12,5,false,false,&frame,PhotonGrid{},.06f);}
            sum+=frame;
        }
        result[row*w+col]=sum/float(count);
        if(col==w-1)std::cout<<"row="<<row+1<<'/'<<h<<'\n'<<std::flush;
    }
    // Explicit little-endian float RGB, row-major, with JSON dimensions.
    static_assert(sizeof(Vec3)==3*sizeof(float));
    std::ofstream binary(std::string(argv[10])+".rgb32f",std::ios::binary);
    binary.write(reinterpret_cast<const char*>(result.data()),result.size()*sizeof(Vec3));
    nlohmann::json record{{"mode",mode},{"samples",count},{"sample_start",start},{"box",{x0,y0,x0+w,y0+h}},
        {"manifest",argv[3]},{"width",w},{"height",h},{"transport","CPU replay of current shader; thin is an explicit diagnostic model override"}};
    std::ofstream(std::string(argv[10])+".json")<<record.dump(2)<<'\n';
}
