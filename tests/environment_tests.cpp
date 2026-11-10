#include "scene/environment.hpp"
#include "scene/scene_manifest.hpp"
#include "renderer/render_fingerprint.hpp"
#include "distributed/assets.hpp"
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <limits>

using Json = nlohmann::json;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
bool near(float a, float b, float tolerance = 1e-5f) { return std::abs(a-b) < tolerance; }
EnvironmentView view(const HostEnvironment& map, float rotation = 0) {
    return EnvironmentView{0,map.alias.data(),map.probabilities.data(),map.width,map.height,1,rotation};
}
HostEnvironment constant(float value) {
    HostEnvironment map; map.width=16; map.height=8;
    map.pixels.assign(map.width*map.height, make_float4(value,value,value,1));
    buildEnvironmentDistribution(map); return map;
}
template<class F> void rejects(F&& f, const char* message) {
    bool failed=false; try { f(); } catch (const std::exception&) { failed=true; }
    require(failed,message);
}

void testDistribution() {
    HostEnvironment map=constant(8);
    EnvironmentView env=view(map);
    double solidAngle=0, probability=0;
    for (uint32_t y=0;y<map.height;++y) solidAngle+=map.width*environmentSolidAngle(y,map.width,map.height);
    for (float p:map.probabilities) probability+=p;
    require(std::abs(solidAngle-4*3.141592653589793)<1e-5 && std::abs(probability-1)<1e-6, "Environment normalization");
    for (Vec3 direction:{Vec3(1,0,0),Vec3(-1,0,0),Vec3(0,1,0),Vec3(0,-1,0),Vec3(0,0,1)})
        require(near(environmentPdf(env,direction),1/(4*3.14159265359f)), "Constant environment PDF must be uniform on sphere");
    require(near(environmentUv(Vec3(1,0,0),0).x,.5f), "+X panorama convention");
    require(near(environmentUv(Vec3(0,0,1),0).x,.75f), "+Z panorama convention");
    require(near(environmentUv(Vec3(0,0,-1),3.14159265359f/2).x,.5f), "Environment yaw convention");
    auto black=constant(0);
    require(near(environmentPdf(view(black),Vec3(0,1,0)),1/(4*3.14159265359f)), "Black-map uniform fallback");
    std::mt19937 generator(41); std::uniform_real_distribution<float> random(0,1);
    double irradiance=0;
    for (int i=0;i<200000;++i) {
        float a=random(generator),b=random(generator),c=random(generator),d=random(generator);
        auto sample=sampleEnvironment(env,a,b,c,d);
        require(near(sample.pdf,environmentPdf(env,sample.direction)), "Sample and evaluated environment PDF disagree");
        irradiance+=fmaxf(0,sample.direction.y)/sample.pdf;
    }
    require(std::abs(irradiance/200000-3.14159265359)<.025, "Constant-map irradiance integral");
    map=constant(0); map.pixels[0]=make_float4(10000,10000,10000,1);buildEnvironmentDistribution(map);
    env=view(map,.37f);
    int bright=0;
    for(int i=0;i<20000;++i) {
        float a=random(generator),b=random(generator),c=random(generator),d=random(generator);
        auto sample=sampleEnvironment(env,a,b,c,d);auto index=environmentTexel(env,sample.direction);
        bright+=index==0;
        require(std::abs(sample.pdf-environmentPdf(env,sample.direction))<1e-3f, "Rotated near-pole sample/PDF mismatch");
    }
    require(std::abs(bright/20000.f-map.probabilities[0])<.015f,"Filtered bright-cell alias frequency");
    require(map.probabilities[map.width*map.height-1]>0,"Black texels retain sampling support");
    require(map.probabilities[map.width-1]>.01f && map.probabilities[1]>.01f && map.probabilities[map.width]>.01f,
        "Bilinear sun shoulders must receive importance mass, including longitude wrap and pole clamp");
    map.pixels[0].x=-1; rejects([&]{buildEnvironmentDistribution(map);},"Negative HDR radiance accepted");
    map.pixels[0].x=std::numeric_limits<float>::infinity();
    rejects([&]{buildEnvironmentDistribution(map);},"Infinite HDR radiance accepted");
    HostEnvironment empty;rejects([&]{buildEnvironmentDistribution(empty);},"Empty map accepted");
}

void writeHdr(const std::filesystem::path& path) {
    std::ofstream out(path,std::ios::binary);
    out << "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n";
    const unsigned char texels[][4]={{128,64,32,133},{128,128,128,129},{128,128,128,128},{128,128,128,130},
        {128,128,128,129},{128,128,128,129},{128,128,128,129},{128,128,128,129}};
    out.write(reinterpret_cast<const char*>(texels),sizeof texels);
}

void testFiles(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);writeHdr(root/"test.hdr");
    auto map=loadEnvironmentMap((root/"test.hdr").c_str());
    require(map.width==4 && map.height==2 && near(map.pixels[0].x,16) && near(map.pixels[0].y,8),"HDR decode must preserve values >1");
    rejects([&]{loadEnvironmentMap((root/"missing.hdr").c_str());},"Missing map accepted");
    {std::ofstream out(root/"ldr.ppm");out<<"P3\n1 1\n255\n255 255 255\n";}
    rejects([&]{loadEnvironmentMap((root/"ldr.ppm").c_str());},"LDR image silently treated as linear HDR");
    auto manifest=root/"scene.json";
    Json document={{"name","environment-test"},{"preset","demo"},
        {"environment",{{"file","test.hdr"},{"intensity",2.5},{"rotation_degrees",90}}}};
    auto write=[&]{std::ofstream out(manifest);out<<document.dump();};write();
    SceneConfig config;std::string error;
    require(loadSceneManifest(manifest.c_str(),config,&error),"Environment manifest rejected");
    require(config.environmentFilename==(root/"test.hdr").string() && config.environmentIntensity==2.5f && config.environmentRotationDegrees==90,"Environment settings/path parsing");
    uint64_t fingerprint=computeSceneFingerprint(config);
    config.environmentIntensity*=2;require(computeSceneFingerprint(config)!=fingerprint,"Intensity missing from fingerprint");config.environmentIntensity/=2;
    config.environmentRotationDegrees+=1;require(computeSceneFingerprint(config)!=fingerprint,"Rotation missing from fingerprint");config.environmentRotationDegrees-=1;
    {std::ofstream file(root/"test.hdr",std::ios::app);file<<"changed";}
    require(computeSceneFingerprint(config)!=fingerprint,"Map contents missing from fingerprint");writeHdr(root/"test.hdr");
    // A procedural scene can distribute its map without a geometry asset.
    DistributedAssetCatalog catalog;
    require(buildDistributedAssetCatalog(config,catalog,&error) && catalog.primaryRelativePath.empty() && catalog.files.size()==1 && catalog.files[0].relativePath==distributedEnvironmentAssetPath,"Environment-only distributed catalog");
    document["environment"]["intensity"]=-1;write();require(!loadSceneManifest(manifest.c_str(),config,&error),"Negative environment intensity accepted");
    document["environment"]["intensity"]=1;document["environment"]["rotation_degrees"]="bad";write();
    require(!loadSceneManifest(manifest.c_str(),config,&error),"Nonnumeric rotation accepted");
    document["environment"]["rotation_degrees"]=0;document["environment"]["unknown"]=true;write();
    require(!loadSceneManifest(manifest.c_str(),config,&error),"Unknown environment option accepted");
    document["environment"].erase("unknown");write();
}

int main(int argc,char** argv) {
    try {testDistribution();testFiles(argc>1?argv[1]:"/tmp/ntasad-environment-fixture");
        std::cout<<"Environment HDR loading, sampling, rotation, configuration, fingerprint and asset checks passed\n";return 0;
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
