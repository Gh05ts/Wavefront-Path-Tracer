#include "scene/environment.hpp"
#include "core/rng.cuh"
#include "renderer/device_sampling.cuh"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

Vec3 lookup(const HostEnvironment& map, Vec3 direction) {
    Vec2 uv=environmentUv(direction,0);
    float tx=uv.x*map.width-.5f,ty=uv.y*map.height-.5f;
    int x=int(floorf(tx)),y=int(floorf(ty));float fx=tx-x,fy=ty-y;
    Vec3 result;
    for(int j=0;j<2;++j)for(int i=0;i<2;++i){
        int xx=(x+i+int(map.width))%map.width,yy=std::clamp(y+j,0,int(map.height)-1);
        auto p=map.pixels[yy*map.width+xx];
        result+=Vec3(p.x,p.y,p.z)*((i?fx:1-fx)*(j?fy:1-fy));
    }return result;
}
int main(int argc,char**argv){
    auto map=loadEnvironmentMap(argv[1]);
    EnvironmentView env{0,map.alias.data(),map.probabilities.data(),map.width,map.height,1,0};
    float peak=0;size_t peakIndex=0;
    for(size_t i=0;i<map.pixels.size();++i)if(map.pixels[i].x>peak){peak=map.pixels[i].x;peakIndex=i;}
    std::cout<<"HDR peak "<<peak<<" at "<<peakIndex%map.width<<','<<peakIndex/map.width<<'\n';
    std::ofstream dump("/tmp/bmw-hdr-rgb.bin",std::ios::binary);
    dump.write(reinterpret_cast<char*>(map.pixels.data()),map.pixels.size()*sizeof(float4));
    for(int pixel: {922*1920+1790,911*1920+1727,826*1920+1359,861*1920+1839,820*1920+1428}){
        Vec3 normal(-.0144418813f,.9998676181f,.0075031393f); // guide at first bright floor pixel
        Vec3 sum;float maximum=0;uint32_t worst=0;Vec3 worstD;float worstPdf=0;Vec3 worstL;
        for(uint32_t sample=500000004;sample<500001028;++sample){
            auto rng=makeRngState(pixelRngKey(pixel,sample,RngStrategy::Pcg32),RngStrategy::Pcg32);
            for(int i=0;i<7;++i)randomFloat(rng); // camera jitter + environment NEE uniforms
            Vec3 d=sampleCosineHemisphere(normal,rng);
            float bsdf=std::max(0.f,dot(normal,d))*.31830988618f, pdf=environmentPdf(env,d);
            float weight=bsdf*bsdf/(bsdf*bsdf+pdf*pdf);
            auto L=lookup(map,d);auto value=L*(weight*.0183606343f);sum+=value;
            if(value.x>maximum){maximum=value.x;worst=sample;worstD=d;worstPdf=pdf;worstL=L;}
        }
        std::cout<<std::setprecision(9)<<"pixel "<<pixel%1920<<','<<pixel/1920<<" direct-BSDF sky-only mean "
            <<sum.x/1024<<','<<sum.y/1024<<','<<sum.z/1024<<" max "<<maximum<<" sample "<<worst
            <<" PDF "<<worstPdf<<" L "<<worstL.x<<','<<worstL.y<<','<<worstL.z
            <<" direction "<<worstD.x<<','<<worstD.y<<','<<worstD.z<<'\n';
    }
}
