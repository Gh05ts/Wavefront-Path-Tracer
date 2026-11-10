// Read-only CPU audit: compare raw-texel and filter-aware importance masses.
#include "scene/environment.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc,char**argv) {
    if(argc!=2)return 2;
    auto map=loadEnvironmentMap(argv[1]);
    auto luminance=[](const float4& p){return .2126*double(p.x)+.7152*double(p.y)+.0722*double(p.z);};
    double rawTotal=0;
    for(size_t i=0;i<map.pixels.size();++i)
        rawTotal+=luminance(map.pixels[i])*environmentSolidAngle(i/map.width,map.width,map.height);
    struct Cell{double ratio,proxy,rawPdf,newPdf;uint32_t index;};
    std::vector<Cell> top;
    double masses[3]={};float peak=0;
    for(uint32_t i=0;i<map.pixels.size();++i) {
        double omega=environmentSolidAngle(i/map.width,map.width,map.height);
        double rawMass=.99*luminance(map.pixels[i])*omega/rawTotal+.01*omega/12.5663706143591729538;
        double ratio=map.probabilities[i]/rawMass;
        for(int j=0;j<3;++j)if(ratio>std::pow(10.,j+1))masses[j]+=map.probabilities[i];
        peak=std::max(peak,map.pixels[i].x);
        if(top.size()<12 || ratio>top.back().ratio) {
            top.push_back({ratio,map.probabilities[i],rawMass/omega,map.probabilities[i]/omega,i});
            std::sort(top.begin(),top.end(),[](auto a,auto b){return a.ratio>b.ratio;});
            if(top.size()>12)top.pop_back();
        }
    }
    std::cout<<std::setprecision(10)<<"dimensions="<<map.width<<'x'<<map.height<<" peak_red="<<peak
        <<" filtered_importance_mass_in_raw_underweighted_cells_gt10_gt100_gt1000="
        <<masses[0]<<','<<masses[1]<<','<<masses[2]<<'\n';
    for(auto c:top)std::cout<<"cell="<<c.index%map.width<<','<<c.index/map.width
        <<" new/raw_mass="<<c.ratio<<" raw_luminance="<<luminance(map.pixels[c.index])
        <<" raw_pdf="<<c.rawPdf<<" filtered_pdf="<<c.newPdf<<'\n';
}
