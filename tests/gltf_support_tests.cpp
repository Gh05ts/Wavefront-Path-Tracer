#include "scene/gltf_loader.hpp"
#include "renderer/specular_glossiness.cuh"
#include "renderer/gltf_pbr.cuh"
#include "json.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

using Json = nlohmann::json;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
bool near(float a, float b, float tolerance = 1e-5f) { return std::abs(a - b) < tolerance; }

void testLights() {
    PunctualLight light;
    light.position = Vec3(0, 0, 2);
    light.intensity = Vec3(8, 4, 2);
    require(near(samplePunctualLight(light, Vec3()).incident.x, 2), "Point inverse-square falloff");
    require(near(samplePunctualLight(light, Vec3(0, 0, -2)).incident.x, .5f), "Point distance doubling");
    light.range = 2;
    require(samplePunctualLight(light, Vec3()).incident.x == 0, "Light range cutoff");
    light.range = 0;
    light.type = PunctualLightType::Spot;
    light.innerConeCos = cosf(.2f); light.outerConeCos = cosf(.4f);
    require(near(samplePunctualLight(light, Vec3()).incident.x, 2), "Spot axis intensity");
    require(samplePunctualLight(light, Vec3(3, 0, 0)).incident.x == 0, "Spot outside cone");
    float angle = .3f;
    float cone = (cosf(angle) - light.outerConeCos) / (light.innerConeCos - light.outerConeCos);
    Vec3 point = light.position + Vec3(sinf(angle), 0, -cosf(angle)) * 2;
    require(near(samplePunctualLight(light, point).incident.x, 2 * cone * cone), "Spot smooth falloff");
    light.type = PunctualLightType::Directional;
    require(near(samplePunctualLight(light, Vec3(100, 0, 0)).incident.x, 8), "Directional no falloff");
    require(samplePunctualLight(light, Vec3()).direction.z == 1, "Directional sign");
}

void testBsdf() {
    Vec3 normal(0, 0, 1), view = normalize(Vec3(.3f, 0, 1)), incoming = normalize(Vec3(-.2f, .1f, 1));
    SpecularGlossinessSurface surface{Vec3(.6f, .3f, .1f), Vec3(.04f, .08f, .12f), .5f};
    float pdf, reversePdf;
    Vec3 f = evaluateSpecGloss(surface, normal, view, incoming, pdf);
    Vec3 reverse = evaluateSpecGloss(surface, normal, incoming, view, reversePdf);
    require(pdf > 0 && near(f.x, reverse.x) && near(f.z, reverse.z), "BSDF reciprocity");
    require(evaluateSpecGloss(surface, normal, view, -normal, pdf).x == 0 && pdf == 0, "BSDF hemisphere");
    surface.roughness = .1f;
    float sharp = evaluateSpecGloss(surface, normal, normal, normal, pdf).x;
    surface.roughness = .7f;
    require(sharp > evaluateSpecGloss(surface, normal, normal, normal, pdf).x, "Glossiness controls highlight width");
    surface.roughness = 0;
    require(std::isfinite(evaluateSpecGloss(surface, normal, normal, normal, pdf).x), "Perfect gloss limit finite");
    surface.roughness = .5f;
    // Independent uniform-hemisphere integral versus the mixture's f*cos/pdf.
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> uniform(0, 1);
    double integral = 0, sampled = 0;
    constexpr int count = 200000;
    for (int i = 0; i < count; ++i) {
        float z = uniform(generator), phi = 6.28318530718f * uniform(generator);
        float r = sqrtf(1-z*z);
        Vec3 wi(r*cosf(phi), r*sinf(phi), z);
        integral += evaluateSpecGloss(surface, normal, view, wi, pdf).x * z * 6.28318530718f;
        float choice = uniform(generator), u = uniform(generator), v = uniform(generator);
        wi = sampleSpecGloss(surface, normal, view, choice, u, v);
        Vec3 value = evaluateSpecGloss(surface, normal, view, wi, pdf);
        if (pdf > 0) sampled += value.x * dot(normal, wi) / pdf;
    }
    require(std::abs(integral/count - sampled/count) < .01, "BSDF sampling estimator disagrees with quadrature");
    require(sampled/count > .4 && sampled/count < 1.0, "BSDF energy sanity");
}

void testPbrBsdf() {
    const Vec3 n(0,0,1),view=normalize(Vec3(.3f,0,1));
    GltfPbrSurface s{Vec3(.5f,.3f,.1f),n,n,.3f,.55f,Vec3(.04f,.04f,.04f),1,.8f,.4f};
    std::mt19937 generator(139);
    std::uniform_real_distribution<float> uniform(0,1);
    double reference=0,estimate=0;
    constexpr double pi=3.14159265358979323846;
    constexpr int count=300000;
    // Independent double-precision BRDF integral (no production D/G/F calls).
    auto kernel=[](double nv,double nl,double nh,double rough) {
        double a2=rough*rough*rough*rough;
        double d=a2/(pi*std::pow(1-nh*nh+a2*nh*nh,2));
        auto g=[a2](double z){return 2*z/(z+std::sqrt(a2+(1-a2)*z*z));};
        return d*g(nv)*g(nl)/(4*nv*nl);
    };
    double nv=dot(n,view),coat=.8*(.04+.96*std::pow(1-nv,5));
    for (int i=0;i<count;++i) {
        double z=uniform(generator),phi=2*pi*uniform(generator),r=std::sqrt(1-z*z);
        Vec3 light(r*std::cos(phi),r*std::sin(phi),z),half=normalize(view+light);
        double vh=dot(view,half),nh=half.z,w=std::pow(1-vh,5);
        double fd=.04+.96*w,fm=.5+.5*w;
        double base=(1-.3)*(1-fd)*.5/pi + ((1-.3)*fd+.3*fm)*kernel(nv,z,nh,.55);
        reference+=((1-coat)*base+coat*kernel(nv,z,nh,.4))*z*2*pi;
        float choice=uniform(generator),u=uniform(generator),v=uniform(generator);
        auto weights=pbrWeights(s,view);
        if (choice<weights.coat) light=pbrSampleReflection(n,view,s.coatRoughness,u,v);
        else if (choice<weights.coat+weights.specular) light=pbrSampleReflection(n,view,s.roughness,u,v);
        else light=specGlossBasis(n,std::sqrt(u)*cosf(2*pi*v),std::sqrt(u)*sinf(2*pi*v),std::sqrt(1-u));
        float pdf;
        Vec3 value=evaluateGltfPbrCos(s,view,light,fmaxf(0.f,light.z)/pi,pdf);
        if (pdf>0) estimate+=value.x/pdf;
    }
    require(std::abs(reference/count-estimate/count)<.007,"glTF layered sampler disagrees with independent quadrature");
    require(estimate/count>0 && estimate/count<1,"glTF layered energy sanity");
    // Factor zero really disables grazing reflection; no metal threshold.
    s.metallic=0;s.coat=0;s.dielectricF0=Vec3();s.dielectricF90=0;
    float pdf;Vec3 value=evaluateGltfPbrCos(s,n,n,1/pi,pdf);
    require(near(value.x,.5/pi)&&near(pdf,1/pi),"Specular zero must recover Lambert exactly");
    s.metallic=.25f;s.roughness=0;s.coat=1;s.coatRoughness=0;
    auto weights=pbrWeights(s,n);
    require(near(weights.diffuse,.96*.75)&&near(weights.specular,.96*.25)&&near(weights.coat,.04),"Mixed metal/coat delta selection masses");
    value=evaluateGltfPbrCos(s,n,n,1/pi,pdf);
    require(near(value.x,.96*.75*.5/pi)&&near(pdf,.96*.75/pi),"Delta lobes must be excluded from finite NEE/PDF");
    s.metallic=1;s.coat=0;
    value=evaluateGltfPbrCos(s,n,n,1/pi,pdf);
    require(lengthSquared(value)==0&&pdf==0,"Perfect metal must retain an exact delta");
    // BMW inner chrome can have a nearly grazing smooth normal while the
    // actual triangle faces the ray. Its ideal reflection was below Ng.
    Vec3 smooth=normalize(Vec3(-.57f,0,.82f)),oblique=normalize(Vec3(.7846f,0,.62f));
    require(dot(n,2.f*dot(smooth,oblique)*smooth-oblique)<-.4f,"Bent-normal fixture must expose invalid mirror reflection");
    Vec3 corrected=gltfReflectionNormal(smooth,n,oblique);
    float reflectedCos=dot(n,2.f*dot(corrected,oblique)*corrected-oblique);
    require(reflectedCos>=.009999f&&reflectedCos<.01001f,"Corrected reflection must meet geometric horizon with minimal rotation");
    require(length(gltfReflectionNormal(n,n,oblique)-n)==0,"Valid reflective frame must remain unchanged");
    s.normal=s.coatNormal=smooth;
    correctGltfReflectionNormals(s,n,oblique);
    require(length(s.normal-smooth)==0&&length(pbrSpecularNormal(s)-corrected)<1.e-6f,"Diffuse normal must remain authored independently of reflective frame");
    s.coat=1;correctGltfReflectionNormals(s,n,oblique);
    require(length(s.coatNormal-corrected)<1.e-6f,"Coat needs its own valid reflective frame");
}

void testImport(const std::filesystem::path& root) {
    std::filesystem::create_directories(root);
    // One upward-facing quad, plus a small external RGB texture.
    float positions[] = {-2,0,-2, 2,0,-2, 2,0,2, -2,0,2};
    uint16_t indices[] = {0,2,1, 0,3,2};
    { std::ofstream binary(root/"quad.bin", std::ios::binary);
      binary.write(reinterpret_cast<char*>(positions), sizeof positions);
      binary.write(reinterpret_cast<char*>(indices), sizeof indices); }
    { std::ofstream texture(root/"pixel.ppm", std::ios::binary);
      texture << "P6\n1 1\n255\n"; const char rgb[] = {char(128), char(64), char(32)}; texture.write(rgb,3); }
    Json doc = Json::parse(R"json({
  "asset": {
    "version": "2.0"
  },
  "extensionsUsed": [
    "KHR_materials_pbrSpecularGlossiness",
    "KHR_lights_punctual",
    "KHR_materials_transmission"
  ],
  "buffers": [
    {
      "uri": "quad.bin",
      "byteLength": 60
    }
  ],
  "bufferViews": [
    {
      "buffer": 0,
      "byteOffset": 0,
      "byteLength": 48
    },
    {
      "buffer": 0,
      "byteOffset": 48,
      "byteLength": 12
    }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "componentType": 5126,
      "count": 4,
      "type": "VEC3"
    },
    {
      "bufferView": 1,
      "componentType": 5123,
      "count": 6,
      "type": "SCALAR"
    }
  ],
  "images": [
    {
      "uri": "pixel.ppm"
    }
  ],
  "textures": [
    {
      "source": 0
    }
  ],
  "materials": [
    {
      "pbrMetallicRoughness": {
        "baseColorFactor": [
          1,
          0,
          0,
          1
        ]
      },
      "alphaMode": "MASK",
      "extensions": {
        "KHR_materials_pbrSpecularGlossiness": {
          "diffuseFactor": [
            0.5,
            0.4,
            0.3,
            0.6
          ],
          "specularFactor": [
            0.04,
            0.08,
            0.12
          ],
          "glossinessFactor": 0.75,
          "diffuseTexture": {
            "index": 0
          },
          "specularGlossinessTexture": {
            "index": 0,
            "extensions": {
              "KHR_texture_transform": {
                "offset": [
                  0.1,
                  0.2
                ],
                "texCoord": 1
              }
            }
          }
        }
      }
    },
    {
      "extensions": {
        "KHR_materials_pbrSpecularGlossiness": {}
      }
    },
    {
      "extensions": {
        "KHR_materials_transmission": {
          "transmissionFactor": 1
        },
        "KHR_materials_ior": {
          "ior": 1.45
        }
      }
    },
    {
      "pbrMetallicRoughness": {
        "metallicFactor": 0.8,
        "roughnessFactor": 0.3
      }
    }
  ],
  "meshes": [
    {
      "primitives": [
        {
          "attributes": {
            "POSITION": 0
          },
          "indices": 1,
          "material": 0
        }
      ]
    }
  ],
  "nodes": [
    {
      "mesh": 0
    },
    {
      "translation": [
        0,
        3,
        0
      ],
      "scale": [
        2,
        3,
        4
      ],
      "rotation": [
        -0.7071067811865476,
        0,
        0,
        0.7071067811865476
      ],
      "children": [
        2,
        3,
        4
      ]
    },
    {
      "extensions": {
        "KHR_lights_punctual": {
          "light": 0
        }
      }
    },
    {
      "translation": [
        1,
        0,
        0
      ],
      "extensions": {
        "KHR_lights_punctual": {
          "light": 1
        }
      }
    },
    {
      "extensions": {
        "KHR_lights_punctual": {
          "light": 2
        }
      }
    },
    {
      "extensions": {
        "KHR_lights_punctual": {
          "light": 3
        }
      }
    }
  ],
  "scenes": [
    {
      "nodes": [
        0,
        1
      ]
    }
  ],
  "scene": 0,
  "extensions": {
    "KHR_lights_punctual": {
      "lights": [
        {
          "type": "point",
          "color": [
            1,
            0.5,
            0.25
          ],
          "intensity": 8,
          "range": 10
        },
        {
          "type": "spot",
          "intensity": 4,
          "spot": {
            "innerConeAngle": 0.2,
            "outerConeAngle": 0.4
          }
        },
        {
          "type": "directional",
          "intensity": 2
        },
        {
          "type": "point",
          "intensity": 999
        }
      ]
    }
  }
})json");
    doc["materials"][3]["extensions"] = {
        {"KHR_materials_specular", {{"specularFactor",.4},{"specularColorFactor",{2,.8,.6}},
            {"specularTexture",{{"index",0}}},{"specularColorTexture",{{"index",0}}}}},
        {"KHR_materials_clearcoat", {{"clearcoatFactor",.9},{"clearcoatRoughnessFactor",.2},
            {"clearcoatTexture",{{"index",0}}},{"clearcoatRoughnessTexture",{{"index",0}}},
            {"clearcoatNormalTexture",{{"index",0},{"scale",.7}}}}}};
    { std::ofstream file(root/"scene.gltf"); file << doc.dump(2); }
    GltfScene scene = loadGltfScene((root/"scene.gltf").c_str());
    require(scene.meshes.size()==1 && scene.instances.size()==1, "Geometry import regression");
    const Material& material = scene.materials.at(1);
    require(material.type==MaterialType::SpecularGlossiness && near(material.albedo.y,.4), "Spec-gloss takes precedence over core fallback");
    require(near(material.specular.z,.12) && near(material.glossiness,.75), "Spec-gloss factors");
    require(material.alphaMasked && near(material.opacity,.6), "Diffuse opacity factor");
    require(material.albedoTexture.texture==0 && material.specularGlossinessTexture.texture==0, "Spec-gloss textures");
    require(material.specularGlossinessTexture.texCoord==1 && near(material.specularGlossinessTexture.offset.x,.1), "Texture coordinate/transform import");
    const Material& defaults = scene.materials.at(2);
    require(defaults.glossiness==1 && defaults.specular.x==1 && defaults.albedo.x==1, "Spec-gloss defaults");
    require(scene.materials.at(3).type==MaterialType::Dielectric && near(scene.materials.at(3).ior,1.45), "Glass import regression");
    require(near(scene.materials.at(4).metallic,.8) && near(scene.materials.at(4).roughness,.3), "Core material regression");
    const auto& pbr=scene.materials.at(4);
    require(pbr.gltfPbrWorkflow&&!material.gltfPbrWorkflow&&!scene.materials.at(3).gltfPbrWorkflow,"Workflow routing");
    require(near(pbr.specularWeight,.4)&&near(pbr.specularColor.x,2)&&near(pbr.clearcoat,.9)&&near(pbr.clearcoatRoughness,.2),"Specular/clearcoat factors");
    require(pbr.specularWeightTexture.texture==0&&pbr.specularColorTexture.texture==0&&pbr.clearcoatTexture.texture==0&&pbr.clearcoatRoughnessTexture.texture==0,"Layer texture import");
    require(pbr.clearcoatNormalTexture.texture==0&&near(pbr.clearcoatNormalScale,.7),"Independent coat normal import");
    require(scene.punctualLights.size()==3, "Only active scene lights imported");
    require(near(scene.punctualLights[0].position.y,3) && near(scene.punctualLights[0].range,10), "Parent light transform / range independent of scale");
    require(near(scene.punctualLights[1].position.x,2) && near(scene.punctualLights[1].direction.y,-1), "Nested spot position/direction");
    require(near(scene.punctualLights[0].intensity.y,4), "Linear light color/intensity");
    require(scene.punctualLights[2].type==PunctualLightType::Directional, "Directional import");
}

int main(int argc, char** argv) {
    try {
        testLights(); testBsdf(); testPbrBsdf();
        testImport(argc > 1 ? argv[1] : "/tmp/ntasad-gltf-support-fixture");
        std::cout << "glTF materials, light transforms/falloff, and BSDF estimator checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
