#pragma once

#include "specular_glossiness.cuh"

// Opaque metallic/roughness + KHR_materials_specular/clearcoat. The coat uses
// the extension's documented NdotV Fresnel layering approximation. Zero
// roughness is a delta lobe, kept separate from the solid-angle density.
struct GltfPbrSurface {
    Vec3 color;
    Vec3 normal, coatNormal;
    float metallic, roughness;
    Vec3 dielectricF0;
    float dielectricF90;
    float coat, coatRoughness;
    // Separate reflective frame: diffuse keeps the authored smooth normal.
    Vec3 reflectionNormal{};
};

struct GltfPbrWeights { float diffuse, specular, coat; };

// A smooth/bump normal can put the entire mirror-lobe center below the real
// triangle. Clipping all such samples makes curved chrome appear black.
// Rotate only the reflective frame towards Ng until its ideal reflection is
// above a small horizon margin. Evaluation and sampling use the same frame.
// The margin follows Cycles' valid-reflection policy; bisection here avoids a
// fragile quadratic root near grazing. This changes the shading-normal model,
// rather than resampling rejected paths with an incorrect density.
__host__ __device__ inline Vec3 gltfReflectionNormal(Vec3 normal, Vec3 geometric, Vec3 view) {
    float geometricView=dot(geometric,view);
    if (geometricView<=0.f) return geometric;
    float target=fminf(.01f,.9f*geometricView);
    Vec3 reflected=2.f*dot(normal,view)*normal-view;
    if (dot(geometric,reflected)>=target) return normal;
    float lower=0.f,upper=1.f;
    #pragma unroll 1
    for (int i=0;i<24;++i) {
        float t=(lower+upper)*.5f;
        Vec3 candidate=normalize(normal*(1.f-t)+geometric*t);
        reflected=2.f*dot(candidate,view)*candidate-view;
        if (dot(geometric,reflected)>=target) upper=t; else lower=t;
    }
    return normalize(normal*(1.f-upper)+geometric*upper);
}

__host__ __device__ inline Vec3 pbrSpecularNormal(const GltfPbrSurface& s) {
    return lengthSquared(s.reflectionNormal)>0.f ? s.reflectionNormal : s.normal;
}

__host__ __device__ inline void correctGltfReflectionNormals(GltfPbrSurface& s,Vec3 geometric,Vec3 view) {
    s.reflectionNormal=gltfReflectionNormal(s.normal,geometric,view);
    if (s.coat>0.f) s.coatNormal=gltfReflectionNormal(s.coatNormal,geometric,view);
}

__host__ __device__ inline float pbrSaturate(float x) { return fminf(1.f, fmaxf(0.f, x)); }
__host__ __device__ inline float pbrSchlick(float cosine) {
    float x = 1.f - pbrSaturate(cosine);
    return x*x*x*x*x;
}
__host__ __device__ inline float pbrCoatWeight(const GltfPbrSurface& s, Vec3 view) {
    return s.coat * (.04f + .96f * pbrSchlick(dot(s.coatNormal, view)));
}
__host__ __device__ inline Vec3 pbrDielectricFresnel(const GltfPbrSurface& s, float cosine) {
    return s.dielectricF0 + (Vec3(s.dielectricF90,s.dielectricF90,s.dielectricF90)-s.dielectricF0)*pbrSchlick(cosine);
}
__host__ __device__ inline Vec3 pbrBaseFresnel(const GltfPbrSurface& s, float cosine) {
    Vec3 metal = s.color + (Vec3(1,1,1)-s.color)*pbrSchlick(cosine);
    return pbrDielectricFresnel(s,cosine)*(1.f-s.metallic)+metal*s.metallic;
}
__host__ __device__ inline GltfPbrWeights pbrWeights(const GltfPbrSurface& s, Vec3 view) {
    float coat = pbrCoatWeight(s,view);
    float specular = s.metallic + (1.f-s.metallic)*specGlossMax(pbrDielectricFresnel(s,dot(pbrSpecularNormal(s),view)));
    return {(1.f-coat)*(1.f-specular), (1.f-coat)*specular, coat};
}
__host__ __device__ inline float pbrAlpha(float roughness) {
    // Numerical floor for finite lobes only; exact zero takes the delta path.
    return fmaxf(1.e-4f,roughness*roughness);
}
__host__ __device__ inline float pbrReflectionPdf(Vec3 normal, Vec3 view, Vec3 light, float roughness) {
    if (roughness<=0 || dot(normal,view)<=0 || dot(normal,light)<=0 || lengthSquared(view+light)<1.e-12f) return 0;
    Vec3 half=normalize(view+light);
    float vh=dot(view,half), nh=dot(normal,half);
    return vh>0 && nh>0 ? specGlossD(pbrSaturate(nh),pbrAlpha(roughness))*nh/(4.f*vh) : 0.f;
}
__host__ __device__ inline float pbrReflectionKernel(Vec3 normal, Vec3 view, Vec3 light, float roughness) {
    float nv=dot(normal,view), nl=dot(normal,light);
    if (roughness<=0 || nv<=0 || nl<=0 || lengthSquared(view+light)<1.e-12f) return 0;
    float alpha=pbrAlpha(roughness), nh=pbrSaturate(dot(normal,normalize(view+light)));
    return specGlossD(nh,alpha)*specGlossG1(nv,alpha)*specGlossG1(nl,alpha)/(4.f*nv*nl);
}
__host__ __device__ inline Vec3 pbrSampleReflection(Vec3 normal, Vec3 view, float roughness, float u, float v) {
    if (roughness<=0) return 2.f*dot(normal,view)*normal-view;
    float alpha=pbrAlpha(roughness);
    float cosine=sqrtf((1.f-u)/((1.f-u)+alpha*alpha*u));
    float sine=sqrtf(fmaxf(0.f,1.f-cosine*cosine)), phi=6.28318530718f*v;
    Vec3 half=specGlossBasis(normal,sine*cosf(phi),sine*sinf(phi),cosine);
    return 2.f*dot(view,half)*half-view;
}

// Returns f * cosine in the path's measure, allowing distinct base/coat normals.
// The caller supplies the *actual* diffuse proposal density (including guiding).
__host__ __device__ inline Vec3 evaluateGltfPbrCos(const GltfPbrSurface& s, Vec3 view, Vec3 light,
    float diffusePdf, float& pdf) {
    GltfPbrWeights w=pbrWeights(s,view);
    float nl=fmaxf(0.f,dot(s.normal,light));
    Vec3 reflectionNormal=pbrSpecularNormal(s);
    pdf=w.diffuse*diffusePdf + w.specular*pbrReflectionPdf(reflectionNormal,view,light,s.roughness)
        + w.coat*pbrReflectionPdf(s.coatNormal,view,light,s.coatRoughness);
    Vec3 base;
    if (lengthSquared(view+light)>1.e-12f) {
        float vh=pbrSaturate(dot(view,normalize(view+light)));
        float diffuseWeight=(1.f-s.metallic)*(1.f-specGlossMax(pbrDielectricFresnel(s,vh)));
        if (dot(s.normal,view)>0) base=s.color*(diffuseWeight*nl*.31830988618f);
        float specularCos=fmaxf(0.f,dot(reflectionNormal,light));
        base+=pbrBaseFresnel(s,vh)*(pbrReflectionKernel(reflectionNormal,view,light,s.roughness)*specularCos);
    }
    float coatCos=fmaxf(0.f,dot(s.coatNormal,light));
    float coatValue=w.coat*pbrReflectionKernel(s.coatNormal,view,light,s.coatRoughness)*coatCos;
    return base*(1.f-w.coat)+Vec3(coatValue,coatValue,coatValue);
}
