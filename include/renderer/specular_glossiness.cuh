#pragma once

#include "../core/vec3.cuh"

// glTF specular-glossiness: Lambert diffuse plus isotropic GGX reflection.
// All colors are linear; perceptual roughness = 1 - glossiness.
struct SpecularGlossinessSurface {
    Vec3 diffuse;
    Vec3 f0;
    float roughness;
};

__host__ __device__ inline float specGlossMax(const Vec3& v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}

__host__ __device__ inline float specGlossProbability(const SpecularGlossinessSurface& s) {
    float specular = specGlossMax(s.f0);
    float diffuse = specGlossMax(s.diffuse) * (1.0f - specular);
    // Leave both lobes sampleable even for black F0 (Schlick grows at grazing).
    return fminf(0.95f, fmaxf(0.05f, specular / fmaxf(1.0e-6f, specular + diffuse)));
}

__host__ __device__ inline float specGlossAlpha(const SpecularGlossinessSurface& s) {
    float roughness = fmaxf(0.02f, s.roughness); // finite GGX, including glossiness=1
    return roughness * roughness;
}

__host__ __device__ inline float specGlossD(float nh, float alpha) {
    constexpr float pi = 3.14159265359f;
    float a2 = alpha * alpha;
    float denominator = (1.0f - nh * nh) + a2 * nh * nh;
    return a2 / (pi * denominator * denominator);
}

__host__ __device__ inline float specGlossG1(float cosine, float alpha) {
    return 2.0f * cosine / (cosine + sqrtf(alpha * alpha + (1.0f - alpha * alpha) * cosine * cosine));
}

__host__ __device__ inline Vec3 evaluateSpecGloss(const SpecularGlossinessSurface& s,
    const Vec3& n, const Vec3& outgoing, const Vec3& incoming, float& pdf) {
    constexpr float invPi = 0.31830988618f;
    pdf = 0.0f;
    float nv = dot(n, outgoing), nl = dot(n, incoming);
    Vec3 sum = outgoing + incoming;
    if (nv <= 0.0f || nl <= 0.0f || lengthSquared(sum) < 1.0e-12f)
        return Vec3();
    Vec3 half = normalize(sum);
    float nh = fminf(1.0f, fmaxf(0.0f, dot(n, half)));
    float vh = fminf(1.0f, fmaxf(1.0e-8f, dot(outgoing, half)));
    float alpha = specGlossAlpha(s);
    float d = specGlossD(nh, alpha);
    float probability = specGlossProbability(s);
    pdf = (1.0f - probability) * nl * invPi + probability * d * nh / (4.0f * vh);
    float schlick = powf(1.0f - vh, 5.0f);
    Vec3 fresnel = s.f0 + (Vec3(1.0f, 1.0f, 1.0f) - s.f0) * schlick;
    Vec3 diffuse = s.diffuse * ((1.0f - specGlossMax(s.f0)) * invPi);
    float g = specGlossG1(nv, alpha) * specGlossG1(nl, alpha);
    return diffuse + fresnel * (d * g / (4.0f * nv * nl));
}

__host__ __device__ inline Vec3 specGlossBasis(const Vec3& n, float x, float y, float z) {
    Vec3 helper = fabsf(n.x) > 0.1f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 tangent = normalize(cross(helper, n));
    return normalize(x * tangent + y * cross(n, tangent) + z * n);
}

__host__ __device__ inline Vec3 sampleSpecGloss(const SpecularGlossinessSurface& s,
    const Vec3& n, const Vec3& outgoing, float choice, float u, float v) {
    float phi = 6.28318530718f * v;
    if (choice < specGlossProbability(s)) {
        float alpha = specGlossAlpha(s);
        float cosine = sqrtf((1.0f - u) / (1.0f + (alpha * alpha - 1.0f) * u));
        float sine = sqrtf(fmaxf(0.0f, 1.0f - cosine * cosine));
        Vec3 half = specGlossBasis(n, sine * cosf(phi), sine * sinf(phi), cosine);
        return 2.0f * dot(outgoing, half) * half - outgoing;
    }
    float radius = sqrtf(u);
    return specGlossBasis(n, radius * cosf(phi), radius * sinf(phi), sqrtf(1.0f - u));
}
