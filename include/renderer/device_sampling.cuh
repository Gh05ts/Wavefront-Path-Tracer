#pragma once

// Header-backed device helpers are included into renderer.cu so nvcc can
// inline the hot path without cross-translation-unit device-call overhead.

__device__
Vec3 sampleCosineHemisphere(const Vec3& normal, RngState& rngState) {
    constexpr float twoPi = 6.28318530718f;

    float u1 = randomFloat(rngState);
    float u2 = randomFloat(rngState);

    float radius = sqrtf(u1);
    float phi = twoPi * u2;

    float x = radius * cosf(phi);
    float y = radius * sinf(phi);
    float z = sqrtf(1.0f - u1);

    Vec3 helper = fabsf(normal.x) > 0.1f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    Vec3 tangent = normalize(cross(helper, normal));
    Vec3 bitangent = cross(normal, tangent);

    return normalize(tangent * x + bitangent * y + normal * z);
}

__device__
Vec3 randomInUnitSphere(RngState& rngState) {
    while (true) {
        Vec3 point(2.0f * randomFloat(rngState) - 1.0f, 2.0f * randomFloat(rngState) - 1.0f, 2.0f * randomFloat(rngState) - 1.0f);

        if (lengthSquared(point) < 1.0f)
            return point;
    }
}

__device__
Vec3 reflect(const Vec3& incident, const Vec3& normal) {
    return incident - 2.0f * dot(incident, normal) * normal;
}

__device__
Vec3 refract(const Vec3& incident, const Vec3& normal, float eta) {
    float cosTheta = fminf(dot(-incident, normal), 1.0f);
    Vec3 perpendicular = eta * (incident + cosTheta * normal);
    Vec3 parallel = -sqrtf(fmaxf(0.0f, 1.0f - lengthSquared(perpendicular))) * normal;
    return perpendicular + parallel;
}

__device__
float schlickReflectance(float cosine, float refractionRatio) {
    float r0 = (1.0f - refractionRatio) / (1.0f + refractionRatio);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

// Unpolarized Fresnel for eta = incident IOR / transmitted IOR. Schlick's
// incident-angle approximation is especially inaccurate on dielectric exits.
__device__ float dielectricReflectance(float cosine, float eta) {
    if (eta == 1.0f) return 0.0f;
    cosine = fminf(1.0f, fmaxf(0.0f, cosine));
    float sin2Transmitted = eta * eta * fmaxf(0.0f, 1.0f - cosine * cosine);
    if (sin2Transmitted >= 1.0f) return 1.0f;
    float transmittedCosine = sqrtf(1.0f - sin2Transmitted);
    float perpendicular = (eta * cosine - transmittedCosine) / (eta * cosine + transmittedCosine);
    float parallel = (cosine - eta * transmittedCosine) / (cosine + eta * transmittedCosine);
    return 0.5f * (perpendicular * perpendicular + parallel * parallel);
}

#define WAVEFRONT_HAS_DIELECTRIC_INTERFACE 1
__device__ DielectricInterface dielectricInterface(const Vec3& incident,
    const Vec3& geometricNormal, Vec3 shadingNormal, float ior) {
    bool frontFace = dot(incident, geometricNormal) < 0.0f;
    Vec3 geometricFacing = frontFace ? geometricNormal : -geometricNormal;
    if (dot(shadingNormal, geometricNormal) < 0.0f) shadingNormal = -shadingNormal;
    Vec3 normal = frontFace ? shadingNormal : -shadingNormal;
    float eta = frontFace ? 1.0f / ior : ior;
    if (dot(incident, normal) >= 0.0f) normal = geometricFacing;
    float reflectance = dielectricReflectance(dot(-incident, normal), eta);
    Vec3 reflection = reflect(incident, normal);
    Vec3 transmission = refract(incident, normal, eta);
    // Smooth/normal-mapped normals may send reflection into the solid or
    // transmission back out of it. Recompute the entire interface consistently
    // before sampling either branch; winding alone defines the medium boundary.
    if (dot(reflection, geometricFacing) <= 0.0f ||
        (reflectance < 1.0f && dot(transmission, geometricFacing) >= 0.0f)) {
        normal = geometricFacing;
        reflectance = dielectricReflectance(dot(-incident, normal), eta);
        reflection = reflect(incident, normal);
        transmission = refract(incident, normal, eta);
    }
    return {normal, reflection, transmission, eta, reflectance, frontFace};
}
