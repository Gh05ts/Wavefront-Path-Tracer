#pragma once

#include <cstdint>
#include "../core/vec3.cuh"

enum class PunctualLightType : uint32_t { Point, Spot, Directional };

struct PunctualLight {
    PunctualLightType type = PunctualLightType::Point;
    Vec3 position;
    Vec3 direction = Vec3(0.0f, 0.0f, -1.0f); // direction of emission
    Vec3 intensity = Vec3(1.0f, 1.0f, 1.0f); // linear RGB * glTF intensity
    float range = 0.0f; // zero denotes unlimited
    float innerConeCos = 1.0f;
    float outerConeCos = 0.70710678118f;
};

struct PunctualLightSample {
    Vec3 direction;
    Vec3 incident;
    float distance = 1.0e30f;
};

__host__ __device__ inline PunctualLightSample samplePunctualLight(const PunctualLight& light, const Vec3& point) {
    PunctualLightSample result;
    if (light.type == PunctualLightType::Directional) {
        result.direction = -light.direction;
        result.incident = light.intensity;
        return result;
    }
    Vec3 offset = light.position - point;
    float distanceSquared = lengthSquared(offset);
    if (!(distanceSquared > 0.0f))
        return result;
    result.distance = sqrtf(distanceSquared);
    result.direction = offset / result.distance;
    float attenuation = 1.0f / distanceSquared;
    if (light.range > 0.0f) {
        float ratio = result.distance / light.range;
        attenuation *= fmaxf(0.0f, 1.0f - ratio * ratio * ratio * ratio);
    }
    if (light.type == PunctualLightType::Spot) {
        float cosine = dot(light.direction, -result.direction);
        float cone = fminf(1.0f, fmaxf(0.0f, (cosine - light.outerConeCos) /
            fmaxf(1.0e-6f, light.innerConeCos - light.outerConeCos)));
        attenuation *= cone * cone;
    }
    result.incident = light.intensity * attenuation;
    return result;
}
