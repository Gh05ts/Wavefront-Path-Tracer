#pragma once

// Lighting interventions for dataset_capture only. Normal rendering and render
// checkpoint fingerprints retain their existing behavior.
#include <cmath>
#include <stdexcept>
#include <string>
#include "json.hpp"
#include "scene/scene_host.hpp"

struct CaptureLighting {
    bool sky = true;
    bool area = false;
    Vec3 center;
    float size = 1.0f;
    Vec3 emission;

    nlohmann::json metadata() const {
        return {{"version", 1}, {"sky", sky}, {"area", area},
            {"center", {center.x, center.y, center.z}}, {"size", size},
            {"emission", {emission.x, emission.y, emission.z}}};
    }
};

inline CaptureLighting parseCaptureLighting(const nlohmann::json& root) {
    if (!root.is_object())
        throw std::invalid_argument("capture_lighting must be an object");
    for (auto it = root.begin(); it != root.end(); ++it)
        if (it.key() != "sky" && it.key() != "area" && it.key() != "center" &&
            it.key() != "size" && it.key() != "emission")
            throw std::invalid_argument("Unknown capture_lighting field: " + it.key());
    CaptureLighting result;
    auto boolean = [&](const char* key, bool fallback) {
        if (!root.contains(key)) return fallback;
        if (!root.at(key).is_boolean())
            throw std::invalid_argument(std::string("capture_lighting.") + key + " must be boolean");
        return root.at(key).get<bool>();
    };
    auto vector = [&](const char* key) {
        const auto& value = root.at(key);
        if (!value.is_array() || value.size() != 3)
            throw std::invalid_argument("Lighting vectors need three finite numbers");
        for (const auto& component : value)
            if (!component.is_number() || !std::isfinite(component.get<float>()))
                throw std::invalid_argument("Lighting vectors need three finite numbers");
        return Vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
    };
    result.sky = boolean("sky", true);
    result.area = boolean("area", false);
    if (result.area) {
        result.center = vector("center");
        result.emission = vector("emission");
        if (!root.contains("size") || !root.at("size").is_number())
            throw std::invalid_argument("Area light size is required and must be numeric");
        result.size = root.at("size").get<float>();
        float luminance = dot(result.emission, Vec3(0.2126f, 0.7152f, 0.0722f));
        if (!std::isfinite(result.size) || result.size <= 0 ||
            result.emission.x < 0 || result.emission.y < 0 || result.emission.z < 0 ||
            !(luminance > 0) || !std::isfinite(result.size * result.size * luminance))
            throw std::invalid_argument("Area light size/power must be finite and positive");
    } else if (root.contains("center") || root.contains("size") || root.contains("emission")) {
        throw std::invalid_argument("Area light parameters require area: true");
    }
    if (!result.sky && !result.area)
        throw std::invalid_argument("Capture lighting needs at least one active source");
    return result;
}

inline void applyCaptureLighting(HostScene& scene, const CaptureLighting& lighting) {
    if (!scene.lights.empty())
        throw std::invalid_argument("Capture lighting expects a scene without explicit lights");
    scene.blackBackground = !lighting.sky;
    if (!lighting.area) return;
    Material material{};
    material.type = MaterialType::Emissive;
    material.albedo = Vec3(1, 1, 1);
    material.emission = lighting.emission;
    material.ior = 1.0f;
    uint32_t materialIndex = static_cast<uint32_t>(scene.materials.size());
    scene.materials.push_back(material);
    const Vec3 u(lighting.size, 0, 0), v(0, 0, lighting.size);
    const Vec3 corner = lighting.center - 0.5f * (u + v);
    for (int index = 0; index < 2; ++index) {
        Triangle triangle{};
        triangle.v0 = corner;
        triangle.v1 = index == 0 ? corner + u : corner + u + v;
        triangle.v2 = index == 0 ? corner + u + v : corner + v;
        triangle.material = materialIndex;
        triangle.lightIndex = static_cast<uint32_t>(scene.lights.size());
        const float area = 0.5f * lighting.size * lighting.size;
        scene.staticTriangles.push_back(triangle);
        scene.lights.push_back(TriangleLight{triangle.v0, triangle.v1, triangle.v2,
            Vec3(0, -1, 0), area, materialIndex, 0.0f});
        scene.lightWeights.push_back(area * dot(lighting.emission, Vec3(0.2126f, 0.7152f, 0.0722f)));
    }
}

inline uint64_t captureLightingFingerprint(uint64_t base, const CaptureLighting& lighting) {
    for (unsigned char byte : std::string("capture-lighting-v1:") + lighting.metadata().dump()) {
        base ^= byte;
        base *= 1099511628211ull;
    }
    return base;
}
