#include "renderer/capture_lighting.hpp"
#include <iostream>
#include <limits>

void require(bool condition) {
    if (!condition) throw std::runtime_error("Capture lighting invariant failed");
}

int main() {
    using Json = nlohmann::json;
    const Json description = {{"sky", false}, {"area", true}, {"center", {0, 6, 0}},
        {"size", 4}, {"emission", {24, 24, 24}}};
    auto lighting = parseCaptureLighting(description);
    HostScene scene;
    scene.spheres.emplace_back(Sphere{Vec3(1, 2, 3), 1, 0});
    scene.materials.push_back(Material{});
    applyCaptureLighting(scene, lighting);
    require(scene.blackBackground && scene.spheres.size() == 1);
    require(scene.spheres[0].geometry.center.x == 1);
    require(scene.lights.size() == 2 && scene.staticTriangles.size() == 2);
    require(std::abs(scene.lightWeights[0] + scene.lightWeights[1] - 384) < 1e-4f);
    for (size_t i = 0; i < 2; ++i) {
        const auto& triangle = scene.staticTriangles[i];
        require(triangle.lightIndex == i);
        require(normalize(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0)).y < -0.99f);
        require(triangle.material == 1 && scene.lights[i].material == 1);
    }
    const auto sky = parseCaptureLighting(Json{{"sky", true}});
    HostScene skyScene;
    applyCaptureLighting(skyScene, sky);
    require(!skyScene.blackBackground && skyScene.lights.empty());
    require(captureLightingFingerprint(123, sky) != captureLightingFingerprint(123, lighting));
    Json small = description;
    small["size"] = 0.5;
    small["emission"] = {1536, 1536, 1536};
    HostScene smallScene;
    applyCaptureLighting(smallScene, parseCaptureLighting(small));
    require(std::abs(smallScene.lightWeights[0] - scene.lightWeights[0]) < 1e-4f);
    require(captureLightingFingerprint(123, parseCaptureLighting(small)) != captureLightingFingerprint(123, lighting));
    for (Json bad : {Json{{"sky", false}}, Json{{"sky", 1}}, Json{{"size", 1}},
                     Json{{"unknown", true}}}) {
        bool rejected = false;
        try { parseCaptureLighting(bad); } catch (const std::exception&) { rejected = true; }
        require(rejected);
    }
    for (Json bad : {Json(-1), Json(0), Json(std::numeric_limits<float>::infinity())}) {
        Json input = description;
        input["size"] = bad;
        bool rejected = false;
        try { parseCaptureLighting(input); } catch (const std::exception&) { rejected = true; }
        require(rejected);
    }
    Json negative = description;
    negative["emission"] = {-1, 1, 1};
    bool rejected = false;
    try { parseCaptureLighting(negative); } catch (const std::exception&) { rejected = true; }
    require(rejected);
    std::cout << "Capture lighting tests passed\n";
}
