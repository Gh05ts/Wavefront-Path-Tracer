#include "scene/scene_host.hpp"
#include "scene/gltf_loader.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>

namespace
{
Triangle makeTriangle(const Vec3& v0, const Vec3& v1, const Vec3& v2, uint32_t material) {
    Triangle triangle{};
    triangle.v0 = v0;
    triangle.v1 = v1;
    triangle.v2 = v2;
    triangle.material = material;
    triangle.lightIndex = invalidLightIndex;
    return triangle;
}

void addQuad(std::vector<Triangle>& triangles, const Vec3& v0, const Vec3& v1, const Vec3& v2, const Vec3& v3, uint32_t material) {
    triangles.push_back(makeTriangle(v0, v1, v2, material));
    triangles.push_back(makeTriangle(v0, v2, v3, material));
}

MeshAsset makeCausticPrism() {
    constexpr float halfLength = 1.25f;
    const Vec3 a0(-halfLength, -0.82f, -0.68f);
    const Vec3 b0(-halfLength, -0.82f,  0.68f);
    const Vec3 c0(-halfLength,  0.38f,  0.00f);
    const Vec3 a1( halfLength, -0.82f, -0.68f);
    const Vec3 b1( halfLength, -0.82f,  0.68f);
    const Vec3 c1( halfLength,  0.38f,  0.00f);

    MeshAsset prism;
    prism.triangles = {
        makeTriangle(a0, a1, b1, 0), makeTriangle(a0, b1, b0, 0),
        makeTriangle(b0, b1, c1, 0), makeTriangle(b0, c1, c0, 0),
        makeTriangle(c0, c1, a1, 0), makeTriangle(c0, a1, a0, 0),
        makeTriangle(a0, c0, b0, 0), makeTriangle(a1, b1, c1, 0)
    };
    return prism;
}

Material makeMaterial(MaterialType type, const Vec3& albedo, const Vec3& emission = Vec3(0.0f, 0.0f, 0.0f)) {
    Material material{};
    material.type = type;
    material.albedo = albedo;
    material.emission = emission;
    material.roughness = 0.0f;
    material.ior = 1.0f;
    return material;
}

void addStaticTriangleLights(HostScene& hostScene) {
    for (Triangle& triangle : hostScene.staticTriangles) {
        if (hostScene.materials[triangle.material].type != MaterialType::Emissive) {
            triangle.lightIndex = invalidLightIndex;
            continue;
        }

        Vec3 normal = normalize(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
        float area = 0.5f * length(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
        const Material& material = hostScene.materials[triangle.material];
        float power = area * (0.2126f * material.emission.x + 0.7152f * material.emission.y + 0.0722f * material.emission.z);
        triangle.lightIndex = static_cast<uint32_t>(hostScene.lights.size());
        hostScene.lights.push_back(TriangleLight{triangle.v0, triangle.v1, triangle.v2, normal, area, triangle.material, 0.0f});
        hostScene.lightWeights.push_back(power);
    }
}

void addMeshTriangleLights(HostScene& hostScene) {
    for (MeshAsset& mesh : hostScene.meshes) {
        uint32_t meshLightCount = 0;
        for (Triangle& triangle : mesh.triangles) {
            if (hostScene.materials[triangle.material].type == MaterialType::Emissive)
                triangle.lightIndex = meshLightCount++;
            else
                triangle.lightIndex = invalidLightIndex;
        }
    }

    for (const SceneInstance& instance : hostScene.instances) {
        const MeshAsset& mesh = hostScene.meshes[instance.meshIndex];
        for (const Triangle& triangle : mesh.triangles) {
            if (triangle.lightIndex == invalidLightIndex)
                continue;

            Vec3 v0 = transformVector(instance.transform, triangle.v0) + instance.transform.translation;
            Vec3 v1 = transformVector(instance.transform, triangle.v1) + instance.transform.translation;
            Vec3 v2 = transformVector(instance.transform, triangle.v2) + instance.transform.translation;
            Vec3 normal = normalize(cross(v1 - v0, v2 - v0));
            float area = 0.5f * length(cross(v1 - v0, v2 - v0));
            const Material& material = hostScene.materials[triangle.material];
            float power = area * (0.2126f * material.emission.x + 0.7152f * material.emission.y + 0.0722f * material.emission.z);

            hostScene.lights.push_back(TriangleLight{v0, v1, v2, normal, area, triangle.material, 0.0f});
            hostScene.lightWeights.push_back(power);
        }
    }
}

void requireAssetFilename(const char* filename, const char* sceneName) {
    if (filename == nullptr) {
        std::cerr << sceneName << " scene requires an asset filename\n";
        std::exit(1);
    }
}
} // namespace

HostScene assembleDemoScene() {
    HostScene hostScene;
    hostScene.spheres = {
        Sphere{Vec3(0.0f, -100.5f, 0.0f), 100.0f, 3},
        Sphere{Vec3(1.35f, 0.08f, 0.0f), 0.5f, 2}
    };

    const Vec3 cube0(-1.85f, -0.42f, -0.5f);
    const Vec3 cube1(-0.85f, -0.42f, -0.5f);
    const Vec3 cube2(-0.85f,  0.58f, -0.5f);
    const Vec3 cube3(-1.85f,  0.58f, -0.5f);
    const Vec3 cube4(-1.85f, -0.42f,  0.5f);
    const Vec3 cube5(-0.85f, -0.42f,  0.5f);
    const Vec3 cube6(-0.85f,  0.58f,  0.5f);
    const Vec3 cube7(-1.85f,  0.58f,  0.5f);
    hostScene.triangles = {
        makeTriangle(cube4, cube5, cube6, 0), makeTriangle(cube4, cube6, cube7, 0),
        makeTriangle(cube0, cube2, cube1, 0), makeTriangle(cube0, cube3, cube2, 0),
        makeTriangle(cube0, cube4, cube7, 0), makeTriangle(cube0, cube7, cube3, 0),
        makeTriangle(cube1, cube2, cube6, 0), makeTriangle(cube1, cube6, cube5, 0),
        makeTriangle(cube3, cube7, cube6, 0), makeTriangle(cube3, cube6, cube2, 0),
        makeTriangle(cube0, cube1, cube5, 0), makeTriangle(cube0, cube5, cube4, 0)
    };

    const Vec3 pyramid0(-0.183f, -0.42f, -0.683f);
    const Vec3 pyramid1( 0.683f, -0.42f, -0.183f);
    const Vec3 pyramid2( 0.183f, -0.42f,  0.683f);
    const Vec3 pyramid3(-0.683f, -0.42f,  0.183f);
    const Vec3 pyramidTop(0.0f, 0.73f, 0.0f);
    hostScene.triangles.insert(hostScene.triangles.end(), {
        makeTriangle(pyramid0, pyramid1, pyramidTop, 1),
        makeTriangle(pyramid1, pyramid2, pyramidTop, 1),
        makeTriangle(pyramid2, pyramid3, pyramidTop, 1),
        makeTriangle(pyramid3, pyramid0, pyramidTop, 1),
        makeTriangle(pyramid0, pyramid2, pyramid1, 1),
        makeTriangle(pyramid0, pyramid3, pyramid2, 1)
    });

    hostScene.materials = {
        makeMaterial(MaterialType::Diffuse, Vec3(0.7f, 0.2f, 0.15f)),
        makeMaterial(MaterialType::Metal, Vec3(0.85f, 0.65f, 0.25f)),
        makeMaterial(MaterialType::Dielectric, Vec3(1.0f, 1.0f, 1.0f)),
        makeMaterial(MaterialType::Diffuse, Vec3(0.75f, 0.75f, 0.75f))
    };
    hostScene.materials[1].roughness = 0.08f;
    hostScene.materials[2].ior = 1.5f;
    return hostScene;
}

HostScene assembleCornellScene(const CornellSceneOptions& options) {
    if (options.objectSource != CornellObjectSource::ProceduralPrism)
        requireAssetFilename(options.filename, "Cornell");

    HostScene hostScene;
    std::vector<SceneInstance> gltfInstances;
    std::vector<Material> objectMaterials;

    if (options.objectSource == CornellObjectSource::ProceduralPrism) {
        hostScene.meshes.push_back(makeCausticPrism());
        Material prismMaterial = makeMaterial(MaterialType::Dielectric, Vec3(0.98f, 0.99f, 1.0f));
        prismMaterial.ior = 1.52f;
        prismMaterial.dispersion = 0.035f;
        hostScene.materials.push_back(prismMaterial);
    } else if (options.objectSource == CornellObjectSource::Gltf) {
        GltfScene source = loadGltfScene(options.filename);
        hostScene.meshes = std::move(source.meshes);
        gltfInstances = std::move(source.instances);
        objectMaterials = std::move(source.materials);
        hostScene.textures = std::move(source.textures);
    } else {
        ObjScene source = loadObjScene(options.filename);
        for (ObjMesh& mesh : source.meshes)
            hostScene.meshes.push_back(MeshAsset{std::move(mesh.triangles)});
        objectMaterials = std::move(source.materials);
        hostScene.textures = std::move(source.textures);
    }

    if (options.convertObjectMaterialsToDielectric) {
        for (Material& material : objectMaterials) {
            material.type = MaterialType::Dielectric;
            material.albedo = Vec3(1.0f, 1.0f, 1.0f);
            material.emission = Vec3(0.0f, 0.0f, 0.0f);
            material.roughness = 0.0f;
            material.ior = 1.52f;
            material.dispersion = 0.02f;
            material.albedoTexture.texture = invalidTextureIndex;
        }
    }

    constexpr uint32_t objectMaterialOffset = 5;
    for (MeshAsset& mesh : hostScene.meshes) {
        for (Triangle& triangle : mesh.triangles)
            triangle.material = options.objectMaterialOverride >= 0 ?
                static_cast<uint32_t>(options.objectMaterialOverride) : triangle.material + objectMaterialOffset;
    }

    if (options.objectSource == CornellObjectSource::Gltf) {
        hostScene.instances = std::move(gltfInstances);
        if (options.removeBackdrop) {
            hostScene.instances.erase(std::remove_if(hostScene.instances.begin(), hostScene.instances.end(), [](const SceneInstance& instance) {
                return instance.meshIndex == 0;
            }), hostScene.instances.end());
        }
        for (SceneInstance& instance : hostScene.instances) {
            instance.transform.localToWorldX = options.objectScale * instance.transform.localToWorldX;
            instance.transform.localToWorldY = options.objectScale * instance.transform.localToWorldY;
            instance.transform.localToWorldZ = options.objectScale * instance.transform.localToWorldZ;
            instance.transform.worldToLocalX = instance.transform.worldToLocalX / options.objectScale;
            instance.transform.worldToLocalY = instance.transform.worldToLocalY / options.objectScale;
            instance.transform.worldToLocalZ = instance.transform.worldToLocalZ / options.objectScale;
            instance.transform.translation = options.objectScale * instance.transform.translation + options.objectTranslation;
        }
    } else {
        for (uint32_t meshIndex = 0; meshIndex < hostScene.meshes.size(); ++meshIndex) {
            InstanceTransform transform = makeScaledTransform(options.objectTranslation, options.objectScale);
            if (options.lightProfile == CornellLightProfile::Prism)
                rotateTransform(transform, Vec3(0.0f, 1.0f, 0.0f), -20.0f);
            hostScene.instances.push_back(SceneInstance{meshIndex, transform});
        }
    }

    constexpr float roomHalfWidth = 2.5f;
    constexpr float floorY = -1.0f;
    constexpr float ceilingY = 3.0f;
    constexpr float backZ = -2.5f;
    constexpr float frontZ = 2.5f;
    addQuad(hostScene.staticTriangles, Vec3(-roomHalfWidth, floorY, frontZ), Vec3(roomHalfWidth, floorY, frontZ), Vec3(roomHalfWidth, floorY, backZ), Vec3(-roomHalfWidth, floorY, backZ), 0);
    addQuad(hostScene.staticTriangles, Vec3(-roomHalfWidth, ceilingY, backZ), Vec3(roomHalfWidth, ceilingY, backZ), Vec3(roomHalfWidth, ceilingY, frontZ), Vec3(-roomHalfWidth, ceilingY, frontZ), 0);
    addQuad(hostScene.staticTriangles, Vec3(-roomHalfWidth, floorY, backZ), Vec3(roomHalfWidth, floorY, backZ), Vec3(roomHalfWidth, ceilingY, backZ), Vec3(-roomHalfWidth, ceilingY, backZ), 0);
    addQuad(hostScene.staticTriangles, Vec3(-roomHalfWidth, floorY, frontZ), Vec3(-roomHalfWidth, floorY, backZ), Vec3(-roomHalfWidth, ceilingY, backZ), Vec3(-roomHalfWidth, ceilingY, frontZ), 1);
    addQuad(hostScene.staticTriangles, Vec3(roomHalfWidth, floorY, backZ), Vec3(roomHalfWidth, floorY, frontZ), Vec3(roomHalfWidth, ceilingY, frontZ), Vec3(roomHalfWidth, ceilingY, backZ), 2);

    bool isCrystal = options.convertObjectMaterialsToDielectric;
    bool isCausticPrism = options.lightProfile == CornellLightProfile::Prism;
    const Vec3 lightCorner = isCrystal ? Vec3(-0.5f, 2.35f, -1.6f) : (isCausticPrism ? Vec3(-0.9f, 2.35f, -1.6f) : Vec3(-0.75f, 2.95f, -0.75f));
    const Vec3 lightEdgeU = isCrystal ? Vec3(1.0f, 0.0f, 0.0f) : (isCausticPrism ? Vec3(1.8f, 0.0f, 0.0f) : Vec3(1.5f, 0.0f, 0.0f));
    const Vec3 lightEdgeV = isCrystal ? Vec3(0.0f, 0.25f, 0.42f) : (isCausticPrism ? Vec3(0.0f, 0.45f, 0.75f) : Vec3(0.0f, 0.0f, 1.5f));
    addQuad(hostScene.staticTriangles, lightCorner, lightCorner + lightEdgeU, lightCorner + lightEdgeU + lightEdgeV, lightCorner + lightEdgeV, 4);

    hostScene.materials.reserve(objectMaterialOffset + objectMaterials.size());
    hostScene.materials.insert(hostScene.materials.begin(), {
        makeMaterial(MaterialType::Diffuse, Vec3(0.73f, 0.73f, 0.73f)),
        makeMaterial(MaterialType::Diffuse, options.neutralRoom ? Vec3(0.68f, 0.68f, 0.68f) : Vec3(0.65f, 0.05f, 0.05f)),
        makeMaterial(MaterialType::Diffuse, options.neutralRoom ? Vec3(0.58f, 0.58f, 0.58f) : Vec3(0.12f, 0.45f, 0.15f)),
        makeMaterial(MaterialType::Diffuse, Vec3(0.72f, 0.72f, 0.72f)),
        makeMaterial(MaterialType::Emissive, Vec3(1.0f, 1.0f, 1.0f), isCrystal ? Vec3(32.0f, 32.0f, 32.0f) : (isCausticPrism ? Vec3(10.0f, 10.0f, 10.0f) : Vec3(16.0f, 16.0f, 16.0f)))
    });
    hostScene.materials.insert(hostScene.materials.end(), objectMaterials.begin(), objectMaterials.end());

    addMeshTriangleLights(hostScene);
    addStaticTriangleLights(hostScene);
    hostScene.blackBackground = true;
    return hostScene;
}

HostScene assembleGltfScene(const char* filename, float sceneScale, bool addTopLight) {
    requireAssetFilename(filename, "glTF");
    GltfScene source = loadGltfScene(filename);
    if (source.meshes.empty() || source.instances.empty()) {
        std::cerr << "glTF scene contains no renderable mesh instances\n";
        std::exit(1);
    }

    HostScene hostScene;
    hostScene.meshes = std::move(source.meshes);
    hostScene.instances = std::move(source.instances);
    hostScene.materials = std::move(source.materials);
    hostScene.textures = std::move(source.textures);

    for (SceneInstance& instance : hostScene.instances) {
        instance.transform.localToWorldX = sceneScale * instance.transform.localToWorldX;
        instance.transform.localToWorldY = sceneScale * instance.transform.localToWorldY;
        instance.transform.localToWorldZ = sceneScale * instance.transform.localToWorldZ;
        instance.transform.worldToLocalX = instance.transform.worldToLocalX / sceneScale;
        instance.transform.worldToLocalY = instance.transform.worldToLocalY / sceneScale;
        instance.transform.worldToLocalZ = instance.transform.worldToLocalZ / sceneScale;
        instance.transform.translation = sceneScale * instance.transform.translation;
    }

    if (addTopLight) {
        uint32_t lightMaterial = static_cast<uint32_t>(hostScene.materials.size());
        hostScene.materials.push_back(makeMaterial(MaterialType::Emissive, Vec3(1.0f, 1.0f, 1.0f), Vec3(80.0f, 80.0f, 80.0f)));
        const Vec3 lightCorner(-4.0f, 9.2f, -4.0f);
        const Vec3 lightEdgeU(8.0f, 0.0f, 0.0f);
        const Vec3 lightEdgeV(0.0f, 0.0f, 8.0f);
        addQuad(hostScene.staticTriangles, lightCorner, lightCorner + lightEdgeU, lightCorner + lightEdgeU + lightEdgeV, lightCorner + lightEdgeV, lightMaterial);
    }

    addMeshTriangleLights(hostScene);
    addStaticTriangleLights(hostScene);
    return hostScene;
}

HostScene assembleObjScene(const ObjSceneOptions& options) {
    requireAssetFilename(options.filename, "OBJ");
    ObjMesh mesh = loadObjMesh(options.filename);

    HostScene hostScene;
    hostScene.acceleration = options.acceleration;
    hostScene.spheres.push_back(Sphere{Vec3(0.0f, -100.5f, 0.0f), 100.0f, 0});
    hostScene.materials.push_back(makeMaterial(MaterialType::Diffuse, Vec3(0.75f, 0.75f, 0.75f)));
    hostScene.materials.insert(hostScene.materials.end(), mesh.materials.begin(), mesh.materials.end());
    hostScene.textures = std::move(mesh.textures);

    for (Triangle& triangle : mesh.triangles)
        triangle.material += 1;

    if (options.acceleration == ObjAccelerationPolicy::TlasBlasBvh2) {
        hostScene.meshes.push_back(MeshAsset{std::move(mesh.triangles)});
        hostScene.instances.push_back(SceneInstance{0, makeScaledTransform(options.objectTranslation, options.objectScale)});
    } else {
        for (Triangle& triangle : mesh.triangles) {
            triangle.v0 = options.objectScale * triangle.v0 + options.objectTranslation;
            triangle.v1 = options.objectScale * triangle.v1 + options.objectTranslation;
            triangle.v2 = options.objectScale * triangle.v2 + options.objectTranslation;
            triangle.lightIndex = invalidLightIndex;
        }
        hostScene.triangles = std::move(mesh.triangles);
    }

    return hostScene;
}
