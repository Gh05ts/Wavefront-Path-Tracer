#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "scene/obj_loader.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

namespace
{
Material makeMaterial(const tinyobj::material_t& source) {
    Material material{};
    material.type = source.emission[0] != 0.0f || source.emission[1] != 0.0f || source.emission[2] != 0.0f
        ? MaterialType::Emissive
        : MaterialType::Diffuse;
    material.albedo = Vec3(source.diffuse[0], source.diffuse[1], source.diffuse[2]);
    material.emission = Vec3(source.emission[0], source.emission[1], source.emission[2]);
    material.roughness = 0.0f;
    material.ior = source.ior > 0.0f ? source.ior : 1.0f;
    material.albedoTexture.texture = invalidTextureIndex;
    return material;
}

Material makeDefaultMaterial() {
    Material material{};
    material.type = MaterialType::Diffuse;
    material.albedo = Vec3(0.65f, 0.3f, 0.18f);
    material.emission = Vec3(0.0f, 0.0f, 0.0f);
    material.roughness = 0.0f;
    material.ior = 1.0f;
    material.albedoTexture.texture = invalidTextureIndex;
    return material;
}

std::string getDirectory(const char* filename) {
    std::string path(filename);
    size_t separator = path.find_last_of("/\\");
    return separator == std::string::npos ? std::string() : path.substr(0, separator + 1);
}
} // namespace

ObjScene loadObjScene(const char* filename) {
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> sourceMaterials;
    std::string warning;
    std::string error;
    std::string directory = getDirectory(filename);

    if (!tinyobj::LoadObj(&attributes, &shapes, &sourceMaterials, &warning, &error, filename, directory.empty() ? nullptr : directory.c_str(), true)) {
        std::cerr << warning << error << '\n';
        std::exit(1);
    }

    if (!warning.empty())
        std::cerr << "tinyobjloader: " << warning;

    ObjScene scene;
    scene.materials.reserve(sourceMaterials.size() + 1);
    scene.materials.push_back(makeDefaultMaterial());

    for (const tinyobj::material_t& sourceMaterial : sourceMaterials)
        scene.materials.push_back(makeMaterial(sourceMaterial));

    std::unordered_map<std::string, uint32_t> textureIndices;

    for (uint32_t materialIndex = 0; materialIndex < sourceMaterials.size(); ++materialIndex) {
        const std::string& textureName = sourceMaterials[materialIndex].diffuse_texname;

        if (textureName.empty())
            continue;

        auto existing = textureIndices.find(textureName);
        if (existing != textureIndices.end()) {
            scene.materials[materialIndex + 1].albedoTexture.texture = existing->second;
            continue;
        }

        int width;
        int height;
        int channels;
        unsigned char* pixels = stbi_load((directory + textureName).c_str(), &width, &height, &channels, STBI_rgb_alpha);

        if (pixels == nullptr) {
            std::cerr << "Failed to load texture " << directory + textureName << ": " << stbi_failure_reason() << '\n';
            std::exit(1);
        }

        ObjTexture texture{};
        texture.width = static_cast<uint32_t>(width);
        texture.height = static_cast<uint32_t>(height);
        texture.pixels.assign(pixels, pixels + 4 * width * height);
        stbi_image_free(pixels);

        uint32_t textureIndex = static_cast<uint32_t>(scene.textures.size());
        scene.textures.push_back(std::move(texture));
        textureIndices.emplace(textureName, textureIndex);
        scene.materials[materialIndex + 1].albedoTexture.texture = textureIndex;
    }

    std::vector<Vec3> generatedNormals(attributes.vertices.size() / 3);

    for (const tinyobj::shape_t& shape : shapes) {
        size_t indexOffset = 0;

        for (uint8_t vertexCount : shape.mesh.num_face_vertices) {
            if (vertexCount == 3) {
                const tinyobj::index_t* indices = &shape.mesh.indices[indexOffset];
                const float* v0 = &attributes.vertices[3 * indices[0].vertex_index];
                const float* v1 = &attributes.vertices[3 * indices[1].vertex_index];
                const float* v2 = &attributes.vertices[3 * indices[2].vertex_index];
                Vec3 faceNormal = cross(Vec3(v1[0], v1[1], v1[2]) - Vec3(v0[0], v0[1], v0[2]), Vec3(v2[0], v2[1], v2[2]) - Vec3(v0[0], v0[1], v0[2]));

                generatedNormals[indices[0].vertex_index] += faceNormal;
                generatedNormals[indices[1].vertex_index] += faceNormal;
                generatedNormals[indices[2].vertex_index] += faceNormal;
            }

            indexOffset += vertexCount;
        }
    }

    for (Vec3& normal : generatedNormals) {
        if (lengthSquared(normal) > 0.0f)
            normal = normalize(normal);
    }

    for (const tinyobj::shape_t& shape : shapes) {
        ObjMesh mesh;
        size_t indexOffset = 0;

        for (size_t face = 0; face < shape.mesh.num_face_vertices.size(); ++face) {
            uint8_t vertexCount = shape.mesh.num_face_vertices[face];

            if (vertexCount != 3) {
                std::cerr << "OBJ loader expected triangulated faces\n";
                std::exit(1);
            }

            Triangle triangle{};
            triangle.lightIndex = invalidLightIndex;
            const tinyobj::index_t* indices = &shape.mesh.indices[indexOffset];

            const float* v0 = &attributes.vertices[3 * indices[0].vertex_index];
            const float* v1 = &attributes.vertices[3 * indices[1].vertex_index];
            const float* v2 = &attributes.vertices[3 * indices[2].vertex_index];

            triangle.v0 = Vec3(v0[0], v0[1], v0[2]);
            triangle.v1 = Vec3(v1[0], v1[1], v1[2]);
            triangle.v2 = Vec3(v2[0], v2[1], v2[2]);

            bool hasObjNormals = indices[0].normal_index >= 0 && indices[1].normal_index >= 0 && indices[2].normal_index >= 0;
            triangle.hasVertexNormals = true;
            if (hasObjNormals) {
                const float* n0 = &attributes.normals[3 * indices[0].normal_index];
                const float* n1 = &attributes.normals[3 * indices[1].normal_index];
                const float* n2 = &attributes.normals[3 * indices[2].normal_index];
                triangle.n0 = Vec3(n0[0], n0[1], n0[2]);
                triangle.n1 = Vec3(n1[0], n1[1], n1[2]);
                triangle.n2 = Vec3(n2[0], n2[1], n2[2]);
            } else {
                triangle.n0 = generatedNormals[indices[0].vertex_index];
                triangle.n1 = generatedNormals[indices[1].vertex_index];
                triangle.n2 = generatedNormals[indices[2].vertex_index];
            }

            triangle.hasTexcoords = indices[0].texcoord_index >= 0 && indices[1].texcoord_index >= 0 && indices[2].texcoord_index >= 0;
            if (triangle.hasTexcoords) {
                const float* uv0 = &attributes.texcoords[2 * indices[0].texcoord_index];
                const float* uv1 = &attributes.texcoords[2 * indices[1].texcoord_index];
                const float* uv2 = &attributes.texcoords[2 * indices[2].texcoord_index];
                triangle.uv0 = Vec2{uv0[0], uv0[1]};
                triangle.uv1 = Vec2{uv1[0], uv1[1]};
                triangle.uv2 = Vec2{uv2[0], uv2[1]};
            }

            int material = shape.mesh.material_ids[face];
            triangle.material = material >= 0 ? static_cast<uint32_t>(material + 1) : 0;
            mesh.triangles.push_back(triangle);

            indexOffset += vertexCount;
        }

        if (!mesh.triangles.empty())
            scene.meshes.push_back(std::move(mesh));
    }

    if (scene.meshes.empty()) {
        std::cerr << "OBJ contains no triangles: " << filename << '\n';
        std::exit(1);
    }

    return scene;
}

ObjMesh loadObjMesh(const char* filename) {
    ObjScene scene = loadObjScene(filename);
    ObjMesh mesh;

    for (ObjMesh& sourceMesh : scene.meshes)
        mesh.triangles.insert(mesh.triangles.end(), sourceMesh.triangles.begin(), sourceMesh.triangles.end());

    mesh.materials = std::move(scene.materials);
    mesh.textures = std::move(scene.textures);

    return mesh;
}
