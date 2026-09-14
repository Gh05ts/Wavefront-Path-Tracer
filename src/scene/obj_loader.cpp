#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include "scene/obj_loader.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

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
    return material;
}

Material makeDefaultMaterial() {
    Material material{};
    material.type = MaterialType::Diffuse;
    material.albedo = Vec3(0.65f, 0.3f, 0.18f);
    material.emission = Vec3(0.0f, 0.0f, 0.0f);
    material.roughness = 0.0f;
    material.ior = 1.0f;
    return material;
}
} // namespace

ObjMesh loadObjMesh(const char* filename) {
    tinyobj::attrib_t attributes;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> sourceMaterials;
    std::string warning;
    std::string error;

    if (!tinyobj::LoadObj(&attributes, &shapes, &sourceMaterials, &warning, &error, filename, nullptr, true)) {
        std::cerr << warning << error << '\n';
        std::exit(1);
    }

    if (!warning.empty())
        std::cerr << "tinyobjloader: " << warning;

    ObjMesh mesh;
    mesh.materials.reserve(sourceMaterials.size() + 1);
    mesh.materials.push_back(makeDefaultMaterial());

    for (const tinyobj::material_t& sourceMaterial : sourceMaterials)
        mesh.materials.push_back(makeMaterial(sourceMaterial));

    for (const tinyobj::shape_t& shape : shapes) {
        size_t indexOffset = 0;

        for (size_t face = 0; face < shape.mesh.num_face_vertices.size(); ++face) {
            uint8_t vertexCount = shape.mesh.num_face_vertices[face];

            if (vertexCount != 3) {
                std::cerr << "OBJ loader expected triangulated faces\n";
                std::exit(1);
            }

            Triangle triangle{};
            const tinyobj::index_t* indices = &shape.mesh.indices[indexOffset];

            const float* v0 = &attributes.vertices[3 * indices[0].vertex_index];
            const float* v1 = &attributes.vertices[3 * indices[1].vertex_index];
            const float* v2 = &attributes.vertices[3 * indices[2].vertex_index];

            triangle.v0 = Vec3(v0[0], v0[1], v0[2]);
            triangle.v1 = Vec3(v1[0], v1[1], v1[2]);
            triangle.v2 = Vec3(v2[0], v2[1], v2[2]);

            int material = shape.mesh.material_ids[face];
            triangle.material = material >= 0 ? static_cast<uint32_t>(material + 1) : 0;
            mesh.triangles.push_back(triangle);

            indexOffset += vertexCount;
        }
    }

    if (mesh.triangles.empty()) {
        std::cerr << "OBJ contains no triangles: " << filename << '\n';
        std::exit(1);
    }

    return mesh;
}
