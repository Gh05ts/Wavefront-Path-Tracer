#define TINYGLTF_NO_STB_IMAGE
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"

#include "stb_image.h"
#include "scene/gltf_loader.hpp"

#include <cstdlib>
#include <array>
#include <filesystem>
#include <iostream>

namespace
{
bool deferImageDecode(tinygltf::Image*, int, std::string*, std::string*, int, int, const unsigned char*, int, void*) {
    return true;
}

float getExtensionNumber(const tinygltf::Material& material, const char* extensionName, const char* propertyName, float defaultValue) {
    auto extension = material.extensions.find(extensionName);

    if (extension == material.extensions.end() || !extension->second.Has(propertyName))
        return defaultValue;

    return static_cast<float>(extension->second.Get(propertyName).GetNumberAsDouble());
}

Vec3 getExtensionColor(const tinygltf::Material& material, const char* extensionName, const char* propertyName, Vec3 defaultValue) {
    auto extension = material.extensions.find(extensionName);

    if (extension == material.extensions.end() || !extension->second.Has(propertyName))
        return defaultValue;

    const tinygltf::Value& color = extension->second.Get(propertyName);

    if (!color.IsArray() || color.ArrayLen() != 3)
        return defaultValue;

    return Vec3(static_cast<float>(color.Get(0).GetNumberAsDouble()), static_cast<float>(color.Get(1).GetNumberAsDouble()), static_cast<float>(color.Get(2).GetNumberAsDouble()));
}

Vec2 getValueVec2(const tinygltf::Value& value, Vec2 defaultValue) {
    if (!value.IsArray() || value.ArrayLen() != 2)
        return defaultValue;

    return Vec2{static_cast<float>(value.Get(0).GetNumberAsDouble()), static_cast<float>(value.Get(1).GetNumberAsDouble())};
}

void applyTextureTransform(TextureBinding& binding, const tinygltf::Value& transform) {
    if (!transform.IsObject())
        return;

    binding.offset = transform.Has("offset") ? getValueVec2(transform.Get("offset"), binding.offset) : binding.offset;
    binding.scale = transform.Has("scale") ? getValueVec2(transform.Get("scale"), binding.scale) : binding.scale;
    binding.rotation = transform.Has("rotation") ? static_cast<float>(transform.Get("rotation").GetNumberAsDouble()) : binding.rotation;
    binding.texCoord = transform.Has("texCoord") ? static_cast<uint32_t>(transform.Get("texCoord").GetNumberAsInt()) : binding.texCoord;
    binding.hasTransform = true;
}

TextureBinding getTextureBinding(const tinygltf::TextureInfo& textureInfo, const tinygltf::Model& model) {
    TextureBinding binding;

    if (textureInfo.index < 0 || static_cast<size_t>(textureInfo.index) >= model.textures.size() || model.textures[textureInfo.index].source < 0)
        return binding;

    binding.texture = static_cast<uint32_t>(model.textures[textureInfo.index].source);
    binding.texCoord = static_cast<uint32_t>(textureInfo.texCoord);
    auto transform = textureInfo.extensions.find("KHR_texture_transform");

    if (transform != textureInfo.extensions.end())
        applyTextureTransform(binding, transform->second);

    return binding;
}

TextureBinding getTextureBinding(const tinygltf::NormalTextureInfo& textureInfo, const tinygltf::Model& model) {
    TextureBinding binding;

    if (textureInfo.index < 0 || static_cast<size_t>(textureInfo.index) >= model.textures.size() || model.textures[textureInfo.index].source < 0)
        return binding;

    binding.texture = static_cast<uint32_t>(model.textures[textureInfo.index].source);
    binding.texCoord = static_cast<uint32_t>(textureInfo.texCoord);
    auto transform = textureInfo.extensions.find("KHR_texture_transform");

    if (transform != textureInfo.extensions.end())
        applyTextureTransform(binding, transform->second);

    return binding;
}

TextureBinding getExtensionTexture(const tinygltf::Material& material, const char* extensionName, const char* propertyName, const tinygltf::Model& model) {
    TextureBinding binding;
    auto extension = material.extensions.find(extensionName);

    if (extension == material.extensions.end() || !extension->second.Has(propertyName))
        return binding;

    const tinygltf::Value& textureInfo = extension->second.Get(propertyName);

    if (!textureInfo.IsObject() || !textureInfo.Has("index"))
        return binding;

    int textureIndex = textureInfo.Get("index").GetNumberAsInt();

    if (textureIndex < 0 || static_cast<size_t>(textureIndex) >= model.textures.size() || model.textures[textureIndex].source < 0)
        return binding;

    binding.texture = static_cast<uint32_t>(model.textures[textureIndex].source);
    binding.texCoord = textureInfo.Has("texCoord") ? static_cast<uint32_t>(textureInfo.Get("texCoord").GetNumberAsInt()) : 0;

    if (textureInfo.Has("extensions")) {
        const tinygltf::Value& extensions = textureInfo.Get("extensions");

        if (extensions.IsObject() && extensions.Has("KHR_texture_transform"))
            applyTextureTransform(binding, extensions.Get("KHR_texture_transform"));
    }

    return binding;
}

std::array<double, 16> multiply(const std::array<double, 16>& a, const std::array<double, 16>& b) {
    std::array<double, 16> result{};

    for (uint32_t column = 0; column < 4; ++column)
        for (uint32_t row = 0; row < 4; ++row)
            for (uint32_t i = 0; i < 4; ++i)
                result[column * 4 + row] += a[i * 4 + row] * b[column * 4 + i];

    return result;
}

std::array<double, 16> getNodeMatrix(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        std::array<double, 16> matrix{};
        std::copy(node.matrix.begin(), node.matrix.end(), matrix.begin());
        return matrix;
    }

    double x = node.rotation.size() == 4 ? node.rotation[0] : 0.0;
    double y = node.rotation.size() == 4 ? node.rotation[1] : 0.0;
    double z = node.rotation.size() == 4 ? node.rotation[2] : 0.0;
    double w = node.rotation.size() == 4 ? node.rotation[3] : 1.0;
    double sx = node.scale.size() == 3 ? node.scale[0] : 1.0;
    double sy = node.scale.size() == 3 ? node.scale[1] : 1.0;
    double sz = node.scale.size() == 3 ? node.scale[2] : 1.0;
    double tx = node.translation.size() == 3 ? node.translation[0] : 0.0;
    double ty = node.translation.size() == 3 ? node.translation[1] : 0.0;
    double tz = node.translation.size() == 3 ? node.translation[2] : 0.0;
    return {sx * (1.0 - 2.0 * y * y - 2.0 * z * z), sx * (2.0 * x * y + 2.0 * w * z), sx * (2.0 * x * z - 2.0 * w * y), 0.0,
        sy * (2.0 * x * y - 2.0 * w * z), sy * (1.0 - 2.0 * x * x - 2.0 * z * z), sy * (2.0 * y * z + 2.0 * w * x), 0.0,
        sz * (2.0 * x * z + 2.0 * w * y), sz * (2.0 * y * z - 2.0 * w * x), sz * (1.0 - 2.0 * x * x - 2.0 * y * y), 0.0,
        tx, ty, tz, 1.0};
}

InstanceTransform makeTransform(const std::array<double, 16>& matrix) {
    Vec3 x(matrix[0], matrix[1], matrix[2]);
    Vec3 y(matrix[4], matrix[5], matrix[6]);
    Vec3 z(matrix[8], matrix[9], matrix[10]);
    float determinant = dot(x, cross(y, z));
    Vec3 row0 = cross(y, z) / determinant;
    Vec3 row1 = cross(z, x) / determinant;
    Vec3 row2 = cross(x, y) / determinant;
    return InstanceTransform{x, y, z,
        Vec3(row0.x, row1.x, row2.x), Vec3(row0.y, row1.y, row2.y), Vec3(row0.z, row1.z, row2.z),
        Vec3(matrix[12], matrix[13], matrix[14])};
}

const unsigned char* getAccessorData(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t& stride) {
    const tinygltf::BufferView& view = model.bufferViews[accessor.bufferView];
    stride = accessor.ByteStride(view);
    return model.buffers[view.buffer].data.data() + view.byteOffset + accessor.byteOffset;
}

Vec3 readVec3(const tinygltf::Model& model, const tinygltf::Accessor& accessor, uint32_t index) {
    size_t stride;
    const float* value = reinterpret_cast<const float*>(getAccessorData(model, accessor, stride) + stride * index);
    return Vec3(value[0], value[1], value[2]);
}

Vec2 readVec2(const tinygltf::Model& model, const tinygltf::Accessor& accessor, uint32_t index) {
    size_t stride;
    const float* value = reinterpret_cast<const float*>(getAccessorData(model, accessor, stride) + stride * index);
    return Vec2{value[0], 1.0f - value[1]};
}

void readTangent(const tinygltf::Model& model, const tinygltf::Accessor& accessor, uint32_t index, Vec3& tangent, float& sign) {
    size_t stride;
    const float* value = reinterpret_cast<const float*>(getAccessorData(model, accessor, stride) + stride * index);
    tangent = Vec3(value[0], value[1], value[2]);
    sign = value[3];
}

uint32_t readIndex(const tinygltf::Model& model, const tinygltf::Accessor& accessor, uint32_t index) {
    size_t stride;
    const unsigned char* value = getAccessorData(model, accessor, stride) + stride * index;

    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
        return *value;
    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
        return *reinterpret_cast<const uint16_t*>(value);
    return *reinterpret_cast<const uint32_t*>(value);
}

void calculateTangentSpace(Triangle& triangle, bool secondSet) {
    if ((!secondSet && !triangle.hasTexcoords) || (secondSet && !triangle.hasTexcoords1))
        return;

    Vec2 uv0 = secondSet ? triangle.uv1_0 : triangle.uv0;
    Vec2 uv1 = secondSet ? triangle.uv1_1 : triangle.uv1;
    Vec2 uv2 = secondSet ? triangle.uv1_2 : triangle.uv2;
    Vec3 edge1 = triangle.v1 - triangle.v0;
    Vec3 edge2 = triangle.v2 - triangle.v0;
    float determinant = (uv1.x - uv0.x) * (uv2.y - uv0.y) - (uv1.y - uv0.y) * (uv2.x - uv0.x);

    if (fabsf(determinant) < 0.000001f)
        return;

    float inverseDeterminant = 1.0f / determinant;
    Vec3 tangent = (edge1 * (uv2.y - uv0.y) - edge2 * (uv1.y - uv0.y)) * inverseDeterminant;
    Vec3 bitangent = (edge2 * (uv1.x - uv0.x) - edge1 * (uv2.x - uv0.x)) * inverseDeterminant;

    if (secondSet) {
        triangle.tangent1 = normalize(tangent);
        triangle.bitangent1 = normalize(bitangent);
    } else {
        triangle.tangent = normalize(tangent);
        triangle.bitangent = normalize(bitangent);
    }
}

void addNodeInstances(GltfScene& scene, const tinygltf::Model& model, const std::vector<std::vector<uint32_t>>& meshAssets, int nodeIndex, const std::array<double, 16>& parent) {
    const tinygltf::Node& node = model.nodes[nodeIndex];
    std::array<double, 16> world = multiply(parent, getNodeMatrix(node));

    if (node.mesh >= 0)
        for (uint32_t meshIndex : meshAssets[node.mesh])
            scene.instances.push_back(SceneInstance{meshIndex, makeTransform(world)});

    for (int child : node.children)
        addNodeInstances(scene, model, meshAssets, child, world);
}

ObjTexture loadTexture(const tinygltf::Image& image, const tinygltf::Model& model, const std::filesystem::path& directory) {
    int width;
    int height;
    int channels;
    unsigned char* pixels = nullptr;

    if (!image.uri.empty())
        pixels = stbi_load((directory / image.uri).string().c_str(), &width, &height, &channels, STBI_rgb_alpha);
    else if (image.bufferView >= 0) {
        const tinygltf::BufferView& view = model.bufferViews[image.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[view.buffer];
        pixels = stbi_load_from_memory(buffer.data.data() + view.byteOffset, static_cast<int>(view.byteLength), &width, &height, &channels, STBI_rgb_alpha);
    }

    if (pixels == nullptr) {
        std::cerr << "Failed to load glTF image: " << image.name << '\n';
        std::exit(1);
    }

    ObjTexture texture{};
    texture.width = width;
    texture.height = height;
    texture.pixels.assign(pixels, pixels + 4 * width * height);
    stbi_image_free(pixels);
    return texture;
}
} // namespace

GltfScene loadGltfScene(const char* filename) {
    tinygltf::TinyGLTF loader;
    loader.SetImageLoader(deferImageDecode, nullptr);
    tinygltf::Model model;
    std::string warning;
    std::string error;
    std::filesystem::path path(filename);
    bool loaded = path.extension() == ".glb" ? loader.LoadBinaryFromFile(&model, &error, &warning, filename) : loader.LoadASCIIFromFile(&model, &error, &warning, filename);

    if (!warning.empty())
        std::cerr << "TinyGLTF: " << warning;
    if (!loaded) {
        std::cerr << "TinyGLTF: " << error << '\n';
        std::exit(1);
    }

    GltfScene scene;
    scene.materials.resize(model.materials.size() + 1);
    scene.materials[0] = Material{MaterialType::Diffuse, Vec3(0.7f, 0.7f, 0.7f), Vec3(), 0.0f, 1.0f};
    for (uint32_t i = 0; i < model.images.size(); ++i)
        scene.textures.push_back(loadTexture(model.images[i], model, path.parent_path()));
    for (uint32_t i = 0; i < model.materials.size(); ++i) {
        const tinygltf::Material& source = model.materials[i];
        Material material{};
        const auto& factor = source.pbrMetallicRoughness.baseColorFactor;
        material.albedo = Vec3(factor[0], factor[1], factor[2]);
        material.emission = Vec3(source.emissiveFactor[0], source.emissiveFactor[1], source.emissiveFactor[2]);
        float transmission = getExtensionNumber(source, "KHR_materials_transmission", "transmissionFactor", 0.0f);
        material.type = lengthSquared(material.emission) > 0.0f || source.emissiveTexture.index >= 0 ? MaterialType::Emissive :
            (transmission > 0.0f ? MaterialType::Dielectric : MaterialType::Diffuse);
        material.roughness = static_cast<float>(source.pbrMetallicRoughness.roughnessFactor);
        material.metallic = static_cast<float>(source.pbrMetallicRoughness.metallicFactor);
        material.normalScale = static_cast<float>(source.normalTexture.scale);
        material.alphaCutoff = static_cast<float>(source.alphaCutoff);
        material.alphaMasked = source.alphaMode == "MASK";
        material.ior = getExtensionNumber(source, "KHR_materials_ior", "ior", 1.5f);
        material.attenuationColor = getExtensionColor(source, "KHR_materials_volume", "attenuationColor", Vec3(1.0f, 1.0f, 1.0f));
        material.attenuationDistance = getExtensionNumber(source, "KHR_materials_volume", "attenuationDistance", 0.0f);
        material.volumeDensity = getExtensionNumber(source, "KHR_materials_volume", "thicknessFactor", 1.0f);
        material.thicknessTexture = getExtensionTexture(source, "KHR_materials_volume", "thicknessTexture", model);
        material.albedoTexture = getTextureBinding(source.pbrMetallicRoughness.baseColorTexture, model);
        material.metallicRoughnessTexture = getTextureBinding(source.pbrMetallicRoughness.metallicRoughnessTexture, model);
        material.emissiveTexture = getTextureBinding(source.emissiveTexture, model);
        material.normalTexture = getTextureBinding(source.normalTexture, model);
        scene.materials[i + 1] = material;
    }
    std::vector<std::vector<uint32_t>> meshAssets(model.meshes.size());

    for (uint32_t meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex) {
        for (const tinygltf::Primitive& primitive : model.meshes[meshIndex].primitives) {
            auto position = primitive.attributes.find("POSITION");
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES || position == primitive.attributes.end() || primitive.indices < 0)
                continue;

            const tinygltf::Accessor& positions = model.accessors[position->second];
            const tinygltf::Accessor& indices = model.accessors[primitive.indices];
            auto normal = primitive.attributes.find("NORMAL");
            auto tangent = primitive.attributes.find("TANGENT");
            auto texcoord = primitive.attributes.find("TEXCOORD_0");
            auto texcoord1 = primitive.attributes.find("TEXCOORD_1");
            const tinygltf::Accessor* normals = normal == primitive.attributes.end() ? nullptr : &model.accessors[normal->second];
            const tinygltf::Accessor* tangents = tangent == primitive.attributes.end() ? nullptr : &model.accessors[tangent->second];
            const tinygltf::Accessor* texcoords = texcoord == primitive.attributes.end() ? nullptr : &model.accessors[texcoord->second];
            const tinygltf::Accessor* texcoords1 = texcoord1 == primitive.attributes.end() ? nullptr : &model.accessors[texcoord1->second];
            MeshAsset asset;

            for (uint32_t index = 0; index < indices.count; index += 3) {
                uint32_t i0 = readIndex(model, indices, index);
                uint32_t i1 = readIndex(model, indices, index + 1);
                uint32_t i2 = readIndex(model, indices, index + 2);
                Triangle triangle{};
                triangle.v0 = readVec3(model, positions, i0);
                triangle.v1 = readVec3(model, positions, i1);
                triangle.v2 = readVec3(model, positions, i2);
                triangle.material = primitive.material >= 0 ? primitive.material + 1 : 0;
                triangle.lightIndex = invalidLightIndex;
                triangle.hasVertexNormals = normals != nullptr;
                triangle.hasVertexTangents = tangents != nullptr;
                triangle.hasTexcoords = texcoords != nullptr;
                triangle.hasTexcoords1 = texcoords1 != nullptr;
                if (normals != nullptr) {
                    triangle.n0 = readVec3(model, *normals, i0);
                    triangle.n1 = readVec3(model, *normals, i1);
                    triangle.n2 = readVec3(model, *normals, i2);
                }
                if (tangents != nullptr) {
                    readTangent(model, *tangents, i0, triangle.vertexTangent0, triangle.vertexTangentSign0);
                    readTangent(model, *tangents, i1, triangle.vertexTangent1, triangle.vertexTangentSign1);
                    readTangent(model, *tangents, i2, triangle.vertexTangent2, triangle.vertexTangentSign2);
                }
                if (texcoords != nullptr) {
                    triangle.uv0 = readVec2(model, *texcoords, i0);
                    triangle.uv1 = readVec2(model, *texcoords, i1);
                    triangle.uv2 = readVec2(model, *texcoords, i2);
                }
                if (texcoords1 != nullptr) {
                    triangle.uv1_0 = readVec2(model, *texcoords1, i0);
                    triangle.uv1_1 = readVec2(model, *texcoords1, i1);
                    triangle.uv1_2 = readVec2(model, *texcoords1, i2);
                }
                calculateTangentSpace(triangle, false);
                calculateTangentSpace(triangle, true);
                asset.triangles.push_back(triangle);
            }

            if (!asset.triangles.empty()) {
                meshAssets[meshIndex].push_back(scene.meshes.size());
                scene.meshes.push_back(std::move(asset));
            }
        }
    }

    std::array<double, 16> identity{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (sceneIndex >= 0 && sceneIndex < model.scenes.size())
        for (int nodeIndex : model.scenes[sceneIndex].nodes)
            addNodeInstances(scene, model, meshAssets, nodeIndex, identity);

    return scene;
}
