#include "scene/scene_manifest.hpp"

#include <filesystem>
#include <fstream>
#include <utility>

#include "json.hpp"

namespace
{
using Json = nlohmann::json;

void setError(std::string* error, const std::string& message) {
    if (error != nullptr)
        *error = message;
}

const char* presetName(ScenePreset preset) {
    switch (preset) {
    case ScenePreset::Sponza: return "sponza";
    case ScenePreset::Cornell: return "cornell";
    case ScenePreset::Hurricane: return "hurricane";
    case ScenePreset::Prism: return "prism";
    case ScenePreset::Crystal: return "crystal";
    case ScenePreset::Deer: return "deer";
    case ScenePreset::Demo: return "demo";
    }
    return "unknown";
}

std::string resolveAssetPath(const std::filesystem::path& manifestPath, const std::string& filename) {
    if (filename.empty() || filename.rfind("__", 0) == 0)
        return filename;

    const std::filesystem::path assetPath(filename);
    if (assetPath.is_absolute())
        return assetPath.lexically_normal().string();
    return (manifestPath.parent_path() / assetPath).lexically_normal().string();
}

bool readVec3(const Json& value, Vec3& output, const char* field, std::string* error) {
    if (!value.is_array() || value.size() != 3 ||
        !value[0].is_number() || !value[1].is_number() || !value[2].is_number()) {
        setError(error, std::string("scene manifest field '") + field + "' must be an array of three numbers");
        return false;
    }
    output = Vec3(value[0].get<float>(), value[1].get<float>(), value[2].get<float>());
    return true;
}

bool parseObjectSource(const std::string& value, CornellObjectSource& source) {
    if (value == "obj") source = CornellObjectSource::Obj;
    else if (value == "gltf" || value == "glb") source = CornellObjectSource::Gltf;
    else if (value == "procedural_prism") source = CornellObjectSource::ProceduralPrism;
    else return false;
    return true;
}

bool parseLightProfile(const std::string& value, CornellLightProfile& profile) {
    if (value == "standard") profile = CornellLightProfile::Standard;
    else if (value == "prism") profile = CornellLightProfile::Prism;
    else if (value == "crystal") profile = CornellLightProfile::Crystal;
    else return false;
    return true;
}

bool parseAcceleration(const std::string& value, ObjAccelerationPolicy& acceleration) {
    if (value == "tlas_blas_bvh2") acceleration = ObjAccelerationPolicy::TlasBlasBvh2;
    else if (value == "nexus_bvh2") acceleration = ObjAccelerationPolicy::NexusBvh2;
    else if (value == "nexus_bvh8") acceleration = ObjAccelerationPolicy::NexusBvh8;
    else if (value == "median_bvh") acceleration = ObjAccelerationPolicy::MedianBvh;
    else if (value == "spatial_split_bvh") acceleration = ObjAccelerationPolicy::SpatialSplitBvh;
    else return false;
    return true;
}
} // namespace

bool loadSceneManifest(const char* filename, SceneConfig& scene, std::string* error) {
    if (filename == nullptr || filename[0] == '\0') {
        setError(error, "scene manifest filename is empty");
        return false;
    }

    std::ifstream input(filename);
    if (!input) {
        setError(error, std::string("could not open scene manifest: ") + filename);
        return false;
    }

    try {
        Json root;
        input >> root;
        if (!root.is_object()) {
            setError(error, "scene manifest root must be an object");
            return false;
        }
        if (!root.contains("preset") || !root["preset"].is_string()) {
            setError(error, "scene manifest requires a string 'preset' field");
            return false;
        }

        ScenePreset preset;
        const std::string presetValue = root["preset"].get<std::string>();
        if (!parseScenePreset(presetValue.c_str(), preset)) {
            setError(error, "scene manifest contains an unknown preset: " + presetValue);
            return false;
        }

        SceneConfig loaded;
        loaded.preset = preset;
        loaded.name = root.value("name", presetName(preset));
        const std::filesystem::path manifestPath = std::filesystem::path(filename);

        if (root.contains("object")) {
            const Json& object = root["object"];
            if (!object.is_object()) {
                setError(error, "scene manifest field 'object' must be an object");
                return false;
            }
            const std::string source = object.value("source", "obj");
            if (!parseObjectSource(source, loaded.objectSource)) {
                setError(error, "scene manifest contains an unknown object source: " + source);
                return false;
            }
            if (object.contains("file")) {
                if (!object["file"].is_string()) {
                    setError(error, "scene manifest object.file must be a string");
                    return false;
                }
                const std::string path = resolveAssetPath(manifestPath, object["file"].get<std::string>());
                loaded.objectFilename = path;
                loaded.gltfFilename = path;
            }
            loaded.objectScale = object.value("scale", loaded.objectScale);
            if (object.contains("translation") &&
                !readVec3(object["translation"], loaded.objectTranslation, "object.translation", error))
                return false;
            loaded.convertObjectMaterialsToDielectric = object.value(
                "dielectric", loaded.convertObjectMaterialsToDielectric);
            loaded.objectMaterialOverride = object.value(
                "material_override", loaded.objectMaterialOverride);
        }

        loaded.gltfScale = root.value("gltf_scale", loaded.gltfScale);
        if (root.contains("room")) {
            const Json& room = root["room"];
            if (!room.is_object()) {
                setError(error, "scene manifest field 'room' must be an object");
                return false;
            }
            loaded.neutralRoom = room.value("neutral", loaded.neutralRoom);
            loaded.removeBackdrop = room.value("remove_backdrop", loaded.removeBackdrop);
        }

        if (root.contains("lighting")) {
            const Json& lighting = root["lighting"];
            if (!lighting.is_object()) {
                setError(error, "scene manifest field 'lighting' must be an object");
                return false;
            }
            const std::string profile = lighting.value("profile", "standard");
            if (!parseLightProfile(profile, loaded.lightProfile)) {
                setError(error, "scene manifest contains an unknown light profile: " + profile);
                return false;
            }
            loaded.addSponzaTopLight = lighting.value(
                "add_sponza_top_light", loaded.addSponzaTopLight);
            loaded.ignoreSponzaLightOcclusion = lighting.value(
                "ignore_sponza_light_occlusion", loaded.ignoreSponzaLightOcclusion);
        }

        if (root.contains("normal_maps")) {
            const Json& normalMaps = root["normal_maps"];
            if (!normalMaps.is_object()) {
                setError(error, "scene manifest field 'normal_maps' must be an object");
                return false;
            }
            loaded.useNormalMaps = normalMaps.value("enabled", loaded.useNormalMaps);
            loaded.normalMapMinimumCosine = normalMaps.value(
                "minimum_cosine", loaded.normalMapMinimumCosine);
        }

        if (root.contains("acceleration")) {
            if (!root["acceleration"].is_string() ||
                !parseAcceleration(root["acceleration"].get<std::string>(), loaded.acceleration)) {
                setError(error, "scene manifest contains an unknown acceleration policy");
                return false;
            }
        }

        scene = std::move(loaded);
        return true;
    } catch (const std::exception& exception) {
        setError(error, std::string("could not parse scene manifest: ") + exception.what());
        return false;
    }
}
