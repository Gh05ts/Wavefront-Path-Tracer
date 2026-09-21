#include "distributed/assets.hpp"

#include "config.hpp"
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_set>

namespace
{
constexpr uint64_t fnvOffset = 1469598103934665603ull;
constexpr uint64_t fnvPrime = 1099511628211ull;

void setError(std::string* error, const std::string& message) {
    if (error != nullptr)
        *error = message;
}

std::string lowerExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
        return static_cast<char>(std::tolower(value));
    });
    return extension;
}

bool isDataUri(const std::string& value) {
    return value.rfind("data:", 0) == 0;
}

bool isWithinRoot(const std::filesystem::path& relative) {
    if (relative.empty() || relative.is_absolute())
        return false;
    for (const auto& component : relative) {
        if (component == "..")
            return false;
    }
    return true;
}

bool addAssetPath(
    const std::filesystem::path& root,
    const std::string& requestedPath,
    std::vector<std::filesystem::path>& paths,
    std::unordered_set<std::string>& seen,
    std::string* error) {
    if (requestedPath.empty() || isDataUri(requestedPath))
        return true;

    const std::filesystem::path normalized =
        (root / std::filesystem::path(requestedPath)).lexically_normal();
    const std::filesystem::path relative = normalized.lexically_relative(root);
    if (!isWithinRoot(relative)) {
        setError(error, "asset dependency escapes the primary asset directory: " + requestedPath);
        return false;
    }

    const std::string key = relative.generic_string();
    if (seen.insert(key).second)
        paths.push_back(normalized);
    return true;
}

bool addGltfDependencies(
    const std::filesystem::path& primary,
    std::vector<std::filesystem::path>& paths,
    std::unordered_set<std::string>& seen,
    std::string* error) {
    std::ifstream input(primary);
    if (!input) {
        setError(error, "could not open glTF asset: " + primary.string());
        return false;
    }

    try {
        nlohmann::json root;
        input >> root;
        for (const char* section : {"buffers", "images"}) {
            if (!root.contains(section) || !root[section].is_array())
                continue;
            for (const nlohmann::json& entry : root[section]) {
                if (entry.is_object() && entry.contains("uri") && entry["uri"].is_string() &&
                    !addAssetPath(primary.parent_path(), entry["uri"].get<std::string>(), paths, seen, error))
                    return false;
            }
        }
    } catch (const std::exception& exception) {
        setError(error, "could not parse glTF dependencies: " + std::string(exception.what()));
        return false;
    }
    return true;
}

bool addObjDependencies(
    const std::filesystem::path& primary,
    std::vector<std::filesystem::path>& paths,
    std::unordered_set<std::string>& seen,
    std::string* error) {
    std::ifstream obj(primary);
    if (!obj) {
        setError(error, "could not open OBJ asset: " + primary.string());
        return false;
    }

    std::vector<std::filesystem::path> materialLibraries;
    std::string line;
    while (std::getline(obj, line)) {
        std::istringstream stream(line);
        std::string directive;
        stream >> directive;
        if (directive != "mtllib")
            continue;

        std::string materialLibrary;
        while (stream >> materialLibrary) {
            if (!addAssetPath(primary.parent_path(), materialLibrary, paths, seen, error))
                return false;
            materialLibraries.push_back((primary.parent_path() / materialLibrary).lexically_normal());
        }
    }

    for (const std::filesystem::path& materialLibrary : materialLibraries) {
        std::ifstream mtl(materialLibrary);
        if (!mtl) {
            setError(error, "could not open OBJ material library: " + materialLibrary.string());
            return false;
        }
        while (std::getline(mtl, line)) {
            std::istringstream stream(line);
            std::string directive;
            stream >> directive;
            if (directive != "map_Kd" && directive != "map_Ka" && directive != "map_Ks" &&
                directive != "map_Bump" && directive != "bump" && directive != "norm")
                continue;

            std::string texture;
            std::string token;
            while (stream >> token)
                texture = token;
            if (!texture.empty() && !addAssetPath(materialLibrary.parent_path(), texture, paths, seen, error))
                return false;
        }
    }
    return true;
}

bool hashFile(
    const std::filesystem::path& path,
    uint64_t& byteSize,
    uint64_t& checksum,
    std::string* error) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        setError(error, "could not open asset dependency: " + path.string());
        return false;
    }

    checksum = fnvOffset;
    byteSize = 0;
    char buffer[64 * 1024];
    while (input.read(buffer, sizeof(buffer)) || input.gcount() > 0) {
        for (std::streamsize i = 0; i < input.gcount(); ++i) {
            checksum ^= static_cast<uint8_t>(static_cast<unsigned char>(buffer[i]));
            checksum *= fnvPrime;
        }
        byteSize += static_cast<uint64_t>(input.gcount());
    }
    if (!input.eof()) {
        setError(error, "could not read asset dependency: " + path.string());
        return false;
    }
    return true;
}

std::string primaryAssetPath(const SceneConfig& scene) {
    if (scene.preset == ScenePreset::Sponza)
        return scene.gltfFilename;
    if (scene.preset == ScenePreset::Cornell || scene.preset == ScenePreset::Hurricane ||
        scene.preset == ScenePreset::Crystal || scene.preset == ScenePreset::Deer) {
        if (scene.objectSource == CornellObjectSource::ProceduralPrism)
            return {};
        return scene.objectFilename;
    }
    if (scene.preset == ScenePreset::Prism && scene.objectSource != CornellObjectSource::ProceduralPrism)
        return scene.objectFilename;
    return {};
}
} // namespace

bool buildDistributedAssetCatalog(
    const SceneConfig& scene,
    DistributedAssetCatalog& catalog,
    std::string* error) {
    catalog = DistributedAssetCatalog{};
    const std::string primaryFilename = primaryAssetPath(scene);
    if (primaryFilename.empty() || primaryFilename.rfind("__", 0) == 0)
        return true;

    const std::filesystem::path primary = std::filesystem::absolute(primaryFilename).lexically_normal();
    if (!std::filesystem::is_regular_file(primary)) {
        setError(error, "primary scene asset does not exist: " + primary.string());
        return false;
    }

    const std::filesystem::path root = primary.parent_path();
    std::vector<std::filesystem::path> paths{primary};
    std::unordered_set<std::string> seen{primary.filename().generic_string()};
    const std::string extension = lowerExtension(primary);
    if (extension == ".gltf" && !addGltfDependencies(primary, paths, seen, error))
        return false;
    if (extension == ".obj" && !addObjDependencies(primary, paths, seen, error))
        return false;

    catalog.primaryRelativePath = primary.filename().generic_string();
    for (const std::filesystem::path& path : paths) {
        const std::filesystem::path relative = path.lexically_relative(root);
        if (!isWithinRoot(relative)) {
            setError(error, "asset dependency is outside the primary asset directory: " + path.string());
            return false;
        }

        DistributedAssetFile file;
        file.relativePath = relative.generic_string();
        file.sourcePath = path.string();
        if (!hashFile(path, file.byteSize, file.checksum, error))
            return false;
        catalog.files.push_back(std::move(file));
    }
    return true;
}

const DistributedAssetFile* findDistributedAsset(
    const DistributedAssetCatalog& catalog,
    const std::string& relativePath) {
    for (const DistributedAssetFile& file : catalog.files) {
        if (file.relativePath == relativePath)
            return &file;
    }
    return nullptr;
}

bool readDistributedAssetChunk(
    const DistributedAssetFile& file,
    uint64_t offset,
    uint32_t maximumBytes,
    std::vector<uint8_t>& bytes,
    bool& finalChunk,
    std::string* error) {
    bytes.clear();
    if (offset > file.byteSize || maximumBytes == 0) {
        setError(error, "invalid asset chunk range");
        return false;
    }

    std::ifstream input(file.sourcePath, std::ios::binary);
    if (!input) {
        setError(error, "could not open asset for transfer: " + file.sourcePath);
        return false;
    }
    input.seekg(static_cast<std::streamoff>(offset));
    if (!input) {
        setError(error, "could not seek asset for transfer: " + file.sourcePath);
        return false;
    }
    const uint64_t remaining = file.byteSize - offset;
    const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(remaining, maximumBytes));
    bytes.resize(count);
    if (count > 0)
        input.read(reinterpret_cast<char*>(bytes.data()), count);
    if (input.gcount() != count) {
        setError(error, "could not read requested asset chunk: " + file.sourcePath);
        return false;
    }
    finalChunk = offset + count == file.byteSize;
    return true;
}
