#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct SceneConfig;

struct DistributedAssetDescriptor {
    std::string relativePath;
    uint64_t byteSize = 0;
    uint64_t checksum = 0;
};

struct DistributedAssetFile {
    std::string relativePath;
    std::string sourcePath;
    uint64_t byteSize = 0;
    uint64_t checksum = 0;
};

struct DistributedAssetCatalog {
    // The primary asset path is relative to the catalog root and is used to
    // rewrite the worker's SceneConfig after transfer.
    std::string primaryRelativePath;
    std::vector<DistributedAssetFile> files;
};

bool buildDistributedAssetCatalog(
    const SceneConfig& scene,
    DistributedAssetCatalog& catalog,
    std::string* error = nullptr);

const DistributedAssetFile* findDistributedAsset(
    const DistributedAssetCatalog& catalog,
    const std::string& relativePath);

bool readDistributedAssetChunk(
    const DistributedAssetFile& file,
    uint64_t offset,
    uint32_t maximumBytes,
    std::vector<uint8_t>& bytes,
    bool& finalChunk,
    std::string* error = nullptr);
