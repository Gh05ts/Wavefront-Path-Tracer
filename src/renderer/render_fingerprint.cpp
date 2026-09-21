#include "renderer/render_fingerprint.hpp"

#include <cstring>
#include <fstream>
#include <string>

namespace
{
class FingerprintBuilder {
public:
    FingerprintBuilder() = default;

    void appendByte(uint8_t byte) {
        value_ ^= byte;
        value_ *= fnvPrime;
    }

    void appendU32(uint32_t value) {
        for (uint32_t byte = 0; byte < sizeof(value); ++byte)
            appendByte(static_cast<uint8_t>((value >> (byte * 8)) & 0xffu));
    }

    void appendU64(uint64_t value) {
        for (uint32_t byte = 0; byte < sizeof(value); ++byte)
            appendByte(static_cast<uint8_t>((value >> (byte * 8)) & 0xffu));
    }

    void appendFloat(float value) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(value), "float fingerprint representation must be four bytes");
        std::memcpy(&bits, &value, sizeof(bits));
        appendU32(bits);
    }

    void appendBool(bool value) {
        appendByte(value ? 1 : 0);
    }

    void appendString(const char* value) {
        if (value == nullptr) {
            appendU64(0);
            return;
        }

        std::string stringValue(value);
        appendU64(stringValue.size());
        for (unsigned char byte : stringValue)
            appendByte(byte);
    }

    void appendAsset(const char* filename) {
        if (filename == nullptr) {
            appendByte(0);
            return;
        }

        std::ifstream input(filename, std::ios::binary);
        if (!input) {
            appendByte(0);
            return;
        }

        appendByte(1);
        char buffer[64 * 1024];
        while (input.read(buffer, sizeof(buffer)) || input.gcount() > 0) {
            for (std::streamsize i = 0; i < input.gcount(); ++i)
                appendByte(static_cast<uint8_t>(static_cast<unsigned char>(buffer[i])));
        }
    }

    uint64_t value() const {
        return value_;
    }

private:
    static constexpr uint64_t fnvPrime = 1099511628211ull;
    uint64_t value_ = 1469598103934665603ull;
};

void appendSceneFingerprint(FingerprintBuilder& builder, const SceneConfig& scene) {
    builder.appendU32(static_cast<uint32_t>(scene.preset));
    builder.appendString(scene.name.c_str());

    switch (scene.preset) {
    case ScenePreset::Sponza:
        builder.appendAsset(scene.gltfFilename.c_str());
        break;
    case ScenePreset::Cornell:
    case ScenePreset::Hurricane:
    case ScenePreset::Crystal:
    case ScenePreset::Deer:
        builder.appendAsset(scene.objectFilename.c_str());
        break;
    case ScenePreset::Prism:
    case ScenePreset::Demo:
        break;
    }

    builder.appendFloat(scene.gltfScale);
    builder.appendFloat(scene.objectScale);
    builder.appendFloat(scene.objectTranslation.x);
    builder.appendFloat(scene.objectTranslation.y);
    builder.appendFloat(scene.objectTranslation.z);
    builder.appendU32(static_cast<uint32_t>(scene.objectSource));
    builder.appendU32(static_cast<uint32_t>(scene.lightProfile));
    builder.appendU32(static_cast<uint32_t>(scene.acceleration));
    builder.appendBool(scene.neutralRoom);
    builder.appendBool(scene.removeBackdrop);
    builder.appendBool(scene.convertObjectMaterialsToDielectric);
    builder.appendU32(static_cast<uint32_t>(scene.objectMaterialOverride));
    builder.appendBool(scene.addSponzaTopLight);
    builder.appendBool(scene.ignoreSponzaLightOcclusion);
    builder.appendBool(scene.useNormalMaps);
    builder.appendFloat(scene.normalMapMinimumCosine);
}

void appendRenderFingerprint(FingerprintBuilder& builder, const RenderConfig& render) {
    builder.appendU32(render.width);
    builder.appendU32(render.height);
    builder.appendU32(render.tileWidth);
    builder.appendU32(render.tileHeight);
    builder.appendU32(render.maxDepth);
    builder.appendU32(render.russianRouletteStartDepth);
    builder.appendU32(render.samplesPerPixel);
    builder.appendU32(render.blockSize);
    builder.appendBool(render.tiledRendering);
    builder.appendBool(render.intersectionDebug);
    builder.appendBool(render.shadingNormalDebug);
    builder.appendBool(render.persistentWavefront);
    builder.appendBool(render.enableCaustics);
    builder.appendBool(render.enableCausticGather);
    builder.appendU32(render.causticPhotonCount);
    builder.appendU32(render.causticMaxDepth);
    builder.appendFloat(render.causticGatherRadius);
}
} // namespace

uint64_t computeRenderFingerprint(const RenderConfig& render, const SceneConfig& scene) {
    FingerprintBuilder builder;
    appendSceneFingerprint(builder, scene);
    appendRenderFingerprint(builder, render);
    return builder.value();
}
