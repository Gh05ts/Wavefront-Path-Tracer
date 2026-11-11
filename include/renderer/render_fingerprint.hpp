#pragma once

#include <cstdint>

#include "../config.hpp"
inline constexpr uint32_t rendererIntegratorVersion = 12;

uint64_t computeRenderFingerprint(const RenderConfig& render, const SceneConfig& scene);

// Scene/config identity for dataset grouping, independent of the display name,
// camera, resolution, RNG, and sample budget. This is not a checkpoint key.
uint64_t computeSceneFingerprint(const SceneConfig& scene);
