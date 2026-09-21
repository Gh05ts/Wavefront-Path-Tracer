#pragma once

#include <cstdint>

#include "../config.hpp"

uint64_t computeRenderFingerprint(const RenderConfig& render, const SceneConfig& scene);
