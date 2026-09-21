#pragma once

#include <cstdint>

#include "../config.hpp"
#include "camera.cuh"
#include "scene.cuh"

Camera createSceneCamera(const SceneConfig& config, uint32_t width, uint32_t height);
DeviceScene createSceneFromConfig(const SceneConfig& config);
void applySceneConfig(DeviceScene& deviceScene, const SceneConfig& config);
