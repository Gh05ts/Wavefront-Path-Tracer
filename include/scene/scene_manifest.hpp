#pragma once

#include <string>

#include "../config.hpp"

// Loads a scene description without allocating CUDA resources. Asset paths
// are resolved relative to the manifest file when they are relative paths.
bool loadSceneManifest(const char* filename, SceneConfig& scene, std::string* error = nullptr);
