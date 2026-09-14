cmake_minimum_required(VERSION 3.22)

project(cuda_wavefront_pathtracer
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(pathtracer
    src/main.cu
    src/core/rng.cu
    src/scene/camera.cu
    src/scene/scene.cu
    src/scene/obj_loader.cpp
    src/renderer/renderer.cu
)

target_include_directories(pathtracer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/tinyobjloader
)

set_target_properties(pathtracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
