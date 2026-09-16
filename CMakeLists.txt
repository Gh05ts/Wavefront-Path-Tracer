cmake_minimum_required(VERSION 3.22)

project(cuda_wavefront_pathtracer
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

set(NEXUSBVH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUSBVH_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/NexusBVH)

add_executable(pathtracer
    src/main.cu
    src/core/rng.cu
    src/scene/camera.cu
    src/scene/scene.cu
    src/scene/bvh.cpp
    src/scene/bvh_sbvh.cpp
    src/scene/obj_loader.cpp
    src/scene/gltf_loader.cpp
    src/renderer/renderer.cu
    src/renderer/render_session.cu
    src/renderer/render_checkpoint.cpp
    src/renderer/tile_scheduler.cpp
)

target_include_directories(pathtracer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/tinyobjloader
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/stb
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/tinygltf
)

target_link_libraries(pathtracer
    PRIVATE
        NXB::NexusBVH
)

set_target_properties(pathtracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
