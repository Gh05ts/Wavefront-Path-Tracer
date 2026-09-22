cmake_minimum_required(VERSION 3.22)

project(cuda_wavefront_pathtracer
    LANGUAGES CXX CUDA
)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

find_package(CUDAToolkit REQUIRED)

set(NEXUSBVH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUSBVH_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/NexusBVH)

add_executable(pathtracer
    src/main.cu
    src/config.cpp
    src/scene/scene_manifest.cpp
    src/core/rng.cu
    src/scene/camera.cu
    src/scene/scene.cu
    src/scene/scene_host.cpp
    src/scene/scene_factory.cpp
    src/scene/bvh.cpp
    src/scene/bvh_sbvh.cpp
    src/scene/obj_loader.cpp
    src/scene/gltf_loader.cpp
    src/renderer/renderer.cu
    src/renderer/photon_mapping.cu
    src/renderer/image_output.cpp
    src/renderer/render_driver.cu
    src/renderer/render_task.cu
    src/renderer/render_fingerprint.cpp
    src/renderer/render_session.cu
    src/renderer/render_checkpoint.cpp
    src/renderer/tile_scheduler.cpp
    src/distributed/task.cpp
    src/distributed/assets.cpp
    src/distributed/checkpoint.cpp
    src/distributed/coordinator.cpp
    src/distributed/protocol.cpp
    src/distributed/transport.cpp
    src/distributed/worker.cpp
    src/distributed/coordinator_app.cpp
    src/distributed/worker_app.cu
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
        CUDA::cudart
        stdc++fs
)

set_target_properties(pathtracer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
