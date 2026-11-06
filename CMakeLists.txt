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
find_package(OpenSSL REQUIRED)

set(NEXUSBVH_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NEXUSBVH_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(extern/NexusBVH)

set(PATHTRACER_SOURCES
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
    src/distributed/checkpoint_writer.cpp
    src/distributed/coordinator.cpp
    src/distributed/protocol.cpp
    src/distributed/transport.cpp
    src/distributed/worker.cpp
    src/distributed/coordinator_app.cpp
    src/distributed/worker_app.cu
)

add_executable(pathtracer src/main.cu ${PATHTRACER_SOURCES})
set(BENCHMARK_SOURCES ${PATHTRACER_SOURCES})
list(FILTER BENCHMARK_SOURCES EXCLUDE REGEX "src/distributed/")
add_executable(render_benchmark EXCLUDE_FROM_ALL benchmarks/render_benchmark.cpp ${BENCHMARK_SOURCES})

option(PATHTRACER_BUILD_DATASET_CAPTURE "Build HDR denoising dataset capture tools (requires OpenEXR)" OFF)
set(RENDERER_TARGETS pathtracer render_benchmark)
if(PATHTRACER_BUILD_DATASET_CAPTURE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENEXR REQUIRED IMPORTED_TARGET OpenEXR)
    add_executable(dataset_capture
        tools/dataset_capture.cu
        src/renderer/frame_export.cpp
        ${BENCHMARK_SOURCES}
    )
    target_link_libraries(dataset_capture PRIVATE PkgConfig::OPENEXR)
    list(APPEND RENDERER_TARGETS dataset_capture)
endif()

foreach(renderer_target IN LISTS RENDERER_TARGETS)
    target_include_directories(${renderer_target}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/extern/tinyobjloader
            ${CMAKE_CURRENT_SOURCE_DIR}/extern/stb
            ${CMAKE_CURRENT_SOURCE_DIR}/extern/tinygltf
    )

    target_link_libraries(${renderer_target}
        PRIVATE
            NXB::NexusBVH
            CUDA::cudart
            OpenSSL::Crypto
            stdc++fs
    )

    set_target_properties(${renderer_target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endforeach()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
