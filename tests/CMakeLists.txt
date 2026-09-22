find_package(Threads REQUIRED)

add_executable(tile_scheduler_tests
    tile_scheduler_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/renderer/tile_scheduler.cpp
)

target_include_directories(tile_scheduler_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(tile_scheduler_tests PRIVATE CUDA::cudart stdc++fs)
add_test(NAME tile_scheduler_tests COMMAND tile_scheduler_tests)

add_executable(distributed_task_tests
    distributed_task_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/task.cpp
)

target_include_directories(distributed_task_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(distributed_task_tests PRIVATE CUDA::cudart stdc++fs)
add_test(NAME distributed_task_tests COMMAND distributed_task_tests)

add_executable(distributed_coordinator_tests
    distributed_coordinator_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/task.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/checkpoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/coordinator.cpp
)

target_include_directories(distributed_coordinator_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(distributed_coordinator_tests PRIVATE CUDA::cudart stdc++fs)
add_test(NAME distributed_coordinator_tests COMMAND distributed_coordinator_tests)

add_executable(distributed_transport_tests
    distributed_transport_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/assets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/task.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/checkpoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/coordinator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/transport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/worker.cpp
)

target_include_directories(distributed_transport_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/NexusBVH/NexusBVH/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/tinygltf
)

target_link_libraries(distributed_transport_tests PRIVATE Threads::Threads CUDA::cudart stdc++fs)
target_link_libraries(distributed_transport_tests PRIVATE Threads::Threads)
add_test(NAME distributed_transport_tests COMMAND distributed_transport_tests)

add_executable(distributed_worker_tests
    distributed_worker_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/assets.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/task.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/checkpoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/coordinator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/protocol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/transport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/distributed/worker.cpp
)

target_include_directories(distributed_worker_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/NexusBVH/NexusBVH/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/tinygltf
)

target_link_libraries(distributed_worker_tests PRIVATE CUDA::cudart stdc++fs)
add_test(NAME distributed_worker_tests COMMAND distributed_worker_tests)

add_executable(config_tests
    config_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/scene/scene_manifest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/renderer/render_fingerprint.cpp
)

target_include_directories(config_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/NexusBVH/NexusBVH/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../extern/tinygltf
)

target_link_libraries(config_tests PRIVATE CUDA::cudart stdc++fs)
add_test(NAME config_tests COMMAND config_tests)

add_executable(checkpoint_tests
    checkpoint_tests.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/renderer/render_checkpoint.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/renderer/render_session.cu
)

target_include_directories(checkpoint_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(checkpoint_tests PRIVATE Threads::Threads)
set_target_properties(checkpoint_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(checkpoint_tests PRIVATE Threads::Threads CUDA::cudart stdc++fs)
add_test(NAME checkpoint_tests COMMAND checkpoint_tests)

add_executable(queue_compaction_tests
    queue_compaction_tests.cu
)

target_include_directories(queue_compaction_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(queue_compaction_tests PRIVATE CUDA::cudart stdc++fs)
add_test(NAME queue_compaction_tests COMMAND queue_compaction_tests)
