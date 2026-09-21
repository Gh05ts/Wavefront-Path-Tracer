# CUDA Wavefront Path Tracer

This is a CUDA path tracer built around a wavefront renderer. It supports two
execution modes:

- captured wavefront execution using a CUDA graph;
- an experimental persistent wavefront kernel using cooperative grid
  synchronization and warp-level queue compaction.

The renderer also has tiled progressive rendering, asynchronous framebuffer
checkpoints, OBJ/MTL and glTF/GLB loading, emissive triangle sampling, texture
maps, normal maps, alpha masking, transmission/volume attenuation, and
optional photon-mapped caustics. NexusBVH supplies the H-PLOC acceleration
structures. The current scene factories use BVH2 TLAS/BLAS traversal by
default; BVH8 traversal code is present but not selected by the active setup.

## Build

The project requires CMake, a CUDA toolkit, and a CUDA-capable compiler. The
default CMake target is configured for compute capability 8.6 in
`CMAKE_CUDA_ARCHITECTURES`; change that setting for another GPU generation.

```sh
cmake -S . -B build
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

## Run

Run from `build/` because the default scene paths are relative to that
directory:

```sh
cd build
./pathtracer --scene sponza
```

Available presets are `sponza`, `cornell`, `hurricane`, `prism`, `crystal`,
`deer`, and `demo`. Useful runtime switches include:

```text
--help                 show command-line help
--list-scenes          list presets
--persistent-wavefront use the cooperative persistent renderer
--profile-queues       report persistent queue occupancy
--caustics             enable photon tracing for caustic-capable presets
--no-caustic-gather    trace photons without camera-side gathering
--resume               load render.checkpoint
```

Named presets are defined in `assets/scenes/*.json`. To render a custom scene
manifest, use `--scene-file` from the build directory (asset paths in the
manifest are resolved relative to that manifest):

```sh
./pathtracer --scene-file ../assets/scenes/prism.json --caustics
```

Distributed workers normally need the same asset bundle. To let the
coordinator transfer the manifest-referenced geometry and texture dependencies
instead, pass `--push-assets` to both processes:

```sh
# coordinator
./pathtracer --coordinator --push-assets --scene sponza

# worker
./pathtracer --worker --push-assets --worker-id gpu-0 \
    --coordinator-host coordinator-hostname
```

Workers cache verified files under `.pathtracer-assets/` by default; override
that location with `--asset-cache DIR`. The coordinator still selects the
scene/render configuration through the command line or `--scene-file`; the
next distributed step is transferring that manifest/configuration itself.

The output image defaults to `render.ppm`. Checkpoints are written to
`render.checkpoint` when the asynchronous checkpoint queue is able to accept
one.

## Where to look

`ARCHITECTURE.md` is the current data-flow and ownership map. The main source
boundaries are:

| Area | Entry points |
| --- | --- |
| CLI and presets | `include/config.hpp`, `src/config.cpp`, `assets/scenes/*.json`, `src/scene/scene_manifest.cpp` |
| Host scene ingestion | `src/scene/obj_loader.cpp`, `src/scene/gltf_loader.cpp`, `src/scene/scene_host.cpp` |
| Scene factory/policy | `include/scene/scene_factory.hpp`, `src/scene/scene_factory.cpp` |
| Device scene and BVHs | `include/scene/scene.cuh`, `src/scene/scene.cu` |
| Path tracing and traversal | `include/renderer/renderer.cuh`, `include/renderer/device_*.cuh`, `src/renderer/renderer.cu` |
| Photon mapping | `src/renderer/photon_mapping.cu` |
| Tiling, persistence, and output | `src/renderer/tile_scheduler.cpp`, `src/renderer/render_session.cu`, `src/renderer/render_checkpoint.cpp`, `src/renderer/image_output.cpp` |
| Render orchestration/resources | `include/renderer/render_driver.cuh`, `src/renderer/render_driver.cu` |
| Application entry point | `src/main.cu` |

The current refactoring priorities and invariants are recorded in
`docs/REFACTORING.md`.

## Current limitations

- Device scene allocations are still mostly raw-pointer based, although render
  execution resources now have an explicit owner.
- Checkpoints include a scene/config fingerprint and reject resumes from a
  different scene or output-affecting render configuration.
- CTest includes CUDA checkpoint and queue-compaction tests; they skip cleanly
  on machines without a CUDA device.
