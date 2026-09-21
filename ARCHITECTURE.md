# CUDA Wavefront Path Tracer

This document describes the current implementation as it exists in the source tree. It is intentionally a map of ownership and data flow, rather than a design proposal.

## Runtime pipeline

`src/main.cu` is the application entry point. `RenderDriver` is the host-side
render coordinator and `RenderResources` owns the CUDA execution resources:

1. Parse render and scene configuration.
2. Load a preset through the scene factory and upload geometry, materials, textures, lights, and acceleration structures.
3. Construct `RenderDriver` with the scene, camera, and configuration.
4. Optionally emit and trace photons, then build the device photon grid.
5. Allocate path, ray, intersection, tile, graph, stream, and checkpoint resources.
6. Render every sample and tile using one of two schedulers.
7. Optionally checkpoint the accumulated framebuffer.
8. Download, normalize, and write the final PPM image.

`main.cu` destroys the device scene only after the driver and all resources have
gone out of scope. This keeps scene BVHs alive until no queued or captured work
can reference them.

## Renderer execution modes

### Captured wavefront

The default path captures a CUDA graph containing:

`generatePrimaryRays -> (intersectScene -> shadePaths) * maxDepth -> advanceSampleIndex`

`shadePaths` appends surviving paths to a secondary queue. Active lanes form a
warp ballot, compute their compacted rank, and lane zero reserves one output
range for the warp. The graph is replayed for each tile and sample after the
host updates the device-side `RenderTile`.

### Persistent wavefront

`--persistent-wavefront` launches `persistentWavefrontTrace` cooperatively. Resident blocks repeatedly drain chunks of the current queue. Each warp uses `__ballot_sync`, a lane-0 atomic reservation, and `__shfl_sync` to compact active continuations. `cooperative_groups::this_grid().sync()` separates queue initialization, each bounce, and queue swapping.

This mode is experimental and requires device cooperative-launch support. `--profile-queues` records queue occupancy by bounce for the last processed tile.

Device responsibilities are split into header-backed CUDA modules included by
`renderer.cu` so the hot device calls remain inline within one compilation
unit:

- `device_sampling.cuh`: cosine sampling, reflection, refraction, and random-vector helpers.
- `device_traversal.cuh`: sphere/triangle/BVH traversal, visibility tests, and material texture lookup needed during hits.
- `device_path_shading.cuh`: captured shading, persistent path-state transitions, and continuation generation.
- `renderer.cu`: primary-ray generation, sample advancement, and persistent scheduling orchestration.

## Scene and acceleration structures

Scene loading is split into host assembly and device upload:

- `src/scene/obj_loader.cpp`: OBJ/MTL input.
- `src/scene/gltf_loader.cpp`: glTF/GLB input, including texture/material bindings.
- `assets/scenes/*.json`: named scene manifests. Each manifest selects the preset, asset source/path, transforms, room/light policy, normal-map policy, and acceleration policy.
- `src/scene/scene_manifest.cpp`: parses manifests and resolves asset paths relative to the manifest file.
- `src/scene/scene_factory.cpp`: maps the parsed scene configuration to explicit scene/camera and build policies.
- `src/scene/scene_host.cpp`: assembles host geometry, materials, transforms, lights, and acceleration policy.
- `src/scene/scene.cu`: converts `HostScene` to device allocations, NexusBVH/legacy BVHs, textures, and scene destruction.

Cornell-style object loading uses `CornellSceneOptions`, which makes the object
source, room palette, backdrop handling, material conversion, and light profile
explicit. OBJ scenes use `ObjSceneOptions` and `ObjAccelerationPolicy` rather
than filename tests or compile-time boolean switches.

Cornell/glTF-style scenes use a TLAS over mesh instances and one NexusBVH BVH2 BLAS per mesh. The simple/demo path can use the direct triangle representation and NexusBVH H-PLOC BVH2. BVH8 support is represented in the scene types and traversal code, but the current scene setup selects BVH2 (`useNexusBvh8 = false`). The legacy host BVH builders remain in `bvh.cpp` and `bvh_sbvh.cpp` for fallback/comparison paths.

Named presets are now data-driven rather than a second set of hard-coded scene
switches. `--scene NAME` loads `assets/scenes/NAME.json`, while
`--scene-file FILE` loads an arbitrary manifest with the same schema. The
existing preset names remain stable for scripts and checkpoints. The manifest
loader only populates host-side `SceneConfig`; it does not introduce JSON into
CUDA code or the distributed wire protocol.

## Progressive rendering and checkpointing

`RenderSession` owns the device framebuffer and sample counter. Tiles accumulate into the full-resolution framebuffer, so tiling limits queue memory without producing independent tile images. The sample index advances only after the final tile of a sample.

Checkpoint files contain a small header (magic, format version, dimensions,
sample index, and scene/config fingerprint) followed by the accumulated `Vec3`
framebuffer. Checkpoints are copied asynchronously into pinned host slots and
written by worker threads through a temporary file followed by rename. Resume
rejects a checkpoint whose fingerprint does not match the current scene or
output-affecting render settings.

## Distributed rendering architecture

The distributed renderer will parallelize independent Monte Carlo work at the
tile/sample-range level. It will not distribute individual rays, bounces, BVH
queries, or photon-map entries across the network. Each worker keeps its scene,
acceleration structures, textures, and optional photon map resident on its local
GPU and returns only accumulated tile radiance.

The local renderer remains the reference execution path. Distributed scheduling
must be independent of whether a worker uses the captured wavefront or the
experimental persistent wavefront backend.

### Planned host/device boundaries

The coordinator is a host-only process. It owns the job specification, task
queue, leases, accumulated framebuffer, completed-task ledger, and distributed
checkpoint. It never needs a CUDA context.

Each worker loads the scene once, builds/uploads its `DeviceScene`, optionally
builds the photon map once, renders assigned tasks, and sends results back to the
coordinator. Worker failures return leased tasks to the pending queue.

The CUDA renderer now exposes a task-oriented host API in addition to the
current full-frame driver:

```text
RenderTaskRenderer::render(task)
    -> local tile radiance sum
```

`RenderTaskRenderer` reuses `RenderDriver`'s scene, photon, graph, and
persistent-wavefront resources. It clears only the requested tile region,
renders the task's absolute sample range, and returns an unnormalized tile
sum. The task uses global image coordinates and the existing deterministic
pixel/sample seed scheme. The current implementation still keeps the driver's
full-resolution device framebuffer for compatibility; tile-local result
transport is established before later resource slimming.

### Distributed files

```text
include/distributed/
    task.hpp            Task IDs, tile/sample ranges, and result metadata
    coordinator.hpp     Host-only task queue, leases, and result commits
    checkpoint.hpp      Distributed checkpoint model and serialization
    job.hpp             Stable job specification and fingerprint inputs (next)
    protocol.hpp        Versioned wire message types
    transport.hpp       Framed TCP transport
    worker.hpp          Worker registration and task loop
    coordinator_app.hpp Coordinator process entry point
    worker_app.hpp      CUDA worker process entry point

src/distributed/
    assets.cpp
    checkpoint.cpp
    coordinator.cpp
    task.cpp
    protocol.cpp
    transport.cpp
    worker.cpp
    coordinator_app.cpp
    worker_app.cu

include/renderer/
    render_task.cuh     GPU task-rendering interface

src/renderer/
    render_task.cu      Tile/sample-range execution using existing kernels
```

The existing `RenderDriver` remains responsible for local full-frame rendering.
Its task execution details should be extracted incrementally into
`render_task.cu`; the distributed layer must not duplicate CUDA graph,
persistent-wavefront, photon, or scene-upload logic.

### Task model

The coordinator deterministically partitions a job into tasks:

```text
TaskId = (tileOrdinal, sampleStart, sampleCount)
```

Each task includes the tile rectangle, an absolute sample range, the job
fingerprint, and a protocol/version identifier. A practical first batch size is
8–16 samples per 256x256 tile, balancing network overhead against retry cost.

Workers return unnormalized RGB sums and the sample count. The coordinator
adds the result to the global accumulated framebuffer and tracks per-pixel
sample counts. `DistributedCoordinator` now commits results by `TaskId`
exactly once, so retrying a timed-out task cannot double-count its radiance.
Its in-process test covers two workers receiving different tasks, duplicate
results, lease expiry, and global tile accumulation.

The current pixel/sample seed derivation is part of the distributed contract:
the absolute sample index must be preserved when a task is retried or rendered
by a different worker.

### Task lifecycle and failure handling

```text
pending -> leased -> completed
             |
             +-- lease timeout / disconnect -> pending
```

Workers register capabilities and the job fingerprint, request work, send a
result, and renew leases for long tasks. The coordinator accepts the first
valid result for a task ID and ignores duplicate attempts. A worker may cache
the scene and photon map by job fingerprint, but task ownership is never stored
only on the worker.

### Distributed checkpoint format

The existing local checkpoint format remains unchanged. Distributed rendering
now uses a separate versioned binary format containing:

- job/config and scene fingerprint;
- resolution, tile dimensions, sample target, and batch size;
- accumulated float framebuffer;
- per-pixel sample counts;
- deterministic task completion bitmap and snapshot sequence number.

`writeDistributedCheckpoint` writes the accumulation buffer, sample counts,
and completion ledger to a temporary file, validates the payload with a
checksum, and atomically renames it into place. Leased tasks are not persisted
as completed. After coordinator restart, `restore` makes all unfinished
leases pending again. A coordinator crash may requeue work after the latest
snapshot, but the task ID prevents any result from being merged twice.

### Scene and photon distribution

The first implementation assumes workers have access to the same asset bundle
and renderer build. The coordinator sends the scene/config manifest and
fingerprint; workers validate their local assets before accepting tasks. The
scene-manifest layer is now in place, and asset shipping/cache population is
available through `--push-assets`. The coordinator builds a catalog of the
primary geometry file and loader dependencies, sends it in the registration
response, and serves missing files in bounded chunks. Workers verify every
chunked file by size and FNV-1a checksum, cache it under a job-fingerprint
directory, rewrite the primary scene path, and validate the final render
fingerprint before creating CUDA scene state. The manifest/configuration itself
is still supplied to each process separately; transferring that job definition
is the next distributed step.

For caustic renders, each worker builds the deterministic photon map once and
reuses it for all assigned tasks. The photon map is derived worker state and
does not need to be included in the distributed checkpoint; a restarted worker
can rebuild it from the job manifest.

### Process layout and implementation order

The intended command roles are:

```text
pathtracer --coordinator --listen-host 0.0.0.0 --listen-port 9000 \
    --distributed-checkpoint render.distributed.chk ...
pathtracer --worker --worker-id gpu-0 \
    --coordinator-host 192.168.1.10 --coordinator-port 9000 ...
```

Progress and remaining implementation sequence:

1. **Complete:** define stable task IDs, job fingerprints, and deterministic
tile/sample coverage.
2. **Complete:** extract and compile local tile/sample-range rendering without
networking.
3. **Complete:** add a coordinator task ledger and in-process worker simulation.
4. **Complete:** add distributed checkpoint snapshots and resume validation.
5. **Complete:** versioned framed TCP transport, worker registration, lease
renewal, disconnect requeue, and persistent task/result exchange.
6. **In progress:** add worker capability reporting and production coordinator
retry/monitoring policies around the completed lease and heartbeat primitives.
7. **Complete:** add coordinator and CUDA worker command modes while
preserving the existing local command and checkpoint format.
8. **In progress:** run the first end-to-end GPU coordinator/worker render and
add worker-failure and coordinator-restart integration tests.
9. **Complete:** move named scene presets to JSON manifests and add
`--scene-file` without changing the scene factory or CUDA interfaces.
10. **Complete:** distribute and cache manifest-referenced assets before
worker scene construction with chunked, checksum-validated transfers.
11. **Next:** transfer the authoritative scene manifest and render
configuration from coordinator to workers, rather than requiring matching
configuration arguments on every process.

The first end-to-end milestone is a single coordinator and one worker producing
the same image as local rendering, followed by worker-failure and coordinator-
restart tests before adding multiple machines.

## Main source boundaries

| Area | Primary files | Responsibility |
|---|---|---|
| CLI/config | `include/config.hpp`, `src/config.cpp` | Presets and command-line options |
| Scene model | `include/scene/*.cuh` | Device-visible geometry/material/instance layouts |
| Scene loading | `assets/scenes/*.json`, `src/scene/*loader*`, `src/scene/scene_manifest.cpp`, `src/scene/scene_factory.cpp`, `src/scene/scene_host.cpp` | Manifest parsing, OBJ/glTF conversion, preset-to-policy mapping, host scene assembly |
| Distributed assets | `include/distributed/assets.hpp`, `src/distributed/assets.cpp`, `include/distributed/protocol.hpp`, `src/distributed/worker.cpp` | Dependency cataloging, chunked transfer, per-job worker cache, and checksum validation |
| Scene upload | `src/scene/scene.cu` | `HostScene` upload, device allocation, NexusBVH, lights, textures |
| Path tracing | `include/renderer/renderer.cuh`, `include/renderer/device_*.cuh`, `src/renderer/renderer.cu` | Device sampling, traversal, shading, and queue scheduling |
| Caustics | `src/renderer/photon_mapping.cu` | Photon emission, tracing, and grid gathering |
| Scheduling | `src/renderer/tile_scheduler.cpp`, `include/renderer/render_driver.cuh`, `src/renderer/render_driver.cu` | Tile generation and sample/tile orchestration |
| CUDA resources | `include/renderer/render_driver.cuh`, `src/renderer/render_driver.cu` | Queue, session, stream, event, graph, photon, and checkpoint ownership |
| Persistence/output | `src/renderer/render_session.cu`, `src/renderer/render_checkpoint.cpp`, `src/renderer/image_output.cpp` | Framebuffer lifetime, checkpoint I/O, and PPM serialization |

## Baseline verification

The existing build and host smoke tests were verified with:

```text
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

It completes successfully in the current workspace.

## Cleanup direction

The remaining cleanup sequence is now focused on validation and performance
instrumentation around the completed ownership and responsibility boundaries.

Future work should validate the RNG and sample coverage rather than relying
only on visual inspection. Add tests for seed collisions, value histograms,
neighbor/sample correlations, and Monte Carlo convergence versus sample count.
Compare the current xorshift32 stream and seed mixer against a stronger
initializer or sampler such as PCG, Philox, or Sobol before changing the
rendering sequence.

The resource/driver, typed scene-policy, host-scene, compaction, and device
responsibility extractions
keep the existing device-visible scene layouts and kernel interfaces stable.
Checkpoint and compaction tests now cover persistence, session reset, bounded
queue writes, and multi-warp active-item preservation. Checkpoint format
version 2 also records a stable scene/config fingerprint, rejecting an
incompatible resume before rendering continues.

## Future work

- Complete an end-to-end multi-process asset-transfer test with a coordinator
  and worker, including cache reuse, interrupted transfers, and a worker that
  starts without the scene assets installed.
- Transfer the JSON scene manifest and render configuration from the
  coordinator so workers do not need manually synchronized `--scene`,
  `--scene-file`, or render-affecting options. Preserve fingerprint validation
  after applying the received configuration.
- Add a manifest `camera` block for position, target/orientation, up vector,
  and vertical field of view. Keep the existing Cornell/demo camera defaults
  when the block is absent.
- Expose the renderer's existing affine instance transform in manifests. The
  underlying `InstanceTransform` already supports translation, rotation,
  nonuniform or negative scale, and nonsingular skew/shear through its 3x3
  basis and inverse; the manifest currently exposes only uniform scale and
  translation.
- Add camera and transform fields to the render fingerprint and regression
  tests for equivalent local and distributed renders.
