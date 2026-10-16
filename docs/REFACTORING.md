# Refactoring notes

This is an incremental cleanup plan for the current implementation. The
renderer is already featureful, so changes should preserve image behavior and
the existing CUDA kernel interfaces until there are regression checks around
them.

## Ownership today

`main.cu` now owns only process-level setup:

1. configuration and preset selection;
2. scene and camera creation;
3. scene policy flags;
4. `RenderDriver` invocation;
5. device-scene destruction after driver teardown.

`RenderDriver` owns photon preparation, resource allocation, graph capture,
sample/tile scheduling, checkpoint policy, final readback, and image output.
`RenderResources` owns the lifetime of device buffers, `RenderSession`, the
trace stream, CUDA events, CUDA graph handles, photon buffers, and the async
checkpoint writer.

`scene_host.cpp` now owns host scene construction, asset-specific policy,
light metadata, and preset transforms. `scene.cu` owns device upload,
light-table upload, acceleration-structure construction, and destruction.
`renderer.cu` contains primary-ray generation, sample advancement, and the
persistent scheduler. Header-backed device modules contain sampling,
traversal/material lookup, and path shading so the hot calls remain inline in
the same CUDA compilation unit. The photon kernels are isolated in
`photon_mapping.cu`, and PPM serialization is isolated in `image_output.cpp`.

## Contracts to preserve

### Tile and sample progression

- `RenderTile` coordinates are full-frame coordinates.
- `PathState::pixelIndex` is always a full-frame framebuffer index.
- Queue-local path indices are reused for each tile.
- Only the final tile of a sample has `advanceSample != 0`.
- The sample counter advances after the final tile, not after every tile.

### Queue capacity and compaction

- Queue capacity is the tile pixel count in tiled mode and the frame pixel
  count otherwise.
- Primary rays use their tile-local pixel index as their queue slot; one
  thread sets the count directly, without atomic reservations or a prior reset.
- The captured path uses a separate intersection-result buffer and compacts
  surviving continuations in `shadePaths` with one atomic reservation per
  warp, followed by ballot/shuffle rank calculation.
- The persistent path uses the same path-state indexing but performs tracing,
  shading, and continuation compaction inside one cooperative kernel.
- A compaction change must prove that no continuation writes past queue
  capacity and that every active lane writes exactly once.

### Resource lifetime

The trace stream must be synchronized before destroying resources used by its
queued work. The CUDA graph must be destroyed before the resources captured by
the graph. Scene BVH objects must outlive every kernel that traverses them.
Checkpoint writer threads must finish before their pinned host buffers and
CUDA events are released.

### Acceleration structure selection

The scene TLAS is a NexusBVH BVH2 over typed sphere and mesh-instance leaves;
accelerated meshes use BVH2 BLASes. `SceneSphere` and `SceneInstance` select
`AccelerationMode::Bvh` (default) or `Direct` on the host. Upload partitions
objects into TLAS leaves and separate direct sphere/instance index lists;
the policy is not uploaded with primitive data. Direct objects remain visible,
cast shadows, and retain their light registration. A mesh BLAS is built only
when at least one accelerated instance needs it. Shared direct instances
bypass that BLAS. Ground spheres explicitly select `Direct` in scene assembly.

The direct NexusBVH BVH2/BVH8 and legacy host BVH triangle paths coexist with
the sphere TLAS. Closest-hit and visibility queries must consider both.
BVH2 traversal caches inverse ray direction once per traversal, and sphere
shading attributes are computed only for the winning hit. CUDA regression
tests compare the resulting intersections and visibility against a host
brute-force reference, including alpha masking and mixed acceleration modes.

## Recommended sequence

1. Keep the current kernels stable while expanding smoke-test coverage; tests
   now cover tile coverage, scene preset parsing, checkpoint round-trips, and
   queue compaction, with CUDA tests skipped when no device is available.
2. Continue validation and performance instrumentation around the completed
   ownership and responsibility boundaries.

## Low-risk cleanup already completed

The photon kernels were moved to `src/renderer/photon_mapping.cu`; the old
disabled copies in `src/renderer/renderer.cu` have been removed. This is a
source-only cleanup and does not change the active kernel implementations.

The render driver/resource extraction is also complete. It keeps kernel
signatures and device-visible layouts unchanged while making resource teardown
explicit and ensuring the device scene outlives the driver.

Preset mapping and BVH selection are now explicit in
`scene_factory.cpp`/`scene.cuh`; the scene builder no longer infers behavior
from asset filenames or compile-time boolean switches.

Host scene assembly and CUDA upload are now separate. `HostScene` is built by
`scene_host.cpp`; `scene.cu` consumes it to allocate device buffers and build
the selected acceleration structure. CUDA tests cover checkpoint round-trips,
session reset, multi-warp compaction, and bounded queue writes.

Checkpoint format version 2 adds a stable scene/config fingerprint. The
fingerprint includes the selected scene and relevant asset contents plus
output-affecting render settings, so an incompatible `--resume` is rejected
before rendering continues.
