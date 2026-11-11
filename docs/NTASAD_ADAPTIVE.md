# NTASAD adaptive rendering integration

The separate denoiser project now has a paper audit, initial adaptive sampler,
spatial-softmax sample-map normalization and independent sample-layer replay.
See [the staged plan and dataset requirements](../../denoiser/docs/ntasad-adaptive-plan.md).
This document identifies the renderer changes needed next; CUDA adaptive
sampling is not enabled by this milestone.

Layer-only companion capture is now implemented separately: `dataset_capture`
accepts `sample_layers` with disjoint offsets and exports only their linear RGB
means. The [v5 dataset workflow](../../denoiser/dataset_generation/V5.md) copies
verified v4 frames, reuses their 4-SPP layer/reference/guides, and captures the
missing 1/2/8/16/32-SPP layers. This supports offline sample replay; live per-pixel
adaptive primary-ray scheduling described below is still pending.

## Existing execution contract

`RenderTaskRenderer::render` in `src/renderer/render_task.cu` accepts a rectangular
tile and one contiguous sample interval shared by every pixel. It clears that
tile, loops over sample rounds, and returns radiance **sums**. The current
`RenderTaskResult::sampleCount` is one scalar, and the distributed task identity
has the same uniform-range assumption.

`generatePrimaryRays` in `src/renderer/renderer.cu` assigns one primary slot per
tile pixel, uses `pixelRngKey(pixel, *sampleIndex, rngStrategy)`, and sets the
primary queue count to the tile's pixel count. The cooperative
`persistentWavefrontTrace` initializes the same dense primary queue. Neither
accepts a per-pixel count map. Tiled/full-frame determinism depends on retaining
the global pixel index and absolute sample index in the RNG key.

`RenderDriver::createWavefrontGraph` captures the uniform primary launch and
sample increment; `launchPersistentTile` supplies the cooperative kernel's
arguments. Any new optional map arguments must be connected through both paths
and have lifetime covering graph replay. `advanceSampleIndex` advances a global
round, not each pixel's completed count.

The frame exporter and the Python frame-v1 contract store uniform positive
counts. The Python reader verifies counts against manifest SPP. Writing a
variable map into that schema without a new explicit contract will fail
validation or misrepresent the capture.

## Proposed local implementation

1. Introduce a local adaptive task/result separate from the distributed uniform
   task identity. Validate a full-image uint32 count map, optional per-pixel
   sample starts, tile bounds, reserved intervals and overflow. The result must
   retain each pixel's actual count alongside its radiance sum.
2. At sample round r, compact only pixels whose target count exceeds r into the
   primary queue. Initialize their paths using the original global pixel ID
   and absolute sample start + r. Do not enqueue zero-count pixels or use compact
   queue offsets as RNG pixel IDs. Reset both primary/secondary counts between
   rounds, including an entirely empty round. Never permit early returns before
   cooperative grid synchronization in the persistent kernel.
3. Reuse existing intersection, shading and secondary compaction. Atomics and
   path slots must retain the correct destination pixel. Queue capacity can
   remain tile-sized for a first implementation, since each round emits at
   most one primary per pixel. Optimize primary compaction only after parity.
4. Normalize each pixel by its own count outside the sum buffer; leave a defined
   zero placeholder for missing radiance and preserve its mask for the adaptive
   denoiser. Guides remain a separate unjittered pass at every pixel.
5. Add versioned adaptive export and optional independent RGB layer capture.
   Preserve the current uniform capture and distributed/checkpoint formats.
   Companion-layer capture must export independent 1/2/4/8/16/32-SPP means with
   disjoint RNG intervals and matching cameras/scene/integrator provenance.

The paper's equation (2) has no explicit lower bound or finite-layer upper
bound. Do not force a uniform pilot sample, silently floor to one, or clip to
63 without recording that policy as a method variant. Independent rounding can
change total paths; record measured totals for comparisons.

## Required verification

* Uniform map reproduces current uniform tasks within the existing FP32 tolerance
  and uses exactly the same paths/sample IDs.
* Zero, one, sparse, checkerboard and concentrated count maps accumulate the
  expected paths; all-zero maps launch no paths and return zero sums/counts.
* Partial edge tiles and full-frame execution agree, with no queue overflow.
* Graph and persistent execution agree on radiance and counts.
* Selected pixels agree with independent renders of their absolute sample ranges;
  independent companion layers agree with their constituent sample sums.
* Input, layer and reference ranges never overlap per pixel/frame, including
  separately captured report references and resumed/out-of-order frame capture.
* Exported counts survive EXR round-trip and adaptive-reader validation.

CPU verification covers sample-map policy, interval metadata and layer replay.
The initial sandbox hid GPU access; elevated execution now exposes an RTX 3090.
Seven layer-capture GPU tests passed: independent sample-range equality,
tiled/full/persistent parity, 16-frame source-copy/capture/resume/read/replay,
and invalid configuration rejection. Production v4 frames remain on the other
machine, so their reuse/pilot checks await their arrival. No real-time performance
or production adaptive quality has been established.
