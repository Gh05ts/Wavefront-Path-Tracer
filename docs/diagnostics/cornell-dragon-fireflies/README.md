# Cornell dragon firefly regression — 2026-11-10

This check uses the existing `assets/scenes/cornell.json` preset, its default
camera, and `DragonAttenuation.gltf`. The room retains its red/green walls,
16-radiance area light, original glass/volume materials and backdrop removal.
There is **no HDR environment**. Maximum path depth is 32, roulette starts at
12, and photon-mapped caustics are disabled, matching the default CLI settings.
No renderer, asset or dataset-generation source was changed for this check.

The build and outputs are separate from the running dataset job:

- Build: `/tmp/dragon-cornell-check-build`, CUDA architecture 86, Release.
- Captures and logs: `/tmp/dragon-cornell-check/`.
- Dataset executable left unchanged: `/tmp/ntasad-capture-build/dataset_capture`.

The build reuses the installed OpenEXR dependency prefix without rebuilding or
changing that installation. Only the `dataset_capture` target is compiled; it
links the actual renderer/shaders used for the image checks. Both renders and
dataset generation share the RTX 3090, so these timings are not benchmarks.

Reproduce from the renderer root, using a fresh capture output directory:

```sh
/tmp/ntasad-venv/bin/cmake -S . -B /tmp/dragon-cornell-check-build \
  -G Ninja -DCMAKE_MAKE_PROGRAM=/tmp/ntasad-venv/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_PREFIX_PATH=/tmp/ntasad-capture-deps \
  -DPATHTRACER_BUILD_DATASET_CAPTURE=ON -DBUILD_TESTING=OFF
/tmp/ntasad-venv/bin/cmake --build /tmp/dragon-cornell-check-build \
  --target dataset_capture -j1
/tmp/ntasad-venv/bin/python docs/diagnostics/cornell-dragon-fireflies/validate.py \
  --output /tmp/dragon-cornell-repeat
```

The [preview configuration](preview.json) also runs directly with that
`dataset_capture` executable. `[{}]` selects the preset's camera. The validation
captures two independent replicas at 256 and 1024 SPP, each with clamp 0 and
10, then renders a 640×360 1024-SPP preview. Images use fixed exposure with
Reinhard plus gamma 1/2.2, with no denoising.

A frozen integrator-10 executable from before the clamp addition supplies an
additional 320×180 1024-SPP baseline. Frozen production-integrator-11 controls
use the same architecture-75 build to separate clamp changes from GPU
architecture differences. The new isolated build targets architecture 86.
The old executable and the active dataset executable are never rebuilt or
overwritten. Digests, settings and numerical comparisons are retained with
the result report.

## Results

The isolated build completed successfully. All nine new-build captures have
finite buffers. All guides and sample counts match between raw/clamped pairs.
The primary-floor mask covers 6,286 pixels in each 320×180 capture, identified
using geometric normal +Y and the authored floor albedo, rather than a rectangle
that includes parts of the dragon.

| SPP | Clamp | Whole-image unstable outliers, two replicas | Floor outliers | Replica noise RMSE |
| --- | --- | --- | --- | --- |
| 256 | 0 | 2 / 8 | 0 / 0 | 0.024035 |
| 256 | 10 | 2 / 5 | 0 / 0 | 0.023793 |
| 1024 | 0 | 0 / 1 | 0 / 0 | 0.012465 |
| 1024 | 10 | 0 / 1 | 0 / 0 | 0.012345 |

The unstable-outlier rule requires linear luminance >0.1, >4 times the median
of eight neighbors, >4 times the other replica and >0.1 excess over that replica.
It can flag real small highlights. Residual glass-highlight variation remains;
the test does not claim that every bright glass pixel is an artifact or that
all sample budgets are noise-free. There is no new floor-firefly regression
detected in these captures. The source area light remains at RGB radiance 16;
the raw/clamped images retain that maximum.

With clamping disabled, the current 1024-SPP renderer matches the frozen
integrator-10 reference within maximum absolute reference RGB difference
**2.384e-7** (RGB RMS **6.409e-9** in the same-architecture production control).
Guides/counts are identical. The native architecture-86 build also has maximum
reference-channel difference 2.384e-7 against that reference. This is a bounded
numerical regression check, not proof of identical paths for every possible
sample. Clamp 10 reduces average sampled brightness by approximately **0.035%**
in the frozen production control, with unchanged guides/counts.

The active dataset executable's SHA-256 is unchanged before and after the test.
Its controller remained active and started a subsequent capture after the
original dataset render completed. No renderer source fix was needed; only
this diagnostic configuration, script, images and report were added.

View the [640×360 1024-SPP preview](preview.png),
[raw/clamped replica comparison](comparison.png), and
[settings, hashes and measurements](report.json). The preview is raw rendered
radiance displayed with the documented transform, without denoising. This check
covers the default preset with photon mapping off, rather than qualifying
photon-mapped caustics or an HDR-lit variant.
