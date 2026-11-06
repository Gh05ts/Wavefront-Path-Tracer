# Renderer and denoiser runbook

Workspace layout:

```text
/home/myid/lv37348/pt/new       CUDA renderer and dataset_capture
/home/myid/lv37348/pt/denoiser  Python datasets, spatial models and experiments
```

The current model is a spatial residual U-Net, not an NTASAD implementation.
All completed experiments are saved; no new training job is launched by this
document. Run these commands from an ordinary terminal with GPU access.

## Status against the original six points

| Point | Completed | Remaining |
| --- | --- | --- |
| 1. Reference convergence | Independent 512/2,048/8,192-SPP replicas on demo/prism/deer; v2 selected 8,192 SPP. Lighting-specific calibration extends to 65,536 SPP for side light plus sky. | Photon-count/gather-radius convergence, photon-map bias, broader cameras/materials and Sponza-specific convergence. Current ML studies have caustics disabled. |
| 2. Broader dataset | Demo/prism training includes diffuse, glossy/metal and glass surfaces. Deer validation; textured Sponza evaluation. Added four controlled lighting conditions and matched-budget training. | Many more training scenes, textured training scenes, caustic training data, 1/2/8/16-SPP coverage and production resolutions. Current trained data is 4 SPP at 128x96. |
| 3. Independent test set | Whole-scene/family split, identity/hash audits, frozen checkpoint provenance, first reserved Sponza evaluation. | Additional untouched test scenes. Sponza is now an observed development benchmark. The lighting study shares geometry across camera partitions and is explicitly a within-scene ablation. |
| 4. Quality and cost | Input/model/reference comparisons, HDR/relative/log/display metrics, edge/high-gradient/bright-region diagnostics, three paired lighting-study seeds. Warm v2 inference measured about 2.4 ms at 128x96; about 9.31 MiB peak PyTorch allocated memory. | Established-denoiser comparison, textured/caustic/high-resolution quality and cost. Existing memory timing is not full process/driver memory, and timing excludes model loading and file I/O. |
| 5. Ordinary render integration | Input-only EXR loading/inference works; paired capture exports radiance/guides. | Input-only renderer export, automatic local final-frame denoising, distributed assembly hook. Current capture still renders a reference; ordinary rendering still writes its original output. Raw render/checkpoint behavior is unchanged. |
| 6. NTASAD | Guide/motion capture and sequence loading provide preparation. | Spatial hierarchical kernel prediction, temporal recurrence and adaptive sampling. The current U-Net does not implement these. |

The next practical order is broader data and caustic-reference qualification,
an established-denoiser/full-resolution comparison, ordinary-render integration,
then NTASAD spatial kernel prediction before temporal/adaptive extensions.

## 1. Build the renderer

The machine already has CUDA toolkit 12.8, Python 3.10 and CMake 3.22. The RTX
3090 uses CUDA architecture 86. The existing build cache was using architecture
52; explicitly configuring 86 below will rebuild the affected targets.

On an Ubuntu/Debian machine, install missing host dependencies if needed:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake pkg-config libssl-dev libopenexr-dev python3-venv
```

This host-dependency command does not install the NVIDIA driver/CUDA toolkit.
Check the existing GPU/toolkit and build:

```bash
nvidia-smi
nvcc --version
cd /home/myid/lv37348/pt/new
cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DPATHTRACER_BUILD_DATASET_CAPTURE=ON \
  -DBUILD_TESTING=ON
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

For ordinary rendering, use `cd build` then `./pathtracer --scene demo`.
That command does not invoke the denoiser. The capture commands below use the
renderer root explicitly so preset/asset lookup is consistent.

## 2. Create a persistent denoiser environment

The completed experiments used a temporary environment. A persistent local
`.venv` is the preferred entry point for subsequent work. Python 3.10+ is
required. Torch 2.7.0 / CUDA 12.8 matches the tested model stack; OpenEXR 3.5.2
was used for Python reading/writing. The renderer separately uses system C++
OpenEXR libraries.

```bash
cd /home/myid/lv37348/pt/denoiser
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install OpenEXR==3.5.2
python -m pip install -e '.[train,test,analysis]'
python -m pip check
python - <<'PY'
import torch, OpenEXR
print('Torch:', torch.__version__)
print('Torch CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
PY
```

The versioned wheel command is supported by the official PyTorch instructions:
https://pytorch.org/get-started/previous-versions/#v270
Torch's wheel supplies its own CUDA runtime dependencies; building the renderer
still requires the separately installed CUDA compiler/toolkit. The remaining
project dependencies use the supported ranges in pyproject.toml, so this is
not a byte-for-byte lockfile for every dependency.

In each new shell:

```bash
cd /home/myid/lv37348/pt/denoiser
source .venv/bin/activate
```

An editable install makes the package available without a temporary PYTHONPATH.

## 3. Use the saved models immediately

The original whole-scene-split baseline is:

```text
runs/spatial-independent-v2/model.pt
```

The lighting-study diverse checkpoint for the predeclared example seed is:

```text
runs/lighting-experiment-v1/models/diverse_seed7/model.pt
```

The latter is a representative seed, not a model selected by test/probe score.
It does not replace the original baseline. Output directories must be new:

```bash
cd /home/myid/lv37348/pt/denoiser
source .venv/bin/activate
python -m wavefront_denoiser denoise \
  runs/lighting-experiment-v1/models/diverse_seed7/model.pt \
  data/spatial-independent-v2/v2_sponza_test/frame_000000 \
  runs/manual-sponza-output --device cuda
```

The output contains `denoised.exr` (float32 linear HDR RGB), `comparison.png`
(input | denoised | reference when present), and `report.json` (hashes/metrics).
The reference is used for the comparison, never as a model input. Inference
also accepts an input-only EXR with the documented noisy radiance/guide channels,
although the capture executable does not yet generate that input-only export.
Use `--device cpu` for CPU inference; it does not remove the renderer's GPU need.

## 4. Capture a new paired input/reference example

This uses the existing demo capture setup, which has four 4-SPP input views
with 8,192-SPP references:

```bash
cd /home/myid/lv37348/pt/new
build/dataset_capture \
  ../denoiser/configs/independent_demo_train_capture_v2.json \
  ../denoiser/data/manual-demo-capture
cd ../denoiser
source .venv/bin/activate
python -m wavefront_denoiser validate data/manual-demo-capture
python -m wavefront_denoiser denoise \
  runs/spatial-independent-v2/model.pt \
  data/manual-demo-capture/frame_000000 \
  runs/manual-demo-output --device cuda
```

The standalone example is not automatically added to a frozen training split.
Edit a copy of a capture JSON to change `samples`, `reference_samples`, image
and tile sizes, scene/preset or camera frames. Use new capture group IDs and
nonoverlapping sample intervals for independent data. A changed scene/view/
resolution/light/caustic configuration needs its own reference qualification.
For camera motion use multiple `frames`; the current denoiser remains spatial.

## 5. Train again on the existing independent dataset

```bash
cd /home/myid/lv37348/pt/denoiser
source .venv/bin/activate
python -m wavefront_denoiser split-audit configs/dataset_split_v2.json
python -m wavefront_denoiser train \
  configs/spatial_independent_train_v2.json \
  runs/spatial-independent-new --device cuda
python -m wavefront_denoiser evaluate \
  runs/spatial-independent-new/model.pt \
  --split-manifest configs/dataset_split_v2.json \
  --partition validation --device cuda > runs/spatial-independent-new/validation.json
python -m wavefront_denoiser denoise \
  runs/spatial-independent-new/model.pt \
  data/spatial-independent-v2/v2_deer_validation/frame_000000 \
  runs/spatial-independent-new/deer-output --device cuda
```

Training emits `model.pt`, expanded `config.json`, `split.json`, split audit,
`history.jsonl` and `summary.json`. It selects a validation checkpoint and saves
weights/settings, not optimizer state for resuming training. Epoch zero means
no trained candidate passed the selection rule and the identity model won.

After freezing settings/checkpoint, explicit Sponza evaluation is:

```bash
python -m wavefront_denoiser evaluate \
  runs/spatial-independent-new/model.pt \
  --split-manifest configs/dataset_split_v2.json \
  --partition test --device cuda > runs/spatial-independent-new/sponza-benchmark.json
```

For this workspace, that is an observed benchmark. Re-rendering the same Sponza
scene/cameras does not make it a fresh independent test.

## 6. Regenerate the entire independent dataset in a new directory

`capture-split` obtains the destination from the split manifest; it has no
output-directory positional argument. The frozen default destination already
exists. The following makes a separate config set and a new data destination
without editing the existing split or artifacts. Use another suffix if any of
these output names already exist.

Run from the denoiser directory with its .venv active:

```bash
python - <<'PY'
import json
from pathlib import Path
import shutil
source = Path('configs')
target = source / 'recapture'
target.mkdir(exist_ok=False)
for path in source.glob('independent_*_capture_v2.json'):
    shutil.copy2(path, target / path.name)
split = json.loads((source / 'dataset_split_v2.json').read_text())
split['data_root'] = '../../data/spatial-independent-recapture'
(target / 'dataset_split_v2.json').write_text(json.dumps(split, indent=2) + '\n')
for name in ['reference_convergence_v2.json', 'spatial_independent_train_v2.json']:
    shutil.copy2(source / name, target / name)
PY
python -m wavefront_denoiser reference-convergence \
  configs/recapture/reference_convergence_v2.json runs/reference-recapture \
  --capture-executable ../new/build/dataset_capture --plot
python -m wavefront_denoiser capture-split \
  configs/recapture/dataset_split_v2.json runs/reference-recapture/report.json \
  --capture-executable ../new/build/dataset_capture
python -m wavefront_denoiser split-audit configs/recapture/dataset_split_v2.json
python -m wavefront_denoiser train \
  configs/recapture/spatial_independent_train_v2.json \
  runs/spatial-recapture --device cuda
```

This reproduces the same scene/camera/sample assignments, rather than adding
new training diversity or a new test scene. The new split hash is linked to its
new convergence evidence. If no reference count qualifies, capture intentionally
stops; inspect the report and design a new experiment with sufficient budgets.

## 7. Re-run the controlled lighting experiment

All-in-one: calibrate references, generate inputs/references, train six models,
freeze them and report probe/observed-Sponza metrics:

```bash
python -m wavefront_denoiser lighting-study configs/lighting_experiment_v1.json \
  runs/lighting-experiment-new \
  --capture-executable ../new/build/dataset_capture --device cuda
```

The checked-in protocol reuses the validated pilot calibration replicas in
`runs/lighting-reference-pilot-v1`. To render every calibration replica again,
make a new protocol with reuse disabled:

```bash
python - <<'PY'
import json
from pathlib import Path
source = Path('configs/lighting_experiment_v1.json')
value = json.loads(source.read_text())
value['reuse_reference_captures'] = None
destination = Path('configs/lighting_experiment_fresh.json')
with destination.open('x') as handle:
    json.dump(value, handle, indent=2)
    handle.write('\n')
PY
python -m wavefront_denoiser lighting-study configs/lighting_experiment_fresh.json \
  runs/lighting-experiment-fresh \
  --capture-executable ../new/build/dataset_capture --device cuda
```

For separate capture/training phases, use the same protocol/output paths:

```bash
python -m wavefront_denoiser lighting-study configs/lighting_experiment_v1.json \
  runs/lighting-experiment-phased --capture-executable ../new/build/dataset_capture \
  --phase capture
python -m wavefront_denoiser lighting-study configs/lighting_experiment_v1.json \
  runs/lighting-experiment-phased --phase train --device cuda
```

The core command saves `report.json`, `freeze.json`, `dataset.json`, capture
configs/logs, reference reports/CSV/replicas, models and per-condition prediction
EXRs/previews. The combined comparison grid and bar chart in the saved v1 run
were generated by its archived analysis script; those two composite figures
are not automatically generated by the CLI.

Saved result entry points:

```text
runs/spatial-independent-v2/summary.json
runs/spatial-independent-v2/benchmark.json
runs/spatial-independent-v2/comparison_grid.png
runs/lighting-experiment-v1/report.json
runs/lighting-experiment-v1/summary.csv
runs/lighting-experiment-v1/comparison_grid.png
```

## 8. Tests

```bash
cd /home/myid/lv37348/pt/new
ctest --test-dir build --output-on-failure
cd ../denoiser
source .venv/bin/activate
python -m pytest -q
WAVEFRONT_CAPTURE_EXECUTABLE=/home/myid/lv37348/pt/new/build/dataset_capture \
  python -m pytest -q tests/test_capture_gpu.py
```

All generated output directories must be new. Existing trained models, frozen
splits and completed experiments should be retained; create new configs/runs
for new data or settings. Do not change frozen manifests to bypass auditing.
