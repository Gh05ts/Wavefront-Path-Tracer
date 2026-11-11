# Training the uniform denoiser and the joint sampler/denoiser

Run one experiment per machine/GPU. The uniform experiment trains the existing
single spatial/temporal denoiser on 4-SPP inputs. The joint experiment trains
that kind of denoiser together with a separate adaptive sampler on v5 layers.
Both start from fresh weights for the main comparison.

All commands below run from the **denoiser checkout**. `../new` is the renderer
checkout in this workspace; adjust that directory name on another machine.
Copy the current worktrees, including uncommitted/untracked source, configs,
tests and the new renderer `tools/` scripts. A clean clone of the committed
revision alone does not contain this work.

## Prepare both machines now

Create/activate a Python environment and install a CUDA-enabled PyTorch build
using the [official installer](https://pytorch.org/get-started/locally/).
Use the same Python/PyTorch versions on both machines; the current validation
used PyTorch 2.7.0. Then install the denoiser package:

```bash
python -m pip install -e '.[train,test]'
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c 'import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
```

Run the small training/resume checks without waiting for production captures:

```bash
python -m pytest -q \
  tests/test_ntasad.py::test_real_training_resume_matches_uninterrupted_run \
  tests/test_joint_training.py::test_two_epochs_equal_one_epoch_plus_resume_cpu
```

The first check uses the available GPU; the second explicitly checks CPU joint
resume. These tests create synthetic fixture data, not production-quality
training results. The focused architecture/gradient checks are also useful:

```bash
python -m pytest -q tests/test_adaptive_model.py tests/test_renderer_gradient.py tests/test_joint_warp.py
```

## Check capture progress

The reported rate is about 30 minutes per eight-frame BMW clip on a 3090.
BMW contains 36 clips, so at that rate the whole scene takes about 18 hours.
If view 0/1/2 means all six motions at each of three anchors, that is half of
BMW and about nine hours of work, consistent with the reported elapsed time.

There are 234 clips total. Multiplying all of them by the BMW rate gives
117 hours (about five days), **only if every scene costs the same**. This is not
a measured completion estimate. Other scenes may be much faster or slower.
Each full-resolution 4+1024-SPP frame requests 2,131,660,800 camera paths.

The read-only checker counts published frames using JSON/image presence,
without taking the capture writer's lock or reading full EXR pixels:

```bash
python ../new/tools/ntasad_capture_status.py ../dataset/denoiser-v4 > capture-before.json
# Run again after an observation interval, e.g. after another clip finishes.
python ../new/tools/ntasad_capture_status.py ../dataset/denoiser-v4 \
  --previous capture-before.json > capture-after.json
```

The report lists complete clips and per-scene counts. It does not qualify
reference convergence or replace the trainers' checksum checks. Use actual
completed-frame progress and GPU utilization to distinguish slowness from a
stall. Keep the active renderer binary, capture plan, sample counts and assets
unchanged. Changing any of those requires a separate dataset experiment.

## Use an idle GPU for an early validation clip

Capture one held-out validation clip in a **separate output**, rather than
waiting until the main job reaches Bathroom/Classroom. On the spare capture
machine, use the same locked v4 plan, exact capture executable bytes and assets
at matching paths. Do not rebuild or replace the running job's executable.

```bash
python tools/generate_dataset.py --stage pilot --scenes bathroom \
  --output ../dataset/denoiser-v4-validation-pilot \
  --renderer /path/to/the/locked/dataset_capture --renderer-root ../new
```

This requests one full eight-frame, 4+1024-SPP Bathroom clip. The 30-minute BMW
timing is not a guarantee for Bathroom. It is useful for model validation and
not used for training. Two capture writers must use different output roots.

Bring the completed BMW and Bathroom clips plus their capture metadata to the
pilot machine. Keep the original source trees intact. The following helper
copies complete clips, checks their EXR hashes and cameras, and refuses sources
whose capture locks differ. It writes only the new pilot snapshot:

```bash
python ../new/tools/snapshot_ntasad_clips.py \
  --source ../dataset/denoiser-v4 \
  --source ../dataset/denoiser-v4-validation-pilot \
  --group bmw_view_00_linear_slow \
  --group bathroom_view_00_linear_slow \
  --output ../dataset/denoiser-v4-pilot-frozen
```

Use group names actually listed as complete by the progress checker. Identical
capture locks require the same binary digest, asset paths/bytes and settings.
Do not edit a lock to make incompatible captures pass. The snapshot retains the
full original partition declaration, even though only selected clips exist.
Never add new data to a pilot snapshot while training/resuming from it.

## Create v5 companion layers

Uniform training only needs v4. Joint training additionally needs the independent
1/2/8/16/32-SPP layers, reusing the existing 4-SPP input and 1024-SPP reference.
This adds 59 paths/pixel, about 5.7% of v4's path count; initialization, I/O and
scene costs mean it is not a wall-clock prediction. References are not rendered
again. The separate layer capture executable must include the current capture
work; see [the v5 build/reuse instructions](../../denoiser/dataset_generation/V5.md).

For the frozen two-clip pilot:

```bash
python tools/generate_adaptive_dataset.py \
  --source ../dataset/denoiser-v4-pilot-frozen --output ../dataset/denoiser-v5-pilot \
  --stage capture --renderer ../new/build/ntasad-v5/dataset_capture --renderer-root ../new \
  --groups bmw_view_00_linear_slow bathroom_view_00_linear_slow
```

Use a distinct v5 output for the full dataset. V5 capture still requires the
source plan's scene/asset paths to resolve; training a completed frozen v5 copy
does not require loading renderer scene assets. Do not train in a v5 directory
while its capture writer is active.

## Prepare and run a representative pilot

Generate aligned settings for both models. This only prepares configurations;
the trainers perform data validation when launched:

```bash
python ../new/tools/prepare_ntasad_training.py \
  --v4 ../dataset/denoiser-v4-pilot-frozen --v5 ../dataset/denoiser-v5-pilot \
  --train-group bmw_view_00_linear_slow --validation-group bathroom_view_00_linear_slow \
  --output configs/pilot-10 --epochs 10
```

On the uniform machine, omit `--v5` if companion metadata/layers are not present;
the helper then creates only `uniform.json`. Uniform training can begin as soon
as its frozen v4 train/validation clips are ready, while the other machine
captures v5 layers.

Keep widths 48/48, five frames, 128-pixel training crops, batch 4 and Adam LR
0.001. Defaults use four patches per window and five deterministic 256-pixel
validation crops. A one-train-clip pilot has four windows, 16 crop examples and
four updates/epoch at batch 4. Ten epochs checks the pipeline; extend the same
subset separately for a meaningful small-set fitting check. Loss need not reach
zero for this constrained kernel-prediction architecture.

On machine A (uniform denoiser):

```bash
python -m wavefront_denoiser train configs/pilot-10/uniform.json runs/uniform-pilot-10 --device cuda
```

On machine B (joint denoiser + sampler):

```bash
python -m wavefront_denoiser train-adaptive configs/pilot-10/joint.json runs/joint-pilot-10 --device cuda
```

Prepare configs on each machine after placing its dataset copies; resolved paths
are part of resume provenance. If memory is insufficient, use a smaller batch
or validation crop before starting a new run and record those differences.
Use local SSD/NVMe for training snapshots when possible. The current loaders
decode/hash full frames for crops, so storage throughput can limit GPU use.

Inspect `history.jsonl`: finite losses, sustained learning over a fitting run,
and finite nonzero sampler/denoiser gradient norms in the joint run. Expected
SPP should be approximately 4; actual rounded SPP and total paths are separately
reported. Inspect zero/saturated/unavailable fractions rather than treating
their presence alone as failure. Time complete epochs including validation and
checkpoint writes, and observe GPU utilization before projecting a long run.

The pilot's group selection is a separate diagnostic experiment. Do not resume
its checkpoint onto a larger training inventory. No main training should be
launched for 1000 epochs before real-data fitting, timing and memory checks.

## Train on the complete dataset

After v4 train/validation groups are complete, prepare/capture full v5 layers:

```bash
python tools/generate_adaptive_dataset.py --source ../dataset/denoiser-v4 \
  --output ../dataset/denoiser-v5 --stage capture \
  --renderer ../new/build/ntasad-v5/dataset_capture --renderer-root ../new
```

By default this captures train/validation groups, leaving test groups reserved.
Then prepare full configurations, omitting the pilot group restrictions:

```bash
python ../new/tools/prepare_ntasad_training.py \
  --v4 ../dataset/denoiser-v4 --v5 ../dataset/denoiser-v5 \
  --output configs/main-10 --epochs 10
```

Machine A can prepare with `--v4` alone and start uniform training while machine
B captures the companion layers. It need not wait for v5 completion.

Launch on the respective machines:

```bash
# Machine A:
python -m wavefront_denoiser train configs/main-10/uniform.json runs/uniform-main-10 --device cuda
# Machine B:
python -m wavefront_denoiser train-adaptive configs/main-10/joint.json runs/joint-main-10 --device cuda
```

These are fresh experiments on the full, frozen inventory. For a fair comparison,
match the underlying v4 references, guides, train/validation scene split, widths,
loss, initialization policy and training budget. Record wall time and optimizer
updates as well as epochs. At the full plan defaults there are 720 updates/epoch.

`model.pt` is the best validation checkpoint within that invocation; `last.pt`
is the latest completed epoch for resume. After the short full run, change only
`epochs` in the same config to a larger total, then use a new run directory:

```bash
python -m wavefront_denoiser train configs/main-10/uniform.json runs/uniform-main-continued \
  --device cuda --resume runs/uniform-main-10/last.pt
python -m wavefront_denoiser train-adaptive configs/main-10/joint.json runs/joint-main-continued \
  --device cuda --resume runs/joint-main-10/last.pt
```

Only run the command for that machine. The filename `main-10` can stay the same;
its configured total epoch target must be increased. Do not change crops,
batch, paths, selection, policies or inventory during ordinary resume. New
joint experiments use `reconstruction_version=pixel_mean_v2`; old v1 pilots
explicitly retain `sample_mass_v1`. Revisit measured throughput before committing
to a long convergence run. Final CUDA/runtime integration is still deferred.

## Validate and test

The trainers already evaluate validation crops at initialization and each epoch.
For a full uniform run, evaluate its selected checkpoint separately:

```bash
python -m wavefront_denoiser evaluate runs/uniform-main-10/model.pt \
  --split-manifest ../dataset/denoiser-v4/split.json --partition validation --device cuda \
  > uniform-validation.json
```

The helper below re-evaluates a joint checkpoint on its exact frozen validation
crops, checking the saved inventory and excluding test pixels:

```bash
python ../new/tools/evaluate_ntasad_joint.py runs/joint-main-10/model.pt --device cuda \
  > joint-validation.json
```

For a pilot, substitute the pilot checkpoint path in the helper. The uniform
pilot uses explicit subset groups and therefore needs group evaluation instead
of the strict whole-dataset split command:

```bash
python -m wavefront_denoiser evaluate runs/uniform-pilot-10/model.pt \
  ../dataset/denoiser-v4-pilot-frozen --sequence-id bathroom_view_00_linear_slow --device cuda \
  > uniform-pilot-validation.json
```

Uniform streaming previews are available now, using the same selected groups:

```bash
python -m wavefront_denoiser denoise-sequence runs/uniform-main-10/model.pt \
  ../dataset/denoiser-v4 runs/uniform-validation-preview \
  --sequence-id bathroom_view_00_linear_slow --device cuda
```

After checkpoint/hyperparameter selection is frozen, the existing uniform final
test command uses the reserved split, when its captures are complete:

```bash
python -m wavefront_denoiser evaluate runs/uniform-main-10/model.pt \
  --split-manifest ../dataset/denoiser-v4/split.json --partition test --device cuda \
  > uniform-test.json
```

The joint helper reports crop L1 and sample/path accounting. It is **not** a
full-sequence joint PSNR/tPSNR/test evaluator. Such an evaluator, allocation-map
export and matched-budget fixed-denoiser sampler comparisons remain required
before final joint results. The uniform `evaluate`/`denoise-sequence` commands
cannot load joint checkpoints. Do not substitute a joint checkpoint into them.

## What the paper comparison can establish

The paper reports PSNR and tPSNR on tonemapped images, and relative MSE on HDR
images. tPSNR is PSNR of temporal frame differences. Training spatial/temporal
L1 is a different measurement. See [NTASAD section 3](https://research.nvidia.com/sites/default/files/pubs/2020-05_Neural-Temporal-Adaptive/NTASAD_preprint.pdf).

The current uniform report includes display PSNR, HDR relative MSE and
transformed-domain temporal L1. Display uses a fixed Reinhard + sRGB transform.
tPSNR reporting and full-sequence joint reporting still need implementation.
Exact paper tonemapper equivalence has not been established. We can implement
the same metric categories with a documented common transform; matching the
paper's numerical scores is not guaranteed with different scenes, motion,
renderer, radiance clamp, references and training diversity. The present dataset
uses moving cameras in static scenes; this differs from the paper's animated
training scenes. Matching eight-frame clips alone does not reproduce that data.

Compare both models on the same held-out frames and actual path budgets, keep
initialization handling identical, and report scene/clip results as well as
averages. Longer held-out trajectories are necessary to assess ghosting and
temporal stability beyond eight frames. The current 1024-SPP references remain
fixed-budget, not convergence-qualified. The paper used approximately 1024-SPP
training and 4096-SPP evaluation references; higher-budget evaluation should
remain independent of checkpoint selection.

## Local readiness check performed

Both current full-width trainers completed a short one-epoch run (16 optimizer
updates each) on the existing captured 17×13 Demo/Prism fixture, followed by
their validation paths. The joint checkpoint uses `pixel_mean_v2` and both
networks have nonzero recorded gradient norms. The setup/progress/snapshot
helpers were exercised on that fixture. Results are in
`build/ntasad-v5/training-readiness/report.json`. This verifies the commands and
interfaces; it is not production training, an overfit qualification, a training
ETA or a denoising-quality comparison.
