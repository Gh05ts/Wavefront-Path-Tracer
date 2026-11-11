# NTASAD continuation handoff — 2026-11-11

Training instructions now live in [NTASAD_TRAINING.md](NTASAD_TRAINING.md), with
portable renderer `tools/` helpers for progress, verified pilot snapshots,
uniform/joint config preparation and joint validation crops. Both current
full-width trainers completed 16-update technical runs on the captured tiny
fixture; artifacts are in `build/ntasad-v5/training-readiness/`. No production
training was started. Full-sequence joint evaluation and tPSNR remain pending.
The user reports an active remote 3090 capture taking about 30 minutes per
eight-frame BMW clip. BMW has 36 clips (~18 h at that rate); extrapolating that
rate to every scene gives ~117 h, but scene costs vary and this is not an ETA.

## Latest follow-up: fidelity audit and pyramid correction

The requested audit/correction is now documented in
[NTASAD_FIDELITY_AUDIT.md](NTASAD_FIDELITY_AUDIT.md) and the sibling's detailed
[paper/supplement comparison](../../denoiser/docs/ntasad-fidelity-audit.md).
Read these before treating the historical status below as current.

The adaptive default uses `pixel_mean_v2`, removing sample-count weighting
from the image pyramid while retaining explicit available-pixel normalization.
Joint checkpoints now use envelope v2 with explicit reconstruction semantics;
original v1 checkpoints still load with their original `sample_mass_v1` policy.
Old training resume requires that policy explicitly. The uniform model and
capture/renderer source were preserved. SMALL remains unimplemented; exact
author-code agreement is not claimed for underspecified reconstruction details.

Current evidence: 266 model/data/checkpoint tests passed on the GPU-enabled
selection, plus full-width five-frame RTX 3090 gradient/checkpoint smoke checks.
A historical pilot checkpoint reproduced its CPU outputs bitwise. The attempted
full run stalled in the first existing renderer capture test; its process was
stopped, and the two capture modules (18 tests) were excluded from the completed
run. The historical 278-test result below is not a fresh full-suite result.
See `build/ntasad-v5/fidelity-audit/` for evidence and the guarded sibling patch.

Performance changes, long production training, live scheduling and CUDA runtime
integration were not introduced by this follow-up. Their ordering stays as
requested below.

## User intent and boundaries

Recreate NVIDIA NTASAD sampling/denoising in the sibling Python denoiser,
preserving the synced v4 camera changes and existing capture work. Improve and
verify the Python implementation and training first. **Live adaptive rendering,
per-pixel scheduling changes to the renderer and CUDA runtime inference are the
absolute last phase.** The user wants sampler effectiveness measured separately.
Training may take one or two days, so throughput matters; exact bitwise resume
must be weighed against its performance cost rather than assumed mandatory.
The latest user instruction makes **denoiser fidelity to the NTASAD paper the
first priority**. Defer grid_sample/fast-mode work and other throughput changes
until after the fidelity work, then revisit them before long production training.

The user will resume in a new window after the performance discussion. This
turn added benchmark artifacts and this handoff; it did not change production
Python or renderer source. The fast training mode discussed below is deferred
work, not the next task and not something already available.

## Repositories and environment

* Renderer workspace: `/home/myid/lv37348/pt/new`.
* Python project: `/home/myid/lv37348/pt/denoiser` (sibling, outside writable sandbox).
* Expected production source: `/home/myid/lv37348/pt/dataset/denoiser-v4`.
  It was not available locally; user explicitly deferred production validation.
* Companion target: `../dataset/denoiser-v5`.
* Python: `/tmp/ntasad-audit/venv/bin/python`, PyTorch 2.7.0, OpenEXR available.
  Use `PYTHONPATH=../denoiser/src` from renderer workspace.
* GPU works outside sandbox with required escalation. One RTX 3090 was visible;
  user says the machine physically has two. Multi-GPU training is not implemented.
* CUDA reproducibility environment: `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
* Both repos have intentional uncommitted work. Preserve all existing changes,
  especially v4 cameras, uniform model/trainer/checkpoint/runtime and the dirty
  `extern/NexusBVH` submodule. No commits were made to the actual repos.
* For edits to sibling, stage/test under `/tmp`, then apply a reviewed patch
  with filesystem escalation. `/tmp/ntasad-joint-work` is the previous tested
  snapshot, with a temporary baseline commit; it is not an authoritative repo.
* Do not spawn subagents unless explicitly requested by the user or instructions.

## Implemented and last validated

The integrated Python suite passed **278 tests, zero skips**, including GPU
capture and the C++ EXR fixture. The capture exporter C++ test also passed.

1. `adaptive.py`: sampler U-Net, eight guide/history features, grayscale colors,
   softmax allocation, independent rounding. No current noisy RGB or reference
   inputs. Default widths follow the reconstructed LARGE topology.
2. `sample_layers.py`: binary replay of independent 1/2/4/8/16/32-SPP RGB means,
   integer counts 0..63, explicit zero samples. Six layers total 63, not 64.
3. `adaptive_dataset.py`, `adaptive_data.py`, `tools/generate_adaptive_dataset.py`:
   locked v5 companion planning/capture/read/resume, inheriting v4's exact
   234 clips / 1872 frames and cameras. Reuse verified completed source 4-SPP
   inputs, 1024-SPP references and guides; capture only new 1/2/8/16/32 layers
   in explicitly disjoint RNG ranges. Existing v4 generation remains intact.
4. `adaptive_model.py`: separate count-aware sparse wrapper around unified
   denoiser U-Net, weighted transformed-space pyramid, masked spatial kernels,
   normalized available-history warp, fallbacks and explicit output availability.
   Original uniform architecture/checkpoint APIs are preserved.
5. `renderer_gradient.py`: exact integer forward replay, linear-domain DASR
   equation (8) count-gradient surrogate. Reference affects backward only.
   Explicit zero extensions `unit_denominator` (default) or `zero`.
6. `allocation.py`: `paper_softmax_v1` rejects capacity overflow;
   `capped_softmax_v1` (default) redistributes continuous budget with an active
   set. The cap is an explicitly documented paper variant. No minimum count or
   exact integer-budget repair; actual path totals are logged.
7. `joint.py`, `joint_data.py`, `joint_training.py`, `joint_checkpoint.py`:
   five-frame joint recurrence, aligned crops, two-network Adam, strict selected
   train/validation inventory, shared capture read lock, test-pixel isolation,
   distinct atomic two-network optimizer/RNG checkpoints and epoch resume.
   `train-adaptive CONFIG OUTPUT --device cuda --resume CHECKPOINT` is available.
   Currently `num_workers=0`, strict deterministic execution and gather warping
   are mandatory in joint training. Saved config/inventory must agree on resume;
   only total target epochs may change. Resume uses a new output directory.
8. Planner adds runnable `train-joint.json`, preserving historical
   `train-joint-plan.json` and edited configs. Template:
   `../denoiser/configs/train-adaptive.json`. Production plan was refreshed at
   `build/ntasad-v5/production-plan` without copying/rendering frames.

Actual-source files match the tested staging version. Last task patches are
under `.ntasad-backup/joint-training*.patch`. Final source checksums and protected
uniform files are recorded in `build/ntasad-v5/joint-validation.json`.

## Evidence and limitations

Real GPU pilot: two eight-frame 17x13 Demo/Prism clips, 16-SPP references,
reduced-width networks, eight optimizer updates over two epochs. Both networks
received gradients; train/validation losses decreased. Two uninterrupted epochs
and one epoch plus resume produced **bitwise-identical parameters**, max delta 0.
CPU resume also has an exact integration test.

Pilot report: `build/ntasad-v5/joint-pilot-v3/report.json`.
Pilot data: `build/ntasad-v5/joint-pilot-v2/v5`.
Older pilot attempts are archived; do not use their checkpoints as current-format
evidence. Tiny-fixture success does not establish trained quality or sampler gains.

Primary implementation docs in sibling:

* `docs/joint-training.md`
* `docs/adaptive-reconstruction.md`
* `docs/ntasad-adaptive-plan.md`
* `docs/ntasad-progress.md` (historical sections contain old status; use latest follow-ups)
* `dataset_generation/V5.md`

Paper copies: `/tmp/ntasad-audit/paper.pdf`, `paper.txt`, `supplement.pdf`,
`supplement.txt`; DASR equation source `/tmp/ntasad-gradient/dasr.pdf`, `dasr.txt`.
The core follows the publication, but exact author-code equivalence is not proven.
Pooling, padding, domain boundaries, sparse fallbacks, zero derivatives and
parameter-count agreement remain documented reconstruction choices/audit items.
Paper SMALL architectures are not implemented; narrower test widths are not SMALL.

## Proposed next work, before any production training

1. FIRST: audit the denoiser against the NTASAD paper and supplement, figure by
   figure and equation by equation. Check topology/layer counts, features,
   radiance-domain boundaries, multiscale reconstruction, recurrence,
   initialization and loss. Also check sampler/replay integration and the
   gradient estimator. Separate specified requirements from underspecified
   details and explicit engineering variants; establish what must be corrected.
2. Implement the required fidelity corrections with numerical tests and a
   representative GPU check. Preserve existing checkpoints through explicit
   versioning where semantics change. Do not claim exact author-code agreement
   for details the sources leave unspecified. Defer performance refactoring.
3. Add offline streaming inference/evaluation for joint checkpoints and separate
   matched-budget uniform/random/learned sampler comparisons with fixed denoiser
   weights, plus jointly trained comparisons, allocation maps, path accounting,
   PSNR/tPSNR and long-trajectory temporal reports. Existing uniform
   `denoise-sequence`/evaluation CLI cannot load joint checkpoints, although
   `JointNTASAD.eval().forward_sequence` already performs target-free inference.
4. AFTER fidelity is addressed, BEFORE long training: revisit the performance
   findings below. Consider explicit strict/fast modes, recording warp backend
   and execution policy in checkpoints; ordinary resume must remain supported.
   Profile the complete loader/trainer, hashes, finite checks, validation and
   GPU utilization. Optimize measured bottlenecks without weakening provenance,
   partition isolation or history gradients. No fast-mode change is requested
   as the immediate next task.
5. When v4 data arrives, validate source reuse/layers, native resolution and a
   meaningful small-set overfit, then train to convergence and assess quality.
6. LAST: live renderer adaptive scheduling and CUDA runtime model inference.

No production training, performance-mode implementation, source optimizations or
renderer/runtime integration were performed during this discussion.

## Deferred performance work — revisit after paper fidelity, before long training

Both warp backends do bilinear interpolation. `grid_sample` uses a specialized
operator. Our `reproject_history_gather` is a correctness reference expressed as
four gathers plus many tensor operations, integer indices and intermediates.
`reproject_available_history` repeats geometry work for RGB and coverage; the
sampler and denoiser also independently warp history. Validation booleans cause
GPU/CPU synchronizations. These offer optimization opportunities.

More importantly, gather backward scatters contributions into source pixels.
PyTorch 2.7 strict deterministic scatter accumulation routes through index-put
with sorting instead of the faster atomic-add path. Thus the earlier benchmark
compared BOTH different implementations AND different determinism policies; it
must not be interpreted as an intrinsic cost of bilinear interpolation.

Primary source:
https://raw.githubusercontent.com/pytorch/pytorch/v2.7.0/aten/src/ATen/native/TensorAdvancedIndexing.cpp
(`gather_backward`, `scatter_add`, deterministic `index_put_with_sort_stub`).
https://docs.pytorch.org/docs/2.7/generated/torch.use_deterministic_algorithms.html
https://docs.pytorch.org/docs/2.7/generated/torch.nn.functional.grid_sample.html

Isolated normalized RGB warp benchmark (RTX 3090, FP32):

| Operation | grid_sample, ordinary backward | Gather, strict deterministic |
| --- | --- | --- |
| Inference, one 1080p image | 2.36 ms | 5.39 ms |
| Forward/backward, batch4 crop128 | 2.86 ms | 10.35 ms |
| Forward/backward, one 1080p image | 2.75 ms | 33.48 ms |

Artifacts: `build/ntasad-v5/benchmark_warps.py`, `warp-benchmark.json`.

A more relevant **two-network training computation** benchmark used default
48/48 widths, batch4, five frames, 128x128 crops, FP32, forward, replay, loss,
backward and Adam. Same initial weights/input; cuDNN deterministic=true and
benchmark=false for all cases. Warp dispatch was patched only in that process;
production source and checkpoint policies were not changed.

| Backend and global deterministic policy | Median computation time |
| --- | --- |
| Gather, strict deterministic | 154.68 ms |
| Gather, ordinary execution | 121.59 ms |
| grid_sample, ordinary execution | 115.28 ms |

12 measured steps after four warmups each, one visible RTX 3090. About 25% less
compute time with grid_sample versus current strict mode; most measured overhead
was associated with strict determinism. Not a full production run estimate:
EXR decoding, augmentation, transfers, validation, inventory, logging and the
trainer's per-parameter finite scans are excluded. Current loader decodes and
rehashes full frames for every requested crop, sequentially; I/O may dominate.
Do not extrapolate a precise one/two-day ETA from synthetic resident batches.

Artifacts: `build/ntasad-v5/benchmark_joint_training.py`,
`build/ntasad-v5/joint-training-benchmark.json`.

### Follow-up: absolute deterministic overhead estimate

The user says deterministic mode is acceptable if its total overhead is only
one or two hours; retain this preference when deciding whether fast mode is
worth implementing. Production-plan defaults contain 180 train clips, 720
five-frame windows, four patches per window and batch4: 720 steps/epoch.
At 1000 epochs, 720,000 updates times the measured 39.4 ms strict-vs-grid
difference gives **7.9 extra training-computation hours** (30.9 h strict versus
23.1 h fast). At 250 epochs the extra is about 2 h; at 100 epochs about 47 min.
This is a synthetic resident-batch extrapolation, excluding loading, validation
and other trainer overhead; it is not a measured production ETA. Validation can
have an additional backend difference. Convergence may require fewer epochs.
