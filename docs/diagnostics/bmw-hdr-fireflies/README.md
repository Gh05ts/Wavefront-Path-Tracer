# BMW HDR firefly diagnosis — 2026-11-09

Current renderer: **integrator 11**, with a configurable indirect contribution
clamp (default 10; zero disables). This follows the user's subsequent approval
of practical suppression after comparing Cycles' approach. Earlier sections
below describe unclamped experiments and their limits; see the final section
for the current implementation and its validation.

The 1024-SPP BMW references contain isolated bright floor pixels in the linear
EXR itself. They are not introduced by PNG tone mapping, guide export, cropping,
camera motion or the denoiser. During the initial read-only diagnosis, the executable, renderer source,
scene files, HDR map and completed frames were left unchanged. Later sections
record the applied renderer corrections.

The new HDR lighting exposes older material transport problems. These suspect
paths already exist in commit `a5f722e`, before the HDR addition `eb7e916`.
The HDR's peak red channel is 71,680; the old procedural sky peaks around one.
That large dynamic range makes rare paths much more visible.

## Confirmed problems

1. **Glass entry/exit orientation.** Both triangle-normal helpers always face
   normals toward the ray. The dielectric shader then computes `frontFace`
   using that already-flipped normal, so triangle exits are interpreted as
   entries and use `1 / ior` rather than `ior`. The photon shader also tests an
   already-faced normal. This is a transport bug, not just insufficient SPP.
2. **Rough-metal sampling limitation.** The older metal branch adds random
   fuzz to reflection, assigns PDF zero, sets `specularBounce=true`, and performs
   no next-event light sampling. Rare fuzzy directions that hit the HDR sun
   receive its full radiance without a matched finite-BRDF/MIS estimator. This
   combines an ad hoc roughness model with poor sampling of concentrated lights.
   With no competing metal light-sampling estimator, the unit miss weight alone
   is not evidence of MIS double counting; the glass orientation error above is
   the independently confirmed transport bug.

At pixel `(1790, 922)`, sample `500000870` follows floor → windscreen glass
(three boundary interactions) → HDR sun. Its red contribution is about 1207.8
before division by 1024, dominating the completed reference pixel.
At `(1839, 861)`, sample `500000865` follows floor → shiny chrome → HDR sun and
contributes about 1109.6 before averaging.

`caustics=false` disables photon tracing/gathering. It does not suppress
camera-traced paths through glass or reflected from metal, so it does not prevent
these paths or ordinary path-traced caustic variance.

## Isolated candidate and results

[`proposed-fix.patch`](proposed-fix.patch) records the original isolated candidate.
It has now been applied after the user stopped the capture. The main implementation
adds BVH2 coverage and an analytic GGX regression beyond the archived candidate.
The correction covers:

* Preserve oriented normals for dielectric triangles in direct and BVH2 traversal,
  keeping the existing facing convention for opaque surfaces.
* Use the original hit normal for photon entry/exit classification.
* Use the existing finite GGX implementation, with a black diffuse lobe, for
  nonzero-roughness metals. Sample area, punctual and environment lights and
  retain a matching PDF for MIS. Keep exact zero-roughness metal as a delta.
* Cover both ordinary and persistent path shaders and bump integrator version
  from 3 to 4 to reject incompatible render checkpoints/fingerprints.
* Add a CUDA regression target for glass boundaries, a Snell-law parallel slab,
  opaque backfaces, rough-metal evaluation/continuation and the delta limit.

Six selected pixels were replayed with existing RNG/sample IDs using the renderer
helpers compiled as CPU code. Two problematic pixels reproduce the actual GPU
references closely:

| Pixel, zero-based | Actual EXR red | Old CPU, 1024 SPP | Candidate CPU, 1024 SPP |
| --- | ---: | ---: | ---: |
| 1790, 922 (glass) | 1.209437 | 1.209182 | 0.029983 |
| 1839, 861 (chrome) | 1.111726 | 1.111555 | 0.027946 |

For the same pixels at 16,384 SPP, old CPU estimates were 0.102739 and 0.096311;
candidate estimates were 0.029162 and 0.028752. Raising SPP alone dilutes the
reproduced spikes without repairing their underlying transport.

The old code fails the glass-exit regression; the candidate passes all seven
checks on the CPU. The candidate CUDA test executable compiled and linked.
Compilation of the candidate renderer/photon CUDA translation units is checked
separately. **No candidate GPU kernel or render was executed during capture.**

CPU replay is a diagnostic, not an exact GPU renderer: it flattens texture-free
BMW instances into a CPU BVH and emulates bilinear HDR texture lookup in software.
Floating-point arithmetic, intersection order and CUDA texture interpolation
differ, so some sample paths diverge. It does not establish full-image quality,
unbiased convergence, speed, or absence of remaining real caustic fireflies.

## Reproduce without changing the capture

Use the existing development Python environment (Python 3.11+ for this helper):

```sh
cd Wavefront-Path-Tracer
/tmp/ntasad-venv/bin/python docs/diagnostics/bmw-hdr-fireflies/run_cpu_checks.py
```

This builds the candidate only in a temporary directory, checks old/new CPU
regressions and 1024/16,384-SPP selected pixels, and writes logs under
`../dataset/diagnostics/bmw-hdr-fireflies/`. It hashes the active binary, selected
assets, original headers and completed frame before/after to check preservation.
The build's existing host object files, CUDA headers and a C++17 compiler are
required. See `run_cpu_checks.py --help` to select another build/results directory.
The candidate patch applies cleanly to renderer commit `95eb205`.

## Applied fix and GPU validation

The user killed the dataset job and requested updating the original build.
Both `/tmp/ntasad-capture-build/pathtracer` and `dataset_capture` are rebuilt
with integrator version 4. The briefly configured separate build was not used.

All **18 CTest tests** passed, including the new GPU glass/metal regression,
environment/material tests, acceleration, checkpoints and distributed handling.
Two old tests needed portable asset paths for an out-of-tree build. The Python
capture/frame/split/convergence tests passed **72 tests**, with one optional
cross-language fixture skipped. Actual CUDA tests cover tiled/persistent agreement,
frame-range resume, guides and unchanged 4-SPP inputs across report budgets.

Two native 1920×1080 BMW frames completed at 4+1024 SPP in 388.7 seconds under
`../../../../dataset/denoiser-v3`. Cameras and sample IDs match v2, so the
comparison isolates the renderer correction. No clamp or reference postprocessing
was applied. In a fixed floor region, isolated luminance outliers above ten times
the local 3×3 median changed as follows:

| Native frame | Old integrator | Corrected integrator |
| --- | ---: | ---: |
| 0 | 124 | 2 |
| 1 | 112 | 0 |

The reproduced glass/chrome pixels now have red values 0.029984 and 0.027948,
matching the CPU candidate closely. All guides and radiance are finite; frame
one has 99.29% valid history. Repeating the pilot captured zero and skipped both
verified frames. This diagnostic outlier count is not a denoiser evaluation metric.
See `../../../../dataset/diagnostics/bmw-hdr-fireflies/gpu-v4/` for full PNGs,
`floor-before-after.png` and `native-1080p-results.json`. The reproducible native
comparison helper is `analyze_gpu_pilot.py`.

## Remaining reference-quality failure — bulk capture held

Independent **320×180** BMW replicas at 1024 and 4096 SPP were checked with the
existing reference-convergence command. Both failed its 0.1 noise-fraction
threshold: whole-image ratios were **0.506** and **0.174**, respectively;
brightest-5% ratios were **0.508** and **0.175**. This is a small-view test,
not a full-resolution certificate and not evidence about the other scenes.
The report is at
`../../../../dataset/diagnostics/bmw-hdr-fireflies/bmw-v4-reference-check/report.json`.

Ten pixels account for 99.93% of the 1024-SPP replica disagreement and 94.51%
at 4096 SPP, mostly on glass and strong reflected/refracted sun highlights.
A CPU replay of one remaining 4096-SPP outlier, pixel `(166,38)` in replica one,
reproduces red 16.33 versus GPU 16.34. Sample `800006808` follows windscreen
boundaries → interior leather → more glass → HDR sun, with roulette amplifying
the surviving path. Its single-sample red contribution is about 66,862. The
logged shading/geometric facing signs agree on that path, unlike the repaired
entry/exit bug. Glass blocks ordinary shadow rays, so environment NEE at the
interior diffuse surface does not importance-sample the refracted sun path.

The code fix is tested, but **1024-SPP references are not yet qualified**. Bulk
generation has not restarted. Further glass/HDR variance work or a separately
qualified higher reference budget is needed before treating these as final
training targets. Do not mix v2 into v3 or remove its provenance lock. The
correction changes both noisy images and references (and some glass guides).
Fingerprint version 4 prevents old/new checkpoint continuation; distributed
nodes need matching renderer builds. Adaptive pixel sampling remains deferred.


## Roulette isolation study

A current-header CPU replay at pixel `(166,38)` compares the existing roulette
start depth 5 with start depth 12 (disabled within the unchanged 12-bounce cap).
Two **nonoverlapping** 65,536-sample ranges start at 800002060 and 800067596.
Camera, HDR, materials and integrator helpers are unchanged. Results are under
`../../../../dataset/diagnostics/bmw-hdr-fireflies/roulette-study/`.

| Roulette start | Range 0 red mean | Range 1 red mean | Largest sample, range 0 |
| --- | ---: | ---: | ---: |
| 5 | 1.026532 | 0.006718 | 66,861.97 |
| 12 | 0.169485 | 0.006548 | 10,671.17 |

The same sample 800006808 reaches the sun in both variants. Without roulette,
throughput remains 0.168 after leather; with roulette it becomes 1 at depth 5
and 1.05263 at depth 6. Roulette therefore amplifies this contribution by about
6.27×. Even without it, one sample contributes about 96% of the first range's
red total. The second range has no red contribution above 100 in either variant.
These selected-pixel results do not estimate a converged mean or qualify any
reference budget. Reported empirical standard errors are unreliable for this
rare-event tail. Different roulette schedules also consume RNG differently after
the first roulette decision, so this is not a generally paired-path comparison.

This isolates two variance sources: roulette amplification and rare refracted
sun discovery. Delaying roulette alone is insufficient evidence for a production
fix. A useful next experiment is a controlled glass-slab/diffuse/HDR-sun scene
with an analytic or independently integrated target, followed by a matched
sampling estimator for the specular connection. Simply making shadow rays pass
straight through glass would not sample the refracted connection correctly.
No renderer policy, executable, production capture or training was changed.

`cpu_replay.cpp` now accepts optional positional `roulette-start` after `x y`
and reports double-precision red accumulation, empirical standard error and
counts above 100. Build against current headers using the host compiler flags
and scene objects in `run_cpu_checks.py` (its archived candidate mode is separate).
Then run `run_roulette_study.py /path/to/current-header-replay`; its default is
65,536 samples per independent range. The helper only writes diagnostic logs.


## Glass investigation and integrator 5

The roulette study was not sufficient to diagnose the glass. Replaying eight
problem pixels in the windshield, headlight and window regions found additional
transport errors, independent of roulette:

* **Missed thin boundaries.** At `(208,53)`, sample 800002719, an unoffset
  triangle probe finds the windshield exit 0.00186529 world units after entry.
  The old 0.001 origin displacement leaves less than the separate 0.001
  intersection cutoff, so traversal skips the exit and reaches leather while
  `mediumMaterial` still says glass. The corrected ray hits the glass exit at
  depth one and clears the medium before leather. This is recorded in
  `glass-investigation/thin-boundary-probe.txt` and
  `glass-investigation/thin-boundary-corrected.txt` under the dataset diagnostics.
* **Smooth-normal boundary errors.** Geometric and smooth normals can disagree
  on entry/exit near grazing angles, and reflected/refracted directions can
  emerge on the wrong geometric side. At window pixel `(242,47)`, the v4
  replays count 479/518 wrong-side events in the two 4096-sample ranges.
  Geometric classification and a consistent normal fallback reduce these to
  zero. The final replay's unpaired-boundary counts are 4/2 versus 488/472;
  residual counts are diagnostic flags, not proof of a watertight mesh.
* **Incorrect near-critical Fresnel approximation.** For an exiting IOR-1.5
  interface at cosine 0.767, Schlick gives 0.04066 while exact unpolarized
  Fresnel gives 0.23812. The interface now uses the exact equations. Camera
  transmission includes the radiance eta-squared factor; its inverse keeps the
  existing roulette schedule/heuristic eta-neutral. Paired boundaries cancel
  the radiance factor. Photon importance transport does not apply it.

The changes follow the radiance/importance distinction described in
[PBRT's dielectric BSDF](https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF).
Interface side checks, Fresnel reciprocity/critical angles, eta cancellation and
an actual 1.5 mm slab are covered by `material_transport_tests.cu`. The expanded
transport kernel also passes when compiled as CPU code (`failure=0`).

There is a separate sampling problem on the headlight: a rare first-interface
Fresnel reflection receives the full HDR sun. The shader now evaluates both
**immediately escaping** HDR branches with their Fresnel weights and samples
continuations for branches that hit geometry. Every connection is occlusion
checked, including against additional glass. A sampled immediate escape is
terminated without adding the environment again. This is terminal-branch
integration, not straight-through transparent shadows or refracted-sun NEE.
The existing depth cap is preserved. GPU constant-environment tests check the
analytic `R + (1-R)*eta^2` energy, occlusion and depth cap.

All ray/shadow/photon spawning now uses an offset scaled by position and hit
travel distance plus outward rounding, rather than the fixed millimeter shift.
The cutoff for triangles, spheres and traversal boxes is 1e-6. Including travel
distance in the offset is necessary: scaling only by the hit position caused
self-intersections after long primary rays. These are FP32 geometric tolerances,
not a guarantee for arbitrarily tiny geometry at arbitrarily large coordinates.

The implementation is shared by ordinary and persistent shading; the original
`/tmp/ntasad-capture-build` is the only renderer build used. Build with `-j1` on
the shared host. Integrator version **5** invalidates v4 render fingerprints.
Existing v2/v3 captures are preserved; do not resume/mix them with this build or
replace their locks. The default dataset plan is still v3. No bulk generation,
training, HDR clamping, scene/material edits or adaptive pixel sampling occurred.

Reproduce the bounded selected-pixel CPU replay with:

```sh
cd Wavefront-Path-Tracer
/tmp/ntasad-venv/bin/python docs/diagnostics/bmw-hdr-fireflies/run_glass_checks.py --label my-current-replay
```

The helper compiles one small executable inside the existing build, saves source
hashes/commands, and writes evidence under the dataset diagnostics. Archived v4
logs are in `glass-investigation/v4-baseline/`; staged experiments are retained
separately, and `v5-final-cpu/` is the final replay. CPU/GPU path divergence and
software HDR interpolation remain the limitations described above.

Selected CPU results at 4096 SPP (same two sample ranges):

| Pixel / region | v4 red means | v5 red means | v4 largest sample | v5 largest sample |
| --- | --- | --- | ---: | ---: |
| 189,83 / headlight | 60.364 / 74.900 | 125.578 / 118.941 | 67,110 | 3,425 |
| 242,47 / window | 0.01746 / 8.69324 | 0.01596 / 0.01859 | 35,537 | 3.79 |
| 219,77 / headlight surround | 0.30078 / 16.21191 | 0.35–0.36 / 0.35–0.36 | 65,071 | below 30 |
| 166,38 / windshield-interior sun | 0.00601 / 16.33048 | 0.00560 / 15.63315 | 66,862 | 64,003 |

The changed headlight mean is not a claim of bias or convergence: Fresnel changed
and the old estimator had a large rare-event tail. The remaining `(166,38)` path
still needs investigation/qualification; it has no detected wrong-side event
or unpaired boundary. These selected pixels do not qualify a reference budget.


### Independent GPU replicas after the glass corrections

Four fresh 320×180 replicas use the same cameras, noisy sample IDs and independent
reference ranges as v4. The result is in
`../../../../dataset/diagnostics/bmw-hdr-fireflies/bmw-v5-glass-reference-check/report.json`.

| Reference SPP | v4 whole-image noise fraction | v5 whole-image | v5 brightest-5% | Existing 0.1 rule |
| --- | ---: | ---: | ---: | --- |
| 1024 | 0.506 | 0.223 | 0.068 | fails |
| 4096 | 0.174 | 0.073 | 0.072 | passes this calibration view |

At 4096 SPP, fixed-crop replica RGB RMS differences change from 0.434→0.254
(windshield), 0.715→0.445 (headlight), and 0.355→0.078 (side window).
`glass-investigation/gpu-v4-v5-glass-4096spp.png` and the 1024-SPP counterpart
show both replicas at the same display transform/exposure, with enlarged crops.
`analyze_glass_reference.py` reproduces these sheets and region metrics.
The optional Matplotlib convergence plot was unavailable; captures, JSON/CSV
reports and these PIL comparison sheets completed without installing packages.

**This does not qualify the 1024-SPP production plan, other views/scenes, or
native 1080p.** The final 1024-SPP windshield crop is actually noisier in these
particular ranges (replica RMS 0.898 versus 0.116 in v4). This heavy-tail behavior
must not be hidden by the aggregate improvement or by tone mapping.

The final 4096-SPP disagreement is still concentrated: the largest pixel accounts
for 42.31% of MSE, and the top ten account for 85.85%. At 1024 SPP these fractions
are 67.95% and 97.73%. The largest 1024-SPP outlier is now `(207,46)`;
CPU replay matches red 60.828 versus GPU 60.824. Its sample 800001824 crosses
multiple glass boundaries, reflects inside glass, reaches leather, then exits
more glass toward the sun. The high-contribution path has matched boundary
pairs and valid geometric sides. A second reproduced headlight outlier also
follows an opaque reflector → glass → sun connection. Glass occludes ordinary
NEE at those opaque vertices; terminal splitting does not importance-sample the
initial diffuse/glossy direction through the specular chain. These observations
support a remaining specular-connection sampling limitation; they do not prove
that all renderer/material/geometry issues have been eliminated. Replay logs are
in `glass-investigation/v5-remaining-paths/` and error concentration is recorded
in `glass-investigation/gpu-noise-concentration.json`.

A read-only exact-position-welded topology audit is reproducible with
`audit_glass_geometry.py`. The four windscreen/window primitives are closed and
have consistent paired-edge winding. The two `LightGlass` primitives each have
64 open boundary edges. All audited vertex normals agree with triangle winding.
The closed rear-light primitive `94_HeadLightGlass` has negative signed volume,
a separate orientation candidate that needs component/nesting interpretation
before changing asset winding. These findings are in `glass-topology.json`.
No automatic asset repair, thin-sheet substitution or scene edit was applied.


### Final validation and numerical repeatability

The final in-place build passes all **18 CTest executables**, including the
expanded glass kernel and a tilted area-emitter self-occlusion/blocker regression.
The relevant Python frame/sequence/split/convergence/generation tests pass 69
with one optional fixture skipped; actual CUDA capture integration adds 11
passes, for **80 passed / 1 skipped**. Both native v3 pilot EXRs still match
their existing manifest SHA-256 values and retain integrator-4 metadata. Final
binary/source hashes and test counts are in `glass-investigation/final-validation.json`.

A BMW-specific 160×90, 4+128-SPP mode comparison has identical guides and counts;
noisy RGB maximum difference is 1.19e-7. Reference maximum difference is 1.96e-5
(RMS 1.26e-7), with one RGB pixel exceeding the stricter per-pixel
`rtol=1e-5, atol=1e-6` check. This is explicitly recorded as **false**, not a strict
mode-equivalence pass. A separate same-binary/same-sample 320×180, 4096-SPP
repeat has identical noisy RGB and guides, but reference maximum difference
3.63e-4 and RMS 3.61e-6. Its underlying cross-process secondary-path difference
has not been isolated, so exact-bit reproducibility on BMW is not established.
This discrepancy is far below the independent-reference noise measured here;
it does not explain the reproduced tens-of-thousands sun contributions.
`check_glass_gpu_modes.py` records these differences and refuses to reuse captures
from a different binary; its JSON contains the actual pass/identity flags.

The official v5 convergence report was regenerated against the exact final
executable. Earlier in-development replicas remain separately under
`bmw-v5-glass-reference-check-before-final-build/`. The final numbers still round
to 0.223 / 0.073 at 1024 / 4096 SPP. No native v5 pilot or production clip was
captured, and no dataset/train/report default budget was changed.


## Visible spot regression and HDR continuation investigation — integrator 6

The user's visual inspection identified a limitation of the v5 conclusion:
**fewer squared replica errors did not mean fewer visible fireflies**. With the
same display transform, the windshield has more scattered bright pixels in v5.
For linear luminance >1 and >10 times the median of its eight neighbors, the
1024-SPP replicas have 1/2 spots in v4 versus 8/8 in v5. At 4096 SPP the counts
are 4/8 versus 7/11. These counts include possible real glints; they are a
diagnostic rather than a ground-truth classification. Less extreme specks also
increase. `analyze_hdr_spots.py` additionally records spots inconsistent with
the other replica, regional RMS, and several absolute brightness thresholds.

The old scalar report measures replica RMSE divided by the error of **one
4-SPP input**. That denominator changed from 0.53277 to 0.76071 between v4/v5.
The absolute 4096-SPP noise estimate did improve, 0.09274 to 0.05537, but large
headlight values dominate squared error. More smaller spots in dark regions
can coexist with that reduction. The display-domain replica PSNR actually
worsened, 37.95 to 36.75 dB. Passing that global empirical rule is not a
firefly-free certificate. The 1024- and 4096-SPP levels also use different,
nonoverlapping sample ranges; they are not nested refinements of one image.

Read-only CPU replay reproduces the chrome-area GPU outliers closely:

* `(105,90)`, replica one at 4096 SPP: CPU red 1.31293 versus GPU 1.31334.
  Sample 800008671 is shiny chrome → diffuse mattefloor → paired windshield
  entry/exit → HDR sun. Chrome continuation weight is 0.871, not an enormous
  BSDF/PDF quotient; the final sun value is about 65,431 and the path contributes
  5,216 before pixel averaging. Boundary sides and pairs are valid on that path.
* `(129,89)` is a diffuse surface seen between grille parts → paired glass →
  sun, contributing 4,192 in sample 800008244. It is not necessary for the first
  visible surface to be glass for the same poorly sampled connection to appear.

Removing only HDR texels with a channel above 100 in a **diagnostic radiance
lookup**, while retaining the original alias table, RNG and sampled geometry,
eliminates the selected extreme samples. A constant-map ablation does too.
Disabling terminal glass integration does not eliminate several reproduced
spikes (the chrome sample becomes 5,442). This distinguishes the HDR-sun path
tail from the terminal integration change. No production clamp or asset edit
is applied. Logs are under `hdr-spot-investigation/` in the dataset diagnostics.

The implementation addresses three sampling issues:

1. **Missing useful continuation proposal for the specular connection.** NEE
   at the diffuse vertex stops at glass, and cosine-only continuation almost
   never discovers the tiny refracted sun. Diffuse continuations now mix equal
   probabilities of cosine and HDR-environment directions. Each continuation
   traces the actual geometry; throughput divides by the complete mixture PDF.
   Direct-light MIS uses that same competing PDF. Glass shadows remain opaque
   for ordinary NEE. Metals retain the matched GGX/delta estimators.
2. **Importance weights ignored bilinear reconstruction shoulders.** The raw
   texel table assigned very little probability to neighboring cells containing
   filtered bright radiance. It now uses the separable cell-average tent weights
   `[1/8, 3/4, 1/8]`, wrapping longitude and clamping latitude, plus the existing
   1% uniform support. Radiance data/filtering are unchanged. The issue is severe
   for an isolated synthetic sun texel; the BMW HDR already has a wider sun
   footprint, and its maximum filtered/raw probability ratio is only 3.293.
   This is not evidence that filtering alone caused the BMW spikes.
3. **Diffuse geometric fallback had an incomplete PDF.** Resampling a rejected
   smooth-normal direction about the geometric normal and reporting just the
   replacement's cosine PDF did not describe the actual sampling distribution.
   Below-geometry samples now terminate with zero contribution. A tilted-normal
   regression checks the known projected-area integral `(1+cos(tilt))/2`.

The mixture is ordinary importance sampling, consistent with the density
accounting in [PBRT's sampling discussion](https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency).
It is a fixed per-path proposal, not adaptive pixel sampling, material
roughening, photon-map substitution or radiance clamping. It helps near-parallel
glass connections; it does not solve arbitrary curved or reflected specular
chains. A separate CPU prototype and archived v5 source hashes reproduce the
guide-only experiment without creating another build tree.

The new GPU regression places a compact 50,000-radiance source behind a parallel
glass slab. Independent quadrature integrates its entire bilinear footprint
and finite-depth Fresnel transmission, without using the importance table.
The guide-only/raw-table candidate gave mean 0.12928 at 65,536 samples versus
target 0.21655; this is a finite-sample failure, not proof of asymptotic bias.
The final filter-aware mixture gives 0.21553 with empirical standard error
0.00156. Direct-sun, constant-map, tilted-normal, occlusion and existing glass
regressions pass. All 18 CTest executables pass in the original build.

Integrator version is **6**. Only `/tmp/ntasad-capture-build` is rebuilt, with
`-j1`; small CPU diagnostics also live there. Existing v2/v3/v4/v5 captures and
their fingerprints are preserved. The production plan and 1024-SPP default are
unchanged. Bulk generation and training remain stopped.

The four v6 GPU calibration images are preserved under
`bmw-v6-hdr-reference-check/`. At 1024 SPP, windshield bright-local counts fall
8/8 → 0/0 and replica RGB RMS falls 0.89816 → 0.02475 compared with v5. At
4096 SPP, windshield RMS is 0.01541 and grille RMS 0.00339 (v5 0.25364/0.03609).
However, the headlight crop changes 0.44451 → 0.45694 at 4096 SPP. Its remaining
outliers expose a connection not addressed by diffuse-only guiding. The v6
whole-image noise estimates are 0.04430/0.03943 at 1024/4096 SPP. Its empirical
ratio rule passes both levels, which still does not qualify native production
data. All 18 renderer tests and 80 relevant Python tests pass, with one optional
fixture skipped. Both native v3 EXRs still match their original hashes and
integrator-4 metadata.


### Chrome inside glass: rejected finite GGX experiment — integrator 7

The residual v6 headlight pixel `(182,90)`, sample 800004240, is paired glass →
rough chrome → paired glass → HDR sun, with **no diffuse vertex**. Its sample
contribution is 14,663. The metal weight is 0.669; the issue is again rare light
discovery after a specular connection, rather than an exploding GGX quotient.

The experiment extended the weighted environment/BSDF proposal to finite GGX continuations,
including rough metals and imported specular/glossiness. The evaluator exposes
the full mixture density to direct-light MIS; the continuation divides by it
too. Exact delta metal/glass directions and the material's GGX model are
unchanged. The controlled sun/slab regression additionally checks rough-metal
energy against independent double-precision GGX quadrature, with and without
glass.

In the two selected 4096-SPP CPU ranges, `(182,90)` means become 0.44473/0.47500
and the largest sample falls 14,663 → 25.88. At `(164,96)`, the earlier 36,897
glass/chrome/sun sample is absent and the new maximum is 15.87, with means
0.71688/0.71330. These are selected-pixel observations, not image qualification.
The `(182,88)` replica still has a sample near 2,494. Another pixel `(188,85)`
has a first-surface Fresnel reflection contribution near 2,936, before any
continuation; its subpixel sun footprint is a separate camera-sampling tail.
Bright glints must not all be classified as identical transport bugs.

**The full-image test rejects this experiment.** Whole-image replica noise
worsened from v6's 0.04430/0.03943 to v7's 0.09799/0.06868 at 1024/4096 SPP.
New large outliers appear around the other headlight: `(156,88)` at 1024 SPP and
`(168,91)`, `(148,94)`, `(158,87)` at 4096. Improving selected paths was not
enough evidence to retain the generic GGX environment proposal. The raw v7
report and images are preserved, with the rejected source/patch and path logs
under `hdr-spot-investigation/v7-source/` and `v7-rejected-paths/`. No qualification
or production-default change follows from its 4096-SPP ratio passing 0.1.

### Retained correction — integrator 8

The final code restores **exactly the v6 traversal/continuation estimator**:
filter-aware HDR importance weights, diffuse environment/cosine continuation,
and proper diffuse geometric rejection. Finite GGX keeps its original matched
BSDF continuation PDF. A new integrator version **8** prevents continuation of
the intervening v7 experiment. `candidate-selection.json` records this decision
and the restored header's equality to the archived v6 source. Independent GGX
quadrature remains in the direct-HDR energy regression; a tight variance bound
for GGX behind glass is not asserted for this incomplete sampling approach.

Reproduction uses `run_hdr_gpu_validation.py --integrator 8` after the original
build is rebuilt with `-j1`. It checks source/capture versions and binary hashes,
runs renderer regressions, and captures only the four 320×180 replicas declared
in `bmw-v8-hdr-reference-plan.json`. V6/v7 captures remain separate. Production
plan/budget and prior native v3 data are unchanged. Remaining chrome/glass
connections need a proposal that accounts for their specular geometry, rather
than accepting a generic world-direction guide on selected-pixel evidence.
Bulk generation/training remain stopped; adaptive pixel sampling stays deferred.

## Appearance regression and glTF feature audit — integrator 9 candidate

The user's brown/pixelated windshield and missing headlight detail report makes
the v6/v8 appearance **unqualified**, despite fewer strict white outliers. The
current candidate adds missing opaque material layers; it is not a declaration
that windshield/headlight reference quality is solved.

`appearance-investigation/asset-audit.json` records the local extended glTF:
114 meshes, 114 active instances, 792,746 triangles, 26 materials, no textures,
vertex colors, skins or animations. All its geometry encodings are supported by
the importer. The headlight primitives are present. The original 114 source PLYs
and original PBRT PNG are only LFS pointers locally, so their geometry/image bytes
have **not** been independently compared. The usable supplied reference is
`gltf/bmw-m6.png`, a Cycles render, not the original PBRT render.

Two required extensions were silently ignored: specular and clearcoat. BMW uses
clearcoat on 12 materials, including the paint, headlight housing and floor.
Its conversion already reduces the base color to account for a coating (paint
0.7 → 0.48503; HeadLight 0.64 → 0.41779). Keeping that base while dropping the
coat loses reflections and changes appearance. Integrator 9 imports opaque
specular/clearcoat factors, textures, independent coat normals, continuous
metallic mixing and zero-roughness delta layers with matching continuation and
direct-light MIS. Legacy procedural/OBJ and specular/glossiness models remain
separate. `gltf_pbr.cuh` documents the extension's approximate layering model.
This is a material-support correction, not an adjustment of assets or exposure.

The windshield's changed mean needs a separate explanation. Controlled **CPU
integrator-8** replays change only the diffuse proposal, with identical BSDFs,
geometry, HDR and geometric rejection. Over 262,144 new samples:

| Pixel | cosine-only red mean ± empirical SE | HDR/cosine red mean ± empirical SE |
| --- | --- | --- |
| (185,53) | 0.26671 ± 0.07139 | 0.21298 ± 0.00282 |
| (206,52) | 0.25123 ± 0.06297 | 0.21535 ± 0.00181 |

Both cosine means are consistent with the guided means at these uncertainties;
their original 4096-SPP replicas mostly missed the contribution. Removing only
sun texels from radiance lookup at (185,53) gives red 0.06751 ± 0.00035, close to
the earlier blue-gray background. Logged paths are camera → windshield pair →
LEATHER → windshield pair → HDR sun. This supports a newly resolved warm caustic
contribution, with substantial remaining sampling tails. It does **not** prove
all changed pixels or the entire complex glass model correct. The empirical
standard errors themselves are fragile for heavy-tailed path distributions.
No production radiance clamp or caustic suppression is applied.

The supplied reference metadata says Cycles 5.3 beta, 1400×1000, 4096 SPP,
OPTIX denoising, Blackman-Harris 1.5 px, Khronos PBR Neutral + sRGB. A bounded
700×500 capture now uses its position/FOV/aspect. The exported glTF camera has a
negative X basis (determinant -1); our viewing PNG alone is flipped horizontally
to align it. Production camera plans are unchanged. `compare_appearance.py`
applies PBR Neutral and sRGB at exposure one, and labels the denoised supplied
reference versus raw captures. This is not a linear-HDR accuracy metric.

Further reference caveats: source PBRT enables `regularize true`; conversion
flattens the LEATHER mix and approximates spectral conductors/coatings. Cycles
[defaults](https://github.com/blender/blender/blob/main/intern/cycles/blender/addon/properties.py)
include glossy filtering 1 and indirect sample clamp 10. The conversion report
does not record overrides for these two settings, so their actual values in
the supplied render are **unknown**; it cannot be assumed to be an unclamped,
unregularized reference. Those defaults are a comparison hypothesis, not proof
of the cause of this particular PNG difference.

There is also a glass semantics gap: BMW declares transmission/IOR but **no
volume**. The glTF [transmission specification](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_transmission)
defines that as thin, without macroscopic Snell refraction; our current shader
treats every nonzero transmission as solid smooth dielectric. The original PBRT
source instead uses dielectric boundaries, and the current Blender importer
wires transmission to Principled BSDF. These are not interchangeable reference
models. Thin/solid tile replays are explicit diagnostic ablations, not silent
asset replacement or production material fixes. Required-extension limitations
now appear in import logs. Rough/partial transmission, complete layered emission,
and full glTF conformance remain outside the implemented opaque correction.

Reproduction: `audit_appearance.py` is read-only; `investigate_appearance.py`
requires the integrator-8 ABI for its controlled proposal comparison and refuses
current material-layout changes. `appearance_tiles.cpp` supports a separate
thin/solid model diagnostic with the current CPU shader. All diagnostic programs
compile inside `/tmp/ntasad-capture-build`; no additional build tree is created.
`run_appearance_validation.py` runs serial renderer/Python checks, four 320×180
reference replicas and one 700×500 appearance capture. Production remains stopped.

### Integrator 9 results and remaining headlight cause

All 18 renderer tests and 80 relevant Python tests pass (one optional fixture
skipped). The native v3 pilot hashes and integrator-4 provenance are unchanged.
The 700×500 v8/v9 captures have identical albedo, normals, geometric normals,
depth, motion, masks and counts. Coating restores paint/floor reflections, but
the smaller headlight cylinder remains dark. Raw references remain noisy:
1024-SPP noise fraction 0.1641 **fails**; 4096 fraction 0.0437 passes only the
empirical rule. Whole-image strict bright-local counts at 1024 are 18/19 (v8:
2/3), and unstable counts are 34/40 (v8: 5/7); at 4096, unstable counts remain
36/44 (v8: 5/8). Restoring missing reflective material layers does not qualify
the previous sample budget. Do not call this a general firefly correction.

Thin/solid CPU ablations preserve the warm windshield contribution and change
the headlight optical shape without restoring the intended center. This rules
out a blind thin-glass substitution as an adequate appearance fix. The real
glTF thin/volume distinction still needs implementation/interpretation before
claiming complete material support. The PBRT headlight area-light declarations
are commented out; adding an emitting light would change the authored scene.

Replay at raw 700×500 pixel `(343,252)`, sample 810000005, traverses a paired
front pane, then LightGlass, then two shinychrome hits. At the last chrome hit
the view cosine is **0.06875** about the smooth normal but **0.62039** about the
triangle normal. Its ideal reflected ray must therefore be below the actual
surface (`2*0.06875*(Ns·Ng)-0.62039 < -0.48289`). Rejecting sampled directions
there absorbs almost the entire narrow reflection lobe and leaves a black
cylinder. This is a reflective shading-normal problem behind glass; the
geometry is loaded. Selected rays are in `headlight-ray-343-252-*.txt`.

The CPU normal ablation raises a defective reflective normal towards Ng until
the ideal reflected ray clears a small geometric horizon margin, following the
policy in [Cycles' reflective-normal helper](https://github.com/blender/blender/blob/main/intern/cycles/kernel/closure/bsdf_util.h).
It applies the same frame in evaluation and sampling. At 128 SPP, the dark-center
red mean changes 0.03279 → 0.08357 while the neighboring cylinder changes only
0.13447 → 0.13746. `reflection-normal-ablation.png` visibly restores the smaller
cylinder. This is separate from the rejected v7 generic environment/GGX proposal.

### Reflective frame correction — integrator 10 candidate

The renderer retains the material layers and diffuse/HDR proposal, and corrects
opaque glTF base and coat reflective frames **before both** evaluation and
continuation. Diffuse retains its authored normal. Valid mirror directions are
unchanged. The invalid frame is rotated only enough to meet
`min(0.01, 0.9*(Ng·V))`; geometric rejection still applies to individual sampled
directions. There is no fallback resampling, PDF substitution, roughness change,
radiance clamp or asset edit. This is an explicit view-dependent smooth-normal
rendering policy, not a proof of a fully reciprocal physical shading-normal
model. Tests exercise the defective chrome configuration, horizon margin,
unchanged valid normals and independent diffuse/coat frames.

Integrator 9 source/validation and all rejected/diagnostic variants are retained.
Current version 10 prevents mixing earlier checkpoints. Reproduce current GPU
checks with `run_appearance_validation.py --integrator 10`; v9-only CPU scripts
refuse to mislabel v10 sources. All compilation stays in the original build,
one job at a time. Generation/training and adaptive sampling remain stopped.

### Integrator 10 final validation and limits

The final original-build executable passes **18 renderer tests and 80 relevant
Python tests**, with one optional fixture skipped. Both original native v3 EXRs
match their recorded hashes and retain integrator-4 provenance. No production
frame was replaced. Source/binary hashes and checks are in
`appearance-investigation/v10-validation.json`.

The GPU confirms the targeted appearance improvement. At matched-camera pixel
`(343,252)`, the smaller headlight cylinder's red mean rises **0.03569 → 0.06877**
at 1024 SPP; its neighbor changes 0.13567 → 0.14358. The cylinder remains different
from the supplied Cycles render: this is a correction of a reproduced reflective
frame defect, not a declaration that every optical/material difference is solved.
All v9/v10 guides, masks and counts are identical. See `headlight-detail-v10.png`,
`windshield-detail-v10.png`, `matched-camera-appearance.png` and
`final-appearance-summary.json` under the appearance investigation.

**Reference quality is still unresolved.** The final 1024/4096-SPP noise estimates
are 0.14049 / 0.04925, with fractions 0.16413 / 0.05753. 1024 fails; 4096 passes
only the empirical calibration rule. The 4096 absolute estimate is **worse** than
v9's 0.03739. Whole-image 4096 strict bright-local counts are 4/13 (v9: 4/11),
and unstable counts are 36/47 (v9: 36/44). These material/normal corrections
change the rendered model and restore contributions; they are retained for
appearance/correctness, not promoted as a reduction of all visible specks.

One remaining 4096-SPP grille pixel `(101,87)` has GPU red 10.1171; CPU replay
gives 10.2530, not an exact match. Its largest CPU sample, 800007353, contributes
41,672 red along chrome → chrome → chrome → paint → paired windshield → HDR sun.
The three chrome continuation weights remain of order one. A generic world-HDR
proposal at paint does not follow the refracted sun direction through the curved
glass, so a rare cosine sample can still carry essentially the full sun value.
The brighter frame is not caused by an enormous GGX BSDF/PDF quotient on this
replayed path. `v10-grille-tail.txt` preserves it and its sampling/throughput data.

The warm windshield, refracted-sun tails, incomplete glTF thin/volume distinction
and native-resolution qualification remain outstanding. Bulk generation/training
stay stopped. Plan-v3 cameras, splits, sample ranges and 1024-SPP default remain
unchanged; v2 and the old pilots remain separate. No adaptive pixel sampling,
production clamp, roughening or transparent-shadow approximation was introduced.

## Practical indirect clamp — integrator 11

Cycles' current source defaults its indirect sample clamp to 10 and glossy
filter to 1. It scales individual fully weighted contributions before writing
the render buffer, using mean absolute RGB magnitude (the internal sum limit is
three times the user value). Sources:
[defaults](https://github.com/blender/blender/blob/main/intern/cycles/blender/addon/properties.py),
[limit conversion](https://github.com/blender/blender/blob/main/intern/cycles/scene/integrator.cpp),
[accumulation](https://github.com/blender/blender/blob/main/intern/cycles/kernel/film/light_passes.h).
Its supplied denoised BMW PNG is not evidence that raw low-SPP unclamped paths
are free of fireflies.

The renderer now applies the same practical clamp policy to indirect
contributions from environment, triangle/punctual lights, emissive hits and
terminal glass/coat branches. Direct first-surface lighting and first-bounce
escapes remain unbounded. Glass counts as scattering. A common RGB scale
preserves color ratios. The change does not clip the environment, modify
roughness, change throughput/PDFs, skip caustics or alter geometry.

Use `--clamp-indirect 10` on the CLI or `"indirect_radiance_clamp": 10` in capture
JSON. Both default to 10. **Set zero for the previous raw estimator.** Nonzero
values deliberately lose some bright indirect energy; this is a practical
variance/bias tradeoff, not an unbiased solution for specular caustics. Clamp
settings enter render fingerprints, worker configuration (protocol 7) and frame
metadata. Old EXRs and their manifests are preserved.

The self-contained [preview config](bmw-clamp-preview.json) uses the local
BMW/HDR scene. The comparison tool captures matched raw/clamped RNG ranges at
64, 256, 1024 and 4096 SPP, with two replicas each, and checks finite buffers
and unchanged guides/counts. It records lost mean energy as well as speck and
replica-noise statistics. No denoising or exposure adjustment is applied:

```sh
/tmp/ntasad-venv/bin/python \
  docs/diagnostics/bmw-hdr-fireflies/validate_indirect_clamp.py \
  --config docs/diagnostics/bmw-hdr-fireflies/bmw-clamp-preview.json \
  --output /tmp/bmw-indirect-clamp-validation
```

The output directory must be fresh. This is bounded appearance validation,
not native-resolution training-reference qualification. Existing generation
plans and data have not been rewritten or restarted.

### Integrator 11 validation results

The existing build was rebuilt serially and **all 18 CTest tests passed** on the
RTX 3090, including the new HDR miss, terminal glass and weighted punctual-light
clamp regressions. The capture binary SHA-256 and source hashes are recorded in
[`clamp-validation.json`](clamp-validation.json). The existing build targets
architecture 75; the driver compiled it for the 3090 on first launch. Startup
times here are not GPU performance measurements.

The 16 paired captures use identical camera/geometry/sample ranges between
clamp settings, no denoising, and the same fixed display transform. All buffers
are finite and all guides/counts are identical. The whole-image unstable-local
outlier counts (two replicas) and empirical replica-noise RMSE are:

| SPP | Raw outliers | Clamp 10 outliers | Raw noise RMSE | Clamp 10 noise RMSE |
| --- | --- | --- | --- | --- |
| 64 | 35 / 16 | 20 / 13 | 1.013516 | 0.299508 |
| 256 | 32 / 18 | 2 / 1 | 0.318541 | 0.081378 |
| 1024 | 37 / 28 | 0 / 0 | 0.108205 | 0.039586 |
| 4096 | 52 / 37 | 0 / 1 | 0.054258 | 0.007546 |

The unstable-local rule is linear luminance >0.1, >4 times the median of eight
neighbors, >4 times the other replica, and >0.1 excess over that replica.
These thresholds can flag real tiny highlights. At 256 SPP, floor counts fall
18/10 → 0/0 and windshield counts 7/7 → 0/0. Both regions also have zero such
outliers in both clamped 1024/4096-SPP replicas. At 64 SPP some remain. The lone
4096-SPP clamped outlier is at headlight pixel (188,85); raw/clamped red values
there are 2.0880/2.0844. This is not a claim that every bright speck is eliminated.

At 256 SPP the empirical whole-image noise RMSE falls by 74.45%; at 4096 by
86.09%. Average sampled RGB brightness falls by 1.91% and 1.26%, respectively.
Those brightness changes describe these finite sampled images, not a reliable
ground-truth bias estimate for the heavy-tailed raw estimator. Direct highlights
remain HDR: the 4096-SPP raw/clamped maxima are approximately 169.35 and 171.58
in the two replicas. No source radiance or final-image pixel cap is applied.

Additional checks captured two 160×90 frames in each wavefront mode, a larger
640×360 raw/clamped BMW pair at 256 SPP, and a 320×180 HDR Breakfast Room smoke
render. Both modes have identical guides/counts; radiance is not bit-identical
(maximum reference-channel difference 0.00557; largest per-channel RMS 8.65e-5).
With clamp zero, the new 320×180 4096-SPP capture has identical guides to the
saved integrator-10 baseline; maximum reference-channel difference is 0.00148
and largest per-channel RMS is 1.30e-5. This is a numerical comparison, not a
bit-exact certificate. The old baseline and all production datasets are intact.

View the [640×360 before/after with floor and windshield crops](clamp-before-after.png)
and the [full sample-budget/replica comparison](clamp-comparison.png). Raw EXRs,
capture configurations and logs for this run remain under `/tmp/bmw-firefly-fix/`.
