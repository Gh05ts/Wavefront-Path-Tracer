# NTASAD fidelity audit follow-up

The complete equation/figure audit is in the sibling denoiser's
[docs/ntasad-fidelity-audit.md](../../denoiser/docs/ntasad-fidelity-audit.md),
including links to the original paper, supplement and DASR derivative source,
PDF checksums, specified requirements, reconstruction choices and limitations.

The adaptive pyramid now downsamples equally weighted available pixel means
(`pixel_mean_v2`). Previously it weighted colors by sample counts, changing even
dense images with unequal SPP. The corrected dense path agrees with the uniform
box-pyramid reconstruction with shared network weights. Fractional availability
coverage preserves equal original-pixel weighting across multiple scales.
Sparse normalization is still an explicit reconstruction choice; the paper
does not provide an executable sparse downsampler.

New joint checkpoints use envelope v2 and record reconstruction semantics.
Original v1 checkpoints automatically retain `sample_mass_v1`. Their inference
outputs remain bitwise identical in the captured pilot comparison. Resuming
their training requires that legacy policy explicitly; an ordinary resume
cannot silently switch reconstruction semantics. Uniform source/checkpoint/
trainer/runtime files and existing camera/capture work were preserved.

Main-paper LARGE topology, feature channels, 5×5 kernel heads, equations (1)–(4),
initialization, radiance transform, five-frame recurrence, augmentation, replay
and cited gradient estimator were checked. The supplement's SMALL models are
distinct architectures and remain unimplemented. Pooling/padding conventions,
precise radiance boundaries, sparse fallbacks, zero-count derivatives and exact
author-code parameter-count equivalence remain qualified in the full audit.

Validation:

* **266 passed** on the GPU-enabled model/data/checkpoint suite, including the
  C++ EXR interoperability fixture. Six added regression tests cover the pyramid
  and legacy checkpoint/trainer behavior.
* RTX 3090 smoke checks use default 48/48 widths, five captured 17×13 frames
  and a synthetic 128×128 crop. All four learned allocations and both networks
  receive finite nonzero gradients; both networks change after an Adam step.
  New checkpoint round-trip outputs match bitwise.
* The pre-audit captured pilot checkpoint reproduces its CPU RGB, validity,
  expected allocations and integer counts bitwise through the legacy loader.
* A full 284-test run was attempted but stalled in the existing renderer capture
  test on frame zero of a 17×13 capture. That process was stopped; both renderer
  capture modules (18 tests) were excluded from the completed 266-test run.
  Renderer diagnosis is deferred, and no fresh full-suite pass is claimed.

Artifacts: `build/ntasad-v5/fidelity-audit/` contains the GPU smoke report,
model-suite JUnit report, reviewed patch, application script, change manifest
and verification report. Pre-change sibling file copies are preserved under
`.ntasad-backup/fidelity-audit-before/`.

These checks establish computation and compatibility, not trained quality or
sampler gains. Performance/backend changes and long training remain deferred.
Revisit throughput before long training; production v4 validation, overfit and
matched-budget uniform/random/learned sampler evaluation remain necessary.
Renderer adaptive scheduling and CUDA runtime integration remain last.
