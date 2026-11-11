# Bistro missing geometry and display exposure

Integrator 12 replaces the fixed `1e-8` triangle determinant cutoff with a
relative floating-point cancellation bound. Bistro has very small mesh-local
triangles beneath a 100x root transform. Local rays intentionally retain their
unnormalized inverse-transformed directions so their hit distances stay in world
ray units. The previous cutoff discarded almost all scooter and streetlamp
triangles, even though the glTF loader imported their instances. This affected
visibility and shadows, not just the exported display image.

The importer also retains reflective materials when an emissive texture has a
zero or omitted `emissiveFactor`. That factor defaults to zero in glTF. Previously
the texture alone selected a terminating emissive material, turning the Bistro
front banner black instead of allowing its specular-glossiness reflection.
Positive emissive factors still select textured emitters.

Bistro's remaining lamp/sun mismatch is fixed in scene data, without additional
renderer branches. `assets/scenes/bistro-lighting.json` scales all punctual
intensities by `2.5*pi/6830`, making the sun approximately 7.854 in renderer units
while preserving the bulb emissive factors. The pi accounts for the renderer's
Lambertian BRDF; Niagara's reference shader multiplies diffuse albedo directly by
its artistic sun value 2.5 and adds bloom. This is an appearance calibration,
not a physical-unit conversion or an exact Niagara reproduction.

`tools/prepare_gltf_lighting.py` writes an ignored derived glTF from the original
PNG-converted asset, rebasing external resources without copying textures or
geometry. `tools/setup_assets.py` generates and verifies it. Both Bistro scene
manifests point at that derived file. Source assets remain reproducible.

PNG previews now use exposure 1 for every scene; `--exposure` is an explicit
optional display override. The earlier Bistro-specific 0.00579 display preset
was removed because it darkened the bulbs along with the excessively bright sun.
No bloom is added to the training EXRs.

## Validation and continuation

On 2026-11-11, the generic geometry/material fixes passed seven renderer checks.
The final pipeline passed 99 focused Python checks and seven asset setup checks.
The geometry regression fails before the cutoff fix. Tests cover scaled camera
and shadow intersections, zero/default emissive factors, display transforms,
immutable capture ancestry, reuse evidence, R2 lock-transition upload order,
unrelated-prefix rejection and archive idempotence.

The final Bistro preview is one 640x360 frame, 4 input / 256 reference SPP,
depth 12 and clamp 10. The scooter, streetlamps, hanging bulbs and front banner
are visible with bright emission. Its metadata, checksums and finite buffers
were checked. This is a small appearance check rather than a reference
convergence qualification.

BMW was compared with the original locked integrator-11 binary and the corrected
integrator-12 binary at two archived camera poses, 480x270 and 128 reference SPP.
Relative HDR RMSE was 0.01045% and 0.01164%, with identical albedo and visibility
masks. One depth pixel changed by over 1% in the first view; none in the second.
This supports the user's explicit BMW reuse decision, rather than proving every
original frame identical. All 288 archived BMW EXRs and their 288 manifests
matched R2 size/ETag metadata. No cloud EXRs were downloaded.

The user requested continuation in the same local directory and R2 prefix.
`dataset/denoiser-v4/capture-history/` retains the original capture lock, BMW
receipt, comparison evidence and invalidated Bistro log. The new lock names
this parent explicitly and authorizes BMW reuse only while its configs, source
asset hashes and recorded validation still match. BMW's receipt and frame
manifests retain their original provenance. On the next scene archive, R2 saves
and verifies the old lock/evidence under immutable history keys before replacing
the live lock. Unrelated remote locks continue to be rejected.

The five old Bistro frames were moved to
`dataset/denoiser-v4/recovery.revision.partial/`, outside normal capture paths;
their active completion log was moved to history so exports restart at frame 0.
The old binary was backed up as
`/tmp/ntasad-capture-build/dataset_capture.integrator11-e4c694544e34`, then the
corrected binary was installed at the original command's renderer path.

From the workspace root, run:

```bash
bash dataset/denoiser-v4/resume.sh
```

It retains `r2://denoiser/denoiser-v4`, skips 288 BMW frames, and starts Bistro
at frame 0. Preparation itself did not launch production or change cloud objects.
Local previews, comparison results and preparation records are in
`dataset/diagnostics/bmw-bistro-resume/`.
