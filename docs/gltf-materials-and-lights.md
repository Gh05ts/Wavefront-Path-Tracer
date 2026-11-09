# glTF specular/glossiness and punctual lights

The loader supports `KHR_materials_pbrSpecularGlossiness` and
`KHR_lights_punctual` for static scenes. The implementation follows the material
parameters and light transforms/falloff in the [Khronos material extension](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness)
and [light extension](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_lights_punctual).
This is not a claim of complete glTF renderer conformance.

## Materials

Specular/glossiness properties take precedence over the core metallic/roughness
fallback. The loader reads diffuse RGBA, specular RGB (normal-incidence F0),
glossiness, diffuse textures, and packed specular/glossiness textures, including
their existing texture-coordinate and texture-transform bindings. Diffuse and
specular texture RGB use the sRGB transfer function; opacity and glossiness
alpha remain linear. Factors are linear. Diffuse alpha also participates in
existing alpha-mask tests.

Opaque specular/glossiness materials use Lambert diffuse plus isotropic GGX,
Schlick Fresnel and Smith masking. Diffuse is scaled by `1-max(F0)` and roughness
is `1-glossiness`. The path samples a mixture of diffuse and GGX reflection,
using its matching PDF for throughput and area-light MIS. Samples below the
surface are absorbed, not resampled. A minimum perceptual roughness of 0.02
regularizes perfectly glossy surfaces; they are finite lobes, not delta mirrors.
The same helpers are used by standard and persistent wavefront rendering.

Imported transmission materials retain the existing dielectric branch, and
emissive materials retain their existing emission behavior. Core metallic/roughness
and procedural/OBJ metal materials still use the legacy shading implementation;
this change does not replace all materials with GGX. Alpha blending, vertex
colors, rough refraction and complete layered emissive/reflective materials are
not added here. Denoising albedo guides use the imported diffuse color/texture.

## Lights

Point, spot and directional lights are instantiated only from nodes in the
selected glTF scene. Parent transforms affect positions and normalized emission
directions. Intensity, range and cone angles are not multiplied by node scale.
Renderer scene/object scaling moves light positions along with geometry;
intensity and range retain their declared values. Very different scene scales
therefore require reviewing lighting rather than assuming exposure invariance.

Point/spot lights use inverse-square attenuation, with the extension's smooth
fourth-power range cutoff. Spotlights use squared smooth cone falloff. Directional
lights have no distance falloff. Light RGB is linear and multiplied by the
declared intensity. These values enter the renderer's linear RGB calculation
directly; no spectral or absolute photometric calibration is claimed.

At each diffuse/glossy hit, every directional light is evaluated and one
point/spot light is selected uniformly with inverse-selection-PDF weighting.
This avoids undersampling Bistro's sun among its 96 point lights. Positional
lights can still be noisy in low-SPP renders; there is no light-tree or spatial
importance sampler. Each light contribution uses a shadow ray, bounded by the
emitter distance for point/spot lights. Delta-light contributions do not receive
area-light/BSDF MIS weights. Existing area-light sampling remains separate.

Punctual lights are not visible emitting geometry. Standalone glTF scenes default
to the simplified sky; a configured [HDR environment](hdr-environments.md) replaces it.
An added preset top light would be extra
illumination; the supplied Bistro manifest disables it. New lights/materials
are not integrated into the optional photon-map caustic emitter/tracer; keep
caustics disabled for these captures.

## Bistro preview and checks

After running the [DDS conversion](dds-gltf-conversion.md), use
`assets/scenes/bistro.json`. Its object points at `bistro-png.gltf`. From the
renderer directory:

```bash
/tmp/ntasad-capture-build/dataset_capture \
  ../Path-Tracer-Denoiser/configs/bistro_preview_capture.json \
  ../Path-Tracer-Denoiser/runs/bistro-preview-new
```

The JSON contains a relative scene-manifest path resolved against the config
itself. The preview uses the asset's camera, 320×180, 4 noisy SPP and a 64-SPP
comparison reference.
It is an appearance/loading smoke test, not convergence evidence or training data.
No new train/validation/test assignment is made for Bistro.

The renderer integrator fingerprint version is now 3 following HDR integration, so existing render
checkpoints cannot silently resume with the new shading/sampling semantics.
This is separate from the denoiser checkpoint format. Requalify references before
using newly rendered material/light scenes for training.

Checks cover material defaults and precedence, texture bindings/transforms,
opacity, retained glass/core material import, active-scene light selection,
parent transforms, light falloff/cones, BSDF reciprocity and a Monte Carlo versus
independent hemisphere-integral comparison. GPU checks cover analytic point
illumination, shadow occlusion and finite shadow segments, directional-plus-point
sums, and sRGB RGB versus linear alpha. Existing scene acceleration, capture
lighting and config checks also pass. A small captured fixture produced identical
radiance, reference and guide buffers in both renderer execution modes.

The completed RTX 3090 preview is in the denoiser's ignored
`runs/bistro-material-light-preview-v2/` directory, including paired HDR EXR,
`comparison.png`, `reference-preview.png` and display/exposure metadata. All
buffers are finite. Import reported 234 specular/glossiness materials and 97
punctual lights, with 551 mesh assets and 2,909 instances. The initial v1 preview
is retained as a diagnostic of excessive sun-sampling variance; use v2 to inspect
the final implementation.
