# glTF materials and punctual lights

The loader supports `KHR_materials_pbrSpecularGlossiness` and
`KHR_lights_punctual` for static scenes. The implementation follows the material
parameters and light transforms/falloff in the [Khronos material extension](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness)
and [light extension](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_lights_punctual).
This is not a claim of complete glTF renderer conformance.

## Opaque metallic/roughness, specular and clearcoat (integrator 9)

Opaque imported metallic/roughness materials now use a diffuse/dielectric-GGX/
metal-GGX mixture with continuous metallic factors. `KHR_materials_specular`
controls dielectric F0 color and reflection strength, including its zero-strength
grazing limit; it does not change metal reflection. Specular strength uses texture
alpha in linear space and specular color texture RGB uses the sRGB transfer.
`KHR_materials_ior` supplies dielectric F0. Legacy procedural/OBJ materials and
the separate specular/glossiness workflow retain their existing models.

`KHR_materials_clearcoat` adds a separate GGX layer, with factor (texture R),
roughness (texture G), and its own normal map and scale. Without a coat normal
map, the coat uses the original interpolated surface normal, independently of
the base normal map. Layering uses the extension's documented NdotV Schlick
Fresnel approximation: coat weight `c*F`, base weight `1-c*F`. This simple
energy-limited approximation is not a full multiple-scattering layered material.
The BRDF evaluator, continuation density and direct-light MIS all include the
same finite lobes. Diffuse retains the HDR/cosine proposal. Zero-roughness base
and coat reflections are exact delta lobes, excluded from solid-angle NEE/PDFs;
visible delta/HDR connections are integrated once with their physical weights.
Finite GGX has a numerical alpha floor of 1e-4, applied only above zero roughness.

Integrator 10 additionally prevents an interpolated/normal-mapped reflective
frame from placing its ideal reflection below the actual triangle. It rotates
only the reflective frame towards the geometric normal as necessary; diffuse
keeps its authored frame. Base and coat are corrected independently. The same
corrected normals determine BRDF values, sampling and competing MIS densities.
This follows the valid-reflection policy used by Cycles, with a small horizon
margin. It fixes a reproduced dark inner-chrome/headlight case without changing
roughness, radiance or mesh data; it remains an approximate smooth-normal policy.

The BMW file requires clearcoat on **12 materials**, including CarPaint,
HeadLight, WheelRim and floor. Ignoring it removed their reflection layers, even
though all 114 meshes were imported. The layer and its specular controls are now
loaded rather than silently discarded. See the
[appearance audit](diagnostics/bmw-hdr-fireflies/README.md) for the supplied
reference's camera, display, denoising and conversion differences.

Support remains partial: clearcoat on emissive/transmitting materials, specular
overrides on transmission, thin/rough/partial transmission, alpha BLEND, vertex
colors, glTF cameras, skinning/morphing/animation, and other material extensions
are not complete. Required unknown extensions and the partial transmission/volume
model produce import diagnostics. Transmission without a nonzero volume thickness
is a thin surface in glTF; the current smooth solid dielectric interpretation
does not implement that distinction. Do not describe this as complete glTF glass
support or qualify production references from an appearance preview alone.

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

Imported transmission materials use the dielectric branch. Dielectric triangles
retain their boundary orientation so glass exits use the correct refraction
ratio; opaque triangles retain their two-sided facing convention. Geometric
normals determine dielectric medium membership. Smooth normals fall back to
the geometric interface when they would classify the wrong side or send a
branch into the wrong hemisphere. Glass uses exact unpolarized Fresnel and
radiance-mode eta-squared transmission, with eta-neutral roulette probabilities.
Precision-scaled ray offsets and a smaller traversal cutoff preserve thin
boundaries that the previous combined millimeter shift/cutoff skipped.
Visible terminal HDR branches are integrated with Fresnel weights; geometry
still blocks these connections, and sampled escapes are not counted twice.
See the [BMW glass investigation](diagnostics/bmw-hdr-fireflies/README.md).
Emissive
materials retain their existing emission behavior. Nonzero-roughness legacy metallic
surfaces, including procedural/OBJ metals, use the GGX helpers
with zero diffuse color, matched PDFs and direct-light sampling. Exact
zero-roughness metals remain delta mirrors. Alpha blending, vertex
colors, rough refraction and complete layered emissive/reflective materials are
not added here. Denoising albedo guides use the imported diffuse color/texture.

## Lights

For Bistro's small instanced geometry, front-banner material, and display
exposure corrections, see the [integrator-12 diagnosis](diagnostics/bistro-scene-fix.md).

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

The renderer integrator fingerprint version is now 10 following the reflective-frame correction, so existing render
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
