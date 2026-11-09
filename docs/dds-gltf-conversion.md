# DDS textures in glTF assets

The current glTF loader decodes images with stb_image, which does not decode DDS.
For the Niagara Bistro asset, use the offline conversion utility:

```bash
python -m pip install 'Pillow>=11'
python tools/convert_dds_gltf.py assets/niagara_bistro/bistro.gltf
```

This writes PNGs next to the DDS textures and creates `bistro-png.gltf` beside
the original. The original glTF, geometry buffers, DDS files and license remain
intact. Point the renderer scene manifest's `object.file` at the derived glTF.
The utility fills missing PNG fallbacks, replaces DDS image URIs, removes
`MSFT_texture_dds` texture sources, and remaps duplicate image entries. This is
necessary because the loader visits every image entry, including unused DDS
alternatives. It refuses to overwrite an existing derived glTF; existing image
files are reused.

Bistro's DDS files use BC7 UNORM / UNORM_SRGB (DXGI 98/99), supported by the
Pillow decoder in the conversion environment. Conversion preserves decoded RGBA
bytes and verifies a PNG round trip. It applies no additional gamma transform,
channel swizzle, vertical flip or resize. PNG adds no further lossy compression,
but cannot recover information lost when the original textures were BC7 encoded.
Only the base mip is exported. The renderer continues to interpret color/data
textures according to their material usage.

This resolves image decoding, not complete Bistro appearance compatibility.
The renderer now imports `KHR_materials_pbrSpecularGlossiness` and
`KHR_lights_punctual`; see [material/light support and its limits](gltf-materials-and-lights.md).
Review materials and lighting in a small preview before dataset capture.

Native DDS support would require selecting `MSFT_texture_dds` sources and adding
a BC7-capable decoder or compressed-texture upload path. Since the current
renderer stores decoded RGBA texels, offline PNG conversion fits its existing
texture pipeline without changing rendering code.
