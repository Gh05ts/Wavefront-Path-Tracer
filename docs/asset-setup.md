# Recreating the scene assets

The two asset repositories are pinned Git submodules. Renderer changes and scene
manifests belong to this repository. The upstream research scenes need their LFS
payloads downloaded; Bistro needs its DDS textures converted to PNG. No fork of
either upstream repository is required for these preparation steps.

From a fresh clone:

```sh
git clone https://github.com/Gh05ts/Wavefront-Path-Tracer.git
cd Wavefront-Path-Tracer
python3 -m venv .venv
. .venv/bin/activate
python -m pip install Pillow==13.0.0
python tools/setup_assets.py
git submodule update --init extern/NexusBVH
```

The setup script initializes the two asset submodules, selects BMW M6, Bathroom,
Breakfast Room, Classroom2, Living Room and the shared environment folder, then
downloads the 12 required runtime LFS payloads. Git LFS is optional: the script
downloads from the immutable upstream revision and verifies each file's size and
SHA-256 using `tools/asset_sources.json`. Source-format LFS payloads remain
pointers because the renderer does not use them. Scene licenses are retained.

Bistro's `bistro.gltf`, geometry and DDS files remain intact. The script creates
PNG textures and `bistro-png.gltf`, which the committed scene manifest selects.
It uses the existing [DDS conversion utility](dds-gltf-conversion.md). The Pillow
version above matches the original conversion environment; preparation needs
several GB of disk space and internet access on the first run.

Rerun `python tools/setup_assets.py` to finish an interrupted preparation. Valid
files are verified and reused without rewriting them. Missing PNGs are converted;
downloads are published only after content verification. Unexpected existing
file contents, edited Bistro sources or a different submodule revision cause an
error, preserving those files for inspection. To check without making changes:

```sh
python tools/setup_assets.py --verify-only
```

Generated Bistro files may appear as untracked content inside its submodule.
They do not need to be committed there: this script recreates them. Commit this
script, the source manifest, converter changes and documentation in the parent
repository, together with `.gitmodules` and the two submodule gitlinks. Push that
parent commit before expecting a new clone to contain the setup workflow.

This recreates the scene preparation; it does not copy rendered datasets or
training checkpoints. Fresh machines also need their own renderer build and
denoiser environment. Existing capture jobs lock exact renderer and asset bytes;
keep their current prepared assets when resuming those jobs.
