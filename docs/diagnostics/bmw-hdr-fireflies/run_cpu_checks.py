#!/usr/bin/env python3
"""Replay BMW problem pixels and test the proposed fix without using a GPU.

Requires the existing dataset_capture build's host object files and CUDA headers.
All candidate source changes/builds occur in a temporary directory. The patch
is never applied to the renderer checkout. Results go to --output.
"""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(command, *, cwd=None, log=None, expected=0):
    result = subprocess.run([str(x) for x in command], cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log:
        log.write_text(result.stdout)
    if result.returncode != expected:
        raise RuntimeError(f"Command returned {result.returncode}, expected {expected}: "
                           f"{command}\n{result.stdout}")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="95eb205", help="Pre-fix renderer revision")
    parser.add_argument("--build", type=Path, default=Path("/tmp/ntasad-capture-build"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    workspace = repo.parent
    output = (args.output or workspace / "dataset/diagnostics/bmw-hdr-fireflies").resolve()
    output.mkdir(parents=True, exist_ok=True)
    build = args.build.resolve()
    object_dir = build / "CMakeFiles/dataset_capture.dir/src/scene"
    objects = [object_dir / (name + ".cpp.o") for name in
               ("gltf_loader", "obj_loader", "environment", "bvh")]
    manifest = workspace / "dataset/denoiser-v2/bmw_view_00_linear_slow/frame_000000/manifest.json"
    frame = manifest.with_name("frame.exr")
    scene = repo / "assets/gltf-research-scenes/scenes/bmw-m6/gltf/bmw_m6_extended.gltf"
    hdr = repo / "assets/gltf-research-scenes/scenes/common/sunflowers_environment.hdr"
    watched = [build / "dataset_capture", scene, hdr, frame,
               repo / "include/renderer/device_traversal.cuh",
               repo / "include/renderer/device_path_shading.cuh"]
    before = {p: sha(p) for p in watched}
    with tempfile.TemporaryDirectory(prefix="bmw-firefly-check-") as temporary:
        candidate = Path(temporary)
        baseline = candidate / "baseline"
        baseline.mkdir()
        shutil.copytree(repo / "include", baseline / "include")
        for relative in ("include/renderer/device_sampling.cuh", "include/renderer/device_traversal.cuh", "include/renderer/device_path_shading.cuh",
                         "include/renderer/render_fingerprint.hpp"):
            text = run(["git", "show", args.baseline_ref + ":" + relative], cwd=repo)
            (baseline / relative).write_text(text)
        shutil.copytree(baseline / "include", candidate / "include")
        for relative in ("src/renderer/photon_mapping.cu", "tests/CMakeLists.txt"):
            target = candidate / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(run(["git", "show", args.baseline_ref + ":" + relative], cwd=repo))
        run(["git", "init", "--quiet"], cwd=candidate)
        run(["git", "apply", here / "proposed-fix.patch"], cwd=candidate)
        common = ["c++", "-O2", "-std=c++17", "-I/usr/local/cuda/include",
                  "-I" + str(repo / "extern/NexusBVH/NexusBVH/include"),
                  "-I" + str(repo / "extern/tinygltf"), "-include", "cuda_runtime.h"]
        rng = repo / "src/core/rng.cu"
        for name, include in (("baseline", baseline / "include"), ("candidate", candidate / "include")):
            executable = candidate / (name + "-replay")
            run(common + ["-I" + str(include), here / "cpu_replay.cpp", "-x", "c++", rng,
                          "-x", "none", *objects, "-lstdc++fs", "-o", executable],
                log=output / (name + "-compile.log"))
            for spp in (1024, 16384):
                text = run([executable, scene, hdr, manifest, spp],
                           log=output / f"{name}-{spp}spp.txt")
                print(text, flush=True)
            regression = (here / "material_transport_cpu.cpp").read_text()
            if name == "baseline":
                regression = regression.replace("continueGlossySurface(", "continueSpecGloss(")
            source = candidate / (name + "-checks.cpp")
            source.write_text(regression)
            executable = candidate / (name + "-checks")
            run(common + ["-I" + str(include), source, "-x", "c++", rng, "-o", executable])
            # Failure 2 on old code is the glass exit-normal regression.
            print(name, run([executable], expected=2 if name == "baseline" else 0,
                            log=output / (name + "-checks.txt")).strip(), flush=True)
    after = {p: sha(p) for p in watched}
    if before != after:
        raise RuntimeError("A watched capture input changed during diagnosis")
    (output / "preserved-inputs.txt").write_text(
        "\n".join(f"{value}  {path}" for path, value in after.items()) + "\n")
    print(f"CPU checks complete. Active binary, selected assets, frame and renderer headers preserved.\n{output}")


if __name__ == "__main__":
    main()
