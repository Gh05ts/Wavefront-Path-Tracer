#!/usr/bin/env python3
"""Bounded BMW glass replay using the existing build; no GPU or bulk capture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", type=Path, default=Path("/tmp/ntasad-capture-build"))
parser.add_argument("--label", required=True)
args = parser.parse_args()
here = Path(__file__).resolve().parent
repo = here.parents[2]
root = repo.parent / "dataset/diagnostics/bmw-hdr-fireflies"
output = root / "glass-investigation" / args.label
output.mkdir(parents=True, exist_ok=True)
objects = args.build / "CMakeFiles/dataset_capture.dir/src/scene"
executable = args.build.resolve() / "bmw-glass-replay"
command = ["c++", "-O2", "-std=c++17", "-I/usr/local/cuda/include",
           "-I" + str(repo / "extern/NexusBVH/NexusBVH/include"),
           "-I" + str(repo / "extern/tinygltf"), "-include", "cuda_runtime.h",
           "-I" + str(repo / "include"), str(here / "cpu_replay.cpp"),
           "-x", "c++", str(repo / "src/core/rng.cu"), "-x", "none",
           *[str(objects / (name + ".cpp.o")) for name in ("gltf_loader", "obj_loader", "environment", "bvh")],
           "-lstdc++fs", "-o", str(executable)]
subprocess.run(command, check=True)
(output / "compile-command.json").write_text(json.dumps(command, indent=2) + "\n")
watched = [repo / name for name in (
    "include/core/ray.cuh", "include/renderer/path_state.cuh",
    "include/renderer/device_sampling.cuh", "include/renderer/device_traversal.cuh",
    "include/renderer/device_path_shading.cuh", "include/renderer/render_fingerprint.hpp")]
(output / "source-sha256.json").write_text(json.dumps(
    {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in watched},
    indent=2) + "\n")
scene = repo / "assets/gltf-research-scenes/scenes/bmw-m6/gltf/bmw_m6_extended.gltf"
hdr = scene.parents[2] / "common/sunflowers_environment.hdr"
for x, y in [(166,38),(219,77),(204,50),(189,83),(142,47),(208,53),(242,47),(189,82)]:
    for replica in (0,1):
        manifest = root / f"bmw-v4-reference-check/captures/bmw_v4_first/s4096/r{replica}/frame_000000/manifest.json"
        start = json.loads(manifest.read_text())["render"]["reference_sample_start"]
        command = [str(executable), str(scene), str(hdr), str(manifest), "4096", str(start), str(x), str(y)]
        (output / f"pixel-{x}-{y}-r{replica}-command.json").write_text(json.dumps(command, indent=2) + "\n")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        (output / f"pixel-{x}-{y}-r{replica}.txt").write_text(result.stdout)
        print(result.stdout, flush=True)

# Run the actual asset-free transport regression kernel as host C++ as well.
# HDR texture-dependent tests remain in the CUDA environment test executable.
transport_source = args.build / "bmw-glass-transport-cpu.cpp"
scaffold = """#include <algorithm>
using std::min; using std::max;
struct FakeIndex{unsigned x=0;} blockIdx,blockDim,threadIdx;
int __popc(unsigned x){return __builtin_popcount(x);}
template<class T>T tex2D(cudaTextureObject_t,float,float){return T{};}
"""
scaffold += (repo / "tests/material_transport_tests.cu").read_text().split("int main()")[0]
scaffold += '\nint main(){int failure=-1;checkMaterialTransport(&failure);std::cout<<"failure="<<failure<<"\\n";return failure;}\n'
transport_source.write_text(scaffold)
transport_executable = args.build / "bmw-glass-transport-cpu"
subprocess.run(["c++", "-O2", "-std=c++17", "-I/usr/local/cuda/include",
                "-I" + str(repo / "extern/NexusBVH/NexusBVH/include"),
                "-I" + str(repo / "include"), "-include", "cuda_runtime.h", str(transport_source),
                "-x", "c++", str(repo / "src/core/rng.cu"), "-o", str(transport_executable)], check=True)
result = subprocess.run([str(transport_executable)], capture_output=True, text=True, check=True)
(output / "transport-regressions.txt").write_text(result.stdout)
print(result.stdout, flush=True)
