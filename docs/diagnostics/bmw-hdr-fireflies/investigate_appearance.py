#!/usr/bin/env python3
"""Bounded CPU replay of changed windshield means, in the existing build only."""
import hashlib
import json
import re
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
repo = here.parents[2]
build = Path('/tmp/ntasad-capture-build')
root = repo.parent / 'dataset/diagnostics/bmw-hdr-fireflies'
output = root / 'appearance-investigation'
output.mkdir(exist_ok=True)
version = int(re.search(r'rendererIntegratorVersion\s*=\s*(\d+)',
    (repo/'include/renderer/render_fingerprint.hpp').read_text()).group(1))
if version != 8:
    raise RuntimeError('This controlled v8 proposal ablation requires integrator-8 headers and matching host objects; do not run against a changed material ABI.')

# Toggle the proposal only. BSDFs, glass, ray offsets, geometric rejection and
# the filtered HDR table remain identical. This is not a production alteration.
header = (repo / 'include/renderer/device_traversal.cuh').read_text()
needle = 'map.probabilities && map.alias ? 0.5f : 0.0f;'
assert header.count(needle) == 1
modified = 'float diagnosticProposalProbability = .5f;\n' + header.replace(
    needle, 'map.probabilities && map.alias ? diagnosticProposalProbability : 0.0f;')
temporary_header = build / 'bmw-appearance-traversal.cuh'
temporary_header.write_text(modified)
source = (here / 'cpu_replay.cpp').read_text().replace(
    '"renderer/device_traversal.cuh"', '"bmw-appearance-traversal.cuh"')
source = source.replace('mode!="no-split")', 'mode!="no-split" && mode!="cosine")')
source = source.replace('diagnosticNoTerminalSplit=mode=="no-split";',
    'diagnosticNoTerminalSplit=mode=="no-split";\n'
    '    diagnosticProposalProbability=mode=="cosine"?0.f:.5f;')
temporary_source = build / 'bmw-appearance-replay.cpp'
temporary_source.write_text(source)
executable = build / 'bmw-appearance-replay'
objects = build / 'CMakeFiles/dataset_capture.dir/src/scene'
command = ['c++', '-O2', '-std=c++17', '-I/usr/local/cuda/include', '-I'+str(repo/'include/renderer'),
    '-I'+str(repo/'extern/NexusBVH/NexusBVH/include'), '-I'+str(repo/'extern/tinygltf'),
    '-include', 'cuda_runtime.h', '-I'+str(repo/'include'), str(temporary_source),
    '-x', 'c++', str(repo/'src/core/rng.cu'), '-x', 'none',
    *[str(objects/(name+'.cpp.o')) for name in ('gltf_loader','obj_loader','environment','bvh')],
    '-lstdc++fs', '-o', str(executable)]
subprocess.run(command, check=True)
(output/'compile-command.json').write_text(json.dumps(command, indent=2)+'\n')
(output/'replay-source-sha256.json').write_text(json.dumps({
    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
    [here/'cpu_replay.cpp', repo/'include/renderer/device_traversal.cuh',
     repo/'include/renderer/device_sampling.cuh', repo/'include/renderer/device_path_shading.cuh',
     temporary_header, temporary_source, executable,
     *[objects/(n+'.cpp.o') for n in ('gltf_loader','obj_loader','environment','bvh')]]
}, indent=2)+'\n')
scene = repo/'assets/gltf-research-scenes/scenes/bmw-m6/gltf/bmw_m6_extended.gltf'
hdr = scene.parents[2]/'common/sunflowers_environment.hdr'
manifest = root/'bmw-v8-hdr-reference-check/captures/bmw_v8_hdr_first/s4096/r0/frame_000000/manifest.json'
jobs = [(185,53,'cosine',262144), (185,53,'original',262144),
        (206,52,'cosine',262144), (206,52,'original',262144),
        (185,53,'no-sun',65536)]
for x,y,mode,count in jobs:
    command = [str(executable),str(scene),str(hdr),str(manifest),str(count),
               '800010260',str(x),str(y),'5',mode]
    name = f'pixel-{x}-{y}-{mode}-{count}'
    (output/(name+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
    result = subprocess.check_output(command,text=True)
    (output/(name+'.txt')).write_text(result)
    print(result.split(' path: ')[0],flush=True)
