#!/usr/bin/env python3
"""Compile one small CPU diagnostic in the existing build; render bounded tiles."""
import hashlib
import json
from pathlib import Path
import subprocess

here=Path(__file__).resolve().parent
repo=here.parents[2]
build=Path('/tmp/ntasad-capture-build')
if 'rendererIntegratorVersion = 9;' not in (repo/'include/renderer/render_fingerprint.hpp').read_text():
    raise RuntimeError('This archived integrator-9 ablation requires v9 sources; current source would mislabel the result.')
output=repo.parent/'dataset/diagnostics/bmw-hdr-fireflies/appearance-investigation'
objects=build/'CMakeFiles/dataset_capture.dir/src/scene'
executable=build/'bmw-appearance-tiles'
command=['c++','-O2','-std=c++17','-I/usr/local/cuda/include',
    '-I'+str(repo/'extern/NexusBVH/NexusBVH/include'),'-I'+str(repo/'extern/tinygltf'),
    '-include','cuda_runtime.h','-I'+str(repo/'include'),str(here/'appearance_tiles.cpp'),
    '-x','c++',str(repo/'src/core/rng.cu'),'-x','none',
    *[str(objects/(name+'.cpp.o')) for name in ('gltf_loader','obj_loader','environment','bvh')],
    '-lstdc++fs','-o',str(executable)]
subprocess.run(command,check=True)
(output/'tiles-compile-command.json').write_text(json.dumps(command,indent=2)+'\n')
def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
paths=[here/'appearance_tiles.cpp',here/'cpu_replay.cpp',repo/'include/renderer/gltf_pbr.cuh',
       repo/'include/scene/material.cuh',repo/'include/renderer/device_traversal.cuh',
       repo/'include/renderer/device_path_shading.cuh',executable,
       *[objects/(name+'.cpp.o') for name in ('gltf_loader','obj_loader','environment','bvh')]]
(output/'tiles-provenance.json').write_text(json.dumps({str(p):sha(p) for p in paths},indent=2)+'\n')
scene=repo/'assets/gltf-research-scenes/scenes/bmw-m6/gltf/bmw_m6_extended.gltf'
hdr=scene.parents[2]/'common/sunflowers_environment.hdr'
manifest=output/'gltf-camera-v8/frame_000000/manifest.json'
for name,spp,box in [('headlight',128,(293,225,132,60)),('windshield',64,(255,85,260,77))]:
    for mode in ('solid','thin'):
        prefix=output/f'tile-v9-{name}-{mode}'
        command=[str(executable),str(scene),str(hdr),str(manifest),mode,str(spp),
                 *map(str,box),str(prefix)]
        (output/(prefix.name+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
        print('Running '+prefix.name,flush=True)
        result=subprocess.check_output(command,text=True,stderr=subprocess.STDOUT)
        (output/(prefix.name+'.txt')).write_text(result)
        print('Completed '+prefix.name,flush=True)
