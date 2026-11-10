#!/usr/bin/env python3
"""Serial checks and five bounded BMW captures in the original build only."""
import hashlib
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--integrator',type=int,choices=(9,10),default=10)
version=parser.parse_args().integrator
here=Path(__file__).resolve().parent
repo=here.parents[2]
workspace=repo.parent
build=Path('/tmp/ntasad-capture-build')
output=workspace/'dataset/diagnostics/bmw-hdr-fireflies/appearance-investigation'
output.mkdir(exist_ok=True)

def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
binary=build/'dataset_capture'
before=sha(binary)

def run(name,command,**kwargs):
    print('Running '+name,flush=True)
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,**kwargs)
    (output/(name+'.txt')).write_text(result.stdout)
    (output/(name+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
    print(result.stdout[-2500:],flush=True)
    if result.returncode: raise RuntimeError(name+' failed')
    if sha(binary)!=before: raise RuntimeError('Capture executable changed during validation')

# The existing validator checks CTest, source version, binary identity, and all
# four 320x180 capture manifests. This is diagnostic reference qualification only.
run(f'v{version}-renderer-and-replicas',[sys.executable,str(here/'run_hdr_gpu_validation.py'),'--integrator',str(version)],cwd=workspace)
env=dict(os.environ,WAVEFRONT_CAPTURE_EXECUTABLE=str(binary),WAVEFRONT_RENDERER_ROOT=str(repo),
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
files=['test_frames.py','test_sequences.py','test_dataset_splits.py','test_reference_convergence.py',
       'test_dataset_generation.py','test_capture_gpu.py']
run(f'v{version}-python',[sys.executable,'-m','pytest','-q',*[str(workspace/'Path-Tracer-Denoiser/tests'/f) for f in files],
    '--basetemp',str(build/f'python-capture-tests-v{version}')],env=env,cwd=workspace/'Path-Tracer-Denoiser')
run(f'v{version}-matched-camera',[str(binary),str(output/'gltf-camera-appearance.json'),str(output/f'gltf-camera-v{version}')],cwd=repo)
manifest=json.loads((output/f'gltf-camera-v{version}/frame_000000/manifest.json').read_text())
if manifest['render']['integrator_version']!=version: raise RuntimeError('Wrong appearance capture integrator')
run(f'v{version}-appearance-sheet',[sys.executable,str(here/'compare_appearance.py')],cwd=workspace)
pilots=[]
for path in sorted((workspace/'dataset/denoiser-v3').rglob('frame_*/manifest.json')):
    record=json.loads(path.read_text());image=path.parent/record['image']
    valid=sha(image)==record['image_sha256']
    if not valid or record['render']['integrator_version']!=4: raise RuntimeError('Native pilot changed')
    pilots.append({'manifest':str(path),'image_sha256':sha(image),'integrator':4,'verified':valid})
if len(pilots)!=2: raise RuntimeError('Expected exactly the two existing native pilots')
watched=['include/scene/material.cuh','include/renderer/gltf_pbr.cuh','src/scene/gltf_loader.cpp',
         'include/renderer/device_traversal.cuh','include/renderer/device_path_shading.cuh','include/renderer/render_fingerprint.hpp']
ctest=(output.parent/'hdr-spot-investigation'/f'ctest-v{version}.txt').read_text()
pytest=(output/f'v{version}-python.txt').read_text()
renderer_count=re.search(r'100% tests passed.*out of (\d+)',ctest)
python_count=re.search(r'(\d+) passed, (\d+) skipped',pytest)
if renderer_count is None or python_count is None or '***Skipped' in ctest:
    raise RuntimeError('Cannot assert all renderer/Python check counts from the logs')
(output/f'v{version}-validation.json').write_text(json.dumps({'integrator':version,'capture_sha256':before,
    'source_sha256':{p:sha(repo/p) for p in watched},'native_pilots':pilots,
    'production_qualified':False,'bulk_generation_started':False,'training_started':False,
    'test_counts':{'renderer_ctest_passed':int(renderer_count[1]),'python_passed':int(python_count[1]),
                  'python_optional_skipped':int(python_count[2])}},indent=2)+'\n')
