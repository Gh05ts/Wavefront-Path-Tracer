#!/usr/bin/env python3
"""CPU-only sampling experiment in the existing build, without renderer edits."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
repo = here.parents[2]
build = Path('/tmp/ntasad-capture-build')
output = repo.parent / 'dataset/diagnostics/bmw-hdr-fireflies/hdr-spot-investigation/continuation-prototype'
output.mkdir(parents=True, exist_ok=True)
lock = json.loads((output / 'v5-source-sha256.json').read_text())
for name, digest in lock.items():
    if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Archived v5 source hash mismatch: '+name)
traversal = (output / 'v5-device_traversal.cuh').read_text()
guided_traversal = traversal.replace(
    'pdf = fmaxf(0.0f, dot(hit.normal, incoming)) * 0.31830988618f;',
    '''pdf = fmaxf(0.0f, dot(hit.normal, incoming)) * 0.31830988618f;
    if (scene.environment.texture) pdf = .5f * pdf + .5f * environmentPdf(scene.environment, incoming);''')
assert guided_traversal != traversal
shading = (output / 'v5-device_path_shading.cuh').read_text()
start = 'Vec3 samplingNormal = shadedHit.normal;'
end = 'path.specularBounce = false;'
guided_shading = shading
for indent in (20, 8):
    begin = guided_shading.index(start)
    finish = guided_shading.index(end, begin) + len(end)
    guided_shading = guided_shading[:begin] + '''if (scene.environment.texture && randomFloat(path.rngState) < .5f) {
        float a=randomFloat(path.rngState), b=randomFloat(path.rngState);
        float coin=randomFloat(path.rngState), u=randomFloat(path.rngState), v=randomFloat(path.rngState);
        direction=sampleEnvironment(scene.environment,double(a)+double(b)/16777216.0,coin,u,v).direction;
    } else direction=sampleCosineHemisphere(shadedHit.normal,path.rngState);
    float cosine=fmaxf(0.f,dot(shadedHit.normal,direction));
    float pdf=cosine*.31830988618f;
    if(scene.environment.texture)pdf=.5f*pdf+.5f*environmentPdf(scene.environment,direction);
    if(cosine<=0 || dot(direction,hit.geometricNormal)<=0 || pdf<=0)path.active=false;
    else path.throughput=hadamard(path.throughput,albedo)*(cosine*.31830988618f/pdf);
    path.previousBsdfPdf=pdf;
    path.specularBounce=false;''' + guided_shading[finish:]
for name, old, new in [('traversal', traversal, guided_traversal), ('shading', shading, guided_shading)]:
    (build / f'bmw-guided-{name}.cuh').write_text(new)
    (output / f'{name}.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True))))
source = (output / 'v5-cpu_replay.cpp').read_text().replace('"renderer/device_traversal.cuh"', '"bmw-guided-traversal.cuh"').replace('"renderer/device_path_shading.cuh"', '"bmw-guided-shading.cuh"')
cpp = build / 'bmw-guided-replay.cpp'
cpp.write_text(source)
command = json.loads((output.parents[1] / 'glass-investigation/v5-final-cpu/compile-command.json').read_text())
command[command.index(str(here / 'cpu_replay.cpp'))] = str(cpp)
command[-1] = str(build / 'bmw-guided-replay')
command.insert(1, '-I'+str(repo / 'include/renderer'))
command.insert(1, '-I'+str(repo / 'extern/stb'))
command[command.index(str(build / 'CMakeFiles/dataset_capture.dir/src/scene/environment.cpp.o'))] = str(output / 'v5-environment.cpp')
subprocess.run(command, check=True)
(output / 'compile-command.json').write_text(json.dumps(command, indent=2)+'\n')
scene = repo / 'assets/gltf-research-scenes/scenes/bmw-m6/gltf/bmw_m6_extended.gltf'
hdr = scene.parents[2] / 'common/sunflowers_environment.hdr'
root = output.parents[1]
for x,y in [(131,95), (105,90), (129,89), (166,38), (207,46), (206,52), (189,83), (242,47)]:
    for replica in (0,1):
        manifest = root / f'bmw-v5-glass-reference-check/captures/bmw_v5_glass_first/s4096/r{replica}/frame_000000/manifest.json'
        start = json.loads(manifest.read_text())['render']['reference_sample_start']
        command = [str(build/'bmw-guided-replay'),str(scene),str(hdr),str(manifest),'4096',str(start),str(x),str(y)]
        result = subprocess.check_output(command, text=True)
        (output / f'pixel-{x}-{y}-r{replica}.txt').write_text(result)
        print(result.split(' path: ')[0], flush=True)
