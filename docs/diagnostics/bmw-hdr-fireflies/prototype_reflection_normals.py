#!/usr/bin/env python3
"""CPU-only valid-reflection-normal ablation; no production shading edit."""
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
header=(repo/'include/renderer/device_traversal.cuh').read_text()
correction='''
// Diagnostic smooth-normal correction. Rotate towards the geometric normal
// only if the ideal reflected ray is below a small geometric-side threshold.
// Recompute the BSDF and full proposal with this same normal; never resample.
Vec3 diagnosticReflectionNormal(Vec3 n,Vec3 geometric,Vec3 view){
    float iz=dot(geometric,view),target=fminf(.01f,.9f*iz);
    if(iz<=0)return geometric;
    auto valid=[&](Vec3 candidate){return dot(geometric,2.f*dot(candidate,view)*candidate-view)>=target;};
    if(valid(n))return n;
    float lower=0,upper=1;
    for(int i=0;i<24;++i){float t=(lower+upper)*.5f;Vec3 candidate=normalize(n*(1-t)+geometric*t);if(valid(candidate))upper=t;else lower=t;}
    return normalize(n*(1-upper)+geometric*upper);
}
void diagnosticCorrectSurface(GltfPbrSurface& s,Vec3 geometric,Vec3 view){
    if(s.metallic>=1.f)s.normal=diagnosticReflectionNormal(s.normal,geometric,view);
    if(s.coat>0)s.coatNormal=diagnosticReflectionNormal(s.coatNormal,geometric,view);
}
'''
header=header.replace('__device__ Vec3 evaluateSurfaceCos(',correction+'\n__device__ Vec3 evaluateSurfaceCos(')
needle='auto surface=getGltfPbrSurface(material,hit,shadedHit,scene);'
assert header.count(needle)==2
# evaluateSurfaceCos has a view parameter; continuation defines it on the next line.
header=header.replace(needle+'\n        return',needle+'\n        diagnosticCorrectSurface(surface,hit.geometricNormal,view);\n        return')
needle='Vec3 view=-path.ray.direction;'
assert header.count(needle)==1
header=header.replace(needle,needle+'\n    diagnosticCorrectSurface(surface,hit.geometricNormal,view);')
(build/'bmw-normal-traversal.cuh').write_text(header)
source=(here/'cpu_replay.cpp').read_text().replace('"renderer/device_traversal.cuh"','"bmw-normal-traversal.cuh"')
(build/'bmw-normal-replay.cpp').write_text(source)
source=(here/'appearance_tiles.cpp').read_text().replace('"cpu_replay.cpp"','"bmw-normal-replay.cpp"')
(build/'bmw-normal-tiles.cpp').write_text(source)
command=json.loads((output/'tiles-compile-command.json').read_text())
command.insert(1,'-I'+str(repo/'include/renderer'))
command[command.index(str(here/'appearance_tiles.cpp'))]=str(build/'bmw-normal-tiles.cpp')
executable=build/'bmw-normal-tiles'
command[-1]=str(executable)
subprocess.run(command,check=True)
(output/'normal-prototype-compile-command.json').write_text(json.dumps(command,indent=2)+'\n')
for name in ['bmw-normal-traversal.cuh','bmw-normal-replay.cpp','bmw-normal-tiles.cpp']:
    (output/name).write_bytes((build/name).read_bytes())
(output/'normal-prototype-sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
    repo/'include/renderer/device_traversal.cuh',here/'appearance_tiles.cpp',here/'cpu_replay.cpp',
    build/'bmw-normal-traversal.cuh',build/'bmw-normal-replay.cpp',build/'bmw-normal-tiles.cpp',executable]},indent=2)+'\n')
scene=repo/'assets/gltf-research-scenes/scenes/bmw-m6/gltf/bmw_m6_extended.gltf'
hdr=scene.parents[2]/'common/sunflowers_environment.hdr'
command=[str(executable),str(scene),str(hdr),str(output/'gltf-camera-v9/frame_000000/manifest.json'),
         'solid','128','293','225','132','60',str(output/'tile-v9-headlight-reflection-normal')]
(output/'normal-prototype-render-command.json').write_text(json.dumps(command,indent=2)+'\n')
with (output/'normal-prototype-render.txt').open('w') as stream:
    subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
print('Completed CPU headlight reflection-normal ablation',flush=True)
