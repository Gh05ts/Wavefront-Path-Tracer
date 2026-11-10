#!/usr/bin/env python3
"""Read-only BMW feature/geometry audit; does not rewrite scene assets or plans."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np

repo = Path(__file__).resolve().parents[3]
asset = repo/'assets/gltf-research-scenes/scenes/bmw-m6'
path = asset/'gltf/bmw_m6_extended.gltf'
scene = json.loads(path.read_text())
output = repo.parent/'dataset/diagnostics/bmw-hdr-fireflies/appearance-investigation'
output.mkdir(exist_ok=True)
counts = Counter()
primitives = []
for i,mesh in enumerate(scene['meshes']):
    for primitive in mesh['primitives']:
        indices = scene['accessors'][primitive['indices']]
        supported = (primitive.get('mode',4)==4 and 'POSITION' in primitive['attributes']
                     and all(scene['accessors'][a]['componentType']==5126
                             and not scene['accessors'][a].get('sparse')
                             for a in primitive['attributes'].values()))
        triangles = indices['count']//3
        counts[primitive['material']] += triangles
        primitives.append({'mesh_index':i,'name':mesh['name'],'triangles':triangles,
                           'material':scene['materials'][primitive['material']]['name'],
                           'supported_geometry_encoding':supported})
materials = []
for i,m in enumerate(scene['materials']):
    materials.append({'index_in_renderer':i+1,'name':m['name'],'triangles':counts[i],
                      'metallic_roughness':m.get('pbrMetallicRoughness',{}),
                      'extensions':m.get('extensions',{}),
                      'alpha_mode':m.get('alphaMode','OPAQUE')})
visited = []
def visit(index):
    node=scene['nodes'][index]
    visited.append(index)
    for child in node.get('children',[]): visit(child)
for index in scene['scenes'][scene.get('scene',0)]['nodes']: visit(index)
camera_node = next(scene['nodes'][i] for i in visited if 'camera' in scene['nodes'][i])
matrix = np.array(camera_node['matrix']).reshape(4,4,order='F')
pb = (asset/'source/bmw-m6.pbrt').read_text()
source_paths = re.findall(r'"string filename"\s*\[\s*"(geometry/[^\"]+\.ply)"',pb)
result = {
    'scene':str(path.relative_to(repo)),
    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'required_extensions':scene.get('extensionsRequired',[]),
    'declared_meshes':len(scene['meshes']),
    'active_mesh_instances':sum('mesh' in scene['nodes'][i] for i in visited),
    'triangles':sum(p['triangles'] for p in primitives),
    'all_geometry_encodings_supported':all(p['supported_geometry_encoding'] for p in primitives),
    'textures':len(scene.get('textures',[])),
    'animations':len(scene.get('animations',[])),
    'skins':len(scene.get('skins',[])),
    'vertex_color_primitives':sum('COLOR_0' in p['attributes'] for m in scene['meshes'] for p in m['primitives']),
    'source_ply_references':len(source_paths),
    'source_ply_lfs_pointers':sum((asset/'source'/p).read_bytes().startswith(b'version https://git-lfs') for p in source_paths),
    'source_geometry_bytes_verified':False,
    'original_pbrt_reference_is_lfs_pointer':(asset/'source/bmw-m6.png').read_bytes().startswith(b'version https://git-lfs'),
    'reference_conversion_metadata':str((asset/'gltf/conversion.yaml').relative_to(repo)),
    'camera':{'perspective':scene['cameras'][camera_node['camera']]['perspective'],
              'world_matrix':camera_node['matrix'],
              'basis_determinant':float(np.linalg.det(matrix[:3,:3])),
              'renderer_imports_gltf_camera':False,
              'comparison_caution':'Exported camera has mirrored X basis; dataset also uses a different aspect and position. Do not change production camera plans to match a reference PNG.'},
    'pre_fix_required_feature_support':{
        'KHR_materials_clearcoat':'ignored: 12 BMW coated materials lose reflection layer',
        'KHR_materials_specular':'ignored: all 26 declare it; BMW opaque nonmetals use factor 0, coincidentally matching old Lambert base',
        'KHR_materials_transmission':'partial: nonzero factor selects solid smooth dielectric; no thin/rough/partial transmission',
        'KHR_materials_ior':'factor imported'},
    'materials':materials,'primitives':primitives,
}
(output/'asset-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('materials','primitives')},indent=2))
