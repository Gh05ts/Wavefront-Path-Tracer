#!/usr/bin/env python3
"""Read-only BMW glass winding/topology audit, welding identical positions."""
import json
from pathlib import Path
import numpy as np

repo=Path(__file__).resolve().parents[3]
path=repo / "assets/gltf-research-scenes/scenes/bmw-m6/gltf/bmw_m6_extended.gltf"
scene=json.loads(path.read_text())
glass={i for i,m in enumerate(scene["materials"]) if "KHR_materials_transmission" in m.get("extensions",{})}
buffers=[np.memmap(path.parent/b["uri"],dtype=np.uint8,mode="r") for b in scene["buffers"]]
def read(index):
    accessor=scene["accessors"][index];view=scene["bufferViews"][accessor["bufferView"]]
    dtype={5126:"<f4",5125:"<u4",5123:"<u2",5121:"u1"}[accessor["componentType"]]
    size=np.dtype(dtype).itemsize;channels={"SCALAR":1,"VEC3":3}[accessor["type"]]
    return np.ndarray((accessor["count"],channels),dtype=dtype,buffer=buffers[view["buffer"]],
        offset=view.get("byteOffset",0)+accessor.get("byteOffset",0),
        strides=(view.get("byteStride",channels*size),size))
results=[]
for mesh in scene["meshes"]:
    for primitive in mesh["primitives"]:
        if primitive.get("material") not in glass: continue
        positions=read(primitive["attributes"]["POSITION"]);indices=read(primitive["indices"]).reshape(-1,3)
        welded,remap=np.unique(positions,axis=0,return_inverse=True);faces=remap[indices]
        edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]);keys=np.sort(edges,axis=1)
        _,inverse,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
        signs=np.where(edges[:,0]<edges[:,1],1,-1);sums=np.bincount(inverse,weights=signs)
        vertices=positions[indices].astype(np.float64)
        geometric=np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0])
        normals=read(primitive["attributes"]["NORMAL"])[indices].mean(axis=1)
        boundary=int(np.sum(counts==1));nonmanifold=int(np.sum(counts>2))
        center=vertices.reshape(-1,3).mean(axis=0);relative=vertices-center
        volume=float(np.sum(np.einsum("ij,ij->i",relative[:,0],np.cross(relative[:,1],relative[:,2])))/6)
        result={"mesh":mesh.get("name"),"material":scene["materials"][primitive["material"]]["name"],
                "triangles":len(faces),"boundary_edges":boundary,"nonmanifold_edges":nonmanifold,
                "same_winding_paired_edges":int(np.sum((counts==2)&(sums!=0))),
                "vertex_normals_opposing_geometry":int(np.sum(np.einsum("ij,ij->i",geometric,normals)<0)),
                "signed_volume_local_if_closed":volume if boundary==0 and nonmanifold==0 else None,
                "bounds_local":[positions.min(axis=0).tolist(),positions.max(axis=0).tolist()]}
        results.append(result);print(result)
output=repo.parent / "dataset/diagnostics/bmw-hdr-fireflies/glass-investigation/glass-topology.json"
output.write_text(json.dumps(results,indent=2)+"\n")
