#!/usr/bin/env python3
"""Record BMW HDR/glass mode differences and final-binary repeatability."""
import json
import hashlib
from pathlib import Path
import subprocess
import numpy as np
from wavefront_denoiser.frames import load_frame

repo=Path(__file__).resolve().parents[3]
root=repo.parent / "dataset/diagnostics/bmw-hdr-fireflies"
output=root / "glass-investigation/gpu-modes"
output.mkdir(parents=True,exist_ok=True)
binary=Path("/tmp/ntasad-capture-build/dataset_capture")
with binary.open("rb") as stream:
    starting_sha=hashlib.file_digest(stream,"sha256").hexdigest()
provenance=output/"binary-provenance.json"
if provenance.exists():
    if json.loads(provenance.read_text())["capture_binary_sha256"]!=starting_sha:
        raise RuntimeError("Existing mode captures belong to a different binary; preserve them")
else:
    if any(output.glob("*/frame_000000/manifest.json")):
        raise RuntimeError("Existing mode captures lack binary provenance; preserve them")
    provenance.write_text(json.dumps({"capture_binary_sha256":starting_sha},indent=2)+"\n")
base=json.loads((root / "bmw-v4-reference-check/capture_configs/reference_bmw_v4_first_s4096_r0.json").read_text())
base.update(width=160,height=90,reference_samples=128,tile_width=64,tile_height=64)
frames=[]
for mode in ("standard","persistent"):
    config=dict(base,sequence_id="glass_mode_"+mode,persistent_wavefront=mode=="persistent")
    filename=output / (mode+".json")
    if filename.exists() and json.loads(filename.read_text())!=config:
        raise RuntimeError("Existing mode configuration differs; preserve its capture")
    filename.write_text(json.dumps(config,indent=2)+"\n")
    if not (output / mode / "frame_000000/manifest.json").exists():
        with (output / (mode+".log")).open("w") as log:
            subprocess.run(["/tmp/ntasad-capture-build/dataset_capture",str(filename),str(output/mode)],
                           cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
    frame=load_frame(output / mode / "frame_000000")
    assert all(np.isfinite(value).all() for value in frame.buffers.values())
    frames.append(frame)
report={}
for name,a in frames[0].buffers.items():
    b=frames[1].buffers[name]
    difference=a.astype(np.float64)-b
    if name not in ("radiance","reference"):
        np.testing.assert_array_equal(a,b,err_msg=name)
    report[name]={"max_absolute_difference":float(np.max(np.abs(difference))),
                  "rms_difference":float(np.sqrt(np.mean(difference**2))),
                  "strict_allclose_rtol_1e-5_atol_1e-6":bool(np.allclose(a,b,rtol=1e-5,atol=1e-6))}
(output/"results.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))

# The last source refinement concerns finite-light shadow endpoints. Recheck
# one full calibration replica with the final binary and report repeatability.
# Complex-asset secondary visibility can differ even with identical sample IDs;
# do not mislabel this diagnostic as an exact-bit certificate.
config=root / "bmw-v5-glass-reference-check/capture_configs/reference_bmw_v5_glass_first_s4096_r0.json"
if not (output / "final-binary-check/frame_000000/manifest.json").exists():
    with (output / "final-binary-check.log").open("w") as log:
        subprocess.run(["/tmp/ntasad-capture-build/dataset_capture",str(config),str(output/"final-binary-check")],
                       cwd=repo,stdout=log,stderr=subprocess.STDOUT,check=True)
actual=load_frame(output / "final-binary-check/frame_000000")
expected=load_frame(root / "bmw-v5-glass-reference-check/captures/bmw_v5_glass_first/s4096/r0/frame_000000")
repeatability={}
for name,a in actual.buffers.items():
    b=expected.buffers[name]
    if name!="reference":
        np.testing.assert_array_equal(a,b,err_msg=name)
    difference=a.astype(np.float64)-b
    repeatability[name]={"identical":bool(np.array_equal(a,b)),
                         "max_absolute_difference":float(np.max(np.abs(difference))),
                         "rms_difference":float(np.sqrt(np.mean(difference**2)))}
with binary.open("rb") as stream:
    binary_sha=hashlib.file_digest(stream,"sha256").hexdigest()
if binary_sha!=starting_sha:
    raise RuntimeError("Capture executable changed during mode validation")
(output/"final-binary-check.json").write_text(json.dumps({
    "all_buffers_identical":all(value["identical"] for value in repeatability.values()),
    "buffers":repeatability,"integrator_version":actual.metadata["render"]["integrator_version"],
    "capture_binary_sha256":binary_sha},indent=2)+"\n")
print("Final-binary 4096-SPP repeatability:",repeatability["reference"])
