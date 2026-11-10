#!/usr/bin/env python3
"""Bounded HDR regression/calibration, using the original single-job build."""
import hashlib
import argparse
import re
import json
from pathlib import Path
import subprocess
import shutil

repo = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--integrator", type=int, choices=(6,7,8,9,10), default=10)
version = parser.parse_args().integrator
source_version = int(re.search(r"rendererIntegratorVersion\s*=\s*(\d+)",
    (repo / "include/renderer/render_fingerprint.hpp").read_text()).group(1))
if source_version != version:
    raise RuntimeError("Requested diagnostic version differs from the current renderer source")
root = repo.parent / "dataset/diagnostics/bmw-hdr-fireflies"
build = Path("/tmp/ntasad-capture-build")
output = root / "hdr-spot-investigation"
binary = build / "dataset_capture"
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

starting = digest(binary)
ctest = ["/tmp/ntasad-venv/bin/ctest", "--test-dir", str(build), "--output-on-failure", "-j1"]
result = subprocess.run(ctest, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(output / f"ctest-v{version}.txt").write_text(result.stdout)
details = build / "Testing/Temporary/LastTest.log"
if details.exists():
    shutil.copyfile(details, output / f"ctest-v{version}-details.txt")
print(result.stdout, flush=True)
result.check_returncode()
capture = ["/tmp/ntasad-venv/bin/wavefront-denoiser", "reference-convergence",
           str(root / f"bmw-v{version}-hdr-reference-plan.json"), str(root / f"bmw-v{version}-hdr-reference-check"),
           "--capture-executable", str(binary), "--renderer-root", str(repo)]
result = subprocess.run(capture, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(output / f"calibration-v{version}.txt").write_text(result.stdout)
print(result.stdout, flush=True)
result.check_returncode()
if digest(binary) != starting:
    raise RuntimeError("Capture binary changed during diagnostics")
report = json.loads((root / f"bmw-v{version}-hdr-reference-check/report.json").read_text())
if any(case["render_settings"]["integrator_version"] != version for case in report["cases"]):
    raise RuntimeError("Capture binary version differs from the diagnostic label")
watched = ["include/renderer/device_traversal.cuh", "include/renderer/device_path_shading.cuh",
           "include/renderer/render_fingerprint.hpp", "src/scene/environment.cpp",
           "tests/environment_tests.cpp", "tests/environment_shading_tests.cu"]
if version >= 9:
    watched += ["include/scene/material.cuh", "include/renderer/gltf_pbr.cuh", "src/scene/gltf_loader.cpp"]
(output / f"gpu-v{version}-provenance.json").write_text(json.dumps({"integrator": version, "build": str(build),
    "build_jobs": 1, "capture_sha256": starting, "commands": [ctest, capture],
    "sources": {name: digest(repo/name) for name in watched}}, indent=2)+"\n")
