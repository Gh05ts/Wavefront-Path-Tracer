#!/usr/bin/env python3
"""Selected-pixel CPU variance experiment; never changes renderer/captures.

Compile cpu_replay.cpp as documented in run_cpu_checks.py against the CURRENT
headers, then pass that executable. This is not full-image qualification.
"""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("executable", type=Path)
parser.add_argument("--samples", type=int, default=65536)
args = parser.parse_args()
workspace = Path(__file__).resolve().parents[4]
root = workspace / "dataset/diagnostics/bmw-hdr-fireflies"
output = root / "roulette-study"
output.mkdir(exist_ok=True)
scene = workspace / "Wavefront-Path-Tracer/assets/gltf-research-scenes/scenes/bmw-m6/gltf/bmw_m6_extended.gltf"
hdr = scene.parents[2] / "common/sunflowers_environment.hdr"
for roulette in (5, 12):
    for replica in (0, 1):
        manifest = root / f"bmw-v4-reference-check/captures/bmw_v4_first/s4096/r{replica}/frame_000000/manifest.json"
        base_manifest = root / "bmw-v4-reference-check/captures/bmw_v4_first/s4096/r0/frame_000000/manifest.json"
        start = json.loads(base_manifest.read_text())["render"]["reference_sample_start"] + replica * args.samples
        command = [str(args.executable.resolve()), str(scene), str(hdr), str(manifest),
                   str(args.samples), str(start), "166", "38", str(roulette)]
        (output / f"pixel-166-38-r{replica}-rr{roulette}-command.json").write_text(
            json.dumps(command, indent=2) + "\n")
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=True)
        (output / f"pixel-166-38-r{replica}-rr{roulette}.txt").write_text(result.stdout)
        print(result.stdout, flush=True)
