#!/usr/bin/env python3
"""Capture raw/clamped replicas and compare linear HDR specks at fixed exposure.

Requires numpy, Pillow and OpenEXR. Output must be a fresh directory. Captures
use the same geometry, camera and RNG ranges for each clamp; no denoising.
Spot counts are diagnostics, not a ground-truth firefly mask or energy test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import Imath
import numpy as np
import OpenEXR
from PIL import Image, ImageDraw


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frame(directory):
    meta = json.loads((directory / "manifest.json").read_text())
    image = directory / meta["image"]
    if digest(image) != meta["image_sha256"]:
        raise RuntimeError("Capture image digest mismatch")
    source = OpenEXR.InputFile(str(image))
    try:
        channels = {
            name: np.frombuffer(source.channel(name, Imath.PixelType(Imath.PixelType.FLOAT)),
                                dtype=np.float32).reshape(meta["height"], meta["width"])
            for name in source.header()["channels"]
        }
    finally:
        source.close()
    if not all(np.isfinite(values).all() for values in channels.values()):
        raise RuntimeError("Non-finite capture")
    rgb = np.stack([channels["reference." + component] for component in "RGB"], axis=-1)
    return meta, channels, rgb.astype(np.float64)


def spots(rgb, other):
    lum = rgb @ np.array([.2126, .7152, .0722])
    padded = np.pad(lum, 1, mode="edge")
    median = np.median([padded[y:y+lum.shape[0], x:x+lum.shape[1]]
                        for y in range(3) for x in range(3) if (x, y) != (1, 1)], axis=0)
    strict = (lum > 1) & (lum > 10 * np.maximum(median, 1e-6))
    unstable = (lum > .1) & (lum > 4 * median) & (lum > 4 * other) & (lum-other > .1)
    height, width = lum.shape
    regions = {"whole": (0, 0, width, height), "floor": (0, height*3//5, width, height),
               "windshield": (width*125//320, height*30//180, width*216//320, height*64//180)}
    return {name: {"strict_bright_local": int(strict[y0:y1, x0:x1].sum()),
                   "unstable_local": int(unstable[y0:y1, x0:x1].sum())}
            for name, (x0, y0, x1, y1) in regions.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=Path("/tmp/ntasad-capture-build/dataset_capture"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, nargs="+", default=[64, 256, 1024, 4096])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads(args.config.read_text())
    if len(base["frames"]) != 1:
        raise ValueError("Use a single-frame diagnostic configuration")
    if "scene_file" in base:
        base["scene_file"] = str((args.config.resolve().parent / base["scene_file"]).resolve())
    binary_hash = digest(args.executable)
    captures = {}
    for spp in args.samples:
        for replica in (0, 1):
            for clamp in (0, 10):
                config = dict(base, reference_samples=spp, indirect_radiance_clamp=clamp,
                              sample_start=800000000 + replica * max(4096, max(args.samples)),
                              sample_stride=max(8192, spp+base.get("samples", 4)))
                name = f"s{spp}-r{replica}-clamp{clamp}"
                config["sequence_id"] = name
                config_path = args.output / (name + ".json")
                config_path.write_text(json.dumps(config, indent=2) + "\n")
                command = [str(args.executable.resolve()), str(config_path.resolve()), str((args.output/name).resolve())]
                print("Rendering " + name, flush=True)
                with (args.output / (name + ".log")).open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                if digest(args.executable) != binary_hash:
                    raise RuntimeError("Executable changed during validation")
                meta, channels, rgb = read_frame(args.output / name / "frame_000000")
                if meta["render"]["integrator_version"] != 11 or meta["render"]["indirect_radiance_clamp"] != clamp:
                    raise RuntimeError("Wrong integrator or clamp in capture metadata")
                captures[spp, replica, clamp] = (meta, channels, rgb)
                if clamp:
                    raw = captures[spp, replica, 0]
                    for channel in channels:
                        if not channel.startswith(("radiance.", "reference.")):
                            if not np.array_equal(raw[1][channel], channels[channel]):
                                raise RuntimeError("Clamp changed guide/count channel " + channel)

    report = {"integrator": 11, "binary_sha256": binary_hash, "guides_counts_equal": True,
              "display": "fixed exposure: Reinhard then gamma 1/2.2; raw, no denoising",
              "caveat": "Clamp is biased; speck counts include genuine glints and are not convergence qualification.",
              "levels": []}
    width, height = next(iter(captures.values()))[2].shape[1::-1]
    sheet = Image.new("RGB", (width*4, (height+26)*len(args.samples)))
    draw = ImageDraw.Draw(sheet)
    for row, spp in enumerate(args.samples):
        for clamp in (0, 10):
            pair = [captures[spp, r, clamp][2] for r in (0, 1)]
            raw_pair = [captures[spp, r, 0][2] for r in (0, 1)]
            record = {"samples": spp, "clamp": clamp,
                      "replica_noise_rmse": float(np.sqrt(np.mean((pair[0]-pair[1])**2)/2)),
                      "replicas": []}
            for replica, rgb in enumerate(pair):
                other = pair[1-replica] @ np.array([.2126, .7152, .0722])
                record["replicas"].append({"spots": spots(rgb, other), "mean_rgb": rgb.mean(axis=(0, 1)).tolist(),
                    "max_rgb": float(rgb.max()), "mean_change_from_raw": (rgb-raw_pair[replica]).mean(axis=(0, 1)).tolist()})
                positive = np.maximum(rgb, 0)
                pixels = np.uint8(np.clip((positive/(1+positive))**(1/2.2)*255, 0, 255))
                x, y = (2*int(clamp > 0)+replica)*width, row*(height+26)
                draw.text((x+4, y+4), f"{spp} SPP, replica {replica}, clamp {clamp}", fill="white")
                sheet.paste(Image.fromarray(pixels), (x, y+24))
            report["levels"].append(record)
    sheet.save(args.output / "comparison.png")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
