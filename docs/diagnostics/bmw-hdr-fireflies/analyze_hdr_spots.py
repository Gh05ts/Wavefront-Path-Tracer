#!/usr/bin/env python3
"""Fixed-exposure HDR spot diagnostics; counts are not a ground-truth firefly mask."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from wavefront_denoiser.frames import load_frame

root = Path(__file__).resolve().parents[4] / "dataset/diagnostics/bmw-hdr-fireflies"
output = root / "hdr-spot-investigation"
output.mkdir(exist_ok=True)
regions = {"windshield": (125, 30, 216, 64), "grille": (79, 82, 143, 108),
           "front shadow": (72, 108, 164, 145), "headlight": (180, 75, 200, 92),
           "side window": (231, 38, 251, 58), "both headlights": (145, 75, 200, 106),
           "whole image": (0, 0, 320, 180)}
records, reports, all_frames = [], [], {}
versions = [(4, "bmw-v4-reference-check", "bmw_v4_first"),
            (5, "bmw-v5-glass-reference-check", "bmw_v5_glass_first")]
if (root / "bmw-v6-hdr-reference-check/report.json").exists():
    versions.append((6, "bmw-v6-hdr-reference-check", "bmw_v6_hdr_first"))
if (root / "bmw-v7-hdr-reference-check/report.json").exists():
    versions.append((7, "bmw-v7-hdr-reference-check", "bmw_v7_hdr_first"))
if (root / "bmw-v8-hdr-reference-check/report.json").exists():
    versions.append((8, "bmw-v8-hdr-reference-check", "bmw_v8_hdr_first"))
if (root / "bmw-v9-hdr-reference-check/report.json").exists():
    versions.append((9, "bmw-v9-hdr-reference-check", "bmw_v9_material_first"))
if (root / "bmw-v10-hdr-reference-check/report.json").exists():
    versions.append((10, "bmw-v10-hdr-reference-check", "bmw_v10_normal_first"))
for version, location, case in versions:
    report = json.loads((root / location / "report.json").read_text())["cases"][0]
    reports.append({"integrator": version, "input_rmse": report["input_rmse_against_anchor"],
                    "levels": [{k: level[k] for k in ("samples", "replica_noise_rmse", "noise_fraction_of_input")}
                               for level in report["levels"]]})
    for spp in (1024, 4096):
        replicas = [load_frame(root / location / f"captures/{case}/s{spp}/r{r}/frame_000000").buffers["reference"].astype(np.float64)
                    for r in (0, 1)]
        for replica in (0, 1):
            rgb = replicas[replica]
            all_frames[version, spp, replica] = rgb
            assert np.isfinite(rgb).all()
            lum = rgb @ np.array([.2126, .7152, .0722])
            padded = np.pad(lum, 1, mode="edge")
            neighbors = [padded[dy:dy+lum.shape[0], dx:dx+lum.shape[1]]
                         for dy in range(3) for dx in range(3) if (dx, dy) != (1, 1)]
            median = np.median(neighbors, axis=0)
            # Require both absolute brightness and local contrast. Several
            # thresholds expose sensitivity; genuine small sun glints also pass.
            spot = (lum > 1) & (lum > 10 * np.maximum(median, 1e-6))
            other = replicas[1-replica] @ np.array([.2126, .7152, .0722])
            unstable = (lum > .1) & (lum > 4*median) & (lum > 4*other) & (lum-other > .1)
            pixels = np.uint8(np.clip((np.maximum(rgb, 0)/(1+np.maximum(rgb, 0)))**(1/2.2)*255, 0, 255))
            sheet = Image.new("RGB", (1100, 610)); draw = ImageDraw.Draw(sheet)
            draw.text((8, 4), f"v{version}, {spp} SPP, replica {replica}; fixed exposure", fill="white")
            sheet.paste(Image.fromarray(pixels).resize((640, 360), Image.Resampling.NEAREST), (0, 24))
            for i, (region, (x0, y0, x1, y1)) in enumerate(regions.items()):
                sl = np.s_[y0:y1, x0:x1]
                ys, xs = np.nonzero(spot[sl])
                order = np.argsort(lum[sl][ys, xs])[::-1][:12]
                records.append({"integrator": version, "spp": spp, "replica": replica, "region": region,
                                "box": [x0, y0, x1, y1], "bright_local_spots": int(spot[sl].sum()),
                                "unstable_local_spots": int(unstable[sl].sum()),
                                "replica_rgb_rms_difference": float(np.sqrt(np.mean((rgb[sl]-replicas[1-replica][sl])**2))),
                                "luminance_gt_1": int((lum[sl] > 1).sum()),
                                "luminance_gt_4": int((lum[sl] > 4).sum()),
                                "luminance_gt_16": int((lum[sl] > 16).sum()),
                                "top_spots": [{"x": int(xs[j]+x0), "y": int(ys[j]+y0),
                                               "luminance": float(lum[ys[j]+y0, xs[j]+x0]),
                                               "local_median": float(median[ys[j]+y0, xs[j]+x0])}
                                              for j in order]})
                if i >= 5:
                    continue
                x, y = (650, 24+i*112) if i < 3 else ((i-3)*330, 400)
                crop = Image.fromarray(pixels).crop((x0, y0, x1, y1))
                scale = min(320/crop.width, 90/crop.height)
                draw.text((x, y), f"{region}: {int(spot[sl].sum())} bright local spots", fill="white")
                sheet.paste(crop.resize((int(crop.width*scale), int(crop.height*scale)), Image.Resampling.NEAREST), (x, y+16))
            sheet.save(output / f"v{version}-s{spp}-r{replica}.png")
(output / "spot-counts.json").write_text(json.dumps({"definition": "linear luminance > 1 and > 10 times median of 8 neighbors; counts include possible genuine glints",
                                                    "unstable_definition": "luminance > .1, >4x median of 8 neighbors, >4x other replica, and >.1 excess over other replica; a diagnostic, not ground truth",
                                                    "ratios": reports, "regions": records}, indent=2)+"\n")
if len(versions) >= 3:
    compared = versions[:2]+[versions[-1]]
    for spp in (1024, 4096):
        sheet = Image.new("RGB", (1920, 820)); draw = ImageDraw.Draw(sheet)
        for column, (version, replica) in enumerate((v, r) for v, _, _ in compared for r in (0, 1)):
            rgb = np.maximum(all_frames[version, spp, replica], 0)
            pixels = np.uint8(np.clip((rgb/(1+rgb))**(1/2.2)*255, 0, 255))
            sheet.paste(Image.fromarray(pixels), (column*320, 24))
            draw.text((column*320+4, 4), f"v{version}, {spp} SPP, replica {replica}", fill="white")
            for row, (name, box) in enumerate((k, regions[k]) for k in ("windshield", "grille", "both headlights", "side window")):
                crop = Image.fromarray(pixels).crop(box); scale = min(312/crop.width, 120/crop.height)
                y = 212+row*150;draw.text((column*320+4, y), name, fill="white")
                sheet.paste(crop.resize((int(crop.width*scale), int(crop.height*scale)), Image.Resampling.NEAREST), (column*320+4, y+20))
        sheet.save(output / f"gpu-v4-v5-v{versions[-1][0]}-{spp}spp.png")
for row in records:
    if row["region"] in ("windshield", "grille", "front shadow"):
        print(row["integrator"], row["spp"], row["replica"], row["region"], row["bright_local_spots"], row["top_spots"][:3])
