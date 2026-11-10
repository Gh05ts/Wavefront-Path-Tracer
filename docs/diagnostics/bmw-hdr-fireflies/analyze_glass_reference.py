#!/usr/bin/env python3
"""Compare v4/v5 diagnostic replicas with fixed exposure and fixed glass regions."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from wavefront_denoiser.frames import load_frame

root = Path(__file__).resolve().parents[4] / "dataset/diagnostics/bmw-hdr-fireflies"
regions = {"windshield": (125,30,216,64), "headlight": (180,75,200,92), "side window": (231,38,251,58)}
results = []
for spp in (1024,4096):
    frames = []
    for version, case in ((4,"bmw_v4_first"),(5,"bmw_v5_glass_first")):
        location = root / ("bmw-v4-reference-check" if version==4 else "bmw-v5-glass-reference-check")
        replicas = [load_frame(location / f"captures/{case}/s{spp}/r{r}/frame_000000").buffers["reference"] for r in (0,1)]
        assert all(np.isfinite(frame).all() for frame in replicas)
        frames.extend(replicas)
        for region, (x0,y0,x1,y1) in regions.items():
            a,b = [frame[y0:y1,x0:x1].astype(np.float64) for frame in replicas]
            results.append({"spp":spp,"integrator":version,"region":region,
                            "replica_rms_difference":float(np.sqrt(np.mean((a-b)**2))),
                            "maximum_channel":float(max(a.max(),b.max()))})
    sheet = Image.new("RGB",(1280,720));draw = ImageDraw.Draw(sheet)
    for column, frame in enumerate(frames):
        rgb=np.maximum(frame,0);pixels=np.uint8(np.clip((rgb/(1+rgb))**(1/2.2)*255,0,255))
        overview=Image.fromarray(pixels);boxes=ImageDraw.Draw(overview)
        for box in regions.values(): boxes.rectangle(box,outline="orange")
        sheet.paste(overview,(column*320,24))
        draw.text((column*320+4,4),f"v{4 if column<2 else 5}, {spp} SPP, replica {column%2}",fill="white")
        for row,(name,box) in enumerate(regions.items()):
            crop=Image.fromarray(pixels).crop(box);scale=min(312/crop.width,140/crop.height)
            crop=crop.resize((int(crop.width*scale),int(crop.height*scale)),Image.Resampling.NEAREST)
            y=212+row*168;draw.text((column*320+4,y),name,fill="white")
            sheet.paste(crop,(column*320+4,y+20))
    sheet.save(root / "glass-investigation" / f"gpu-v4-v5-glass-{spp}spp.png")
(root / "glass-investigation" / "gpu-region-comparison.json").write_text(json.dumps(results,indent=2)+"\n")
for result in results: print(result)
