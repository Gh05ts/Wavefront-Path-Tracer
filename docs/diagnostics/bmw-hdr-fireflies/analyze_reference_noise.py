from pathlib import Path
import numpy as np
import json
from wavefront_denoiser.frames import load_frame
from PIL import Image, ImageDraw

workspace=Path(__file__).resolve().parents[4]
root=workspace/Path('dataset/diagnostics/bmw-hdr-fireflies/bmw-v4-reference-check/captures/bmw_v4_first')
report=[]
for spp in (1024,4096):
    a=load_frame(root/f's{spp}/r0/frame_000000')
    b=load_frame(root/f's{spp}/r1/frame_000000')
    error=((a.buffers['reference'].astype('float64')-b.buffers['reference'])**2).mean(axis=-1)
    ids=np.argsort(error.ravel())[::-1]
    fraction1=float(error.ravel()[ids[0]]/error.sum())
    fraction10=float(error.ravel()[ids[:10]].sum()/error.sum())
    print('SPP',spp,'maximums',float(a.buffers['reference'].max()),float(b.buffers['reference'].max()),'top1 fraction',fraction1,'top10 fraction',fraction10)
    for idx in ids[:8]:
        y,x=divmod(int(idx),320)
        print('pixel',x,y,'a',a.buffers['reference'][y,x].tolist(),'b',b.buffers['reference'][y,x].tolist(),'albedo',a.buffers['albedo'][y,x].tolist(),'normal',a.buffers['normal'][y,x].tolist(),'depth',a.buffers['depth'][y,x].tolist())
    report.append({'spp':spp,'mse_top1_fraction':fraction1,'mse_top10_fraction':fraction10})
    im=Image.new('RGB',(640,204));draw=ImageDraw.Draw(im)
    for col,frame in enumerate((a,b)):
        rgb=frame.buffers['reference'].clip(0)
        pixels=np.uint8(np.clip((rgb/(1+rgb))**(1/2.2)*255,0,255))
        im.paste(Image.fromarray(pixels),(320*col,24))
        draw.text((320*col+4,4),f'{spp} SPP independent reference {col}',fill='white')
    im.save(root.parent.parent/f'replicas-{spp}spp.png')
(workspace/'dataset/diagnostics/bmw-hdr-fireflies/gpu-v4/remaining-reference-noise.json').write_text(json.dumps(report,indent=2)+'\n')
