#!/usr/bin/env python3
"""Display-only matched-camera comparison; never edits captured linear HDR."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from wavefront_denoiser.frames import load_frame

repo=Path(__file__).resolve().parents[3]
root=repo.parent/'dataset/diagnostics/bmw-hdr-fireflies/appearance-investigation'
reference=repo/'assets/gltf-research-scenes/scenes/bmw-m6/gltf/bmw-m6.png'

def neutral(rgb):
    # Khronos PBR Neutral reference algorithm, followed by exact sRGB transfer.
    # https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/pbrNeutral.glsl
    color=np.maximum(rgb.astype(np.float64),0)
    x=color.min(axis=-1,keepdims=True)
    color-=np.where(x<.08,x-6.25*x*x,.04)
    peak=color.max(axis=-1,keepdims=True)
    compressed=1-.24*.24/np.maximum(peak+.24-.76,1.e-12)
    high=color*compressed/np.maximum(peak,1.e-12)
    g=1-1/(.15*(peak-compressed)+1)
    high=high*(1-g)+compressed*g
    color=np.clip(np.where(peak<.76,color,high),0,1)
    srgb=np.where(color<=.0031308,12.92*color,1.055*color**(1/2.4)-.055)
    return Image.fromarray(np.uint8(np.clip(srgb*255+.5,0,255)))

columns=[('Supplied Cycles: 4096 SPP + OPTIX denoise',Image.open(reference).convert('RGB').resize((700,500),Image.Resampling.LANCZOS))]
metadata=[]
frames={}
for v in (8,9,10):
    directory=root/f'gltf-camera-v{v}/frame_000000'
    if not directory.exists(): continue
    frame=load_frame(directory)
    frames[v]=frame
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['render']['integrator_version']==v
    image=neutral(frame.buffers['reference']).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    image.save(root/f'gltf-camera-v{v}-neutral-mirrored.png')
    columns.append((f'Our v{v}: 1024 SPP, raw; PBR Neutral + sRGB',image))
    metadata.append({'integrator':v,'manifest':str(directory/'manifest.json'),
                     'display_mirrored_x':True,'exposure':1,'tonemapper':'Khronos PBR Neutral','transfer':'sRGB'})

sheet=Image.new('RGB',(700*len(columns),930));draw=ImageDraw.Draw(sheet)
regions=[('windshield / dashboard',(185,85,445,162)),('headlight cylinders',(275,225,407,284)),('paint / floor',(80,265,280,465))]
for i,(title,image) in enumerate(columns):
    x=i*700;draw.text((x+8,6),title,fill='white');sheet.paste(image,(x,28))
    for row,(name,box) in enumerate(regions):
        crop=image.crop(box)
        scale=min(680/crop.width,100/crop.height)
        y=540+row*126;draw.text((x+8,y),name,fill='white')
        sheet.paste(crop.resize((int(crop.width*scale),int(crop.height*scale)),Image.Resampling.NEAREST),(x+8,y+18))
sheet.save(root/'matched-camera-appearance.png')
if 9 in frames and 10 in frames:
    latest=dict((v,neutral(frames[v].buffers['reference']).transpose(Image.Transpose.FLIP_LEFT_RIGHT)) for v in (9,10))
    for name,box in regions[:2]:
        detail=Image.new('RGB',(1500,250));d=ImageDraw.Draw(detail)
        ref=Image.open(reference).convert('RGB').resize((700,500),Image.Resampling.LANCZOS)
        for col,(label,im) in enumerate([('Supplied Cycles: denoised',ref),('v9: coating only, raw 1024 SPP',latest[9]),('v10: corrected reflective frames, raw 1024 SPP',latest[10])]):
            crop=im.crop(box);scale=min(490/crop.width,220/crop.height)
            d.text((col*500+5,5),label,fill='white')
            detail.paste(crop.resize((int(crop.width*scale),int(crop.height*scale)),Image.Resampling.NEAREST),(col*500+5,25))
        filename='headlight-detail-v10.png' if name.startswith('headlight') else 'windshield-detail-v10.png'
        detail.save(root/filename)
guides={}
if 8 in frames and 9 in frames:
    for name in frames[8].buffers:
        if name not in ('radiance','reference'):
            guides[name]=bool(np.array_equal(frames[8].buffers[name],frames[9].buffers[name],equal_nan=True))
(root/'display-comparison.json').write_text(json.dumps({'reference':str(reference),'comparison':metadata,
    'v8_v9_guide_and_count_equality':guides,
    'v9_v10_guide_and_count_equality':{name:bool(np.array_equal(frames[9].buffers[name],frames[10].buffers[name],equal_nan=True)) for name in frames[9].buffers if name not in ('radiance','reference')} if 9 in frames and 10 in frames else {},
    'cautions':['Reference is 1400x1000, displayed at 700x500; ours is 700x500, raw box-filtered 1024 SPP.',
                'Exported camera has a mirrored X basis; only our viewing PNG is flipped. Captures/plans are unchanged.',
                'PNG-only display comparison cannot establish linear-radiance accuracy or reference convergence.']},indent=2)+'\n')
