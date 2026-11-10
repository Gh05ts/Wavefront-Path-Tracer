from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from wavefront_denoiser.frames import load_frame

workspace=Path(__file__).resolve().parents[4]
old_root=workspace/'dataset/denoiser-v2/bmw_view_00_linear_slow'
new_root=workspace/'dataset/denoiser-v3/bmw_view_00_linear_slow'
output=workspace/'dataset/diagnostics/bmw-hdr-fireflies/gpu-v4'
output.mkdir(parents=True,exist_ok=True)

def display(values):
    v=np.maximum(values,0)
    return Image.fromarray(np.uint8(np.clip((v/(1+v))**(1/2.2)*255,0,255)))

def floor_stats(frame):
    rgb=frame.buffers['reference']
    luma=rgb@np.array([.2126,.7152,.0722])
    pad=np.pad(luma,1,mode='edge')
    h,w=luma.shape
    median=np.median(np.stack([pad[j:j+h,i:i+w] for j in range(3) for i in range(3)]),axis=0)
    y,x=np.indices(luma.shape)
    mask=(y>760)&(y<1040)&(x>1150)&(x<1900)&(frame.buffers['normal'][...,1]>.98)
    mask &= np.max(np.abs(frame.buffers['albedo']-.0183606343),axis=-1)<1e-6
    ratios=luma[mask]/np.maximum(median[mask],1e-8)
    return {'pixels':int(mask.sum()),'isolated_luminance_outliers_above_10x_local_median':int((ratios>10).sum()),
            'max_local_luminance_ratio':float(ratios.max()),'max_reference_red':float(rgb[...,0][mask].max())}

records=[]
for index in range(2):
    if not (new_root/f'frame_{index:06d}'/'manifest.json').is_file():
        continue
    old=load_frame(old_root/f'frame_{index:06d}')
    new=load_frame(new_root/f'frame_{index:06d}')
    assert new.metadata['render']['integrator_version']==4
    assert old.metadata['camera']==new.metadata['camera']
    assert old.metadata['render_fingerprint']!=new.metadata['render_fingerprint']
    for key in ('sample_start','reference_sample_start','samples','reference_samples','max_depth','sample_stride'):
        assert old.metadata['render'][key]==new.metadata['render'][key]
    old_image=display(old.buffers['reference']);new_image=display(new.buffers['reference'])
    new_image.save(output/f'frame_{index:06d}-reference-1024spp.png')
    record={'frame':index,'old':floor_stats(old),'new':floor_stats(new),
            'all_buffers_finite':all(np.isfinite(v).all() for v in new.buffers.values()),
            'history_valid_fraction':float(new.buffers['history_valid'].mean())}
    if index==0:
        record['reproduced_pixels']=[]
        for x,y in ((1790,922),(1839,861),(1727,911),(1359,826),(1428,820)):
            record['reproduced_pixels'].append({'x':x,'y':y,'old_reference_rgb':old.buffers['reference'][y,x].tolist(),
                                                'new_reference_rgb':new.buffers['reference'][y,x].tolist()})
        crop=(1150,760,1900,1040)
        sheet=Image.new('RGB',(1500,312));draw=ImageDraw.Draw(sheet)
        for col,(im,label) in enumerate(((old_image,'Before | integrator 3 | 1024 SPP'),(new_image,'After | integrator 4 | 1024 SPP'))):
            sheet.paste(im.crop(crop),(750*col,32));draw.text((750*col+8,8),label,fill='white')
        sheet.save(output/'floor-before-after.png')
    records.append(record)
    print(json.dumps(record,indent=2),flush=True)
(output/'native-1080p-results.json').write_text(json.dumps({'render_width':1920,'render_height':1080,'reference_spp':1024,'frames':records,
    'note':'Diagnostic outlier heuristic, not a denoiser evaluation metric or a convergence certificate.'},indent=2)+'\n')
