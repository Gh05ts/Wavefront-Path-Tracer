#!/usr/bin/env python3
"""Display explicit CPU glass-model ablations against the supplied PNG."""
import json
import numpy as np
from PIL import Image, ImageDraw
from compare_appearance import neutral, root, reference

ref=Image.open(reference).convert('RGB').resize((700,500),Image.Resampling.LANCZOS)
sheet=Image.new('RGB',(1500,520));draw=ImageDraw.Draw(sheet)
for row,name in enumerate(('headlight','windshield')):
    for col,mode in enumerate(('reference','solid','thin')):
        prefix=root/f'tile-v9-{name}-{mode}'
        record_path=prefix.with_suffix('.json')
        if mode=='reference':
            record_path=(root/f'tile-v9-{name}-solid').with_suffix('.json')
        if not record_path.exists():continue
        m=json.loads(record_path.read_text())
        x0,y0,x1,y1=m['box']
        if mode=='reference':
            image=ref.crop((700-x1,y0,700-x0,y1))
            title=f'{name}: supplied Cycles (denoised)'
        else:
            rgb=np.fromfile(prefix.with_suffix('.rgb32f'),dtype='<f4').reshape(m['height'],m['width'],3)
            image=neutral(rgb).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            title=f'CPU v9 {mode} glass, {m["samples"]} SPP (raw)'
        scale=min(492/image.width,215/image.height)
        x=col*500+4;y=row*260+4
        draw.text((x,y),title,fill='white')
        sheet.paste(image.resize((int(image.width*scale),int(image.height*scale)),Image.Resampling.NEAREST),(x,y+22))
sheet.save(root/'glass-model-ablation.png')
