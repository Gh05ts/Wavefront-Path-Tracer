#!/usr/bin/env python3
"""Bounded Cornell dragon regression captures in a separate build/output tree.

Uses the exact Cornell preset and camera. Compares raw/clamped replicas with
no denoising. Outlier thresholds are diagnostics, not a ground-truth mask.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, default=Path('/tmp/dragon-cornell-check-build/dataset_capture'))
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    spec = importlib.util.spec_from_file_location('hdr_validation', here.parent/'bmw-hdr-fireflies/validate_indirect_clamp.py')
    hdr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hdr)
    args.output.mkdir(parents=True, exist_ok=False)
    base = json.loads((here/'preview.json').read_text())
    base['scene_file'] = str((here/base['scene_file']).resolve())
    binary_hash = hdr.digest(args.executable)
    watched = Path('/tmp/ntasad-capture-build/dataset_capture')
    active_hash = hdr.digest(watched) if watched.exists() else None
    records = {}
    commands = []

    def capture(name, config):
        config = dict(config, sequence_id=name)
        filename = args.output/(name+'.json')
        filename.write_text(json.dumps(config, indent=2)+'\n')
        command = [str(args.executable.resolve()), str(filename.resolve()), str((args.output/name).resolve())]
        commands.append(command)
        print('Rendering '+name, flush=True)
        with (args.output/(name+'.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=repo)
        if hdr.digest(args.executable) != binary_hash or (active_hash and hdr.digest(watched) != active_hash):
            raise RuntimeError('A watched executable changed during the check')
        meta, channels, rgb = hdr.read_frame(args.output/name/'frame_000000')
        if meta['render']['integrator_version'] != 11 or meta['render']['environment']['enabled']:
            raise RuntimeError('Wrong integrator or unexpected HDR environment')
        if meta['render']['indirect_radiance_clamp'] != config['indirect_radiance_clamp']:
            raise RuntimeError('Wrong clamp setting')
        return meta, channels, rgb

    for spp in (256, 1024):
        for replica in (0, 1):
            for clamp in (0, 10):
                config = dict(base, width=320, height=180, reference_samples=spp,
                              sample_start=900000000+replica*8192, indirect_radiance_clamp=clamp)
                result = capture(f's{spp}-r{replica}-clamp{clamp}', config)
                records[spp, replica, clamp] = result
                if clamp:
                    for name, values in result[1].items():
                        if not name.startswith(('radiance.', 'reference.')):
                            np.testing.assert_array_equal(values, records[spp, replica, 0][1][name], err_msg=name)

    report = {'integrator': 11, 'binary_sha256': binary_hash, 'active_dataset_binary_unchanged': True,
              'active_dataset_binary_sha256': active_hash,
              'all_buffers_finite': True, 'guides_and_counts_equal': True, 'commands': commands,
              'base_config': base, 'levels': [],
              'outlier_rule': 'luminance > .1, >4x eight-neighbor median, >4x other replica, and excess > .1; can include real glints',
              'floor_mask': 'primary guide geometric normal Y > .9999 and RGB albedo within .0001 of .73',
              'display': 'Reinhard at exposure one then gamma 1/2.2; no denoising'}
    sheet = Image.new('RGB', (1280, 412))
    draw = ImageDraw.Draw(sheet)
    for row, spp in enumerate((256, 1024)):
        for clamp in (0, 10):
            pair = [records[spp, r, clamp][2] for r in (0, 1)]
            level = {'samples': spp, 'clamp': clamp,
                     'replica_noise_rmse': float(np.sqrt(np.mean((pair[0]-pair[1])**2)/2)), 'replicas': []}
            for replica, rgb in enumerate(pair):
                lum = rgb @ np.array([.2126, .7152, .0722])
                other = pair[1-replica] @ np.array([.2126, .7152, .0722])
                padded = np.pad(lum, 1, mode='edge')
                median = np.median([padded[y:y+180, x:x+320] for y in range(3) for x in range(3)
                                    if (x, y) != (1, 1)], axis=0)
                unstable = (lum>.1) & (lum>4*median) & (lum>4*other) & (lum-other>.1)
                strict = (lum>1) & (lum>10*np.maximum(median, 1e-6))
                guides = records[spp, replica, clamp][1]
                floor = guides['geometric_normal.Y'] > .9999
                for component in 'RGB':
                    floor &= np.abs(guides['albedo.'+component]-.73) < .0001
                level['replicas'].append({'unstable_whole': int(unstable.sum()),
                    'unstable_floor': int(unstable[floor].sum()), 'floor_pixels': int(floor.sum()),
                    'strict_bright_local': int(strict.sum()),
                    'max_rgb': float(rgb.max()), 'mean_rgb': rgb.mean(axis=(0, 1)).tolist()})
                display = np.maximum(rgb, 0)
                pixels = np.uint8(np.clip((display/(1+display))**(1/2.2)*255, 0, 255))
                x, y = (2*int(clamp>0)+replica)*320, row*206
                draw.text((x+4, y+4), f'{spp} SPP, replica {replica}, clamp {clamp}', fill='white')
                sheet.paste(Image.fromarray(pixels), (x, y+24))
            report['levels'].append(level)
    sheet.save(args.output/'comparison.png')

    if args.baseline:
        baseline = hdr.read_frame(args.baseline)
        current = records[1024, 0, 0]
        comparison = {}
        for name, a in baseline[1].items():
            b = current[1][name]
            difference = a.astype(np.float64)-b
            comparison[name] = {'identical': bool(np.array_equal(a, b)),
                'max_absolute_difference': float(np.abs(difference).max()),
                'rms_difference': float(np.sqrt(np.mean(difference**2)))}
        report['preclamp_integrator_10_comparison'] = comparison

    hero = capture('preview-640', base)
    rgb = np.maximum(hero[2], 0)
    Image.fromarray(np.uint8(np.clip((rgb/(1+rgb))**(1/2.2)*255, 0, 255))).save(args.output/'preview.png')
    report['preview'] = {'width': 640, 'height': 360, 'samples': 1024, 'clamp': 10}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
