#!/usr/bin/env python3
"""Copy complete train/validation clips into a verified, separate v4 pilot.

Reads published source frames without taking a writer lock or changing source
files. Separate sources must have identical capture locks and split metadata.
Incomplete attempts remain in the explicitly named output.partial directory.
"""
import argparse
import json
from pathlib import Path
import shutil

from wavefront_denoiser.dataset_generation import digest, verify_frame


def snapshot(sources, groups, output):
    sources = [p.resolve() for p in sources]
    output = output.resolve()
    staging = output.with_name(output.name+'.partial')
    if output.exists() or staging.exists():
        raise FileExistsError("Choose a new output; an existing snapshot/partial is preserved")
    if any(p == output or p in output.parents or output in p.parents for p in sources):
        raise ValueError("Snapshot and source trees must be separate")
    lock = json.loads((sources[0]/'capture_lock.json').read_text())
    split = json.loads((sources[0]/'split.json').read_text())
    for source in sources[1:]:
        if (json.loads((source/'capture_lock.json').read_text()) != lock
                or json.loads((source/'split.json').read_text()) != split):
            raise ValueError("Source capture locks differ: renderer bytes, asset paths/bytes and settings must agree")
    allowed = set(split['partitions']['train']+split['partitions']['validation'])
    if not groups or len(set(groups)) != len(groups) or set(groups)-allowed:
        raise ValueError("Select unique registered train/validation groups only")
    origins = {}
    for group in groups:
        config = lock['configs'][group]
        candidates = [s for s in sources if all((s/group/f'frame_{i:06d}'/'manifest.json').is_file()
                                                for i in range(len(config['frames'])))]
        if not candidates:
            raise ValueError(f"No complete source clip: {group}")
        for source in candidates:
            for i in range(len(config['frames'])):
                verify_frame(source/group/f'frame_{i:06d}',config,i)
        if any(digest(candidates[0]/group/f'frame_{i:06d}'/'manifest.json') !=
               digest(source/group/f'frame_{i:06d}'/'manifest.json')
               for source in candidates[1:] for i in range(len(config['frames']))):
            raise ValueError(f"Duplicate source clips disagree: {group}")
        origins[group] = candidates[0]
    staging.mkdir(parents=True,exist_ok=False)
    for name in ('capture_lock.json','split.json'):
        shutil.copy2(sources[0]/name,staging/name)
    (staging/'capture_configs').mkdir()
    for group,config in lock['configs'].items():
        (staging/'capture_configs'/f'{group}.json').write_text(json.dumps(config,indent=2)+'\n')
    frames = 0
    for group,source in origins.items():
        config = lock['configs'][group]
        for index in range(len(config['frames'])):
            original = source/group/f'frame_{index:06d}'
            target = staging/group/f'frame_{index:06d}'
            target.mkdir(parents=True)
            metadata = verify_frame(original,config,index)
            for name in ('manifest.json',metadata['image']):
                shutil.copy2(original/name,target/name)
            if verify_frame(target,config,index) != metadata or digest(original/'manifest.json') != digest(target/'manifest.json'):
                raise ValueError(f"Copied frame differs: {group}/{index}")
            frames += 1
    report = {'output':str(output),'frames':frames,'origins':{g:str(s) for g,s in origins.items()},
              'scope':'diagnostic_subset_full_production_split_preserved', 'source_files_changed':False}
    (staging/'snapshot-info.json').write_text(json.dumps(report,indent=2)+'\n')
    staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',action='append',required=True,type=Path)
    parser.add_argument('--group',action='append',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(snapshot(args.source,args.group,args.output),indent=2))
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f'clip snapshot: {error}\n')


if __name__ == '__main__':
    main()
