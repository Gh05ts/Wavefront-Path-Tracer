#!/usr/bin/env python3
"""Inspect published v4 capture metadata without locking or reading EXR pixels."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def status(root, previous=None):
    root = Path(root).resolve()
    split = json.loads((root / "split.json").read_text())
    planned = Counter()
    completed = Counter()
    complete_groups = []
    newest = None
    errors = []
    for name, group in split["groups"].items():
        expected = group.get("frame_count")
        if expected is None:
            config = json.loads((root / group["capture_config"]).read_text())
            expected = len(config["frames"])
        scene = group["scene"]
        planned[scene] += expected
        count = 0
        for index in range(expected):
            manifest = root / name / f"frame_{index:06d}" / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text())
                if (data["sequence_id"] != name or data["frame_index"] != index
                        or not (manifest.parent / data["image"]).is_file()):
                    raise ValueError("Published frame identity or image is missing")
            except (ValueError, KeyError, OSError) as error:
                errors.append({"manifest":str(manifest), "error":str(error)})
                continue
            count += 1
            modified = manifest.stat().st_mtime
            if newest is None or modified > newest[0]:
                newest = (modified, str(manifest))
        completed[scene] += count
        if count == expected:
            complete_groups.append(name)
    now = time.time()
    report = {"root":str(root), "snapshot_unix":now,
              "snapshot_utc":datetime.fromtimestamp(now, timezone.utc).isoformat(),
              "completed_frames":sum(completed.values()), "planned_frames":sum(planned.values()),
              "scenes":{s:{"completed":completed[s], "planned":planned[s]} for s in planned},
              "complete_groups":complete_groups, "metadata_errors":errors,
              "verification":"metadata_and_image_presence_only_not_checksums_or_convergence"}
    if newest:
        report["newest_frame"] = {"manifest":newest[1], "manifest_mtime_unix":newest[0],
                                  "seconds_since_manifest_mtime":now-newest[0]}
    if previous:
        before = json.loads(Path(previous).read_text())
        if before["root"] != str(root) or before["planned_frames"] != report["planned_frames"]:
            raise ValueError("Previous snapshot belongs to a different dataset")
        elapsed = now-before["snapshot_unix"]
        delta = report["completed_frames"]-before["completed_frames"]
        if elapsed <= 0 or delta < 0:
            raise ValueError("Progress snapshots are inconsistent")
        report["interval"] = {"seconds":elapsed, "new_completed_frames":delta,
                              "frames_per_hour":delta*3600/elapsed,
                              "note":"Scene costs vary; this rate is not a whole-dataset ETA"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset",type=Path)
    parser.add_argument("--previous",type=Path,help="Earlier JSON snapshot to measure observed progress")
    args = parser.parse_args()
    try:
        print(json.dumps(status(args.dataset,args.previous),indent=2,allow_nan=False))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1,f"capture status: {error}\n")


if __name__ == "__main__":
    main()
