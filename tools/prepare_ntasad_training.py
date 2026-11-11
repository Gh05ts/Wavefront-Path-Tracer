#!/usr/bin/env python3
"""Create separate, matching uniform and joint training configurations."""
import argparse
import json
from pathlib import Path

from wavefront_denoiser.adaptive_data import read_v5_dataset
from wavefront_denoiser.dataset_splits import read_split_manifest
from wavefront_denoiser.training_defaults import dataset_training_config, joint_training_config


def prepare(args):
    if args.output.exists():
        raise FileExistsError(f"Choose a new config directory: {args.output}")
    spec = read_split_manifest(args.v4 / "split.json")
    if args.v5:
        companion = read_v5_dataset(args.v5)
        if spec["partitions"] != companion["partitions"]:
            raise ValueError("v4 and v5 must use identical declared partitions")
    if bool(args.train_group) != bool(args.validation_group):
        raise ValueError("Provide both train and validation selections, or neither")
    selections = {"train":args.train_group,"validation":args.validation_group}
    for partition,names in selections.items():
        if names and (len(names) != len(set(names)) or set(names)-set(spec["partitions"][partition])):
            raise ValueError(f"Invalid {partition} group selection")
    for key in ("epochs","batch_size","patch_size","validation_patch_size"):
        if getattr(args,key) < 1:
            raise ValueError(f"{key} must be positive")
    uniform = dataset_training_config()
    uniform["split_manifest"] = str((args.v4 / "split.json").resolve())
    joint = joint_training_config(str(args.v5.resolve())) if args.v5 else None
    for config in [uniform] + ([joint] if joint else []):
        config.update(epochs=args.epochs,batch_size=args.batch_size,patch_size=args.patch_size,
                      validation_patch_size=args.validation_patch_size)
    if args.train_group:
        # Pilot subsets are explicit, separate experiments. The uniform strict
        # whole-dataset split loader requires all registered train/val groups.
        # Keep production split files intact and constrain this diagnostic config.
        uniform.pop("split_manifest")
        uniform.update(data_root=str(args.v4.resolve()),train_sequences=args.train_group,
                       validation_sequences=args.validation_group)
        if joint:
            joint.update(train_groups=args.train_group,validation_groups=args.validation_group)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,config in [("uniform.json",uniform)] + ([("joint.json",joint)] if joint else []):
        (args.output/name).write_text(json.dumps(config,indent=2)+'\n')
    return {"uniform":str((args.output/'uniform.json').resolve()),
            "joint":str((args.output/'joint.json').resolve()) if joint else None,
            "scope":"diagnostic_subset" if args.train_group else "full_registered_train_validation",
            "train_groups":args.train_group or spec["partitions"]["train"],
            "validation_groups":args.validation_group or spec["partitions"]["validation"],
            "pixel_inventory_verified":False,"test_groups_selected":False,
            "note":"Metadata-only preparation; trainers verify their selected data before training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4",required=True,type=Path)
    parser.add_argument("--v5",type=Path,help="Prepare a joint config too; omit for uniform-only setup")
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--batch-size",type=int,default=4)
    parser.add_argument("--patch-size",type=int,default=128)
    parser.add_argument("--validation-patch-size",type=int,default=256)
    parser.add_argument("--train-group",action="append")
    parser.add_argument("--validation-group",action="append")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args),indent=2,allow_nan=False))
    except (OSError,ValueError,KeyError) as error:
        parser.exit(1,f"training setup: {error}\n")


if __name__ == "__main__":
    main()
