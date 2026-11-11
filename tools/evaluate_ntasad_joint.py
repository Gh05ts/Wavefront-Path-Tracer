#!/usr/bin/env python3
"""Validate a joint checkpoint on its frozen, deterministic validation crops.

This reports training-domain L1 and path accounting, not full-sequence PSNR or
tPSNR. Test-partition benchmarking is intentionally outside this command.
"""
import argparse
import json
from pathlib import Path

import torch

from wavefront_denoiser.device import select_device
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint
from wavefront_denoiser.joint_data import JointPatchDataset, joint_dataset_read_lock, joint_inventory
from wavefront_denoiser.joint_training import evaluate_joint


def evaluate(checkpoint, device_name, data_root=None):
    device = select_device(device_name)
    model,payload = load_joint_checkpoint(checkpoint,device)
    config = payload["config"]
    root = Path(data_root or config["data_root"]).resolve()
    frozen = payload["dataset_split"]
    torch.set_num_threads(config["cpu_threads"])
    with joint_dataset_read_lock(root):
        if joint_inventory(root,frozen["selected"]) != frozen:
            raise ValueError("Dataset inventory changed since training; use the frozen training snapshot")
        dataset = JointPatchDataset(root,partition="validation",sequence_ids=frozen["selected"]["validation"],
            length=config["window_length"],patch_size=config["validation_patch_size"],
            patches_per_window=config["validation_crops_per_window"],augment=False)
        metrics = evaluate_joint(model,dataset,device)
    return {"checkpoint":str(Path(checkpoint).resolve()),"device":str(device),
            "mode":"deterministic_validation_crops", "metrics":metrics,
            "model_spec":model.specification(),"validation_groups":frozen["selected"]["validation"],
            "test_pixels_loaded":False,"paper_psnr_or_tpsnr":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--data-root",type=Path,help="An identical frozen v5 copy at a different location")
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.checkpoint,args.device,args.data_root),indent=2,allow_nan=False))
    except (OSError,RuntimeError,ValueError,KeyError) as error:
        parser.exit(1,f"joint validation: {error}\n")


if __name__ == "__main__":
    main()
