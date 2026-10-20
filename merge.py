from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from transformers import AutoModel

from utils.config import add_config_arg, ensure_dir, get_task_order, load_config
from merging import get_merger


def strip_head_only(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    head_state = {k.replace("mlp.", "", 1): v.detach().cpu() for k, v in state_dict.items() if k.startswith("mlp.")}
    if not head_state:
        raise KeyError("Checkpoint has no mlp.* Cox head parameters.")
    return head_state


def build_task_checkpoint_map(fold_src_dir: Path) -> Dict[str, Path]:
    task_to_checkpoint: Dict[str, Path] = {}

    for meta_path in sorted(fold_src_dir.glob("finetune_task_*_metadata.json")):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        task_name = str(metadata["task_name"]).lower()
        checkpoint_path = meta_path.with_name(meta_path.name.replace("_metadata.json", ".pt"))
        task_to_checkpoint[task_name] = checkpoint_path

    if not task_to_checkpoint:
        raise FileNotFoundError(
            f"No finetune_task_*_metadata.json files found in {fold_src_dir}. "
            "Task metadata is required to reuse the same experts across different task orders."
        )

    return task_to_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual model merging for WSI survival")
    add_config_arg(parser)
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "opcm",
            "weight_average",
            "task_arithmetic",
            "ties",
            "dop",
            "nufilt",
            "dare",
            "opcm_original",
        ],
    )
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--src_finedtuned_checkpoints", type=str, default=None)
    parser.add_argument("--des_merged_checkpoints", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_order = get_task_order(cfg)[: int(args.num_tasks or cfg.training.num_tasks)]
    num_tasks = len(task_order)
    src_root = Path(args.src_finedtuned_checkpoints or cfg.paths.finetuned_checkpoints)
    dst_root = ensure_dir(Path(args.des_merged_checkpoints or cfg.paths.merged_checkpoints) / args.method)

    merger = get_merger(args.method)

    print("Loading TITAN base model ...")
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)

    for fold_idx in range(int(cfg.training.num_folds)):
        fold = f"fold_{fold_idx}"
        dst_fold_dir = ensure_dir(dst_root / fold)

        fold_src_dir = src_root / fold
        task_checkpoint_map = build_task_checkpoint_map(fold_src_dir)

        missing_tasks = [task for task in task_order if task.lower() not in task_checkpoint_map]
        if missing_tasks:
            raise FileNotFoundError(
                f"[{fold}] Missing fine-tuned expert metadata/checkpoints for tasks: {missing_tasks}"
            )

        task_paths = [task_checkpoint_map[task.lower()] for task in task_order]
        missing = [str(p) for p in task_paths if not p.exists()]
        if missing:
            raise FileNotFoundError("Missing fine-tuned checkpoints:\n" + "\n".join(missing))

        print(f"[{fold}] Task order: {task_order}")
        for step, (task_name, task_path) in enumerate(zip(task_order, task_paths)):
            print(f"  step {step}: {task_name.upper()} <- {task_path.name}")

        print(f"[{fold}] Merging backbone with {args.method} ...")
        merger(task_paths=task_paths, dst_fold_dir=dst_fold_dir, base_model=base_model, cfg=cfg,)

        seen_heads: Dict[int, Dict[str, Tensor]] = {}
        for task_id, task_path in enumerate(task_paths):
            seen_heads[task_id] = strip_head_only(torch.load(task_path, map_location="cpu"))
            out_heads = dst_fold_dir / f"task_heads_until_task_{task_id}.pth"
            torch.save({"task_heads": seen_heads, "task_names": task_order[: task_id + 1]}, out_heads)

            package = {
                "method": args.method,
                "fold": fold,
                "seen_task_ids": list(range(task_id + 1)),
                "seen_task_names": task_order[: task_id + 1],
                "backbone_path": str(dst_fold_dir / f"merged_backbone_task_{task_id}.pth"),
                "task_heads_path": str(out_heads),
            }

            with open(dst_fold_dir / f"merged_package_task_{task_id}.json", "w", encoding="utf-8") as f:
                json.dump(package, f, indent=2)
            print(f"Saved {out_heads}")


if __name__ == "__main__":
    main()
