from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from merging.common import strip_backbone_only




def merge(task_paths: List[Path], dst_fold_dir: Path, base_model=None,cfg=None,):
    task_weight = strip_backbone_only(torch.load(task_paths[0], map_location="cpu",), strict=False)

    merged_weight = {k: v.clone() for k, v in task_weight.items()}
    torch.save(merged_weight, dst_fold_dir / "merged_backbone_task_0.pth",)

    print("[Weight Average] Initialize with task 0")

    for task_id, task_path in enumerate(task_paths[1:], start=1,):
        task_weight = strip_backbone_only(torch.load(task_path, map_location="cpu",), strict=False)

        for key in merged_weight:
            if torch.is_floating_point(merged_weight[key]):
                merged_weight[key] = (task_id * merged_weight[key] + task_weight[key]) / (task_id + 1)

            else:
                merged_weight[key] = (task_weight[key].clone())

        torch.save({k: v.detach().cpu() for k, v in merged_weight.items()}, dst_fold_dir / f"merged_backbone_task_{task_id}.pth",)

        print(
            f"[Weight Average] "
            f"Merged task {task_id}"
        )