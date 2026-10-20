from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from merging.common import strip_backbone_only




@torch.no_grad()
def merge(task_paths: List[Path], dst_fold_dir: Path, base_model=None, cfg=None,):
    """
    Continual Task Arithmetic.

    Faithful adaptation of FusionBench
    ContinualTaskArithmeticForCLIP:

        merged <- merged
                  + scaling_factor
                    * (task - pretrained)

    The merged model starts from the pretrained backbone.
    """

    scaling_factor = float(getattr(cfg.training, "task_arithmetic_scaling_factor", 0.3,))

    print(
        f"[Task Arithmetic] "
        f"scaling_factor={scaling_factor}"
    )

    pretrained_weight = {k: v.detach().cpu().clone() for k, v in base_model.vision_encoder.state_dict().items()}

    merged_weight = {k: v.clone() for k, v in pretrained_weight.items()}

    parameter_names = {name for name, _ in base_model.vision_encoder.named_parameters()}

    for task_id, task_path in enumerate(task_paths):
        print(f"[Task Arithmetic] Merge task {task_id}")

        task_weight = strip_backbone_only(torch.load(task_path, map_location="cpu",))

        for key in parameter_names:
            merged_weight[key] = (merged_weight[key] + scaling_factor * (task_weight[key] - pretrained_weight[key]))

        torch.save({k: v.detach().cpu() for k, v in merged_weight.items()}, dst_fold_dir / f"merged_backbone_task_{task_id}.pth",)