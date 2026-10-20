from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor, nn
from merging.common import strip_backbone_only
from tqdm.auto import tqdm

from utils.model import get_task_vector_norm, is_leaf_module, svd




def merge_linear_weights(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    param_name: str,
    alpha: float,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    original_device = merged_W.device

    merged_W = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv = task_W - pretrained_W

    u, _s, v = svd(previous_merged_tv)

    projected_task_tv = u.T @ task_tv @ v
    projected_task_tv.diag().fill_(0)

    cleaned_task_tv = u @ projected_task_tv @ v.T

    new_merged_W = (
        pretrained_W
        + (
            previous_lambda_t * previous_merged_tv
            + cleaned_task_tv
        )
        / lambda_t
    )

    return new_merged_W.to(original_device)


def merge_other_parameters(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    param_name: str,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    original_device = merged_W.device

    merged_W = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv = task_W - pretrained_W

    new_merged_W = (
        pretrained_W
        + (
            previous_lambda_t * previous_merged_tv
            + task_tv
        )
        / lambda_t
    )

    return new_merged_W.to(original_device)


def merge(task_paths, dst_fold_dir, base_model=None, cfg=None,):
    alpha = float(getattr(cfg.training, "opcm_alpha",0.5,))
    
    base_weight = {k: v.detach().cpu() for k, v in base_model.vision_encoder.state_dict().items()}

    all_task_vector_norm: List[float] = []

    # First task
    task_weight = strip_backbone_only(torch.load(task_paths[0], map_location="cpu"))

    merged_weight = dict(task_weight)

    all_task_vector_norm.append(float(get_task_vector_norm(task_weight, base_weight)))

    previous_lambda_t = 1.0

    torch.save(
        {k: v.detach().cpu() for k, v in merged_weight.items()},
        dst_fold_dir / "merged_backbone_task_0.pth",
    )

    # Remaining tasks
    for model_idx, task_path in enumerate(
        task_paths[1:], start=1
    ):
        task_weight = strip_backbone_only(
            torch.load(task_path, map_location="cpu")
        )

        all_task_vector_norm.append(
            float(get_task_vector_norm(task_weight, base_weight))
        )

        avg_task_vector_norm = float(
            np.mean(all_task_vector_norm)
        )

        lambda_t = 1.0

        for module_name, module in tqdm(
            list(base_model.vision_encoder.named_modules()),
            desc=f"OPCM merge task {model_idx}",
            leave=False,
        ):
            if not is_leaf_module(module):
                continue

            if isinstance(module, nn.Linear):
                weight_key = f"{module_name}.weight"

                merged_weight[weight_key] = merge_linear_weights(
                    merged_weight[weight_key].detach(),
                    base_model.vision_encoder
                    .get_submodule(module_name)
                    .weight.detach()
                    .cpu(),
                    task_weight[weight_key].detach(),
                    weight_key,
                    alpha,
                    previous_lambda_t,
                    lambda_t,
                )

                if module.bias is not None:
                    bias_key = f"{module_name}.bias"

                    merged_weight[bias_key] = merge_other_parameters(
                        merged_weight[bias_key].detach(),
                        base_model.vision_encoder
                        .get_submodule(module_name)
                        .bias.detach()
                        .cpu(),
                        task_weight[bias_key].detach(),
                        bias_key,
                        previous_lambda_t,
                        lambda_t,
                    )

            else:
                for param_name, _ in module.named_parameters(
                    recurse=False
                ):
                    key = f"{module_name}.{param_name}"

                    merged_weight[key] = merge_other_parameters(
                        merged_weight[key].detach(),
                        base_model.vision_encoder
                        .get_submodule(module_name)
                        .get_parameter(param_name)
                        .detach()
                        .cpu(),
                        task_weight[key].detach(),
                        key,
                        previous_lambda_t,
                        lambda_t,
                    )

        task_vector_norm = float(
            get_task_vector_norm(
                merged_weight,
                base_weight,
            )
        )

        lambda_t *= (
            task_vector_norm
            / max(avg_task_vector_norm, 1e-12)
        )

        previous_lambda_t = lambda_t

        for param_name, _ in (
            base_model.vision_encoder.named_parameters()
        ):
            base_param = (
                base_model.vision_encoder
                .get_parameter(param_name)
                .detach()
                .cpu()
            )

            merged_weight[param_name] = (
                base_param
                + (
                    merged_weight[param_name]
                    - base_param
                )
                * (
                    avg_task_vector_norm
                    / max(task_vector_norm, 1e-12)
                )
            )

        torch.save(
            {
                k: v.detach().cpu()
                for k, v in merged_weight.items()
            },
            dst_fold_dir
            / f"merged_backbone_task_{model_idx}.pth",
        )