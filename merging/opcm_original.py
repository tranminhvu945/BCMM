from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from merging.common import strip_backbone_only



def _svd(
    w: Tensor,
    full_matrices: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:

    original_device = w.device
    original_dtype = w.dtype

    try:
        u, s, vh = torch.linalg.svd(
            w,
            full_matrices=full_matrices,
            driver="gesvd" if w.is_cuda else None,
        )

    except torch._C._LinAlgError:
        w_cpu = (
            w.detach()
            .cpu()
            .double()
        )

        u, s, vh = torch.linalg.svd(
            w_cpu,
            full_matrices=full_matrices,
        )

        u = u.to(
            device=original_device,
            dtype=original_dtype,
        )

        s = s.to(
            device=original_device,
            dtype=original_dtype,
        )

        vh = vh.to(
            device=original_device,
            dtype=original_dtype,
        )

    return u, s, vh.T

def get_task_vector_norm(
    state: Dict[str, Tensor],
    pretrained_state: Dict[str, Tensor],
) -> float:
    """
    Equivalent to:

        || state_dict_to_vector(
             state - pretrained_state
           ) ||_2

    used by official OPCM.
    """

    total = torch.tensor(
        0.0,
        dtype=torch.float64,
    )

    for key, value in state.items():

        if key not in pretrained_state:
            continue

        if not (
            torch.is_floating_point(value)
            or torch.is_complex(value)
        ):
            continue

        diff = (
            value.detach()
            .cpu()
            .double()
            - pretrained_state[key]
            .detach()
            .cpu()
            .double()
        )

        total += torch.sum(
            diff.reshape(-1) ** 2
        )

    return float(
        torch.sqrt(total).item()
    )

def merge_linear_weight(
    merged_w: Tensor,
    pretrained_w: Tensor,
    task_w: Tensor,
    previous_lambda_t: float,
    lambda_t: float,
    alpha: float,
) -> Tensor:
    """
    Exact core operator from official OPCM:

      previous_merged_tv = merged - pretrained
      task_tv            = task - pretrained

      U,S,V = SVD(previous_merged_tv)

      split_rank =
          first cumulative singular-value ratio > alpha

      projected = U^T task_tv V

      projected.diag() = 0
      projected[:split_rank, :split_rank] = 0

      cleaned = U projected V^T

      new =
          pretrained
          + (
              previous_lambda * previous_tv
              + cleaned
            ) / lambda_t
    """

    original_dtype = merged_w.dtype

    merged_w = merged_w.float()
    pretrained_w = pretrained_w.float()
    task_w = task_w.float()

    previous_merged_tv = (
        merged_w
        - pretrained_w
    )

    task_tv = (
        task_w
        - pretrained_w
    )

    if not torch.isfinite(
        previous_merged_tv
    ).all():
        raise RuntimeError(
            "Non-finite previous OPCM task vector."
        )

    if not torch.isfinite(
        task_tv
    ).all():
        raise RuntimeError(
            "Non-finite incoming OPCM task vector."
        )

    u, s, v = _svd(
        previous_merged_tv,
        full_matrices=True,
    )

    if float(s.sum()) > 0.0:

        cumulative = (
            s.cumsum(dim=0)
            / s.sum()
        )

        split_rank = (
            (
                cumulative > alpha
            )
            .float()
            .argmax()
            .item()
        )

    else:
        split_rank = 0


    projected_task_tv = (
        u.T
        @ task_tv
        @ v
    )

    # Official OPCM:
    projected_task_tv.diag().fill_(0)

    projected_task_tv[
        :split_rank,
        :split_rank
    ] = 0

    cleaned_task_tv = (
        u
        @ projected_task_tv
        @ v.T
    )

    new_merged_w = (
        pretrained_w
        + (
            previous_lambda_t
            * previous_merged_tv
            + cleaned_task_tv
        )
        / lambda_t
    )

    return (
        new_merged_w
        .to(original_dtype)
    )

def merge_other_parameter(
    merged_w: Tensor,
    pretrained_w: Tensor,
    task_w: Tensor,
    previous_lambda_t: float,
    lambda_t: float,
) -> Tensor:
    """
    Official OPCM does no projection for biases and
    non-Linear parameters.
    """

    original_dtype = merged_w.dtype

    merged_w = merged_w.float()
    pretrained_w = pretrained_w.float()
    task_w = task_w.float()

    previous_merged_tv = (
        merged_w
        - pretrained_w
    )

    task_tv = (
        task_w
        - pretrained_w
    )

    new_merged_w = (
        pretrained_w
        + (
            previous_lambda_t
            * previous_merged_tv
            + task_tv
        )
        / lambda_t
    )

    return (
        new_merged_w
        .to(original_dtype)
    )

@torch.no_grad()
def merge(
    task_paths: List[Path],
    dst_fold_dir: Path,
    base_model=None,
    cfg=None,
) -> None:
    """
    Original/upstream OPCM.

    Faithful adaptation of:

        tanganke/opcm
        fusion_bench/method/opcm/opcm.py

    Only I/O is adapted to the BCMM/TITAN checkpoint format.
    """

    alpha = float(
        getattr(
            cfg.training,
            "opcm_original_alpha",
            0.5,
        )
    )

    print(
        f"[OPCM-Original] alpha={alpha}"
    )

    pretrained_state = {
        k: v.detach().cpu().clone()
        for k, v
        in base_model
        .vision_encoder
        .state_dict()
        .items()
    }

    parameter_names = {
        name
        for name, _
        in base_model
        .vision_encoder
        .named_parameters()
    }

    linear_weight_keys = set()

    for module_name, module in (
        base_model
        .vision_encoder
        .named_modules()
    ):

        if isinstance(
            module,
            nn.Linear,
        ):
            linear_weight_keys.add(
                f"{module_name}.weight"
            )

    first_state = (
        strip_backbone_only(
            torch.load(
                task_paths[0],
                map_location="cpu",
            )
        )
    )

    merged_state = {
        k: v.clone()
        for k, v
        in first_state.items()
    }

    first_norm = get_task_vector_norm(
        first_state,
        pretrained_state,
    )

    all_task_vector_norms = [
        first_norm
    ]

    avg_task_vector_norm = (
        first_norm
    )

    previous_lambda_t = 1.0

    torch.save(
        {
            k: v.detach().cpu()
            for k, v
            in merged_state.items()
        },
        dst_fold_dir
        / "merged_backbone_task_0.pth",
    )

    print(
        "[OPCM-Original] "
        f"Initialize with task 0 | "
        f"task_norm={first_norm:.6f}"
    )

    for task_id, task_path in enumerate(
        task_paths[1:],
        start=1,
    ):

        print(
            f"[OPCM-Original] "
            f"Merge task {task_id}"
        )

        task_state = (
            strip_backbone_only(
                torch.load(
                    task_path,
                    map_location="cpu",
                )
            )
        )

        current_task_norm = (
            get_task_vector_norm(
                task_state,
                pretrained_state,
            )
        )

        all_task_vector_norms.append(
            current_task_norm
        )

        avg_task_vector_norm = float(
            np.mean(
                all_task_vector_norms
            )
        )

        lambda_t = 1.0

        new_state = {
            k: v.clone()
            for k, v
            in merged_state.items()
        }


        for name in parameter_names:

            merged_param = (
                merged_state[name]
            )

            pretrained_param = (
                pretrained_state[name]
            )

            task_param = (
                task_state[name]
            )

            if (
                name in linear_weight_keys
                and merged_param.ndim == 2
            ):

                new_state[name] = (
                    merge_linear_weight(
                        merged_w=merged_param,
                        pretrained_w=pretrained_param,
                        task_w=task_param,
                        previous_lambda_t=(
                            previous_lambda_t
                        ),
                        lambda_t=lambda_t,
                        alpha=alpha,
                    )
                )

            else:

                new_state[name] = (
                    merge_other_parameter(
                        merged_w=merged_param,
                        pretrained_w=pretrained_param,
                        task_w=task_param,
                        previous_lambda_t=(
                            previous_lambda_t
                        ),
                        lambda_t=lambda_t,
                    )
                )

        merged_task_vector_norm = (
            get_task_vector_norm(
                new_state,
                pretrained_state,
            )
        )

        lambda_t *= (
            merged_task_vector_norm
            / avg_task_vector_norm
        )

        norm_scale = (
            avg_task_vector_norm
            / merged_task_vector_norm
        )

        for name in parameter_names:

            if not torch.is_floating_point(
                new_state[name]
            ):
                continue

            new_state[name] = (
                pretrained_state[name]
                + (
                    new_state[name]
                    - pretrained_state[name]
                )
                * norm_scale
            )

        previous_lambda_t = (
            lambda_t
        )

        merged_state = (
            new_state
        )

        final_norm = (
            get_task_vector_norm(
                merged_state,
                pretrained_state,
            )
        )

        print(
            "[OPCM-Original] "
            f"task={task_id} | "
            f"task_norm={current_task_norm:.6f} | "
            f"avg_norm={avg_task_vector_norm:.6f} | "
            f"raw_norm={merged_task_vector_norm:.6f} | "
            f"lambda={lambda_t:.6f} | "
            f"final_norm={final_norm:.6f}"
        )

        torch.save(
            {
                k: v.detach().cpu()
                for k, v
                in merged_state.items()
            },
            dst_fold_dir
            / (
                f"merged_backbone_task_"
                f"{task_id}.pth"
            ),
        )