from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torch import Tensor
from merging.common import strip_backbone_only




def dare_mask(
    task_vector: Tensor,
    drop_rate: float,
) -> Tensor:
    """
    DARE: Drop And REscale.

    Official rule:

        mask ~ Bernoulli(drop_rate)
        delta = delta * (1 - mask)
        delta = delta / (1 - drop_rate)
    """

    if not 0.0 <= drop_rate < 1.0:
        raise ValueError(
            f"DARE drop_rate must be in [0, 1), got {drop_rate}"
        )

    if drop_rate == 0.0:
        return task_vector.clone()

    drop_mask = torch.bernoulli(
        torch.full_like(
            task_vector,
            fill_value=drop_rate,
        )
    )

    masked = (
        task_vector
        * (1.0 - drop_mask)
    )

    masked = (
        masked
        / (1.0 - drop_rate)
    )

    return masked


@torch.no_grad()
def merge(
    task_paths: List[Path],
    dst_fold_dir: Path,
    base_model=None,
    cfg=None,
) -> None:
    """
    Continual DARE + Task Arithmetic.

    Each task vector is DARE-masked once when it arrives:

        tau_i = theta_i - theta_pretrained

        tau_i_DARE =
            Drop(tau_i) / (1 - p)

    Running merged model:

        theta_merge^t =
            theta_pretrained
            + lambda * sum_{i=0}^t tau_i_DARE

    This is mathematically equivalent to applying
    DARE + Task Arithmetic to all observed experts,
    while allowing sequential implementation.
    """

    drop_rate = float(
        getattr(
            cfg.training,
            "dare_drop_rate",
            0.9,
        )
    )

    scaling_factor = float(
        getattr(
            cfg.training,
            "dare_scaling_factor",
            0.3,
        )
    )

    seed = int(
        getattr(
            cfg.training,
            "dare_seed",
            42,
        )
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(
        "[DARE] "
        f"drop_rate={drop_rate}, "
        f"scaling_factor={scaling_factor}, "
        f"seed={seed}"
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

    cumulative_tv = {
        name: torch.zeros_like(
            pretrained_state[name]
        )
        for name in parameter_names
    }

    for task_id, task_path in enumerate(
        task_paths
    ):

        print(
            f"[DARE] Process task {task_id}"
        )

        task_state = strip_backbone_only(
            torch.load(
                task_path,
                map_location="cpu",
            )
        )

        for name in parameter_names:

            task_vector = (
                task_state[name]
                - pretrained_state[name]
            )

            if not torch.is_floating_point(
                task_vector
            ):
                continue

            masked_tv = dare_mask(
                task_vector,
                drop_rate=drop_rate,
            )

            cumulative_tv[name] = (
                cumulative_tv[name]
                + masked_tv
            )

        merged_state = {
            k: v.clone()
            for k, v
            in pretrained_state.items()
        }

        for name in parameter_names:

            if not torch.is_floating_point(
                pretrained_state[name]
            ):
                continue

            merged_state[name] = (
                pretrained_state[name]
                + scaling_factor
                * cumulative_tv[name]
            )

        torch.save(
            {
                k: v.detach().cpu()
                for k, v
                in merged_state.items()
            },
            dst_fold_dir
            / f"merged_backbone_task_{task_id}.pth",
        )

        print(
            f"[DARE] Saved task {task_id}"
        )