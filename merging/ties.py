from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor, nn
from merging.common import strip_backbone_only




# =========================================================
# Utilities adapted directly from FusionBench TIES
# =========================================================

def state_dict_to_vector(
    state_dict: Dict[str, Tensor],
    remove_keys: List[str] | None = None,
) -> Tensor:
    if remove_keys is None:
        remove_keys = []

    shared_state_dict = copy.deepcopy(
        state_dict
    )

    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]

    sorted_shared_state_dict = OrderedDict(
        sorted(shared_state_dict.items())
    )

    return nn.utils.parameters_to_vector(
        [
            value.reshape(-1)
            for _, value
            in sorted_shared_state_dict.items()
        ]
    )


def vector_to_state_dict(
    vector: Tensor,
    state_dict: Dict[str, Tensor],
    remove_keys: List[str] | None = None,
) -> Dict[str, Tensor]:
    if remove_keys is None:
        remove_keys = []

    reference_dict = copy.deepcopy(
        state_dict
    )

    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]

    sorted_reference_dict = OrderedDict(
        sorted(reference_dict.items())
    )

    nn.utils.vector_to_parameters(
        vector,
        sorted_reference_dict.values(),
    )

    return sorted_reference_dict


def topk_values_mask(
    M: Tensor,
    K: float = 0.7,
    return_mask: bool = False,
):
    if K > 1:
        K /= 100

    original_shape = M.shape

    if M.dim() == 1:
        M = M.unsqueeze(0)

    _, d = M.shape

    k = int(d * K)
    k = d - k

    kth_values, _ = M.abs().kthvalue(
        k,
        dim=1,
        keepdim=True,
    )

    mask = M.abs() >= kth_values

    final_mask = (
        mask.squeeze()
        if original_shape == M.squeeze().shape
        else mask
    )

    if return_mask:
        return (
            M * final_mask,
            final_mask.float().mean(dim=1),
            final_mask,
        )

    return (
        M * final_mask,
        final_mask.float().mean(dim=1),
    )


def resolve_zero_signs(
    sign_to_mult: Tensor,
    method: str = "majority",
) -> Tensor:
    majority_sign = torch.sign(
        sign_to_mult.sum()
    )

    if method == "majority":
        sign_to_mult[
            sign_to_mult == 0
        ] = majority_sign

    elif method == "minority":
        sign_to_mult[
            sign_to_mult == 0
        ] = -1 * majority_sign

    return sign_to_mult


def resolve_sign(
    v: Tensor,
) -> Tensor:
    sign_to_mult = torch.sign(
        v.sum(dim=0)
    )

    sign_to_mult = resolve_zero_signs(
        sign_to_mult,
        "majority",
    )

    return sign_to_mult


def disjoint_merge(
    v: Tensor,
    merge_func: str,
    sign_to_mult: Tensor,
) -> Tensor:
    merge_func = merge_func.split("-")[-1]

    if sign_to_mult is not None:
        rows_to_keep = torch.where(
            sign_to_mult.unsqueeze(0) > 0,
            v > 0,
            v < 0,
        )

        selected_entries = (
            v * rows_to_keep
        )

    else:
        rows_to_keep = v != 0

        selected_entries = (
            v * rows_to_keep
        )

    if merge_func == "mean":
        non_zero_counts = (
            selected_entries != 0
        ).sum(
            dim=0
        ).float()

        disjoint_aggs = (
            torch.sum(
                selected_entries,
                dim=0,
            )
            / torch.clamp(
                non_zero_counts,
                min=1,
            )
        )

    elif merge_func == "sum":
        disjoint_aggs = torch.sum(
            selected_entries,
            dim=0,
        )

    elif merge_func == "max":
        disjoint_aggs = (
            selected_entries
            .abs()
            .max(dim=0)[0]
        )

        disjoint_aggs *= sign_to_mult

    else:
        raise ValueError(
            f"Merge method {merge_func} "
            f"is not defined."
        )

    return disjoint_aggs


def ties_merging(
    flat_task_checks: Tensor,
    reset_thresh: float | None = None,
    merge_func: str = "",
) -> Tensor:
    all_checks = (
        flat_task_checks.clone()
    )

    updated_checks, *_ = (
        topk_values_mask(
            all_checks,
            K=reset_thresh,
            return_mask=False,
        )
    )

    print("RESOLVING SIGN")

    final_signs = resolve_sign(
        updated_checks
    )

    print(
        f"Disjoint AGGREGATION: "
        f"{merge_func}"
    )

    merged_tv = disjoint_merge(
        updated_checks,
        merge_func,
        final_signs,
    )

    return merged_tv

@torch.no_grad()
def merge(
    task_paths: List[Path],
    dst_fold_dir: Path,
    base_model=None,
    cfg=None,
) -> None:
    """
    Continualized TIES-Merging.

    FusionBench TIES itself is an offline merging method.

    For continual step t, we apply the same FusionBench
    TIES operation to all task models observed so far:

        theta_t =
            theta_pretrained
            + lambda * TIES(
                tau_0, ..., tau_t
            )

    where:
        tau_i = theta_i - theta_pretrained
    """

    scaling_factor = float(
        getattr(
            cfg.training,
            "ties_scaling_factor",
            0.3,
        )
    )

    threshold = float(
        getattr(
            cfg.training,
            "ties_threshold",
            20,
        )
    )

    merge_func = str(
        getattr(
            cfg.training,
            "ties_merge_func",
            "sum",
        )
    )

    remove_keys: List[str] = []

    print(
        f"[TIES] scaling_factor="
        f"{scaling_factor}, "
        f"threshold={threshold}, "
        f"merge_func={merge_func}"
    )
    
    ptm_check = {
        k: v.detach().cpu().clone()
        for k, v
        in base_model
        .vision_encoder
        .state_dict()
        .items()
    }

    flat_ptm = state_dict_to_vector(
        ptm_check,
        remove_keys=remove_keys,
    )

    task_vectors = []

    for task_path in task_paths:
        ft_check = strip_backbone_only(
            torch.load(
                task_path,
                map_location="cpu",
            )
        )

        flat_ft = state_dict_to_vector(
            ft_check,
            remove_keys=remove_keys,
        )

        task_vectors.append(
            flat_ft - flat_ptm
        )

    for step in range(
        len(task_paths)
    ):
        print(
            f"[TIES] Merge step {step} "
            f"with {step + 1} task(s)"
        )

        tv_flat_checks = torch.vstack(
            task_vectors[
                : step + 1
            ]
        )

        merged_tv = ties_merging(
            tv_flat_checks,
            reset_thresh=threshold,
            merge_func=merge_func,
        )
        
        merged_check = (
            flat_ptm
            + scaling_factor
            * merged_tv
        )

        merged_state = (
            vector_to_state_dict(
                merged_check,
                ptm_check,
                remove_keys=remove_keys,
            )
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
                f"{step}.pth"
            ),
        )