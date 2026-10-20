from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor, nn
from merging.common import strip_backbone_only
from tqdm.auto import tqdm



def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def _svd(
    w: Tensor,
    full_matrices: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:

    u, s, vh = torch.linalg.svd(
        w,
        full_matrices=full_matrices,
        driver="gesvd" if w.is_cuda else None,
    )

    v = vh.T

    return u, s, v

def simple_average_tensor(
    a: Tensor,
    b: Tensor,
) -> Tensor:
    if torch.is_floating_point(a):
        return (a + b) / 2.0
    return b.clone()


def min_norm_two(
    grad_0: Tensor,
    grad_1: Tensor,
) -> List[float]:

    v1v1 = float(
        torch.sum(grad_0 * grad_0)
        .detach()
        .cpu()
        .item()
    )

    v1v2 = float(
        torch.sum(grad_0 * grad_1)
        .detach()
        .cpu()
        .item()
    )

    v2v2 = float(
        torch.sum(grad_1 * grad_1)
        .detach()
        .cpu()
        .item()
    )

    if v1v2 >= v1v1:
        gamma = 0.999

    elif v1v2 >= v2v2:
        gamma = 0.001

    else:
        gamma = -(
            (v1v2 - v2v2)
            /
            (
                v1v1
                + v2v2
                - 2.0 * v1v2
            )
        )

    return [
        float(gamma),
        float(1.0 - gamma),
    ]

def projection_loss(
    delta_tv: Tensor,
    proj_s: Tensor,
    proj_u: Tensor,
    proj_v: Tensor,
    proj_space: str,
) -> Tensor:

    if proj_s.numel() == 0:
      return delta_tv.sum() * 0.0

    diag_s = torch.diag(proj_s)

    proj_delta_u = (
        diag_s
        @ proj_u.T
        @ delta_tv
    )

    proj_delta_v = (
        delta_tv
        @ proj_v
        @ diag_s
    )

    loss_u = (
        torch.linalg.matrix_norm(
            proj_delta_u,
            ord="fro",
        )
        ** 2
    )

    loss_v = (
        torch.linalg.matrix_norm(
            proj_delta_v,
            ord="fro",
        )
        ** 2
    )

    if proj_space == "uv":
        return loss_u + loss_v

    if proj_space == "u":
        return loss_u

    if proj_space == "v":
        return loss_v

    raise ValueError(
        f"Invalid svd_proj_space: {proj_space}"
    )

def optimize_linear_weight(
    pretrained_weight: Tensor,
    previous_weight: Tensor,
    current_weight: Tensor,
    lr: float,
    num_steps: int,
    mgda: bool,
    alpha: float,
    ema: bool,
    ema_beta: float,
    svd_epsilon: float,
    svd_proj_space: str,
    device: torch.device,
    module_name: str,
) -> Tensor:

    pretrained_weight = (
        pretrained_weight
        .detach()
        .to(device)
        .float()
    )

    previous_weight = (
        previous_weight
        .detach()
        .to(device)
        .float()
    )

    current_weight = (
        current_weight
        .detach()
        .to(device)
        .float()
    )

    finetuned_weights = [
        previous_weight,
        current_weight,
    ]

    merged_weight = nn.Parameter(
        (
            previous_weight
            + current_weight
        )
        / 2.0,
        requires_grad=True,
    )

    proj_u_list = []
    proj_v_list = []
    proj_s_list = []

    for finetuned_weight in finetuned_weights:

        finetuned_tv = (finetuned_weight - pretrained_weight)

        if not torch.isfinite(finetuned_tv).all():
            raise RuntimeError(f"[DOP] Non-finite task vector before SVD at layer {module_name}")
    
        tv_norm = torch.linalg.vector_norm(finetuned_tv)
    
        if tv_norm <= 1e-12:
            rows, cols = finetuned_tv.shape
    
            proj_u_list.append(
                torch.empty(
                    rows,
                    0,
                    device=device,
                    dtype=finetuned_tv.dtype,
                )
            )
    
            proj_v_list.append(
                torch.empty(
                    cols,
                    0,
                    device=device,
                    dtype=finetuned_tv.dtype,
                )
            )
    
            proj_s_list.append(
                torch.empty(
                    0,
                    device=device,
                    dtype=finetuned_tv.dtype,
                )
            )
            continue
            
        u, s, v = _svd(
            finetuned_tv,
            full_matrices=True,
        )

        epsilon = min(
            float(svd_epsilon),
            1.0,
        )

        cumsum_ratio = (
            s.cumsum(dim=0)
            / s.sum()
        )

        split_rank = int(
            torch.searchsorted(
                cumsum_ratio,
                epsilon,
            ).item()
        )

        u_main = u[:, :split_rank]
        v_main = v[:, :split_rank]
        s_main = s[:split_rank]

        proj_u_list.append(u_main)
        proj_v_list.append(v_main)
        proj_s_list.append(s_main)

    optimizer = torch.optim.Adam(
        [merged_weight],
        lr=lr,
    )

    if mgda:

        ema_sol = [
            float(alpha),
            float(1.0 - alpha),
        ]

        for _ in tqdm(
            range(num_steps),
            desc=f"DOP {module_name}",
            leave=False,
        ):

            losses_data = {}
            grads = {}
            
            for i in range(2):

                delta_tv = (
                    merged_weight
                    - finetuned_weights[i]
                )

                loss_i = projection_loss(
                    delta_tv=delta_tv,
                    proj_s=proj_s_list[i],
                    proj_u=proj_u_list[i],
                    proj_v=proj_v_list[i],
                    proj_space=svd_proj_space,
                )

                losses_data[i] = float(
                    loss_i.detach()
                    .cpu()
                    .item()
                )

                optimizer.zero_grad()

                loss_i.backward()

                grads[i] = (
                    merged_weight.grad
                    .detach()
                    .clone()
                )

            eps = 1e-12

            grads[0] = (
                grads[0]
                / max(float(losses_data[0]), eps)
            )
            
            grads[1] = (
                grads[1]
                / max(float(losses_data[1]), eps)
            )

            sol = min_norm_two(
                grads[0],
                grads[1],
            )

            if ema:
                ema_sol = [
                    (
                        ema_beta
                        * ema_sol[i]
                        + (1.0 - ema_beta)
                        * float(sol[i])
                    )
                    for i in range(2)
                ]

                sol = ema_sol

            loss = 0.0

            for i in range(2):

                delta_tv = (
                    merged_weight
                    - finetuned_weights[i]
                )

                loss_i = projection_loss(
                    delta_tv=delta_tv,
                    proj_s=proj_s_list[i],
                    proj_u=proj_u_list[i],
                    proj_v=proj_v_list[i],
                    proj_space=svd_proj_space,
                )

                loss = (
                    loss
                    + float(sol[i]) * loss_i
                )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            if not torch.isfinite(merged_weight).all():
                raise RuntimeError(
                    f"[DOP] Non-finite merged weight "
                    f"at layer={module_name}, "
                    f"optimization_step={_}"
                )

    else:

        for _ in tqdm(
            range(num_steps),
            desc=f"DOP {module_name}",
            leave=False,
        ):

            loss = 0.0

            for i in range(2):

                delta_tv = (
                    merged_weight
                    - finetuned_weights[i]
                )

                loss_i = projection_loss(
                    delta_tv=delta_tv,
                    proj_s=proj_s_list[i],
                    proj_u=proj_u_list[i],
                    proj_v=proj_v_list[i],
                    proj_space=svd_proj_space,
                )

                if i == 0:
                    loss = loss + alpha * loss_i
                else:
                    loss = (
                        loss
                        + (1.0 - alpha) * loss_i
                    )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            if not torch.isfinite(merged_weight).all():
                raise RuntimeError(
                    f"[DOP] Non-finite merged weight "
                    f"at layer={module_name}, "
                    f"optimization_step={_}"
                )

    return (
        merged_weight
        .detach()
        .cpu()
        .to(previous_weight.dtype)
    )

def merge(
    task_paths: List[Path],
    dst_fold_dir: Path,
    base_model=None,
    cfg=None,
) -> None:

    lr = float(
        getattr(
            cfg.training,
            "dop_lr",
            1e-4,
        )
    )

    num_steps = int(
        getattr(
            cfg.training,
            "dop_num_steps",
            200,
        )
    )

    mgda = bool(
        getattr(
            cfg.training,
            "dop_mgda",
            True,
        )
    )

    alpha = float(
        getattr(
            cfg.training,
            "dop_alpha",
            0.8,
        )
    )

    ema = bool(
        getattr(
            cfg.training,
            "dop_ema",
            True,
        )
    )

    ema_beta = float(
        getattr(
            cfg.training,
            "dop_ema_beta",
            0.999,
        )
    )

    svd_epsilon = float(
        getattr(
            cfg.training,
            "dop_svd_epsilon",
            0.99999,
        )
    )

    svd_proj_space = str(
        getattr(
            cfg.training,
            "dop_svd_proj_space",
            "uv",
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "[DOP] "
        f"lr={lr}, "
        f"steps={num_steps}, "
        f"mgda={mgda}, "
        f"alpha={alpha}, "
        f"ema={ema}, "
        f"ema_beta={ema_beta}, "
        f"svd_epsilon={svd_epsilon}, "
        f"proj={svd_proj_space}"
    )

    pretrained_state = {
        k: v.detach().cpu().clone()
        for k, v
        in base_model
        .vision_encoder
        .state_dict()
        .items()
    }

    first_state = strip_backbone_only(
        torch.load(
            task_paths[0],
            map_location="cpu",
        )
    )

    merged_state = {
        k: v.clone()
        for k, v in first_state.items()
    }

    torch.save(
        merged_state,
        dst_fold_dir
        / "merged_backbone_task_0.pth",
    )

    print(
        "[DOP] Initialize merged model with task 0"
    )

    for task_id, task_path in enumerate(
        task_paths[1:],
        start=1,
    ):

        print(
            f"[DOP] Merge task {task_id}"
        )

        current_state = strip_backbone_only(
            torch.load(
                task_path,
                map_location="cpu",
            )
        )

        new_state = {
            k: v.clone()
            for k, v
            in merged_state.items()
        }

        for module_name, module in (
            base_model
            .vision_encoder
            .named_modules()
        ):

            if not is_leaf_module(module):
                continue

            if isinstance(
                module,
                nn.Linear,
            ):

                weight_key = (
                    f"{module_name}.weight"
                )

                # DOP optimization is only for Linear weight
                new_state[weight_key] = (
                    optimize_linear_weight(
                        pretrained_weight=(
                            pretrained_state[
                                weight_key
                            ]
                        ),
                        previous_weight=(
                            merged_state[
                                weight_key
                            ]
                        ),
                        current_weight=(
                            current_state[
                                weight_key
                            ]
                        ),
                        lr=lr,
                        num_steps=num_steps,
                        mgda=mgda,
                        alpha=alpha,
                        ema=ema,
                        ema_beta=ema_beta,
                        svd_epsilon=(
                            svd_epsilon
                        ),
                        svd_proj_space=(
                            svd_proj_space
                        ),
                        device=device,
                        module_name=(
                            module_name
                        ),
                    )
                )

                # Official DOP simply averages bias
                if module.bias is not None:

                    bias_key = (
                        f"{module_name}.bias"
                    )

                    new_state[bias_key] = (
                        simple_average_tensor(
                            merged_state[
                                bias_key
                            ],
                            current_state[
                                bias_key
                            ],
                        )
                    )

            else:

                for param_name, _ in (
                    module.named_parameters(
                        recurse=False
                    )
                ):

                    key = (
                        f"{module_name}."
                        f"{param_name}"
                    )

                    new_state[key] = (
                        simple_average_tensor(
                            merged_state[key],
                            current_state[key],
                        )
                    )

        merged_state = new_state

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