from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor, nn
from merging.common import strip_backbone_only
from tqdm.auto import tqdm




def svd(
    w: Tensor,
    full_matrices: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:

    u, s, vh = torch.linalg.svd(
        w,
        full_matrices=full_matrices,
        driver="gesvd" if w.is_cuda else None,
    )

    v = vh.T

    return u, s, v


# =========================================================
# Residual Projection Gate
#
# Official NUFILT:
#
#   P = I - U U^T
#
#   W_gate = P + B A
# =========================================================

class ResidualProjGate(nn.Module):
    def __init__(
        self,
        d: int,
        r: int,
        U: Tensor | None = None,
    ):
        super().__init__()

        self.U = U

        r = max(int(r), 1)

        self.A = nn.Parameter(
            torch.zeros(
                r,
                d,
                device=U.device if U is not None else None,
            )
        )

        self.B = nn.Parameter(
            torch.zeros(
                d,
                r,
                device=U.device if U is not None else None,
            )
        )

        nn.init.normal_(
            self.A,
            std=0.02,
        )

        nn.init.zeros_(
            self.B
        )

    @torch.no_grad()
    def get_weight(self) -> Tensor:

        d = self.A.shape[1]

        device = self.A.device
        dtype = self.A.dtype

        if self.U is not None:
            P = (
                torch.eye(
                    d,
                    device=device,
                    dtype=dtype,
                )
                - self.U @ self.U.T
            )

        else:
            P = torch.eye(
                d,
                device=device,
                dtype=dtype,
            )

        return (
            P
            + self.B @ self.A
        )

class FilterLoRAWeight(nn.Module):
    def __init__(
        self,
        base_weight: Tensor,
        task_tv: Tensor,
        previous_merged_tv: Tensor,
        null_space: bool,
        rank: int,
        null_rank: int,
        grad_rank: int,
    ):
        super().__init__()

        self.register_buffer(
            "base_weight",
            base_weight.detach().clone(),
        )

        self.register_buffer(
            "task_tv",
            task_tv.detach().clone(),
        )

        self.register_buffer(
            "prev_tv",
            previous_merged_tv.detach().clone(),
        )

        _, _, pre_V = svd(
            previous_merged_tv,
            full_matrices=False,
        )

        _, _, task_V = svd(
            task_tv,
            full_matrices=False,
        )

        self.pre_V = pre_V[
            :,
            :null_rank,
        ]

        if grad_rank > 0:
            self.task_V = task_V[
                :,
                :grad_rank,
            ]
        else:
            self.task_V = None

        self.U = (
            self.pre_V
            if null_space
            else None
        )

        self.gate = ResidualProjGate(
            d=base_weight.shape[1],
            r=rank,
            U=self.U,
        ).to(base_weight.device)


    def solve_lora(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
    ) -> Tensor:
        T = self.task_tv

        A = self.gate.A
        B = self.gate.B

        prev = self.prev_tv

        U = self.gate.U

        loss = torch.tensor(
            0.0,
            device=T.device,
            dtype=T.dtype,
        )

        has_proj1 = (
            self.task_V is not None
            and lambda1 != 0.0
        )

        if has_proj1:
            V = self.task_V

            target = T @ V

            prev_v = prev @ V

            if U is not None:
                u_t_v = U.T @ V
                u_utv = U @ u_t_v

                p_v = V - u_utv
            else:
                p_v = V

            t_p_v = T @ p_v

            m0_v = (
                prev_v
                + t_p_v
            )

            residual = (
                target
                - m0_v
            )

            a_v = A @ V
            b_av = B @ a_v
            t_bav = T @ b_av

            r = (
                residual
                - t_bav
            )

            loss = (
                loss
                + lambda1
                * torch.sum(r * r)
            )

        has_proj2 = (
            self.pre_V is not None
            and lambda2 != 0.0
        )

        if has_proj2:
            V = self.pre_V

            target = prev @ V

            prev_v = prev @ V

            if U is not None:
                u_t_v = U.T @ V
                u_utv = U @ u_t_v

                p_v = V - u_utv
            else:
                p_v = V

            t_p_v = T @ p_v

            m0_v = (
                prev_v
                + t_p_v
            )

            residual = (
                target
                - m0_v
            )

            a_v = A @ V
            b_av = B @ a_v
            t_bav = T @ b_av

            r = (
                residual
                - t_bav
            )

            loss = (
                loss
                + lambda2
                * torch.sum(r * r)
            )

        if not has_proj1 and not has_proj2:

            if U is not None:
                t_u = T @ U
                t_u_ut = (
                    t_u @ U.T
                )

                t_p = (
                    T - t_u_ut
                )
            else:
                t_p = T

            m0 = (
                prev
                + t_p
            )

            t_b = T @ B
            delta = t_b @ A

            m = (
                m0
                + delta
            )

            r_full1 = T - m
            r_full2 = prev - m

            loss = (
                loss
                + torch.sum(
                    r_full1 * r_full1
                )
                + torch.sum(
                    r_full2 * r_full2
                )
            )

        return loss


    @torch.no_grad()
    def merge_to_base(self) -> Tensor:
        """
        Fuse filter + LoRA into the Linear weight.
        No extra inference-time parameters remain.
        """

        gate_w = (
            self.gate.get_weight()
        )

        return (
            self.base_weight
            + self.task_tv @ gate_w
        )

def merge(
    task_paths: List[Path],
    dst_fold_dir: Path,
    base_model=None,
    cfg=None,
) -> None:

    seed = int(
        getattr(
            cfg.training,
            "nufilt_seed",
            42,
        )
    )

    null_space = bool(
        getattr(
            cfg.training,
            "nufilt_null_space",
            True,
        )
    )

    lr = float(
        getattr(
            cfg.training,
            "nufilt_lr",
            1e-3,
        )
    )

    max_steps = int(
        getattr(
            cfg.training,
            "nufilt_max_steps",
            50,
        )
    )

    lora_r = int(
        getattr(
            cfg.training,
            "nufilt_lora_r",
            64,
        )
    )

    grad_r = int(
        getattr(
            cfg.training,
            "nufilt_grad_r",
            8,
        )
    )

    null_r = int(
        getattr(
            cfg.training,
            "nufilt_null_r",
            128,
        )
    )

    lora_layers = list(
        getattr(
            cfg.training,
            "nufilt_lora_layers",
            ["attn", "fc1"],
        )
    )

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "[NUFILT] "
        f"null_space={null_space}, "
        f"lr={lr}, "
        f"steps={max_steps}, "
        f"lora_r={lora_r}, "
        f"grad_r={grad_r}, "
        f"null_r={null_r}, "
        f"layers={lora_layers}"
    )

    pretrained_state = {
        k: v.detach().cpu().clone()
        for k, v
        in base_model
        .vision_encoder
        .state_dict()
        .items()
    }
    
    merged_state = {
        k: v.clone()
        for k, v
        in pretrained_state.items()
    }

    selected_layers = []

    for module_name, module in (
        base_model
        .vision_encoder
        .named_modules()
    ):
        if not isinstance(
            module,
            nn.Linear,
        ):
            continue

        if not any(
            key in module_name
            for key in lora_layers
        ):
            continue

        selected_layers.append(
            module_name
        )

    print(
        f"[NUFILT] "
        f"Selected {len(selected_layers)} "
        f"Linear layers"
    )

    if not selected_layers:
        raise RuntimeError(
            "NUFILT did not match any Linear layers. "
            f"Patterns: {lora_layers}"
        )

    for task_id, task_path in enumerate(
        task_paths
    ):

        print(
            f"[NUFILT] Merge task {task_id}"
        )

        task_state = (
            strip_backbone_only(
                torch.load(
                    task_path,
                    map_location="cpu",
                )
            )
        )

        filters = []

        for module_name in selected_layers:

            weight_key = (
                f"{module_name}.weight"
            )

            pretrained_weight = (
                pretrained_state[
                    weight_key
                ]
                .to(device)
                .float()
            )

            merged_weight = (
                merged_state[
                    weight_key
                ]
                .to(device)
                .float()
            )

            current_weight = (
                task_state[
                    weight_key
                ]
                .to(device)
                .float()
            )

            task_tv = (
                current_weight
                - pretrained_weight
            )

            previous_merged_tv = (
                merged_weight
                - pretrained_weight
            )

            filter_module = (
                FilterLoRAWeight(
                    base_weight=merged_weight,
                    task_tv=task_tv,
                    previous_merged_tv=(
                        previous_merged_tv
                    ),
                    null_space=null_space,
                    rank=lora_r,
                    null_rank=null_r,
                    grad_rank=grad_r,
                )
                .to(device)
            )

            filters.append(
                (
                    module_name,
                    weight_key,
                    filter_module,
                )
            )

        if (
            lora_r > 0
            and task_id > 0
        ):

            trainable_params = []

            for _, _, module in filters:
                trainable_params.extend(
                    [
                        module.gate.A,
                        module.gate.B,
                    ]
                )

            optimizer = torch.optim.Adam(
                trainable_params,
                lr=lr,
            )

            pbar = tqdm(
                range(max_steps),
                desc=(
                    f"NUFILT LoRA "
                    f"task {task_id}"
                ),
                leave=False,
            )

            for _ in pbar:

                loss = torch.tensor(
                    0.0,
                    device=device,
                )

                for _, _, module in filters:
                    loss = (
                        loss
                        + module.solve_lora()
                    )

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                pbar.set_postfix(
                    loss=float(
                        loss.detach()
                        .cpu()
                        .item()
                    )
                )

        new_state = {
            k: v.clone()
            for k, v
            in merged_state.items()
        }

        with torch.no_grad():

            for (
                module_name,
                weight_key,
                module,
            ) in filters:

                new_weight = (
                    module
                    .merge_to_base()
                    .detach()
                    .cpu()
                    .to(
                        merged_state[
                            weight_key
                        ].dtype
                    )
                )

                new_state[
                    weight_key
                ] = new_weight

        merged_state = (
            new_state
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

        print(
            f"[NUFILT] "
            f"Saved task {task_id}"
        )

        del filters

        if torch.cuda.is_available():
            torch.cuda.empty_cache()