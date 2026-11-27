from __future__ import annotations

import argparse
import json
import time
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from utils.config import add_config_arg, ensure_dir, get_task_order, load_config, seed_everything
from datasets.survival import Sequential_Generic_MIL_Survival_Dataset
from utils.survival import (
    cox_ph_loss,
    concordance_index,
    concordance_index_ipcw,
)
from utils.model import seed_torch
from models.survival import CustomSequential, TaskSpecificCoxHead
from utils.sampling import stable_int_hash as _stable_int_hash, sample_patches_train as sample_patches


EMBED_DIM = 768
TITAN_PS_ARG = 1024


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    warmup_length = max(1, int(warmup_length))
    steps = max(warmup_length + 1, int(steps))

    def _assign_learning_rate(new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr * param_group.get("lr_scale", 1.0)

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        _assign_learning_rate(lr)
        return lr

    return _lr_adjuster

def make_grad_scaler():
    if torch.cuda.is_available():
        return torch.amp.GradScaler("cuda", enabled=True)
    return torch.amp.GradScaler("cpu", enabled=False)
    

def sanitize_features(features: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(features.float(), nan=0.0, posinf=0.0, neginf=0.0)


def _to_numpy_list(tensor_or_array):
    if torch.is_tensor(tensor_or_array):
        return tensor_or_array.detach().cpu().numpy().tolist()
    return np.asarray(tensor_or_array).tolist()


def _events_from_censorship(censorship: torch.Tensor | np.ndarray | list) -> np.ndarray:
    c = np.asarray(_to_numpy_list(censorship), dtype=int)
    return (1 - c).astype(int)


def forward_bag_batch(
    model,
    feature_list,
    coord_list,
    device,
    num_patches,
    ps,
    slide_ids=None,
    sampling_mode: str = "random",
    sampling_seed: int = 0,
):
    risks: List[torch.Tensor] = []

    if slide_ids is None:
        slide_ids = [None] * len(feature_list)

    for bag_idx, (features, coords, slide_id) in enumerate(zip(feature_list, coord_list, slide_ids)):
        features = sanitize_features(features.to(device, non_blocking=True))
        coords = coords.long().to(device, non_blocking=True)

        seed = None
        if sampling_mode == "fixed_random":
            seed = _stable_int_hash(slide_id if slide_id is not None else bag_idx, base_seed=sampling_seed)

        features, coords = sample_patches(
            features,
            coords,
            num_patches,
            mode=sampling_mode,
            seed=seed,
        )
        risk = model(features, coords, ps)
        risks.append(risk.view(-1))
    return torch.cat(risks, dim=0)


def _dataset_times_events(loader):
    """Return stable train split times/events for IPCW censoring estimation."""
    dataset = loader.dataset
    if not hasattr(dataset, "slide_data") or not hasattr(dataset, "time_col"):
        return None, None
    df = dataset.slide_data
    times = df[dataset.time_col].values.astype(float)
    events = (1 - df["censorship"].values.astype(int)).astype(int)
    return times.tolist(), events.tolist()


def evaluate_full_split(
    loader,
    model,
    device,
    num_patches,
    ps,
    cfg,
    sampling_mode: str = "fixed_random",
    sampling_seed: int = 0,
):
    model.eval()

    all_risks: List[torch.Tensor] = []
    all_times: List[torch.Tensor] = []
    all_censorships: List[torch.Tensor] = []

    skipped_batches = 0
    total_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for (features, coords, _time_bin, censorship, survival_time, slide_id,) in loader:

            censorship = censorship.float().to(device,non_blocking=True,)
            survival_time = survival_time.float().to(device,non_blocking=True,)

            try:
                with torch.amp.autocast("cuda",dtype=torch.bfloat16,enabled=torch.cuda.is_available(),):
                    risks = forward_bag_batch(
                        model=model,
                        feature_list=features,
                        coord_list=coords,
                        device=device,
                        num_patches=num_patches,
                        ps=ps,
                        slide_ids=slide_id,
                        sampling_mode=sampling_mode,
                        sampling_seed=sampling_seed,
                    )

                    events = 1.0 - censorship

                    loss = cox_ph_loss(
                        risks,
                        survival_time,
                        events,
                        eps=float(cfg.survival.get("cox_eps", 1e-7,)),)

            except (FloatingPointError, RuntimeError) as exc:
                skipped_batches += 1
                print(f"[Warning] Skip eval batch: {exc}")
                continue

            if not torch.isfinite(loss):
                skipped_batches += 1
                print(
                    "[Warning] Skip eval batch because "
                    "loss is non-finite."
                )
                continue

            all_risks.append(risks.detach().float().cpu())
            all_times.append(survival_time.detach().float().cpu())
            all_censorships.append(censorship.detach().float().cpu())
            total_loss += float(loss.detach().cpu().item())
            valid_batches += 1

    if not all_risks:
        return {
            "loss": float("nan"),
            "times": [],
            "censorships": [],
            "events": [],
            "risks": [],
            "cindex": 0.0,
            "skipped_batches": skipped_batches,
        }

    risks_all = torch.cat(all_risks, dim=0)
    times_all = torch.cat(all_times, dim=0)
    censorships_all = torch.cat(all_censorships, dim=0,)

    times_np = (times_all.numpy().astype(float).tolist())

    censorships_np = (censorships_all.numpy().astype(int).tolist())
    events_np = (1 - np.asarray(censorships_np, dtype=int,)).astype(int).tolist()

    risks_np = (risks_all.numpy().astype(float).tolist())
    cindex = concordance_index(times_np, risks_np, events_np,)

    avg_loss = total_loss / max(1,valid_batches,)

    return {
        "loss": avg_loss,
        "cindex": cindex,
        "times": times_np,
        "censorships": censorships_np,
        "events": events_np,
        "risks": risks_np,
        "skipped_batches": skipped_batches,
    }

def train(train_loader, val_loader, model, num_epochs, lr, head_lr, weight_decay, device, train_num_patches, cfg):
    named_parameters = list(model.named_parameters())

    # Param groups: train TITAN aggregator conservatively, train survival/prompt heads faster.
    backbone_params, head_params, no_decay_params = [], [], []
    for name, p in named_parameters:
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "bias" in name or "ln" in name.lower() or "norm" in name.lower():
            no_decay_params.append(p)
        elif name.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": no_decay_params, "weight_decay": 0.0, "lr_scale": 1.0},
            {"params": backbone_params, "weight_decay": weight_decay, "lr_scale": 1.0},
            {"params": head_params, "weight_decay": weight_decay, "lr_scale": float(head_lr) / max(float(lr), 1e-12)},
        ],
        lr=lr,
    )
    lr_scheduler = cosine_lr(optimizer, lr, int(len(train_loader) * num_epochs * 0.1), len(train_loader) * num_epochs)

    scaler = make_grad_scaler()
    step = 0
    ps = torch.tensor(TITAN_PS_ARG).int().to(device)

    for epoch in tqdm(range(num_epochs), desc="train epochs"):
        model.train()
        total_train_loss = 0.0
        valid_train_batches = 0
        skipped_train_batches = 0
        train_times, train_censorships, train_risks = [], [], []

        for features, coords, _time_bin, censorship, survival_time, _slide_id in tqdm(train_loader, leave=False):
            lr_scheduler(step)
            censorship = censorship.float().to(device, non_blocking=True)
            survival_time = survival_time.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    risks = forward_bag_batch(
                        model,
                        features,
                        coords,
                        device,
                        train_num_patches,
                        ps,
                        slide_ids=_slide_id,
                        sampling_mode="random",
                    )
                    events = 1.0 - censorship.float()
                    loss = cox_ph_loss(risks, survival_time, events, eps=float(cfg.survival.get("cox_eps", 1e-7)),)
                    
            except (FloatingPointError, RuntimeError) as exc:
                skipped_train_batches += 1
                print(f"[Warning] Skip train batch: {exc}")
                step += 1
                continue

            if not torch.isfinite(loss):
                skipped_train_batches += 1
                print("[Warning] Skip train batch because loss is non-finite.")
                step += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in list(model.parameters()) if p.requires_grad], max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_risks.extend(risks.detach().float().cpu().numpy().tolist())
            train_times.extend(_to_numpy_list(survival_time))
            train_censorships.extend(censorship.detach().cpu().numpy().astype(int).tolist())
            total_train_loss += float(loss.detach().cpu().item())
            valid_train_batches += 1
            step += 1

        avg_train_loss = total_train_loss / max(1, valid_train_batches)
        train_events = (1 - np.asarray(train_censorships, dtype=int)).tolist()
        train_cindex = concordance_index(train_times, train_risks, train_events)
        train_cindex_ipcw = concordance_index_ipcw(train_times, train_events, train_times, train_risks, train_events)

        denom_tr = max(1, valid_train_batches)
        tqdm.write(
            f"epoch {epoch}/{num_epochs}, "
            f"loss={avg_train_loss:.4f} "
            f"cindex={train_cindex:.4f}, "
            f"cindex_ipcw={train_cindex_ipcw:.4f}, "
            f"skipped_train={skipped_train_batches}"
        )

    eval_num_patches = int(cfg.training.get("eval_num_patches", train_num_patches))
    val_sampling_mode = str(cfg.training.get("val_sampling_mode", "fixed_random"))
    val_eval = evaluate_full_split(
        loader=val_loader,
        model=model,
        device=device,
        num_patches=eval_num_patches,
        ps=ps,
        cfg=cfg,
        sampling_mode=val_sampling_mode,
        sampling_seed=int(cfg.training.seed),
    )
    
    avg_val_loss = val_eval["loss"]
    val_cindex = val_eval["cindex"]
    
    ipcw_train_times, ipcw_train_events = (_dataset_times_events(train_loader))
    
    if ipcw_train_times is None:
        ipcw_train_times = train_times
        ipcw_train_events = train_events
    
    val_cindex_ipcw = concordance_index_ipcw(
        ipcw_train_times,
        ipcw_train_events,
        val_eval["times"],
        val_eval["risks"],
        val_eval["events"],
    )
    
    print(
        f"Final held-out evaluation after "
        f"{num_epochs} epochs: "
        f"loss={avg_val_loss:.4f}, "
        f"cindex={val_cindex:.4f}, "
        f"ipcw={val_cindex_ipcw:.4f}"
    )
    
    training_info = {
        "num_epochs_configured": int(num_epochs),
        "num_epochs_completed": int(num_epochs),
        "final_epoch": int(num_epochs - 1),
    
        "checkpoint_policy": "final_epoch",
    
        "heldout_policy":
            "evaluate_once_after_final_epoch",
    
        "heldout_loss": float(avg_val_loss),
        "heldout_cindex": float(val_cindex),
        "heldout_cindex_ipcw": float(
            val_cindex_ipcw
        ),
    }

    model.eval()
    
    return model, training_info


def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser(description="Stage-1 offline task-specific TITAN survival fine-tuning")
    add_config_arg(parser)
    parser.add_argument("--save_dir", type=str, default=None, help="Override paths.finetuned_checkpoints")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(int(cfg.training.seed), device)

    try:
        seed_torch(device, int(cfg.training.seed))
    except Exception:
        pass

    task_order = get_task_order(cfg)[: int(cfg.training.num_tasks)]
    num_tasks = len(task_order)
    save_root = ensure_dir(args.save_dir or cfg.paths.finetuned_checkpoints)

    seq_dataset = Sequential_Generic_MIL_Survival_Dataset(cfg=cfg)

    for fold_id in range(int(cfg.training.num_folds)):
        fold_dir = ensure_dir(save_root / f"fold_{fold_id}")

        for task_id, task in enumerate(task_order):
            train_loader, val_loader, _ = seq_dataset.get_data_loaders(fold_id, task_id)
            titan = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)

            survival_head = TaskSpecificCoxHead(
                in_dim=EMBED_DIM,
                hidden_dim=int(cfg.training.get("survival_hidden_dim", 512)),
                dropout=float(cfg.training.get("survival_dropout", 0.25)),
            ).to(device)
            wrapped = CustomSequential(titan, survival_head).to(device)

            start = time.time()
            wrapped, training_info = train(
                train_loader=train_loader,
                val_loader=val_loader,
                model=wrapped,
                num_epochs=int(cfg.training.num_epochs),
                lr=float(cfg.training.lr),
                head_lr=float(cfg.training.get("head_lr", cfg.training.lr)),
                weight_decay=float(cfg.training.weight_decay),
                device=device,
                train_num_patches=int(cfg.training.train_num_patches),
                cfg=cfg,
            )
            elapsed = time.time() - start

            out_path = fold_dir / f"finetune_task_{task_id}.pt"
            torch.save(wrapped.state_dict(), out_path)

            metadata = {
                "stage": "stage1_offline_task_specific_survival_finetuning",
                "task_id": task_id,
                "task_name": task,
                "checkpoint": str(out_path),
                "survival_head": "TaskSpecificCoxHead",
                "main_loss": "Cox partial likelihood",
                "training": training_info,
            }
            meta_path = fold_dir / f"finetune_task_{task_id}_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            print(f"Saved {out_path}, {meta_path} in {elapsed:.2f}s")

            del titan, wrapped, survival_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
