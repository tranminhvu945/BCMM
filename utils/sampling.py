from __future__ import annotations

import hashlib
import torch


def stable_int_hash(value, base_seed: int = 0) -> int:
    """Stable deterministic patch-selection seed (original md5 algorithm)."""
    if isinstance(value, (list, tuple)):
        value = "_".join(str(v) for v in value)
    key = f"{base_seed}_{str(value)}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def _sample_indices(n: int, k: int, device, mode: str, seed, *, evaluation: bool):
    if mode == "random" and not evaluation:
        return torch.randperm(n, device=device)[:k]
    if mode == "fixed_random":
        if evaluation:
            # Original evaluation: int(seed or 0)
            effective_seed = int(seed or 0)
        else:
            # Original training: seed=None becomes 0, otherwise int(seed)
            effective_seed = int(0 if seed is None else seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(effective_seed % (2**31 - 1))
        return torch.randperm(n, generator=generator, device="cpu")[:k].to(device)
    if mode == "first":
        return torch.arange(k, device=device)
    if mode == "all" and evaluation:
        return torch.arange(n, device=device)
    if evaluation:
        raise ValueError(mode)
    raise ValueError(f"Unknown patch sampling mode: {mode!r}")


def sample_patches_train(features, coords, k, mode: str = "random", seed: int | None = None):
    """Preserve training modes, default and 2-D indexing."""
    k = min(int(k), features.shape[0])
    n = int(features.shape[0])
    indices = _sample_indices(n, k, features.device, mode, seed, evaluation=False)
    return features[indices, :], coords[indices, :]


def sample_patches_eval(features, coords, k, mode="fixed_random", seed=None):
    """Preserve evaluation modes, default and arbitrary-dim indexing."""
    k = min(int(k), int(features.shape[0]))
    n = int(features.shape[0])
    idx = _sample_indices(n, k, features.device, mode, seed, evaluation=True)
    return features[idx], coords[idx]
