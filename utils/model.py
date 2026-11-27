import os
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import torch.nn as nn
from torch import Tensor
import numpy as np
import torch
import copy

StateDictType = Dict[str, Tensor]


def seed_torch(device, seed=0):
    # ------------------------------------------------------------------------------------------
    # References:
    # HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    # ------------------------------------------------------------------------------------------
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def state_dict_sub(
    a: StateDictType, b: StateDictType, strict: bool = True, device=None
):
    if strict:
        assert set(a.keys()) == set(b.keys())

    diff = OrderedDict()
    for k in a:
        if k in b:
            diff[k] = a[k] - b[k]
            if device is not None:
                diff[k] = diff[k].to(device, non_blocking=True)
    return diff

def state_dict_to_vector(
    state_dict: StateDictType,
    remove_keys: Optional[List[str]] = None,
):
    remove_keys = remove_keys if remove_keys is not None else []
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return nn.utils.parameters_to_vector(
        [value.reshape(-1) for key, value in sorted_shared_state_dict.items()]
    )

def _svd(w: Tensor, full_matrices=True) -> Tuple[Tensor, Tensor, Tensor]:
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


def svd(
    w: Tensor, full_matrices=True, accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u.to(original_device), s.to(original_device), v.to(original_device)


def is_leaf_module(module: nn.Module) -> bool:
    return len(list(module.children())) == 0


def get_task_vector_norm(model: nn.Module, pretrained_model: nn.Module) -> Tensor:
    return torch.linalg.norm(
        state_dict_to_vector(
            state_dict_sub(model, pretrained_model)
        )
    )

