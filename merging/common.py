from __future__ import annotations

from typing import Dict
from torch import Tensor


def strip_backbone_only(
    state_dict: Dict[str, Tensor],
    strict: bool = True
) -> Dict[str, Tensor]:
    backbone_state = {
        k.replace("backbone.", "", 1): v.detach().cpu()
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }

    if strict and not backbone_state:
        raise KeyError("Checkpoint has no backbone.* parameters.")

    return backbone_state

