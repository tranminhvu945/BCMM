from __future__ import annotations

import torch
import torch.nn as nn

EMBED_DIM = 768


class CustomSequential(nn.Module):
    def __init__(self, model, survival_head):
        super().__init__()
        self.backbone = model.vision_encoder
        # Keep name `mlp` for compatibility with merging scripts that strip or
        # optionally process non-backbone heads.
        self.mlp = survival_head

    def forward(self, features, coords, ps):
        z = self.backbone(features, coords, ps)
        return self.mlp(z)



class TaskSpecificCoxHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM, hidden_dim: int = 512, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.risk = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor):
        z = torch.nan_to_num(z.float(), nan=0.0, posinf=0.0, neginf=0.0)
        hidden = self.net(z)
        risk = self.risk(hidden).squeeze(-1).clamp(min=-20.0, max=20.0)
        return risk

