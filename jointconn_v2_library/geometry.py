from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnergyMap(nn.Module):
    """Build detached token-aligned edge energy from a depth map."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    @torch.no_grad()
    def forward(self, depth: torch.Tensor, token_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            depth: [B, 1 or 3, H, W], expected in [0, 1].
            token_hw: token grid height and width.

        Returns:
            Detached [B, h*w] edge energy in [0, 1].
        """
        if depth.ndim != 4:
            raise ValueError(f"depth must be [B,C,H,W], got {tuple(depth.shape)}")
        if depth.shape[1] != 1:
            depth = depth.mean(dim=1, keepdim=True)

        depth = depth.float().clamp(0.0, 1.0)
        gx = F.conv2d(depth, self.sobel_x.to(depth.device), padding=1)
        gy = F.conv2d(depth, self.sobel_y.to(depth.device), padding=1)
        mag = torch.sqrt(gx.square() + gy.square() + self.eps)

        b = mag.shape[0]
        flat = mag.flatten(1)
        min_v = flat.min(dim=1).values.view(b, 1, 1, 1)
        max_v = flat.max(dim=1).values.view(b, 1, 1, 1)
        mag = (mag - min_v) / (max_v - min_v + self.eps)

        mag = F.interpolate(mag, size=token_hw, mode="area")
        return mag.flatten(1).detach()


def zero_edge(batch_size: int, num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(batch_size, num_tokens, device=device, dtype=dtype)
