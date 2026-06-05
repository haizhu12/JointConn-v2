from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class RelativePositionBias2D(nn.Module):
    """Learned 2D relative position bias for token grids."""

    def __init__(self, num_heads: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.num_heads = num_heads
        self.max_h = max_h
        self.max_w = max_w
        self.table = nn.Parameter(torch.zeros(num_heads, 2 * max_h - 1, 2 * max_w - 1))
        nn.init.trunc_normal_(self.table, std=0.02)
        self._index_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    def _indices(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        key = (h, w, device)
        if key in self._index_cache:
            return self._index_cache[key]
        ys = torch.arange(h, device=device)
        xs = torch.arange(w, device=device)
        coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).view(-1, 2)
        rel = coords[:, None, :] - coords[None, :, :]
        rel_y = rel[..., 0] + self.max_h - 1
        rel_x = rel[..., 1] + self.max_w - 1
        indices = rel_y * (2 * self.max_w - 1) + rel_x
        self._index_cache[key] = indices.long()
        return self._index_cache[key]

    def forward(self, h: int, w: int) -> torch.Tensor:
        if h > self.max_h or w > self.max_w:
            raise ValueError(f"token grid {(h, w)} exceeds max grid {(self.max_h, self.max_w)}")
        indices = self._indices(h, w, self.table.device)
        flat_table = self.table.view(self.num_heads, -1)
        return flat_table[:, indices]


class LocalGaussianKernel2D(nn.Module):
    """Cached spatial compatibility kernel for pairwise edge bias."""

    def __init__(self, sigma: float = 3.0):
        super().__init__()
        self.sigma = sigma
        self._cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    def forward(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (h, w, device)
        if key not in self._cache:
            ys = torch.arange(h, device=device, dtype=torch.float32)
            xs = torch.arange(w, device=device, dtype=torch.float32)
            coords = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1).view(-1, 2)
            dist2 = (coords[:, None, :] - coords[None, :, :]).square().sum(dim=-1)
            kernel = torch.exp(-dist2 / (2.0 * self.sigma * self.sigma))
            self._cache[key] = kernel
        return self._cache[key].to(dtype=dtype)
