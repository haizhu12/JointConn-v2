from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from jointconn_v2_library.relative_position import LocalGaussianKernel2D, RelativePositionBias2D


@dataclass
class JointConnStats:
    gate: torch.Tensor
    r_x: torch.Tensor
    r_y: torch.Tensor
    r_0: torch.Tensor


def _mask_value(task_masks: Optional[Dict[str, torch.Tensor]], name: str, like: torch.Tensor, default: float = 1.0) -> torch.Tensor:
    batch = like.shape[0] // 2
    if task_masks is None or name not in task_masks or task_masks[name] is None:
        return torch.full((batch,), default, device=like.device, dtype=like.dtype)
    value = task_masks[name].to(device=like.device, dtype=like.dtype)
    if value.ndim > 1:
        value = value.view(value.shape[0], -1)[:, 0]
    if value.shape[0] == like.shape[0]:
        value = value[:batch]
    return value


class ContentGate(nn.Module):
    def __init__(self, hidden_size: int, hidden_dim: int = 128):
        super().__init__()
        hidden_dim = min(hidden_dim, hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_x: torch.Tensor,
        h_y: torch.Tensor,
        t_x: torch.Tensor,
        t_y: torch.Tensor,
        m_x: torch.Tensor,
        m_y: torch.Tensor,
        lambda_x: torch.Tensor,
        lambda_y: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        gap_x = h_x.detach().mean(dim=1)
        gap_y = h_y.detach().mean(dim=1)
        layer = torch.full_like(t_x, float(layer_index) / 100.0)
        scalar = torch.stack([t_x, t_y, (t_x - t_y).abs(), m_x, m_y, lambda_x, lambda_y, layer], dim=-1)
        return torch.sigmoid(self.net(torch.cat([gap_x, gap_y, scalar], dim=-1))).view(h_x.shape[0], 1, 1)


class RegionalRouting(nn.Module):
    def __init__(self, hidden_size: int, hidden_dim: int = 64, routing_type: str = "three_way"):
        super().__init__()
        self.routing_type = routing_type
        out_dim = 3 if routing_type == "three_way" else 2
        hidden_dim = min(hidden_dim, hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h_x: torch.Tensor, h_y: torch.Tensor, e_att: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor):
        b, n, _ = h_x.shape
        t_feat = torch.stack([t_x, t_y], dim=-1)[:, None, :].expand(b, n, 2)
        logits = self.net(torch.cat([h_x, h_y, e_att[:, :, None], t_feat], dim=-1))
        if self.routing_type == "three_way":
            routing = torch.softmax(logits, dim=-1)
            return routing[..., 0:1], routing[..., 1:2], routing[..., 2:3]
        r = torch.sigmoid(logits)
        r_x, r_y = r[..., 0:1], r[..., 1:2]
        r_0 = (1.0 - torch.maximum(r_x, r_y)).clamp(min=0.0)
        return r_x, r_y, r_0


class LayerwiseCouplingSchedule(nn.Module):
    def __init__(self, residual_w_max: float = 1.0):
        super().__init__()
        self.residual_w_max = residual_w_max
        self.net = nn.Sequential(
            nn.Linear(8, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        t_x: torch.Tensor,
        t_y: torch.Tensor,
        m_x: torch.Tensor,
        m_y: torch.Tensor,
        lambda_x: torch.Tensor,
        lambda_y: torch.Tensor,
        layer_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        layer = torch.full_like(t_x, float(layer_index) / 100.0)
        feat = torch.stack([t_x, t_y, (t_x - t_y).abs(), m_x, m_y, lambda_x, lambda_y, layer], dim=-1)
        weights = self.residual_w_max * torch.sigmoid(self.net(feat))
        return weights[:, 0:1, None], weights[:, 1:2, None]


class JointConnV2Block(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        beta_att: float = 1.0,
        local_kernel_sigma: float = 3.0,
        routing_type: str = "three_way",
        gate_hidden_dim: int = 128,
        routing_hidden_dim: int = 64,
        output_bottleneck_dim: int = 128,
        residual_w_max: float = 1.0,
        max_token_hw: int = 64,
        max_bias_tokens: int = 2048,
        zero_init_output_proj: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.beta_att = beta_att
        self.max_bias_tokens = max_bias_tokens
        self.rel_pos = RelativePositionBias2D(num_heads, max_h=max_token_hw, max_w=max_token_hw)
        self.local_kernel = LocalGaussianKernel2D(local_kernel_sigma)
        self.content_gate = ContentGate(hidden_size, hidden_dim=gate_hidden_dim)
        self.router = RegionalRouting(hidden_size, hidden_dim=routing_hidden_dim, routing_type=routing_type)
        self.schedule = LayerwiseCouplingSchedule(residual_w_max=residual_w_max)
        output_bottleneck_dim = min(output_bottleneck_dim, hidden_size * 2)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, output_bottleneck_dim),
            nn.SiLU(),
            nn.Linear(output_bottleneck_dim, hidden_size * 2),
        )
        if zero_init_output_proj:
            nn.init.zeros_(self.out_proj[-1].weight)
            nn.init.zeros_(self.out_proj[-1].bias)
        self.last_stats: Optional[JointConnStats] = None

    def _bias(self, e_att: torch.Tensor, token_hw: Tuple[int, int], dtype: torch.dtype) -> Optional[torch.Tensor]:
        b, n = e_att.shape
        h, w = token_hw
        if n != h * w or n > self.max_bias_tokens:
            return None
        rel = self.rel_pos(h, w).to(device=e_att.device, dtype=dtype)
        kernel = self.local_kernel(h, w, e_att.device, dtype=dtype)
        edge_pair = e_att.to(dtype=dtype)[:, :, None] * e_att.to(dtype=dtype)[:, None, :]
        geom = self.beta_att * edge_pair[:, None, :, :] * kernel[None, None, :, :]
        return rel[None, :, :, :] + geom

    def forward_from_qkv(
        self,
        h: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        e_att: Optional[torch.Tensor],
        token_hw: Tuple[int, int],
        timesteps: torch.Tensor,
        task_masks: Optional[Dict[str, torch.Tensor]],
        layer_index: int,
    ) -> torch.Tensor:
        batch_total, num_tokens, _ = h.shape
        if batch_total % 2 != 0:
            raise ValueError("JointConn-v2 expects RGB/depth pairs packed as an even batch.")
        pair_batch = batch_total // 2

        h_x, h_y = h[:pair_batch], h[pair_batch:]
        q_x, q_y = q[:pair_batch], q[pair_batch:]
        k_x, k_y = k[:pair_batch], k[pair_batch:]
        v_x, v_y = v[:pair_batch], v[pair_batch:]

        t_x = timesteps[:pair_batch].to(dtype=h.dtype, device=h.device)
        t_y = timesteps[pair_batch:].to(dtype=h.dtype, device=h.device)
        m_x = _mask_value(task_masks, "m_x", h)
        m_y = _mask_value(task_masks, "m_y", h)
        lambda_x = _mask_value(task_masks, "lambda_x", h)
        lambda_y = _mask_value(task_masks, "lambda_y", h)

        if e_att is None:
            e_att = torch.zeros(pair_batch, num_tokens, device=h.device, dtype=h.dtype)
        else:
            e_att = e_att.to(device=h.device, dtype=h.dtype)
            if e_att.shape[0] == batch_total:
                e_att = e_att[pair_batch:]
            if e_att.shape[0] != pair_batch:
                raise ValueError(f"e_att batch {e_att.shape[0]} does not match pair batch {pair_batch}")

        bias = self._bias(e_att.detach(), token_hw, q.dtype)
        o_x = F.scaled_dot_product_attention(q_x, k_y, v_y, attn_mask=bias)
        o_y = F.scaled_dot_product_attention(q_y, k_x, v_x, attn_mask=bias)
        o_x = rearrange(o_x, "B H L D -> B L (H D)")
        o_y = rearrange(o_y, "B H L D -> B L (H D)")

        o_pair = self.out_proj(torch.cat([o_x, o_y], dim=-1))
        o_x, o_y = torch.chunk(o_pair, 2, dim=-1)

        gate = self.content_gate(h_x, h_y, t_x, t_y, m_x, m_y, lambda_x, lambda_y, layer_index)
        r_x, r_y, r_0 = self.router(h_x, h_y, e_att.detach(), t_x, t_y)
        w_x, w_y = self.schedule(t_x, t_y, m_x, m_y, lambda_x, lambda_y, layer_index)

        res_x = lambda_x[:, None, None] * w_x * gate * r_x * o_x
        res_y = lambda_y[:, None, None] * w_y * gate * r_y * o_y
        self.last_stats = JointConnStats(
            gate=gate.detach(),
            r_x=r_x.detach(),
            r_y=r_y.detach(),
            r_0=r_0.detach(),
        )
        return torch.cat([res_x, res_y], dim=0)
