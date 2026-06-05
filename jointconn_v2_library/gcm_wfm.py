from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class TaskBatch:
    t_x: torch.Tensor
    t_y: torch.Tensor
    m_x: torch.Tensor
    m_y: torch.Tensor
    lambda_x: torch.Tensor
    lambda_y: torch.Tensor
    task_id: Optional[torch.Tensor] = None

    def as_masks(self) -> Dict[str, torch.Tensor]:
        return {
            "m_x": self.m_x,
            "m_y": self.m_y,
            "lambda_x": self.lambda_x,
            "lambda_y": self.lambda_y,
        }


@dataclass
class GCMWFMLossOutput:
    loss: torch.Tensor
    loss_x: torch.Tensor
    loss_y: torch.Tensor
    alpha_x: torch.Tensor
    alpha_y: torch.Tensor
    diagnostics: Dict[str, float]


class GCMWFMLoss(nn.Module):
    def __init__(
        self,
        beta_loss: float = 2.0,
        gamma_t: float = 1.0,
        alpha_min: float = 0.25,
        alpha_max: float = 4.0,
        alpha_eps: float = 1e-6,
    ):
        super().__init__()
        self.beta_loss = beta_loss
        self.gamma_t = gamma_t
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_eps = alpha_eps

    @classmethod
    def from_args(cls, args) -> "GCMWFMLoss":
        return cls(
            beta_loss=getattr(args, "jointconn_beta_loss", 2.0),
            gamma_t=getattr(args, "jointconn_gamma_t", 1.0),
            alpha_min=getattr(args, "jointconn_alpha_min", 0.25),
            alpha_max=getattr(args, "jointconn_alpha_max", 4.0),
            alpha_eps=getattr(args, "jointconn_alpha_eps", 1e-6),
        )

    def _time_weight(self, t: torch.Tensor) -> torch.Tensor:
        return 1.0 + self.gamma_t * t * (1.0 - t)

    def _build_alpha(self, t: torch.Tensor, e_loss: torch.Tensor) -> torch.Tensor:
        alpha = self._time_weight(t).to(e_loss.dtype)[:, None] * (1.0 + self.beta_loss * e_loss)
        denom = alpha.mean(dim=1, keepdim=True) + self.alpha_eps
        alpha = (alpha / denom).clamp(self.alpha_min, self.alpha_max)
        return alpha.detach()

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mask = mask.to(values.dtype)
        return (values * mask).sum() / (mask.sum() + eps)

    @staticmethod
    def _ess(alpha: torch.Tensor) -> torch.Tensor:
        n = alpha.shape[1]
        return alpha.sum(dim=1).square() / (n * alpha.square().sum(dim=1).clamp_min(1e-6))

    def forward(
        self,
        v_x: torch.Tensor,
        v_y: torch.Tensor,
        tau_x: torch.Tensor,
        tau_y: torch.Tensor,
        e_loss: torch.Tensor,
        task: TaskBatch,
    ) -> GCMWFMLossOutput:
        e_loss = e_loss.to(device=v_x.device, dtype=v_x.dtype).detach()
        alpha_x = self._build_alpha(task.t_x.to(device=v_x.device, dtype=v_x.dtype), e_loss)
        alpha_y = self._build_alpha(task.t_y.to(device=v_y.device, dtype=v_y.dtype), e_loss)

        err_x = (v_x.float() - tau_x.float()).square().mean(dim=-1)
        err_y = (v_y.float() - tau_y.float()).square().mean(dim=-1)
        loss_x_sample = (alpha_x.float() * err_x).mean(dim=1)
        loss_y_sample = (alpha_y.float() * err_y).mean(dim=1)

        m_x = task.m_x.to(device=v_x.device, dtype=loss_x_sample.dtype)
        m_y = task.m_y.to(device=v_y.device, dtype=loss_y_sample.dtype)
        loss_x = self._masked_mean(loss_x_sample, m_x)
        loss_y = self._masked_mean(loss_y_sample, m_y)
        loss = loss_x + loss_y

        with torch.no_grad():
            diagnostics = {
                "loss_x": float(loss_x.detach().cpu()),
                "loss_y": float(loss_y.detach().cpu()),
                "alpha_x_mean": float(alpha_x.mean().detach().cpu()),
                "alpha_y_mean": float(alpha_y.mean().detach().cpu()),
                "alpha_x_min": float(alpha_x.min().detach().cpu()),
                "alpha_y_min": float(alpha_y.min().detach().cpu()),
                "alpha_x_max": float(alpha_x.max().detach().cpu()),
                "alpha_y_max": float(alpha_y.max().detach().cpu()),
                "alpha_ess_x": float(self._ess(alpha_x.float()).mean().detach().cpu()),
                "alpha_ess_y": float(self._ess(alpha_y.float()).mean().detach().cpu()),
            }

        return GCMWFMLossOutput(loss, loss_x, loss_y, alpha_x, alpha_y, diagnostics)
