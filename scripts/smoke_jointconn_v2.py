from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from jointconn_v2_library.gcm_wfm import GCMWFMLoss, TaskBatch
from jointconn_v2_library.geometry import EdgeEnergyMap
from jointconn_v2_library.jointconn_v2 import JointConnV2Block


def smoke_edge_energy() -> None:
    depth = torch.rand(2, 1, 16, 16, requires_grad=True)
    edge = EdgeEnergyMap()(depth, (4, 4))
    assert edge.shape == (2, 16)
    assert not edge.requires_grad
    assert torch.isfinite(edge).all()
    assert edge.min() >= 0.0
    assert edge.max() <= 1.0


def smoke_jointconn_block() -> None:
    pair_bsz = 2
    heads = 2
    tokens = 16
    head_dim = 4
    hidden = heads * head_dim
    total_bsz = pair_bsz * 2

    block = JointConnV2Block(
        hidden_size=hidden,
        num_heads=heads,
        max_token_hw=4,
        max_bias_tokens=64,
        zero_init_output_proj=False,
    )
    h = torch.randn(total_bsz, tokens, hidden)
    q = torch.randn(total_bsz, heads, tokens, head_dim)
    k = torch.randn(total_bsz, heads, tokens, head_dim)
    v = torch.randn(total_bsz, heads, tokens, head_dim)
    timesteps = torch.tensor([0.2, 0.7, 0.0, 0.0])
    e_att = torch.rand(pair_bsz, tokens)
    masks = {
        "m_x": torch.ones(pair_bsz),
        "m_y": torch.zeros(pair_bsz),
        "lambda_x": torch.ones(pair_bsz),
        "lambda_y": torch.zeros(pair_bsz),
    }

    out = block.forward_from_qkv(
        h=h,
        q=q,
        k=k,
        v=v,
        e_att=e_att,
        token_hw=(4, 4),
        timesteps=timesteps,
        task_masks=masks,
        layer_index=3,
    )
    assert out.shape == h.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out[pair_bsz:], torch.zeros_like(out[pair_bsz:]))
    assert block.last_stats is not None
    assert block.last_stats.gate.shape == (pair_bsz, 1, 1)


def smoke_gcm_wfm_loss() -> None:
    pair_bsz = 2
    tokens = 16
    channels = 8
    v_x = torch.randn(pair_bsz, tokens, channels, requires_grad=True)
    v_y = torch.randn(pair_bsz, tokens, channels, requires_grad=True)
    tau_x = torch.randn_like(v_x)
    tau_y = torch.randn_like(v_y)
    e_loss = torch.rand(pair_bsz, tokens, requires_grad=True)
    task = TaskBatch(
        t_x=torch.tensor([0.2, 0.7]),
        t_y=torch.tensor([0.0, 0.5]),
        m_x=torch.ones(pair_bsz),
        m_y=torch.tensor([0.0, 1.0]),
        lambda_x=torch.ones(pair_bsz),
        lambda_y=torch.tensor([0.0, 1.0]),
    )

    loss_fn = GCMWFMLoss(beta_loss=2.0, gamma_t=1.0, alpha_min=0.25, alpha_max=4.0)
    out = loss_fn(v_x, v_y, tau_x, tau_y, e_loss, task)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)
    assert out.alpha_x.min() >= 0.25
    assert out.alpha_x.max() <= 4.0
    assert not out.alpha_x.requires_grad
    out.loss.backward()
    assert v_x.grad is not None
    assert v_y.grad is not None
    assert e_loss.grad is None


def main() -> None:
    torch.manual_seed(0)
    smoke_edge_energy()
    smoke_jointconn_block()
    smoke_gcm_wfm_loss()
    print("JointConn-v2 smoke test passed.")


if __name__ == "__main__":
    main()
