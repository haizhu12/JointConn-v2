import json
import os
from dataclasses import replace
from typing import List, Optional, Tuple, Union

import einops
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import CLIPConfig, CLIPTextModel, T5Config, T5EncoderModel

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from jointconn_v2_library import jointconn_v2_model
from jointconn_v2_library.lora_adapter import (
    add_linear_to_double_blocks,
    add_linear_to_single_blocks,
    add_lora_to_attention,
)
from jointconn_v2_library.jointconn_v2 import JointConnV2Block

from library.utils import load_safetensors
from library.flux_utils import analyze_checkpoint_state

MODEL_VERSION_FLUX_V1 = "flux1"
MODEL_NAME_DEV = "dev"
MODEL_NAME_SCHNELL = "schnell"

def load_empty_flux_model(
    ckpt_path: str, dtype: Optional[torch.dtype], device: Union[str, torch.device], disable_mmap: bool = False, noload = False
) -> Tuple[bool, jointconn_v2_model.JointConnV2Model]:
    is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths = analyze_checkpoint_state(ckpt_path)
    name = MODEL_NAME_DEV if not is_schnell else MODEL_NAME_SCHNELL

    # build model
    logger.info(f"Building Flux model {name} from {'Diffusers' if is_diffusers else 'BFL'} checkpoint")
    with torch.device("meta"):
        params = jointconn_v2_model.configs[name].params

        # set the number of blocks
        if params.depth != num_double_blocks:
            logger.info(f"Setting the number of double blocks from {params.depth} to {num_double_blocks}")
            params = replace(params, depth=num_double_blocks)
        if params.depth_single_blocks != num_single_blocks:
            logger.info(f"Setting the number of single blocks from {params.depth_single_blocks} to {num_single_blocks}")
            params = replace(params, depth_single_blocks=num_single_blocks)

        model = jointconn_v2_model.JointConnV2Model(params)
        if dtype is not None:
            model = model.to(dtype)

    return is_schnell, model

def setup_jointconn_v2_model(
    model: nn.Module,
    lora_rank: int = 64,
    input_phase_configs: Optional[List[Tuple[str, int, float]]] = None,
    enable_jointconn_v2: bool = False,
    jointconn_config: Optional[dict] = None,
    lora_branch_mode: str = "depth_only",
) -> nn.Module:
    """
    Extend a Flux model with JointConn-v2 adapters:
    - Insert Linear adapters (joint1/joint2) into each block.
    - Attach LoRA modules to input-phase layers and attention layers.
    """
    if not enable_jointconn_v2:
        add_linear_to_double_blocks(model)
        add_linear_to_single_blocks(model)

    if input_phase_configs is None:
        input_phase_configs = [
            ("vector_in.in_layer", 512, 512/2),
            ("vector_in.out_layer", 1024, 1024/2),
            ("txt_in", 1024, 1024/2),
        ]
    for name, r, alpha in input_phase_configs:
        add_lora_to_attention(model, name, r=r, alpha=alpha)

    lora_alpha = lora_rank / 2
    for name in [
        "img_mod.lin", "img_attn.qkv", "txt_mod.lin",
        "txt_attn.qkv", "img_attn.proj", "txt_attn.proj"
    ]:
        add_lora_to_attention(model, name, r=lora_rank, alpha=lora_alpha)

    for name in ["linear1", "modulation.lin"]:
        add_lora_to_attention(model, name, r=lora_rank, alpha=lora_alpha)

    if lora_branch_mode != "depth_only":
        logger.warning(
            "Only lora_branch_mode='depth_only' is implemented in the current adapter. "
            "Falling back to the legacy paired LoRA behavior."
        )

    if enable_jointconn_v2:
        jointconn_config = jointconn_config or {}
        _attach_jointconn_v2(model, jointconn_config)

    return model


def _attach_jointconn_v2(model: nn.Module, config: dict):
    def make_block(block):
        module = JointConnV2Block(
            hidden_size=block.hidden_size,
            num_heads=block.num_heads,
            beta_att=config.get("beta_att", 1.0),
            local_kernel_sigma=config.get("local_kernel_sigma", 3.0),
            routing_type=config.get("routing_type", "three_way"),
            gate_hidden_dim=config.get("gate_hidden_dim", 128),
            routing_hidden_dim=config.get("routing_hidden_dim", 64),
            output_bottleneck_dim=config.get("output_bottleneck_dim", 128),
            residual_w_max=config.get("residual_w_max", 1.0),
            max_token_hw=config.get("max_token_hw", 64),
            max_bias_tokens=config.get("max_bias_tokens", 2048),
            zero_init_output_proj=config.get("zero_init_output_proj", True),
        )
        try:
            module.to(device=model.device, dtype=model.dtype)
        except Exception:
            pass
        return module

    for block in model.double_blocks:
        block.add_module("jointconn_v2", make_block(block))
    for block in model.single_blocks:
        block.add_module("jointconn_v2", make_block(block))

    logger.info("JointConn-v2 modules attached to all double and single blocks.")

def save_added_params(model: torch.nn.Module,
                      path: str,
                      dtype: torch.dtype = torch.bfloat16):
    """
    추가된 파라미터만 뽑아서 지정한 dtype으로 변환한 뒤 저장
    """
    added = {}
    for name, tensor in model.state_dict().items():
        if any(k in name for k in ("joint1", "joint2", "jointconn_v2", "lora_A", "lora_B")):
            # 원하는 dtype으로 캐스팅
            added[name] = tensor.to(dtype).cpu()
    save_file(added, path)
    print(f"Saved {len(added)} tensors as {dtype} ▶ {path}")

def get_small_timesteps(
    args, noise_scheduler, latents, noise, device, dtype, timesteps=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, _, h, w = latents.shape
    sigmas = None

    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        # Simple random t-based noise sampling
        if args.timestep_sampling == "sigmoid":
            # https://github.com/XLabs-AI/x-flux/tree/main
            t = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device))
        else:
            t = torch.rand((bsz,), device=device)

        timesteps = t * 1000.0
        t = t.view(-1, 1, 1, 1)
        noisy_model_input = (1 - t) * latents + t * noise
    elif args.timestep_sampling == "shift":
        shift = 0.25

        logits_norm = torch.randn(bsz, device=device)
        logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
        timesteps = logits_norm.sigmoid()
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)

        t = timesteps.view(-1, 1, 1, 1)
        timesteps = timesteps * 1000.0

    elif args.timestep_sampling == "flux_shift":
        logits_norm = torch.randn(bsz, device=device)
        logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
        timesteps = logits_norm.sigmoid()
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        timesteps = time_shift(mu, 1.0, timesteps)

        t = timesteps.view(-1, 1, 1, 1)
        timesteps = timesteps * 1000.0
        noisy_model_input = (1 - t) * latents + t * noise
    else:
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)

        # Add noise according to flow matching.
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

    return timesteps.to(dtype)


def get_normal_timesteps(
    args, noise_scheduler, latents, noise, device, dtype, timesteps=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, _, h, w = latents.shape
    sigmas = None

    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        # Simple random t-based noise sampling
        if args.timestep_sampling == "sigmoid":
            # https://github.com/XLabs-AI/x-flux/tree/main
            t = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device))
        else:
            t = torch.rand((bsz,), device=device)

        timesteps = t * 1000.0
        t = t.view(-1, 1, 1, 1)
        noisy_model_input = (1 - t) * latents + t * noise
    elif args.timestep_sampling == "shift":
        shift = args.discrete_flow_shift

        logits_norm = torch.randn(bsz, device=device)
        logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
        timesteps = logits_norm.sigmoid()
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)

        t = timesteps.view(-1, 1, 1, 1)
        timesteps = timesteps * 1000.0

    elif args.timestep_sampling == "flux_shift":
        logits_norm = torch.randn(bsz, device=device)
        logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
        timesteps = logits_norm.sigmoid()
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        timesteps = time_shift(mu, 1.0, timesteps)

        t = timesteps.view(-1, 1, 1, 1)
        timesteps = timesteps * 1000.0
        noisy_model_input = (1 - t) * latents + t * noise
    else:
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)

        # Add noise according to flow matching.
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

    return timesteps.to(dtype)



def get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents, noise, device, dtype, timesteps=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, _, h, w = latents.shape
    sigmas = None

    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        # Simple random t-based noise sampling
        if args.timestep_sampling == "sigmoid":
            # https://github.com/XLabs-AI/x-flux/tree/main
            t = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device))
        else:
            t = torch.rand((bsz,), device=device)

        timesteps = t * 1000.0
        t = t.view(-1, 1, 1, 1)
        noisy_model_input = (1 - t) * latents + t * noise
    elif args.timestep_sampling == "shift":
        shift = args.discrete_flow_shift

        if timesteps is None:
            logits_norm = torch.randn(bsz, device=device)
            logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
            timesteps = logits_norm.sigmoid()
            timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)

            t = timesteps.view(-1, 1, 1, 1)
            timesteps = timesteps * 1000.0
        else:
            t = timesteps.view(-1, 1, 1, 1) / 1000
        noisy_model_input = (1 - t) * latents + t * noise
    elif args.timestep_sampling == "flux_shift":
        logits_norm = torch.randn(bsz, device=device)
        logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
        timesteps = logits_norm.sigmoid()
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        timesteps = time_shift(mu, 1.0, timesteps)

        t = timesteps.view(-1, 1, 1, 1)
        timesteps = timesteps * 1000.0
        noisy_model_input = (1 - t) * latents + t * noise
    else:
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)

        # Add noise according to flow matching.
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas
