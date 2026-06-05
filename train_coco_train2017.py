# training with captions

# Swap blocks between CPU and GPU:
# This implementation is inspired by and based on the work of 2kpr.
# Many thanks to 2kpr for the original concept and implementation of memory-efficient offloading.
# The original idea has been adapted and extended to fit the current project's needs.

# Key features:
# - CPU offloading during forward and backward passes
# - Use of fused optimizer and grad_hook for efficient gradient processing
# - Per-block fused optimizer instances

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import os
import sys
from multiprocessing import Value
import time
from typing import List, Optional, Tuple, Union
import toml

sys.path.append(os.path.join(os.path.dirname(__file__), 'sd-scripts'))

from tqdm import tqdm

import torch
import torch.nn as nn
from library import utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from library import deepspeed_utils, flux_train_utils, flux_utils, strategy_base, strategy_flux
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util

# import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments
from dataset.jointconn_v2_dataset import JointDataset
from safetensors.torch import load_file
from jointconn_v2_library import inference_pipeline
from jointconn_v2_library import jointconn_v2_utils
from jointconn_v2_library.geometry import EdgeEnergyMap
from jointconn_v2_library.gcm_wfm import GCMWFMLoss, TaskBatch
from jointconn_v2_library.jointconn_v2_utils import load_empty_flux_model, setup_jointconn_v2_model, save_added_params

import pdb
import json

def load_state_dict_from_cache(pretrained_ckpt):
    print("Loading model...")
    pretrained_weight_path = pretrained_ckpt
    print("Loading pretrained weights from %s" % pretrained_weight_path)
    state_dict = load_file(pretrained_weight_path)
    return state_dict


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used from a forked multiprocessing child"""
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def _ensure_torch_device(dev):
    """Normalize accelerate.device into torch.device"""
    if isinstance(dev, torch.device):
        return dev
    if isinstance(dev, int):
        return torch.device(f"cuda:{dev}")
    if isinstance(dev, str):
        return torch.device(dev)
    # fallback: assume cuda:0
    return torch.device("cuda:0")


def _prepare_swapped_optimizer_step(model, optimizer, device):
    """Move trainable swapped-block params/states back to device before AdamW updates."""
    if hasattr(model, "wait_for_pending_block_swaps"):
        model.wait_for_pending_block_swaps()

    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.device != device:
                param.data = param.data.to(device, non_blocking=True)
            if param.grad is not None and param.grad.device != device:
                param.grad.data = param.grad.data.to(device, non_blocking=True)

            state = optimizer.state.get(param)
            if not state:
                continue
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.device != device:
                    state[key] = value.to(device, non_blocking=True)

def train(args):

    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)

    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    if args.cpu_offload_checkpointing and not args.gradient_checkpointing:
        logger.warning(
            "cpu_offload_checkpointing is enabled, so gradient_checkpointing is also enabled / cpu_offload_checkpointingが有効になっているため、gradient_checkpointingも有効になります"
        )
        args.gradient_checkpointing = True

    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ) or not args.cpu_offload_checkpointing, (
        "blocks_to_swap is not supported with cpu_offload_checkpointing / blocks_to_swapはcpu_offload_checkpointingと併用できません"
    )

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)

    _, is_schnell, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # load FLUX
    _, jointconn_model = load_empty_flux_model(
        args.pretrained_model_name_or_path,
        weight_dtype,
        device="cpu",
        disable_mmap=args.disable_mmap_load_safetensors
    )

    # normalize device for safe usage (avoid int device)
    device_norm = _ensure_torch_device(accelerator.device)
    jointconn_model = jointconn_model.to_empty(device=device_norm)

    # 2) Load base Flux weights only
    base_ckpt = load_state_dict_from_cache(args.pretrained_model_name_or_path)
    jointconn_model.load_state_dict(base_ckpt, strict=False)
    jointconn_model.requires_grad_(False)

    # 3) Attach JointConn-v2 adapters
    jointconn_config = {
        "beta_att": args.jointconn_beta_att,
        "local_kernel_sigma": args.jointconn_local_kernel_sigma,
        "routing_type": args.jointconn_routing_type,
        "gate_hidden_dim": args.jointconn_gate_hidden_dim,
        "routing_hidden_dim": args.jointconn_routing_hidden_dim,
        "output_bottleneck_dim": args.jointconn_output_bottleneck_dim,
        "max_token_hw": args.jointconn_max_token_hw,
        "max_bias_tokens": args.jointconn_max_bias_tokens,
        "zero_init_output_proj": True,
    }
    jointconn_model = setup_jointconn_v2_model(
        jointconn_model,
        lora_rank=args.jointconn_lora_rank,
        enable_jointconn_v2=args.enable_jointconn_v2,
        jointconn_config=jointconn_config,
        lora_branch_mode=args.lora_branch_mode,
    )

    if args.gradient_checkpointing:
        jointconn_model.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing)

    # backward compatibility
    if args.blocks_to_swap is None:
        blocks_to_swap = args.double_blocks_to_swap or 0
        if args.single_blocks_to_swap is not None:
            blocks_to_swap += args.single_blocks_to_swap // 2
        if blocks_to_swap > 0:
            logger.warning(
                "double_blocks_to_swap and single_blocks_to_swap are deprecated. Use blocks_to_swap instead."
                " / double_blocks_to_swapとsingle_blocks_to_swapは非推奨です。blocks_to_swapを使ってください。"
            )
            logger.info(
                f"double_blocks_to_swap={args.double_blocks_to_swap} and single_blocks_to_swap={args.single_blocks_to_swap} are converted to blocks_to_swap={blocks_to_swap}."
            )
            args.blocks_to_swap = blocks_to_swap
        del blocks_to_swap

    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

    if is_swapping_blocks:
        # Swap blocks between CPU and GPU to reduce memory usage, in forward and backward passes.
        # This idea is based on 2kpr's great work. Thank you!
        logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        device_for_swap = _ensure_torch_device(accelerator.device)
        jointconn_model.enable_block_swap(args.blocks_to_swap, device_for_swap)

    # load VAE here if not cached
    ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu")
    ae.requires_grad_(False)
    ae.eval()
    ae.to(device_norm, dtype=weight_dtype)

    training_models = []
    params_to_optimize = []
    training_models.append(jointconn_model)
    name_and_params = list(jointconn_model.named_parameters())
    params_to_optimize.append({
        "params": [p for _, p in name_and_params if p.requires_grad],
        "lr": args.learning_rate
    })
    param_names = [[n for n, p in name_and_params if p.requires_grad]]

    # calculate number of trainable parameters
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"number of trainable parameters: {n_params}")
    accelerator.print("prepare optimizer, data loader etc.")

    if args.blockwise_fused_optimizers:
        # fused backward pass (build per-block optimizers)
        grouped_params = []
        param_group = {}
        for group in params_to_optimize:
            named_parameters = list(jointconn_model.named_parameters())
            assert len(named_parameters) == len(group["params"]), "number of parameters does not match"
            for p, np in zip(group["params"], named_parameters):
                block_type = "other"
                if np[0].startswith("double_blocks"):
                    block_index = int(np[0].split(".")[1])
                    block_type = "double"
                elif np[0].startswith("single_blocks"):
                    block_index = int(np[0].split(".")[1])
                    block_type = "single"
                else:
                    block_index = -1

                key = (block_type, block_index)
                if key not in param_group:
                    param_group[key] = []
                param_group[key].append(p)

        block_types_and_indices = []
        for key, params in param_group.items():
            block_types_and_indices.append(key)
            grouped_params.append({"params": params, "lr": args.learning_rate})
            num_params = sum(p.numel() for p in params)
            accelerator.print(f"block {key}: {num_params} parameters")

        # prepare optimizers per group
        optimizers = []
        for group in grouped_params:
            _, _, opt = train_util.get_optimizer(args, trainable_params=[group])
            optimizers.append(opt)
        optimizer = optimizers[0]

        logger.info(f"using {len(optimizers)} optimizers for blockwise fused optimizers")

        if train_util.is_schedulefree_optimizer(optimizers[0], args):
            raise ValueError("Schedule-free optimizer is not supported with blockwise fused optimizers")
        optimizer_train_fn = lambda: None
        optimizer_eval_fn = lambda: None
    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

    # prepare dataloader
    train_dataset_group = JointDataset(args=args, image_folder=args.image_folder, drop_text=0.10, depth_transform=args.depth_transform)

    # ---- Windows-safe DataLoader (no prefetch_factor & no persistent workers when workers=0)
    is_windows = (os.name == "nt")
    if is_windows:
        n_workers = 0
        persistent_workers_flag = False
        dl_kwargs = dict(num_workers=0, pin_memory=True)
    else:
        n_workers = max(1, os.cpu_count() // 8)
        persistent_workers_flag = args.persistent_data_loader_n_workers if hasattr(args, "persistent_data_loader_n_workers") else args.persistent_data_loader_workers
        # 上行做兼容：不同版本里字段名可能有所变化；若没有 n_workers 字段，继续使用 persistent_data_loader_workers
        if isinstance(persistent_workers_flag, bool):
            pass
        else:
            persistent_workers_flag = args.persistent_data_loader_workers
        dl_kwargs = dict(num_workers=n_workers, pin_memory=True, prefetch_factor=4)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=args.train_batch_size,
        shuffle=True,
        persistent_workers=persistent_workers_flag,
        **dl_kwargs,
    )

    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / Number of steps until the specified epoch: {args.max_train_steps}"
        )

    # データセット側にも学習ステップを送信
    if args.blockwise_fused_optimizers:
        lr_schedulers = [train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes) for optimizer in optimizers]
        lr_scheduler = lr_schedulers[0]
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        jointconn_model.to(weight_dtype)
    elif args.full_bf16:
        assert args.mixed_precision == "bf16", "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        jointconn_model.to(weight_dtype)

    clean_memory_on_device(device_norm)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(args, mmdit=jointconn_model)
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(ds_model, optimizer, train_dataloader, lr_scheduler)
        training_models = [ds_model]
    else:
        jointconn_model = accelerator.prepare(jointconn_model, device_placement=[not is_swapping_blocks])
        if is_swapping_blocks:
            accelerator.unwrap_model(jointconn_model).move_to_device_except_swap_blocks(device_norm)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.fused_backward_pass:
        import library.adafactor_fused
        library.adafactor_fused.patch_adafactor_fused(optimizer)
        for param_group, param_name_group in zip(optimizer.param_groups, param_names):
            for parameter, param_name in zip(param_group["params"], param_name_group):
                if parameter.requires_grad:
                    def create_grad_hook(p_name, p_group):
                        def grad_hook(tensor: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                            optimizer.step_param(tensor, p_group)
                            tensor.grad = None
                        return grad_hook
                    parameter.register_post_accumulate_grad_hook(create_grad_hook(param_name, param_group))

    elif args.blockwise_fused_optimizers:
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        global optimizer_hooked_count
        global num_parameters_per_group
        global parameter_optimizer_map

        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, optimizer_i in enumerate(optimizers):
            for param_group in optimizer_i.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:
                        def grad_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, args.max_grad_norm)
                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)
                        parameter.register_post_accumulate_grad_hook(grad_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    # epoch数を計算する
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    # 学習する
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print("Total batch size: %d" % total_batch_size)
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.__len__()}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

    global_step = 0

    noise_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    edge_energy = EdgeEnergyMap().to(device_norm) if args.enable_jointconn_v2 else None
    gcm_wfm_loss = GCMWFMLoss.from_args(args) if args.enable_jointconn_v2 else None

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    if is_swapping_blocks:
        accelerator.unwrap_model(jointconn_model).prepare_block_swap_before_forward()

    if args.is_latent_training:
        clip_l = None
        t5xxl = None
    else:
        clip_l = flux_utils.load_clip_l(
            args.clip_l, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
        )
        t5xxl = flux_utils.load_t5xxl(
            args.t5xxl, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
        )
        for m in (clip_l, t5xxl):
            m.eval().requires_grad_(False)
            m.to(device_norm)

        t5xxl_max_len = 512
        flux_tok = strategy_flux.FluxTokenizeStrategy(t5xxl_max_len)
        strategy_base.TokenizeStrategy.set_strategy(flux_tok)
        txt_enc = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)
        strategy_base.TextEncodingStrategy.set_strategy(txt_enc)

    # For --sample_at_first
    optimizer_eval_fn()
    inference_pipeline.sample_images(accelerator, args, 0, global_step, jointconn_model, ae, [clip_l, t5xxl], weight_dtype)
    optimizer_train_fn()

    if len(accelerator.trackers) > 0:
        accelerator.log({}, step=0)

    loss_recorder = train_util.LossRecorder()
    epoch = 0  # avoid error when max_train_steps is 0

    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if args.blockwise_fused_optimizers:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}

            with accelerator.accumulate(*training_models):

                if args.is_latent_training:
                    rgb_latents = batch["image_latent"].to(device_norm, dtype=weight_dtype)
                    depth_latents = batch["depth_latent"].to(device_norm, dtype=weight_dtype)

                    l_pooled = batch["clip_latent"].to(device_norm, dtype=weight_dtype)
                    t5_out = batch["t5_latents"].to(device_norm, dtype=weight_dtype)

                    if args.is_txt_ids_training:
                        # keep integer dtype for txt_ids
                        txt_ids = batch["txt_ids"].to(device_norm)
                    else:
                        txt_ids = torch.zeros(t5_out.shape[0], t5_out.shape[1], 3, device=device_norm)

                    l_pooled = torch.cat([l_pooled, l_pooled], dim=0)
                    t5_out = torch.cat([t5_out, t5_out], dim=0)
                    txt_ids = torch.cat([txt_ids, txt_ids], dim=0)

                    if args.is_attnmask_training:
                        t5_attn_mask = batch["attn_mask"].to(device_norm)
                        t5_attn_mask = torch.cat([t5_attn_mask, t5_attn_mask], 0)
                    else:
                        t5_attn_mask = None

                    if torch.any(torch.isnan(rgb_latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        rgb_latents = torch.nan_to_num(rgb_latents, 0, out=rgb_latents)

                    if torch.any(torch.isnan(depth_latents)):
                        accelerator.print("NaN found in depth latents, replacing with zeros")
                        depth_latents = torch.nan_to_num(depth_latents, 0, out=depth_latents)

                else:
                    image = batch["image"].to(device_norm, dtype=weight_dtype)
                    depth = batch["depth"].to(device_norm, dtype=weight_dtype)
                    text_prompt = batch["text"]

                    with torch.no_grad():
                        rgb_latents = ae.encode(image)
                        depth_latents = ae.encode(depth)

                        # ensure list[str] for tokenizer
                        text_list = text_prompt if isinstance(text_prompt, list) else [text_prompt]
                        tokens_and_masks = flux_tok.tokenize(text_list)
                        inputs = [t.to(device_norm) for t in tokens_and_masks]
                        conds = txt_enc.encode_tokens(flux_tok, [clip_l, t5xxl], inputs, args.apply_t5_attn_mask)
                        if args.full_fp16:
                            conds = [c.to(weight_dtype) for c in conds]
                        l_pooled, t5_out, txt_ids, t5_attn_mask = conds

                        l_pooled = torch.cat([l_pooled, l_pooled], dim=0)
                        t5_out = torch.cat([t5_out, t5_out], dim=0)
                        txt_ids = torch.cat([txt_ids, txt_ids], dim=0)

                        if not args.is_txt_ids_training:
                            txt_ids = torch.zeros(t5_out.shape[0], t5_out.shape[1], 3, device=device_norm)

                        if not args.is_attnmask_training:
                            t5_attn_mask = None
                        else:
                            t5_attn_mask = t5_attn_mask.repeat(2, 1)

                # Merge RGB and Depth inputs
                latents = torch.cat([rgb_latents, depth_latents], dim=0)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                pair_bsz = bsz // 2

                if args.enable_jointconn_v2:
                    eps_t = 1e-5

                    def rand_time():
                        return torch.rand(pair_bsz, device=device_norm, dtype=weight_dtype) * (1.0 - 2 * eps_t) + eps_t

                    is_joint = torch.rand(pair_bsz, device=device_norm) < args.p_joint_task
                    is_sync = torch.rand(pair_bsz, device=device_norm) < args.p_sync

                    t_joint_x = rand_time()
                    t_joint_y_ind = rand_time()
                    t_joint_y = torch.where(is_sync, t_joint_x, t_joint_y_ind)
                    t_cond_x = rand_time()
                    t_cond_y = torch.zeros_like(t_cond_x)

                    timesteps_rgb = torch.where(is_joint, t_joint_x, t_cond_x)
                    timesteps_dep = torch.where(is_joint, t_joint_y, t_cond_y)

                    m_x = torch.ones(pair_bsz, device=device_norm, dtype=weight_dtype)
                    m_y = is_joint.to(weight_dtype)
                    lambda_x = torch.ones(pair_bsz, device=device_norm, dtype=weight_dtype)
                    lambda_y = is_joint.to(weight_dtype)
                    task_batch = TaskBatch(timesteps_rgb, timesteps_dep, m_x, m_y, lambda_x, lambda_y)

                    t_rgb_view = timesteps_rgb.view(pair_bsz, 1, 1, 1)
                    t_dep_view = timesteps_dep.view(pair_bsz, 1, 1, 1)
                    noisy_model_input_rgb = (1 - t_rgb_view) * rgb_latents + t_rgb_view * noise[:pair_bsz]
                    noisy_model_input_dep = (1 - t_dep_view) * depth_latents + t_dep_view * noise[pair_bsz:]
                    noisy_model_input = torch.cat([noisy_model_input_rgb, noisy_model_input_dep], dim=0)
                    timesteps = torch.cat([timesteps_rgb, timesteps_dep], dim=0)

                    packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)
                    packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
                    img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=device_norm)
                    guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=device_norm)

                    target = noise - latents
                    packed_target = flux_utils.pack_latents(target)
                    tau_x = packed_target[:pair_bsz]
                    tau_y = packed_target[pair_bsz:]

                    if not args.is_latent_training and "depth" in batch:
                        depth_1ch = ((batch["depth"].to(device_norm, dtype=weight_dtype) + 1.0) / 2.0).mean(dim=1, keepdim=True)
                        e_loss = edge_energy(depth_1ch, (packed_latent_height, packed_latent_width)).to(device_norm, dtype=weight_dtype)
                    elif "depth_edge" in batch:
                        e_loss = batch["depth_edge"].to(device_norm, dtype=weight_dtype)
                    else:
                        e_loss = torch.zeros(pair_bsz, packed_noisy_model_input.shape[1], device=device_norm, dtype=weight_dtype)

                    e_att_zero = torch.zeros_like(e_loss)
                    is_depth_conditioned = (~is_joint).view(pair_bsz, 1)
                    e_att = torch.where(is_depth_conditioned, e_loss, e_att_zero)

                    with accelerator.autocast():
                        model_pred = jointconn_model(
                            img=packed_noisy_model_input,
                            img_ids=img_ids,
                            txt=t5_out,
                            txt_ids=txt_ids,
                            y=l_pooled,
                            timesteps=timesteps,
                            guidance=guidance_vec,
                            txt_attention_mask=t5_attn_mask,
                            e_att=e_att,
                            task_masks=task_batch.as_masks(),
                        )

                    v_x = model_pred[:pair_bsz]
                    v_y = model_pred[pair_bsz:]
                    loss_out = gcm_wfm_loss(v_x, v_y, tau_x, tau_y, e_loss, task_batch)
                    loss = loss_out.loss
                else:
                    # Legacy paired RGB-depth timestep sampling strategy.
                    scenario_indices = torch.randint(0, 4, (pair_bsz,), device=device_norm)
                    small_steps = jointconn_v2_utils.get_small_timesteps(
                        args, noise_scheduler_copy, latents[:pair_bsz], noise[:pair_bsz], device_norm, weight_dtype)
                    normal_steps = jointconn_v2_utils.get_normal_timesteps(
                        args, noise_scheduler_copy, latents[:pair_bsz], noise[:pair_bsz], device_norm, weight_dtype)

                    timesteps_rgb = torch.zeros_like(small_steps)
                    timesteps_dep = torch.zeros_like(small_steps)

                    for i in range(pair_bsz):
                        scenario = scenario_indices[i].item()
                        if scenario == 0 or scenario == 1:
                            timesteps_rgb[i] = normal_steps[i]
                            timesteps_dep[i] = normal_steps[i]
                        elif scenario == 2:
                            timesteps_rgb[i] = normal_steps[i]
                            timesteps_dep[i] = small_steps[i]
                        else:
                            timesteps_rgb[i] = small_steps[i]
                            timesteps_dep[i] = normal_steps[i]

                    noisy_model_input_rgb, timesteps_rgb, sigmas = jointconn_v2_utils.get_noisy_model_input_and_timesteps(
                        args, noise_scheduler_copy, latents[:pair_bsz], noise[:pair_bsz], device_norm, weight_dtype, timesteps=timesteps_rgb
                    )
                    noisy_model_input_dep, timesteps_dep, sigmas_dep = jointconn_v2_utils.get_noisy_model_input_and_timesteps(
                        args, noise_scheduler_copy, latents[pair_bsz:], noise[pair_bsz:], device_norm, weight_dtype, timesteps=timesteps_dep
                    )

                    noisy_model_input = torch.cat([noisy_model_input_rgb, noisy_model_input_dep], dim=0)
                    timesteps = torch.cat([timesteps_rgb, timesteps_dep], dim=0)

                    packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
                    packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
                    img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=device_norm)
                    guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=device_norm)

                    with accelerator.autocast():
                        model_pred = jointconn_model(
                            img=packed_noisy_model_input,
                            img_ids=img_ids,
                            txt=t5_out,
                            txt_ids=txt_ids,
                            y=l_pooled,
                            timesteps=timesteps / 1000,
                            guidance=guidance_vec,
                            txt_attention_mask=t5_attn_mask,
                        )

                    model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)
                    model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)
                    target = noise - latents

                    huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
                    loss = train_util.conditional_loss(model_pred.float(), target.float(), args.loss_type, "none", huber_c)
                    if weighting is not None:
                        loss = loss * weighting
                    if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])

                    loss_weights = 1.0
                    loss = (loss * loss_weights).mean()

                # backward
                accelerator.backward(loss)

                if not (args.fused_backward_pass or args.blockwise_fused_optimizers):
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(p for p in m.parameters() if p.requires_grad)
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    if is_swapping_blocks:
                        _prepare_swapped_optimizer_step(accelerator.unwrap_model(jointconn_model), optimizer, device_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    lr_scheduler.step()
                    if args.blockwise_fused_optimizers:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

            # optimization step done?
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                optimizer_eval_fn()
                inference_pipeline.sample_images(
                    accelerator, args, None, global_step, jointconn_model, ae, [clip_l, t5xxl], weight_dtype
                )

                if global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()

                    save_path = os.path.join(args.output_dir, f"jointconn_v2_addons_step_{global_step:06d}.safetensors")
                    save_added_params(jointconn_model, save_path, dtype=torch.bfloat16)

                    flags = {
                        "is_txt_ids_training": args.is_txt_ids_training,
                        "is_attnmask_training": args.is_attnmask_training,
                        "depth_transform": args.depth_transform,
                        "enable_jointconn_v2": args.enable_jointconn_v2,
                        "jointconn_beta_att": args.jointconn_beta_att,
                        "jointconn_local_kernel_sigma": args.jointconn_local_kernel_sigma,
                        "jointconn_routing_type": args.jointconn_routing_type,
                        "jointconn_gate_hidden_dim": args.jointconn_gate_hidden_dim,
                        "jointconn_routing_hidden_dim": args.jointconn_routing_hidden_dim,
                        "jointconn_output_bottleneck_dim": args.jointconn_output_bottleneck_dim,
                        "jointconn_max_token_hw": args.jointconn_max_token_hw,
                        "jointconn_max_bias_tokens": args.jointconn_max_bias_tokens,
                        "jointconn_beta_loss": args.jointconn_beta_loss,
                        "jointconn_gamma_t": args.jointconn_gamma_t,
                        "jointconn_alpha_min": args.jointconn_alpha_min,
                        "jointconn_alpha_max": args.jointconn_alpha_max,
                        "jointconn_alpha_eps": args.jointconn_alpha_eps,
                        "p_joint_task": args.p_joint_task,
                        "p_sync": args.p_sync,
                        "jointconn_lora_rank": args.jointconn_lora_rank,
                        "lora_branch_mode": args.lora_branch_mode,
                    }
                    json_path = os.path.join(args.output_dir, f"jointconn_v2_addons_step_{global_step:06d}.json")
                    with open(json_path, "w") as jf:
                        json.dump(flags, jf, indent=4)

                    print(f"Saved training flags to {json_path}")

                optimizer_train_fn()

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=True)
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        optimizer_eval_fn()
        optimizer_train_fn()

    is_main_process = accelerator.is_main_process
    jointconn_model = accelerator.unwrap_model(jointconn_model)

    accelerator.end_training()
    optimizer_eval_fn()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator

    if is_main_process:
        if getattr(args, "skip_train_end_full_model_save", False):
            logger.info("skip full model save on train end.")
        else:
            flux_train_utils.save_flux_model_on_train_end(args, save_dtype, epoch, global_step, jointconn_model)
            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)  # TODO split this
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    add_custom_train_arguments(parser)  # TODO remove this from here
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)

    parser.add_argument(
        "--mem_eff_save",
        action="store_true",
        help="[EXPERIMENTAL] use memory efficient custom model saving method / メモリ効率の良い独自のモデル保存方法を使う",
    )
    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="**this option is not working** will be removed in the future / このオプションは動作しません。将来削除されます",
    )
    parser.add_argument(
        "--blockwise_fused_optimizers",
        action="store_true",
        help="enable blockwise optimizers for fused backward pass and optimizer step / fused backward passとoptimizer step のためブロック単位のoptimizerを有効にする",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead / 代わりに 'skip_cache_check' を使用してください",
    )
    parser.add_argument(
        "--double_blocks_to_swap",
        type=int,
        default=None,
        help="[Deprecated] use 'blocks_to_swap' instead / 代わりに 'blocks_to_swap' を使用してください",
    )
    parser.add_argument(
        "--single_blocks_to_swap",
        type=int,
        default=None,
        help="[Deprecated] use 'blocks_to_swap' instead / 代わりに 'blocks_to_swap' を使用してください",
    )
    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing / チェックポイント時にテンソルをCPUにオフロードする",
    )
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument(
        "--is_latent_training",
        action="store_true",
        help="If set, enables latent training mode."
    )
    parser.add_argument(
        "--is_txt_ids_training",
        action="store_true",
        help="If set, use the t5 txt_ids for training (default: false)."
    )
    parser.add_argument(
        "--is_attnmask_training",
        action="store_true",
        help="If set, use the t5 attention mask for training (default: false)."
    )
    parser.add_argument(
        '--depth_transform',
        type=str,
        choices=["none", "inverse"],
        default="none",
        help="Specify how to transform the depth map: 'none' or 'inverse'"
    )
    parser.add_argument(
        "--output_resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help=("Output resolution for joint_generation mode, as two integers: width height, e.g., 512 512 or 1024 1024.")
    )
    parser.add_argument("--exp_name", type=str, default="training")
    parser.add_argument("--skip_train_end_full_model_save", action="store_true", help="Skip saving the full FLUX model at train end; useful for adapter-only dry runs.")
    parser.add_argument("--enable_jointconn_v2", action="store_true", help="Enable JointConn-v2 connector and GCM-WFM training path.")
    parser.add_argument("--jointconn_beta_att", type=float, default=1.0)
    parser.add_argument("--jointconn_local_kernel_sigma", type=float, default=3.0)
    parser.add_argument("--jointconn_routing_type", type=str, choices=["three_way", "two_sigmoid"], default="three_way")
    parser.add_argument("--jointconn_gate_hidden_dim", type=int, default=128)
    parser.add_argument("--jointconn_routing_hidden_dim", type=int, default=64)
    parser.add_argument("--jointconn_output_bottleneck_dim", type=int, default=128)
    parser.add_argument("--jointconn_max_token_hw", type=int, default=64)
    parser.add_argument("--jointconn_max_bias_tokens", type=int, default=2048)
    parser.add_argument("--jointconn_beta_loss", type=float, default=2.0)
    parser.add_argument("--jointconn_gamma_t", type=float, default=1.0)
    parser.add_argument("--jointconn_alpha_min", type=float, default=0.25)
    parser.add_argument("--jointconn_alpha_max", type=float, default=4.0)
    parser.add_argument("--jointconn_alpha_eps", type=float, default=1e-6)
    parser.add_argument("--p_joint_task", type=float, default=0.5)
    parser.add_argument("--p_sync", type=float, default=0.5)
    parser.add_argument("--joint_train_e_att_mode", type=str, choices=["zero"], default="zero")
    parser.add_argument("--jointconn_lora_rank", type=int, default=64)
    parser.add_argument("--lora_branch_mode", type=str, choices=["depth_only", "shared_both", "separate"], default="depth_only")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    # Load depth_transform flag from depth_transform.json in the specified folder
    import os as _os, json as _json
    depth_json = _os.path.join(args.image_folder, "depth_transform.json")
    if _os.path.isfile(depth_json):
        with open(depth_json, "r", encoding="utf-8") as f:
            data = _json.load(f)
        args.depth_transform = data.get("depth_transform", getattr(args, "depth_transform", "none"))
        print(f"Loaded depth_transform from {depth_json}: {args.depth_transform}")
    else:
        args.depth_transform = getattr(args, "depth_transform", "none")
        print(f"No depth_transform.json found in {args.image_folder}; using {args.depth_transform!r}")

    train(args)
