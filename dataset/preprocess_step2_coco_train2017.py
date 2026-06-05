# -*- coding: utf-8 -*-
import argparse
import glob
import json
import logging
import os
import sys
from functools import lru_cache
from typing import List
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np
from accelerate.utils import set_seed
from safetensors.torch import load_file
from PIL import Image

# --- add local paths for sd-scripts utils ---
sys.path.append('./sd-scripts')

from library import (
    deepspeed_utils,
    strategy_flux,
    strategy_base,
    flux_train_utils,
    flux_utils,
)
from library.device_utils import init_ipex, clean_memory_on_device
import library.train_util as train_util
import library.config_util as config_util

init_ipex()
logger = logging.getLogger(__name__)

# -------------------------
# Dataset
# -------------------------
class PreprocessDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder: str, depth_transform: str):
        self.image_dir = os.path.join(image_folder, "processed_images")
        self.depth_dir = os.path.join(image_folder, "depthmaps")
        self.text_dir  = os.path.join(image_folder, "text_prompts")
        self.depth_transform = depth_transform

        image_paths = glob.glob(os.path.join(self.image_dir, "*.jpg")) + \
                      glob.glob(os.path.join(self.image_dir, "*.png"))

        self.filenames = []
        for p in sorted(image_paths):
            name = os.path.splitext(os.path.basename(p))[0]
            if (os.path.exists(os.path.join(self.depth_dir, name + ".npy"))
                and os.path.exists(os.path.join(self.text_dir, name + ".txt"))):
                self.filenames.append(name)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # image → [-1,1], CHW
        img_path = os.path.join(self.image_dir, fname + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, fname + ".png")
        image = Image.open(img_path).convert("RGB")
        image = np.asarray(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3,H,W]
        image = image * 2 - 1

        # depth (0~1 相对视差) → [-1,1]，升成 3 通道 CHW
        depth = np.load(os.path.join(self.depth_dir, fname + ".npy")).astype(np.float32)
        #（稳妥）再做一次 min-max，防止个别异常
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max - d_min < 1e-8:
            depth = np.zeros_like(depth, dtype=np.float32)
        else:
            depth = (depth - d_min) / (d_max - d_min)
        if self.depth_transform == "inverse":
            depth = 1.0 - depth
        depth = depth * 2 - 1
        depth = torch.from_numpy(depth).float().unsqueeze(0).repeat(3, 1, 1)

        # text
        with open(os.path.join(self.text_dir, fname + ".txt"), "r", encoding="utf-8") as f:
            prompt = f.read().strip()

        return {"image": image, "depth": depth, "text": prompt, "filename": fname}

# -------------------------
# Helpers
# -------------------------
@lru_cache(maxsize=1)
def load_state_dict_from_cache(pretrained_ckpt: str) -> dict:
    print(f"Loading pretrained weights from {pretrained_ckpt}")
    state_dict = load_file(pretrained_ckpt)
    return state_dict

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute latents (images/depth/text)")
    # sd-scripts common args
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    config_util.add_config_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)

    parser.add_argument('--image_folder', default=None, help='data/train2017')
    parser.add_argument('--depth_transform', type=str, choices=['none', 'inverse'], required=True)
    return parser

# -------------------------
# Main
# -------------------------
def preprocess_step2(args: argparse.Namespace):
    # deepspeed / seed
    deepspeed_utils.prepare_deepspeed_args(args)
    if args.seed is not None:
        set_seed(args.seed)

    # accelerator & dtypes
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # --- VAE ---
    ae = flux_utils.load_ae(args.ae, weight_dtype, device="cpu")
    ae.eval().requires_grad_(False)
    ae.to(accelerator.device, dtype=weight_dtype)
    clean_memory_on_device(accelerator.device)

    # --- Text models: CLIP + T5-XXL ---
    clip_l = flux_utils.load_clip_l(
        args.clip_l, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
    )
    t5xxl = flux_utils.load_t5xxl(
        args.t5xxl, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
    )
    for m in (clip_l, t5xxl):
        m.eval().requires_grad_(False)
        m.to(accelerator.device)

    # tokenizer / encoding strategies
    t5xxl_max_len = 512
    flux_tok = strategy_flux.FluxTokenizeStrategy(t5xxl_max_len)
    strategy_base.TokenizeStrategy.set_strategy(flux_tok)
    txt_enc = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)
    strategy_base.TextEncodingStrategy.set_strategy(txt_enc)

    # DataLoader: Windows → workers=0（更稳）
    is_windows = (os.name == 'nt')
    n_workers = 0 if is_windows else max(1, os.cpu_count() // 8)
    dataset = PreprocessDataset(image_folder=args.image_folder, depth_transform=args.depth_transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(not is_windows) and torch.cuda.is_available(),
        persistent_workers=False if is_windows else (n_workers > 0),
    )
    dataloader = accelerator.prepare(dataloader)

    # 兼容 --full_bf16 / --full_fp16
    full_low_precision = bool(getattr(args, 'full_bf16', False) or getattr(args, 'full_fp16', False))
    if full_low_precision and getattr(args, 'full_fp16', False):
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # 输出目录
    out_dirs = {
        "image_latents": os.path.join(args.image_folder, "image_latents"),
        "depth_latents": os.path.join(args.image_folder, "depth_latents"),
        "clip_latents":  os.path.join(args.image_folder, "clip_latents"),
        "t5_latents":    os.path.join(args.image_folder, "t5_latents"),
        "txt_ids":       os.path.join(args.image_folder, "txt_ids"),
        "attn_masks":    os.path.join(args.image_folder, "attn_masks"),
    }
    for p in out_dirs.values():
        os.makedirs(p, exist_ok=True)

    # 记录 depth_transform
    with open(os.path.join(args.image_folder, "depth_transform.json"), "w") as f:
        json.dump({"depth_transform": args.depth_transform}, f, indent=2)

    # 逐样本处理
    pbar = tqdm(dataloader, desc="Precompute latents", total=len(dataloader))
    for batch in pbar:
        image = batch["image"].to(accelerator.device, dtype=weight_dtype)  # [1,3,H,W]
        depth = batch["depth"].to(accelerator.device, dtype=weight_dtype)  # [1,3,H,W]
        fname = batch["filename"][0]

        # ---- VAE encode ----
        with torch.no_grad():
            image_latent = ae.encode(image)[0]   # [16, H//8, W//8]
            depth_latent = ae.encode(depth)[0]   # [16, H//8, W//8]

        # ---- Text encode ----
        text_prompt = batch["text"]
        # DataLoader 会把字符串批成 list[str]；确保是 list 传给 tokenizer
        if isinstance(text_prompt, list):
            text_list = text_prompt
        else:
            text_list = [text_prompt]

        with torch.no_grad():
            tokens_and_masks = flux_tok.tokenize(text_list)  # batch 形式
            inputs = [t.to(accelerator.device) for t in tokens_and_masks]
            conds = txt_enc.encode_tokens(flux_tok, [clip_l, t5xxl], inputs, args.apply_t5_attn_mask)
            if full_low_precision:
                conds = [c.to(weight_dtype) for c in conds]
            l_pooled, t5_out, txt_ids, attn_mask = conds  # [B,...]

        # 取 batch=1 的第 0 个样本
        l_pooled  = l_pooled[0]   # [768]
        t5_out    = t5_out[0]     # [512,4096]
        txt_ids   = txt_ids[0]    # [512,3]
        attn_mask = attn_mask[0]  # [512]

        # ---- Save ----
        torch.save(image_latent, os.path.join(out_dirs["image_latents"], f"{fname}.pt"))
        torch.save(depth_latent, os.path.join(out_dirs["depth_latents"], f"{fname}.pt"))
        torch.save(l_pooled,  os.path.join(out_dirs["clip_latents"],  f"{fname}.pt"))
        torch.save(t5_out,    os.path.join(out_dirs["t5_latents"],    f"{fname}.pt"))
        torch.save(txt_ids,   os.path.join(out_dirs["txt_ids"],       f"{fname}.pt"))
        torch.save(attn_mask.cpu().to(torch.int32), os.path.join(out_dirs["attn_masks"], f"{fname}.pt"))

        # 显存清理（更稳）
        clean_memory_on_device(accelerator.device)

# -------------------------
if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    # 验证与合并配置
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    # 执行
    preprocess_step2(args)
