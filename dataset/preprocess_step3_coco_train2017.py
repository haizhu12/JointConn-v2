# -*- coding: utf-8 -*-
import argparse
import os
import sys
import json
import shutil
from functools import lru_cache
from typing import List
from tqdm import tqdm

import torch
from accelerate.utils import set_seed
from safetensors.torch import load_file

# 让脚本能找到 sd-scripts 的库
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

@lru_cache(maxsize=1)
def load_state_dict_from_cache(pretrained_ckpt: str) -> dict:
    print(f"Loading pretrained weights from {pretrained_ckpt}")
    return load_file(pretrained_ckpt)

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step3: Precompute latents for evaluation prompts"
    )
    # ====== 和 step2 一样把这些参数都加上（包含 deepspeed）======
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    config_util.add_config_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)  # ← 关键修复：注入 args.deepspeed 等

    # 自定义评测文本与输出根目录
    parser.add_argument("--eval_txt", default="dataset/evaluation_prompts.txt",
                        help="Path to evaluation prompts txt (one prompt per line).")
    parser.add_argument("--out_root", default="dataset",
                        help="Root folder to save evaluation/empty prompts latents.")
    return parser

def _ensure_deepspeed_defaults(args):
    """双保险：即使上面没注入，也补上必须字段，防止 AttributeError。"""
    if not hasattr(args, "deepspeed"):
        args.deepspeed = False
    if not hasattr(args, "ds_config"):
        args.ds_config = None

def preprocess_step3(args: argparse.Namespace):
    # deepspeed / seed
    _ensure_deepspeed_defaults(args)                     # ← 兜底
    deepspeed_utils.prepare_deepspeed_args(args)
    if args.seed is not None:
        set_seed(args.seed)

    # accelerator & dtype
    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # 文本模型（与 step2 对齐）
    clean_memory_on_device(accelerator.device)
    clip_l = flux_utils.load_clip_l(
        args.clip_l, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
    )
    t5xxl = flux_utils.load_t5xxl(
        args.t5xxl, weight_dtype, device="cpu", disable_mmap=args.disable_mmap_load_safetensors
    )
    for m in (clip_l, t5xxl):
        m.eval().requires_grad_(False)
        m.to(accelerator.device)

    # tokenizer & encoding
    t5xxl_max_len = 512
    flux_tok = strategy_flux.FluxTokenizeStrategy(t5xxl_max_len)
    strategy_base.TokenizeStrategy.set_strategy(flux_tok)
    txt_enc = strategy_flux.FluxTextEncodingStrategy(args.apply_t5_attn_mask)
    strategy_base.TextEncodingStrategy.set_strategy(txt_enc)

    # 兼容 --full_bf16 / --full_fp16
    full_low_precision = bool(getattr(args, "full_bf16", False) or getattr(args, "full_fp16", False))
    if getattr(args, "full_fp16", False):
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # 读入评测文本
    eval_txt_path = args.eval_txt
    if not os.path.exists(eval_txt_path):
        raise FileNotFoundError(f"Evaluation prompts txt not found: {eval_txt_path}")
    with open(eval_txt_path, "r", encoding="utf-8") as f:
        prompt_list = [line.strip() for line in f if line.strip()]

    # 在首位加入“空提示”
    prompt_list.insert(0, " ")

    # 输出目录（放 dataset 下，和 README/训练脚本一致）
    eval_dir  = os.path.join(args.out_root, "evaluation_prompts")
    empty_dir = os.path.join(args.out_root, "empty_prompts")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(empty_dir, exist_ok=True)
    shutil.copy(eval_txt_path, os.path.join(eval_dir, "evaluation_prompts.txt"))

    latent_dirs = {
        "clip_latents": os.path.join(eval_dir,  "clip_latents"),
        "t5_latents":   os.path.join(eval_dir,  "t5_latents"),
        "txt_ids":      os.path.join(eval_dir,  "txt_ids"),
        "attn_masks":   os.path.join(eval_dir,  "attn_masks"),
    }
    empty_latent_dirs = {
        "clip_latents": os.path.join(empty_dir, "clip_latents"),
        "t5_latents":   os.path.join(empty_dir, "t5_latents"),
        "txt_ids":      os.path.join(empty_dir, "txt_ids"),
        "attn_masks":   os.path.join(empty_dir, "attn_masks"),
    }
    for d in (latent_dirs, empty_latent_dirs):
        for p in d.values():
            os.makedirs(p, exist_ok=True)

    # 编码 & 保存
    for idx, text_prompt in enumerate(tqdm(prompt_list, desc="Encoding evaluation prompts")):
        # 统一用 batch 列表形式，以和 step2 保持一致
        text_batch = [text_prompt]
        with torch.no_grad():
            tokens_and_masks = flux_tok.tokenize(text_batch)
            inputs = [t.to(accelerator.device) for t in tokens_and_masks]
            conds = txt_enc.encode_tokens(flux_tok, [clip_l, t5xxl], inputs, args.apply_t5_attn_mask)
            if full_low_precision:
                conds = [c.to(weight_dtype) for c in conds]
            l_pooled, t5_out, txt_ids, attn_mask = conds  # [B,...]

        # 取 batch=1 的第 0 个
        l_pooled  = l_pooled[0]
        t5_out    = t5_out[0]
        txt_ids   = txt_ids[0]
        attn_mask = attn_mask[0]

        if idx == 0:
            # 空提示
            torch.save(l_pooled,  os.path.join(empty_latent_dirs["clip_latents"], "empty.pt"))
            torch.save(t5_out,    os.path.join(empty_latent_dirs["t5_latents"],   "empty.pt"))
            torch.save(txt_ids,   os.path.join(empty_latent_dirs["txt_ids"],      "empty.pt"))
            torch.save(attn_mask.cpu().to(torch.int32),
                       os.path.join(empty_latent_dirs["attn_masks"], "empty.pt"))
        else:
            fname = f"{idx-1:05d}"
            torch.save(l_pooled,  os.path.join(latent_dirs["clip_latents"], fname + ".pt"))
            torch.save(t5_out,    os.path.join(latent_dirs["t5_latents"],   fname + ".pt"))
            torch.save(txt_ids,   os.path.join(latent_dirs["txt_ids"],      fname + ".pt"))
            torch.save(attn_mask.cpu().to(torch.int32),
                       os.path.join(latent_dirs["attn_masks"], fname + ".pt"))

    print("\n[Step3] Done. Saved to:")
    print(f"  {eval_dir}")
    print(f"  {empty_dir}")

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    # 验证 + 读取 config（和 step2 行为一致）
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    preprocess_step3(args)
