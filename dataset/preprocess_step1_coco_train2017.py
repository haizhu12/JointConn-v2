# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path
from glob import glob
import shutil
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch

def add_depth_repo_to_path(depth_repo: str):
    """If user supplied a Depth-Anything-V2 repo path, append to sys.path."""
    if depth_repo:
        depth_repo = str(Path(depth_repo).resolve())
        if depth_repo not in sys.path:
            sys.path.append(depth_repo)

def try_import_depth_anything():
    """
    Try to import Depth-Anything-V2 model class.
    Expecting: from depth_anything_v2.dpt import DepthAnythingV2
    """
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
        return DepthAnythingV2
    except Exception as e:
        print("[ERROR] Cannot import Depth-Anything-V2. "
              "Set --depth_repo to your repo root or ensure it's on PYTHONPATH.")
        raise

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image_file(p: Path):
    return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]

def resize_and_center_crop_rgb(img_rgb: np.ndarray, target: int = 512) -> np.ndarray:
    """
    Short-side resize to 'target', then center-crop to (target, target).
    img_rgb: HxWx3, uint8 RGB
    """
    h, w = img_rgb.shape[:2]
    scale = float(target) / min(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = max(0, (nh - target) // 2)
    x0 = max(0, (nw - target) // 2)
    cropped = resized[y0:y0 + target, x0:x0 + target]
    if cropped.shape[0] != target or cropped.shape[1] != target:
        cropped = cv2.resize(cropped, (target, target), interpolation=cv2.INTER_AREA)
    return cropped

def minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)

def safe_symlink_or_copy(src_dir: Path, dst_dir: Path):
    """
    Try to symlink dst_dir -> src_dir. If symlink not permitted (e.g., Windows without admin),
    fallback to copying files (mirror). If dst exists, skip.
    """
    if dst_dir.exists():
        return
    try:
        ensure_dir(dst_dir.parent)
        os.symlink(src_dir, dst_dir, target_is_directory=True)
        print(f"[OK] Created symlink: {dst_dir} -> {src_dir}")
    except Exception as e:
        print(f"[WARN] Symlink failed ({e}). Fallback to copying files...")
        ensure_dir(dst_dir)
        for txt in src_dir.glob("*.txt"):
            shutil.copy2(txt, dst_dir / txt.name)
        print(f"[OK] Copied {len(list(dst_dir.glob('*.txt')))} txt files to {dst_dir}")

def load_depth_model(depth_repo: str, weights: str, device: str, precision: str):
    add_depth_repo_to_path(depth_repo)
    DepthAnythingV2 = try_import_depth_anything()
    model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    ckpt = torch.load(weights, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    if precision == 'fp16':
        model.half()
    return model

def infer_disparity_01(model, rgb_512: np.ndarray, device: str, precision: str) -> np.ndarray:
    """
    Runs depth model and returns [0,1] disparity-like map (min-max normalized).
    rgb_512: HxWx3, uint8 RGB (512x512)
    """
    with torch.no_grad():
        if precision == 'fp16':
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model.infer_image(rgb_512)
        else:
            out = model.infer_image(rgb_512)
    disp01 = minmax01(out)
    return disp01

def main():
    ap = argparse.ArgumentParser("COCO train2017 → crop512 + DA-V2 disparity[0,1]")
    ap.add_argument("--image_folder", required=True, help="Path to data\\train2017")
    ap.add_argument("--out_root", default=None,
                    help="Where to write processed_images/ depthmaps/ (default: same as image_folder)")
    ap.add_argument("--prompts_dir", default=None,
                    help="Existing prompts dir (e.g., data\\train2017_text_prompts)")
    ap.add_argument("--connect_prompts", action="store_true",
                    help="Create a symlink (or copy if symlink fails) at image_folder\\text_prompts -> prompts_dir")
    ap.add_argument("--target_size", type=int, default=512, help="Crop size")
    ap.add_argument("--recursive", action="store_true", help="Recursively search images")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N images (smoke test)")
    # Depth-Anything related
    ap.add_argument("--depth_repo", default=None,
                    help="Path to Depth-Anything-V2 repo root if not importable")
    ap.add_argument("--depth_weights", default="checkpoints\\depth_anything_v2_vitl.pth",
                    help="Depth-Anything-V2 ViT-L checkpoint")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--save_depth_vis", action="store_true",
                    help="Also save pseudo-color depth png for quick inspection")
    args = ap.parse_args()

    img_root = Path(args.image_folder).resolve()
    if not img_root.exists():
        raise FileNotFoundError(f"image_folder not found: {img_root}")

    out_root = Path(args.out_root).resolve() if args.out_root else img_root
    processed_root = out_root / "processed_images"
    depth_root     = out_root / "depthmaps"
    txt_root       = img_root / "text_prompts"

    ensure_dir(processed_root)
    ensure_dir(depth_root)

    # Connect or validate text prompts
    if args.prompts_dir and args.connect_prompts:
        src_prompts = Path(args.prompts_dir).resolve()
        if not src_prompts.exists():
            raise FileNotFoundError(f"prompts_dir not found: {src_prompts}")
        safe_symlink_or_copy(src_prompts, txt_root)
    else:
        if not txt_root.exists():
            print(f"[WARN] {txt_root} does not exist. "
                  f"If you already have prompts in data\\train2017_text_prompts, "
                  f"use --prompts_dir and --connect_prompts to link/copy them here.")

    # Collect images
    if args.recursive:
        files = [p for p in img_root.rglob("*") if p.is_file() and is_image_file(p)]
    else:
        files = [p for p in img_root.iterdir() if p.is_file() and is_image_file(p)]
    files = sorted(files)
    if args.limit > 0:
        files = files[:args.limit]
    if len(files) == 0:
        print("[WARN] No images found. Check your --image_folder and extensions.")
        return

    # Load depth model
    model = load_depth_model(args.depth_repo, args.depth_weights, args.device, args.precision)

    # Process
    n_processed = 0
    n_skipped_img = 0
    n_skipped_dep = 0
    pbar = tqdm(files, desc="Preprocess (crop512 + depth)")

    for img_path in pbar:
        stem = img_path.stem
        ext  = img_path.suffix.lower()

        out_img_path = processed_root / f"{stem}{ext}"
        out_npy_path = depth_root / f"{stem}.npy"
        out_vis_path = depth_root / f"{stem}.png"

        # 1) crop 512x512
        do_img = args.overwrite or (not out_img_path.exists())
        if do_img:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[WARN] Failed to read {img_path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb512 = resize_and_center_crop_rgb(rgb, target=args.target_size)
            cv2.imwrite(str(out_img_path), cv2.cvtColor(rgb512, cv2.COLOR_RGB2BGR))
        else:
            n_skipped_img += 1
            rgb512 = cv2.cvtColor(cv2.imread(str(out_img_path)), cv2.COLOR_BGR2RGB)

        # 2) depth → disparity[0,1]
        do_dep = args.overwrite or (not out_npy_path.exists())
        if do_dep:
            disp01 = infer_disparity_01(model, rgb512, args.device, args.precision)
            np.save(str(out_npy_path), disp01.astype(np.float32))
            if args.save_depth_vis:
                vis = (disp01 * 255.0).clip(0, 255).astype(np.uint8)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
                cv2.imwrite(str(out_vis_path), vis)
        else:
            n_skipped_dep += 1

        n_processed += 1

    print("\n=== Summary ===")
    print(f"Images scanned     : {len(files)}")
    print(f"Processed (img+dep): {n_processed}")
    print(f"Skipped images     : {n_skipped_img} (already existed)")
    print(f"Skipped depths     : {n_skipped_dep} (already existed)")
    if txt_root.exists():
        n_txt = len(list(txt_root.glob('*.txt')))
        print(f"Text prompts found : {n_txt} under {txt_root}")
    else:
        print("Text prompts dir   : MISSING (see warning above)")

if __name__ == "__main__":
    main()
