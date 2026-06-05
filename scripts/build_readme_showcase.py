from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Pair:
    key: str
    image: Path
    depth: Path
    mtime: float


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def pair_outputs(folder: Path) -> list[Pair]:
    images = {p.name.removeprefix("image_"): p for p in folder.glob("image_*.png")}
    depths = {p.name.removeprefix("depth_"): p for p in folder.glob("depth_*.png")}
    pairs: list[Pair] = []
    for key in sorted(set(images) & set(depths)):
        image = images[key]
        depth = depths[key]
        pairs.append(Pair(key=key, image=image, depth=depth, mtime=max(image.stat().st_mtime, depth.stat().st_mtime)))
    return sorted(pairs, key=lambda item: item.mtime, reverse=True)


def fit_square(path: Path, size: int) -> Image.Image:
    with Image.open(path) as src:
        img = src.convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def rounded_panel(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], radius: int, fill: str, outline: str) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=2)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) // 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def make_frame(pair: Pair, index: int, total: int, mode: str) -> Image.Image:
    canvas_w, canvas_h = 1280, 720
    margin = 54
    image_size = 480
    top = 142
    gap = 54
    left_a = margin
    left_b = margin + image_size + gap

    frame = Image.new("RGB", (canvas_w, canvas_h), "#f8fafc")
    draw = ImageDraw.Draw(frame)
    title_font = load_font(34, bold=True)
    label_font = load_font(25, bold=True)
    small_font = load_font(20, bold=False)
    mono_font = load_font(18, bold=False)

    if mode == "depth_to_image":
        title = "Depth-conditioned Image Generation"
        first_label = "Input Depth"
        second_label = "Generated RGB"
        first = fit_square(pair.depth, image_size)
        second = fit_square(pair.image, image_size)
        accent = "#2563eb"
    else:
        title = "Joint RGB-Depth Generation"
        first_label = "Generated RGB"
        second_label = "Generated Depth"
        first = fit_square(pair.image, image_size)
        second = fit_square(pair.depth, image_size)
        accent = "#047857"

    draw.text((margin, 42), title, font=title_font, fill="#172033")
    draw.text((margin, 91), f"Example {index + 1:02d} / {total:02d}", font=small_font, fill="#687385")
    draw.text((360, 91), pair.key.replace(".png", ""), font=mono_font, fill="#687385")

    rounded_panel(draw, (38, 30, canvas_w - 38, canvas_h - 32), 18, "#ffffff", "#dbe3ef")
    draw.text((margin, 42), title, font=title_font, fill="#172033")
    draw.text((margin, 91), f"Example {index + 1:02d} / {total:02d}", font=small_font, fill="#687385")
    draw.text((360, 91), pair.key.replace(".png", ""), font=mono_font, fill="#687385")

    frame.paste(first, (left_a, top))
    frame.paste(second, (left_b, top))
    draw.rectangle((left_a, top, left_a + image_size, top + image_size), outline="#e2e8f0", width=2)
    draw.rectangle((left_b, top, left_b + image_size, top + image_size), outline="#e2e8f0", width=2)

    draw_centered_text(draw, first_label, (left_a, 104, left_a + image_size, 140), label_font, "#172033")
    draw_centered_text(draw, second_label, (left_b, 104, left_b + image_size, 140), label_font, "#172033")

    x = left_b + image_size + 35
    y = top + 176
    draw.rounded_rectangle((x, y, x + 86, y + 86), radius=43, fill="#ffffff", outline="#e2e8f0", width=2)
    draw.polygon([(x + 37, y + 28), (x + 37, y + 58), (x + 58, y + 43)], fill=accent)
    draw.rounded_rectangle((left_a + image_size - 120, y, left_a + image_size - 34, y + 86), radius=43, fill="#ffffff", outline="#e2e8f0", width=2)
    draw.polygon(
        [
            (left_a + image_size - 72, y + 28),
            (left_a + image_size - 72, y + 58),
            (left_a + image_size - 93, y + 43),
        ],
        fill=accent,
    )

    dots_y = top + image_size + 34
    dots_x = left_a + image_size + gap // 2 - (total * 24) // 2
    for i in range(total):
        dot_fill = accent if i == index else "#cbd5e1"
        draw.rounded_rectangle((dots_x + i * 24, dots_y, dots_x + i * 24 + 16, dots_y + 16), radius=8, fill=dot_fill)

    return frame


def save_showcase(pairs: list[Pair], out_gif: Path, out_png: Path, mode: str, max_frames: int) -> None:
    selected = pairs[:max_frames]
    if not selected:
        raise RuntimeError(f"No paired examples found for {mode}.")
    frames = [make_frame(pair, idx, len(selected), mode) for idx, pair in enumerate(selected)]
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_png, format="PNG", optimize=True)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=1700,
        loop=0,
        optimize=True,
    )


def select_pairs(pairs: list[Pair], keys: str | None, max_frames: int) -> list[Pair]:
    if not keys:
        return pairs[:max_frames]
    by_key = {pair.key: pair for pair in pairs}
    selected: list[Pair] = []
    missing: list[str] = []
    for raw_key in keys.split(","):
        key = raw_key.strip()
        if not key:
            continue
        if not key.endswith(".png"):
            key = f"{key}.png"
        pair = by_key.get(key)
        if pair is None:
            missing.append(key)
        else:
            selected.append(pair)
    if missing:
        raise RuntimeError(f"Missing paired examples: {', '.join(missing)}")
    return selected[:max_frames]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-to-image", type=Path, default=Path("outputs/depth_to_image"))
    parser.add_argument("--joint-generation", type=Path, default=Path("outputs/joint_generation"))
    parser.add_argument("--out-dir", type=Path, default=Path("assets/readme_showcase"))
    parser.add_argument("--max-frames", type=int, default=4)
    parser.add_argument(
        "--depth-keys",
        type=str,
        default="test20251117,test_digital_art_2,000000265518,20251004_212117",
        help="Comma-separated paired output keys for depth-to-image showcase.",
    )
    parser.add_argument(
        "--joint-keys",
        type=str,
        default="20260604_225206,20251109_012630,20251004_203121,20251024_180841",
        help="Comma-separated paired output keys for joint-generation showcase.",
    )
    args = parser.parse_args()

    d2i_pairs = pair_outputs(args.depth_to_image)
    joint_pairs = pair_outputs(args.joint_generation)
    selected_d2i = select_pairs(d2i_pairs, args.depth_keys, args.max_frames)
    selected_joint = select_pairs(joint_pairs, args.joint_keys, args.max_frames)
    save_showcase(
        selected_d2i,
        args.out_dir / "depth_to_image_carousel.gif",
        args.out_dir / "depth_to_image_preview.png",
        mode="depth_to_image",
        max_frames=args.max_frames,
    )
    save_showcase(
        selected_joint,
        args.out_dir / "joint_generation_carousel.gif",
        args.out_dir / "joint_generation_preview.png",
        mode="joint_generation",
        max_frames=args.max_frames,
    )
    print(f"Depth-to-image pairs: {len(d2i_pairs)}")
    print(f"Joint-generation pairs: {len(joint_pairs)}")
    print(f"Showcase assets written to: {args.out_dir}")


if __name__ == "__main__":
    main()
