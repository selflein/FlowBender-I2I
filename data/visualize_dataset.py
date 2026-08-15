"""Visualize (image, condition, caption) triplets from the dataset.

Optionally generates images from the pretrained ControlNet conditioned on the
depth map + caption, so each column shows: source image | condition | generated.

Usage (triplets only):
    python -m data.visualize_dataset \
        --image-dir data/test \
        --condition-dir depths_moge/test \
        --prompt-dir prompts_sd3-long-captioner-v2/test \
        --num-samples 8 --out grid.png

Usage (with ControlNet generation):
    python -m data.visualize_dataset \
        --image-dir data/test \
        --condition-dir depths_moge/test \
        --prompt-dir prompts_sd3-long-captioner-v2/test \
        --controlnet stabilityai/stable-diffusion-3.5-large-controlnet-depth \
        --base-model stabilityai/stable-diffusion-3.5-large \
        --num-samples 4 --out grid_with_gen.png
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path
from typing import Literal

import cyclopts
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from forward_models import FORWARD_MODELS

VAE_PATCH_SIZE = 64

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

app = cyclopts.App(help="Visualize dataset triplets (image / condition / caption).")

from dataset import collect_samples


def condition_to_displayable(t: torch.Tensor) -> np.ndarray:
    """Normalize a condition tensor to [0, 1] for display."""
    arr = t.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    return arr


def load_pipeline(controlnet: str, base_model: str, device: str):
    """Build a StableDiffusion3ControlNetPipeline for generation."""
    from diffusers import SD3ControlNetModel, StableDiffusion3ControlNetPipeline

    print(f"Loading ControlNet from {controlnet} ...")
    cn = SD3ControlNetModel.from_pretrained(controlnet, torch_dtype=torch.float16)

    print(f"Loading base model from {base_model} ...")
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(base_model, controlnet=cn, torch_dtype=torch.float16)
    pipe.to(device)
    return pipe


def _center_crop_to_aspect(pil: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center crop *pil* so its aspect ratio matches target_w / target_h."""
    w, h = pil.size
    target_aspect = target_w / target_h
    current_aspect = w / h

    if current_aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        pil = pil.crop((left, 0, left + new_w, h))
    elif current_aspect < target_aspect:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        pil = pil.crop((0, top, w, top + new_h))

    return pil


def raw_condition_to_pil(cond_tensor: torch.Tensor, height: int, width: int) -> Image.Image:
    """Convert a raw condition tensor to a 3-channel PIL image for the pipeline.

    Crops/resizes first so the min/max rescaling reflects the visible region,
    then normalizes to [0, 255] uint8 and tiles single-channel to RGB.
    """
    arr = cond_tensor.cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)

    # Crop and resize in float space before rescaling so that normalization
    # is computed over the final visible region only.
    pil_f = Image.fromarray(arr)
    if pil_f.size != (width, height):
        pil_f = _center_crop_to_aspect(pil_f, width, height)
        pil_f = pil_f.resize((width, height), Image.BILINEAR)
    arr = np.array(pil_f)

    lo, hi = arr.min(), arr.max()
    if hi - lo > 1e-8:
        arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0, 1)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)

    return Image.fromarray((arr * 255).astype(np.uint8))


def round_to_nearest(x: float, step: float) -> float:
    """Round *x* to the nearest multiple of *step*."""
    return int(round(x / step) * step)


def get_height_width(aspect_ratio: float, resolution: int) -> tuple[int, int]:
    if aspect_ratio < 1:  # Aspect ratio is w / h, so this is portrait orientation
        return resolution, round_to_nearest(resolution * aspect_ratio, VAE_PATCH_SIZE)
    elif aspect_ratio > 1:  # Aspect ratio is h / w, so this is landscape orientation
        return round_to_nearest(resolution / aspect_ratio, VAE_PATCH_SIZE), resolution
    else:
        return resolution, resolution


@app.default
def main(
    *,
    image_dir: str,
    condition_dir: str,
    prompt_dir: str,
    forward_model: Literal["moge2"] = "moge2",
    controlnet: str | None = None,
    base_model: str = "stabilityai/stable-diffusion-3.5-large",
    num_inference_steps: int = 60,
    guidance_scale: float = 5.0,
    controlnet_conditioning_scale: float = 0.7,
    resolution: int = 1024,
    aspect_ratio: float | None = None,
    device: str = "cuda",
    num_samples: int = 8,
    cols: int = 4,
    seed: int | None = None,
    out: str | None = None,
) -> None:
    """Visualize dataset triplets with optional ControlNet generation.

    Args:
        image_dir: Directory containing the source images.
        condition_dir: Directory containing the condition maps (e.g. depth
            PNGs).
        prompt_dir: Directory containing the caption .txt files.
        forward_model: Forward model whose condition_loader / preprocess to
            use.
        controlnet: HF id or local path for the SD3 ControlNet. If
            provided, generates images and adds them to the grid.
        base_model: Base SD3 model id (used when --controlnet is set).
        num_inference_steps: Diffusion sampling steps.
        guidance_scale: Classifier-free guidance scale.
        controlnet_conditioning_scale: ControlNet conditioning strength.
        resolution: Resolution for generated images. SD3.5 was trained at
            1024.
        device: Torch device for generation.
        num_samples: Number of triplets to display.
        cols: Number of columns in the grid.
        seed: Random seed for sample selection and generation.
        out: Save the figure to this path instead of displaying it.
    """

    image_dir_path = Path(image_dir)
    condition_dir_path = Path(condition_dir)
    prompt_dir_path = Path(prompt_dir)

    fm_cls = FORWARD_MODELS[forward_model]
    triplets = collect_triplets(image_dir_path, condition_dir_path, prompt_dir_path)

    if not triplets:
        print(
            f"No matching triplets found.\n"
            f"  images:     {image_dir_path}\n"
            f"  conditions: {condition_dir_path}\n"
            f"  prompts:    {prompt_dir_path}"
        )
        return

    n = min(num_samples, len(triplets))

    if seed is not None:
        random.seed(seed)
        selected = random.sample(triplets, n)
    else:
        selected = triplets[:n]

    do_generate = controlnet is not None
    pipe = load_pipeline(controlnet, base_model, device) if do_generate else None
    generator = torch.Generator(device=device).manual_seed(seed) if do_generate and seed is not None else None

    rows_per_sample = 3 if do_generate else 2
    grid_cols = min(cols, n)
    rows = ((n + grid_cols - 1) // grid_cols) * rows_per_sample

    grid_rows = (n + grid_cols - 1) // grid_cols
    fig, axes = plt.subplots(
        rows, grid_cols, figsize=(4 * grid_cols, (3.5 * rows_per_sample + 1.5) * grid_rows), squeeze=False
    )
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for pos, (img_path, cond_path, prompt_path) in enumerate(selected):
        col = pos % grid_cols
        row_base = (pos // grid_cols) * rows_per_sample

        img = np.array(Image.open(img_path).convert("RGB"))
        cond_tensor = fm_cls.load_condition(cond_path)
        cond_np = condition_to_displayable(cond_tensor)

        aspect_ratio_to_generate = aspect_ratio if aspect_ratio else cond_np.shape[1] / cond_np.shape[0]
        height, width = get_height_width(aspect_ratio_to_generate, resolution)
        print(f"Generating at height={height}, width={width}")

        caption = prompt_path.read_text().strip()

        ax_img = axes[row_base, col]
        ax_img.imshow(img)
        wrapped = textwrap.fill(caption, width=35)
        ax_img.set_title(wrapped, fontsize=8, pad=4)

        ax_cond = axes[row_base + 1, col]
        if cond_np.ndim == 2:
            ax_cond.imshow(cond_np, cmap="inferno")
        else:
            ax_cond.imshow(cond_np)
        ax_cond.set_title("condition", fontsize=8, pad=4)

        if do_generate:
            cond_pil = raw_condition_to_pil(cond_tensor, height, width)
            gen_img = pipe(
                prompt=caption,
                control_image=cond_pil,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                generator=generator,
            ).images[0]
            ax_gen = axes[row_base + 2, col]
            ax_gen.imshow(np.array(gen_img))
            ax_gen.set_title("generated", fontsize=8, pad=4)

        print(f"[{pos + 1}/{n}] {img_path.name}")

    fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    app()
