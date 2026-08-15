"""Batch depth estimation using Depth Anything V2 for an image folder.

Saves 16-bit grayscale PNG depth maps to a separate folder mirroring the image
directory structure:
    images/train/abc.jpg  ->  depths/train/abc.png

The depth maps are in [0, 65535] range (16-bit) for maximum precision, suitable as
conditioning inputs for SD3 ControlNet depth.

Usage:
    python batch_depth.py --image-dir images/ --output-dir depths/
    python batch_depth.py --image-dir images/ --output-dir depths/ --model depth-anything/Depth-Anything-V2-Large-hf
"""

from pathlib import Path

import cyclopts
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

app = cyclopts.App(help="Batch depth estimation with Depth Anything V2.")


def output_path_for(image_path: Path, image_root: Path, output_root: Path) -> Path:
    """Map an image path to its corresponding depth output path with .png extension."""
    return output_root / image_path.relative_to(image_root).with_suffix(".png")


def collect_image_paths(image_dir: Path, output_dir: Path) -> list[Path]:
    """Gather all image paths (recursively) that don't already have a depth map."""
    paths = sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not output_path_for(p, image_dir, output_dir).exists()
    )
    return paths


def load_image(path: Path) -> tuple[Image.Image, tuple[int, int]]:
    """Load an image as a PIL RGB image and return it with its original size."""
    img = Image.open(path).convert("RGB")
    orig_size = img.size  # (w, h)
    return img, orig_size


def save_depth(depth_np: np.ndarray, out_path: Path, orig_size: tuple[int, int]):
    """Save a [0, 1] float depth map as 16-bit grayscale PNG at original resolution."""
    depth_uint16 = (np.clip(depth_np, 0, 1) * 65535).astype(np.uint16)
    depth_img = Image.fromarray(depth_uint16, mode="I;16")
    if depth_img.size != orig_size:
        depth_img = depth_img.resize(orig_size, resample=Resampling.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    depth_img.save(out_path)


@app.default
def main(*, image_dir: str, output_dir: str, model: str = "depth-anything/Depth-Anything-V2-Large-hf") -> None:
    """Run batch depth estimation.

    Args:
        image_dir: Root directory of images.
        output_dir: Output directory for depth maps (mirrors image_dir structure).
        model: HuggingFace model id for Depth Anything V2.
    """
    torch_device = torch.device("cuda")
    dtype = torch.bfloat16

    image_root = Path(image_dir)
    output_root = Path(output_dir)

    print(f"Loading Depth Anything V2 from {model} ...")
    depth_processor = AutoImageProcessor.from_pretrained(model)
    depth_model = AutoModelForDepthEstimation.from_pretrained(model, torch_dtype=dtype).to(torch_device)
    depth_model.eval()

    image_paths = collect_image_paths(image_root, output_root)
    total = len(image_paths)
    print(f"Found {total} images without depth maps in {image_dir}")

    for i, img_path in enumerate(image_paths):
        img, orig_size = load_image(img_path)

        inputs = depth_processor(images=img, return_tensors="pt").to(torch_device, dtype)

        with torch.inference_mode():
            outputs = depth_model(**inputs)
            predicted_depth = outputs.predicted_depth  # (1, H_small, W_small)

        depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=orig_size[::-1],  # (H, W)
            mode="bicubic",
            align_corners=False,
        )
        depth_np = depth[0, 0].cpu().float().numpy()

        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 0:
            depth_np = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_np = np.zeros_like(depth_np)

        out_path = output_path_for(img_path, image_root, output_root)
        save_depth(depth_np, out_path, orig_size)

        name = str(img_path.relative_to(image_root))
        print(f"[{i + 1}/{total}] Depth: {name}")

    print("Done.")


if __name__ == "__main__":
    app()
