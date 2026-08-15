"""Batch center-crop and resize images to 1024x1024 for SD3.5 ControlNet training.

Saves processed images to an output folder mirroring the input directory structure:
    raw_images/train/abc.jpg  ->  images/train/abc.png

Usage:
    python batch_image.py --image-dir raw_images/ --output-dir images/
    python batch_image.py --image-dir raw_images/ --output-dir images/ --resolution 1024
"""

from pathlib import Path

import cyclopts
from PIL import Image
from PIL.Image import Resampling

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

app = cyclopts.App(help="Batch center-crop and resize images for SD3.5 ControlNet.")


def output_path_for(image_path: Path, image_root: Path, output_root: Path) -> Path:
    """Map an image path to its corresponding output path with .png extension."""
    return output_root / image_path.relative_to(image_root).with_suffix(".png")


def _is_valid_image(path: Path) -> bool:
    """Return True if PIL can open and verify the file."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def collect_image_paths(image_dir: Path, output_dir: Path) -> list[Path]:
    """Gather image paths whose output is missing or corrupt."""
    paths = sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not _is_valid_image(output_path_for(p, image_dir, output_dir))
    )
    return paths


def center_crop_and_resize(img: Image.Image, resolution: int) -> Image.Image:
    """Center-crop to square then resize to the target resolution."""
    w, h = img.size
    crop_size = min(w, h)
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    img = img.resize((resolution, resolution), resample=Resampling.LANCZOS)
    return img


@app.default
def main(*, image_dir: str, output_dir: str, resolution: int = 1024) -> None:
    """Run batch center-crop and resize.

    Args:
        image_dir: Root directory of source images.
        output_dir: Output directory for processed images (mirrors image_dir
            structure).
        resolution: Target square resolution in pixels.
    """
    image_root = Path(image_dir)
    output_root = Path(output_dir)

    image_paths = collect_image_paths(image_root, output_root)
    total = len(image_paths)
    print(f"Found {total} images to process in {image_dir}")

    for i, img_path in enumerate(image_paths):
        img = Image.open(img_path).convert("RGB")
        img = center_crop_and_resize(img, resolution)

        out_path = output_path_for(img_path, image_root, output_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

        name = str(img_path.relative_to(image_root))
        print(f"[{i + 1}/{total}] {name} -> {resolution}x{resolution}")

    print("Done.")


if __name__ == "__main__":
    app()
