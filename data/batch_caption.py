"""Batch caption images in a folder using Florence-2.

Saves captions to a separate folder mirroring the image directory structure:
    images/train/abc.jpg  ->  prompts/train/abc.txt

Usage:
    python batch_caption.py --image-dir images/ --output-dir prompts/
    python batch_caption.py --image-dir images/ --output-dir prompts/ --batch-size 8
"""

import re
from pathlib import Path
from typing import Literal

import cyclopts
import torch
from PIL import Image
from transformers import AutoProcessor, Florence2ForConditionalGeneration

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
_IMAGE_PREFIX_RE = re.compile(
    r"^(the|this)\s+image\s+(shows|depicts|displays|features|presents|captures|is|are)\s+", re.IGNORECASE
)
MODEL_ID = "florence-community/Florence-2-large"
PROMPT = "<CAPTION>"

app = cyclopts.App(help="Batch caption images with Florence-2.")


def strip_image_prefix(caption: str) -> str:
    """Remove 'The/This image shows/depicts ...' prefix from a caption."""
    stripped = _IMAGE_PREFIX_RE.sub("", caption)
    if stripped and stripped[0].islower():
        stripped = stripped[0].upper() + stripped[1:]
    return stripped


def output_path_for(image_path: Path, image_root: Path, output_root: Path) -> Path:
    """Map an image path to its corresponding output path with .txt extension."""
    return output_root / image_path.relative_to(image_root).with_suffix(".txt")


def collect_image_paths(image_dir: Path, output_dir: Path) -> list[Path]:
    """Gather all image paths (recursively) that don't already have a caption."""
    paths = sorted(
        p
        for p in image_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not output_path_for(p, image_dir, output_dir).exists()
    )
    return paths


def load_images(paths: list[Path]) -> list[Image.Image]:
    images = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        images.append(img)
    return images


def caption_batch(
    model: Florence2ForConditionalGeneration,
    processor: AutoProcessor,
    images: list[Image.Image],
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int,
) -> list[str]:
    prompts = [PROMPT] * len(images)
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device, dtype)

    with torch.inference_mode():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=False,
        )

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    captions = []
    for text, img in zip(generated_texts, images):
        parsed = processor.post_process_generation(text, task=PROMPT, image_size=(img.width, img.height))
        # Remove and capitalize the image prefix
        # Example: "The image shows a cat sitting on a chair" -> "A cat sitting on a chair"
        captions.append(strip_image_prefix(parsed[PROMPT]).capitalize())
    return captions


@app.default
def main(
    *,
    image_dir: str,
    output_dir: str,
    batch_size: int = 4,
    max_new_tokens: int = 256,
    device: str | None = None,
    dtype: Literal["fp32", "fp16", "bf16"] = "bf16",
) -> None:
    """Run batch captioning.

    Args:
        image_dir: Root directory of images.
        output_dir: Output directory for captions (mirrors image_dir
            structure).
        batch_size: Batch size for inference.
        max_new_tokens: Max tokens to generate per caption.
        device: Device (cuda / cpu). Auto-detected if omitted.
        dtype: Model dtype.
    """
    torch_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    torch_dtype = dtype_map[dtype]

    image_root = Path(image_dir)
    output_root = Path(output_dir)

    print(f"Loading model {MODEL_ID} on {torch_device} ({dtype}) ...")
    model = (
        Florence2ForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch_dtype, attn_implementation="sdpa")
        .to(torch_device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    image_paths = collect_image_paths(image_root, output_root)
    total = len(image_paths)
    print(f"Found {total} images without captions in {image_dir}")

    captioned = 0
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = load_images(batch_paths)

        captions = caption_batch(model, processor, images, torch_device, torch_dtype, max_new_tokens)

        for path, caption in zip(batch_paths, captions):
            out_path = output_path_for(path, image_root, output_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(caption, encoding="utf-8")

        captioned += len(batch_paths)
        names = ", ".join(str(p.relative_to(image_root)) for p in batch_paths)
        print(f"[{captioned}/{total}] Captioned: {names}")

    print("Done.")


if __name__ == "__main__":
    app()
