"""Dataset for ControlNet training with pre-computed text embeddings.

Loads quadruplets from a directory structure produced by the preprocessing
pipeline (``data/preprocess.sh``).
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

logger = logging.getLogger(__name__)


class PreprocessedControlNetDataset(Dataset):
    """Dataset for ControlNet training with pre-computed text embeddings.

    Returns dicts with keys ``pixel_values``, ``conditioning_pixel_values``,
    ``prompt_embeds``, and ``pooled_prompt_embeds``.

    Args:
        img_dir: Directory containing training images.
        cond_dir: Directory containing conditioning images (e.g. depth maps).
        text_embeds_dir: Directory containing pre-computed ``.pt`` embedding
            files mirroring the prompt directory structure.
        prompt_dir: Optional directory containing ``.txt`` caption files.
            Only used if raw prompts are needed (e.g. for logging).
        resolution: Target spatial resolution (square center-crop).
    """

    def __init__(
        self,
        img_dir: str | Path,
        cond_dir: str | Path,
        text_embeds_dir: str | Path,
        prompt_dir: str | Path | None = None,
        resolution: int = 1024,
        condition_type: str = "depth",
        scale_factor: int = 4,
        jpeg_quality: int = 10,
    ) -> None:
        super().__init__()
        self.image_dir = Path(img_dir)
        self.cond_dir = Path(cond_dir)
        self.text_embeds_dir = Path(text_embeds_dir)
        self.prompt_dir = Path(prompt_dir) if prompt_dir is not None else None
        self.resolution = resolution
        self.condition_postprocess = None

        if condition_type == "depth":
            # Depth Anything V2 outputs inverted depth maps in the range [0, 1] (0=far, 1=close)
            self.condition_loader = lambda path: _load_depth_condition(path, invert=True)
            self.condition_ext = ".png"
        elif condition_type == "super_resolution":
            self.condition_loader = _load_rgb_condition
            self.condition_ext = ".png"
            self.condition_postprocess = _make_sr_degradation(scale_factor)
        elif condition_type == "jpeg_restoration":
            self.condition_loader = _load_rgb_condition
            self.condition_ext = ".png"
            self.condition_postprocess = _make_jpeg_degradation(jpeg_quality)
        elif condition_type == "edge":
            # The condition is derived on-the-fly from the GT image via
            # ``EdgeForwardProcess.compute_condition`` in the train/eval
            # loops; the dataset just loads the image itself as a 3-channel
            # ``[-1, 1]`` tensor placeholder.
            self.condition_loader = _load_rgb_condition
            self.condition_ext = ".png"
        else:
            raise ValueError(f"Unsupported condition type: {condition_type}")

        self.samples = _collect_samples(self.image_dir, self.cond_dir, self.text_embeds_dir, self.condition_ext)
        logger.info("Found %d samples", len(self.samples))

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.condition_resize = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_rel, cond_rel, embed_rel = self.samples[idx]

        image = Image.open(self.image_dir / img_rel).convert("RGB")
        pixel_values = self.image_transform(image)

        raw_condition = self.condition_loader(self.cond_dir / cond_rel)
        raw_condition = self.condition_resize(raw_condition)
        if self.condition_postprocess is not None:
            raw_condition = self.condition_postprocess(raw_condition)

        embeds = torch.load(self.text_embeds_dir / embed_rel, map_location="cpu", weights_only=True)
        prompt_embeds = embeds["prompt_embeds"]
        pooled_prompt_embeds = embeds["pooled_prompt_embeds"]
        item: dict = {
            "id": Path(img_rel).stem,
            "pixel_values": pixel_values,
            "conditioning_pixel_values": raw_condition,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

        if self.prompt_dir is not None:
            stem_rel = Path(img_rel).with_suffix(".txt")
            prompt_path = self.prompt_dir / stem_rel
            if prompt_path.exists():
                item["prompts"] = prompt_path.read_text().strip()

        return item


def collate_fn(batch: list[dict]) -> dict:
    """Stack tensors into a batch dict compatible with the SD3 training loop."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([b["conditioning_pixel_values"] for b in batch])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    result: dict = {
        "pixel_values": pixel_values,  # [B, 3, H, W], range [-1, 1]
        "conditioning_pixel_values": conditioning_pixel_values,  # [B, 3, H, W], range [-1, 1]
        "prompt_embeds": torch.stack([b["prompt_embeds"] for b in batch]),  # [B, seq_len, hidden_dim]
        "pooled_prompt_embeds": torch.stack([b["pooled_prompt_embeds"] for b in batch]),  # [B, pooled_dim]
        "id": [b["id"] for b in batch],  # list[str]
    }

    if "prompts" in batch[0]:
        result["prompts"] = [b["prompts"] for b in batch]  # list[str]

    return result


def _load_rgb_condition(path: Path) -> Tensor:
    """Load an RGB image as a float tensor [3, H, W] in [0, 1]."""
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def _make_sr_degradation(scale_factor: int):
    """Return a callable that degrades a normalised condition tensor.

    Applies avg-pool downsampling then a matching nearest-neighbour upsample so
    the stored conditioning is a single coherent operator's output at full
    spatial size. Expects and returns tensors in ``[-1, 1]``.

    Both kernels are delegated to ``_sr_downsample`` / ``_sr_upsample`` in the
    SR forward-process module so the dataset and the forward process always
    agree on ``A`` (single source of truth).
    """
    from sd3.super_resolution import _sr_downsample, _sr_upsample

    logger.info(f"Making SR degradation with scale_factor: {scale_factor}")

    def _degrade(cond: Tensor) -> Tensor:
        cond_01 = (cond * 0.5 + 0.5).unsqueeze(0)  # [-1, 1] → [0, 1], add batch dim
        lr = _sr_downsample(cond_01, scale_factor)
        upsampled = _sr_upsample(lr, target_size=cond.shape[-2:]).squeeze(0)
        return upsampled * 2.0 - 1.0  # [0, 1] → [-1, 1]

    return _degrade


def _make_jpeg_degradation(quality: int):
    """Return a callable that JPEG-encodes/decodes the condition at fixed quality.

    Runs in DataLoader workers (CPU). Expects and returns tensors in
    ``[-1, 1]`` so it composes with the existing ``Normalize([0.5], [0.5])``
    pipeline.  Uses ``torchvision.io`` for tensor-native JPEG encode/decode.
    """
    logger.info(f"Making JPEG degradation with quality: {quality}")
    from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg

    def _degrade(cond: Tensor) -> Tensor:
        cond_01 = cond * 0.5 + 0.5  # [-1, 1] → [0, 1]
        cond_uint8 = (cond_01.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        encoded = encode_jpeg(cond_uint8, quality=quality)
        decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB).float() / 255.0
        return decoded * 2.0 - 1.0  # [0, 1] → [-1, 1]

    return _degrade


def _load_depth_condition(path: Path, invert: bool = False) -> Tensor:
    depth = np.array(Image.open(path), dtype=np.float32) / 65535.0
    if invert:
        depth = 1 - depth
    return torch.from_numpy(depth).unsqueeze(0).repeat(3, 1, 1)


def _collect_samples(
    image_dir: Path, cond_dir: Path, text_embeds_dir: Path, condition_ext: str
) -> list[tuple[str, str, str]]:
    """Return sorted list of (image, condition, text_embed) relative paths."""

    # Create a unique cache file name based on the directory paths
    key = "_".join([p.resolve().absolute().as_posix() for p in [image_dir, cond_dir, text_embeds_dir]])
    cache_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_path = image_dir.parent / f"{cache_hash}_samples.npy"
    if cache_path.exists():
        logger.info("Loading cached sample list from %s", cache_path)
        return list(np.load(cache_path, allow_pickle=True))

    image_paths = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)

    samples: list[tuple[str, str, str]] = []
    for img_path in image_paths:
        rel = img_path.relative_to(image_dir)
        stem_rel = rel.with_suffix("")

        cond_path = cond_dir / stem_rel.with_suffix(condition_ext)
        if not cond_path.exists():
            continue

        embed_path = text_embeds_dir / stem_rel.with_suffix(".pt")
        if not embed_path.exists():
            continue

        samples.append((str(rel), str(cond_path.relative_to(cond_dir)), str(embed_path.relative_to(text_embeds_dir))))

    logger.info(
        "Collected %d samples (%d images skipped due to missing cond/embeds)",
        len(samples),
        len(image_paths) - len(samples),
    )
    np.save(cache_path, samples)
    return samples
