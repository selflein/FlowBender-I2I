from typing import Literal

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvf
from transformers import AutoModelForDepthEstimation

# Depth-Anything-V2 (``DPTImageProcessor``): ImageNet normalisation, 518 px.
_DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
_DEPTH_INPUT_SIZE = 518
_DEPTH_IMAGE_MEAN = (0.485, 0.456, 0.406)
_DEPTH_IMAGE_STD = (0.229, 0.224, 0.225)


class DepthForwardProcess(nn.Module):
    """Forward process for depth conditioning.

    Uses Depth-Anything-V2 (matching the offline ``data/batch_depth.py``
    conditioning) to predict disparity (close=high).

    Args:
        alignment: How to align predicted vs. target depth when computing the
            residual ("minmax" or "affine_lstsq").
    """

    def __init__(self, alignment: Literal["minmax"] = "minmax"):
        super().__init__()
        print(f"DepthForwardProcess initialized with alignment={alignment} ({_DEPTH_MODEL_ID}@{_DEPTH_INPUT_SIZE})")
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(_DEPTH_MODEL_ID)
        self.register_buffer("image_mean", torch.tensor(_DEPTH_IMAGE_MEAN)[None, :, None, None], persistent=False)
        self.register_buffer("image_std", torch.tensor(_DEPTH_IMAGE_STD)[None, :, None, None], persistent=False)
        self.alignment = alignment
        self.input_size = _DEPTH_INPUT_SIZE

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        # VAE decoded output: [B, 3, H, W], range: [0, 1]
        H, W = img.shape[-2:]

        x = tvf.resize(img, size=self.input_size, interpolation=tvf.InterpolationMode.BILINEAR)
        x = (x - self.image_mean) / self.image_std

        outputs = self.depth_model(x)
        predicted_depth = outputs.predicted_depth  # [B, H_small, W_small]
        # DA-V2 outputs disparity (close=high); flip orientation so downstream
        # "close=0/far=1" minmax / affine alignment is directly meaningful.
        predicted_depth = 1 - predicted_depth

        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
        )
        return prediction

    def compute_condition(self, img: torch.Tensor) -> torch.Tensor:
        """Run the depth predictor and per-image min-max normalise to ``[0, 1]``.

        Mirrors the offline ``data/batch_depth.py`` pipeline so on-the-fly
        conditioning is interchangeable with the pre-computed PNGs (close=0,
        far=1) and matches the post-``invert`` convention of
        ``_load_depth_condition`` in ``sd3.dataset``.

        Args:
            img: [B, 3, H, W] in [0, 1].

        Returns:
            [B, 1, H, W] in [0, 1] (close=0, far=1).
        """
        depth = self.predict(img)
        d_min = torch.amin(depth, dim=(1, 2, 3), keepdim=True)
        d_max = torch.amax(depth, dim=(1, 2, 3), keepdim=True)
        return (depth - d_min) / (d_max - d_min + 1e-8)

    def get_target(self, cond: torch.Tensor) -> torch.Tensor:
        """Returns the target depth in the range [0, 1]."""
        target = cond * 0.5 + 0.5  # Range: [-1, 1] -> [0, 1]
        return target

    def get_residual(self, cur_img: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target = self.get_target(cond)  # (B, 3, H, W) in [0, 1]
        cur_depth = self.predict(cur_img)  # (B, 1, H, W) in [0, 1]

        if self.alignment == "minmax":
            min_depth = torch.amin(cur_depth, dim=(1, 2, 3), keepdim=True)
            max_depth = torch.amax(cur_depth, dim=(1, 2, 3), keepdim=True)
            cur_depth = (cur_depth - min_depth) / (max_depth - min_depth + 1e-8)
        return cur_depth - target, cur_depth
