"""HED edge forward process for ControlNet conditioning.

The HED predictor is the differentiable port from the official ControlNet
repository (Apache 2.0).  Compared to the original Saining Xie HED model it
takes RGB (not BGR) input, so it composes naturally with our pixel-space
``[0, 1]`` conventions.

Edge conditioning is *derived* from the GT image at train/eval time -- the
same model serves as both the offline edge detector (``compute_condition``)
and the online forward process used for residual / gradient feedback
(``predict`` / ``get_residual``).  No on-disk edge maps are needed.

Reference:
    https://github.com/lllyasviel/ControlNet/blob/main/annotator/hed/__init__.py
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download


class _DoubleConvBlock(nn.Module):
    """A stack of 3x3 convs followed by a 1x1 projection to a single channel.

    Mirrors the upstream ControlNetHED block exactly so the released
    ``ControlNetHED.pth`` state dict loads without remapping.
    """

    def __init__(self, input_channel: int, output_channel: int, layer_number: int):
        super().__init__()
        self.convs = nn.Sequential()
        self.convs.append(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1)
        )
        for _ in range(1, layer_number):
            self.convs.append(
                nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1, padding=1)
            )
        self.projection = nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, down_sampling: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        if down_sampling:
            h = F.max_pool2d(h, kernel_size=2, stride=2)
        for conv in self.convs:
            h = F.relu(conv(h))
        return h, self.projection(h)


class _ControlNetHED_Apache2(nn.Module):
    """Differentiable HED edge detector (Apache 2.0).

    Operates on RGB inputs in the ``[0, 255]`` range with a learned
    per-channel mean (``self.norm``); returns the five multi-scale side
    outputs (logits) used by the upstream HED loss.
    """

    def __init__(self):
        super().__init__()
        self.norm = nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = _DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = _DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = _DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = _DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = _DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        _, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5


def _load_hed_checkpoint() -> Path:
    """Download (if needed) and return the path to ``ControlNetHED.pth``.

    Uses ``huggingface_hub`` so the checkpoint lands in ``HF_HOME`` like
    every other model in this repo.
    """
    return Path(hf_hub_download(repo_id="lllyasviel/Annotators", filename="ControlNetHED.pth"))


class EdgeForwardProcess(nn.Module):
    """Forward process for HED-edge conditioning.

    ``predict`` runs the differentiable HED detector to produce a single-
    channel edge probability map in ``[0, 1]``.  The conditioning channel
    convention matches the depth task: a 3-channel ``[-1, 1]`` tensor where
    all channels carry the same edge map.

    The same model is used both to *derive* the conditioning from the GT
    image (``compute_condition``, called by the train / eval loops) and to
    compute residuals during steering (``get_residual``).
    """

    def __init__(self):
        super().__init__()
        ckpt_path = _load_hed_checkpoint()
        print(f"EdgeForwardProcess loading HED checkpoint from {ckpt_path}")
        self.hed = _ControlNetHED_Apache2()
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.hed.load_state_dict(state_dict)
        self.hed.eval()

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        """Run the HED edge detector.

        Args:
            img: ``[B, 3, H, W]`` in ``[0, 1]``.

        Returns:
            Single-channel edge probability map ``[B, 1, H, W]`` in ``[0, 1]``.
        """
        H, W = img.shape[-2:]
        x = img * 255.0  # HED was trained on RGB in [0, 255]
        projections = self.hed(x)
        upsampled = [F.interpolate(p, size=(H, W), mode="bilinear", align_corners=False) for p in projections]
        # Mean across the 5 scales then sigmoid to get an edge probability.
        edge_logits = torch.stack(upsampled, dim=0).mean(dim=0)
        return torch.sigmoid(edge_logits)

    @torch.no_grad()
    def compute_condition(self, img: torch.Tensor) -> torch.Tensor:
        """Derive the conditioning edge map from a GT image.

        Mirrors the offline preprocessing step the depth task uses, but
        runs on the fly: callers pass the GT image and get back a single-
        channel edge map ready to be tiled into the 3-channel ``[-1, 1]``
        condition tensor expected by the ControlNet.

        Args:
            img: ``[B, 3, H, W]`` in ``[0, 1]``.

        Returns:
            ``[B, 1, H, W]`` in ``[0, 1]``.
        """
        return self.predict(img)

    def get_target(self, cond: torch.Tensor) -> torch.Tensor:
        """Map the 3-channel ``[-1, 1]`` edge condition to a ``[0, 1]`` target.

        Args:
            cond: ``[B, 3, H, W]`` in ``[-1, 1]`` (channels are tiled copies
                of the same edge map).

        Returns:
            ``[B, 3, H, W]`` in ``[0, 1]``.
        """
        return cond * 0.5 + 0.5

    def get_residual(self, cur_img: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the edge residual between predicted and target edges.

        Returns ``(residual, prediction)`` where ``residual`` broadcasts to
        the 3-channel target shape (matches the depth task convention so
        the rest of the pipeline -- VAE-encode, ``_apply_residual_variant``,
        gradient losses -- works unchanged).

        Args:
            cur_img: Generated image estimate ``[B, 3, H, W]`` in ``[0, 1]``.
            cond: Conditioning edges ``[B, 3, H, W]`` in ``[-1, 1]``.

        Returns:
            Tuple of:
                - residual ``[B, 3, H, W]`` (broadcasted single-channel diff).
                - prediction ``[B, 1, H, W]``.
        """
        target = self.get_target(cond)  # [B, 3, H, W]
        cur_edge = self.predict(cur_img)  # [B, 1, H, W]
        return cur_edge - target, cur_edge
