"""Super-resolution forward process for FlowChef steering.

Degradation operator: average-pooling downsampling (matching FlowChef's
``SuperResolutionOperator``). The down/up kernels are kept consistent so the
rendered cond and ``predict`` outputs are a single coherent operator: avg-pool
down + nearest-neighbour up (block-constant).

Reference: https://github.com/FlowChef/FlowChef/blob/9b705fcd91cb/src/inverse_operators.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sr_downsample(img: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Downsample an image to ``H/s, W/s`` with average pooling.

    Single source of truth for the SR forward operator. Used both by the
    on-the-fly degradation in ``sd3/dataset.py`` and by
    ``SuperResolutionForwardProcess`` so the dataset condition and the
    metric/feedback path always agree on ``A``.

    Args:
        img: ``[B, C, H, W]`` (or ``[C, H, W]`` for the dataset path).
        scale_factor: Integer downsample factor ``s``.

    Returns:
        Downsampled tensor at ``H/s, W/s``.
    """
    return F.avg_pool2d(img, scale_factor)


def _sr_upsample(lr: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Nearest-neighbour upsample an LR image back to ``target_size``.

    Pairs with :func:`_sr_downsample` so the round-trip is a single coherent
    operator: block-constant rendering matching what the trained ControlNet
    has seen.

    Args:
        lr: Low-resolution tensor ``[B, C, H', W']``.
        target_size: Output spatial size ``(H, W)``.

    Returns:
        Tensor at ``target_size``.
    """
    return F.interpolate(lr, size=target_size, mode="nearest")


class SuperResolutionForwardProcess(nn.Module):
    """Forward process that downsamples images by a fixed integer factor.

    The degradation operator ``A`` is ``F.avg_pool2d(x, s)`` (matching the
    FlowChef baseline used at training time). During FlowChef steering the
    residual ``A(x_hat) - y`` drives the latent optimisation towards an image
    whose low-frequency content matches the observed low-resolution input.
    """

    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        print(f"SuperResolutionForwardProcess initialized with scale_factor: {scale_factor}")

    def downsample(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the configured downsample operator at the LR resolution.

        Args:
            img: ``[B, C, H, W]``.

        Returns:
            ``[B, C, H/s, W/s]`` downsampled tensor.
        """
        return _sr_downsample(img, self.scale_factor)

    def predict(self, img: torch.Tensor) -> torch.Tensor:
        """Downsample then nearest-neighbour upsample back to the input size.

        Produces a block-constant image (the training-time rendering); the
        down/up pair is a single coherent operator.

        Args:
            img: [B, 3, H, W] in [0, 1].

        Returns:
            [B, 3, H, W] in float32, at the original spatial size.
        """
        return _sr_upsample(self.downsample(img), img.shape[-2:])

    def get_target(self, cond: torch.Tensor) -> torch.Tensor:
        """Returns the target image in the range [0, 1]. It is already the low-resolution version.

        Args:
            cond: [B, 3, H, W] in [-1, 1].

        Returns:
            [B, 3, H, W] in [0, 1], float32.
        """
        return cond * 0.5 + 0.5

    def get_residual(self, cur_img: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the residual between the degraded estimate and the observation.

        Both sides are at the full input resolution (block-constant after the
        down/up round-trip), so the residual is also full resolution.

        Args:
            cur_img: Generated image estimate [B, 3, H, W] in [0, 1].
            cond: Conditioning tensor [B, 3, H', W'] in [-1, 1].

        Returns:
            Tuple of (residual, prediction), both [B, 3, H, W] float32.
        """
        target = self.get_target(cond)
        pred = self.predict(cur_img)
        return pred - target, pred
