"""JPEG restoration forward process.

Degradation operator: a JPEG encode/decode round-trip at a fixed quality
factor, implemented via ``torchvision.io.encode_jpeg`` / ``decode_jpeg``.

JPEG quantisation is non-differentiable, so this forward process supports
the residual feedback path only -- there is no useful gradient for the
gradient/combined modes.
"""

import torch
import torch.nn as nn
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg


class JpegRestorationForwardProcess(nn.Module):
    """Forward process that JPEG-encodes/decodes images at a fixed quality.

    The degradation operator ``A`` is a libjpeg encode/decode round-trip at
    the configured ``quality`` factor.  During steering the residual
    ``A(x_hat) - y`` drives the latent optimisation towards an image whose
    JPEG-degraded version matches the observed compressed input.

    Args:
        quality: JPEG quality factor in ``[1, 100]`` (lower is more
            compression).  Default ``10`` matches the heavy-compression
            restoration benchmark.
    """

    def __init__(self, quality: int = 10):
        super().__init__()
        self.quality = int(quality)
        print(f"JpegRestorationForwardProcess initialized with quality: {self.quality}")

    @torch.no_grad()
    def predict(self, img: torch.Tensor) -> torch.Tensor:
        """JPEG-encode then decode each image in the batch.

        Iterates per-sample because torchvision's encode/decode operate on
        single ``[C, H, W]`` tensors only.  The JPEG codec runs on CPU; the
        result is cast back to the input device and dtype so callers never
        observe the round-trip.

        Args:
            img: ``[B, 3, H, W]`` in ``[0, 1]``.

        Returns:
            ``[B, 3, H, W]`` in ``[0, 1]`` matching the input device/dtype.
        """
        in_dtype = img.dtype
        in_device = img.device
        img_uint8 = (img.detach().float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
        out = torch.empty_like(img_uint8)
        for i in range(img_uint8.shape[0]):
            encoded = encode_jpeg(img_uint8[i], quality=self.quality)
            out[i] = decode_jpeg(encoded, mode=ImageReadMode.RGB)
        return out.to(device=in_device, dtype=in_dtype) / 255.0

    def get_target(self, cond: torch.Tensor) -> torch.Tensor:
        """Map conditioning from ``[-1, 1]`` to ``[0, 1]``.

        Args:
            cond: ``[B, 3, H, W]`` in ``[-1, 1]``.

        Returns:
            ``[B, 3, H, W]`` in ``[0, 1]``, float32.
        """
        return cond.float() * 0.5 + 0.5

    def get_residual(self, cur_img: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pixel residual between the JPEG-degraded estimate and the observation.

        Args:
            cur_img: Generated image estimate ``[B, 3, H, W]`` in ``[0, 1]``.
            cond: JPEG-degraded conditioning ``[B, 3, H, W]`` in ``[-1, 1]``.

        Returns:
            Tuple of ``(residual, prediction)``, both ``[B, 3, H, W]``, where
            ``residual = predict(cur_img) - get_target(cond)``.
        """
        target = self.get_target(cond).to(cur_img.dtype)
        pred = self.predict(cur_img)
        return pred - target, pred
