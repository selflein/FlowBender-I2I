"""Shared image-quality metrics (PSNR / SSIM / LPIPS / MAE)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

_LPIPS_CACHE: dict[str, nn.Module] = {}


def get_lpips(device: torch.device) -> nn.Module:
    """Lazily build and cache an AlexNet-LPIPS module per device.

    Inputs are expected in ``[-1, 1]`` (``normalize=False``).
    """
    key = str(device)
    if key not in _LPIPS_CACHE:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        module = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=False)
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
        _LPIPS_CACHE[key] = module.to(device)
    return _LPIPS_CACHE[key]


@torch.no_grad()
def compute_psnr_ssim_lpips(gen: torch.Tensor, gt: torch.Tensor) -> dict[str, list[float]]:
    """Per-sample PSNR, SSIM, LPIPS, MAE for full images in ``[0, 1]``.

    Args:
        gen: Generated images ``[B, 3, H, W]`` in ``[0, 1]``.
        gt: Ground-truth images ``[B, 3, H, W]`` in ``[0, 1]``.

    Returns:
        Dict with keys ``psnr``, ``ssim``, ``lpips``, ``mae``; each a length-``B``
        list.  ``mae`` is the spatial mean of ``|gen - gt|`` per sample (range
        ``[0, 1]``; lower is better).
    """
    gen = gen.float().clamp(0.0, 1.0)
    gt = gt.float().clamp(0.0, 1.0)

    psnr = peak_signal_noise_ratio(gen, gt, data_range=1.0, dim=(1, 2, 3), reduction="none")
    ssim = structural_similarity_index_measure(gen, gt, data_range=1.0, reduction="none")
    mae = (gen - gt).abs().mean(dim=(1, 2, 3))

    lpips_fn = get_lpips(gen.device)
    # LPIPS wants inputs in [-1, 1].
    gen_lp = gen * 2.0 - 1.0
    gt_lp = gt * 2.0 - 1.0
    lpips_vals = [float(lpips_fn(gen_lp[i : i + 1], gt_lp[i : i + 1]).item()) for i in range(gen.shape[0])]

    return {
        "psnr": [float(v) for v in psnr.tolist()],
        "ssim": [float(v) for v in ssim.tolist()],
        "lpips": lpips_vals,
        "mae": [float(v) for v in mae.tolist()],
    }
