"""Depth alignment and metrics used for evaluation.

Predicted depth is aligned to ground truth via per-image min-max
normalisation (close=0, far=1) before computing delta1 / MAE -- this is the
alignment used to report depth metrics in the paper.
"""

import torch


def scale_min_max(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    min_depth = torch.amin(t, dim=(1,), keepdim=True)
    max_depth = torch.amax(t, dim=(1,), keepdim=True)
    return (t - min_depth) / (max_depth - min_depth).clamp(min=eps)


def delta1_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample fraction of pixels with ``max(gt/pred, pred/gt) < 1.25``. Inputs ``[B, N]``."""
    return (torch.maximum(gt / pred, pred / gt) < 1.25).float().mean(dim=-1)


def mae_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample mean absolute depth error. Inputs ``[B, N]``."""
    return torch.abs(pred - gt).mean(dim=-1)


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    """Per-sample depth metrics under min-max alignment.

    Returns lists of length ``B`` per metric key: ``delta1``, ``mae``.
    """
    B = pred.shape[0]
    pred_orig_shape = pred.shape  # [B, H, W]
    pred = pred.view(B, -1)
    gt = gt.view(B, -1)

    pred_minmax = scale_min_max(pred)
    gt_minmax = scale_min_max(gt)

    metrics: dict[str, list[float]] = {
        "delta1": [float(v) for v in delta1_depth(pred_minmax, gt_minmax).tolist()],
        "mae": [float(v) for v in mae_depth(pred_minmax, gt_minmax).tolist()],
    }
    preds = {"pred": pred_minmax.view(pred_orig_shape), "gt": gt_minmax.view(pred_orig_shape)}

    return preds, metrics
