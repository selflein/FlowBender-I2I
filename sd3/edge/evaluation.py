"""Edge-specific evaluation: metric wrapper and debug visualization."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sd3.image_metrics import compute_psnr_ssim_lpips


@torch.no_grad()
def _compute_edge_metrics(
    gen_tensors: torch.Tensor,
    cond_tensors: torch.Tensor,
    forward_process: torch.nn.Module,
    gt_tensors: torch.Tensor | None = None,
    **kwargs,
) -> tuple[dict, dict[str, list[float]]]:
    """Compute edge-fidelity metrics plus image quality vs GT.

    Reports:

    * ``edge_mae`` / ``edge_mse`` -- mean absolute / squared error between
      the HED-predicted edge map of the generated image and the
      conditioning edge map (both in ``[0, 1]``).
    * ``psnr`` / ``ssim`` / ``lpips`` -- generated image vs GT (perceptual
      quality, mirrors the SR / JPEG-restoration tasks).

    Args:
        gen_tensors: Generated images ``[B, 3, H, W]`` in ``[0, 1]``.
        cond_tensors: Conditioning edges ``[B, 3, H, W]`` in ``[-1, 1]``.
        forward_process: ``EdgeForwardProcess`` instance.
        gt_tensors: Ground-truth images ``[B, 3, H, W]`` in ``[0, 1]``.

    Returns:
        Tuple of ``(aligned dict for debug viz, metrics dict)``.
    """
    target = forward_process.get_target(cond_tensors)[:, :1]  # [B, 1, H, W] in [0, 1]
    pred = forward_process.predict(gen_tensors)  # [B, 1, H, W] in [0, 1]

    diff = pred - target
    abs_diff = diff.abs()
    edge_mae = abs_diff.mean(dim=(1, 2, 3))
    edge_mse = diff.pow(2).mean(dim=(1, 2, 3))

    metrics: dict[str, list[float]] = {
        "edge_mae": [float(v) for v in edge_mae.tolist()],
        "edge_mse": [float(v) for v in edge_mse.tolist()],
    }
    if gt_tensors is not None:
        metrics.update(compute_psnr_ssim_lpips(gen_tensors, gt_tensors))

    aligned = {"pred_edge": pred, "target_edge": target, "gen": gen_tensors, "gt": gt_tensors}
    return aligned, metrics


def _save_edge_comparison(
    gt_rgb: np.ndarray,
    gen_rgb: np.ndarray,
    target_edge: np.ndarray,
    pred_edge: np.ndarray,
    save_path: str | Path,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a 1-row debug plot for HED-edge evaluation.

    Panels: GT image, generated image, target edges (HED of GT),
    predicted edges (HED of generated), edge residual (|pred - target|).

    Args:
        gt_rgb: Ground-truth RGB image ``[H, W, 3]`` in ``[0, 1]``.
        gen_rgb: Generated RGB image ``[H, W, 3]`` in ``[0, 1]``.
        target_edge: Target edge map ``[H, W]`` in ``[0, 1]``.
        pred_edge: Predicted edge map ``[H, W]`` in ``[0, 1]``.
        save_path: Output file path.
        metrics: Optional dict of metric name to value.
    """
    residual = np.abs(pred_edge - target_edge)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    panels = [
        ("GT Image", gt_rgb, None, {}),
        ("Generated Image", gen_rgb, None, {}),
        ("Target Edges (HED of GT)", target_edge, "gray", {"vmin": 0, "vmax": 1}),
        ("Predicted Edges (HED of Gen)", pred_edge, "gray", {"vmin": 0, "vmax": 1}),
        ("Residual |Pred - Target|", residual, "viridis", {"vmin": 0}),
    ]

    for ax, (title, img, cmap, extra_kwargs) in zip(axes, panels):
        if cmap is not None:
            im = ax.imshow(img, cmap=cmap, **extra_kwargs)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    if metrics:
        edge_parts = [f"{n}: {metrics[n]:.4f}" for n in ("edge_mae", "edge_mse") if n in metrics]
        img_parts = [f"{n}: {metrics[n]:.4f}" for n in ("psnr", "ssim", "lpips") if n in metrics]
        lines = []
        if edge_parts:
            lines.append("edge - " + "  ".join(edge_parts))
        if img_parts:
            lines.append("img  - " + "  ".join(img_parts))
        if lines:
            fig.suptitle("\n".join(lines), fontsize=10, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.95] if metrics else [0, 0, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
