import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ResidualDebugInfo:
    """Intermediate tensors from a single residual / gradient computation."""

    z0_hat_decoded: torch.Tensor  # [B, 3, H, W] decoded x0 estimate
    forward_pred: torch.Tensor  # [B, 1, H, W] forward-process prediction (e.g. depth)
    condition: torch.Tensor  # [B, C, H, W] conditioning target
    residual: torch.Tensor  # [B, C, H, W] forward_pred - condition
    z0_hat: Optional[torch.Tensor] = None  # [B, C, H, W] latent-space x0 estimate (pre-decode)


@dataclass
class StepDebugInfo:
    """Debug snapshot collected at one denoising step."""

    step_index: int
    sigma: float
    z0_hat_decoded: torch.Tensor  # [B, 3, H, W]
    forward_pred: Optional[torch.Tensor] = None  # [B, 1, H, W]
    condition: Optional[torch.Tensor] = None  # [B, C, H, W]
    residual: Optional[torch.Tensor] = None  # [B, C, H, W]
    inner_residual_debug: Optional[ResidualDebugInfo] = None


def _resolve_row_tensor(step: StepDebugInfo, key: str) -> torch.Tensor | None:
    """Resolve a dotted key like ``"inner.z0_hat_decoded"`` on a StepDebugInfo."""
    if key.startswith("inner."):
        inner = step.inner_residual_debug
        if inner is None:
            return None
        return getattr(inner, key[6:], None)
    return getattr(step, key, None)


def save_z0_visualization(
    steps: list[StepDebugInfo],
    output_path: str | None = None,
    batch_idx: int = 0,
    final_image: Image.Image | None = None,
    dpi: int = 150,
) -> Image.Image | None:
    """Render a multi-row grid showing debug quantities at each sampled step.

    Rows rendered (top to bottom, rows with no data are omitted):
        z0 (outer), fwd_pred (outer), condition, residual (outer),
        z0 (inner), fwd_pred (inner), residual (inner).

    Residual rows use a diverging colormap (``RdBu_r``) with a separate
    colorbar per residual group (outer / inner), each with its own scale.
    The outer residual colorbar is labelled "raw, unscaled" to distinguish
    it from the inner residual which may have been rescaled.

    Args:
        steps: Debug snapshots collected during denoising.
        output_path: Where to write the grid. If ``None``, the image is
            returned as a PIL Image instead of being saved to disk.
        batch_idx: Which sample in the batch to visualize.
        final_image: Optional final pipeline output appended as last column.
        dpi: Resolution of the saved figure.

    Returns:
        A PIL Image when *output_path* is ``None``, otherwise ``None``.
    """
    if not steps:
        return

    import matplotlib.pyplot as plt
    import numpy as np

    has_outer_fwd = any(s.forward_pred is not None for s in steps)
    has_inner = any(s.inner_residual_debug is not None for s in steps)

    # (dotted_key, row_label, render_mode)
    #   "rgb_sym": [-1,1]->RGB, "rgb_or_gray_unit": [0,1]->RGB or gray (auto),
    #   "residual": diverging colormap
    row_defs: list[tuple[str, str, str]] = [("z0_hat_decoded", "z0 (outer)", "rgb_sym")]
    if has_outer_fwd:
        row_defs.append(("forward_pred", "fwd (outer)", "rgb_or_gray_unit"))
        row_defs.append(("condition", "condition", "rgb_sym"))
        row_defs.append(("residual", "res (outer)", "residual"))
    if has_inner:
        row_defs.append(("inner.z0_hat_decoded", "z0 (inner)", "rgb_sym"))
        row_defs.append(("inner.forward_pred", "fwd (inner)", "rgb_or_gray_unit"))
        row_defs.append(("inner.residual", "res (inner)", "residual"))

    n_step_cols = len(steps)
    has_final = final_image is not None
    n_cols = n_step_cols + (1 if has_final else 0)
    n_rows = len(row_defs)
    residual_keys = {key for key, _, mode in row_defs if mode == "residual"}

    # Per-key vmax so each residual group gets its own colorbar range.
    res_vmax: dict[str, float] = {key: 0.0 for key in residual_keys}
    for step in steps:
        for key in residual_keys:
            tensor = _resolve_row_tensor(step, key)
            if tensor is not None:
                arr = tensor[batch_idx].detach().float().cpu()
                res_vmax[key] = max(res_vmax[key], arr.abs().max().item())
    res_vmax = {k: max(v, 1e-6) for k, v in res_vmax.items()}

    cell_w, cell_h = 2.0, 2.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(cell_w * n_cols + 1.2, cell_h * n_rows), squeeze=False)
    fig.subplots_adjust(wspace=0.02, hspace=0.15)

    # Track one imshow handle + row index per residual key for colorbars.
    residual_handles: dict[str, tuple[Any, int]] = {}

    for col, step in enumerate(steps):
        for row, (key, _, mode) in enumerate(row_defs):
            ax = axes[row, col]
            tensor = _resolve_row_tensor(step, key)
            if tensor is None:
                ax.axis("off")
                continue

            arr = tensor[batch_idx].detach().float().cpu()

            if mode == "rgb_sym":
                rgb = (arr.clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()
                ax.imshow(rgb)
            elif mode == "rgb_or_gray_unit":
                if arr.shape[0] >= 3:
                    rgb = arr[:3].clamp(0, 1).permute(1, 2, 0).numpy()
                    ax.imshow(rgb)
                else:
                    gray = arr[0].clamp(0, 1).numpy()
                    ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            elif mode == "residual":
                vmax = res_vmax[key]
                gray = arr[0].numpy()
                im = ax.imshow(gray, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                if key not in residual_handles:
                    residual_handles[key] = (im, row)

            ax.set_xticks([])
            ax.set_yticks([])

        axes[0, col].set_title(f"t={step.step_index}\nσ={step.sigma:.3f}", fontsize=7)

    if has_final:
        col = n_step_cols
        for row in range(n_rows):
            axes[row, col].axis("off")
        ax_final = axes[0, col]
        ax_final.imshow(np.asarray(final_image))
        ax_final.set_xticks([])
        ax_final.set_yticks([])
        ax_final.set_title("final", fontsize=7)

    for row, (_, label, _) in enumerate(row_defs):
        axes[row, 0].set_ylabel(label, fontsize=7, rotation=90, labelpad=10)

    residual_labels = {"residual": "residual (raw, unscaled)", "inner.residual": "residual (inner)"}
    for key, (im, row) in residual_handles.items():
        fig.colorbar(im, ax=axes[row, :].tolist(), shrink=0.8, pad=0.01, label=residual_labels.get(key, key))

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug("z0 derivation visualization saved to %s", output_path)
        return None

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    pil_img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
    plt.close(fig)
    return pil_img
