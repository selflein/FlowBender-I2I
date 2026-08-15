#!/usr/bin/env python
"""Evaluate an SD3 ControlNet or FlowChef pipeline.

Supports multi-GPU via ``accelerate``::
accelerate launch --num_processes 4 sd3/evaluate.py eval_baseline=controlnet model_dir=/path/to/checkpoint

Or single-GPU:
python sd3/evaluate.py eval_baseline=controlnet model_dir=/path/to/checkpoint
"""

import gc
import json
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, SD3Transformer2DModel, attention_backend
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from sd3.dataset import PreprocessedControlNetDataset, collate_fn
from sd3.edge import _compute_edge_metrics, _save_edge_comparison
from sd3.flowbender import SD3ControlNetModelFlowBender, StableDiffusion3FlowBenderPipeline
from sd3.image_metrics import compute_psnr_ssim_lpips
from sd3.residual_utils import FORWARD_PROCESS_REGISTRY
from sd3.vis import save_z0_visualization


def _save_depth_comparison(
    gt_rgb: np.ndarray,
    gen_rgb: np.ndarray,
    condition: np.ndarray,
    pred_gt_minmax: tuple[np.ndarray, np.ndarray],
    save_path: str | Path,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a 2-row debug plot: images, min-max-aligned depths.

    Row 1: GT image, generated image, condition depth.
    Row 2: GT depth (minmax), predicted depth (minmax), residual (minmax).

    Args:
        gt_rgb: Ground-truth RGB image [H, W, 3] in [0, 1].
        gen_rgb: Generated RGB image [H, W, 3] in [0, 1].
        condition: Single-channel conditioning depth [H, W].
        pred_gt_minmax: Tuple of (predicted depth, ground-truth depth) [H, W].
        save_path: Output file path.
        metrics: Optional dict of per-sample metric name to value to display.
    """
    pred_minmax, gt_minmax = pred_gt_minmax
    residual_minmax = np.abs(pred_minmax - gt_minmax)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    rows = [
        [
            ("GT Image", gt_rgb, None, {}),
            ("Generated Image", gen_rgb, None, {}),
            ("Condition Depth", condition, "inferno", {}),
        ],
        [
            ("GT Depth (MinMax)", gt_minmax, "inferno", {}),
            ("Pred Depth (MinMax)", pred_minmax, "inferno", {}),
            ("Residual (MinMax)", residual_minmax, "viridis", {"vmin": 0}),
        ],
    ]

    for row_idx, row in enumerate(rows):
        for col_idx, (title, img, cmap, extra_kwargs) in enumerate(row):
            ax = axes[row_idx, col_idx]
            if cmap is not None:
                im = ax.imshow(img, cmap=cmap, **extra_kwargs)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(img)
            ax.set_title(title, fontsize=12)
            ax.axis("off")

    if metrics:
        parts = [f"{name}: {metrics[name]:.4f}" for name in ("delta1", "mae") if name in metrics]
        if parts:
            fig.suptitle("  ".join(parts), fontsize=10, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.95] if metrics else [0, 0, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _null_embeds_cache_dir(base_model: str, max_seq_len: int) -> Path:
    """Build a deterministic cache path under $TMPDIR for null text embeddings."""
    import hashlib
    import os

    tmpdir = Path(os.environ.get("TMPDIR", "/tmp"))
    model_id = base_model.replace("/", "--")
    key = hashlib.sha256(f"{base_model}:{max_seq_len}".encode()).hexdigest()[:12]
    return tmpdir / "null_embeds_cache" / f"{model_id}_{key}"


def _compute_negative_embeddings(base_model, device, weight_dtype, max_seq_len):
    """Encode the empty string once to obtain unconditional embeddings for CFG."""
    from sd3.text_utils import get_null_text_embeds

    neg_embeds, neg_pooled = get_null_text_embeds(
        pretrained_model_name_or_path=base_model,
        cache_dir=_null_embeds_cache_dir(base_model, max_seq_len),
        device=device,
        weight_dtype=weight_dtype,
        max_sequence_length=max_seq_len,
    )
    return neg_embeds.unsqueeze(0), neg_pooled.unsqueeze(0)


@torch.no_grad()
def _compute_depth_metrics(
    gen_tensors: torch.Tensor,
    cond_tensors: torch.Tensor,
    forward_process: torch.nn.Module,
    threshold: float = 1e-3,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, list[float]]]:
    """Compute MAE and delta<1.25 between generated and conditioning depth.

    Both the predicted and GT depth are converted to the *standard* convention
    (close=0, far=1) via per-image min-max normalisation before comparison.

    Args:
        gen_tensors: Generated images [B, 3, H, W] in [0, 1].
        cond_tensors: Conditioning depth images [B, 3, H, W] in [-1, 1]
        forward_process: ``DepthForwardProcess`` instance.
        threshold: Ignore GT pixels below this value to avoid division
            instability near zero.
    """
    B, _, H, W = gen_tensors.shape
    from sd3.depth.alignment import compute_metrics

    target = forward_process.get_target(cond_tensors)[:, 0]
    pred = forward_process.predict(gen_tensors)[:, 0]  # [B, H, W]
    aligned, metrics = compute_metrics(pred, target)
    return aligned, metrics


@torch.no_grad()
def _compute_sr_metrics(
    gen_tensors: torch.Tensor,
    cond_tensors: torch.Tensor,
    forward_process: torch.nn.Module,
    gt_tensors: torch.Tensor | None = None,
) -> tuple[dict, dict[str, list[float]]]:
    """Compute super-resolution metrics.

    Reports two groups of per-sample PSNR / SSIM / LPIPS / MAE:

    * ``psnr`` / ``ssim`` / ``lpips`` / ``mae`` — full-resolution generation vs GT.
    * ``psnr_lr`` / ``ssim_lr`` / ``lpips_lr`` / ``mae_lr`` — condition alignment
      at the LR scale: ``downsample(gen, s)`` vs ``downsample(get_target(cond), s)``.
      Measures how well the generation respects the low-frequency content
      of the LR observation, independent of perceptual quality on the GT.

    Args:
        gen_tensors: Generated images [B, 3, H, W] in [0, 1].
        cond_tensors: Conditioning images [B, 3, H, W] in [-1, 1].
        forward_process: ``SuperResolutionForwardProcess`` instance.
        gt_tensors: Ground-truth images [B, 3, H, W] in [0, 1].

    Returns:
        Tuple of (aligned dict for debug viz, metrics dict).
    """
    if gt_tensors is None:
        raise ValueError("gt_tensors is required for super-resolution metrics")

    full = compute_psnr_ssim_lpips(gen_tensors, gt_tensors)

    # Compare A(gen) to A(GT) directly so the LR-consistency metric stays a
    # clean operator residual under any down/up kernel pair. (In avg_pool
    # mode this is bit-identical to comparing against `get_target(cond)` --
    # avg_pool is a left-inverse of NN-up. In bilinear mode the cond round-trip
    # would otherwise apply bilinear_down twice and lose extra HF content.)
    lr_gen = forward_process.downsample(gen_tensors.float())
    lr_target = forward_process.downsample(gt_tensors)
    lr = compute_psnr_ssim_lpips(lr_gen, lr_target)

    metrics = {**full, **{f"{k}_lr": v for k, v in lr.items()}}
    aligned = {"gen": gen_tensors, "gt": gt_tensors, "lr_gen": lr_gen, "lr_target": lr_target}
    return aligned, metrics


def _save_sr_comparison(
    gt_rgb: np.ndarray,
    gen_rgb: np.ndarray,
    condition_rgb: np.ndarray,
    save_path: str | Path,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a 2-row debug plot for super-resolution evaluation.

    Row 1: GT image, generated image, condition (LR upsampled).
    Row 2: |GT - generated| residual (per-channel mean).

    Args:
        gt_rgb: Ground-truth RGB image [H, W, 3] in [0, 1].
        gen_rgb: Generated RGB image [H, W, 3] in [0, 1].
        condition_rgb: LR-upsampled conditioning image [H, W, 3] in [0, 1].
        save_path: Output file path.
        metrics: Optional dict of metric name to value.
    """
    residual = np.abs(gt_rgb - gen_rgb)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    panels = [
        ("GT Image", gt_rgb, None, {}),
        ("Generated Image", gen_rgb, None, {}),
        ("Condition (LR upsampled)", condition_rgb, None, {}),
        ("Residual |GT - Gen|", residual.mean(axis=-1), "viridis", {"vmin": 0}),
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
        full_parts = [f"{n}: {metrics[n]:.4f}" for n in ("psnr", "ssim", "lpips", "mae") if n in metrics]
        lr_parts = [f"{n}: {metrics[n]:.4f}" for n in ("psnr_lr", "ssim_lr", "lpips_lr", "mae_lr") if n in metrics]
        lines = []
        if full_parts:
            lines.append("full - " + "  ".join(full_parts))
        if lr_parts:
            lines.append("lr   - " + "  ".join(lr_parts))
        if lines:
            fig.suptitle("\n".join(lines), fontsize=10, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.95] if metrics else [0, 0, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def _compute_jpeg_metrics(
    gen_tensors: torch.Tensor,
    cond_tensors: torch.Tensor,
    forward_process: torch.nn.Module,
    gt_tensors: torch.Tensor | None = None,
) -> tuple[dict, dict[str, list[float]]]:
    """Compute JPEG restoration metrics.

    Reports two groups of per-sample PSNR / SSIM / LPIPS / MAE:

    * ``psnr`` / ``ssim`` / ``lpips`` / ``mae`` -- full-resolution generation vs GT.
    * ``psnr_jpeg`` / ``ssim_jpeg`` / ``lpips_jpeg`` / ``mae_jpeg`` -- JPEG-consistency:
      ``A(gen)`` vs ``y = get_target(cond)``, i.e. how well the generation
      respects the compressed observation, analogous to the SR ``*_lr``
      metrics.

    Args:
        gen_tensors: Generated images [B, 3, H, W] in [0, 1].
        cond_tensors: JPEG-degraded conditioning [B, 3, H, W] in [-1, 1].
        forward_process: ``JpegRestorationForwardProcess`` instance.
        gt_tensors: Ground-truth images [B, 3, H, W] in [0, 1].

    Returns:
        Tuple of (aligned dict for debug viz, metrics dict).
    """
    if gt_tensors is None:
        raise ValueError("gt_tensors is required for jpeg_restoration metrics")

    full = compute_psnr_ssim_lpips(gen_tensors, gt_tensors)

    jpeg_gen = forward_process.predict(gen_tensors.float())
    jpeg_target = forward_process.get_target(cond_tensors)
    jpeg = compute_psnr_ssim_lpips(jpeg_gen, jpeg_target)

    metrics = {**full, **{f"{k}_jpeg": v for k, v in jpeg.items()}}
    aligned = {"gen": gen_tensors, "gt": gt_tensors, "jpeg_gen": jpeg_gen, "jpeg_target": jpeg_target}
    return aligned, metrics


def _save_jpeg_comparison(
    gt_rgb: np.ndarray,
    gen_rgb: np.ndarray,
    condition_rgb: np.ndarray,
    save_path: str | Path,
    metrics: dict[str, float] | None = None,
) -> None:
    """Save a debug plot for JPEG restoration evaluation.

    Panels: GT image, generated image, JPEG-degraded condition,
    ``|GT - generated|`` residual.

    Args:
        gt_rgb: Ground-truth RGB image [H, W, 3] in [0, 1].
        gen_rgb: Generated RGB image [H, W, 3] in [0, 1].
        condition_rgb: JPEG-degraded conditioning image [H, W, 3] in [0, 1].
        save_path: Output file path.
        metrics: Optional dict of metric name to value.
    """
    residual = np.abs(gt_rgb - gen_rgb)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    panels = [
        ("GT Image", gt_rgb, None, {}),
        ("Generated Image", gen_rgb, None, {}),
        ("Condition (JPEG)", condition_rgb, None, {}),
        ("Residual |GT - Gen|", residual.mean(axis=-1), "viridis", {"vmin": 0}),
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
        full_parts = [f"{n}: {metrics[n]:.4f}" for n in ("psnr", "ssim", "lpips", "mae") if n in metrics]
        jpeg_parts = [
            f"{n}: {metrics[n]:.4f}" for n in ("psnr_jpeg", "ssim_jpeg", "lpips_jpeg", "mae_jpeg") if n in metrics
        ]
        lines = []
        if full_parts:
            lines.append("full - " + "  ".join(full_parts))
        if jpeg_parts:
            lines.append("jpeg - " + "  ".join(jpeg_parts))
        if lines:
            fig.suptitle("\n".join(lines), fontsize=10, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.95] if metrics else [0, 0, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


MODALITY_TO_METRICS = {
    "depth": _compute_depth_metrics,
    "super_resolution": _compute_sr_metrics,
    "jpeg_restoration": _compute_jpeg_metrics,
    "edge": _compute_edge_metrics,
}


# Maps a forward-process type to the config key that stores its dataset root
# (relative to which we append `/test` to build `cond_dir`). Used only when
# the user didn't pass `task=...` and we auto-derive from the ControlNet
# config; `super_resolution` has no dedicated root and reads from `images/`.
_DEFAULT_COND_ROOT_KEYS: dict[str, str] = {"depth": "depths_root"}


def _resolve_task(cfg: DictConfig, controlnet: SD3ControlNetModelFlowBender | None) -> tuple[str, dict, Path]:
    """Determine forward-process type, kwargs and cond_dir for this evaluation.

    When a ControlNet is loaded, its saved config supplies the base
    ``forward_process_type`` and ``forward_process_kwargs``; the
    ``forward_process_type`` cannot be changed (the trained ControlNet only
    speaks one task), but ``forward_process_kwargs`` from the Hydra config
    are merged on top so the same checkpoint can be evaluated under a
    different forward process (e.g. change SR ``scale_factor``).  For non-ControlNet baselines both values come from
    the Hydra task config.

    ``cond_dir`` is always overridable via ``data.cond_dir`` / ``cond_root`` so
    that the same trained model can be evaluated against different datasets.

    Raises:
        ValueError: if the user passes ``task=...`` that disagrees with the
            ControlNet's trained task, or if ``cond_dir`` cannot be derived.
    """
    cond_type_override = cfg.data.condition_type
    cond_dir_override = cfg.data.cond_dir
    cond_root = cfg.get("cond_root", None)

    cfg_fp_kwargs_raw = cfg.get("forward_process_kwargs", {}) or {}
    cfg_fp_kwargs = (
        OmegaConf.to_container(cfg_fp_kwargs_raw, resolve=True)
        if OmegaConf.is_config(cfg_fp_kwargs_raw)
        else dict(cfg_fp_kwargs_raw)
    )

    if controlnet is not None:
        forward_process_type = controlnet.config.get("forward_process_type")
        ckpt_fp_kwargs = controlnet.config.get("forward_process_kwargs", {}) or {}
        if cond_type_override and cond_type_override != forward_process_type:
            raise ValueError(
                f"Task mismatch: ControlNet was trained for {forward_process_type!r}, "
                f"but `task=` requested {cond_type_override!r}. Omit `task=` to use "
                f"the ControlNet's trained task."
            )
        # Checkpoint kwargs are the base; Hydra cfg overrides win on a per-key basis.
        fp_kwargs = {**dict(ckpt_fp_kwargs), **cfg_fp_kwargs}
    else:
        forward_process_type = cond_type_override or "depth"
        fp_kwargs = cfg_fp_kwargs

    if cond_dir_override:
        cond_dir = Path(cond_dir_override)
    elif cond_root:
        cond_dir = Path(cond_root) / "test"
    elif forward_process_type in ("super_resolution", "jpeg_restoration", "edge"):
        cond_dir = Path(cfg.data_root) / "images" / "test"
    else:
        key = _DEFAULT_COND_ROOT_KEYS.get(forward_process_type)
        if key is None:
            raise ValueError(
                f"Cannot auto-derive cond_dir for forward_process_type="
                f"{forward_process_type!r}. Pass `task=...` or set `data.cond_dir`."
            )
        cond_dir = Path(cfg[key]) / "test"

    return forward_process_type, fp_kwargs, cond_dir


def _aggregate_metrics(eval_dir: Path) -> dict[str, float]:
    """Merge the per-rank ``metrics_*.json`` files into a single summary.

    Args:
        eval_dir: Evaluation output directory containing ``metrics_*.json``.

    Returns:
        Dict mapping metric names to their mean values.
    """
    per_rank_files = sorted(eval_dir.glob("metrics_*.json"))
    if not per_rank_files:
        print("Warning: no metrics_*.json files found, skipping aggregation.")
        return {}

    combined: dict[str, list[float]] = defaultdict(list)
    for path in per_rank_files:
        rank_metrics = json.loads(path.read_text())
        for key, values in rank_metrics.items():
            combined[key].extend(values)

    means = {k: float(np.mean(v)) for k, v in combined.items()}
    n = len(next(iter(combined.values())))
    print(f"Aggregated {len(per_rank_files)} rank file(s), {n} samples total.")
    for k, v in sorted(means.items()):
        print(f"  {k}: {v:.6f}")

    return means


def aggregate_and_compute_fid(eval_dir: Path, ref_dir: Path, *, mode: str = "clean", num_workers: int = 8) -> None:
    """Aggregate per-rank metrics, compute FID, and write ``metrics.json``.

    Args:
        eval_dir: Evaluation output directory (contains ``generated/`` and
            ``metrics_*.json``).
        ref_dir: Directory of reference (ground-truth) images.
        mode: clean-fid mode ("clean", "legacy_pytorch", "legacy_tensorflow").
        num_workers: Number of dataloader workers for feature extraction.
    """
    from cleanfid import fid

    summary = _aggregate_metrics(eval_dir)

    score = fid.compute_fid(str(eval_dir / "generated"), str(ref_dir), mode=mode, num_workers=num_workers)
    print(f"FID ({mode}): {score:.4f}")
    summary["fid"] = score
    summary["fid_mode"] = mode

    out_path = eval_dir / "metrics.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved to {out_path}")


@torch.no_grad()
def run_evaluation(cfg: DictConfig) -> None:
    """Evaluate an SD3 pipeline."""
    model_dir = Path(cfg.model_dir) if cfg.model_dir else None
    output_dir = Path(cfg.output_dir)
    base_model = cfg.base_model

    pipe_cls = hydra.utils.get_class(cfg.pipeline._target_)
    uses_controlnet = issubclass(pipe_cls, StableDiffusion3FlowBenderPipeline)
    additional_call_kwargs = OmegaConf.to_container(cfg.pipeline.get("additional_call_kwargs", {}), resolve=True)

    gen_cfg = cfg.generation
    resolution = gen_cfg.resolution
    max_sequence_length = gen_cfg.max_sequence_length
    num_inference_steps = gen_cfg.num_inference_steps
    guidance_scale = gen_cfg.guidance_scale
    seed = gen_cfg.seed
    batch_size = gen_cfg.batch_size
    max_samples = gen_cfg.max_samples

    img_dir = Path(cfg.data.img_dir)
    text_embeds_dir = Path(cfg.data.text_embeds_dir)
    prompt_dir = Path(cfg.data.prompt_dir) if cfg.data.prompt_dir else None

    accelerator = Accelerator()
    device = accelerator.device
    is_main = accelerator.is_main_process
    rank = accelerator.process_index
    num_processes = accelerator.num_processes
    weight_dtype = torch.bfloat16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_main:
        print(f"Evaluating on {accelerator.num_processes} GPU(s)")
        print(f"Pipeline: {cfg.pipeline._target_}")
        print(f"Model: {model_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"metrics_{rank}.json"
    metrics: dict[str, list[float]] = defaultdict(list)
    if metrics_path.exists():
        # Resume: load prior values. Wrap in defaultdict so newly-added metric
        # keys auto-create on first access without KeyError.
        metrics.update(json.loads(metrics_path.read_text()))
        finished_count = len(next(iter(metrics.values())))
    else:
        finished_count = 0

    # --- Load controlnet (ControlNet baselines only) ---------------------
    controlnet = None
    if uses_controlnet:
        if model_dir is None:
            raise ValueError("model_dir is required for ControlNet baselines")
        ctrl_src = model_dir / "controlnet" if (model_dir / "controlnet").is_dir() else model_dir
        controlnet = SD3ControlNetModelFlowBender.from_pretrained(str(ctrl_src), torch_dtype=weight_dtype)
        setattr(controlnet, "_repeated_blocks", ["JointTransformerBlock", "SD3SingleTransformerBlock"])
        print("Compiling controlnet ...")
        controlnet.compile_repeated_blocks(fullgraph=True)

        if is_main:
            print(f"  Feedback mode: {controlnet.config.get('feedback_mode', 'vanilla')}")

    # --- Resolve task from controlnet config (single source of truth) ----
    forward_process_type, fp_kwargs, cond_dir = _resolve_task(cfg, controlnet)
    cfg.data.condition_type = forward_process_type
    cfg.data.cond_dir = str(cond_dir)
    if is_main:
        print(f"  Task: {forward_process_type}  (cond_dir={cond_dir})")
        print(f"  Forward process kwargs: {fp_kwargs}")
        (output_dir / "eval_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # --- Dataset (with global indices) ----------------------------------
    dataset: Dataset = PreprocessedControlNetDataset(
        img_dir=str(img_dir),
        cond_dir=str(cond_dir),
        text_embeds_dir=str(text_embeds_dir),
        prompt_dir=str(prompt_dir) if prompt_dir else None,
        resolution=resolution,
        condition_type=forward_process_type,
        scale_factor=fp_kwargs.get("scale_factor", None),
        jpeg_quality=fp_kwargs.get("quality", None),
    )
    # Read sample_seq_len from the full dataset before subsetting so this still
    # works on ranks whose subset is empty (e.g. when resuming a run where the
    # rank already finished all its samples).
    sample_seq_len = dataset[0]["prompt_embeds"].shape[0]

    end = min(max_samples, len(dataset)) if max_samples is not None else len(dataset)
    indices = list(range(rank, end, num_processes))
    dataset = Subset(dataset, indices[finished_count:])

    if is_main:
        print(f"Evaluation samples: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- Negative embeddings for CFG ------------------------------------
    target_seq_len = 77 + max_sequence_length
    if sample_seq_len > target_seq_len and is_main:
        print(
            f"  Truncating prompt_embeds from {sample_seq_len} → "
            f"{target_seq_len} tokens (max_sequence_length={max_sequence_length})"
        )

    if is_main:
        print("Computing negative prompt embeddings ...")
    neg_prompt_embeds, neg_pooled_prompt_embeds = _compute_negative_embeddings(
        base_model, device, weight_dtype, max_sequence_length
    )

    if is_main:
        print(f"Loading {forward_process_type} with {fp_kwargs} model for task-specific metrics ...")
    forward_process = FORWARD_PROCESS_REGISTRY[forward_process_type](**fp_kwargs)
    forward_process.to(device).eval()
    forward_process.requires_grad_(False)

    # --- Load base transformer ------------------------------------------
    if model_dir is not None and (model_dir / "transformer").is_dir():
        transformer = SD3Transformer2DModel.from_pretrained(model_dir / "transformer", torch_dtype=weight_dtype)
    else:
        transformer = SD3Transformer2DModel.from_pretrained(
            base_model, subfolder="transformer", torch_dtype=weight_dtype
        )
    setattr(transformer, "_repeated_blocks", ["JointTransformerBlock", "SD3SingleTransformerBlock"])
    print("Compiling transformer ...")
    transformer.compile_repeated_blocks(fullgraph=True)

    # --- Load VAE -------------------------------------------------------
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=weight_dtype)

    # --- Build generation pipeline (no text encoders) -------------------
    if is_main:
        print(f"Building {pipe_cls.__name__} pipeline ...")
    pipe_kwargs = dict(
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        torch_dtype=weight_dtype,
    )
    if uses_controlnet:
        pipe_kwargs["controlnet"] = controlnet
    pipe = pipe_cls.from_pretrained(base_model, **pipe_kwargs)
    pipe._forward_process = forward_process
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    use_cfg = guidance_scale > 1.0

    # --- Output dirs ----------------------------------------------------
    gen_dir = output_dir / "generated"
    if is_main:
        gen_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    stopped_early = False

    metric_fun = MODALITY_TO_METRICS[forward_process_type]

    recompute_cond = bool(cfg.get("recompute_cond_from_image", False))
    # Edge conditioning is always derived from the GT image on the fly --
    # there is no parallel on-disk edge directory.
    if forward_process_type == "edge":
        recompute_cond = True
    if recompute_cond:
        if forward_process_type not in ("depth", "edge"):
            raise ValueError(
                f"recompute_cond_from_image=True is only implemented for depth and edge; "
                f"forward_process_type={forward_process_type!r} derives the condition "
                f"on the fly inside the dataset already."
            )
        if is_main:
            print(
                f"  recompute_cond_from_image=True: deriving {forward_process_type} condition "
                "from GT image at eval time."
            )

    # --- Generation + metrics -------------------------------------------
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating", disable=not is_main)):
        img_ids = batch["id"]
        B = batch["pixel_values"].shape[0]

        cond_tensors = batch["conditioning_pixel_values"]  # [B, 3, H, W], range [-1, 1]
        prompt_embeds = batch["prompt_embeds"][:, :target_seq_len]
        pooled_prompt_embeds = batch["pooled_prompt_embeds"]

        if recompute_cond:
            # Run the condition extractor on the GT image and use its output as
            # the ControlNet conditioning. Mirrors data/batch_depth.py
            # for depth; for edge there is no offline pipeline -- HED is always
            # applied here.
            gt_01 = (batch["pixel_values"].to(device, dtype=torch.float32) + 1.0) / 2.0
            with torch.no_grad():
                cond_01 = forward_process.compute_condition(gt_01)  # [B, 1, H, W] in [0, 1]
            cond_tensors = cond_01.repeat(1, 3, 1, 1) * 2.0 - 1.0  # [B, 3, H, W] in [-1, 1]

        # Skip batch if every image was already generated (resume support)
        save_paths = [gen_dir / f"{img_ids[j]}.png" for j in range(B)]
        is_debug_batch = batch_idx < 3
        debug_info = None
        if all(p.exists() for p in save_paths):
            gen_images = [Image.open(p).convert("RGB") for p in save_paths]
        else:
            generators = [
                torch.Generator(device=device).manual_seed(seed + batch_idx * batch_size + j) for j in range(B)
            ]

            visualize_kwargs = {}
            if is_debug_batch:
                visualize_kwargs["visualize_z0"] = True
                visualize_kwargs["visualize_z0_every_n"] = max(1, num_inference_steps // 8)

            with attention_backend("_native_flash"):
                result = pipe(
                    prompt_embeds=prompt_embeds.to(device, dtype=weight_dtype),
                    negative_prompt_embeds=(
                        neg_prompt_embeds.expand(B, -1, -1).to(device, dtype=weight_dtype) if use_cfg else None
                    ),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, dtype=weight_dtype),
                    negative_pooled_prompt_embeds=(
                        neg_pooled_prompt_embeds.expand(B, -1).to(device, dtype=weight_dtype) if use_cfg else None
                    ),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=resolution,
                    width=resolution,
                    generator=generators,
                    control_image=cond_tensors.to(device, dtype=weight_dtype),
                    **additional_call_kwargs,
                    **visualize_kwargs,
                )
            gen_images = result.images
            debug_info = getattr(result, "debug_info", None)

            for j in range(B):
                gen_images[j].save(save_paths[j])

        # --- Per-batch metrics ------------------------------------------
        # Generated images (B, 3, H, W), range [0, 1]
        gen_t = torch.stack([transforms.ToTensor()(img) for img in gen_images]).to(device, dtype=torch.float32)
        # Ground truth images (B, 3, H, W), range [0, 1]
        gt_t = (batch["pixel_values"].to(device, dtype=torch.float32) + 1.0) / 2.0  # [-1, 1] → [0, 1]

        with torch.no_grad():
            aligned, task_metrics = metric_fun(
                gen_t, cond_tensors.to(device, dtype=torch.float32), forward_process, gt_tensors=gt_t
            )
            for k, v in task_metrics.items():
                metrics[k].extend(v)

        if is_debug_batch:
            debug_dir = output_dir / f"debug_{forward_process_type}"
            debug_dir.mkdir(parents=True, exist_ok=True)

            batch_metrics = {k: v[0] for k, v in task_metrics.items()}
            if forward_process_type == "super_resolution":
                cond_rgb = (cond_tensors * 0.5 + 0.5).clamp(0, 1)
                for j in range(B):
                    _save_sr_comparison(
                        gt_rgb=gt_t[j].permute(1, 2, 0).cpu().numpy(),
                        gen_rgb=gen_t[j].permute(1, 2, 0).cpu().numpy(),
                        condition_rgb=cond_rgb[j].permute(1, 2, 0).cpu().numpy(),
                        save_path=debug_dir / f"{img_ids[j]}.png",
                        metrics=batch_metrics,
                    )
            elif forward_process_type == "jpeg_restoration":
                cond_rgb = (cond_tensors * 0.5 + 0.5).clamp(0, 1)
                for j in range(B):
                    _save_jpeg_comparison(
                        gt_rgb=gt_t[j].permute(1, 2, 0).cpu().numpy(),
                        gen_rgb=gen_t[j].permute(1, 2, 0).cpu().numpy(),
                        condition_rgb=cond_rgb[j].permute(1, 2, 0).cpu().numpy(),
                        save_path=debug_dir / f"{img_ids[j]}.png",
                        metrics=batch_metrics,
                    )
            elif forward_process_type == "edge":
                target_edge = aligned["target_edge"][:, 0].cpu().numpy()
                pred_edge = aligned["pred_edge"][:, 0].cpu().numpy()
                for j in range(B):
                    _save_edge_comparison(
                        gt_rgb=gt_t[j].permute(1, 2, 0).cpu().numpy(),
                        gen_rgb=gen_t[j].permute(1, 2, 0).cpu().numpy(),
                        target_edge=target_edge[j],
                        pred_edge=pred_edge[j],
                        save_path=debug_dir / f"{img_ids[j]}.png",
                        metrics=batch_metrics,
                    )
            else:
                cond_display = cond_tensors[:, 0].to(device, dtype=torch.float32)
                for j in range(B):
                    _save_depth_comparison(
                        gt_rgb=gt_t[j].permute(1, 2, 0).cpu().numpy(),
                        gen_rgb=gen_t[j].permute(1, 2, 0).cpu().numpy(),
                        condition=cond_display[j].cpu().numpy(),
                        pred_gt_minmax=(
                            aligned["pred"][j].cpu().numpy(),
                            aligned["gt"][j].cpu().numpy(),
                        ),
                        save_path=debug_dir / f"{img_ids[j]}.png",
                        metrics=batch_metrics,
                    )

            if debug_info:
                z0_dir = output_dir / "debug_z0"
                z0_dir.mkdir(parents=True, exist_ok=True)
                for j in range(B):
                    save_z0_visualization(
                        debug_info,
                        output_path=str(z0_dir / f"{img_ids[j]}.png"),
                        batch_idx=j,
                        final_image=gen_images[j],
                    )

        # Flush metrics to disk
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # --- Gather metrics across processes --------------------------------
    accelerator.wait_for_everyone()

    to_delete = [pipe, transformer, vae]
    if controlnet is not None:
        to_delete.append(controlnet)
    del to_delete
    gc.collect()
    torch.cuda.empty_cache()

    if is_main and not stopped_early:
        aggregate_and_compute_fid(output_dir, img_dir)


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
