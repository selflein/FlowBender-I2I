#!/usr/bin/env python
"""Diagnose how closely the shortcut-sampling approximation matches the full probe.

For a small handful of samples, run a custom denoising loop that, at each step,
computes BOTH the full-probe feedback signal (residual or gradient) and the
shortcut signal derived from the previous step's cached ``z0_hat``, then report
per-step divergence metrics.

Sampling is always driven by the full-probe signal so measured divergence at
step ``i`` reflects only the step-``i`` approximation, not compounded errors
from earlier shortcut steps.

The primary metrics are:
    - gradient mode: cosine similarity of ``dLdLatents``.
    - residual mode: mean L1 distance of the residual tensor.
Both also record the free ``z0_hat`` cosine similarity (the underlying
approximation ``v_theta([z_{t+1}, g_{t+1}], t+1) ~ v_theta([z_t, 0], t)``).

Usage::

    python sd3/diagnose_shortcut.py \
        model_dir=/path/to/checkpoint \
        output_dir=/path/to/output \
        ++diagnose.num_samples=4

CFG is not supported; run with ``generation.guidance_scale=1.0`` (the eval
default). Single-GPU only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL, SD3Transformer2DModel, attention_backend
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet import retrieve_timesteps
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from sd3.dataset import PreprocessedControlNetDataset, collate_fn
from sd3.evaluate import _resolve_task
from sd3.flowbender import SD3ControlNetModelFlowBender, StableDiffusion3FlowBenderPipeline
from sd3.residual_utils import (
    FORWARD_PROCESS_REGISTRY,
    _predict_z0_hat,
    get_residual_condition,
    get_residual_gradient,
    vae_encode,
)

logger = logging.getLogger(__name__)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample cosine similarity over flattened tensors. Returns [B]."""
    af, bf = a.flatten(1).float(), b.flatten(1).float()
    num = (af * bf).sum(dim=1)
    denom = af.norm(dim=1).clamp_min(1e-8) * bf.norm(dim=1).clamp_min(1e-8)
    return num / denom


def mean_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).abs().flatten(1).mean(dim=1)


def l2_ratio(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.flatten(1).float().norm(dim=1) / b.flatten(1).float().norm(dim=1).clamp_min(1e-8)


def psnr_per_sample(reference: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample PSNR (dB) of *target* vs *reference*.

    The peak (data range) is taken per-sample from the reference signal, so the
    metric is meaningful even when residuals don't share a fixed [-1, 1] range
    across feedback variants. Returns [B].
    """
    a = reference.float().flatten(1)
    b = target.float().flatten(1)
    mse = ((a - b) ** 2).mean(dim=1).clamp_min(1e-12)
    data_range = (a.amax(dim=1) - a.amin(dim=1)).clamp_min(1e-8)
    return 10.0 * torch.log10(data_range**2 / mse)


def _call_feedback_helper(
    *,
    feedback_mode: str,
    pipe: StableDiffusion3FlowBenderPipeline,
    latents: torch.Tensor,
    t: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    orig_control_image: torch.Tensor,
    feedback_variant: Any,
    gradient_cond_kwargs: dict[str, Any],
    cached_z0_hat: torch.Tensor | None,
) -> tuple[torch.Tensor, Any]:
    """Dispatch to the right feedback helper with a cached-z0_hat override.

    Returns ``(signal, debug)`` where ``signal`` is the residual tensor (residual
    mode) or the gradient tensor (gradient mode) and ``debug`` carries the
    probe-derived ``z0_hat``.
    """
    if feedback_mode == "gradient":
        # Force skip_denoiser_grad=True on both paths so both produce dL/dz0_hat
        # with different z0_hat (probe vs cached) — apples-to-apples.
        kwargs = dict(gradient_cond_kwargs)
        kwargs["skip_denoiser_grad"] = True
        signal, debug, _ = get_residual_gradient(
            noisy_latents=latents,
            timesteps=t,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            transformer=pipe.transformer,
            vae=pipe.vae,
            scheduler=pipe.scheduler,
            raw_cond_img=orig_control_image,
            forward_process=pipe._forward_process,
            controlnet=pipe.controlnet,
            rescale=feedback_variant,
            cached_z0_hat=cached_z0_hat,
            **kwargs,
        )
        return signal, debug
    if feedback_mode == "residual":
        signal, debug, _ = get_residual_condition(
            noisy_latents=latents,
            timesteps=t,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            transformer=pipe.transformer,
            vae=pipe.vae,
            scheduler=pipe.scheduler,
            raw_cond_img=orig_control_image,
            forward_process=pipe._forward_process,
            controlnet=pipe.controlnet,
            variant=feedback_variant,
            cached_z0_hat=cached_z0_hat,
        )
        return signal, debug
    raise ValueError(f"Unsupported feedback_mode for diagnostic: {feedback_mode!r}")


@torch.no_grad()
def _diagnose_batch(
    pipe: StableDiffusion3FlowBenderPipeline,
    batch: dict[str, Any],
    *,
    num_inference_steps: int,
    seed: int,
    resolution: int,
    weight_dtype: torch.dtype,
    device: torch.device,
    feedback_mode: str,
    feedback_variant: Any,
    gradient_cond_kwargs: dict[str, Any],
    target_seq_len: int,
) -> list[dict[str, Any]]:
    """Run the diagnostic loop on one batch. Returns per-(sample, step) metrics."""
    prompt_embeds = batch["prompt_embeds"][:, :target_seq_len].to(device, dtype=weight_dtype)
    pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(device, dtype=weight_dtype)
    orig_control_image = batch["conditioning_pixel_values"].to(device, dtype=weight_dtype)
    img_ids = batch["id"]
    B = prompt_embeds.shape[0]

    controlnet_ref = pipe.controlnet if isinstance(pipe.controlnet, SD3ControlNetModel) else pipe.controlnet.nets[0]
    controlnet_config = controlnet_ref.config

    # --- Encode control image into latent space (mirrors pipeline) --------
    vae_shift_factor = 0 if controlnet_config.force_zeros_for_pooled_projection else pipe.vae.config.shift_factor
    control_image_latents = pipe.vae.encode(orig_control_image).latent_dist.sample()
    control_image_latents = (control_image_latents - vae_shift_factor) * pipe.vae.config.scaling_factor

    # --- Controlnet extra inputs ------------------------------------------
    if controlnet_config.force_zeros_for_pooled_projection:
        controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
    else:
        controlnet_pooled_projections = pooled_prompt_embeds
    controlnet_encoder_hidden_states = prompt_embeds if controlnet_config.joint_attention_dim is not None else None

    # --- Timesteps --------------------------------------------------------
    timesteps, _ = retrieve_timesteps(pipe.scheduler, num_inference_steps, device)

    # --- Initial latents (fixed seed for reproducibility) -----------------
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(
        B, pipe.vae.config.latent_channels, resolution, resolution, weight_dtype, device, generator, None
    )

    metrics: list[dict[str, Any]] = []
    cached_z0_hat: torch.Tensor | None = None

    for i, t in enumerate(tqdm(timesteps, desc="denoising", leave=False)):
        sigma = float(t.item() / pipe.scheduler.config.num_train_timesteps)

        # --- Full probe (drives sampling) ---------------------------------
        signal_full, debug_full = _call_feedback_helper(
            feedback_mode=feedback_mode,
            pipe=pipe,
            latents=latents,
            t=t,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            orig_control_image=orig_control_image,
            feedback_variant=feedback_variant,
            gradient_cond_kwargs=gradient_cond_kwargs,
            cached_z0_hat=None,
        )
        z0_hat_full = debug_full.z0_hat.to(device, dtype=weight_dtype)

        # --- Shortcut (only once cache is seeded by a previous step) ------
        if cached_z0_hat is not None:
            signal_short, _ = _call_feedback_helper(
                feedback_mode=feedback_mode,
                pipe=pipe,
                latents=latents,
                t=t,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                orig_control_image=orig_control_image,
                feedback_variant=feedback_variant,
                gradient_cond_kwargs=gradient_cond_kwargs,
                cached_z0_hat=cached_z0_hat,
            )

            z0_cos = cosine_sim(z0_hat_full, cached_z0_hat).cpu().tolist()
            if feedback_mode == "gradient":
                cos_g = cosine_sim(signal_full, signal_short).cpu().tolist()
                l2r = l2_ratio(signal_short, signal_full).cpu().tolist()
                for b in range(B):
                    metrics.append(
                        {
                            "sample_id": img_ids[b],
                            "step": i,
                            "sigma": sigma,
                            "cos_grad": float(cos_g[b]),
                            "l2_ratio": float(l2r[b]),
                            "z0_cos": float(z0_cos[b]),
                        }
                    )
            else:  # residual
                l1 = mean_l1(signal_full, signal_short).cpu().tolist()
                cos_r = cosine_sim(signal_full, signal_short).cpu().tolist()
                psnr_r = psnr_per_sample(signal_full, signal_short).cpu().tolist()
                for b in range(B):
                    metrics.append(
                        {
                            "sample_id": img_ids[b],
                            "step": i,
                            "sigma": sigma,
                            "l1_residual": float(l1[b]),
                            "cos_residual": float(cos_r[b]),
                            "psnr_residual": float(psnr_r[b]),
                            "z0_cos": float(z0_cos[b]),
                        }
                    )

        # --- Prepare guided forward using the full signal -----------------
        if feedback_mode == "gradient":
            cur_control_image_latents = control_image_latents
            latent_model_input = torch.cat([latents, signal_full], dim=1)
        else:  # residual
            enc_residuals = vae_encode(pipe.vae, signal_full)
            cur_control_image_latents = torch.cat([control_image_latents, enc_residuals], dim=1)
            latent_model_input = latents

        timestep = t.expand(latent_model_input.shape[0])
        control_block_samples = pipe.controlnet(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=controlnet_encoder_hidden_states,
            pooled_projections=controlnet_pooled_projections,
            controlnet_cond=cur_control_image_latents,
            return_dict=False,
        )[0]

        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_samples,
            return_dict=False,
        )[0]

        # --- Update cache before the scheduler step (latents still = z_t) --
        cached_z0_hat = _predict_z0_hat(latents, noise_pred, t, pipe.scheduler).detach()

        # --- Euler step ----------------------------------------------------
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    return metrics


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------


def _plot_metrics(metrics: list[dict[str, Any]], feedback_mode: str, save_path: Path) -> None:
    """Render per-sample lines + mean/std envelope for each metric against sigma."""
    if not metrics:
        logger.warning("No metrics collected; skipping plot.")
        return

    metric_keys = ("cos_grad", "l2_ratio", "l1_residual", "cos_residual", "psnr_residual")
    sample_ids = sorted({m["sample_id"] for m in metrics})
    by_sample: dict[str, dict[str, list[float]]] = {
        sid: {k: [] for k in ("sigma", "z0_cos", *metric_keys)} for sid in sample_ids
    }
    for m in metrics:
        row = by_sample[m["sample_id"]]
        row["sigma"].append(m["sigma"])
        row["z0_cos"].append(m["z0_cos"])
        for k in metric_keys:
            if k in m:
                row[k].append(m[k])

    if feedback_mode == "gradient":
        panels = [("cos_grad", "cosine(grad_full, grad_short)", "cos sim")]
    else:
        panels = [
            ("l1_residual", "L1(residual_full, residual_short)", "mean |delta|"),
            ("psnr_residual", "PSNR(residual_full, residual_short)", "PSNR (dB)"),
        ]
    panels.append(("z0_cos", "cosine(z0_hat_full, z0_hat_cached)", "cos sim"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    for ax, (key, title, ylabel) in zip(axes[0], panels):
        all_x: list[list[float]] = []
        all_y: list[list[float]] = []
        for sid in sample_ids:
            xs = by_sample[sid]["sigma"]
            ys = by_sample[sid][key]
            if not xs:
                continue
            order = np.argsort(xs)[::-1]  # high sigma (early step) to low sigma (late step)
            xs_arr = np.array(xs)[order]
            ys_arr = np.array(ys)[order]
            ax.plot(xs_arr, ys_arr, alpha=0.3, linewidth=1.0, label=sid)
            all_x.append(xs_arr.tolist())
            all_y.append(ys_arr.tolist())

        if len(all_y) > 1 and all(len(y) == len(all_y[0]) for y in all_y):
            y_mean = np.mean(all_y, axis=0)
            y_std = np.std(all_y, axis=0)
            x_ref = np.array(all_x[0])
            ax.plot(x_ref, y_mean, color="black", linewidth=2.0, label="mean")
            ax.fill_between(x_ref, y_mean - y_std, y_mean + y_std, color="black", alpha=0.15)

        ax.set_xlabel("sigma (= t / num_train_timesteps)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()  # denoising progresses from high sigma to low sigma
        if len(sample_ids) <= 8:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"shortcut vs full-probe divergence (feedback_mode={feedback_mode})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info("Plot saved to %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(cfg: DictConfig) -> None:
    model_dir = Path(cfg.model_dir) if cfg.model_dir else None
    if model_dir is None:
        raise ValueError("model_dir is required for the shortcut diagnostic")
    base_model = cfg.base_model

    # --- Diagnostic-specific knobs (with sensible defaults) --------------
    diagnose_cfg = cfg.get("diagnose", {}) or {}
    num_samples = int(diagnose_cfg.get("num_samples", 4))
    output_subdir = str(diagnose_cfg.get("output_subdir", "diagnose"))

    output_dir = Path(cfg.output_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_cfg = cfg.generation
    if gen_cfg.guidance_scale > 1.0:
        raise ValueError(
            f"Diagnostic script does not support CFG (guidance_scale={gen_cfg.guidance_scale}). "
            "Run with generation.guidance_scale=1.0."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Load ControlNet (must be a FlowBender checkpoint) ----------------------
    ctrl_src = model_dir / "controlnet" if (model_dir / "controlnet").is_dir() else model_dir
    controlnet = SD3ControlNetModelFlowBender.from_pretrained(str(ctrl_src), torch_dtype=weight_dtype)
    feedback_mode = controlnet.config.get("feedback_mode", "vanilla")
    feedback_variant = controlnet.config.get("feedback_variant", None)
    gradient_cond_kwargs_raw = controlnet.config.get("gradient_cond_kwargs", {}) or {}
    gradient_cond_kwargs = (
        OmegaConf.to_container(gradient_cond_kwargs_raw, resolve=True)
        if OmegaConf.is_config(gradient_cond_kwargs_raw)
        else dict(gradient_cond_kwargs_raw)
    )

    if feedback_mode not in ("gradient", "residual"):
        raise ValueError(
            f"Diagnostic supports only 'gradient' and 'residual' feedback modes; "
            f"controlnet config has feedback_mode={feedback_mode!r}."
        )
    if feedback_mode == "gradient" and feedback_variant == "rescale_noise_norm":
        raise ValueError(
            "feedback_variant='rescale_noise_norm' needs a live model_pred and is incompatible "
            "with the shortcut (cached z0_hat) path, so the diagnostic cannot compare both sides."
        )

    # --- Task resolution (same logic as evaluate.py) ----------------------
    forward_process_type, fp_kwargs, cond_dir = _resolve_task(cfg, controlnet)
    cfg.data.condition_type = forward_process_type
    cfg.data.cond_dir = str(cond_dir)
    logger.info("feedback_mode=%s  task=%s  cond_dir=%s", feedback_mode, forward_process_type, cond_dir)

    forward_process = FORWARD_PROCESS_REGISTRY[forward_process_type](**fp_kwargs)
    forward_process.to(device).eval()
    forward_process.requires_grad_(False)

    # --- Load transformer + VAE -------------------------------------------
    if (model_dir / "transformer").is_dir():
        transformer = SD3Transformer2DModel.from_pretrained(model_dir / "transformer", torch_dtype=weight_dtype)
    else:
        transformer = SD3Transformer2DModel.from_pretrained(
            base_model, subfolder="transformer", torch_dtype=weight_dtype
        )
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=weight_dtype)

    # --- Build pipeline ---------------------------------------------------
    # The diagnostic always needs the FlowBender controlnet pipeline (it pokes at
    # `get_residual_gradient`/`get_residual_condition` directly), so hard-code
    # the class instead of reading `pipeline._target_` from the eval config —
    # that field is a `???` placeholder meant for `eval_baseline=...` overrides.
    pipe = StableDiffusion3FlowBenderPipeline.from_pretrained(
        base_model,
        transformer=transformer,
        vae=vae,
        controlnet=controlnet,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        torch_dtype=weight_dtype,
    )
    pipe._forward_process = forward_process
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # --- Dataset ----------------------------------------------------------
    dataset = PreprocessedControlNetDataset(
        img_dir=str(cfg.data.img_dir),
        cond_dir=str(cond_dir),
        text_embeds_dir=str(cfg.data.text_embeds_dir),
        prompt_dir=str(cfg.data.prompt_dir) if cfg.data.prompt_dir else None,
        resolution=gen_cfg.resolution,
        condition_type=forward_process_type,
        scale_factor=fp_kwargs.get("scale_factor", None),
    )
    dataset = Subset(dataset, list(range(min(num_samples, len(dataset)))))
    dataloader = DataLoader(dataset, batch_size=gen_cfg.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    target_seq_len = 77 + gen_cfg.max_sequence_length

    # --- Run diagnostic loop ---------------------------------------------
    all_metrics: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Diagnosing")):
        seed = gen_cfg.seed + batch_idx * gen_cfg.batch_size
        with attention_backend("_native_flash"):
            batch_metrics = _diagnose_batch(
                pipe,
                batch,
                num_inference_steps=gen_cfg.num_inference_steps,
                seed=seed,
                resolution=gen_cfg.resolution,
                weight_dtype=weight_dtype,
                device=device,
                feedback_mode=feedback_mode,
                feedback_variant=feedback_variant,
                gradient_cond_kwargs=gradient_cond_kwargs,
                target_seq_len=target_seq_len,
            )
        all_metrics.extend(batch_metrics)

    # --- Persist outputs --------------------------------------------------
    metrics_path = output_dir / "per_step_metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2))
    logger.info("Saved %d metric rows to %s", len(all_metrics), metrics_path)

    plot_path = output_dir / "shortcut_divergence.png"
    _plot_metrics(all_metrics, feedback_mode, plot_path)

    # Brief console summary grouped by step index
    by_step: dict[int, list[dict[str, Any]]] = {}
    for m in all_metrics:
        by_step.setdefault(m["step"], []).append(m)
    print(f"\nSummary (feedback_mode={feedback_mode}, N={num_samples} samples):")
    summary_keys = ["cos_grad"] if feedback_mode == "gradient" else ["l1_residual", "psnr_residual"]
    header = f"  {'step':>5}  {'sigma':>7}"
    for k in summary_keys:
        header += f"  {k:>16}"
    header += f"  {'z0_cos':>14}"
    print(header)
    for step in sorted(by_step):
        rows = by_step[step]
        z0s = [r["z0_cos"] for r in rows]
        line = f"  {step:>5d}  {rows[0]['sigma']:>7.3f}"
        for k in summary_keys:
            vals = [r[k] for r in rows if k in r]
            if vals:
                line += f"  {np.mean(vals):>8.3f}+/-{np.std(vals):>5.3f}"
            else:
                line += f"  {'-':>16}"
        line += f"  {np.mean(z0s):>6.3f}+/-{np.std(z0s):>5.3f}"
        print(line)


if __name__ == "__main__":
    main()
