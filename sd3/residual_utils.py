import logging
from typing import Literal

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel
from einops import reduce
from torch import nn

from sd3.depth.forward_process import DepthForwardProcess
from sd3.edge import EdgeForwardProcess
from sd3.jpeg_restoration import JpegRestorationForwardProcess
from sd3.super_resolution import SuperResolutionForwardProcess
from sd3.vis import ResidualDebugInfo

logger = logging.getLogger(__name__)


FORWARD_PROCESS_REGISTRY = {
    "depth": DepthForwardProcess,
    "super_resolution": SuperResolutionForwardProcess,
    "jpeg_restoration": JpegRestorationForwardProcess,
    "edge": EdgeForwardProcess,
}


def vae_encode(vae: AutoencoderKL, img: torch.Tensor) -> torch.Tensor:
    enc_cond_image = vae.encode(img.to(vae.dtype)).latent_dist.sample()
    return (enc_cond_image - vae.config.shift_factor) * vae.config.scaling_factor


def _predict_z0_hat(
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
) -> torch.Tensor:
    """Derive the latent-space x_0 estimate from noisy latents and predicted velocity.

    Flow matching: z_t = (1 - sigma) * x_0 + sigma * noise
    Model predicts v = noise - x_0, so x_0 = z_t - sigma * v

    Returns:
        Tensor with same shape as *noisy_latents* (latent space, before VAE decode).
    """
    sigmas = timestep.float().reshape(-1) / scheduler.config.num_train_timesteps
    while len(sigmas.shape) < noisy_latents.ndim:
        sigmas = sigmas.unsqueeze(-1)
    sigmas = sigmas.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    return noisy_latents - sigmas * noise_pred


def _decode_latent(z0_hat: torch.Tensor, vae: AutoencoderKL) -> torch.Tensor:
    """VAE-decode a latent-space tensor to pixel space.

    Returns:
        Tensor of shape [B, 3, H, W], range approximately [-1, 1].
    """
    z0_hat_unscaled = z0_hat / vae.config.scaling_factor + vae.config.shift_factor
    return vae.decode(z0_hat_unscaled.to(vae.dtype).contiguous(), return_dict=False)[0]


def _decode_z0_hat(
    noisy_latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: AutoencoderKL,
) -> torch.Tensor:
    """Derive x_0 estimate and VAE-decode to pixel space.

    Returns:
        Tensor of shape [B, 3, H, W], range: [-1, 1]
    """
    z0_hat = _predict_z0_hat(noisy_latents, noise_pred, timestep, scheduler)
    return _decode_latent(z0_hat, vae)


@torch.no_grad()
def get_residual_condition(
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    transformer: SD3Transformer2DModel,
    vae: AutoencoderKL,
    scheduler: FlowMatchEulerDiscreteScheduler,
    raw_cond_img: torch.Tensor,
    forward_process: nn.Module,
    enc_cond_image: torch.Tensor | None = None,
    controlnet: SD3ControlNetModel | None = None,
    variant: str | None = None,
    cached_z0_hat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ResidualDebugInfo, torch.Tensor | None]:
    """Calculates z0_hat, decodes, runs forward process, returns residual + debug info.

    Args:
        variant: ``"scale_vae_range"`` normalises the residual per-sample to
            [-1, 1]; ``"predicted_depth"`` returns the predicted depth (mapped
            to [-1, 1]) instead of the residual.
        cached_z0_hat: Optional pre-computed ``z0_hat`` (e.g. from the previous
            denoising step's guided prediction).  When provided, skips the
            internal ControlNet + transformer probe used to derive ``z0_hat``
            and uses this tensor instead. See the Euler-step identity
            ``z0_hat_{t+1} = z_t - sigma_t * v_{t+1}``.

    Returns:
        ``(residual, debug, unguided_pred)`` where ``unguided_pred`` is the
        detached probe ``model_pred`` (positive text + zero feedback) — the
        natural CFG baseline for the feedback axis. ``None`` when
        ``cached_z0_hat`` short-circuits the probe.
    """
    dtype = noisy_latents.dtype
    unguided_pred: torch.Tensor | None = None

    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0).expand(noisy_latents.shape[0])

    if cached_z0_hat is not None:
        z0_hat = cached_z0_hat
    elif controlnet is not None:
        # --- INFORMED GUESS: encode null condition in latent space ---
        if enc_cond_image is None:
            enc_cond_image = vae_encode(vae, raw_cond_img)
        null_condition = torch.cat([enc_cond_image.to(dtype), torch.zeros_like(enc_cond_image)], dim=1)

        using_encoder_hidden_states = controlnet.context_embedder is not None
        control_block_samples = controlnet(
            hidden_states=noisy_latents,
            timestep=timesteps,
            # encoder_hidden_states is unused in SD3ControlNets
            # https://github.com/huggingface/diffusers/blob/d791c5c024014784df4a8dac2601e19fb4d300fc/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py#L1143
            encoder_hidden_states=prompt_embeds if using_encoder_hidden_states else None,
            pooled_projections=pooled_prompt_embeds,
            controlnet_cond=null_condition,
            return_dict=False,
        )[0]
        control_block_samples = [s.to(dtype=dtype) for s in control_block_samples]

        model_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            block_controlnet_hidden_states=control_block_samples,
            return_dict=False,
        )[0]
        z0_hat = _predict_z0_hat(noisy_latents, model_pred, timesteps, scheduler)
        unguided_pred = model_pred.detach().to(dtype)
    else:
        # --- BLIND GUESS ---
        model_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        z0_hat = _predict_z0_hat(noisy_latents, model_pred, timesteps, scheduler)
        unguided_pred = model_pred.detach().to(dtype)

    decoded = _decode_latent(z0_hat, vae)
    forward_model_inp = decoded * 0.5 + 0.5  # Range: [-1, 1] -> [0, 1]
    residual, forward_pred = forward_process.get_residual(cur_img=forward_model_inp, cond=raw_cond_img)

    residual = _apply_residual_variant(residual, forward_pred, variant)

    debug = ResidualDebugInfo(
        z0_hat_decoded=decoded.detach().cpu(),
        forward_pred=forward_pred.detach().cpu(),
        condition=raw_cond_img.detach().cpu(),
        residual=residual.detach().cpu(),
        z0_hat=z0_hat.detach().cpu(),
    )
    return residual.to(dtype), debug, unguided_pred


def _rescale_gradient(
    dLdLatents: torch.Tensor,
    model_pred: torch.Tensor,
    rescale: Literal["rescale_noise_norm", "rescale_std_normal", "rescale_std_normal_per_channel", "rescale_std_only"]
    | None,
) -> torch.Tensor:
    """Rescale gradient signal according to the chosen strategy.

    Args:
        dLdLatents: Raw gradient of shape [B, C, H, W].
        model_pred: Transformer noise prediction of same spatial shape.
        rescale: Scaling strategy
            - ``"rescale_noise_norm"`` matches per-sample L2 norm of *model_pred*;
            - ``"rescale_std_normal"`` standardizes to zero-mean unit-variance per sample;
            - ``"rescale_std_normal_per_channel"`` standardizes to zero-mean unit-variance per channel.
            - ``"rescale_std_only"`` divides by per-sample std (preserves the gradient's mean).
    """
    if rescale is None:
        return dLdLatents

    if rescale == "rescale_noise_norm":
        flat = dLdLatents.flatten(1)
        pred_flat = model_pred.detach().flatten(1)
        grad_norm_val = flat.norm(dim=1).clamp(min=1e-8)
        pred_norm = pred_flat.norm(dim=1)
        scale = (pred_norm / grad_norm_val).view(-1, 1, 1, 1)
        return dLdLatents * scale
    elif rescale == "rescale_std_normal":
        std, mean = torch.std_mean(dLdLatents, dim=[1, 2, 3], keepdim=True)
        return (dLdLatents - mean) / std.clamp(min=1e-8)
    elif rescale == "rescale_std_normal_per_channel":
        std, mean = torch.std_mean(dLdLatents, dim=1, keepdim=True)
        return (dLdLatents - mean) / std.clamp(min=1e-8)
    elif rescale == "rescale_std_only":
        std = dLdLatents.std(dim=[1, 2, 3], keepdim=True).clamp(min=1e-8)
        return dLdLatents / std
    else:
        raise ValueError(f"Unknown gradient rescale mode: {rescale}")


def get_residual_gradient(
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    transformer: SD3Transformer2DModel,
    vae: AutoencoderKL,
    scheduler: FlowMatchEulerDiscreteScheduler,
    raw_cond_img: torch.Tensor,
    forward_process: nn.Module,
    enc_cond_image: torch.Tensor | None = None,
    controlnet: SD3ControlNetModel | None = None,
    rescale: str | None = None,
    loss_func: Literal["l1", "mse", "nll"] = "l1",
    loss_reduction: Literal["mean", "sum"] = "mean",
    skip_denoiser_grad: bool = False,
    cached_z0_hat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ResidualDebugInfo, torch.Tensor | None]:
    """Calculates z0_hat, decodes, runs forward process, returns gradient + debug info.

    Args:
        skip_denoiser_grad: When ``True`` the gradient is taken w.r.t. the predicted clean latent ``z0_hat`` instead of
            the noisy latent ``z_t``, avoiding back-propagation through the denoiser.  This is an approximation
            justified by the flow-matching identity ``dz0/dz_t = I - sigma * dv/dz_t ≈ I``.
            Also see https://arxiv.org/abs/2412.00100.
        cached_z0_hat: Optional pre-computed ``z0_hat`` (e.g. from the previous denoising step's guided prediction).
            When provided, skips the internal ControlNet + transformer probe and uses this tensor directly as the
            leaf for the gradient computation.  Implies ``skip_denoiser_grad=True`` behaviour.  Incompatible with
            ``rescale='rescale_noise_norm'`` because that scaling needs a live ``model_pred``.

    Returns:
        ``(dLdLatents, debug, unguided_pred)`` where ``unguided_pred`` is the
        detached probe ``model_pred`` (positive text + zero feedback) — the
        natural CFG baseline for the feedback axis. ``None`` when
        ``cached_z0_hat`` short-circuits the probe.
    """
    if cached_z0_hat is not None and rescale == "rescale_noise_norm":
        raise ValueError(
            "rescale='rescale_noise_norm' requires a live model_pred; it is incompatible with cached_z0_hat."
        )

    device = noisy_latents.device
    dtype = noisy_latents.dtype

    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0).expand(noisy_latents.shape[0])

    # When a cached z0_hat is supplied we skip the probe entirely — neither the ControlNet nor the
    # transformer are needed to derive z0_hat, which halves per-step denoiser activations.
    need_probe = cached_z0_hat is None
    model_pred: torch.Tensor | None = None
    unguided_pred: torch.Tensor | None = None

    control_block_samples = None
    if need_probe and controlnet is not None:
        if enc_cond_image is None:
            enc_cond_image = vae_encode(vae, raw_cond_img)

        using_encoder_hidden_states = controlnet.context_embedder is not None
        control_block_samples = controlnet(
            hidden_states=noisy_latents,
            timestep=timesteps,
            # encoder_hidden_states is unused in SD3ControlNets
            # https://github.com/huggingface/diffusers/blob/d791c5c024014784df4a8dac2601e19fb4d300fc/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py#L1143
            encoder_hidden_states=prompt_embeds if using_encoder_hidden_states else None,
            pooled_projections=pooled_prompt_embeds,
            controlnet_cond=enc_cond_image,
            return_dict=False,
        )[0]
        control_block_samples = [s.to(dtype=dtype) for s in control_block_samples]

    # A cached z0_hat always routes through the "gradient w.r.t. z0_hat" branch — back-propagating through the
    # denoiser is impossible without a live transformer forward.
    take_grad_wrt_z0_hat = skip_denoiser_grad or cached_z0_hat is not None

    if take_grad_wrt_z0_hat:
        if cached_z0_hat is None:
            with torch.no_grad():
                model_pred = transformer(
                    hidden_states=torch.cat([noisy_latents, torch.zeros_like(noisy_latents)], dim=1),
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_samples,
                    return_dict=False,
                )[0]
            z0_hat = _predict_z0_hat(noisy_latents, model_pred, timesteps, scheduler)
            unguided_pred = model_pred.detach().to(dtype)
        else:
            z0_hat = cached_z0_hat

        # Leaf variable for the local graph: gradient is taken w.r.t. z0_hat and used as a proxy for dL/dz_t
        # (flow-matching identity dz0/dz_t = I - sigma * dv/dz_t ≈ I).
        z0_hat = z0_hat.detach().requires_grad_(True)

        with torch.enable_grad():
            decoded = _decode_latent(z0_hat, vae)
            forward_model_inp = decoded * 0.5 + 0.5
            residual, forward_pred = forward_process.get_residual(cur_img=forward_model_inp, cond=raw_cond_img)

            # Gradient checkpointing will move models to the CPU but we need them on the GPU here to backprop through
            vae.to(device)
            forward_process.to(device)

            loss = 1000 * _compute_loss(residual.float(), loss_func, loss_reduction)
            dLdLatents = torch.autograd.grad(loss, z0_hat)[0]
            if dLdLatents.abs().mean() < 1e-8:
                logger.warning("dLdLatents underflow")

            if not torch.isfinite(dLdLatents).all() or torch.isnan(dLdLatents).any():
                logger.warning("dLdLatents is not finite or nan, setting to 0")
                torch.nan_to_num_(dLdLatents, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        noisy_latents_for_grad = noisy_latents.detach().requires_grad_(True)

        with torch.enable_grad():
            model_pred = transformer(
                hidden_states=torch.cat([noisy_latents_for_grad, torch.zeros_like(noisy_latents_for_grad)], dim=1),
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                block_controlnet_hidden_states=control_block_samples,
                return_dict=False,
            )[0]

            z0_hat = _predict_z0_hat(noisy_latents_for_grad, model_pred, timesteps, scheduler)
            decoded = _decode_latent(z0_hat, vae)
            forward_model_inp = decoded * 0.5 + 0.5
            residual, forward_pred = forward_process.get_residual(cur_img=forward_model_inp, cond=raw_cond_img)

            # Gradient checkpointing will move models to the CPU but we need them on the GPU here to backprop through
            transformer.to(device)
            vae.to(device)
            forward_process.to(device)

            loss = 1000 * _compute_loss(residual.float(), loss_func, loss_reduction)
            dLdLatents = torch.autograd.grad(loss, noisy_latents_for_grad)[0]
            if dLdLatents.abs().mean() < 1e-8:
                logger.warning("dLdLatents underflow")

            if not torch.isfinite(dLdLatents).all() or torch.isnan(dLdLatents).any():
                logger.warning("dLdLatents is not finite or nan, setting to 0")
                torch.nan_to_num_(dLdLatents, nan=0.0, posinf=0.0, neginf=0.0)
        unguided_pred = model_pred.detach().to(dtype)

    logger.debug(f"dLdLatents: {dLdLatents.shape}, {dLdLatents.min()}, {dLdLatents.max()}")

    debug = ResidualDebugInfo(
        z0_hat_decoded=decoded.detach().cpu(),
        forward_pred=forward_pred.detach().cpu(),
        condition=raw_cond_img.detach().cpu(),
        residual=residual.detach().cpu(),
        z0_hat=z0_hat.detach().cpu(),
    )

    dLdLatents = _rescale_gradient(dLdLatents, model_pred, rescale)
    # Return in the caller's latent dtype (matches noisy_latents); when upcast_vae is on,
    # vae.dtype != noisy_latents.dtype and mixing would upcast the transformer input to fp32.
    return dLdLatents.detach().to(dtype), debug, unguided_pred


def _compute_loss(
    residual: torch.Tensor, loss_func: Literal["l1", "mse", "nll"], loss_reduction: Literal["mean", "sum"]
) -> torch.Tensor:
    if loss_func == "l1":
        return reduce(residual.abs(), "... -> ", reduction=loss_reduction)
    elif loss_func == "mse":
        return reduce(residual**2, "... -> ", reduction=loss_reduction)
    elif loss_func == "nll":
        return reduce((1 - residual).clamp(min=1e-8).log().neg(), "... -> ", reduction=loss_reduction)
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")


def _apply_residual_variant(residual: torch.Tensor, forward_pred: torch.Tensor, variant: str | None) -> torch.Tensor:
    """Post-process the raw residual according to the chosen variant."""
    if variant is None:
        return residual
    if variant == "scale_vae_range":
        max_abs = residual.abs().amax(dim=[1, 2, 3], keepdim=True).clamp(min=1e-8)
        return residual / max_abs
    if variant == "scale_vae_range_[-1,1]":
        min_val = residual.amin(dim=[1, 2, 3], keepdim=True).clamp(min=1e-8)
        max_val = residual.amax(dim=[1, 2, 3], keepdim=True).clamp(min=1e-8)
        return (residual - min_val) / (max_val - min_val) * 2 - 1
    if variant == "predicted_depth":
        return forward_pred.expand_as(residual) * 2 - 1
    raise ValueError(f"Unknown residual variant: {variant}")


def get_residual_and_gradient_condition(
    noisy_latents: torch.Tensor,
    timesteps: torch.LongTensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    transformer: SD3Transformer2DModel,
    vae: AutoencoderKL,
    scheduler: FlowMatchEulerDiscreteScheduler,
    raw_cond_img: torch.Tensor,
    forward_process: nn.Module,
    enc_cond_image: torch.Tensor | None = None,
    controlnet: SD3ControlNetModel | None = None,
    variant: str | None = None,
    rescale: str | None = None,
    loss_func: Literal["l1", "mse", "identity"] = "l1",
    loss_reduction: Literal["mean", "sum"] = "mean",
    skip_denoiser_grad: bool = False,
    cached_z0_hat: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, ResidualDebugInfo, torch.Tensor | None]:
    """Compute both the residual condition and the gradient condition in a single forward pass.

    Shares the transformer → z0 → VAE decode → forward-process computation
    between the two signals, avoiding the redundant second pass that would
    occur when calling ``get_residual_condition`` and ``get_residual_gradient``
    separately.

    Args:
        cached_z0_hat: Optional pre-computed ``z0_hat`` (e.g. from the previous denoising step's guided
            prediction). When provided, skips the internal ControlNet + transformer probe and uses this
            tensor directly as the leaf for the gradient computation. Implies ``skip_denoiser_grad=True``
            behaviour. Incompatible with ``rescale='rescale_noise_norm'``.

    Returns:
        ``(residual_condition, dLdLatents, debug_info, unguided_pred)`` where
        ``unguided_pred`` is the detached probe ``model_pred`` (positive text +
        zero feedback) — the natural CFG baseline for the feedback axis.
        ``None`` when ``cached_z0_hat`` short-circuits the probe.
    """
    if cached_z0_hat is not None and rescale == "rescale_noise_norm":
        raise ValueError(
            "rescale='rescale_noise_norm' requires a live model_pred; it is incompatible with cached_z0_hat."
        )

    device = noisy_latents.device
    dtype = noisy_latents.dtype

    if timesteps.dim() == 0:
        timesteps = timesteps.unsqueeze(0).expand(noisy_latents.shape[0])

    # When a cached z0_hat is supplied we skip the probe entirely — neither the ControlNet nor the
    # transformer are needed to derive z0_hat.
    need_probe = cached_z0_hat is None
    model_pred: torch.Tensor | None = None
    unguided_pred: torch.Tensor | None = None

    # ControlNet forward — use null residual condition (zeros in extra channels)
    control_block_samples = None
    if need_probe and controlnet is not None:
        if enc_cond_image is None:
            enc_cond_image = vae_encode(vae, raw_cond_img)
        null_condition = torch.cat([enc_cond_image.to(dtype), torch.zeros_like(enc_cond_image)], dim=1)

        using_encoder_hidden_states = controlnet.context_embedder is not None
        control_block_samples = controlnet(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds if using_encoder_hidden_states else None,
            pooled_projections=pooled_prompt_embeds,
            controlnet_cond=null_condition,
            return_dict=False,
        )[0]
        control_block_samples = [s.to(dtype=dtype) for s in control_block_samples]

    # A cached z0_hat always routes through the "gradient w.r.t. z0_hat" branch.
    take_grad_wrt_z0_hat = skip_denoiser_grad or cached_z0_hat is not None

    if take_grad_wrt_z0_hat:
        if cached_z0_hat is None:
            # Transformer input has widened channels (gradient half is zeros)
            transformer_input = torch.cat([noisy_latents, torch.zeros_like(noisy_latents)], dim=1)
            with torch.no_grad():
                model_pred = transformer(
                    hidden_states=transformer_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_samples,
                    return_dict=False,
                )[0]
            z0_hat = _predict_z0_hat(noisy_latents, model_pred, timesteps, scheduler)
            unguided_pred = model_pred.detach().to(dtype)
        else:
            z0_hat = cached_z0_hat

        # Leaf variable for the local graph: gradient is taken w.r.t. z0_hat and used as a proxy for dL/dz_t
        # (flow-matching identity dz0/dz_t = I - sigma * dv/dz_t ≈ I).
        z0_hat = z0_hat.detach().requires_grad_(True)

        with torch.enable_grad():
            decoded = _decode_latent(z0_hat, vae)
            forward_model_inp = decoded * 0.5 + 0.5
            residual, forward_pred = forward_process.get_residual(cur_img=forward_model_inp, cond=raw_cond_img)

            vae.to(device)
            forward_process.to(device)

            loss = 1000 * _compute_loss(residual.float(), loss_func, loss_reduction)
            dLdLatents = torch.autograd.grad(loss, z0_hat)[0]
            if dLdLatents.abs().mean() < 1e-8:
                logger.warning("dLdLatents underflow")

            if not torch.isfinite(dLdLatents).all() or torch.isnan(dLdLatents).any():
                logger.warning("dLdLatents is not finite or nan, setting to 0")
                torch.nan_to_num_(dLdLatents, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        noisy_latents_for_grad = noisy_latents.detach().requires_grad_(True)
        transformer_input = torch.cat([noisy_latents_for_grad, torch.zeros_like(noisy_latents_for_grad)], dim=1)

        with torch.enable_grad():
            model_pred = transformer(
                hidden_states=transformer_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                block_controlnet_hidden_states=control_block_samples,
                return_dict=False,
            )[0]

            z0_hat = _predict_z0_hat(noisy_latents_for_grad, model_pred, timesteps, scheduler)
            decoded = _decode_latent(z0_hat, vae)
            forward_model_inp = decoded * 0.5 + 0.5
            residual, forward_pred = forward_process.get_residual(cur_img=forward_model_inp, cond=raw_cond_img)

            transformer.to(device)
            vae.to(device)
            forward_process.to(device)

            loss = 1000 * _compute_loss(residual.float(), loss_func, loss_reduction)
            dLdLatents = torch.autograd.grad(loss, noisy_latents_for_grad)[0]
            if dLdLatents.abs().mean() < 1e-8:
                logger.warning("dLdLatents underflow")

            if not torch.isfinite(dLdLatents).all() or torch.isnan(dLdLatents).any():
                logger.warning("dLdLatents is not finite or nan, setting to 0")
                torch.nan_to_num_(dLdLatents, nan=0.0, posinf=0.0, neginf=0.0)
        unguided_pred = model_pred.detach().to(dtype)

    # Residual condition (detached, with variant post-processing)
    residual_cond = _apply_residual_variant(residual.detach(), forward_pred.detach(), variant)

    debug = ResidualDebugInfo(
        z0_hat_decoded=decoded.detach().cpu(),
        forward_pred=forward_pred.detach().cpu(),
        condition=raw_cond_img.detach().cpu(),
        residual=residual.detach().cpu(),
        z0_hat=z0_hat.detach().cpu(),
    )

    dLdLatents = _rescale_gradient(dLdLatents, model_pred, rescale)
    # dLdLatents is cat'd into the latent-space transformer input and must match its dtype;
    # residual_cond is pixel-space and will be re-cast by vae_encode, but we use the same dtype
    # for consistency. See note in get_residual_gradient.
    return residual_cond.to(dtype), dLdLatents.detach().to(dtype), debug, unguided_pred


def flowchef_steer_step(
    latents: torch.Tensor,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: AutoencoderKL,
    forward_process: nn.Module,
    control_image: torch.Tensor,
    learning_rate: float,
    num_steps: int = 1,
    loss_func: str = "mse",
    loss_reduction: str = "mean",
    rescale: str | None = None,
    loss_scale: float = 1000.0,
    **kwargs,
) -> tuple[torch.Tensor, ResidualDebugInfo]:
    """FlowChef gradient-steering (Algorithm 1, lines 6-9).

    See: https://github.com/FlowChef/FlowChef/blob/9b705fcd91cb3cb8bf9d485ef6badc539201651c/src/fluxcombined.py#L916-L929
    """
    if num_steps == 0:
        return latents.detach(), None

    with torch.enable_grad():
        latents_opt = latents.detach().clone().requires_grad_(True)
        for step in range(num_steps):
            z0_hat = _predict_z0_hat(latents_opt, noise_pred, timestep, scheduler)
            decoded = _decode_latent(z0_hat, vae)
            fwd_inp = decoded * 0.5 + 0.5  # [-1, 1] -> [0, 1]
            residual, fwd_pred = forward_process.get_residual(cur_img=fwd_inp, cond=control_image)
            loss = loss_scale * _compute_loss(residual, loss_func, loss_reduction)  # Scale loss to prevent underflow

            grad = torch.autograd.grad(loss, latents_opt, create_graph=False)[0] / loss_scale
            grad_rescaled = _rescale_gradient(grad, noise_pred, rescale)

            logger.debug(
                "FlowChef steer [%d/%d]: loss=%.6f, |grad|=%.2e, |grad_rescaled|=%.2e",
                step + 1,
                num_steps,
                loss.item(),
                grad.abs().mean().item(),
                grad_rescaled.abs().mean().item(),
            )
            latents_opt = latents_opt - learning_rate * grad_rescaled

    debug = ResidualDebugInfo(
        z0_hat_decoded=decoded.detach().cpu(),
        forward_pred=fwd_pred.detach().cpu(),
        condition=control_image.detach().cpu(),
        residual=residual.detach().cpu(),
    )
    return latents_opt.detach().clone(), debug


def compute_reward_loss(
    noisy_model_input: torch.Tensor,
    noise_pred: torch.Tensor,
    timesteps: torch.Tensor,
    sigmas: torch.Tensor,
    raw_cond_img: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: AutoencoderKL,
    forward_process: nn.Module,
    sigma_threshold: float = 0.2,
    loss_func: Literal["l1", "mse", "nll"] = "mse",
    loss_reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor | None:
    """ControlNet++ reward consistency loss (arXiv 2404.07987, Eq. 8-9).

    For samples whose sigma <= sigma_threshold, decode the single-step z0
    estimate, run the forward process, and return the consistency loss.
    Returns None when no samples in the batch qualify.
    """
    mask = sigmas.view(-1) <= sigma_threshold
    if not mask.any():
        return None

    z0_hat = _predict_z0_hat(noisy_model_input, noise_pred, timesteps, scheduler)

    # Only decode the qualifying subset to save compute
    z0_sub = z0_hat[mask]
    cond_sub = raw_cond_img[mask]

    decoded = _decode_latent(z0_sub, vae)
    fwd_inp = decoded * 0.5 + 0.5  # [-1, 1] -> [0, 1]
    residual, _ = forward_process.get_residual(cur_img=fwd_inp, cond=cond_sub)
    return _compute_loss(residual, loss_func, loss_reduction)
