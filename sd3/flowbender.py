"""FlowBender SD3.5 ControlNet model and pipeline."""

import logging
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.pipelines.controlnet_sd3 import StableDiffusion3ControlNetPipeline
from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet import retrieve_timesteps
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput

from sd3.residual_utils import (
    FORWARD_PROCESS_REGISTRY,
    _decode_z0_hat,
    _predict_z0_hat,
    flowchef_steer_step,
    get_residual_and_gradient_condition,
    get_residual_condition,
    get_residual_gradient,
    vae_encode,
)
from sd3.vis import ResidualDebugInfo, StepDebugInfo

logger = logging.getLogger(__name__)


class StableDiffusion3FlowBenderPipeline(StableDiffusion3ControlNetPipeline):
    """Denoise with conditioning on the residual w.r.t. conditioning image.

    Computed by applying the forward process; optionally applies FlowChef steering.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: PipelineImageInput = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        controlnet_pooled_projections: Optional[torch.FloatTensor] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        visualize_z0: bool = False,
        visualize_z0_every_n: int = 1,
        # Optional FlowChef steering (applied on top of ControlNet)
        flowchef_kwargs: dict[str, Any] | None = None,
        # Fraction of the *last* denoising iterations (small t) that reuse the previous step's
        # post-Euler z0_hat instead of running a fresh "probe" transformer+ControlNet pass.
        # `0.0` keeps today's 2N-eval baseline; `1.0` gives N+1 evals (only step 0 runs a full probe
        # to seed the cache). Applies only when feedback_mode != "vanilla".
        shortcut_fraction: float = 0.0,
        # Orthogonal feedback-axis CFG. Reuses the helper's positive-text + zero-feedback `model_pred`
        # (already computed by the probe) as the unconditional baseline:
        #   noise_pred <- unguided_pred + feedback_guidance_scale * (noise_pred - unguided_pred)
        # `1.0` is a no-op; `>1.0` amplifies the feedback signal. Only applies when feedback_mode !=
        # "vanilla". Silently no-ops on shortcut steps (probe skipped → no unguided_pred available).
        feedback_guidance_scale: float = 1.0,
    ):
        if not 0.0 <= shortcut_fraction <= 1.0:
            raise ValueError(f"shortcut_fraction must be in [0, 1]; got {shortcut_fraction}")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        controlnet_config = (
            self.controlnet.config
            if isinstance(self.controlnet, SD3ControlNetModel)
            else self.controlnet.nets[0].config
        )
        controlnet_ref = self.controlnet if isinstance(self.controlnet, SD3ControlNetModel) else self.controlnet.nets[0]
        feedback_mode = controlnet_ref.config.get("feedback_mode", "vanilla")
        feedback_variant = controlnet_ref.config.get("feedback_variant", None)
        gradient_cond_kwargs = controlnet_ref.config.get("gradient_cond_kwargs", {})

        # Honour an externally-injected `_forward_process` (e.g. set by
        # `evaluate.py` so FlowChef can steer vanilla checkpoints) regardless
        # of feedback_mode; only construct one from the controlnet config when
        # feedback_mode != "vanilla" and nothing was injected.
        forward_process = getattr(self, "_forward_process", None)
        if feedback_mode != "vanilla" and forward_process is None:
            fp_type = controlnet_ref.config.get("forward_process_type", "depth")
            fp_kwargs = controlnet_ref.config.get("forward_process_kwargs", {})
            fp_cls = FORWARD_PROCESS_REGISTRY[fp_type]
            self._forward_process = fp_cls(**fp_kwargs)
            self._forward_process.requires_grad_(False)
            forward_process = self._forward_process

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(self.controlnet, SD3MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        if forward_process is not None:
            forward_process.to(device)

        (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = (
            self.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_3=prompt_3,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                negative_prompt_3=negative_prompt_3,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                clip_skip=self.clip_skip,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

        # 3. Prepare control image
        if controlnet_config.force_zeros_for_pooled_projection:
            # instantx sd3 controlnet does not apply shift factor
            vae_shift_factor = 0
        else:
            vae_shift_factor = self.vae.config.shift_factor

        if isinstance(self.controlnet, SD3ControlNetModel):
            # This normalizes control_image to the range [-1, 1] if it is a PIL image
            # If it is a tensor it is not normalized (just expanded for potential CFG)
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=torch.float32,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                guess_mode=False,
            )
            orig_control_image = control_image  # Save to compute residual later
            height, width = control_image.shape[-2:]

            control_image_latents = self.vae.encode(control_image.to(dtype)).latent_dist.sample()
            control_image_latents = (control_image_latents - vae_shift_factor) * self.vae.config.scaling_factor
        elif isinstance(self.controlnet, SD3MultiControlNetModel):
            raise NotImplementedError("MultiControlNet is not supported yet")
        else:
            assert False

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Shortcut scheduling: run the last `num_shortcut_steps` iterations (small t) with a cached
        # z0_hat from the previous step instead of re-deriving it via a transformer+ControlNet probe.
        # The clamp `min(..., N - 1)` guarantees step 0 always runs a full probe to seed the cache.
        _N = len(timesteps)
        num_shortcut_steps = min(round(shortcut_fraction * _N), _N - 1) if _N > 0 else 0
        first_shortcut_idx = _N - num_shortcut_steps  # equals _N when feature disabled
        if num_shortcut_steps > 0 and feedback_mode != "vanilla":
            logger.info(
                "Shortcut sampling enabled: %d/%d iterations (fraction=%.3f, first_shortcut_idx=%d) "
                "will reuse the previous step's z0_hat (feedback_mode=%s).",
                num_shortcut_steps,
                _N,
                shortcut_fraction,
                first_shortcut_idx,
                feedback_mode,
            )

        # 5. Prepare latent variables — always use VAE latent channels (16),
        # independent of whether the transformer's input proj was widened for
        # gradient conditioning.
        num_channels_latents = self.vae.config.latent_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(self.controlnet, SD3ControlNetModel) else keeps)

        if controlnet_config.force_zeros_for_pooled_projection:
            # instantx sd3 controlnet used zero pooled projection
            controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
        else:
            controlnet_pooled_projections = controlnet_pooled_projections or pooled_prompt_embeds

        if controlnet_config.joint_attention_dim is not None:
            controlnet_encoder_hidden_states = prompt_embeds
        else:
            # SD35 official 8b controlnet does not use encoder_hidden_states
            controlnet_encoder_hidden_states = None

        # 7. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # FlowChef steering setup
        _flowchef_active = flowchef_kwargs is not None and forward_process is not None

        debug_steps: list[StepDebugInfo] = []

        # Cache for the previous step's post-Euler z0_hat (used when `shortcut_fraction > 0`).
        cached_z0_hat: torch.Tensor | None = None

        # 8. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                #######################################################
                # Inject residual information
                latent_model_input = latents
                controlnet_input = latents
                cur_control_image_latents = control_image_latents
                inner_debug: Optional[ResidualDebugInfo] = None
                # Probe's positive-text + zero-feedback prediction; populated by non-vanilla helpers
                # when the probe runs. Used as the baseline for feedback-axis CFG.
                unguided_pred: Optional[torch.Tensor] = None

                # Reuse prev-step z0_hat when we're past the warmup and the cache has been seeded.
                use_shortcut = i >= first_shortcut_idx and cached_z0_hat is not None
                cached_arg = cached_z0_hat if use_shortcut else None

                if feedback_mode != "vanilla":
                    # Residual/gradient helpers expect matching batch dims.
                    # When CFG is active the embeddings & orig_control_image
                    # are [neg, pos] along batch; slice the positive half.
                    if self.do_classifier_free_guidance:
                        _B = latents.shape[0]
                        cond_pe = prompt_embeds[_B:]
                        cond_ppe = pooled_prompt_embeds[_B:]
                        cond_raw_img = orig_control_image[_B:]
                    else:
                        cond_pe = prompt_embeds
                        cond_ppe = pooled_prompt_embeds
                        cond_raw_img = orig_control_image

                if feedback_mode == "residual":
                    residuals, inner_debug, unguided_pred = get_residual_condition(
                        noisy_latents=latents,
                        timesteps=t,
                        prompt_embeds=cond_pe,
                        pooled_prompt_embeds=cond_ppe,
                        transformer=self.transformer,
                        vae=self.vae,
                        scheduler=self.scheduler,
                        raw_cond_img=cond_raw_img,
                        forward_process=forward_process,
                        controlnet=self.controlnet,
                        variant=feedback_variant,
                        cached_z0_hat=cached_arg,
                    )
                    enc_residuals = vae_encode(self.vae, residuals)
                    if self.do_classifier_free_guidance:  # TODO: validte CFG behavior
                        enc_residuals = torch.cat([torch.zeros_like(enc_residuals), enc_residuals], dim=0)
                    cur_control_image_latents = torch.cat([control_image_latents, enc_residuals], dim=1)
                elif feedback_mode == "gradient":
                    dLdLatents, inner_debug, unguided_pred = get_residual_gradient(
                        noisy_latents=latents,
                        timesteps=t,
                        prompt_embeds=cond_pe,
                        pooled_prompt_embeds=cond_ppe,
                        transformer=self.transformer,
                        vae=self.vae,
                        scheduler=self.scheduler,
                        raw_cond_img=cond_raw_img,
                        forward_process=forward_process,
                        controlnet=self.controlnet,
                        rescale=feedback_variant,
                        cached_z0_hat=cached_arg,
                        **gradient_cond_kwargs,
                    )
                    cur_control_image_latents = control_image_latents
                    latent_model_input = torch.cat([latents, dLdLatents], dim=1)
                elif feedback_mode == "combined":
                    rescale, variant = feedback_variant
                    residuals, dLdLatents, inner_debug, unguided_pred = get_residual_and_gradient_condition(
                        noisy_latents=latents,
                        timesteps=t,
                        prompt_embeds=cond_pe,
                        pooled_prompt_embeds=cond_ppe,
                        transformer=self.transformer,
                        vae=self.vae,
                        scheduler=self.scheduler,
                        raw_cond_img=cond_raw_img,
                        forward_process=forward_process,
                        controlnet=self.controlnet,
                        variant=variant,  # For residual condition
                        rescale=rescale,  # For gradient condition
                        cached_z0_hat=cached_arg,
                        **gradient_cond_kwargs,
                    )
                    enc_residuals = vae_encode(self.vae, residuals)
                    if self.do_classifier_free_guidance:
                        enc_residuals = torch.cat([torch.zeros_like(enc_residuals), enc_residuals], dim=0)
                    cur_control_image_latents = torch.cat([control_image_latents, enc_residuals], dim=1)
                    latent_model_input = torch.cat([latents, dLdLatents], dim=1)
                elif feedback_mode == "vanilla":
                    pass
                else:
                    raise ValueError(f"Invalid feedback mode: {feedback_mode}")
                #########################################################

                # expand the latents if we are doing classifier free guidance
                if self.do_classifier_free_guidance:
                    if feedback_mode in ("gradient", "combined"):
                        uncond_input = torch.cat([latents, torch.zeros_like(dLdLatents)], dim=1)
                        latent_model_input = torch.cat([uncond_input, latent_model_input], dim=0)
                    else:
                        latent_model_input = torch.cat([latent_model_input] * 2)
                    controlnet_input = torch.cat([controlnet_input] * 2)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                # controlnet(s) inference — always pass original latent channels,
                # NOT the gradient-augmented input used by the transformer.
                control_block_samples = self.controlnet(
                    hidden_states=controlnet_input,
                    timestep=timestep,
                    encoder_hidden_states=controlnet_encoder_hidden_states,
                    pooled_projections=controlnet_pooled_projections,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    controlnet_cond=cur_control_image_latents,
                    conditioning_scale=cond_scale,
                    return_dict=False,
                )[0]

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_samples,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                noise_pred_text = None
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update the z0_hat cache for the next step's shortcut.  Done before FlowChef /
                # scheduler.step so `latents` is still z_t and the Euler identity holds:
                #   z0_hat_t = z_t - sigma_t * v_cond_t
                # When CFG is active we use the positive-conditional velocity (matches the helper's
                # internal probe call, which uses positive prompt only); otherwise noise_pred is
                # already the conditional prediction.  The cache update is intentionally before
                # feedback CFG so it tracks the post-text-CFG conditional velocity (preserves the
                # existing shortcut semantics).
                if num_shortcut_steps > 0 and feedback_mode != "vanilla":
                    v_for_cache = noise_pred_text if self.do_classifier_free_guidance else noise_pred
                    cached_z0_hat = _predict_z0_hat(latents, v_for_cache, t, self.scheduler).detach()

                # Orthogonal feedback-axis CFG.  Treats the helper's positive-text + zero-feedback
                # `model_pred` as the unconditional baseline:
                #   noise_pred <- unguided_pred + s * (noise_pred - unguided_pred)
                # No-op when scale=1.0, in vanilla mode, or on shortcut steps where the probe was
                # skipped (unguided_pred is None).
                if feedback_mode != "vanilla" and feedback_guidance_scale != 1.0 and unguided_pred is not None:
                    noise_pred = unguided_pred + feedback_guidance_scale * (noise_pred - unguided_pred)

                # Optional FlowChef steering (on top of ControlNet)
                # it is done post cfg application following
                # https://github.com/FlowChef/FlowChef/blob/9b705fcd91cb3cb8bf9d485ef6badc539201651c/src/pipeline_rf.py#L700-L706
                if _flowchef_active:
                    _flowchef_max_steps = flowchef_kwargs.get("max_steps", num_inference_steps)
                    if i <= _flowchef_max_steps:
                        latents, _fc_debug = flowchef_steer_step(
                            latents=latents,
                            noise_pred=noise_pred,
                            timestep=t,
                            scheduler=self.scheduler,
                            vae=self.vae,
                            forward_process=forward_process,
                            control_image=orig_control_image,
                            **flowchef_kwargs,
                        )
                        if inner_debug is None:
                            inner_debug = _fc_debug

                ### Visualize the prediction once the steering has been incorporated
                if visualize_z0 and i % visualize_z0_every_n == 0:
                    sigma_val = t.item() / self.scheduler.config.num_train_timesteps
                    decoded_z0 = _decode_z0_hat(latents, noise_pred, t, self.scheduler, self.vae)

                    fwd_pred_outer = None
                    cond_outer = None
                    residual_outer = None
                    if forward_process is not None:
                        cond_for_fp = orig_control_image[: decoded_z0.shape[0]]
                        residual_outer, fwd_pred_outer = forward_process.get_residual(
                            cur_img=decoded_z0, cond=cond_for_fp
                        )
                        cond_outer = cond_for_fp.detach().cpu()
                        fwd_pred_outer = fwd_pred_outer.detach().cpu()
                        residual_outer = residual_outer.detach().cpu()

                    debug_steps.append(
                        StepDebugInfo(
                            step_index=i,
                            sigma=sigma_val,
                            z0_hat_decoded=decoded_z0.detach().cpu(),
                            forward_pred=fwd_pred_outer,
                            condition=cond_outer,
                            residual=residual_outer,
                            inner_residual_debug=inner_debug,
                        )
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, debug_steps if debug_steps else None)

        output = StableDiffusion3PipelineOutput(images=image)
        output.debug_info = debug_steps if debug_steps else None
        return output


class SD3ControlNetModelFlowBender(SD3ControlNetModel):
    """SD3ControlNetModel that persists feedback_mode and forward_process_type
    in its config so they survive save_pretrained / from_pretrained.

    Architecture is identical to SD3ControlNetModel — no __init__ override so
    diffusers' config extraction works unchanged.  Custom config keys are
    restored after the parent finishes loading.
    """

    _FLOWBENDER_CONFIG_DEFAULTS = {
        "feedback_mode": "vanilla",
        "feedback_variant": None,
        "forward_process_type": "depth",
        "forward_process_kwargs": {},
        "gradient_cond_kwargs": {},
    }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):

        # Peek at saved config to grab our custom keys before the parent
        # filters them out during extract_init_dict.
        full_config = cls.load_config(pretrained_model_name_or_path, **kwargs)
        logger.info(f"Loading FlowBender controlnet from {pretrained_model_name_or_path} with config {full_config}")
        if isinstance(full_config, tuple):
            full_config = full_config[0]
        flowbender_values = {k: full_config.get(k, default) for k, default in cls._FLOWBENDER_CONFIG_DEFAULTS.items()}

        model = SD3ControlNetModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.__class__ = cls
        model.register_to_config(**flowbender_values)
        return model
