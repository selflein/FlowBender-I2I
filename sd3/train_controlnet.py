#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import math
import os
import random
import shutil
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Callable

import diffusers
import hydra
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.models.controlnets.controlnet import zero_module
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils.torch_utils import is_compiled_module
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from sd3.dataset import PreprocessedControlNetDataset, collate_fn
from sd3.evaluate import MODALITY_TO_METRICS
from sd3.flowbender import SD3ControlNetModelFlowBender
from sd3.flowbender import StableDiffusion3FlowBenderPipeline as StableDiffusion3ControlNetPipeline
from sd3.residual_utils import (
    FORWARD_PROCESS_REGISTRY,
    compute_reward_loss,
    get_residual_and_gradient_condition,
    get_residual_condition,
    get_residual_gradient,
    vae_encode,
)
from sd3.text_utils import align_null_embeds_to_prompt, get_null_text_embeds
from sd3.vis import save_z0_visualization

logger = get_logger(__name__)


def _flatten_dict(d, parent_key="", sep="."):
    """Flatten a nested dict into a single-level dict with dotted keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _widen_conv2d_in_channels(old_proj: torch.nn.Conv2d) -> torch.nn.Conv2d:
    """Double a Conv2d's in_channels, copying old weights and zero-initializing the new half."""
    bias = old_proj.bias is not None
    in_channels = old_proj.in_channels
    new_proj = torch.nn.Conv2d(
        in_channels=in_channels * 2,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=bias,
    )
    torch.nn.init.zeros_(new_proj.weight)
    new_proj.weight.data[:, :in_channels, :, :] = old_proj.weight.data.clone()
    if bias:
        new_proj.bias.data = old_proj.bias.data.clone()
    return new_proj.to(old_proj.weight.device)


def _cleanup_checkpoints(output_dir: Path, keep_last_n: int | None, total_limit: int | None, best_ckpt_path: Path):
    """Remove old numbered checkpoints, keeping the last N and the best."""
    limit = keep_last_n if keep_last_n is not None else total_limit
    if limit is None:
        return

    all_ckpts = [d for d in output_dir.iterdir() if d.name.startswith("checkpoint-") and d.name != best_ckpt_path.name]
    all_ckpts = sorted(all_ckpts, key=lambda x: int(x.name.split("-")[1]))

    if len(all_ckpts) <= limit:
        return

    to_remove = all_ckpts[: len(all_ckpts) - limit]
    logger.info(
        f"{len(all_ckpts)} checkpoints exist, removing {len(to_remove)}: {', '.join(str(p) for p in to_remove)}"
    )
    for candidate in to_remove:
        shutil.rmtree(str(candidate))


def _maybe_save_best_checkpoint(
    cfg: DictConfig,
    accelerator: Accelerator,
    val_metrics: dict[str, float],
    best_metric_value: float | None,
    best_ckpt_path: str,
    global_step: int,
) -> float | None:
    """Save best checkpoint if the tracked metric improved. Returns the updated best value."""
    metric_key = cfg.checkpointing.get("best_metric")
    if metric_key is None:
        return best_metric_value

    if metric_key not in val_metrics:
        logger.warning(
            f"Best-metric key '{metric_key}' not found in validation metrics "
            f"(available: {list(val_metrics.keys())}). Skipping best-checkpoint logic."
        )
        return best_metric_value

    current = val_metrics[metric_key]
    mode = cfg.checkpointing.get("best_mode", "max")
    is_better = (
        best_metric_value is None
        or (mode == "max" and current > best_metric_value)
        or (mode == "min" and current < best_metric_value)
    )

    if is_better:
        logger.info(
            f"New best {metric_key}: {current:.6f} (prev: {best_metric_value}). "
            f"Saving best checkpoint at step {global_step}."
        )
        if os.path.exists(best_ckpt_path):
            shutil.rmtree(best_ckpt_path)
        accelerator.save_state(best_ckpt_path)
        best_metric_value = current

    return best_metric_value


@torch.no_grad()
def log_validation(
    controlnet: torch.nn.Module,
    transformer: torch.nn.Module,
    vae: torch.nn.Module,
    cfg: DictConfig,
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    step: int,
    forward_process: torch.nn.Module,
    metric_fun: Callable,
):
    """Run validation distributed across all accelerator processes."""
    logger.info("Running validation... ")
    controlnet = accelerator.unwrap_model(controlnet)

    val_dataset = PreprocessedControlNetDataset(
        img_dir=cfg.validation.img_dir,
        cond_dir=cfg.validation.cond_dir,
        text_embeds_dir=cfg.validation.text_embeds_dir,
        prompt_dir=cfg.validation.prompt_dir,
        resolution=cfg.data.resolution,
        condition_type=cfg.data.get("condition_type", "depth"),
        scale_factor=cfg.forward_process_kwargs.get("scale_factor", None),
    )

    # All ranks use the same RNG to pick the same global indices.
    n = min(cfg.validation.num_samples, len(val_dataset))
    rng = random.Random(cfg.training.seed)
    indices = rng.sample(range(len(val_dataset)), n)

    # Shard samples across ranks.
    rank = accelerator.process_index
    num_processes = accelerator.num_processes
    local_indices = indices[rank::num_processes]
    local_samples = [val_dataset[idx] for idx in local_indices]

    # Materialise edge conditioning from the GT image for the edge task.
    # The dataset stores the image itself as the conditioning placeholder;
    # HED is applied here so both the pipeline call and the metric loop
    # below see the same edge map.
    if cfg.data.condition_type == "edge":
        fp_unwrapped = accelerator.unwrap_model(forward_process)
        for sample in local_samples:
            gt_01 = (sample["pixel_values"].unsqueeze(0).to(accelerator.device, dtype=torch.float32) + 1.0) / 2.0
            with torch.no_grad():
                edge_01 = fp_unwrapped.compute_condition(gt_01)
            sample["conditioning_pixel_values"] = (edge_01.repeat(1, 3, 1, 1) * 2.0 - 1.0).squeeze(0).cpu()

    # Use pre-computed embeddings from the dataset instead of loading text encoders.
    use_cfg = cfg.validation.guidance_scale > 1.0
    neg_prompt_embeds = neg_pooled_prompt_embeds = None
    if use_cfg:
        neg_pe, neg_pooled = get_null_text_embeds(
            pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
            cache_dir=Path(cfg.validation.text_embeds_dir),
            device=accelerator.device,
            weight_dtype=weight_dtype,
            max_sequence_length=cfg.text.max_sequence_length,
        )
        neg_prompt_embeds = neg_pe.unsqueeze(0)
        neg_pooled_prompt_embeds = neg_pooled.unsqueeze(0)

    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        controlnet=controlnet,
        safety_checker=None,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        vae=vae,
        transformer=transformer,
        revision=cfg.model.revision,
        variant=cfg.model.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)

    image_logs = []
    inference_ctx = torch.autocast(accelerator.device.type)

    for i, sample in enumerate(tqdm(local_samples, desc="Running validation", disable=not accelerator.is_main_process)):
        cond_tensor = sample["conditioning_pixel_values"]
        cond_np = ((cond_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        cond_img = Image.fromarray(cond_np)

        train_tensor = sample["pixel_values"]
        train_np = ((train_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
        train_img = Image.fromarray(train_np)

        pe = sample["prompt_embeds"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        pooled_pe = sample["pooled_prompt_embeds"].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)

        neg_pe_i = None
        neg_pooled_pe_i = None
        if use_cfg:
            neg_pe_i, neg_pooled_pe_i = align_null_embeds_to_prompt(
                neg_prompt_embeds, neg_pooled_prompt_embeds, pe, pooled_pe
            )

        n_per_prompt = cfg.validation.num_samples_per_prompt
        generators = [
            torch.Generator(device=accelerator.device).manual_seed(cfg.training.seed + j) for j in range(n_per_prompt)
        ]
        with inference_ctx:
            result = pipeline(
                prompt_embeds=pe,
                negative_prompt_embeds=neg_pe_i,
                pooled_prompt_embeds=pooled_pe,
                negative_pooled_prompt_embeds=neg_pooled_pe_i,
                control_image=cond_img,
                num_images_per_prompt=n_per_prompt,
                num_inference_steps=cfg.validation.num_denoising_steps,
                guidance_scale=cfg.validation.guidance_scale,
                generator=generators,
                visualize_z0=True,
                visualize_z0_every_n=max(1, cfg.validation.num_denoising_steps // 8),
            )
        images = result.images

        z0_images = []
        if result.debug_info:
            for j in range(len(images)):
                z0_img = save_z0_visualization(result.debug_info, final_image=images[j])
                if z0_img is not None:
                    z0_images.append(z0_img)

        image_logs.append(
            {
                "train_image": train_img,
                "cond_image": cond_img,
                "images": images,
                "z0_images": z0_images,
                "validation_prompt": sample.get("prompts", ""),
            }
        )

    # Compute metrics on this rank's samples.
    forward_process = accelerator.unwrap_model(forward_process)
    metrics = defaultdict(list)

    img_preproc = transforms.ToTensor()
    for i, sample in enumerate(local_samples):
        cond_t = sample["conditioning_pixel_values"].unsqueeze(0).to(accelerator.device, dtype=torch.float32)
        gt_t = (sample["pixel_values"].unsqueeze(0).to(accelerator.device, dtype=torch.float32) * 0.5 + 0.5).clamp(0, 1)
        for gen_img in image_logs[i]["images"]:
            gen_t = img_preproc(gen_img).unsqueeze(0).to(accelerator.device, dtype=torch.float32)
            _, batch_metrics = metric_fun(gen_t, cond_t, forward_process, gt_tensors=gt_t)
            for k, v in batch_metrics.items():
                metrics[k].extend(v)

    # Gather metrics across all ranks via all_reduce.
    avg_metrics = {}
    if metrics:
        for k, v in metrics.items():
            local_sum = torch.tensor(sum(v), device=accelerator.device, dtype=torch.float64)
            local_count = torch.tensor(float(len(v)), device=accelerator.device, dtype=torch.float64)
            if num_processes > 1:
                torch.distributed.all_reduce(local_sum)
                torch.distributed.all_reduce(local_count)
            avg_metrics[f"val/{k}"] = (local_sum / local_count).item()

    logger.info(f"Metrics at step {step}: {avg_metrics}")
    if accelerator.is_main_process:
        accelerator.log(avg_metrics, step=step)

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                for idx, log in enumerate(image_logs[:3]):
                    all_imgs = [log["train_image"], log["cond_image"]] + log["images"]
                    h = max(img.size[1] for img in all_imgs)
                    resized = []
                    for img in all_imgs:
                        if img.size[1] != h:
                            w_new = int(img.size[0] * h / img.size[1])
                            img = img.resize((w_new, h), Image.LANCZOS)
                        resized.append(np.asarray(img))
                    grid = np.concatenate(resized, axis=1)
                    caption = log["validation_prompt"] or "validation"
                    tag = f"validation/{idx}"
                    log_dict = {tag: wandb.Image(grid, caption=caption)}
                    for j, z0_img in enumerate(log.get("z0_images", [])):
                        log_dict[f"{tag}/z0_gen{j}"] = wandb.Image(z0_img, caption=caption)
                    tracker.log(log_dict, step=step)
            else:
                logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    free_memory()

    controlnet.to(accelerator.device)
    return image_logs, avg_metrics


def _sync_new_wandb_id() -> str:
    """Generate a wandb run ID on rank 0 and broadcast to all ranks.

    Uses torch.distributed broadcast (gloo backend) instead of filesystem sync
    to avoid unreliable metadata visibility on distributed filesystems (Lustre).
    """
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size <= 1:
        return wandb.util.generate_id()

    need_init = not dist.is_initialized()
    if need_init:
        dist.init_process_group(backend="gloo", timeout=timedelta(minutes=2))

    wid_list = [wandb.util.generate_id() if rank == 0 else ""]
    dist.broadcast_object_list(wid_list, src=0)

    if need_init:
        dist.destroy_process_group()

    return wid_list[0]


def _format_output_dir(output_dir: str, exp_name: str, wandb_id: str) -> Path:
    """Get output directory.

    Example:
        output_dir: /p/a/out, wandb_id: abc123 -> /p/a/out-abc123
        output_dir: /p/a/out/, wandb_id: abc123 -> /p/a/out-abc123
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return Path(output_dir) / f"{exp_name}-{wandb_id}"


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
    )

    if cfg.checkpointing.resume_from:
        raise NotImplementedError("Manual resume is not supported yet.")

    trainings_dir = Path(cfg.training.output_dir)
    exp_name = cfg.exp_name
    wandb_id = _sync_new_wandb_id()
    exp_output_dir = _format_output_dir(trainings_dir, exp_name, wandb_id)

    # Scale down gradient accumulation to keep the same effective batch size
    # when using multiple GPUs.  E.g. grad_accum=4 on 1 GPU → grad_accum=1 on
    # 4 GPUs gives the same effective batch of batch_size * 4.
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    orig_grad_accum = cfg.training.gradient_accumulation_steps
    cfg.training.gradient_accumulation_steps = max(1, orig_grad_accum // num_processes)
    if cfg.training.gradient_accumulation_steps != orig_grad_accum:
        print(
            f"Adjusted gradient_accumulation_steps: {orig_grad_accum} -> "
            f"{cfg.training.gradient_accumulation_steps} for {num_processes} processes"
        )
        if orig_grad_accum % num_processes != 0:
            print(
                f"WARNING: gradient_accumulation_steps ({orig_grad_accum}) is not "
                f"evenly divisible by num_processes ({num_processes}). "
                f"Effective batch size will differ from the single-GPU setting."
            )

    logging_dir = exp_output_dir / cfg.training.logging_dir
    accelerator_project_config = ProjectConfiguration(project_dir=str(exp_output_dir), logging_dir=str(logging_dir))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=20))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        mixed_precision=cfg.accelerator.mixed_precision,
        log_with=cfg.accelerator.log_with,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )

    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        exp_output_dir.mkdir(exist_ok=True)

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.model.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKL.from_pretrained(
        cfg.model.pretrained_model_name_or_path, subfolder="vae", revision=cfg.model.revision, variant=cfg.model.variant
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=cfg.model.revision,
        variant=cfg.model.variant,
    )

    # Controlnet setup
    if cfg.model.controlnet_model_name_or_path:  # Pre-trained
        logger.info("Loading existing controlnet weights")
        controlnet = SD3ControlNetModelFlowBender.from_pretrained(cfg.model.controlnet_model_name_or_path)

        # Controlnet uses pos_embed from the transformer
        # https://github.com/huggingface/diffusers/blob/d791c5c024014784df4a8dac2601e19fb4d300fc/src/diffusers/pipelines/controlnet_sd3/pipeline_stable_diffusion_3_controlnet.py#L214
        # Here we always add the weights since it only negligly increases the model size.
        if controlnet.pos_embed is None:
            controlnet.pos_embed = controlnet._get_pos_embed_from_transformer(transformer)
            # Copy relevant config options from transformer
            controlnet.register_to_config(
                use_pos_embed=True, pos_embed_max_size=transformer.config.pos_embed_max_size, pos_embed_type="sincos"
            )
    else:
        # From scratch, matching SD3.5 ControlNet architecture
        # See https://huggingface.co/stabilityai/stable-diffusion-3.5-large-controlnet-depth/blob/main/config.json
        logger.info("Initializing controlnet weights from transformer")
        config = dict(transformer.config)
        config["num_layers"] = 19
        config["extra_conditioning_channels"] = 0
        config["joint_attention_dim"] = None  # Match pre-trained SD3.5 controlnets and do not use text-conditioning
        config["dual_attention_layers"] = []
        config["use_pos_embed"] = True
        config["pos_embed_type"] = "sincos"
        config["force_zeros_for_pooled_projection"] = False
        controlnet = SD3ControlNetModelFlowBender.from_config(config)

        # {'use_pos_embed', 'pos_embed_type', 'dual_attention_layers', 'force_zeros_for_pooled_projection'} was not found in config. Values will be initialized to default values.
        controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
        controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
        controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
        controlnet.pos_embed_input = zero_module(controlnet.pos_embed_input)

    # Setup the input for the residual feedback
    if cfg.feedback_mode in ("residual", "combined"):
        old_proj = controlnet.pos_embed_input.proj
        controlnet.pos_embed_input.proj = _widen_conv2d_in_channels(old_proj)
        controlnet.pos_embed_input.proj.requires_grad_(True)
        controlnet.register_to_config(extra_conditioning_channels=old_proj.in_channels)

    if cfg.feedback_mode in ("gradient", "combined"):
        logger.info("Using gradient conditioning for the DiT")
        old_proj = transformer.pos_embed.proj
        transformer.pos_embed.proj = _widen_conv2d_in_channels(old_proj)
        print("Transformer pos_embed.proj.weight:\n", transformer.pos_embed.proj.weight)
        transformer.register_to_config(in_channels=old_proj.in_channels * 2)

    fp_kwargs_cfg = cfg.get("forward_process_kwargs", {})
    fp_kwargs = (
        OmegaConf.to_container(fp_kwargs_cfg, resolve=True) if OmegaConf.is_config(fp_kwargs_cfg) else fp_kwargs_cfg
    )
    grad_kwargs_cfg = cfg.get("gradient_cond_kwargs", {})
    gradient_cond_kwargs = (
        OmegaConf.to_container(grad_kwargs_cfg, resolve=True)
        if OmegaConf.is_config(grad_kwargs_cfg)
        else grad_kwargs_cfg
    )
    feedback_variant = cfg.feedback_variant
    if OmegaConf.is_config(feedback_variant):
        feedback_variant = OmegaConf.to_container(feedback_variant, resolve=True)
    controlnet.register_to_config(
        feedback_mode=cfg.feedback_mode,
        feedback_variant=feedback_variant,
        forward_process_type=cfg.data.condition_type,
        forward_process_kwargs=fp_kwargs,
        gradient_cond_kwargs=gradient_cond_kwargs,
    )

    fp_cls = FORWARD_PROCESS_REGISTRY[controlnet.config.forward_process_type]
    forward_process = fp_cls(**fp_kwargs)
    forward_process.requires_grad_(False)

    transformer.requires_grad_(False)
    if cfg.feedback_mode in ("gradient", "combined"):
        transformer.pos_embed.proj.requires_grad_(True)
    vae.requires_grad_(False)
    controlnet.train()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            i = len(weights) - 1

            while len(weights) > 0:
                weights.pop()
                model = models[i]
                i -= 1

                if not hasattr(model, "save_pretrained"):
                    continue

                sub_dir = "controlnet"
                model.save_pretrained(os.path.join(output_dir, sub_dir))

            if cfg.feedback_mode in ("gradient", "combined"):
                transformer.save_pretrained(os.path.join(output_dir, "transformer"))

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()

            if not isinstance(model, SD3ControlNetModelFlowBender):
                continue

            load_model = SD3ControlNetModelFlowBender.from_pretrained(input_dir, subfolder="controlnet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

        if cfg.feedback_mode in ("gradient", "combined"):
            transformer_ckpt = os.path.join(input_dir, "transformer")
            if os.path.exists(transformer_ckpt):
                loaded_transformer = SD3Transformer2DModel.from_pretrained(transformer_ckpt)
                transformer.load_state_dict(loaded_transformer.state_dict())
                del loaded_transformer

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if cfg.training.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        transformer.enable_gradient_checkpointing()
        # VAE decode runs under autograd in gradient/combined feedback modes
        # (and in the reward loss); checkpointing saves decoder
        # activations at the cost of a second forward pass. It's a no-op for
        # the no_grad vae_encode calls that dominate most paths. Especially
        # important with upcast_vae where activations are fp32.
        vae.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    if cfg.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.training.scale_lr:
        cfg.training.learning_rate = (
            cfg.training.learning_rate
            * cfg.training.gradient_accumulation_steps
            * cfg.training.batch_size
            * accelerator.num_processes
        )

    params_to_optimize = list(controlnet.parameters())
    if cfg.feedback_mode in ("gradient", "combined"):
        params_to_optimize += [p for p in transformer.parameters() if p.requires_grad]
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=params_to_optimize)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and transformer to device and cast to weight_dtype
    if cfg.model.upcast_vae:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)

    train_dataset = PreprocessedControlNetDataset(
        img_dir=cfg.data.img_dir,
        cond_dir=cfg.data.cond_dir,
        text_embeds_dir=cfg.data.text_embeds_dir,
        prompt_dir=cfg.data.get("prompt_dir"),
        resolution=cfg.data.resolution,
        condition_type=cfg.data.get("condition_type", "depth"),
        scale_factor=cfg.forward_process_kwargs.get("scale_factor", None),
        jpeg_quality=cfg.forward_process_kwargs.get("quality", None),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.loader.num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if cfg.training.max_steps is None:
        cfg.training.max_steps = cfg.training.num_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        cfg.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.training.max_steps * accelerator.num_processes,
        num_cycles=cfg.lr_scheduler.num_cycles,
        power=cfg.lr_scheduler.power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler, forward_process = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler, forward_process
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.training.gradient_accumulation_steps)
    if overrode_max_train_steps:
        cfg.training.max_steps = cfg.training.num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.training.num_epochs = math.ceil(cfg.training.max_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = {
            k: str(v) if not isinstance(v, (int, float, bool)) else v
            for k, v in _flatten_dict(OmegaConf.to_container(cfg, resolve=True)).items()
            if k != "validation"
        }
        wandb_kwargs = {
            "name": f"{cfg.exp_name}_{wandb_id}",
            "dir": cfg.wandb.output_dir,
            "id": wandb_id,
            "resume": "allow",
        }
        accelerator.init_trackers(
            cfg.wandb.tracker_project_name, config=tracker_config, init_kwargs={"wandb": wandb_kwargs}
        )
        wandb_id_file = exp_output_dir / "wandb_run_id.txt"
        if not wandb_id_file.exists():
            with open(wandb_id_file, "w") as f:
                f.write(wandb_id)

    # Pre-compute null text embeddings for CFG training (encoding of "")
    null_prompt_embeds = null_pooled_prompt_embeds = None
    if cfg.training.null_text_probability > 0:
        null_prompt_embeds, null_pooled_prompt_embeds = get_null_text_embeds(
            pretrained_model_name_or_path=cfg.model.pretrained_model_name_or_path,
            cache_dir=Path(cfg.data.text_embeds_dir),
            device=accelerator.device,
            weight_dtype=weight_dtype,
            max_sequence_length=cfg.text.max_sequence_length,
        )

    # Train!
    total_batch_size = cfg.training.batch_size * accelerator.num_processes * cfg.training.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"Num examples = {len(train_dataset)}")
    logger.info(f"Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"Num Epochs = {cfg.training.num_epochs}")
    logger.info(f"Instantaneous batch size per device = {cfg.training.batch_size}")
    logger.info(f"Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"Gradient Accumulation steps = {cfg.training.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps = {cfg.training.max_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.checkpointing.resume_from:
        if cfg.checkpointing.resume_from != "latest":
            path = os.path.basename(cfg.checkpointing.resume_from)
        else:
            # Get the most recent numbered checkpoint (skip checkpoint-best)
            dirs = [
                d.name
                for d in exp_output_dir.iterdir()
                if d.name.startswith("checkpoint-") and d.name.split("-")[1].isdigit()
            ]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.checkpointing.resume_from}' does not exist. Starting a new training run."
            )
            cfg.checkpointing.resume_from = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(str(exp_output_dir / path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    best_metric_value = None
    best_ckpt_path = exp_output_dir / "checkpoint-best"

    progress_bar = tqdm(
        range(0, cfg.training.max_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    metric_fun = MODALITY_TO_METRICS[unwrap_model(controlnet).config.forward_process_type]

    # Visualize at start of training (all processes participate)
    if initial_global_step == 0:
        log_validation(
            controlnet,
            transformer,
            vae,
            cfg,
            accelerator,
            weight_dtype,
            global_step,
            forward_process=forward_process,
            metric_fun=metric_fun,
        )
    accelerator.wait_for_everyone()

    for epoch in range(first_epoch, cfg.training.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                model_input = vae_encode(vae, pixel_values).to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=cfg.flow_matching.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=cfg.flow_matching.logit_mean,
                    logit_std=cfg.flow_matching.logit_std,
                    mode_scale=cfg.flow_matching.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Get the text embedding for conditioning
                prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype)

                # CFG training: randomly replace text with null (encoding of "")
                if null_prompt_embeds is not None and random.random() < cfg.training.null_text_probability:
                    prompt_embeds_for_model, pooled_prompt_embeds_for_model = align_null_embeds_to_prompt(
                        null_prompt_embeds, null_pooled_prompt_embeds, prompt_embeds, pooled_prompt_embeds
                    )
                else:
                    prompt_embeds_for_model = prompt_embeds
                    pooled_prompt_embeds_for_model = pooled_prompt_embeds

                # controlnet(s) inference,
                # conditioning image is in range [-1, 1]
                raw_cond_img = batch["conditioning_pixel_values"]
                if cfg.data.condition_type == "edge":
                    # Edge conditioning is derived on the fly from the GT
                    # image -- the dataset just stores the image itself as
                    # a placeholder. Apply HED here once so the same map
                    # feeds both the ControlNet and the residual feedback.
                    gt_01 = (batch["pixel_values"].to(accelerator.device, dtype=torch.float32) + 1.0) / 2.0
                    with torch.no_grad():
                        edge_01 = unwrap_model(forward_process).compute_condition(gt_01)
                    raw_cond_img = (edge_01.repeat(1, 3, 1, 1) * 2.0 - 1.0).to(raw_cond_img.dtype)
                enc_cond_image = vae_encode(vae, raw_cond_img)

                #############################
                # Inject residual information
                controlnet_latents_input = noisy_model_input
                noisy_model_input = noisy_model_input
                if cfg.feedback_mode != "vanilla":
                    if cfg.feedback_mode == "residual":
                        if random.random() <= cfg.training.null_residual_probability:
                            residuals = torch.zeros_like(enc_cond_image)
                        else:
                            residuals, _, _ = get_residual_condition(
                                noisy_latents=noisy_model_input,
                                timesteps=timesteps,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                transformer=transformer,
                                vae=vae,
                                scheduler=noise_scheduler,
                                raw_cond_img=raw_cond_img,
                                enc_cond_image=enc_cond_image,
                                forward_process=forward_process,
                                controlnet=unwrap_model(controlnet),
                                variant=cfg.feedback_variant,
                            )
                            residuals = vae_encode(vae, residuals)

                        enc_cond_image = torch.cat([enc_cond_image, residuals], dim=1)
                    elif cfg.feedback_mode == "gradient":
                        if random.random() <= cfg.training.null_residual_probability:
                            dLdLatents = torch.zeros_like(noisy_model_input)
                        else:
                            dLdLatents, _, _ = get_residual_gradient(
                                noisy_latents=noisy_model_input,
                                timesteps=timesteps,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                transformer=transformer,
                                vae=vae,
                                scheduler=noise_scheduler,
                                raw_cond_img=raw_cond_img,
                                enc_cond_image=enc_cond_image,
                                forward_process=forward_process,
                                controlnet=unwrap_model(controlnet),
                                rescale=cfg.feedback_variant,
                                **gradient_cond_kwargs,
                            )

                        noisy_model_input = torch.cat([noisy_model_input, dLdLatents], dim=1)
                    elif cfg.feedback_mode == "combined":
                        null_both = random.random() <= cfg.training.null_residual_probability
                        rescale, variant = cfg.feedback_variant
                        if null_both:
                            residuals = torch.zeros_like(enc_cond_image)
                            dLdLatents = torch.zeros_like(noisy_model_input)
                        else:
                            residuals, dLdLatents, _, _ = get_residual_and_gradient_condition(
                                noisy_latents=noisy_model_input,
                                timesteps=timesteps,
                                prompt_embeds=prompt_embeds,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                transformer=transformer,
                                vae=vae,
                                scheduler=noise_scheduler,
                                raw_cond_img=raw_cond_img,
                                enc_cond_image=enc_cond_image,
                                forward_process=forward_process,
                                controlnet=unwrap_model(controlnet),
                                variant=variant,
                                rescale=rescale,
                                **gradient_cond_kwargs,
                            )
                            residuals = vae_encode(vae, residuals)

                        enc_cond_image = torch.cat([enc_cond_image, residuals], dim=1)
                        noisy_model_input = torch.cat([noisy_model_input, dLdLatents], dim=1)
                    else:
                        raise ValueError(f"Invalid feedback mode: {cfg.feedback_mode}")
                #############################

                control_block_res_samples = controlnet(
                    hidden_states=controlnet_latents_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds_for_model
                    if unwrap_model(controlnet).context_embedder is not None
                    else None,
                    pooled_projections=pooled_prompt_embeds_for_model,
                    controlnet_cond=enc_cond_image,
                    return_dict=False,
                )[0]
                control_block_res_samples = [sample.to(dtype=weight_dtype) for sample in control_block_res_samples]

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds_for_model,
                    pooled_projections=pooled_prompt_embeds_for_model,
                    block_controlnet_hidden_states=control_block_res_samples,
                    return_dict=False,
                )[0]

                # Save raw velocity prediction for reward loss before preconditioning
                raw_model_pred = model_pred

                # Follow: Section 5 of https://huggingface.co/papers/2206.00364.
                # Preconditioning of the model outputs.
                if cfg.flow_matching.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + controlnet_latents_input

                # these weighting schemes use a uniform timestep sampling and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=cfg.flow_matching.weighting_scheme, sigmas=sigmas
                )

                # flow matching loss
                if cfg.flow_matching.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), dim=1
                )
                loss = loss.mean()

                # ControlNet++ reward consistency loss (arXiv 2404.07987)
                if cfg.reward_loss.enabled:
                    reward_loss = compute_reward_loss(
                        noisy_model_input=controlnet_latents_input,
                        noise_pred=raw_model_pred,
                        timesteps=timesteps,
                        sigmas=sigmas,
                        raw_cond_img=raw_cond_img,
                        scheduler=noise_scheduler,
                        vae=vae,
                        forward_process=forward_process,
                        sigma_threshold=cfg.reward_loss.sigma_threshold,
                        loss_func=cfg.reward_loss.loss_func,
                    )
                    if reward_loss is not None:
                        loss = loss + cfg.reward_loss.weight * reward_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(controlnet.parameters())
                    if cfg.feedback_mode in ("gradient", "combined"):
                        params_to_clip += [p for p in transformer.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(params_to_clip, cfg.training.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=cfg.training.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg.checkpointing.steps == 0:
                        save_path = exp_output_dir / f"checkpoint-{global_step}"
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                        _cleanup_checkpoints(
                            exp_output_dir,
                            keep_last_n=cfg.checkpointing.get("keep_last_n"),
                            total_limit=cfg.checkpointing.total_limit,
                            best_ckpt_path=best_ckpt_path,
                        )

                # All processes participate in validation.
                if cfg.validation.get("img_dir") is not None and global_step % cfg.validation.run_every_n_steps == 0:
                    _, val_metrics = log_validation(
                        controlnet,
                        transformer,
                        vae,
                        cfg,
                        accelerator,
                        weight_dtype,
                        global_step,
                        forward_process=forward_process,
                        metric_fun=metric_fun,
                    )

                    if accelerator.is_main_process:
                        best_metric_value = _maybe_save_best_checkpoint(
                            cfg=cfg,
                            accelerator=accelerator,
                            val_metrics=val_metrics,
                            best_metric_value=best_metric_value,
                            best_ckpt_path=best_ckpt_path,
                            global_step=global_step,
                        )

                accelerator.wait_for_everyone()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            if cfg.reward_loss.enabled and reward_loss is not None:
                logs["reward_loss"] = reward_loss.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= cfg.training.max_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(exp_output_dir / "controlnet")
        if cfg.feedback_mode in ("gradient", "combined"):
            transformer.save_pretrained(exp_output_dir / "transformer")

    accelerator.end_training()


if __name__ == "__main__":
    main()
