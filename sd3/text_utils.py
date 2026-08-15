"""Shared utilities for text-encoder embeddings (SD3.5)."""

import gc
import logging
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline

logger = logging.getLogger(__name__)


def get_null_text_embeds(
    pretrained_model_name_or_path: str,
    cache_dir: str | Path,
    device: torch.device,
    weight_dtype: torch.dtype = torch.float32,
    max_sequence_length: int = 77,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(prompt_embeds, pooled_prompt_embeds)`` for the empty string.

    Returns:
        Tuple of CPU tensors with batch dimension squeezed:
        ``prompt_embeds`` of shape ``(seq_len, hidden_dim)`` and
        ``pooled_prompt_embeds`` of shape ``(pooled_dim,)``.
    """
    cache_path = Path(cache_dir) / "null_text_embeds.pt"
    if cache_path.exists():
        logger.info("Loading cached null text embeddings from %s", cache_path)
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        return data["prompt_embeds"], data["pooled_prompt_embeds"]

    logger.info("Computing null text embeddings (one-time cost) ...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path, transformer=None, torch_dtype=weight_dtype
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    with torch.no_grad():
        neg_embeds, _, neg_pooled, _ = pipe.encode_prompt(
            prompt=[""],
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    neg_embeds = neg_embeds.squeeze(0).cpu()
    neg_pooled = neg_pooled.squeeze(0).cpu()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt_embeds": neg_embeds, "pooled_prompt_embeds": neg_pooled}, cache_path)
    logger.info("Cached null text embeddings to %s", cache_path)
    return neg_embeds, neg_pooled


def align_null_embeds_to_prompt(
    null_prompt_embeds: torch.Tensor,
    null_pooled_prompt_embeds: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape cached null text embeddings to match a prompt batch for CFG.

    Args:
        null_prompt_embeds: Null tokens of shape ``(seq_len, hidden_dim)`` or ``(1, seq_len, hidden_dim)``.
        null_pooled_prompt_embeds: Pooled null embedding of shape ``(pooled_dim,)`` or ``(1, pooled_dim)``.
        prompt_embeds: Target prompt tensor of shape ``(B, target_seq_len, hidden_dim)``.
        pooled_prompt_embeds: Target pooled tensor of shape ``(B, pooled_dim)``.

    Returns:
        ``(null_prompt_embeds, null_pooled_prompt_embeds)`` matching the
        shape, dtype, and device of ``prompt_embeds`` / ``pooled_prompt_embeds``.
    """
    null_pe = null_prompt_embeds.unsqueeze(0) if null_prompt_embeds.dim() == 2 else null_prompt_embeds
    null_pooled = (
        null_pooled_prompt_embeds.unsqueeze(0) if null_pooled_prompt_embeds.dim() == 1 else null_pooled_prompt_embeds
    )

    target_seq_len = prompt_embeds.shape[1]
    cur_seq_len = null_pe.shape[1]
    if cur_seq_len < target_seq_len:
        null_pe = torch.nn.functional.pad(null_pe, (0, 0, 0, target_seq_len - cur_seq_len))
    elif cur_seq_len > target_seq_len:
        null_pe = null_pe[:, :target_seq_len]

    null_pe = null_pe.expand_as(prompt_embeds).to(dtype=prompt_embeds.dtype, device=prompt_embeds.device)
    null_pooled = null_pooled.expand_as(pooled_prompt_embeds).to(
        dtype=pooled_prompt_embeds.dtype, device=pooled_prompt_embeds.device
    )
    return null_pe, null_pooled
