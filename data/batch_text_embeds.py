"""Batch pre-compute SD3.5 text embeddings (CLIP-L, CLIP-G, T5-XXL).

Reads ``.txt`` caption files from a prompt directory and writes per-sample
``.pt`` files containing ``prompt_embeds`` and ``pooled_prompt_embeds`` ready
for ControlNet training, matching the format expected by
``sd3/train_controlnet.py``.

Output mirrors the input directory structure::

    prompts/train/abc.txt  ->  text_embeds/train/abc.pt

Each ``.pt`` file stores a dict::

    {
        "prompt_embeds":        Tensor (seq_len, hidden_dim),
        "pooled_prompt_embeds": Tensor (pooled_dim,),
    }

Usage::

    python batch_text_embeds.py --prompt-dir prompts/ --output-dir text_embeds/
    python batch_text_embeds.py --prompt-dir prompts/ --output-dir text_embeds/ \
        --model stabilityai/stable-diffusion-3.5-large --batch-size 64
"""

from pathlib import Path

import cyclopts
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast

app = cyclopts.App(help="Batch pre-compute SD3.5 text embeddings.")


def output_path_for(prompt_path: Path, prompt_root: Path, output_root: Path) -> Path:
    """Map a prompt .txt path to its corresponding .pt output path."""
    return output_root / prompt_path.relative_to(prompt_root).with_suffix(".pt")


def collect_prompt_paths(prompt_dir: Path, output_dir: Path) -> list[Path]:
    """Gather all .txt prompt paths that don't already have embeddings."""
    return sorted(
        p for p in prompt_dir.rglob("*.txt") if p.is_file() and not output_path_for(p, prompt_dir, output_dir).exists()
    )


def import_model_class(pretrained_model_name_or_path: str, revision: str | None, subfolder: str = "text_encoder"):
    """Resolve the correct text encoder class from the model config."""
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        return T5EncoderModel
    raise ValueError(f"{model_class} is not supported.")


def load_tokenizers(model_id: str, revision: str | None) -> tuple[CLIPTokenizer, CLIPTokenizer, T5TokenizerFast]:
    tokenizer_one = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", revision=revision)
    tokenizer_two = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", revision=revision)
    tokenizer_three = T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3", revision=revision)
    return tokenizer_one, tokenizer_two, tokenizer_three


def load_text_encoders(
    model_id: str, revision: str | None, variant: str | None, device: torch.device, dtype: torch.dtype
):
    cls_one = import_model_class(model_id, revision, "text_encoder")
    cls_two = import_model_class(model_id, revision, "text_encoder_2")
    cls_three = import_model_class(model_id, revision, "text_encoder_3")

    te_one = (
        cls_one.from_pretrained(model_id, subfolder="text_encoder", revision=revision, variant=variant)
        .to(device, dtype=dtype)
        .eval()
    )

    te_two = (
        cls_two.from_pretrained(model_id, subfolder="text_encoder_2", revision=revision, variant=variant)
        .to(device, dtype=dtype)
        .eval()
    )

    te_three = (
        cls_three.from_pretrained(model_id, subfolder="text_encoder_3", revision=revision, variant=variant)
        .to(device, dtype=dtype)
        .eval()
    )

    return te_one, te_two, te_three


def _encode_clip(
    text_encoder, tokenizer, prompts: list[str], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode prompts with a CLIP text encoder.

    Returns (prompt_embeds, pooled_prompt_embeds).
    """
    inputs = tokenizer(prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    outputs = text_encoder(inputs.input_ids.to(device), output_hidden_states=True)
    pooled = outputs[0]
    hidden = outputs.hidden_states[-2]
    hidden = hidden.to(dtype=text_encoder.dtype, device=device)
    return hidden, pooled


def _encode_t5(
    text_encoder, tokenizer, prompts: list[str], max_sequence_length: int, device: torch.device
) -> torch.Tensor:
    """Encode prompts with the T5 text encoder."""
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    embeds = text_encoder(inputs.input_ids.to(device))[0]
    return embeds.to(dtype=text_encoder.dtype, device=device)


def encode_prompt(
    text_encoders, tokenizers, prompts: list[str], max_sequence_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full SD3 triple-encoder prompt encoding.

    Mirrors ``encode_prompt`` in ``sd3/train_controlnet.py``.

    Returns:
        prompt_embeds: (B, seq_len, hidden_dim)
        pooled_prompt_embeds: (B, pooled_dim)
    """
    clip_embeds_list = []
    clip_pooled_list = []
    for tok, enc in zip(tokenizers[:2], text_encoders[:2]):
        hidden, pooled = _encode_clip(enc, tok, prompts, device)
        clip_embeds_list.append(hidden)
        clip_pooled_list.append(pooled)

    clip_embeds = torch.cat(clip_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_list, dim=-1)

    t5_embeds = _encode_t5(text_encoders[-1], tokenizers[-1], prompts, max_sequence_length, device)

    clip_embeds = torch.nn.functional.pad(clip_embeds, (0, t5_embeds.shape[-1] - clip_embeds.shape[-1]))
    prompt_embeds = torch.cat([clip_embeds, t5_embeds], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


@app.default
def main(
    *,
    prompt_dir: str,
    output_dir: str,
    model: str = "stabilityai/stable-diffusion-3.5-large",
    revision: str | None = None,
    variant: str | None = None,
    batch_size: int = 32,
    max_sequence_length: int = 256,
) -> None:
    """Run batch text embedding extraction.

    Args:
        prompt_dir: Root directory of .txt caption files.
        output_dir: Output directory for .pt embedding files (mirrors
            prompt_dir structure).
        model: HuggingFace model id for SD3 / SD3.5.
        revision: Model revision.
        variant: Model variant (e.g. "fp16").
        batch_size: Number of prompts to encode at once.
        max_sequence_length: Max token length for the T5 encoder.
    """
    torch_device = torch.device("cuda")
    dtype = torch.bfloat16

    prompt_root = Path(prompt_dir)
    output_root = Path(output_dir)

    prompt_paths = collect_prompt_paths(prompt_root, output_root)
    total = len(prompt_paths)
    print(f"Found {total} prompts without embeddings in {prompt_dir}")
    if total == 0:
        return

    print(f"Loading text encoders from {model} ...")
    tokenizers = load_tokenizers(model, revision)
    text_encoders = load_text_encoders(model, revision, variant, torch_device, dtype)

    processed = 0
    for i in range(0, total, batch_size):
        batch_paths = prompt_paths[i : i + batch_size]
        prompts = [p.read_text(encoding="utf-8").strip() for p in batch_paths]

        with torch.inference_mode():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompts, max_sequence_length, torch_device
            )

        for j, pp in enumerate(batch_paths):
            out_path = output_path_for(pp, prompt_root, output_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"prompt_embeds": prompt_embeds[j].cpu(), "pooled_prompt_embeds": pooled_prompt_embeds[j].cpu()},
                out_path,
            )

        processed += len(batch_paths)
        print(f"[{processed}/{total}] Encoded batch of {len(batch_paths)}")

    print("Done.")


if __name__ == "__main__":
    app()
