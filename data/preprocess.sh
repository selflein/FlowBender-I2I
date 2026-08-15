#!/usr/bin/env bash
#
# Data preprocessing pipeline for FlowBender SD3.5 ControlNet training.
#
# Takes the raw images downloaded by data/download_unsplash.py in
# {data_root}/data/{train,test} and produces the folders that training and
# evaluation read from:
#   {data_root}/images/                                  (resized images)
#   {data_root}/depths_Depth-Anything-V2-Large-hf/       (Depth Anything V2)
#   {data_root}/prompts_Florence-2-large/                (Florence-2 captions)
#   {data_root}/text_embeds_stable-diffusion-3.5-large/  (SD3.5 text embeds)
#
# Run from the repository root:
#   bash data/preprocess.sh /path/to/data_root
#
set -euo pipefail

DATA_ROOT="${1:?Usage: bash data/preprocess.sh <data_root>}"

RAW_DIR="$DATA_ROOT/data"
IMAGES_DIR="$DATA_ROOT/images"
DEPTHS_DIR="$DATA_ROOT/depths_Depth-Anything-V2-Large-hf"
PROMPTS_DIR="$DATA_ROOT/prompts_Florence-2-large"
TEXT_EMBEDS_DIR="$DATA_ROOT/text_embeds_stable-diffusion-3.5-large"

# 1) Center-crop & resize to 1024x1024 (CPU only).
python data/batch_image.py --image-dir "$RAW_DIR" --output-dir "$IMAGES_DIR"

# 2) Depth maps with Depth Anything V2 (GPU).
python data/batch_depth.py --image-dir "$IMAGES_DIR" --output-dir "$DEPTHS_DIR"

# 3) Captions with Florence-2 (GPU).
python data/batch_caption.py --image-dir "$IMAGES_DIR" --output-dir "$PROMPTS_DIR" --batch-size 16

# 4) Pre-compute SD3.5 text embeddings (GPU).
python data/batch_text_embeds.py --prompt-dir "$PROMPTS_DIR" --output-dir "$TEXT_EMBEDS_DIR" --batch-size 128

echo "Preprocessing complete under: $DATA_ROOT"
