# Data preparation

Training and evaluation read from a single `data_root` directory that holds the
raw images, resized images, depth maps, captions, and pre-computed text
embeddings. Two commands build it end to end:

```bash
python data/download_unsplash.py --data-root {data_root}
bash data/preprocess.sh {data_root}
```

where `{data_root}` matches `data_root` in your user config
(`sd3/conf/user/<you>.yaml`). Both scripts are idempotent — they skip files that
already exist, so an interrupted run can simply be re-run.

## Dataset

We use the [Unsplash Lite dataset](https://github.com/unsplash/datasets#lite-dataset)
(~25,000 photos). It ships as TSV metadata rather than image files: each row of
the `photos.tsv*` table has a `photo_image_url` field pointing to the image.

[`data/download_unsplash.py`](../data/download_unsplash.py) downloads the metadata
archive (into `{data_root}/unsplash-lite`), fetches the images, and writes them
to `{data_root}/data/{train,test}`.

The train/test split is **fixed**: the repo ships the canonical manifest
[`data/unsplash_test_ids.txt`](../data/unsplash_test_ids.txt) (one photo id per
line) — **19,998 train / 5,000 test** — so you reproduce the exact split used in
the paper.

Usage of Unsplash photos is subject to the
[Unsplash Dataset terms](https://github.com/unsplash/datasets#terms).

## Layout

The two commands produce exactly:

```
{data_root}/
├── data/                                       # raw downloaded images (.jpg)
│   ├── train/<photo_id>.jpg
│   └── test/<photo_id>.jpg
├── images/                                     # center-cropped, resized RGB (.png)
│   ├── train/<photo_id>.png
│   └── test/<photo_id>.png
├── depths_Depth-Anything-V2-Large-hf/          # 16-bit depth maps (.png)
├── prompts_Florence-2-large/                   # captions (.txt)
├── text_embeds_stable-diffusion-3.5-large/     # cached SD3.5 text embeds (.pt)
└── unsplash-lite/                              # Lite metadata cache
```

Every folder except `data/` and `unsplash-lite/` mirrors the `train/`/`test/`
split. These names are what `sd3/conf/config.yaml` and `sd3/conf/eval.yaml`
expect. Only `depth` uses an on-disk conditioning directory
(`depths_Depth-Anything-V2-Large-hf/`) — `super_resolution`, `jpeg_restoration`,
and `edge` derive their condition from the image on the fly.

## Pipeline steps

`data/preprocess.sh` runs the four steps below in order. Each is a standalone
script that mirrors its input directory structure and skips already-produced
files.

1. **Resize** (`data/batch_image.py`, CPU) — center-crop / resize to 1024×1024,
   `data/` → `images/`.
2. **Depth** (`data/batch_depth.py`, GPU) —
   [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)
   16-bit depth maps, `images/` → `depths_Depth-Anything-V2-Large-hf/`.
3. **Captions** (`data/batch_caption.py`, GPU) —
   [Florence-2](https://huggingface.co/florence-community/Florence-2-large)
   captions, `images/` → `prompts_Florence-2-large/`.
4. **Text embeddings** (`data/batch_text_embeds.py`, GPU) — SD3.5 text-encoder
   outputs (CLIP-L, CLIP-G, T5-XXL) cached as `prompt_embeds` /
   `pooled_prompt_embeds`, `prompts_Florence-2-large/` →
   `text_embeds_stable-diffusion-3.5-large/`.

## Notes

- Models are pulled from the Hugging Face Hub into `$HF_HOME` on first use.
- The depth predictor used here (Depth Anything V2) must match the one inside the
  FlowBender depth forward process (`sd3/depth/forward_process.py`).
