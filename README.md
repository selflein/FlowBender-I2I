<div align="center">
<h1>FlowBender: Feedback-Aware Training for Self-Correcting Conditional Flows</h1>

<a href="https://arxiv.org/abs/2606.20404"><img src="https://img.shields.io/badge/arXiv-2606.20404-b31b1b.svg" alt="arXiv"></a>
<a href="https://flow-bender.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/papers/2606.20404"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-blue" alt="Hugging Face Paper"></a>

**[Technion](https://www.technion.ac.il/en/)** &nbsp;&nbsp;&nbsp; **[NVIDIA](https://www.nvidia.com/)** &nbsp;&nbsp;&nbsp; **[University of Toronto](https://www.utoronto.ca/)** &nbsp;&nbsp;&nbsp; **[Vector Institute](https://vectorinstitute.ai/)**

[Daniel Gilo](https://openreview.net/profile?id=~Daniel_Gilo1), [Sven Elflein](https://selflein.github.io/), Ido Sobol<!-- TODO: add homepage -->, [Or Litany](https://orlitany.github.io/)
</div>

---

This repository contains the **image-to-image experiments** from FlowBender:
training and evaluating self-correcting [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
ControlNets with flow matching. Conditional flow models often fail to satisfy
the task constraint they are conditioned on. FlowBender closes the loop by
letting the model *see its own deviation* — measured through the task's forward
operator — and correct for it, using either **residual** (zero-order) or **gradient** (first-order)
feedback.

## Repository structure

```bash
├── data/                     # dataset creation (resize, depth, caption, text embeds)
│   └── preprocess.sh         # runs the full preprocessing pipeline
├── docs/                     # data / training / evaluation / paper-table guides
└── sd3/                      # Stable Diffusion 3.5 pipeline and experiments
    ├── dataset.py            # loads images + conditions + cached prompt embeds
    ├── train_controlnet.py   # training entry point
    ├── evaluate.py           # evaluation entry point (multi-GPU, metrics + FID)
    ├── flowbender.py         # FlowBender ControlNet model + pipeline (residual/gradient feedback)
    ├── residual_utils.py     # feedback-conditioning helpers + forward-process registry
    ├── flowchef.py           # FlowChef baseline
    ├── conf/                 # Configurations
        ├── config.yaml       # Main configuration file (basically the CLI for train_controlnet.py). See https://hydra.cc/docs/advanced/override_grammar/basic/#basic-override-syntax
        ├── ...
        └── experiments/      # Experiment configuration (each training corresponds to a config file which can be set via experiments=...)
    └── <task>/               # per-task forward process + metrics
```

## Installation

```bash
conda create -n flowbender-i2i python=3.12
# You might want to use another CUDA version (https://pytorch.org/get-started/previous-versions/)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
```

`stabilityai/stable-diffusion-3.5-large` is gated — request access on the
[model page](https://huggingface.co/stabilityai/stable-diffusion-3.5-large), then `huggingface-cli login`.

## Usage

1. **Prepare data** — download the [Unsplash Lite](https://github.com/unsplash/datasets#lite-dataset)
   photos (~25k) and build the image / depth / caption / text-embedding dataset.
   See [docs/data.md](docs/data.md).

   ```bash
   python data/download_unsplash.py --data-root {data_root}
   bash data/preprocess.sh {data_root}
   ```

2. **Train** — Hydra + Accelerate; pick an experiment per task.
   See [docs/training.md](docs/training.md), and
   [docs/experiments.md](docs/experiments.md) for which config reproduces which
   paper table.

   ```bash
   accelerate launch --num_processes 4 \
       sd3/train_controlnet.py experiments=depth/vanilla user=<you>
   ```

3. **Evaluate** — generate, score task metrics, and compute FID.
   See [docs/evaluation.md](docs/evaluation.md).

   ```bash
   accelerate launch --num_processes 4 sd3/evaluate.py \
       eval_baseline=controlnet model_dir=/path/to/run output_dir=/path/to/eval user=<you>
   ```

## Acknowledgments

This code builds on [Stable Diffusion 3.5](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
and 🤗 [Diffusers](https://github.com/huggingface/diffusers), with data produced
by [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf)
and [Florence-2](https://huggingface.co/florence-community/Florence-2-large). The
baselines follow [FlowChef](https://github.com/FlowChef/FlowChef) and
[ControlNet++](https://arxiv.org/abs/2404.07987).

## Citation

```bibtex
@misc{gilo2026flowbenderfeedbackawaretrainingselfcorrecting,
  title         = {FlowBender: Feedback-Aware Training for Self-Correcting Conditional Flows},
  author        = {Daniel Gilo and Sven Elflein and Ido Sobol and Or Litany},
  year          = {2026},
  eprint        = {2606.20404},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2606.20404},
}
```
