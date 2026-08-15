# Reproducing the paper tables

Each row of the paper's image-to-image tables corresponds to one experiment
config under [`sd3/conf/experiments/<task>/`](../sd3/conf/experiments). Train it
with

```bash
accelerate launch --num_processes 4 \
    sd3/train_controlnet.py experiments=<task>/<config> user=<you>
```

then evaluate it (see [evaluation.md](evaluation.md)). This repo covers the
image-to-image results — **Tables 1–3**. Table 4 (mesh texturing) is a separate
codebase.

## Table 1 — Image-to-Image (Super-Resolution, Depth→RGB, Edge→RGB)

`experiments=` values (config filenames, without the `.yaml`):

| Paper row | `super_resolution` | `depth` | `edge` |
| --- | --- | --- | --- |
| Standard FT | `super_resolution/vanilla` | `depth/vanilla` | `edge/vanilla` |
| FT + ℒ_align | `super_resolution/controlnetpp` | `depth/controlnetpp` | `edge/controlnetpp` |
| IT Guidance | `eval_baseline=flowchef` | `eval_baseline=flowchef` | `eval_baseline=flowchef` |
| First-order (w.r.t. 𝐱_t) | `super_resolution/gradient-std-normal-mse` | `depth/gradient-std-normal-mse` | `edge/gradient-std-normal-mse` |
| First-order (w.r.t. 𝐱̂₁) | `super_resolution/gradient-std-normal-mse-x0` | `depth/gradient-std-normal-mse-x0` | `edge/gradient-std-normal-mse-x0` |
| Zero-order | `super_resolution/residual-scaled` | `depth/residual-scaled` | `edge/residual-scaled` |
| Combined (w.r.t. 𝐱_t) | `super_resolution/combined` | `depth/combined` | `edge/combined` |
| Combined (w.r.t. 𝐱̂₁) | `super_resolution/combined-x0` | `depth/combined-x0` | `edge/combined-x0` |

## Table 2 — JPEG Restoration

| Paper row | Config |
| --- | --- |
| Standard FT | `jpeg_restoration/vanilla` |
| Zero-order (Ours) | `jpeg_restoration/residual-scaled` |

## Table 3 — Ablation of p_un (Super-Resolution, zero-order)

`p_un` is the null-feedback dropout probability (`training.null_residual_probability`).

| p_un | Config |
| --- | --- |
| 0.0 | `super_resolution/residual-scaled-nullres-p00` |
| 0.1 | `jpeg_restoration/residual-scaled` |
| 0.2 | `super_resolution/residual-scaled-nullres-p02` |
| 0.3 | `super_resolution/residual-scaled-nullres-p03` |
