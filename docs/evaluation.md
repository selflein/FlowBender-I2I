# Evaluation

Evaluation generates images for the test split, computes task metrics, and then
computes FID. It is Hydra-configured
([`sd3/conf/eval.yaml`](../sd3/conf/eval.yaml)) and launched with Accelerate for
multi-GPU generation.

## Generate + score

```bash
accelerate launch --num_processes 4 sd3/evaluate.py \
    eval_baseline=controlnet \
    model_dir=/path/to/{exp_name}-{wandb-id} \
    output_dir=/path/to/eval_output \
    user=<you>
```

- `model_dir` is a training output directory (or a Hugging Face ControlNet id).
- The task (`depth`, `super_resolution`, `jpeg_restoration`, `edge`) is
  auto-detected from the trained ControlNet's saved `forward_process_type`, and
  the conditioning directory is derived accordingly. Pass `task=<task>` only to
  override.
- Common `generation.*` overrides: `generation.batch_size`,
  `generation.num_inference_steps`, `generation.guidance_scale`,
  `generation.max_samples` (cap the number of samples for a quick run).

### Baselines (`eval_baseline=`)

| Value | Pipeline |
| --- | --- |
| `controlnet` | Trained SD3.5 ControlNet. Requires `model_dir` and can be pointed to a model trained with a ControlNet++ config as at inference time they are the same. |
| `controlnet_cfg` | Same, with classifier-free guidance. Guidance scale can be set via `geneartion.guidance_scale=<value>`  |
| `base_sd3` | Plain SD3.5, no conditioning. |
| `flowchef` | Training-free [FlowChef](https://github.com/FlowChef/FlowChef) steering. |

### Shortcut sampling

For `feedback_mode` in `{residual, gradient, combined}` you can reuse the
previous step's look-ahead prediction on the final iterations to skip the extra
probe passes:

```bash
... +pipeline.additional_call_kwargs.shortcut_fraction=1.0
```

`0.0` disables it (2N model evals); `1.0` keeps only step 0 as a full probe
(N+1 evals). See the note in [`eval.yaml`](../sd3/conf/eval.yaml) and Figure 6 in the paper.

## Output structure

```
{output_dir}/
├── generated/           # generated images
├── debug_{task}/        # debug visualizations for a subset of samples
├── metrics_{rank}.json  # per-GPU metrics
└── metrics.json         # aggregated metrics incl. FID
```

Per-task metrics: `depth` reports delta1 / MAE (under min-max alignment);
`super_resolution` and `jpeg_restoration` report PSNR / SSIM / LPIPS; `edge`
reports edge MAE. FID is reported for every task.
