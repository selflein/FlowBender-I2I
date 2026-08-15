# Training

Training is configured with [Hydra](https://hydra.cc/) and launched with
[Accelerate](https://huggingface.co/docs/accelerate). The base config is
[`sd3/conf/config.yaml`](../sd3/conf/config.yaml); each run selects an
experiment from [`sd3/conf/experiments/<task>/`](../sd3/conf/experiments).

## One-time setup

Point the code at your data and output directories by creating a user
config (keeps machine-specific paths out of the shared config):

```bash
cp sd3/conf/user/default.yaml sd3/conf/user/<you>.yaml
```

Edit `data_root`, `wandb_dir`, and `training.output_dir`, then pass `user=<you>` on the command line.

## Hardware

An 80GB GPU is recommended.

## Launch a run

Single GPU:

```bash
python sd3/train_controlnet.py experiments=depth/vanilla user=<you>
```

Multi-GPU:

```bash
accelerate launch --num_processes 4 \
    sd3/train_controlnet.py experiments=depth/vanilla user=<you>
```

Any config value can be overridden inline, e.g.
`training.learning_rate=1e-5 training.batch_size=8 checkpointing.steps=1000`.

## Tasks and feedback modes

The task is selected by the experiment's `override /task:` (one of `depth`,
`edge`, `super_resolution`, `jpeg_restoration`). The steering behaviour is set
by `feedback_mode`:

| `feedback_mode` | Description |
| --- | --- |
| `vanilla` | Standard ControlNet, no feedback. |
| `residual` | Feedback from the forward-operator residual (zero-order feedback). |
| `gradient` | Feedback from the gradient of the forward-operator consistency loss (first-order feedback). |
| `combined` | Residual + gradient feedback. |

The ControlNet++ baseline (Standard FT + L_align) is not a separate `feedback_mode`; its `*-controlnetpp`
configs use `feedback_mode: vanilla` with the reward-consistency loss enabled
(`reward_loss.enabled: true`).

For `gradient`/`combined` modes, `gradient_cond_kwargs.skip_denoiser_grad` selects the
first-order variant: `false` (default) takes the gradient w.r.t. `x_t`; `true` takes it
w.r.t. the predicted clean latent `x̂1` (in this repo, the clean latent is referred to by `x0` instead of `x̂1` due to the existing convention of the SD/ControlNet code).

Configs are named `<mode>[-variant].yaml` under each task directory (e.g.
`depth/vanilla`, `super_resolution/gradient-std-normal-mse`). Browse
[`sd3/conf/experiments/`](../sd3/conf/experiments) for the full list per task,
and see [experiments.md](experiments.md) for which config reproduces which paper
table.

## Output structure

Runs are written under `training.output_dir` as `{exp_name}-{wandb-id}`:

```
{output_dir}/{exp_name}-{wandb-id}
├── checkpoint-{step}/
│   ├── controlnet/...
│   ├── transformer/...          # only for gradient/combined modes
│   ├── optimizer.bin
│   ├── scheduler.bin
│   └── random_states_0.pkl
├── controlnet/...               # final model
├── transformer/...              # final model (gradient/combined modes)
└── wandb_run_id.txt
```

Metrics are logged to Weights & Biases (project `training.tracker_project_name`,
default `flowbender-i2i`). Set `WANDB_MODE=offline` to disable online logging.
