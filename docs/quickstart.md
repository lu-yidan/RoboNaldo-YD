# Quickstart

This page expands the repository README into a complete local setup flow. Run
all commands from the repository root unless noted otherwise.

## Environment

Install Isaac Sim + Isaac Lab, then activate the Python environment that can
import `isaaclab`, `isaaclab_tasks`, and `isaaclab_rl`. The upstream baseline
(Isaac Sim 4.5.0 / Isaac Lab 2.1.0 / Python 3.10) works; this fork is currently
validated on **Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + Python 3.11 +
torch 2.7.0+cu128 + rsl-rl-lib 3.1.2**. See `ADAM_KICK_HANDOFF.md` §3 for the
exact native-install recipe and dependency-ordering pitfalls.

Install the RoboNaldo training extension:

```bash
python -m pip install -e source/whole_body_tracking
```

## Robot Assets

Download the Unitree description assets into the repo-level `assets/` directory
(where `ASSET_DIR` resolves):

```bash
mkdir -p assets
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
tar -xzf unitree_description.tar.gz -C assets/
rm unitree_description.tar.gz
test -f assets/unitree_description/urdf/g1/main.urdf
```

The path is resolved by `whole_body_tracking/assets.py` (`ASSET_DIR =
parents[3]/"assets"`).

## Motion Data

The repository includes the open-source right-foot kick reference motion:

```text
motions/right_kick_reference.csv
```

It has 612 frames at 50 Hz. Convert it to the NPZ format used by the training
environment:

```bash
python scripts/csv_to_npz.py \
  --input_file motions/right_kick_reference.csv \
  --input_fps 50 \
  --output_name right_kick \
  --headless
```

This creates `motions/right_kick.npz` by default.

Optional: upload to a W&B registry:

```bash
python scripts/upload_npz.py \
  --artifact_path motions/right_kick.npz \
  --entity <entity> \
  --name right_kick \
  --alias latest
```

## Train

Use `scripts/rsl_rl/train.py` directly:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --motion_file motions/right_kick.npz \
  --yaml right_kick/tracking_params.yaml \
  --headless \
  --logger wandb \
  --log_project_name kick \
  --run_name right_kick_tracking
```

Use `--registry_name <entity>/wandb-registry-motions/right_kick:latest` instead
of `--motion_file` if your reference motions live in a W&B artifact registry.

Stage progression:

| Stage | Preset | Notes |
| --- | --- | --- |
| Plane tracking | `right_kick/tracking_params.yaml` | Learns the human-kick motion prior on a flat plane. |
| Mixed-terrain tracking, optional | `right_kick/tracking_mixed_params.yaml` | Fine-tunes a plane checkpoint on light roughness and slopes. |
| Static adaptation | `right_kick/task_params_1.yaml` | Enables task rewards in a small ball-spawn range. |
| Static shooting | `right_kick/task_params_2.yaml` | Widened stationary-ball target shooting. |
| Dynamic shooting | `right_kick/task_params_3.yaml` | Incoming balls with adapted motion and jump trigger. |

Do not use the mixed-terrain preset as the default scratch run. Train
`right_kick/tracking_params.yaml` first, then resume into
`right_kick/tracking_mixed_params.yaml`:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --motion_file motions/right_kick.npz \
  --yaml right_kick/tracking_mixed_params.yaml \
  --resume True \
  --load_run <plane_tracking_run_folder> \
  --checkpoint model_<iter>.pt \
  --headless
```

Resume by passing `--resume True --load_run <run_folder> --checkpoint <model.pt>`
for local logs, or by using `--wandb_path <entity>/<project>/<run_id>`.
`--wandb_path` also accepts W&B UI URLs and loads the latest `model_*.pt` by
default.

## Play

A known Stage-2 hot-test policy run can be played with `scripts/rsl_rl/play.py`:

```bash
python scripts/rsl_rl/play.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --wandb_path <your_checkpoint_path> \
  --yaml right_kick/task_params_2.yaml \
  --motion_file motions/right_kick.npz \
  --num_envs 1 \
  --headless
```

Use `--yaml` to override the archived preset and `--motion_file` to use a local
motion NPZ.

`play.py` also exports a deployment ONNX artifact to
`<checkpoint_folder>/exported/policy-obs.onnx`. The file embeds joint names, PD
gains, default poses, and observation/action metadata for
[RoboNaldo_Deploy](https://github.com/OpenDriveLab/RoboNaldo_Deploy/tree/f60f24459aaabc3aea9187a2b13f8923049b629c). W&B training
runs (`--logger wandb`) additionally write `<run_name>.onnx` beside each saved
checkpoint.

## Evaluate

Use `scripts/rsl_rl/eval.py`:

```bash
python scripts/rsl_rl/eval.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --wandb_path <your_checkpoint_path> \
  --yaml right_kick/task_params_2.yaml \
  --motion_file motions/right_kick.npz \
  --num_envs 6000 \
  --headless
```

Evaluation writes JSON records and aggregate metrics to `logs/rsl_rl/eval/`.

## Resume From the Hot-Test Run

Use the Stage-2 run as a resume source:

```bash
python scripts/rsl_rl/train.py \
  --task Tracking-Body-Frame-Flat-G1-v0 \
  --wandb_path <your_checkpoint_path> \
  --motion_file motions/right_kick.npz \
  --yaml right_kick/task_params_2.yaml \
  --headless
```

`--wandb_path` resolves the checkpoint and archived task parameters. The
`--motion_file` argument provides the local reference motion generated from the
open-source CSV. Use `--registry_name` instead when the motion lives in a WandB
artifact registry.
