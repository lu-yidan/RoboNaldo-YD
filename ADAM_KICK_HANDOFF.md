# Adam kick — migration handoff

Goal: train an **Adam Inspire (29 DOF)** version of the RoboNaldo right-foot
kick, reusing the G1 BeyondMimic tracking pipeline in this repo.

This is the entry point for a new agent. Deep detail on motor params / hazards
lives in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/adam/README_ADAM_HAZARDS.md`
— read that too. This file adds: **phase status, environment state,
exact next commands, and known model gaps**.

---

## 1. Phase status

| Phase | What | Status |
|---|---|---|
| 0 | Understand G1 pipeline + Adam motion data (BFS order, WXYZ quat) | DONE |
| 0c | Isaac Sim 5.1 / Isaac Lab 2.3 **native** env for training | DONE (see §3) |
| 1 | `robots/adam.py` articulation cfg (pnd_rl_lab motor params) | DONE (draft, gaps in §5) |
| 2 | Task registration + Adam YAML preset + G1-hardcode fixes | DONE (scaffold; not run e2e) |
| 3 | Convert Adam reference NPZ -> training NPZ (Isaac FK replay) | DONE + validated |
| 4 | Stage-1 flat tracking training smoke test | DONE (runs e2e, 5 iters) |
| 5 | Facing fix + motor-gain tuning + full Stage-1 run | DONE (reward ~22, 41% episodes complete; see §1b) |
| 5b | Full 10k-iter Stage-1 run | DONE (reward ~38, 92% episodes complete) |
| 6 | Stage-2 (ball/goal rewards) resume-train from Stage-1 | DONE (see §1c) |

The env is installed and verified. Phase 3 output is `motions/right_kick_adam.npz`
(611 frames @ 50 fps, 29 joints, 30 bodies). Full Stage-1 training now runs to 2000
iters and the policy learns the kick (walks in, kicks the ball, stays balanced) —
see §1b.

**Stage-1 training command (current, 4096 envs / 2000 iters, ~50 min on a 4070 Ti):**
```bash
python scripts/rsl_rl/train.py --task Tracking-Body-Frame-Flat-Adam-v0 \
  --motion_file motions/right_kick_adam.npz \
  --yaml right_kick_adam/tracking_params.yaml \
  --headless --logger tensorboard --run_name adam_kick_stage1 \
  --num_envs 4096 --max_iterations 2000
```
Logs/checkpoints under `logs/rsl_rl/adam_flat/<ts>_adam_kick_stage1/`.

**Play a checkpoint + record video** (needs the GPU free, i.e. not while training):
```bash
python scripts/rsl_rl/play.py --task Tracking-Body-Frame-Flat-Adam-v0 \
  --load_run <ts>_adam_kick_stage1 --checkpoint model_1999.pt \
  --yaml right_kick_adam/tracking_params.yaml \
  --motion_file motions/right_kick_adam.npz --num_envs 1 \
  --video --video_length 650 --video_views isometric --video_closeup --headless
# -> logs/videos/rl-video-step-0.mp4
```

**Validation of Phase 3 output** (right-foot kick, looks correct):
- pelvis z 0.86-0.93 (mean 0.91); toeRight (kicking foot) swings up during the
  kick, toeLeft planted; quats normalized, no NaNs.
- `joint_pos` range matches the input dof range exactly -> BFS joint mapping OK.
- All 14 `ADAM_TRACKED_BODY_NAMES` + anchor/feet/wrist/torso names exist in the
  robot body order printed by the converter.
- **Facing FIXED:** the raw Adam kick data travels along world **-Y** (forward is
  body **+X**, but the initial root yaw points body +X at world -Y). The ball /
  target / over-line logic all assume **+Y** (so does the deploy
  `freekick_motion.npz`, which also travels +Y). The converter now applies
  `--rotate_z_deg 180` so forward = +Y and the robot faces the ball. (An earlier
  note called the -Y drift "not a bug" — true kinematically, but it faced away
  from the ball, so we rotate it to match the +Y convention.)

## 1b. Latest training results (Stage-1, flat, 2000 iters, 4096 envs)

Two fixes unblocked a working kick, in order:

1. **Facing fix** (`--rotate_z_deg 180` in the converter): the robot now faces the
   ball. Stage-1 has `goal_weight=0`, so this alone doesn't change tracking reward,
   but it's required for Stage-2 (ball/goal rewards) and for playback to look right.
2. **Motor-gain + termination fix (the big one):** with pnd_rl_lab's soft ankle
   gains the pelvis anchor drifted out of the 0.25 m bound in ~1 s. Switching ankle
   + arm to the stiffer instinctMj gains and relaxing `anchor_pos` 0.25 -> 0.45:

   | metric | soft gains / 0.25 | stiff ankle+arm / 0.45 |
   |---|---|---|
   | reward | 2.4 | **22.1** |
   | mean episode length | 48 steps | **344 steps** |
   | episodes reaching timeout | ~0% | **41%** |
   | `anchor_pos` termination | 0.83 | **0.00** |
   | `error_joint_vel` | ~16 | **6.5** |

   Playback: the robot walks in, kicks the ball (it flies several metres), and
   stays balanced. **Remaining:** `ee_body_pos` termination is still ~0.59 (the
   fast foot swing trips the 0.25 m end-effector bound at contact) and
   `error_joint_pos ~1.4` still has slack — 2000 iters is smoke scale, 10k+ should
   tighten tracking; relaxing the `ee_body_pos` threshold during the kick window is
   the other lever.

## 1c. Stage-2 results (ball/goal rewards, resume-trained, 10k iters, 4096 envs)

Stage-2 turns on the full ball/goal reward set on top of the frozen-in tracking
skill. **Resume from a Stage-1 checkpoint** (do NOT train from scratch):

```bash
cd /home/luyd/workspace/RoboNaldo-YD
export WBT_TASK_PARAMS_YAML=right_kick_adam/task_params_2.yaml   # REQUIRED: bakes the
      # ball-contact sensor prim path to toeRight at import time (else defaults to G1)
python scripts/rsl_rl/train.py --task Tracking-Body-Frame-Flat-Adam-v0 \
  --motion_file motions/right_kick_adam.npz \
  --yaml right_kick_adam/task_params_2.yaml \
  --resume --load_run <stage1_run_folder> --checkpoint model_9999.pt \
  --headless --logger tensorboard --run_name adam_kick_stage2 \
  --num_envs 4096 --max_iterations 10000
```
(~4 h on a 4070 Ti. Note `--resume` continues the iteration counter, so the final
checkpoint of a 10k resume is `model_19998.pt`.)

Result after 10k Stage-2 iters (run `2026-07-02_22-28-43_adam_kick_stage2`):
- **Tracking preserved / improved:** `time_out` 0.976 (episodes complete), `anchor_pos`
  termination ~0, `error_body_pos` **0.056** (better than Stage-1's 0.068).
- **Ball interaction learned:** `last_episode_had_shot` 0.37 (up from 0.24 at start),
  `max_ball_velocity` ~3.1 m/s — the robot walks in, contacts the ball, kicks it away,
  and stays balanced (verified in playback).
- **Remaining (Stage-3 job):** `shot_success_count` 0, `last_episode_shot_error` ~8 —
  the ball is kicked but not accurately toward the target/over-line. This is expected:
  Stage-3 (`right_kick/task_params_3.yaml` analog) adds `adapt_motion_flag`, `jump_flag`,
  `use_ontime_ball_reset`, higher `goal_weight` (1.0) and `error_ball_to_target` (20),
  and tighter difficulty to convert "kicks the ball" into "scores".

**Next step (Phase 7):** create `right_kick_adam/task_params_3.yaml` (Adam-name port of
the G1 Stage-3 preset) and resume-train from the Stage-2 checkpoint.

## 2. What changed (this work)

Main repo (`RoboNaldo-YD`):
- `source/whole_body_tracking/whole_body_tracking/robots/adam.py` — **NEW + TUNED**.
  Adam Inspire 29-DOF `ArticulationCfg`. Motor Kp/Kd/effort for hip/knee/waist from
  `pnd_rl_lab`; **ankle (pitch 30->130, roll 3->70) and arm (shoulder/elbow ->60)
  now use the stiffer `instinctMj` gains** — pnd's were too soft for the kick (§1b);
  wrist gains + armature from `instinctMj`. Also computes `ADAM_ACTION_SCALE`
  (auto-recomputed from the gains, so it tracks these changes).
- `.../tasks/tracking/config/adam/` — **NEW**. Task registration
  (`Tracking-Flat-Adam-v0`, `Tracking-Body-Frame-Flat-Adam-v0`), `flat_env_cfg.py`
  (remaps every config-overridable G1 name -> Adam URDF name), RSL-RL agent cfg,
  and `README_ADAM_HAZARDS.md`.
- `.../tasks/tracking/yaml/right_kick_adam/tracking_params.yaml` — **NEW + TUNED**.
  Adam Stage-1 preset (`main_foot_name=toeRight`, self-collision off, etc.).
  `anchor_pos` termination threshold relaxed 0.25 -> 0.45 (§1b).
- `.../tasks/tracking/yaml/right_kick_adam/task_params_2.yaml` — **NEW**. Adam Stage-2
  preset (Adam-name port of `right_kick/task_params_2.yaml`): `stage: task`,
  `goal_weight 0.8`, full ball/goal reward block, `main_foot_name=toeRight`, and
  `ee_body_pos` termination keyed by Adam links (toeLeft/toeRight/wristYaw*). All goal
  reward terms verified to resolve on Adam (dry-run) — see §1c.
- `.../tasks/tracking/mdp/rewards.py` — **MODIFIED**. Fixed two G1-hardcoded spots
  that would crash Adam: `penalize_weak_foot_contact` (optional `weak_foot_name`
  param) and `arm_default_pose_penalty` (elbow match by substring). G1 behavior
  unchanged.
- `scripts/adam_npz_to_npz.py` — **NEW + rotate_z**. Converts an Adam reference NPZ
  (root_pos/root_rot/dof_pos, BFS, WXYZ) into the body-level training NPZ by
  kinematically replaying it on `ADAM_INSPIRE_CFG` and logging Isaac FK. Added
  `--rotate_z_deg` (a true rotation about world Z, pivoting on the start pose —
  unlike `--turn_y_axis`, which mirrors and flips handedness). We build
  `right_kick_adam.npz` with `--rotate_z_deg 180` so forward = +Y (faces the ball).
- `source/whole_body_tracking/whole_body_tracking/assets.py` — **FIXED**.
  `ASSET_DIR` pointed at a non-existent package-local `assets/` folder; now
  `parents[3]/"assets"` = the repo-level `assets/` repo (holds
  `unitree_description`, `pnd_description`, ...). This also fixes G1 asset paths.

IsaacLab 2.1 -> 2.3 / rsl-rl 3.x API fixes (needed for training to run):
- `scripts/rsl_rl/train.py` — **FIXED**. `handle_deprecated_rsl_rl_cfg` no longer
  exists in isaaclab_rl v2.3.2; wrapped the import with a no-op fallback.
- `source/whole_body_tracking/whole_body_tracking/utils/my_on_policy_runner.py` —
  **FIXED**. rsl-rl 3.x exposes `self.logger_type` (was `self.logger.logger_type`).
- Installed `tensordict` + switched `rsl-rl-lib` 2.3.3 -> **3.1.2** (isaaclab_rl
  v2.3.2's wrapper `get_observations()` returns a single `TensorDict`, which the
  3.x runner expects; 2.3.3's runner did `obs, extras = get_observations()` and
  crashed). rsl-rl-lib 3.1.2 needs only `torch>=2.6`, so torch 2.7 is fine.

Assets repo (`RoboNaldo-YD/assets`, **separate git repo**):
- `pnd_description/` — **NEW**. `adam_inspire/` (29 DOF, kick target) and
  `adam_lite/` (23 DOF), each with `urdf/mjcf/usd/meshes/assets`. `waistRoll`
  joint limit widened `±0.226 -> ±0.6` in both `adam_inspire` URDFs (motion
  clipped otherwise). Legacy `*_description/` dirs left untouched.

Not committed on purpose: `motions/*.mp4` previews (large binaries), and the
pre-existing unrelated working-tree changes in the main repo (`.gitignore` add of
`assets`, `RoboNaldo_Deploy` submodule deletion, `assets/teaser-crop.png`
deletion) — leave those for the repo owner to decide.

## 3. Environment (Phase 0c) — native, DONE

Docker was abandoned. Host was upgraded to **Ubuntu 22.04.5** (kernel 6.8, glibc
new enough), NVIDIA driver **580.159.03**, GPU **RTX 4070 Ti (12 GB)**. Everything
runs natively in a conda env — no container.

**Versions (verified working together):**
- conda env **`isaaclab51`**, Python **3.11**
- Isaac Sim **5.1.0** (pip `isaacsim[all,extscache]`)
- Isaac Lab source at `/home/luyd/workspace/IsaacLab`, tag **`v2.3.2`** (editable)
- `torch 2.7.0+cu128` / `torchvision 0.22.0+cu128` / `torchaudio 2.7.0`
- `rsl-rl-lib 3.1.2` (matches IsaacLab v2.3.2's pin; NOT 2.3.3), `tensordict 0.13.0`,
  `isaaclab 0.54.2`, `isaaclab_tasks 0.11.12`, `warp-lang 1.14.0`

**Activate + run** (no `isaaclab.sh -p` needed; just use the env's `python`):
```bash
source /home/luyd/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab51
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y   # needed for headless launch
```

**Verify (headless smoke — already passes):**
```bash
python /tmp/isaac_smoke.py          # AppLauncher headless + pxr + all imports + torch.cuda
```

### Gotchas that cost real time (do NOT re-fight these)
1. **torch must stay pinned to 2.7.0+cu128.** `isaaclab_rl[all]` / RL frameworks'
   resolver will happily pull `torch 2.12.x` (CPU/other CUDA) and break the Isaac
   Sim ABI. Always install with a constraints file. `/tmp/torch_pin.txt` holds:
   ```
   torch==2.7.0+cu128
   torchvision==0.22.0+cu128
   torchaudio==2.7.0
   packaging==23.0
   flatdict==4.0.1
   ```
   Use `pip install ... -c /tmp/torch_pin.txt --extra-index-url https://download.pytorch.org/whl/cu128`.
2. **`flatdict==4.0.1` won't build** (sdist-only, legacy `setup.py` does
   `import pkg_resources`) under the latest setuptools (82 dropped `pkg_resources`).
   Fix: pin build toolchain `setuptools==75.8.0` + `wheel==0.43.0` + `packaging==23.0`,
   then `pip install flatdict==4.0.1 --no-build-isolation`.
3. **Install IsaacLab extensions individually with `--no-build-isolation`**, NOT
   `./isaaclab.sh -i` (which pulls `[all]` RL frameworks and triggers gotcha #1).
   Order that worked: `flatdict` -> `-e source/isaaclab` -> `-e source/isaaclab_assets
   -e source/isaaclab_tasks` -> `rsl-rl-lib==2.3.3`. (`isaaclab_contrib/_mimic/_rl`
   were already editable-installed.)
4. **Benign pip conflict notices** — isaacsim-kernel hard-pins `click==8.1.7`,
   `psutil==5.9.8`, `typing_extensions==4.12.2`; isaaclab/wandb bump them (8.4.2 /
   7.2.2 / 4.15.0). The headless app launches fine anyway — do **not** downgrade
   (would break isaaclab/transformers/wandb).

If the env is ever rebuilt from scratch, follow §3 gotchas in that exact order.

## 4. Next steps (in order)

First: `conda activate isaaclab51` and `export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y` (see §3).

### 4a. Convert the Adam motion (Phase 3)
```bash
cd /home/luyd/workspace/RoboNaldo-YD
python scripts/adam_npz_to_npz.py \
  --input_file motions/right_kick_adam_v2_bfs.npz \
  --output_name right_kick_adam \
  --output_dir motions \
  --headless
```
Validate: the printed `robot body order` must match `ADAM_TRACKED_BODY_NAMES`
selection in `flat_env_cfg.py`; sanity-check output `body_pos_w[...,2]` (pelvis
height ~0.9 m) and that the foot trajectory looks like a kick. `JOINT_NAMES_BFS`
in the script is the assumed column order — re-verify if joints look scrambled.

### 4b. Stage-1 flat tracking smoke test (Phase 4)
```bash
cd /home/luyd/workspace/RoboNaldo-YD
python scripts/rsl_rl/train.py \
  --task Tracking-Body-Frame-Flat-Adam-v0 \
  --motion_file motions/right_kick_adam.npz \
  --yaml right_kick_adam/tracking_params.yaml \
  --headless --logger tensorboard \
  --run_name adam_kick_smoke
```
Start with a few hundred iters just to confirm it runs and reward trends up.
Note (from hazards doc): the base `tracking_env_cfg.py` reads `main_foot_name`
from the selected YAML **at import time**, so also export the preset if a run
path needs it: `export WBT_TASK_PARAMS_YAML=right_kick_adam/tracking_params.yaml`.

### 4c. Optional G1 regression smoke
Run the G1 task once to confirm the `rewards.py` edits didn't regress G1.

## 5. Known model gaps (the "adam.py may still need changes" items)

These are the most likely things to revisit after the first runs:

1. **Arm/ankle gains — RESOLVED (swapped to instinctMj).** pnd_rl_lab's ankle Kp
   (30 pitch / 3 roll) and arm Kp (18/9) were too soft: the support foot could not
   hold balance and episodes died in ~1 s (83% `anchor_pos` terminations, no kick).
   `adam.py` now uses the stiffer instinctMj ankle (130/70) + arm (60) gains, which
   fixed it — see §1b. hip/knee/waist stay on pnd_rl_lab (there pnd is as stiff or
   stiffer). Comparison table in `README_ADAM_HAZARDS.md` §4.
2. **Wrist gains + all armature are `instinctMj` fallbacks.** Our model is a full
   **29 DOF** (all wrists `revolute`); only pnd_rl_lab's *reference* URDF fixes
   the wrists (23 DOF) and sets no armature, so those gains had no pnd source and
   were taken from instinctMj `adam_sp`. Not validated in Isaac; adjust if wrists
   jitter or joints feel over/under-damped.
3. **`effort_limit_sim`/`velocity_limit_sim`** come from pnd_rl_lab's torque-speed
   curve peaks; pnd_rl_lab actually uses a custom `DelayedPDActuator` (T-N curve +
   friction + motor-strength DR). We use plain `ImplicitActuatorCfg`. Consider
   porting the custom actuator later for sim-to-real fidelity.
4. **`waistRoll` limit widened to ±0.6** to fit the motion — confirm the hardware
   actually allows this before deploying.
5. **Self-collision disabled** (`enabled_self_collisions=False` + Stage-1 preset
   `self_collision: false`). `make_self_collision_termination()` hardcodes G1
   ankle links; re-enable with `toeLeft`/`toeRight` when needed.
6. **Init height `pos.z=0.89`** is the pnd_rl_lab default; reset overwrites from
   the motion, but fix if the fallback spawn clips/floats.
7. **URDF spawn each run is slow** (`UrdfFileCfg` converts to USD at startup).
   Converting once to a committed USD under `pnd_description/adam_inspire/usd/`
   would speed iteration.

## 6. Handy references
- `README_ADAM_HAZARDS.md` — motor param comparison, full G1->Adam name map,
  all remapped reward terms, asset layout, reusable `pnd_rl_lab` resources.
- `/home/luyd/workspace/pnd_rl_lab` — Isaac-Lab-native Adam project (validated
  cfg, custom actuator, full BeyondMimic tracking env; 23-DOF, `EE_*_hand` links).
- `/home/luyd/workspace/instinctMj` (branch `shadow`) — mjlab Adam params
  (`src/instinct_mj/assets/adam_sp.py`); source for wrist + armature fallbacks.
- `scripts/csv_to_npz.py` — the G1 conversion that `adam_npz_to_npz.py` mirrors.
