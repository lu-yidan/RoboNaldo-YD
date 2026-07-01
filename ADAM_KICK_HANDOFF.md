# Adam kick — migration handoff

Goal: train an **Adam Inspire (29 DOF)** version of the RoboNaldo right-foot
kick, reusing the G1 BeyondMimic tracking pipeline in this repo.

This is the entry point for a new agent. Deep detail on motor params / hazards
lives in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/adam/README_ADAM_HAZARDS.md`
— read that too. This file adds: **phase status, environment/container state,
exact next commands, and known model gaps**.

---

## 1. Phase status

| Phase | What | Status |
|---|---|---|
| 0 | Understand G1 pipeline + Adam motion data (BFS order, WXYZ quat) | DONE |
| 0c | Isaac Sim/Lab **Docker** env for training | IN PROGRESS (see §3) |
| 1 | `robots/adam.py` articulation cfg (pnd_rl_lab motor params) | DONE (draft, gaps in §5) |
| 2 | Task registration + Adam YAML preset + G1-hardcode fixes | DONE (scaffold; not run e2e) |
| 3 | Convert Adam reference NPZ -> training NPZ (Isaac FK replay) | SCRIPT WRITTEN, NOT RUN (needs container) |
| 4 | Stage-1 flat tracking training smoke test | NOT STARTED |

Nothing has been run end-to-end inside Isaac yet. Phases 3 and 4 are blocked on
finishing the container install (§3).

## 2. What changed (this work)

Main repo (`RoboNaldo-YD`):
- `source/whole_body_tracking/whole_body_tracking/robots/adam.py` — **NEW**. Adam
  Inspire 29-DOF `ArticulationCfg`. Motor Kp/Kd/effort/velocity from `pnd_rl_lab`;
  wrist gains + armature from `instinctMj`. Also computes `ADAM_ACTION_SCALE`.
- `.../tasks/tracking/config/adam/` — **NEW**. Task registration
  (`Tracking-Flat-Adam-v0`, `Tracking-Body-Frame-Flat-Adam-v0`), `flat_env_cfg.py`
  (remaps every config-overridable G1 name -> Adam URDF name), RSL-RL agent cfg,
  and `README_ADAM_HAZARDS.md`.
- `.../tasks/tracking/yaml/right_kick_adam/tracking_params.yaml` — **NEW**. Adam
  Stage-1 preset (`main_foot_name=toeRight`, self-collision off, etc.).
- `.../tasks/tracking/mdp/rewards.py` — **MODIFIED**. Fixed two G1-hardcoded spots
  that would crash Adam: `penalize_weak_foot_contact` (optional `weak_foot_name`
  param) and `arm_default_pose_penalty` (elbow match by substring). G1 behavior
  unchanged.
- `scripts/adam_npz_to_npz.py` — **NEW**. Converts an Adam reference NPZ
  (root_pos/root_rot/dof_pos, BFS, WXYZ) into the body-level training NPZ by
  kinematically replaying it on `ADAM_INSPIRE_CFG` and logging Isaac FK.

Assets repo (`RoboNaldo-YD/assets`, **separate git repo**):
- `pnd_description/` — **NEW**. `adam_inspire/` (29 DOF, kick target) and
  `adam_lite/` (23 DOF), each with `urdf/mjcf/usd/meshes/assets`. `waistRoll`
  joint limit widened `±0.226 -> ±0.6` in both `adam_inspire` URDFs (motion
  clipped otherwise). Legacy `*_description/` dirs left untouched.

Not committed on purpose: `motions/*.mp4` previews (large binaries), and the
pre-existing unrelated working-tree changes in the main repo (`.gitignore` add of
`assets`, `RoboNaldo_Deploy` submodule deletion, `assets/teaser-crop.png`
deletion) — leave those for the repo owner to decide.

## 3. Environment / container (Phase 0c)

Host OS glibc is too old for Isaac Lab natively, so everything runs in Docker.

- Container: **`isaac-adam`** from `nvcr.io/nvidia/isaac-sim:4.5.0`, GPU
  passthrough, host network, with host dirs mounted:
  - `/home/luyd/workspace/IsaacLab`  -> `/workspace/IsaacLab`
  - `/home/luyd/workspace/RoboNaldo-YD` -> `/workspace/RoboNaldo-YD`
- Fix applied: IsaacLab's `_isaac_sim` symlink pointed at a host path missing in
  the container; created container-local `/home/luyd/isaacsim -> /isaac-sim`.
- Enter the container:
  ```bash
  docker exec -it isaac-adam bash
  ```
- Install (in progress, was building when handed off):
  ```bash
  cd /workspace/IsaacLab && TERM=xterm ./isaaclab.sh -i rsl_rl
  ```
  Known snag: `flatdict==4.0.1` fails under pip build isolation
  (`No module named 'pkg_resources'`). Workaround already applied:
  ```bash
  ./isaaclab.sh -p -m pip install "flatdict==4.0.1" --no-build-isolation
  ./isaaclab.sh -p -m pip install -e source/isaaclab
  ```
- Also install this repo's package inside the container before running:
  ```bash
  cd /workspace/RoboNaldo-YD && /workspace/IsaacLab/isaaclab.sh -p -m pip install -e source/whole_body_tracking
  ```
- Verify env is ready:
  ```bash
  /workspace/IsaacLab/isaaclab.sh -p -c "import isaaclab, isaaclab_tasks, rsl_rl; import whole_body_tracking; print('OK')"
  ```

**FIRST THING A NEW AGENT SHOULD DO:** confirm the isaaclab core editable install
finished (the command above), then install `whole_body_tracking`, then verify.

## 4. Next steps (in order)

### 4a. Convert the Adam motion (Phase 3) — inside container
```bash
cd /workspace/RoboNaldo-YD
/workspace/IsaacLab/isaaclab.sh -p scripts/adam_npz_to_npz.py \
  --input_file motions/right_kick_adam_v2_bfs.npz \
  --output_name right_kick_adam \
  --output_dir motions \
  --headless
```
Validate: the printed `robot body order` must match `ADAM_TRACKED_BODY_NAMES`
selection in `flat_env_cfg.py`; sanity-check output `body_pos_w[...,2]` (pelvis
height ~0.9 m) and that the foot trajectory looks like a kick. `JOINT_NAMES_BFS`
in the script is the assumed column order — re-verify if joints look scrambled.

### 4b. Stage-1 flat tracking smoke test (Phase 4) — inside container
```bash
cd /workspace/RoboNaldo-YD
/workspace/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
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

1. **Arm/ankle gains may be too soft for a fast kick.** pnd_rl_lab ankle Kp is
   much lower than instinctMj (30 vs 130 pitch, 3 vs 70 roll); shoulders 18/9 vs
   60. If the struck leg or arms lag the reference, swap those groups to the
   stiffer instinctMj values (comparison table in `README_ADAM_HAZARDS.md` §4).
2. **Wrist gains + all armature are `instinctMj` fallbacks**, because pnd_rl_lab
   treats Inspire wrists as fixed (23 DOF) and sets no armature. Not validated in
   Isaac; adjust if wrists jitter or joints feel over/under-damped.
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
