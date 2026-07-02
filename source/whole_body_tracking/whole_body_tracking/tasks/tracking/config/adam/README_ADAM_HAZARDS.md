# Adam migration — hazards & open items

Status: **Phase 2 scaffolding only.** The Adam tracking task is registered
(`Tracking-Flat-Adam-v0`, `Tracking-Body-Frame-Flat-Adam-v0`) and every
*config-overridable* G1 name is remapped in `flat_env_cfg.py`. It does **not**
run end-to-end yet — the items in section 1 must be fixed and validated in the
Isaac Sim container first.

Decision on record (updated): motor / actuator numbers now follow
**pnd_rl_lab** (`assets/robots/pnd.py::PND_ADAM_INSPIRE_CFG` + `pnd_actuators.py`),
which is **Isaac-Lab-native** and targets this same `adam_inspire` URDF — the
most directly applicable source. Exceptions:
- **Wrist joints** (`wrist{Yaw,Pitch,Roll}_*`): our robot is a full **29 DOF** —
  the `adam_inspire` URDF we train on defines all 6 wrist joints as `revolute`
  (so does instinctMj's `adam_sp.xml`). Only pnd_rl_lab's *reference* URDF
  (`pnd_robots/adam_inspire`) marks the wrists `type="fixed"` and actuates 23
  DOF, so its config has no wrist actuator to copy. Wrist Kp/Kd therefore come
  from **instinctMj** `adam_sp`. This is a *gain-sourcing* note, **not** a DOF
  limitation of our model.
- **Armature**: pnd_rl_lab AdamInspire actuators do not set armature; instinctMj
  per-group armature values (physically derived) are retained.

The earlier instinctMj-vs-pnd comparison is kept in section 4 for reference.
See also section 6 for the shared `pnd_rl_lab` resources worth reusing.

---

## 1. Hardcoded G1 names *inside `mdp` code* — RESOLVED

The two blockers below were parameterized in `mdp/rewards.py` (G1 defaults kept
unchanged) and are now fed Adam names from `flat_env_cfg.py`.

| File | Function | Was | Fix |
|---|---|---|---|
| `mdp/rewards.py` | `penalize_weak_foot_contact` | derived `weak_foot_name` from hardcoded `*_ankle_roll_link` | added optional `weak_foot_name` param; config sets `ADAM_WEAK_FOOT_BODY="toeLeft"`. (weight `-0.5*goal_weight`; only Stage-2, but would raise on Adam if evaluated) |
| `mdp/rewards.py` | `arm_default_pose_penalty` | `arm_joint_names=None` -> G1 patterns; elbow via `endswith("_elbow_joint")` | **would crash Stage-1** (`find_joints` raises when G1 patterns match no Adam joint). Config now sets `arm_joint_names=ADAM_ARM_JOINT_PATTERNS`; elbow match changed to `"elbow" in name.lower()` (works for G1 + Adam). |
| `mdp/rewards.py` | `robot_torso_ball_distance` | default `body_names=["torso_link"]` | overridden via config to `["torso"]` — OK |
| `mdp/rewards.py` | `locomotion_phase_torso_orientation_l2` | default `torso_body_name="torso_link"` | overridden via config to `"torso"` — OK |
| `mdp/commands.py` | `MotionCommandCfg` | default `main_foot_name="right_ankle_roll_link"` | overridden via YAML `main_foot_name="toeRight"` — OK |

Impact if left unfixed: `arm_default_pose_penalty` breaks **Stage-1** at startup
(nonzero weight + unmatched joint patterns -> `find_joints` raises).
`penalize_weak_foot_contact` breaks **Stage-2** (and Stage-1 too if the reward
manager evaluates zero-weight terms). Both are now safe.

## 2. Module-import coupling

- `tracking_env_cfg.py` reads `MAIN_FOOT_NAME = _P["main_foot_name"]` **at import
  time** and bakes it into the `ball_contact_forces` sensor prim path
  (`{ENV}/Robot/{MAIN_FOOT_NAME}`) and `CommandsCfg.motion`. Therefore Adam runs
  **require** selecting the Adam preset via the env var, e.g.:

  ```bash
  export WBT_TASK_PARAMS_YAML=right_kick_adam/tracking_params.yaml
  ```

  Selecting the task id alone is not enough.

## 3. Config-level remaps already applied (`flat_env_cfg.py`)

Mapping G1 (MJCF) → Adam (URDF):

| G1 body | Adam URDF body |
|---|---|
| torso_link | torso |
| left/right_ankle_roll_link | toeLeft / toeRight |
| left/right_wrist_yaw_link | wristYawLeft / wristYawRight |
| left/right_hip_roll_link | hipRollLeft / hipRollRight |
| left/right_knee_link | shinLeft / shinRight |
| left/right_shoulder_roll_link | shoulderRollLeft / shoulderRollRight |
| left/right_elbow_link | elbowLeft / elbowRight |
| pelvis | pelvis |

Remapped terms: action clip (joint patterns), `events.base_com`,
`rewards.{motion_feet_lin_vel, locomotion_phase_torso_orientation,
stand_still_base_anchor_vel, robot_head_torso_ball_distance,
penalize_self_contact_feet, no_fly, hand_height_penalty,
arm_pitch_same_sign_penalty, feet_slip, feet_contact_force, undesired_contacts,
ee_body_pos_termination_penalty}`, `terminations.ee_body_pos`, plus
`commands.motion.{anchor_body_name, body_names}` and `actions.joint_pos.scale`.

## 4. Motor-parameter comparison (pnd_rl_lab [used] vs instinctMj adam_sp)

`adam.py` uses the pnd_rl_lab column for hip/knee/waist, but **ankle (pitch+roll)
and arm (shoulder/elbow/wrist) now use the instinctMj column** — pnd's arm/ankle
gains were too soft (see the UPDATE below). Kp/Kd diverge sharply on arms/ankles;
effort/velocity for the pnd groups come from the pnd_rl_lab actuator T-N curve
(`Y1` / rated).

| Joint | pnd Kp/Kd (used) | pnd eff/vel (used) | instinctMj Kp/Kd | instinctMj eff/vel |
|---|---|---|---|---|
| hipPitch / knee | 305 / 6.1 | 230 / 15 | 300 / 7 | 230 / 60* |
| hipRoll | 700 / 30 | 160 / 8 | 600 / 10 | 180 / 60* |
| hipYaw | 405 / 6.1 | 105 / 8 | 300 / 2 | 105 / 60* |
| anklePitch | 30 / 3.5 | 40 / 20 | 130 / 3.5 | 80 / 60* |
| ankleRoll | 3 / 0.35 | 12 / 20 | 70 / 2 | 40 / 60* |
| waistRoll/Pitch | 405 / 6.1 | 110 / 8 | 400 / 11 | 150 / 60* |
| waistYaw | 205 / 4.1 | 110 / 8 | 400 / 11 | 150 / 60* |
| shoulderPitch | 18 / 0.9 | 65 / 8 | 60 / 3 | 65 / 60* |
| shoulderRoll/Yaw | 9 / 0.9 | 65 / 8 | 60 / 3 | 65 / 60* |
| elbow | 9 / 0.9 | 30 / 8 | 60 / 3 | 30 / 60* |
| wrist* (instinctMj) | 20 / 1.0 | 6.4 / 20 | 20 / 1.0 | 6.4 / — |

`*` instinctMj velocity was a flat placeholder (60) in the earlier draft, not a
physical limit. pnd_rl_lab velocities are the actuator rated speeds.

Biggest risk if pnd_rl_lab arm/ankle gains prove too soft for a fast kick: the
struck leg / arms may lag the reference. If tracking underfits, the instinctMj
(stiffer) arm/ankle gains are the fallback — swap per-group in `adam.py`.

**UPDATE (done, this is what happened):** pnd_rl_lab's arm/ankle gains *were* too
soft — the soft-gain Stage-1 run died in ~1 s (83% `anchor_pos` terminations, no
kick, reward ~2.4). `adam.py` now uses the **instinctMj ankle + arm gains**
(anklePitch 130, ankleRoll 70, shoulder/elbow 60); hip/knee/waist stay pnd_rl_lab.
Combined with relaxing the `anchor_pos` termination 0.25 -> 0.45, reward went
2.4 -> 22, mean episode length 48 -> 344 steps, and the kick works. Details in
`ADAM_KICK_HANDOFF.md` §1b.

## 5. Open decisions — status

- **Anchor body**: RESOLVED -> `pelvis`. pnd_rl_lab's Adam tracking explicitly
  uses `pelvis` ("not torso_link like G1"), consistent with the NPZ root height
  (~0.9 m). `flat_env_cfg.py` sets `anchor_body_name = ADAM_ANCHOR_BODY`
  ("pelvis") and randomizes pelvis CoM.
- **Tracked hand/wrist link**: our 29-DOF URDF ends the arm at `wristRoll{Left,
  Right}`; there is **no `EE_L_hand`/`EE_R_hand` link** (that exists only in
  pnd_rl_lab's fixed-wrist reference URDF). We track `wristYaw{Left,Right}`
  (matches the YAML `ee_body_pos`). Keep this consistent with Phase-3 conversion.
- **DOF**: training target is the **29-DOF** `adam_inspire` URDF under
  `assets/pnd_description/adam_inspire/` (copied from RoboNaldo's legacy
  `adam_inspire_description`; all 6 wrists `revolute`; matches the kick NPZ).
  This is **not** pnd_rl_lab's `pnd_robots/adam_inspire` URDF, which fixes the
  wrists (`type="fixed"`, 23 DOF) — we do not use that file for training.
- **Init height**: `pos.z=0.89` (pnd_rl_lab default). Reset overwrites from the
  reference; adjust if fallback spawns clip/float.
- **Self-collision termination**: `make_self_collision_termination()` in
  `task_overrides.py` hardcodes the G1 ankle pair. Stage-1 preset disables it
  (`self_collision: false`); re-enable with Adam links (`toeLeft`/`toeRight`)
  when needed.

## 6. Reusable `pnd_rl_lab` resources (Isaac-Lab-native, high value)

`pnd_rl_lab` is a full Isaac Lab RL project for PND/Adam robots — closer to this
repo than instinctMj (mjlab). Worth mining before writing Phase 3/4:

| Resource | Path (in `pnd_rl_lab`) | Use |
|---|---|---|
| Adam Inspire articulation cfg | `assets/robots/pnd.py::PND_ADAM_INSPIRE_CFG` | validated joint names + PD gains (source for `adam.py`) |
| Custom actuator (T-N curve + friction + motor-strength DR) | `assets/robots/pnd_actuators.py` | more realistic actuator model than Implicit PD |
| Full Adam BeyondMimic tracking env | `tasks/mimic/beyond_mimic/robots/pnd_inspire/dance/tracking_env_cfg.py` | reference reward/obs/event/termination/curriculum tuned for Adam (pelvis anchor, arm-tracking rewards) |
| Adam mimic reward impls | `tasks/mimic/beyond_mimic/mdp/rewards.py` | Adam-specific reward functions |
| Example motion NPZ (dance/taichi) | same dir, `*.npz` | reference NPZ format for Phase 3 |

Caveat: pnd_rl_lab's tracking env is 23-DOF and uses `EE_*_hand` links + a
different anchor/reward weighting; adapt names to our 29-DOF URDF when porting.

## 7. Asset layout

Robot assets consolidated under `assets/pnd_description/` (mirrors the
`unitree_description` idea, per-robot subdirs):

```
assets/pnd_description/
  adam_inspire/{urdf,mjcf,usd,meshes,assets}/   # 29-DOF (kick target)
  adam_lite/{urdf,mjcf,meshes,assets}/          # 23-DOF
```

`adam.py` points at `pnd_description/adam_inspire/urdf/adam_inspire.urdf`. Mesh
refs were rewritten (`meshes/`->`../meshes/`, XML `meshdir`->`../assets`) for the
subdir layout. The legacy `adam_inspire_description/` + `adam_lite_description/`
are kept unchanged to avoid breaking other consumers. Note: `assets/` is its own
git repo. Known pre-existing quirk carried over: `adam_inspire/mjcf/scene.xml`
fails to load standalone ("repeated name 'floor'") because the model xml already
defines a floor — load `adam_inspire.xml` directly, or use `adam_lite/scene.xml`.
