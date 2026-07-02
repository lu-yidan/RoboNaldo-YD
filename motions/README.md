# Reference Motions

## `right_kick_reference.csv`

Open-source right-foot kick reference motion used in RoboNaldo.

Source:

- Retarget: GVHMR+GMR.
- Artifact entry: `motion.npz`

Format:

- 612 frames
- 50 Hz
- 36 comma-separated columns per row
- Columns: root position `(x, y, z)`, root quaternion `(x, y, z, w)`, then 29
  Unitree G1 joint positions in the order used by `scripts/csv_to_npz.py`

Convert to training NPZ:

```bash
python scripts/csv_to_npz.py \
  --input_file motions/right_kick_reference.csv \
  --input_fps 50 \
  --output_name right_kick \
  --headless
```

## Adam Inspire (29 DOF) kick motions

The Adam kick reuses the same right-foot kick, retargeted to the Adam Inspire
29-DOF body. Lineage:

```
right_kick_adam_v2_bfs.npz  --(adam_npz_to_npz.py, --rotate_z_deg 180)-->  right_kick_adam.npz
   (raw, -Y facing)                                                          (training input, +Y facing)
```

Files:

- `right_kick_adam_v2_bfs.npz` — raw retargeted reference (`root_pos`, `root_rot`
  WXYZ, `dof_pos`/`dof_vel`; 612 frames @ 50 Hz). Joints are in **BFS** order (see
  `JOINT_NAMES_BFS` in `scripts/adam_npz_to_npz.py`). Travels along world **-Y**
  (forward = body +X, but the initial yaw points body +X at world -Y).
  `right_kick_adam_v2.npz` is an equivalent copy.
- `right_kick_adam.npz` — **training input (current)**. Body-level NPZ
  (`body_pos_w/quat_w/lin_vel_w/ang_vel_w`, `joint_pos/vel`; 611 frames, 29 joints,
  30 bodies) produced from the raw file by kinematic FK replay on
  `ADAM_INSPIRE_CFG`, with a **180° world-Z rotation** so forward = **+Y** (faces
  the ball, matching the ball/target/over-line convention and the deploy
  `freekick_motion.npz`, which also travels +Y).
- `right_kick_adam_posY_rot180_backup.npz` — byte-identical backup of the current
  `right_kick_adam.npz` (+Y).
- `right_kick_adam_negY_backup.npz` — older backup of the pre-rotation **-Y**
  version (robot faced away from the ball); kept for reference only.

Convert (raw -> training NPZ, with the facing fix):

```bash
python scripts/adam_npz_to_npz.py \
  --input_file motions/right_kick_adam_v2_bfs.npz \
  --output_name right_kick_adam \
  --output_dir motions \
  --quat_order wxyz \
  --rotate_z_deg 180 \
  --headless
```
