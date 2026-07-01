"""Convert a retargeted **Adam (Inspire, 29 DOF)** motion NPZ into the body-level
training NPZ format consumed by ``MotionLoader`` / ``MotionCommand``.

Input NPZ (e.g. ``motions/right_kick_adam_v2_bfs.npz``) holds the *reference*:
    fps       : (1,)          output/input fps (Hz)
    root_pos  : (T, 3)        pelvis world position (m)
    root_rot  : (T, 4)        pelvis world orientation, WXYZ by default
    dof_pos   : (T, 29)       joint positions in **BFS** order (see JOINT_NAMES_BFS)
    dof_vel   : (T, 29)       joint velocities (unused; recomputed from pos)

Output NPZ (``--output_name``) holds full-body kinematics logged from Isaac Lab
FK while kinematically replaying the reference on ``ADAM_INSPIRE_CFG``:
    fps, joint_pos, joint_vel,
    body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w   (all Adam bodies)

The body dimension/order matches ``robot.data.body_*`` of ``ADAM_INSPIRE_CFG``,
which is exactly what ``MotionLoader`` indexes via ``robot.find_bodies(...)``.

Must run inside the Isaac Lab / Isaac Sim environment (needs ``isaaclab``).

Example (inside the container):
    python scripts/adam_npz_to_npz.py \
        --input_file motions/right_kick_adam_v2_bfs.npz \
        --output_name right_kick_adam --output_dir motions --headless
"""

import argparse
import numpy as np
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay an Adam reference NPZ and export the training NPZ.")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input Adam reference NPZ.")
parser.add_argument("--input_fps", type=int, default=None, help="Override input fps (default: read from NPZ 'fps').")
parser.add_argument("--output_fps", type=int, default=50, help="Output motion fps.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="Inclusive 1-indexed frame range START END. Default: all frames.",
)
parser.add_argument("--output_name", type=str, required=True, help="Output NPZ name (without extension).")
parser.add_argument("--output_dir", type=Path, default=Path("motions"), help="Directory for the output NPZ.")
parser.add_argument(
    "--quat_order",
    type=str,
    default="wxyz",
    choices=["wxyz", "xyzw"],
    help="Quaternion layout of root_rot in the input NPZ (default wxyz; Isaac Lab wants wxyz).",
)
parser.add_argument(
    "--turn_y_axis",
    action="store_true",
    help="Mirror the motion about the x-axis (+y -> -y). Off by default for Adam.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.adam import ADAM_INSPIRE_CFG

# BFS joint order matching the input NPZ 'dof_pos' columns. Derived from the
# adam_inspire URDF kinematic tree (breadth-first from pelvis) and cross-checked
# against per-joint URDF limits (BFS -> 0.15% out-of-limit vs DFS -> 6.8%).
JOINT_NAMES_BFS = [
    "hipPitch_Left", "hipPitch_Right", "waistRoll",
    "hipRoll_Left", "hipRoll_Right", "waistPitch",
    "hipYaw_Left", "hipYaw_Right", "waistYaw",
    "kneePitch_Left", "kneePitch_Right",
    "shoulderPitch_Left", "shoulderPitch_Right",
    "anklePitch_Left", "anklePitch_Right",
    "shoulderRoll_Left", "shoulderRoll_Right",
    "ankleRoll_Left", "ankleRoll_Right",
    "shoulderYaw_Left", "shoulderYaw_Right",
    "elbow_Left", "elbow_Right",
    "wristYaw_Left", "wristYaw_Right",
    "wristPitch_Left", "wristPitch_Right",
    "wristRoll_Left", "wristRoll_Right",
]


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Scene with a ground plane, sky light and the Adam articulation."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = ADAM_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class AdamMotionLoader:
    """Load an Adam reference NPZ, resample to output_fps, and derive velocities.

    Mirrors ``scripts/csv_to_npz.py::MotionLoader`` but reads the NPZ arrays
    (root_pos/root_rot/dof_pos) instead of CSV columns.
    """

    def __init__(self, motion_file, input_fps, output_fps, device, frame_range, quat_order, turn_y_axis):
        self.motion_file = motion_file
        self.output_fps = output_fps
        self.output_dt = 1.0 / output_fps
        self.device = device
        self.frame_range = frame_range
        self.quat_order = quat_order
        self.turn_y_axis = turn_y_axis
        self.current_idx = 0

        data = np.load(motion_file)
        npz_fps = int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50
        self.input_fps = int(input_fps) if input_fps is not None else npz_fps
        self.input_dt = 1.0 / self.input_fps

        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_rot = np.asarray(data["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        if frame_range is not None:
            s, e = frame_range[0] - 1, frame_range[1]
            root_pos, root_rot, dof_pos = root_pos[s:e], root_rot[s:e], dof_pos[s:e]

        if dof_pos.shape[1] != len(JOINT_NAMES_BFS):
            raise ValueError(f"dof_pos has {dof_pos.shape[1]} cols, expected {len(JOINT_NAMES_BFS)}.")

        base_pos = torch.from_numpy(root_pos).to(device)
        base_rot = torch.from_numpy(root_rot).to(device)
        if quat_order == "xyzw":
            base_rot = base_rot[:, [3, 0, 1, 2]]  # -> wxyz
        # normalize quaternions defensively
        base_rot = base_rot / base_rot.norm(dim=-1, keepdim=True)

        if turn_y_axis:
            base_pos[:, 1] = -base_pos[:, 1]
            q = base_rot.clone()
            q[:, 0] = -base_rot[:, 3]
            q[:, 1] = -base_rot[:, 2]
            q[:, 2] = +base_rot[:, 1]
            q[:, 3] = +base_rot[:, 0]
            base_rot = q

        self.motion_base_poss_input = base_pos
        self.motion_base_rots_input = base_rot
        self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(device)
        self.input_frames = dof_pos.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded ({motion_file}), duration {self.duration:.3f}s, frames {self.input_frames}, "
              f"in_fps {self.input_fps}, quat {quat_order}, turn_y {turn_y_axis}")

        self._interpolate_motion()
        self._compute_velocities()

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        i0, i1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(self.motion_base_poss_input[i0], self.motion_base_poss_input[i1], blend[:, None])
        self.motion_base_rots = self._slerp(self.motion_base_rots_input[i0], self.motion_base_rots_input[i1], blend)
        self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[i0], self.motion_dof_poss_input[i1], blend[:, None])
        print(f"Motion interpolated -> {self.output_frames} frames @ {self.output_fps} fps")

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], blend[i])
        return out

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        i0 = (phase * (self.input_frames - 1)).floor().long()
        i1 = torch.minimum(i0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - i0
        return i0, i1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self):
        i = self.current_idx
        state = (
            self.motion_base_poss[i : i + 1],
            self.motion_base_rots[i : i + 1],
            self.motion_base_lin_vels[i : i + 1],
            self.motion_base_ang_vels[i : i + 1],
            self.motion_dof_poss[i : i + 1],
            self.motion_dof_vels[i : i + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(sim, scene, joint_names):
    motion = AdamMotionLoader(
        motion_file=args_cli.input_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
        quat_order=args_cli.quat_order,
        turn_y_axis=args_cli.turn_y_axis,
    )

    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False

    while simulation_app.is_running():
        (
            (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel),
            reset_flag,
        ) = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = base_rot
        root_states[:, 7:10] = base_lin_vel
        root_states[:, 10:] = base_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = dof_pos
        joint_vel[:, robot_joint_indexes] = dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
                log[k] = np.stack(log[k], axis=0)
            args_cli.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = args_cli.output_dir / f"{args_cli.output_name}.npz"
            np.savez(output_path, **log)
            print(f"[INFO]: Motion saved to {output_path}")
            print(f"[INFO]: joint_pos {log['joint_pos'].shape}, body_pos_w {log['body_pos_w'].shape}")
            print(f"[INFO]: robot body order: {list(robot.body_names)}")
            exit(0)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene, joint_names=JOINT_NAMES_BFS)


if __name__ == "__main__":
    main()
    simulation_app.close()
