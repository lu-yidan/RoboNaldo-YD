"""Adam (Inspire, 29 DOF) tracking task config.

Ported from ``config/g1/flat_env_cfg.py``. The shared ``TrackingEnvCfg`` base
hardcodes many G1 body/joint names (MJCF-style), so ``_apply_adam_robot_settings``
remaps every config-overridable name onto the Adam URDF naming
(camelCase + trailing side, e.g. ``hipPitch_Left``, ``toeRight``, ``torso``).

IMPORTANT — this task does NOT run end-to-end yet. Some G1 names are hardcoded
inside ``mdp`` function bodies (not reachable from config) and must be patched
first. See ``config/adam/README_ADAM_HAZARDS.md`` for the full list.
"""

from isaaclab.utils import configclass

from whole_body_tracking.robots.adam import ADAM_ACTION_SCALE, ADAM_INSPIRE_CFG
from whole_body_tracking.tasks.tracking.tracking_env_cfg import (
    TrackingEnvCfg,
    TrackingWorldPosEnvCfg,
)

# ---------------------------------------------------------------------------
# G1 (MJCF) -> Adam (URDF) name mapping used across the remap below.
# ---------------------------------------------------------------------------
# Anchor body: pnd_rl_lab PND_ADAM_INSPIRE tracking uses "pelvis" as the anchor
# (explicitly "not torso_link like G1"). torso is still tracked + used for the
# torso-orientation reward, but the motion anchor / CoM randomization is pelvis.
ADAM_ANCHOR_BODY = "pelvis"
ADAM_TORSO_BODY = "torso"
ADAM_FEET_BODIES = ["toeLeft", "toeRight"]
ADAM_WRIST_BODIES = ["wristYawLeft", "wristYawRight"]
ADAM_EE_BODIES = ADAM_FEET_BODIES + ADAM_WRIST_BODIES  # order: L/R foot, L/R wrist

# Tracked bodies, mirroring G1_TRACKED_BODY_NAMES order/semantics.
ADAM_TRACKED_BODY_NAMES = [
    "pelvis",
    "hipRollLeft",
    "shinLeft",       # G1 left_knee_link  (knee child link)
    "toeLeft",        # G1 left_ankle_roll_link (foot)
    "hipRollRight",
    "shinRight",
    "toeRight",
    "torso",          # G1 torso_link (anchor)
    "shoulderRollLeft",
    "elbowLeft",
    "wristYawLeft",
    "shoulderRollRight",
    "elbowRight",
    "wristYawRight",
]

# Per-joint action clip, remapped from the base ActionsCfg (G1 joint patterns).
ADAM_ACTION_CLIP = {
    "hipYaw_.*": [-6.0, 6.0],
    "hipRoll_.*": [-6.0, 6.0],
    "hipPitch_.*": [-6.0, 6.0],
    "kneePitch_.*": [-6.0, 6.0],
    "anklePitch_.*": [-3.0, 3.0],
    "ankleRoll_.*": [-0.6, 0.6],
    "waistYaw": [-3.0, 3.0],
    "waistRoll": [-1.2, 1.2],
    "waistPitch": [-1.2, 1.2],
    "shoulderPitch_.*": [-2.0, 2.0],
    "shoulderRoll_.*": [-3.0, 3.0],
    "shoulderYaw_.*": [-3.0, 3.0],
    "elbow_.*": [-3.0, 3.0],
    "wristRoll_.*": [-3.0, 3.0],
    "wristPitch_.*": [-3.0, 3.0],
    "wristYaw_.*": [-3.0, 3.0],
}

# Undesired-contact regex: every robot body EXCEPT feet + wrists (Adam names).
ADAM_UNDESIRED_CONTACT_REGEX = (
    r"^(?!toeLeft$)(?!toeRight$)(?!wristYawLeft$)(?!wristYawRight$).+$"
)

# Arm joint patterns for arm_default_pose_penalty (base default is G1-style and
# matches no Adam joint -> find_joints would raise). These are the 14 arm DOFs.
ADAM_ARM_JOINT_PATTERNS = [
    "shoulderPitch_.*",
    "shoulderRoll_.*",
    "shoulderYaw_.*",
    "elbow_.*",
    "wristRoll_.*",
    "wristPitch_.*",
    "wristYaw_.*",
]

# Non-kicking (weak) foot body for penalize_weak_foot_contact. Main foot is
# toeRight (YAML main_foot_name), so the weak foot is toeLeft.
ADAM_WEAK_FOOT_BODY = "toeLeft"


def _set_param(term, key, value):
    """Safely set a param on a manager term if the term exists."""
    if term is not None and getattr(term, "params", None) is not None:
        term.params[key] = value


def _apply_adam_robot_settings(env_cfg) -> None:
    # --- core robot wiring (mirrors _apply_g1_robot_settings) ---
    env_cfg.scene.robot = ADAM_INSPIRE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    env_cfg.actions.joint_pos.scale = ADAM_ACTION_SCALE
    env_cfg.actions.joint_pos.clip = ADAM_ACTION_CLIP
    env_cfg.commands.motion.anchor_body_name = ADAM_ANCHOR_BODY  # pelvis (per pnd_rl_lab)
    env_cfg.commands.motion.body_names = ADAM_TRACKED_BODY_NAMES

    # --- events ---
    _set_param(env_cfg.events.base_com, "asset_cfg", _replace_body(env_cfg.events.base_com, ADAM_ANCHOR_BODY))

    # --- rewards: body-name params remapped to Adam URDF ---
    r = env_cfg.rewards
    _set_param(r.motion_feet_lin_vel, "body_names", ADAM_FEET_BODIES)
    _set_param(r.locomotion_phase_torso_orientation, "torso_body_name", ADAM_TORSO_BODY)
    _set_param(r.stand_still_base_anchor_vel, "body_names", ADAM_EE_BODIES)
    _set_param(r.robot_head_torso_ball_distance, "body_names", [ADAM_TORSO_BODY])
    _set_param(r.penalize_self_contact_feet, "body_names", ADAM_FEET_BODIES)
    _set_param(r.no_fly, "body_names", ADAM_FEET_BODIES)
    _set_param(r.hand_height_penalty, "body_names", ADAM_WRIST_BODIES)
    _set_param(r.arm_pitch_same_sign_penalty, "left_joint_name", "shoulderPitch_Left")
    _set_param(r.arm_pitch_same_sign_penalty, "right_joint_name", "shoulderPitch_Right")
    # Formerly-hardcoded G1 names now fed from config (see mdp/rewards.py patches):
    _set_param(getattr(r, "arm_default_pose", None), "arm_joint_names", ADAM_ARM_JOINT_PATTERNS)
    _set_param(getattr(r, "penalize_weak_foot_contact", None), "weak_foot_name", ADAM_WEAK_FOOT_BODY)

    # feet_slip / feet_contact_force / undesired_contacts carry SceneEntityCfg sensors.
    _remap_sensor_bodies(r.feet_slip, "sensor_cfg", ADAM_FEET_BODIES)
    _set_param(r.feet_slip, "body_names", ADAM_FEET_BODIES)
    _remap_sensor_bodies(r.feet_contact_force, "sensor_cfg", ADAM_FEET_BODIES)
    _remap_sensor_bodies(r.undesired_contacts, "sensor_cfg", [ADAM_UNDESIRED_CONTACT_REGEX])

    # ee_body_pos_termination_penalty: remap body_names only. The per-body
    # `threshold` dict is keyed by body name and is supplied by the Adam YAML
    # preset (already Adam-keyed), matching how G1 sources it from YAML.
    _set_param(r.ee_body_pos_termination_penalty, "body_names", ADAM_EE_BODIES)

    # --- terminations ---
    ee_term = getattr(env_cfg.terminations, "ee_body_pos", None)
    _set_param(ee_term, "body_names", ADAM_EE_BODIES)


def _replace_body(term, body_name):
    """Return a SceneEntityCfg copy of an existing asset_cfg with new body_names."""
    asset_cfg = term.params["asset_cfg"]
    asset_cfg.body_names = body_name
    return asset_cfg


def _remap_sensor_bodies(term, key, body_names):
    if term is None or getattr(term, "params", None) is None:
        return
    sensor_cfg = term.params.get(key)
    if sensor_cfg is not None:
        sensor_cfg.body_names = body_names


@configclass
class AdamFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_adam_robot_settings(self)


@configclass
class AdamFlatBodyFrameEnvCfg(TrackingWorldPosEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_adam_robot_settings(self)
