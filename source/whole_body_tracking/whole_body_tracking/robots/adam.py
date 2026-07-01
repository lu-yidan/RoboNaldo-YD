"""PND Adam (Inspire, 29 DOF) articulation config for whole-body tracking.

Motor PD / effort / velocity values follow the **Isaac-Lab-native** reference
``pnd_rl_lab`` (``assets/robots/pnd.py::PND_ADAM_INSPIRE_CFG`` +
``pnd_actuators.py``), which targets this same ``adam_inspire`` URDF and is the
most directly applicable source for Isaac Lab. See ``README_ADAM_HAZARDS.md``
for the full comparison against ``instinctMj`` (mjlab) ``adam_sp.py``.

Sourcing notes:
  * Kp / Kd  -> pnd_rl_lab ``PND_ADAM_INSPIRE_CFG.actuators``.
  * effort   -> pnd_rl_lab ``pnd_actuators`` peak torque ``Y1``.
  * velocity -> pnd_rl_lab actuator docstring rated speed (rad/s).
  * armature -> ``instinctMj`` adam_sp (pnd_rl_lab AdamInspire actuators do not
    specify armature; instinctMj values are physically derived, so kept here).
  * wrist    -> ``instinctMj`` adam_sp. pnd_rl_lab treats the Inspire wrists as
    *fixed* (23 DOF); our URDF + motion data are 29 DOF, so the 3 wrist joints
    per arm use instinctMj gains as the only available reference.

Joint naming (URDF): camelCase + underscore side, e.g. ``hipPitch_Left``,
``kneePitch_Right``, ``elbow_Left``, ``wristYaw_Left``, ``waistRoll``.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ADAM_INSPIRE_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/pnd_description/adam_inspire/urdf/adam_inspire.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    # Default pose from pnd_rl_lab PND_ADAM_INSPIRE_CFG (root == pelvis here).
    # NOTE: reset events overwrite this from the motion reference; kept as a fallback.
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.89),
        joint_pos={
            "hipPitch_.*": -0.32,
            "hipRoll_.*": 0.0,
            "hipYaw_Left": -0.18,
            "hipYaw_Right": 0.18,
            "kneePitch_.*": 0.66,
            "anklePitch_.*": -0.39,
            "ankleRoll_.*": 0.0,
            "waistRoll": 0.0,
            "waistPitch": 0.0,
            "waistYaw": 0.0,
            "shoulderPitch_.*": 0.0,
            "shoulderRoll_Left": 0.1,
            "shoulderRoll_Right": -0.1,
            "shoulderYaw_.*": 0.0,
            "elbow_.*": -0.3,
            "wrist.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # hipPitch + kneePitch (pnd_rl_lab: Kp305 Kd6.1, Y1=230, ~15 rad/s)
        "hip_knee_pitch": ImplicitActuatorCfg(
            joint_names_expr=["hipPitch_.*", "kneePitch_.*"],
            effort_limit_sim=230.0,
            velocity_limit_sim=15.0,
            stiffness=305.0,
            damping=6.1,
            armature=0.13426,
        ),
        # hipRoll (pnd_rl_lab: Kp700 Kd30, Y1=160, ~8 rad/s)
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=["hipRoll_.*"],
            effort_limit_sim=160.0,
            velocity_limit_sim=8.0,
            stiffness=700.0,
            damping=30.0,
            armature=0.281573,
        ),
        # hipYaw (pnd_rl_lab: Kp405 Kd6.1, Y1=105, ~8 rad/s)
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=["hipYaw_.*"],
            effort_limit_sim=105.0,
            velocity_limit_sim=8.0,
            stiffness=405.0,
            damping=6.1,
            armature=0.23409,
        ),
        # anklePitch (pnd_rl_lab: Kp30 Kd3.5, Y1=40, ~20 rad/s)
        "ankle_pitch": ImplicitActuatorCfg(
            joint_names_expr=["anklePitch_.*"],
            effort_limit_sim=40.0,
            velocity_limit_sim=20.0,
            stiffness=30.0,
            damping=3.5,
            armature=0.0549,
        ),
        # ankleRoll (pnd_rl_lab: Kp3 Kd0.35, Y1=12, ~20 rad/s)
        "ankle_roll": ImplicitActuatorCfg(
            joint_names_expr=["ankleRoll_.*"],
            effort_limit_sim=12.0,
            velocity_limit_sim=20.0,
            stiffness=3.0,
            damping=0.35,
            armature=0.0549,
        ),
        # waist roll/pitch (Kp405 Kd6.1) + yaw (Kp205 Kd4.1); Y1=110, ~8 rad/s
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waistRoll", "waistPitch", "waistYaw"],
            effort_limit_sim=110.0,
            velocity_limit_sim=8.0,
            stiffness={"waistRoll": 405.0, "waistPitch": 405.0, "waistYaw": 205.0},
            damping={"waistRoll": 6.1, "waistPitch": 6.1, "waistYaw": 4.1},
            armature=0.23409,
        ),
        # shoulder pitch (Kp18) / roll+yaw (Kp9), Kd0.9; Y1=65, ~8 rad/s
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulderPitch_.*", "shoulderRoll_.*", "shoulderYaw_.*"],
            effort_limit_sim=65.0,
            velocity_limit_sim=8.0,
            stiffness={"shoulderPitch_.*": 18.0, "shoulderRoll_.*": 9.0, "shoulderYaw_.*": 9.0},
            damping=0.9,
            armature=0.01,
        ),
        # elbow (pnd_rl_lab: Kp9 Kd0.9, Y1=30, ~8 rad/s)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_.*"],
            effort_limit_sim=30.0,
            velocity_limit_sim=8.0,
            stiffness=9.0,
            damping=0.9,
            armature=0.01,
        ),
        # wrist yaw/pitch/roll -> instinctMj adam_sp (pnd_rl_lab has no wrist actuator; 23 DOF)
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wristYaw_.*", "wristPitch_.*", "wristRoll_.*"],
            effort_limit_sim=6.4,
            velocity_limit_sim=20.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
        ),
    },
)

# BeyondMimic action scale: 0.25 * effort / stiffness per joint expr (matches G1_ACTION_SCALE).
ADAM_ACTION_SCALE = {}
for a in ADAM_INSPIRE_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            ADAM_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
