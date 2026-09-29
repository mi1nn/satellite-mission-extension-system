"""Astrobee docking damping assist: grab the Ares1 probe near its root and damp the
arm + payload swing with thrust while the probe is brought onto the docking axis.

Why: the Canadarm3 + 3 t MEP has a lightly damped mode (period ~20 s, see
`coupled_dock.py`). Measured in `full_6dof` (coupled_predictive): the probe-tip lateral
error swings by ~0.1-0.2 m at ~0.03-0.1 m/s through PRE_DOCK_APPROACH and
POSITION_ATTITUDE_ALIGN. Damping it from the arm side (`docking.swing_damping`) was
tried and destabilised the drives, so the damper here is external and collocated.

Physical scale (why the thrust limit is NOT a real Astrobee's): on a 1-D model of that
mode (3000 kg, 20 s, `tests/test_astrobee_assist.py`) a saturated damper takes 4 F / k
off the amplitude per cycle (k = m w^2 = 296 N/m, i.e. ~8 mm per cycle at 0.6 N). Over
an ~80 s alignment window a 100 mm swing ends at ~78 mm undamped, ~49 mm with a real
NASA Astrobee's ~0.6 N, ~5 mm with 2 N and ~2 mm with 5 N; a 200 mm swing needs ~5 N.
`max_force_n` (default 5 N) therefore models a hypothetical *scaled-up* Astrobee-class
free-flyer; say so wherever results are shown.

Sequence (one shot per run; `DampingAssist.step`):

    ASTROBEE_ASSIST_RENDEZVOUS      fly to a standby point beside the grip point
                                    (`standby_distance_m`, perpendicular to the probe)
    ASTROBEE_ASSIST_FINAL_APPROACH  close in slowly to `grip_standoff_m`, velocity matched
    ASTROBEE_ASSIST_GRASPED         attached, no thrust (mission not in a damping state yet)
    ASTROBEE_DAMPING_ASSIST         attached, F = clip(-c (v_grip - v_ref), F_max)
    ASTROBEE_ASSIST_RELEASE         let go and back off to the standby distance
    ASTROBEE_ASSIST_STANDBY         hold beside the MEP (camera on the probe)

During `standby_states` (the MEP is still free-floating: arm deploy .. capture) it only
flies to the standby point and waits there; the final approach and the grasp need the
MEP held by the arm (`engage_states` or `damping_states`). The Astrobee starts on its
observation ring (~80 m from the probe) and HOLDING -> PRE_DOCK_APPROACH takes ~2 s, so
without the early start it would arrive after the damping window. Release as soon as the
state is in none of the lists (so Z_APPROACH, i.e. before the probe reaches the nozzle,
and any failure). Released means released for the rest of the run: a docking rollback to the
aligning states does not re-grasp.

Model (simplifications, all deliberate):
- The Astrobee stays kinematic (visual only, no collider). Its flight is acceleration
  and speed limited (`fly_towards`); once grasped it rides rigidly with the MEP.
- The grasp is not a physics joint: the thrust is applied to the MEP as an external
  force at the grip point (force at the COM + r x F torque, `vision_capture_demo`).
  The Astrobee's own mass (~10 kg vs 3000 kg) is neglected.
- v_ref is the arm reference velocity at the grip point, so only the deviation from the
  commanded motion (the swing) is damped, never the approach itself.

Frames: world W (Z up, metres, m/s, N). MEP body frame M as `docking.py` (probe tip and
direction in M). Pure numpy (`project/tests/test_astrobee_assist.py`).
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .frames import Frame

ASSIST_PHASES = (
    "ASTROBEE_ASSIST_RENDEZVOUS",
    "ASTROBEE_ASSIST_FINAL_APPROACH",
    "ASTROBEE_ASSIST_GRASPED",
    "ASTROBEE_DAMPING_ASSIST",
    "ASTROBEE_ASSIST_RELEASE",
    "ASTROBEE_ASSIST_STANDBY",
)
ATTACHED_PHASES = frozenset({"ASTROBEE_ASSIST_GRASPED", "ASTROBEE_DAMPING_ASSIST"})

# Mission states (`vision_capture_demo.State`) in which the probe is at or inside the
# nozzle: never allowed in `engage_states` / `damping_states`
CONTACT_STATES = frozenset({"Z_APPROACH", "FINAL_INSERTION", "DOCK_READY", "DOCKED", "DOCK_HOLDING"})
# Every mission state name the lists may use (checked against the demo's `State` enum
# by the tests; kept here because the demo module imports Isaac Lab)
MISSION_STATES = frozenset({
    "ARM_DEPLOY", "SEARCH", "TAG_DETECTED", "POSE_ESTIMATED", "PREDICTING", "APPROACHING", "SLOW_APPROACH",
    "CAPTURE_ATTEMPT", "CAPTURED", "HOLDING", "RETREAT", "DOCK_TARGET_ACQUIRE", "PRE_DOCK_APPROACH", "XY_ALIGN", "POSITION_ATTITUDE_ALIGN",
    "ORIENTATION_ALIGN", "ALIGNMENT_CHECK", "CLIENT_RELEASE", "CLIENT_CRUISE", "CHASE", "VELOCITY_MATCHING",
    "RENDEZVOUS",
})


##############
### Config ###
##############


@dataclass
class AstrobeeAssistCfg:
    """`astrobee.assist:` section of `vision_capture.yaml`."""

    enabled: bool = False
    # Thrust limit [N]. NOT a real Astrobee (~0.6 N): see the module docstring
    max_force_n: float = 5.0
    # Damper gain c [N s/m]: F = -c (v_grip - v_ref). 300 adds a damping ratio of
    # c / (2 m w) = 0.16 on the 3 t, 20 s mode and saturates F_max at 0.017 m/s
    damping_n_s_per_m: float = 300.0
    # Grip point on the probe rod: 0 = root (MEP body side), 1 = tip
    grip_fraction_from_root: float = 0.15
    # Astrobee body centre to grip point once grasped [m] (display-scaled model + arm)
    grip_standoff_m: float = 1.0
    # Standby point: this far from the grip point, beside the rod [m]
    standby_distance_m: float = 3.0
    standby_tolerance_m: float = 0.3
    # Flight limits relative to the target motion [m/s], [m/s^2]
    transit_speed_mps: float = 2.0
    final_approach_speed_mps: float = 0.1
    retreat_speed_mps: float = 0.1
    max_accel_mps2: float = 0.2
    # Grasp gate: position [m] and relative speed [m/s] to the grip standoff point
    grasp_position_tol_m: float = 0.02
    grasp_speed_tol_mps: float = 0.01
    # Mission states: fly to the standby point and wait (MEP not held yet) during these ...
    standby_states: List[str] = field(default_factory=lambda: [
        "ARM_DEPLOY", "SEARCH", "TAG_DETECTED", "POSE_ESTIMATED", "PREDICTING", "APPROACHING", "SLOW_APPROACH",
        "CAPTURE_ATTEMPT", "CAPTURED",
    ])
    # ... fly in and grasp during these ...
    engage_states: List[str] = field(default_factory=lambda: [
        "HOLDING", "RETREAT", "CLIENT_RELEASE", "CLIENT_CRUISE", "CHASE", "VELOCITY_MATCHING", "RENDEZVOUS",
        "DOCK_TARGET_ACQUIRE",
    ])
    # ... thrust (damp) during these; release in any other state (before contact)
    damping_states: List[str] = field(default_factory=lambda: [
        "PRE_DOCK_APPROACH", "XY_ALIGN", "ORIENTATION_ALIGN", "POSITION_ATTITUDE_ALIGN", "ALIGNMENT_CHECK",
    ])


def validate_assist_cfg(cfg: AstrobeeAssistCfg):
    if cfg.max_force_n <= 0.0 or cfg.damping_n_s_per_m < 0.0:
        raise ValueError("astrobee.assist.max_force_n must be > 0 and damping_n_s_per_m >= 0")
    if not 0.0 <= cfg.grip_fraction_from_root <= 1.0:
        raise ValueError("astrobee.assist.grip_fraction_from_root must be in [0, 1]")
    if cfg.grip_standoff_m <= 0.0 or cfg.standby_distance_m <= cfg.grip_standoff_m or cfg.standby_tolerance_m <= 0.0:
        raise ValueError("astrobee.assist: need 0 < grip_standoff_m < standby_distance_m and standby_tolerance_m > 0")
    if min(cfg.transit_speed_mps, cfg.final_approach_speed_mps, cfg.retreat_speed_mps, cfg.max_accel_mps2) <= 0.0:
        raise ValueError("astrobee.assist speeds and max_accel_mps2 must be > 0")
    if cfg.grasp_position_tol_m <= 0.0 or cfg.grasp_speed_tol_mps <= 0.0:
        raise ValueError("astrobee.assist grasp tolerances must be > 0")
    for name in ("standby_states", "engage_states", "damping_states"):
        states = set(getattr(cfg, name))
        if states & CONTACT_STATES:
            raise ValueError(f"astrobee.assist.{name} must not contain contact states {sorted(states & CONTACT_STATES)}")
        if not states <= MISSION_STATES:
            raise ValueError(f"astrobee.assist.{name}: unknown mission states {sorted(states - MISSION_STATES)}")
    if not cfg.damping_states:
        raise ValueError("astrobee.assist.damping_states must not be empty")


###############
### Physics ###
###############


def clip_norm(v, limit: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v * (limit / n) if n > limit else v.copy()


def damping_force(grip_velocity, reference_velocity, damping: float, max_force: float) -> np.ndarray:
    """F = clip(-c (v_grip - v_ref), F_max) [N, world]: a thrust-limited viscous damper
    on the deviation of the grip point from the arm's commanded motion."""
    err = np.asarray(grip_velocity, dtype=float) - np.asarray(reference_velocity, dtype=float)
    return clip_norm(-float(damping) * err, float(max_force))


def grip_point_in_mep(tip, direction, length: float, fraction_from_root: float) -> np.ndarray:
    """Grip point on the probe rod in the MEP frame [m]: root + fraction * length."""
    d = np.asarray(direction, dtype=float)
    root = np.asarray(tip, dtype=float) - float(length) * d
    return root + float(fraction_from_root) * float(length) * d


def fly_towards(pos, vel, target, target_vel, dt: float, max_rel_speed: float,
                max_accel: float) -> Tuple[np.ndarray, np.ndarray]:
    """One step of acceleration-limited flight onto a moving target point.

    Desired velocity = target velocity + a relative closing velocity that is limited by
    `max_rel_speed` and by the braking distance sqrt(2 a d); the change of velocity per
    step is limited by `max_accel` [m/s^2]."""
    pos, vel = np.asarray(pos, dtype=float), np.asarray(vel, dtype=float)
    e = np.asarray(target, dtype=float) - pos
    d = float(np.linalg.norm(e))
    closing = min(float(max_rel_speed), math.sqrt(2.0 * max_accel * d), d / max(dt, 1e-9))
    v_des = np.asarray(target_vel, dtype=float) + (e / d * closing if d > 1e-12 else 0.0)
    vel = vel + clip_norm(v_des - vel, max_accel * dt)
    return pos + vel * dt, vel


###############
### Planner ###
###############


@dataclass
class MepContext:
    """What the mission side hands the assist each step (world frame)."""

    mep: Frame  # MEP body frame in W
    grip_in_mep: np.ndarray  # grip point, MEP frame [m]
    axis_in_mep: np.ndarray  # probe direction (unit), MEP frame
    grip_velocity: np.ndarray  # measured velocity of the grip point [m/s]
    reference_velocity: np.ndarray  # arm reference velocity at the grip point [m/s]


@dataclass
class AssistOutput:
    phase: str
    pos: np.ndarray  # Astrobee body centre [m, W]
    aim: np.ndarray  # camera aim point [m, W]
    force_w: Optional[np.ndarray]  # thrust on the MEP at the grip point [N, W]; None = none
    grip_w: np.ndarray  # grip point [m, W]


class DampingAssist:
    """State machine of the assist (see the module docstring). `step` returns None while
    the assist has not engaged, so the caller keeps its own (observation) flight."""

    def __init__(self, cfg: AstrobeeAssistCfg):
        self.cfg = cfg
        self.phase: Optional[str] = None
        self._vel = np.zeros(3)
        self._side_in_mep: Optional[np.ndarray] = None  # unit, perpendicular to the rod
        self._t_grasp: Optional[float] = None
        self._grasp_rel_speed = math.nan
        self._damping_s = 0.0
        self._energy = 0.0
        self._max_force = 0.0
        self._impulse = 0.0

    def _goto(self, phase: str, t: float, detail: str = ""):
        if phase != self.phase:
            self.phase = phase
            print(f"[ASTROBEE] {phase}  (t={t:.2f} s){'  ' + detail if detail else ''}", flush=True)

    def _side_point(self, ctx: MepContext, distance: float) -> np.ndarray:
        return ctx.mep.point(ctx.grip_in_mep + distance * self._side_in_mep)

    def _choose_side(self, ctx: MepContext, pos: np.ndarray):
        """Approach side: from the grip point towards the Astrobee, perpendicular to the rod
        (so it never sits in front of the tip), frozen in the MEP frame."""
        a = np.asarray(ctx.axis_in_mep, dtype=float) / np.linalg.norm(ctx.axis_in_mep)
        r = ctx.mep.rot.T @ (np.asarray(pos, dtype=float) - ctx.mep.point(ctx.grip_in_mep))
        s = r - float(r @ a) * a
        if np.linalg.norm(s) < 1e-6:
            s = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        self._side_in_mep = s / np.linalg.norm(s)

    def step(self, t: float, dt: float, mission_state: Optional[str], pos, vel,
             ctx: Optional[MepContext]) -> Optional[AssistOutput]:
        c = self.cfg
        if not c.enabled or ctx is None:
            return None
        may_grasp = mission_state in c.engage_states or mission_state in c.damping_states
        active = may_grasp or mission_state in c.standby_states
        if self.phase is None:
            if not active:
                return None
            self._vel = np.asarray(vel, dtype=float).copy()
            self._choose_side(ctx, pos)
            self._goto("ASTROBEE_ASSIST_RENDEZVOUS", t, f"mission {mission_state}")
        grip = ctx.mep.point(ctx.grip_in_mep)
        pos = np.asarray(pos, dtype=float)
        if not active and self.phase not in ("ASTROBEE_ASSIST_RELEASE", "ASTROBEE_ASSIST_STANDBY"):
            self._goto("ASTROBEE_ASSIST_RELEASE", t, f"mission {mission_state}: letting go before contact")
        force = None
        if self.phase == "ASTROBEE_ASSIST_RENDEZVOUS":
            target = self._side_point(ctx, c.standby_distance_m)
            pos, self._vel = fly_towards(pos, self._vel, target, ctx.grip_velocity, dt, c.transit_speed_mps, c.max_accel_mps2)
            if may_grasp and float(np.linalg.norm(target - pos)) < c.standby_tolerance_m:
                self._goto("ASTROBEE_ASSIST_FINAL_APPROACH", t)
        elif self.phase == "ASTROBEE_ASSIST_FINAL_APPROACH":
            target = self._side_point(ctx, c.grip_standoff_m)
            pos, self._vel = fly_towards(pos, self._vel, target, ctx.grip_velocity, dt,
                                         c.final_approach_speed_mps, c.max_accel_mps2)
            rel = float(np.linalg.norm(self._vel - ctx.grip_velocity))
            if float(np.linalg.norm(target - pos)) < c.grasp_position_tol_m and rel < c.grasp_speed_tol_mps:
                self._t_grasp, self._grasp_rel_speed = t, rel
                self._goto("ASTROBEE_ASSIST_GRASPED", t, f"probe root held, relative speed {rel * 1000:.1f} mm/s")
        if self.phase in ATTACHED_PHASES:
            pos = self._side_point(ctx, c.grip_standoff_m)
            self._vel = np.asarray(ctx.grip_velocity, dtype=float).copy()
            if mission_state in c.damping_states:
                self._goto("ASTROBEE_DAMPING_ASSIST", t, f"F_max {c.max_force_n:g} N, c {c.damping_n_s_per_m:g} N s/m")
                force = damping_force(ctx.grip_velocity, ctx.reference_velocity, c.damping_n_s_per_m, c.max_force_n)
                err = np.asarray(ctx.grip_velocity, dtype=float) - np.asarray(ctx.reference_velocity, dtype=float)
                f = float(np.linalg.norm(force))
                self._damping_s += dt
                self._energy += -float(force @ err) * dt
                self._impulse += f * dt
                self._max_force = max(self._max_force, f)
            else:
                self._goto("ASTROBEE_ASSIST_GRASPED", t)
        elif self.phase == "ASTROBEE_ASSIST_RELEASE":
            target = self._side_point(ctx, c.standby_distance_m)
            pos, self._vel = fly_towards(pos, self._vel, target, ctx.grip_velocity, dt, c.retreat_speed_mps, c.max_accel_mps2)
            if float(np.linalg.norm(target - pos)) < 0.01:
                self._goto("ASTROBEE_ASSIST_STANDBY", t)
        elif self.phase == "ASTROBEE_ASSIST_STANDBY":
            target = self._side_point(ctx, c.standby_distance_m)
            pos, self._vel = fly_towards(pos, self._vel, target, ctx.grip_velocity, dt, c.retreat_speed_mps, c.max_accel_mps2)
        return AssistOutput(self.phase, pos, grip, force, grip)

    def summary(self) -> dict:
        return {
            "enabled": bool(self.cfg.enabled),
            "max_force_limit_n": self.cfg.max_force_n,
            "damping_n_s_per_m": self.cfg.damping_n_s_per_m,
            "final_phase": self.phase,
            "grasped": self._t_grasp is not None,
            "grasp_time_s": self._t_grasp,
            "grasp_relative_speed_mps": self._grasp_rel_speed,
            "damping_time_s": self._damping_s,
            "energy_removed_j": self._energy,
            "impulse_n_s": self._impulse,
            "max_force_n": self._max_force,
        }
