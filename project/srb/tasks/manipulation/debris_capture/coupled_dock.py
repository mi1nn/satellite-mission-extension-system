"""Predictive coupled position/attitude docking control (`docking_control.mode`).

Pure numpy (no Isaac Sim imports), so every rule here is unit-testable offline
(`project/tests/test_coupled_dock.py`). The state machine that uses it lives in
`vision_capture_demo.py`; `docking_control.mode: legacy` (the default) leaves the
verified sequential XY_ALIGN -> ORIENTATION_ALIGN -> ALIGNMENT_CHECK path untouched.

Two ideas are taken from the tumbling-target docking literature, in their simplest
verifiable form (no sliding mode, no MPC):

- Zhou, Liu, Cai, "Motion-planning and pose-tracking based rendezvous and docking with
  a tumbling target": the chaser tracks a *desired docking pose planned from the
  target's motion*, not the target's current pose.
- Ye, Lu, Mu, "Compound control for autonomous docking to a three-axis tumbling
  target": position and attitude are tracked *together*, relative to the target.

Frames and units: world frame, [m], [m/s], [rad], [rad/s] unless a name says `_deg`.
Rotation matrices have the frame axes as columns (`frames.Frame`); angular velocities
are world-frame vectors, so a constant rate integrates as R(t + T) = Exp([w] T) R(t)
(left multiplication; a body-frame rate would multiply on the right). Quaternions, where
logged, are (w, x, y, z) as everywhere in this package.

Controlled frame: the probe tip (PROBE_DOCK_POINT) against SAT_DOCK_POINT, both from
`docking.py`. Control law, per step (`CoupledPoseTracker.step`):

    D      = desired probe-tip pose at t + T  (predicted dock frame, offset `axial_cmd`
             along its +Z = docking axis)
    C      = commanded probe-tip pose (the IK reference) propagated to t + T
    e_p    = D.pos - C.pos                  e_R = Log(D.rot C.rot^T)
    v_rel  = v_tip - v_D                    w_rel = w_tip - w_client
    v_cmd  = v_D + clip(Kp_p e_p) - Kd_p v_rel
    w_cmd  = w_client + clip(Kp_R e_R) - Kd_R w_rel

The proportional term acts on the *commanded* tip pose and the damping on the
*measured* tip motion. Feeding the measured tip pose into the integrating reference
instead limit-cycles on the lightly damped arm + 3 t payload mode (period ~20 s):
checked on a 1-D model of that mode in `test_coupled_dock.py`. `tip_feedback_weight`
blends the measured pose into the proportional term for experiments (0 = off).

The gates (soft approach gate, hysteresis, emergency stop, stable-duration check) all
use the MEASURED probe tip against the CURRENT dock frame -- what physically docks.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .frames import Frame, rotation_angle
from .moving_dock import StableTimer, clip_norm

MODES = ("legacy", "coupled_predictive")

##############
### Config ###
##############


@dataclass
class DockingControlCfg:
    """`docking_control:` section -- controller selection and coupled-tracker gains."""

    # legacy: the verified sequential XY_ALIGN -> ORIENTATION_ALIGN -> ALIGNMENT_CHECK
    # (unchanged). coupled_predictive: POSITION_ATTITUDE_ALIGN -> ALIGNMENT_CHECK ->
    # Z_APPROACH with the predictive coupled tracker, soft gate and hysteresis below.
    mode: str = "legacy"
    # Horizon of the client-motion prediction [s] (constant twist about the client CoM)
    prediction_horizon_sec: float = 0.3
    # v_cmd = v_target + clip(Kp e_p) - Kd v_rel  (Kp [1/s], Kd [-])
    kp_position_hz: float = 0.15
    kd_position: float = 0.6
    # w_cmd = w_client + clip(Kp e_R) - Kd w_rel
    kp_attitude_hz: float = 0.15
    kd_attitude: float = 0.6
    # Correction limits (on top of the target's own motion)
    # far-field speed ~ max_correction_speed_mps / (1 + kd_position) (see vision_capture.yaml)
    max_correction_speed_mps: float = 0.03
    max_correction_rate_deg_s: float = 2.0
    # Change of the commanded tip velocity per second: stepping it kicks the 3 t payload
    max_accel_mps2: float = 0.01
    max_angular_accel_deg_s2: float = 1.0
    # 0: proportional term on the commanded tip pose only (stable on the payload mode);
    # (0, 1]: blend the measured tip pose in (experiments; needs Isaac validation)
    tip_feedback_weight: float = 0.0
    # Anti-windup: the commanded tip may lead the measured one by at most this much
    max_reference_lead_m: float = 0.10
    max_reference_lead_deg: float = 5.0
    # Z_APPROACH: the axial set point advances at the gated speed, never more than this
    # ahead of the measured axial position [m]
    max_axial_lead_m: float = 0.05
    # ... and inside the nozzle: a set point far ahead of a tip that touches the nozzle
    # pushes the client (measured: +0.2 mm/s on the 578 t client in 0.3 s of DOCK_READY,
    # which made the MRV station keeping accelerate). DOCK_READY freezes it at the tip.
    max_axial_lead_inside_m: float = 0.01
    # Soft approach gate on the measured tip: full approach speed below the `full_*`
    # errors, zero at / above the `zero_*` errors, smooth in between (product of both)
    soft_gate_full_position_m: float = 0.01
    soft_gate_zero_position_m: float = 0.08
    soft_gate_full_orientation_deg: float = 1.0
    soft_gate_zero_orientation_deg: float = 7.0
    # Emergency hard stop (the approach stops at once and falls back to aligning)
    hard_stop_position_m: float = 0.15
    hard_stop_orientation_deg: float = 12.0
    hard_stop_relative_speed_mps: float = 0.10


@dataclass
class DockingAlignmentCfg:
    """`docking_alignment:` section -- entry / rollback thresholds (coupled mode)."""

    # ALIGNMENT_CHECK -> Z_APPROACH once ALL of these hold for `stable_duration_sec`.
    # Position = lateral offset of the probe tip from the docking axis (dock frame X/Y),
    # orientation = full 3-axis angle between the probe and dock frames (roll included).
    position_threshold_m: float = 0.05
    orientation_threshold_deg: float = 4.0
    relative_velocity_threshold_mps: float = 0.02
    relative_angular_velocity_threshold_deg_s: float = 1.0
    stable_duration_sec: float = 0.5
    # Hysteresis: back to POSITION_ATTITUDE_ALIGN only above these (must be wider than
    # the entry thresholds), held for `rollback_hold_sec` (0: at once)
    rollback_position_m: float = 0.08
    rollback_orientation_deg: float = 7.0
    rollback_hold_sec: float = 0.2


def validate_docking_control_cfg(c: DockingControlCfg, a: DockingAlignmentCfg):
    """Raise on a configuration that cannot work (called from `load_vision_config`)."""
    if c.mode not in MODES:
        raise ValueError(f"docking_control.mode must be one of {MODES}, got '{c.mode}'")
    if c.prediction_horizon_sec < 0.0:
        raise ValueError("docking_control.prediction_horizon_sec must be >= 0")
    for name in ("kp_position_hz", "kd_position", "kp_attitude_hz", "kd_attitude"):
        if getattr(c, name) < 0.0:
            raise ValueError(f"docking_control.{name} must be >= 0")
    for name in ("max_correction_speed_mps", "max_correction_rate_deg_s", "max_accel_mps2", "max_angular_accel_deg_s2",
                 "max_reference_lead_m", "max_reference_lead_deg", "max_axial_lead_m", "max_axial_lead_inside_m"):
        if not getattr(c, name) > 0.0:
            raise ValueError(f"docking_control.{name} must be > 0")
    if not 0.0 <= c.tip_feedback_weight <= 1.0:
        raise ValueError("docking_control.tip_feedback_weight must be in [0, 1]")
    if not 0.0 <= c.soft_gate_full_position_m < c.soft_gate_zero_position_m:
        raise ValueError("docking_control requires 0 <= soft_gate_full_position_m < soft_gate_zero_position_m")
    if not 0.0 <= c.soft_gate_full_orientation_deg < c.soft_gate_zero_orientation_deg:
        raise ValueError("docking_control requires 0 <= soft_gate_full_orientation_deg < soft_gate_zero_orientation_deg")
    for name in ("position_threshold_m", "orientation_threshold_deg", "relative_velocity_threshold_mps",
                 "relative_angular_velocity_threshold_deg_s"):
        if not getattr(a, name) > 0.0:
            raise ValueError(f"docking_alignment.{name} must be > 0")
    if a.stable_duration_sec <= 0.0:
        raise ValueError("docking_alignment.stable_duration_sec must be > 0 (a single step is never enough)")
    if a.rollback_hold_sec < 0.0:
        raise ValueError("docking_alignment.rollback_hold_sec must be >= 0")
    # enter < rollback <= hard stop: otherwise the state flips on noise, or the hard stop
    # fires before the (gentler) rollback could
    if not a.position_threshold_m < a.rollback_position_m <= c.hard_stop_position_m:
        raise ValueError("requires docking_alignment.position_threshold_m < rollback_position_m <= docking_control.hard_stop_position_m")
    if not a.orientation_threshold_deg < a.rollback_orientation_deg <= c.hard_stop_orientation_deg:
        raise ValueError("requires docking_alignment.orientation_threshold_deg < rollback_orientation_deg "
                         "<= docking_control.hard_stop_orientation_deg")
    if not c.hard_stop_relative_speed_mps > 0.0:
        raise ValueError("docking_control.hard_stop_relative_speed_mps must be > 0")


##############
### SO(3) ###
##############


def _hat(w) -> np.ndarray:
    x, y, z = (float(v) for v in w)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(phi) -> np.ndarray:
    """Rotation matrix of the rotation vector `phi` [rad] (Rodrigues)."""
    phi = np.asarray(phi, dtype=float)
    th = float(np.linalg.norm(phi))
    k = _hat(phi)
    if th < 1e-9:
        return np.eye(3) + k
    return np.eye(3) + math.sin(th) / th * k + (1.0 - math.cos(th)) / (th * th) * (k @ k)


def so3_log(r: np.ndarray) -> np.ndarray:
    """Rotation vector [rad] of a rotation matrix (|result| <= pi)."""
    r = np.asarray(r, dtype=float)
    c = max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))
    th = math.acos(c)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if th < 1e-7:
        return 0.5 * v
    if math.pi - th < 1e-4:  # near pi: the axis from the symmetric part
        axis = np.sqrt(np.maximum(np.diag(r) - c, 0.0) / (1.0 - c))
        i = int(np.argmax(axis))
        for j in range(3):  # signs relative to the largest component
            if j != i:
                axis[j] = math.copysign(axis[j], r[i, j] + r[j, i])
        return th * axis / np.linalg.norm(axis)
    return th / (2.0 * math.sin(th)) * v


##################
### Prediction ###
##################


def point_velocity(p, com, v_com, w) -> np.ndarray:
    """Velocity of the body-fixed point `p` of a rigid body (v_com + w x (p - com))."""
    return np.asarray(v_com, dtype=float) + np.cross(np.asarray(w, dtype=float), np.asarray(p, dtype=float) - np.asarray(com, dtype=float))


def predict_body_frame(f: Frame, com, v_com, w, horizon: float) -> Frame:
    """Pose of a frame fixed on a rigid body after `horizon` seconds of constant twist.

    The body translates with its CoM (v_com) and turns about it (world-frame w):

        R(t+T) = Exp([w] T) R(t)
        p(t+T) = com + v_com T + Exp([w] T) (p(t) - com)

    For w = 0 this is p + v T; for a tumbling client the docking port swings around the
    CoM instead of moving on its current tangent (which `p + v_port T` would assume).
    """
    t = max(0.0, float(horizon))
    q = so3_exp(np.asarray(w, dtype=float) * t)
    com = np.asarray(com, dtype=float)
    pos = com + np.asarray(v_com, dtype=float) * t + q @ (f.pos - com)
    return Frame(pos, q @ f.rot)


def predict_frame(f: Frame, v, w, horizon: float) -> Frame:
    """`f` moved by its own twist (origin velocity v, world rate w) for `horizon` s."""
    t = max(0.0, float(horizon))
    return Frame(f.pos + np.asarray(v, dtype=float) * t, so3_exp(np.asarray(w, dtype=float) * t) @ f.rot)


def desired_tip(dock_pred: Frame, axial_cmd: float) -> Frame:
    """Desired probe-tip pose: on the predicted docking axis, `axial_cmd` from the dock
    point (< 0: outside), coaxial and roll-aligned (`probe_dock.tip_goal` on the
    predicted frame)."""
    return Frame(dock_pred.pos + float(axial_cmd) * dock_pred.rot[:, 2], dock_pred.rot.copy())


##############
### Errors ###
##############


def pose_error(desired: Frame, current: Frame) -> Tuple[np.ndarray, np.ndarray]:
    """(e_p [m], e_R [rad]) world-frame vectors taking `current` onto `desired`.

    e_R = Log(R_d R_c^T): the rotation vector (never an Euler difference)."""
    return desired.pos - current.pos, so3_log(desired.rot @ current.rot.T)


def dock_frame_errors(tip: Frame, dock: Frame) -> Dict[str, float]:
    """Gate quantities of the MEASURED tip against the CURRENT dock frame:
    lateral [m] (dock X/Y), axial [m] (dock Z, < 0 outside) and the full 3-axis
    orientation error [deg] (rotation angle, roll included)."""
    rel = dock.inv() @ tip
    x, y, z = (float(v) for v in rel.pos)
    return {"lateral": math.hypot(x, y), "axial": z,
            "orientation_deg": math.degrees(rotation_angle(rel.rot, np.eye(3)))}


#################
### Soft gate ###
#################


def ramp_down(x: float, full: float, zero: float) -> float:
    """1 at x <= full, 0 at x >= zero, smoothstep in between (C1, no speed step)."""
    if not math.isfinite(x) or x >= zero:
        return 0.0
    if x <= full:
        return 1.0
    s = (zero - x) / (zero - full)
    return s * s * (3.0 - 2.0 * s)


def alignment_scale(c: DockingControlCfg, position_m: float, orientation_deg: float) -> float:
    """Approach-speed factor in [0, 1] from the measured lateral and orientation errors."""
    return (ramp_down(position_m, c.soft_gate_full_position_m, c.soft_gate_zero_position_m)
            * ramp_down(orientation_deg, c.soft_gate_full_orientation_deg, c.soft_gate_zero_orientation_deg))


def emergency_stop(c: DockingControlCfg, position_m: float, orientation_deg: float, rel_speed_mps: float,
                   clearance_m: float, min_clearance_m: float) -> Tuple[bool, List[str]]:
    """Unsafe state -> hard stop. The soft gate handles everything below these."""
    bad = []
    if not position_m <= c.hard_stop_position_m:
        bad.append(f"lateral {position_m*1000:.1f} mm > {c.hard_stop_position_m*1000:.0f} mm (hard stop)")
    if not orientation_deg <= c.hard_stop_orientation_deg:
        bad.append(f"orientation {orientation_deg:.2f} deg > {c.hard_stop_orientation_deg:g} (hard stop)")
    if not rel_speed_mps <= c.hard_stop_relative_speed_mps:
        bad.append(f"relative speed {rel_speed_mps*1000:.0f} mm/s > {c.hard_stop_relative_speed_mps*1000:.0f} (hard stop)")
    if not clearance_m >= min_clearance_m:  # inf outside the nozzle
        bad.append(f"wall clearance {clearance_m*1000:.0f} mm < {min_clearance_m*1000:.0f} mm (hard stop)")
    return bool(bad), bad


##########################
### Alignment gate FSM ###
##########################


class AlignmentGate:
    """Entry condition held for `stable_duration_sec`, rollback on wider thresholds.

    `entry()` is used in ALIGNMENT_CHECK: every condition must hold continuously (sim
    time, not step count). `rollback()` is used while approaching: it only fires above
    the wider `rollback_*` thresholds, held for `rollback_hold_sec`, so noise around the
    entry threshold cannot bounce ALIGN <-> APPROACH.
    """

    def __init__(self, cfg: DockingAlignmentCfg):
        self.cfg = cfg
        self._entry = StableTimer()
        self._rollback = StableTimer()

    def reset(self):
        self._entry.reset()
        self._rollback.reset()

    def entry_conditions(self, position_m: float, orientation_deg: float, rel_speed_mps: float,
                         rel_rate_deg_s: float) -> Tuple[bool, List[str]]:
        a = self.cfg
        bad = []
        if not position_m < a.position_threshold_m:
            bad.append(f"lateral {position_m*1000:.1f} mm >= {a.position_threshold_m*1000:.0f} mm")
        if not orientation_deg < a.orientation_threshold_deg:
            bad.append(f"orientation {orientation_deg:.2f} deg >= {a.orientation_threshold_deg:g}")
        if not rel_speed_mps < a.relative_velocity_threshold_mps:
            bad.append(f"relative speed {rel_speed_mps*1000:.1f} mm/s >= {a.relative_velocity_threshold_mps*1000:.0f}")
        if not rel_rate_deg_s < a.relative_angular_velocity_threshold_deg_s:
            bad.append(f"relative rate {rel_rate_deg_s:.2f} deg/s >= {a.relative_angular_velocity_threshold_deg_s:g}")
        return not bad, bad

    def entry(self, t: float, position_m: float, orientation_deg: float, rel_speed_mps: float,
              rel_rate_deg_s: float, blocked: Sequence[str] = ()) -> Tuple[bool, float, List[str]]:
        """(passed, held [s], reasons): passed once all conditions held long enough.
        `blocked`: reasons from outside (MRV station, depth calibration) that also break
        the stable interval."""
        ok, bad = self.entry_conditions(position_m, orientation_deg, rel_speed_mps, rel_rate_deg_s)
        bad = bad + list(blocked)
        ok = ok and not blocked
        held = self._entry.update(ok, t)
        return ok and held >= self.cfg.stable_duration_sec, held, bad

    def rollback(self, t: float, position_m: float, orientation_deg: float) -> Tuple[bool, List[str]]:
        a = self.cfg
        bad = []
        if not position_m <= a.rollback_position_m:
            bad.append(f"lateral {position_m*1000:.1f} mm > {a.rollback_position_m*1000:.0f} mm")
        if not orientation_deg <= a.rollback_orientation_deg:
            bad.append(f"orientation {orientation_deg:.2f} deg > {a.rollback_orientation_deg:g}")
        held = self._rollback.update(bool(bad), t)
        return bool(bad) and held >= a.rollback_hold_sec, bad


class TwistEstimator:
    """Probe-tip twist from consecutive poses, low-passed with time constant `tau_s`.

    The MEP is dragged through the capture joint by an arm whose base may be moved
    kinematically (moving client); its PhysX velocity then misses the base motion while
    its pose is right (`moving_dock.VelocityEstimator`), so the damping term and the
    relative-velocity gates use pose differences instead."""

    def __init__(self, tau_s: float = 0.1, max_gap_s: float = 0.5):
        self.tau = float(tau_s)
        self.max_gap = float(max_gap_s)
        self.reset()

    def reset(self):
        self._t: Optional[float] = None
        self._f: Optional[Frame] = None
        self.v: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None

    def update(self, t: float, f: Frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._t is not None and (t - self._t > self.max_gap or t <= self._t):
            if t <= self._t:  # same instant twice: keep the estimate
                return self.v, self.w
            self.reset()
        if self._t is not None:
            dt = t - self._t
            v = (f.pos - self._f.pos) / dt
            w = so3_log(f.rot @ self._f.rot.T) / dt
            a = min(1.0, dt / max(self.tau, 1e-9))
            self.v = v if self.v is None else self.v + a * (v - self.v)
            self.w = w if self.w is None else self.w + a * (w - self.w)
        self._t, self._f = float(t), Frame(f.pos.copy(), f.rot.copy())
        return self.v, self.w


###############
### Tracker ###
###############


@dataclass
class TrackerOutput:
    """One step of a docking tracker. `ref` is the new commanded probe-tip pose."""

    ref: Frame
    v_cmd: np.ndarray
    w_cmd: np.ndarray
    e_p: np.ndarray
    e_r: np.ndarray


class CoupledPoseTracker:
    """Coupled position + attitude PD tracker of a predicted desired tip pose.

    Kept behind this small interface (`reset`, `step`) so a later phase can swap in a
    sliding-mode or MPC tracker without touching the state machine.
    """

    def __init__(self, cfg: DockingControlCfg):
        self.cfg = cfg
        self.v = np.zeros(3)
        self.w = np.zeros(3)

    def reset(self, v0=None, w0=None):
        self.v = np.zeros(3) if v0 is None else np.asarray(v0, dtype=float).copy()
        self.w = np.zeros(3) if w0 is None else np.asarray(w0, dtype=float).copy()

    def step(self, ref: Frame, tip: Frame, v_tip, w_tip, desired: Frame, v_des, w_des, dt: float) -> TrackerOutput:
        """`ref`: commanded tip pose; `tip`, `v_tip`, `w_tip`: measured tip pose / twist;
        `desired`, `v_des`, `w_des`: desired tip pose at t + T and its twist."""
        c = self.cfg
        T = c.prediction_horizon_sec
        v_tip, w_tip = np.asarray(v_tip, dtype=float), np.asarray(w_tip, dtype=float)
        v_des, w_des = np.asarray(v_des, dtype=float), np.asarray(w_des, dtype=float)
        # Current pose seen by the proportional term (commanded, optionally blended with
        # the measured one), propagated to t + T with the commanded twist
        cur = ref
        if c.tip_feedback_weight > 0.0:
            wgt = c.tip_feedback_weight
            cur = Frame((1.0 - wgt) * ref.pos + wgt * tip.pos,
                        so3_exp(wgt * so3_log(tip.rot @ ref.rot.T)) @ ref.rot)
        e_p, e_r = pose_error(desired, predict_frame(cur, self.v, self.w, T))
        v_goal = v_des + clip_norm(c.kp_position_hz * e_p, c.max_correction_speed_mps) - c.kd_position * (v_tip - v_des)
        w_goal = (w_des + clip_norm(c.kp_attitude_hz * e_r, math.radians(c.max_correction_rate_deg_s))
                  - c.kd_attitude * (w_tip - w_des))
        self.v = self.v + clip_norm(v_goal - self.v, c.max_accel_mps2 * dt)
        self.w = self.w + clip_norm(w_goal - self.w, math.radians(c.max_angular_accel_deg_s2) * dt)
        pos = ref.pos + self.v * dt
        rot = so3_exp(self.w * dt) @ ref.rot
        # Anti-windup: the command never runs away from the measured tip
        pos = tip.pos + clip_norm(pos - tip.pos, c.max_reference_lead_m)
        lead = so3_log(rot @ tip.rot.T)
        lead_max = math.radians(c.max_reference_lead_deg)
        if float(np.linalg.norm(lead)) > lead_max:
            rot = so3_exp(clip_norm(lead, lead_max)) @ tip.rot
        return TrackerOutput(Frame(pos, rot), self.v.copy(), self.w.copy(), e_p, e_r)


TRACKERS = {"coupled_predictive": CoupledPoseTracker}


###############
### Metrics ###
###############

# States the alignment/approach switching count looks at (both modes)
ALIGN_STATES = ("XY_ALIGN", "ORIENTATION_ALIGN", "POSITION_ATTITUDE_ALIGN", "ALIGNMENT_CHECK")
APPROACH_STATES = ("Z_APPROACH", "FINAL_INSERTION")


class DockingControlMetrics:
    """Comparison metrics of one docking run (identical for both modes)."""

    def __init__(self):
        self.start_t: Optional[float] = None
        self.dock_t: Optional[float] = None
        self.max_position_error_m = 0.0
        self.max_orientation_error_deg = 0.0
        self.max_contact_speed_mps = 0.0
        self.rollbacks = 0
        self.emergency_stops = 0
        self.switches = 0
        self.at_dock: Optional[Dict[str, float]] = None

    def update(self, t: float, state: str, lateral_m: float, orientation_deg: float, rel_speed_mps: float,
               insertion_depth_m: float):
        if self.start_t is None:
            self.start_t = t
        if state in ALIGN_STATES + APPROACH_STATES + ("DOCK_READY",):
            if math.isfinite(lateral_m):
                self.max_position_error_m = max(self.max_position_error_m, lateral_m)
            if math.isfinite(orientation_deg):
                self.max_orientation_error_deg = max(self.max_orientation_error_deg, orientation_deg)
        if insertion_depth_m > 0.0 and state in APPROACH_STATES + ("DOCK_READY",) and math.isfinite(rel_speed_mps):
            self.max_contact_speed_mps = max(self.max_contact_speed_mps, rel_speed_mps)

    def transition(self, old: str, new: str):
        """Count ALIGN <-> APPROACH switches and approach -> align rollbacks."""
        if (old in ALIGN_STATES and new in APPROACH_STATES) or (old in APPROACH_STATES and new in ALIGN_STATES):
            self.switches += 1
        if old in APPROACH_STATES and new in ALIGN_STATES:
            self.rollbacks += 1

    def docked(self, t: float, values: Dict[str, float]):
        self.dock_t = t
        self.at_dock = dict(values)

    def summary(self, mode: str, success: bool) -> dict:
        return {
            "mode": mode,
            "docking_success": bool(success),
            "docking_time_s": (None if self.dock_t is None or self.start_t is None else self.dock_t - self.start_t),
            "probe_tip_position_error_at_dock_m": None if self.at_dock is None else self.at_dock.get("position_error_m"),
            "probe_tip_orientation_error_at_dock_deg": None if self.at_dock is None else self.at_dock.get("orientation_error_deg"),
            "relative_velocity_at_dock_mps": None if self.at_dock is None else self.at_dock.get("relative_speed_mps"),
            "max_position_error_m": self.max_position_error_m,
            "max_orientation_error_deg": self.max_orientation_error_deg,
            "rollback_count": self.rollbacks,
            "emergency_stop_count": self.emergency_stops,
            "alignment_state_switching_count": self.switches,
            "max_contact_velocity_mps": self.max_contact_speed_mps,
        }


###############
### Logging ###
###############


def _xyz(prefix: str) -> Tuple[str, ...]:
    return tuple(f"{prefix}_{a}" for a in "xyz")


def _wxyz(prefix: str) -> Tuple[str, ...]:
    return tuple(f"{prefix}_q{a}" for a in "wxyz")


# `<run>_docking_control.csv`, one row per `logging.rate_hz` tick of the docking phase,
# written in BOTH modes (the legacy run logs the same quantities for comparison; its
# desired pose is then the prediction it did not use). World frame, SI units, degrees
# where named. `position_error_*` / `orientation_error_*` are measured tip vs desired
# (predicted) tip; `lateral_error_m` / `orientation_error_deg` are measured tip vs the
# CURRENT dock frame (what the gates use).
CSV_COLUMNS: Tuple[str, ...] = (
    "run_id", "timestamp", "mode", "fsm_state",
    *_xyz("client_dock_position"), *_wxyz("client_dock_orientation"),
    *_xyz("client_linear_velocity"), *_xyz("client_angular_velocity"),
    *_xyz("desired_dock_position"), *_wxyz("desired_dock_orientation"),
    *_xyz("probe_tip_position"), *_wxyz("probe_tip_orientation"),
    *_xyz("position_error"), "position_error_m", *_xyz("orientation_error_rotvec"),
    "lateral_error_m", "axial_error_m", "orientation_error_deg",
    *_xyz("relative_linear_velocity"), "relative_speed_mps",
    *_xyz("relative_angular_velocity"), "relative_angular_rate_deg_s",
    "prediction_horizon", "axial_command_m", "approach_speed", "alignment_scale",
    *_xyz("command_linear_velocity"), *_xyz("command_angular_velocity"),
    "rollback_count", "emergency_stop_count", "docking_success",
)
