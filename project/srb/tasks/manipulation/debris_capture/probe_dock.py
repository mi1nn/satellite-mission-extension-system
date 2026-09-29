"""Ares1 probe -> satellite thruster docking: relative pose, depth check, conditions.

Pure numpy (no Isaac Sim imports), so every rule here is unit-testable offline
(`project/tests/test_probe_dock.py`). The state machine that uses it lives in
`vision_capture_demo.py`; the measured frames come from `docking.py`
(`DockingGeometry.probe_dock`, `DockingGeometry.sat_dock`).

Frames (`frames.Frame`, rotation matrices with the axes as columns):

    T_W_P  probe dock frame  : origin = probe tip, +Z = insertion direction (out of the tip)
    T_W_D  satellite dock    : origin = SAT_DOCK_POINT (`dock_depth` inside the nozzle
                               exit), +Z = into the nozzle, +X = satellite body +X

Everything is expressed in the **satellite dock frame D**, never in world XYZ: the
error components are then the docking-axis distance (Z) and the two lateral offsets
(X, Y), whatever the satellite's world orientation is.

    e = T_W_D^-1 @ T_W_P          (probe tip in the dock frame)
    e.pos = [e_x, e_y, e_z]       e_z < 0: the tip is still outside (before the dock point)
    remaining insertion = -e_z

Sign convention: `axial` is `e_z`, so it grows from negative towards 0 as the probe
goes in, and `depth` (from the nozzle exit) grows from negative to positive.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .frames import Frame, axis_angle, rotation_angle

##############
### Config ###
##############


@dataclass
class DockingVisionCfg:
    """`docking:` section of `config/vision_capture.yaml` (see the YAML for units)."""

    # The capture demo only runs the docking phase when this is on (`--dock`)
    enabled: bool = True
    # Skip the capture phase: attach the MEP at its nominal grasp pose and start docking
    # (iteration speed; the capture itself is covered by the other scenarios)
    skip_capture: bool = False
    # Holding time before the docking phase starts [s] (0: straight on)
    settle_time_s: float = 1.0

    ## Satellite free drift (this phase: translation only, no rotation)
    satellite_velocity_mps: float = 0.0
    satellite_drift_direction: List[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])

    ## Key poses
    # Probe tip standoff in front of the nozzle *exit* at the pre-dock pose [m]
    pre_dock_distance_m: float = 1.0

    ## Alignment gates (checked in the dock frame, before any insertion)
    align_lateral_m: float = 0.01
    align_axis_deg: float = 0.5
    align_roll_deg: float = 1.0
    # The alignment has to hold for this long before the Z approach starts [s]
    align_hold_s: float = 0.5
    # The tip must also be at rest: max probe-tip displacement over `settle_window_s`
    # The window has to span at least half of the arm + payload period (~20 s): a short
    # window is satisfied at every turning point of an oscillation that is still running.
    settle_window_s: float = 12.0
    settle_window_m: float = 0.008

    ## Transport to the pre-dock pose (free space, no insertion yet)
    transport_speed_mps: float = 0.11

    ## Docking-axis approach
    # Above `slow_zone_m` remaining insertion distance, move at `approach_speed_mps`
    approach_speed_mps: float = 0.11
    slow_zone_m: float = 0.30
    near_speed_mps: float = 0.03
    # Final insertion (inside the nozzle, below `insertion_zone_m` remaining)
    insertion_zone_m: float = 0.10
    insertion_speed_mps: float = 0.01
    # Lateral / angular correction rate while approaching
    # Slow on purpose: a correction finished in much less than the arm + 3 t payload
    # period (~20 s) re-excites that mode, and the residual swing then sits right on the
    # alignment gate (measured: a 0.02 m/s correction left a +-12 mm swing vs a 10 mm gate)
    align_speed_mps: float = 0.008
    align_speed_deg_s: float = 1.0
    # Continuous deceleration instead of an on/off target: v <= decel_gain * remaining,
    # never below `creep`. Stop-and-go of the reference excites the lightly damped
    # arm + 3 t payload mode (period ~20 s), which a straight rate limit cannot settle.
    decel_gain_hz: float = 0.05
    # Docking-axis (z) approach only: faster deceleration profile than `decel_gain_hz`
    z_decel_gain_hz: float = 0.3
    creep_speed_mps: float = 0.002
    # Ramp-up limit [m/s^2]. Stepping the commanded speed from 0 to the approach speed
    # kicks the 3 t payload sideways (measured: +-20 mm within 1 s of starting the Z
    # approach, which tripped the corridor check and bounced back to aligning).
    accel_mps2: float = 0.005
    # Active damping of the arm + payload swing: the joint velocity target is biased
    # against the measured probe-tip velocity error, so the drives' own velocity gain
    # brakes the mode instead of only holding a position target. 0 disables it.
    # (An outer damping loop on the joint *velocity* target was tried at gain 1.0 and
    # destabilised the arm within 0.1 s -- the drives already have a large velocity gain.
    # Left configurable, off by default.)
    swing_damping: float = 0.0
    # Anti-windup for the held joint target, in multiples of the per-step joint limit
    joint_target_windup: float = 5.0
    # The approach stops (and the state machine falls back to aligning) when the
    # alignment leaves these, i.e. the closed-loop check during the Z approach
    approach_abort_lateral_m: float = 0.02
    approach_abort_axis_deg: float = 1.5

    ## Docking conditions (all must hold)
    dock_axial_m: float = 0.01
    dock_radial_m: float = 0.01
    dock_axis_deg: float = 0.5
    dock_roll_deg: float = 1.0
    max_relative_velocity_mps: float = 0.02
    # Minimum clearance between the probe surface and the nozzle inner wall [m]
    min_wall_clearance_m: float = 0.02

    ## Depth camera
    depth_enabled: bool = True
    # Median of a square patch of this many pixels at the principal point
    depth_patch_px: int = 9
    # A depth reading is only used within this range [m]
    depth_min_m: float = 0.05
    depth_max_m: float = 12.0
    # The docking-axis distance from the depth camera and from the frames must agree
    # within this; otherwise the depth is not trusted and the probe does not advance
    depth_agreement_m: float = 0.15
    # Distance from SAT_DOCK_POINT to the surface the principal ray hits, along the
    # docking axis [m]. The nozzle back plate built by `build_thruster_collider` is a
    # collision-only prim (not rendered), so the ray actually lands on the thruster mesh
    # behind it; `auto_calibrate` measures this offset once at the pre-dock pose (where
    # the geometric distance is known) and logs it next to `backstop_gap` for comparison.
    depth_surface_offset_m: float = 0.1
    auto_calibrate: bool = True
    # The calibration is only taken ON the docking axis. The ray lands on the inside of
    # the bell nozzle, so its depth depends on the lateral offset: calibrated 41 mm off
    # the axis, every later on-axis reading disagreed by a constant 320 mm and the
    # approach never advanced (measured, static and moving client alike). The offset is
    # the median of `depth_calibration_samples` readings within these limits, and the
    # docking-axis approach does not start before it exists.
    depth_calibration_max_lateral_m: float = 0.015
    depth_calibration_max_axis_deg: float = 1.0
    depth_calibration_samples: int = 10
    # The Z approach requires a valid depth reading (fail-safe, never a blind advance)
    require_depth: bool = True

    ## Timeouts [s]
    stage_timeout_s: float = 240.0
    # Whole docking phase (a state that keeps bouncing resets `stage_timeout_s`)
    phase_timeout_s: float = 600.0
    hold_duration_s: float = 3.0
    # DOCK_READY gives up (-> DOCK_FAILED) if every condition has not been true at once
    # for this long. Must span at least half the arm + payload swing period (~20 s, see
    # `settle_window_s`): a shorter window can land on the outward half of that swing and
    # declare failure while the tip is still oscillating through the gate, not diverging
    # (measured: radial recovered to <25 mm mid-window, then swung back out to >65 mm by
    # a 5 s cutoff, while the MEP-client relative velocity gate cleared only after that).
    dock_ready_timeout_s: float = 20.0


@dataclass
class ProbeCameraCfg:
    """`probe_camera:` section: the RGB-D camera on the Ares1 probe.

    RGB is for the operator (the approach seen from the probe); the depth channel
    (`distance_to_image_plane`) measures the docking-axis distance. The camera sits on
    the probe axis `offset_from_tip_m` *in front of* the tip and looks along the
    insertion direction, so the rod never occludes the principal ray.
    """

    enabled: bool = True
    name: str = "cam_probe"
    width: int = 640
    height: int = 480
    horizontal_fov_deg: float = 70.0
    horizontal_aperture_mm: float = 20.955
    # Near clip must stay below the final probe-to-back-plate gap (`backstop_gap`)
    clipping_range_m: List[float] = field(default_factory=lambda: [0.01, 30.0])
    offset_from_tip_m: float = 0.02
    # Roll about the viewing direction [deg], clockwise as seen from behind the camera.
    # 90: image upright (was 0 = rolled 90 deg CCW)
    image_roll_deg: float = 90.0
    # Save an RGB frame every this many seconds (0: off); written next to the metrics CSV
    rgb_every_sec: float = 1.0

    @property
    def focal_length_mm(self) -> float:
        return self.horizontal_aperture_mm / (2.0 * math.tan(math.radians(self.horizontal_fov_deg) / 2.0))


def validate_probe_camera_cfg(cfg: ProbeCameraCfg):
    if cfg.width < 16 or cfg.height < 16:
        raise ValueError("probe_camera.width / .height must be >= 16 px")
    if not 1.0 < cfg.horizontal_fov_deg < 179.0:
        raise ValueError("probe_camera.horizontal_fov_deg must be in (1, 179)")
    lo, hi = cfg.clipping_range_m
    if not 0.0 < lo < hi:
        raise ValueError("probe_camera.clipping_range_m must be [near, far] with 0 < near < far")
    if cfg.offset_from_tip_m < 0.0:
        raise ValueError("probe_camera.offset_from_tip_m must be >= 0 (in front of the tip)")


PHASES = ("MEP_CAPTURED", "DOCK_TARGET_ACQUIRE", "PRE_DOCK_APPROACH", "XY_ALIGN", "POSITION_ATTITUDE_ALIGN",
          "ORIENTATION_ALIGN", "ALIGNMENT_CHECK", "Z_APPROACH", "FINAL_INSERTION",
          "DOCK_READY", "DOCKED", "DOCK_HOLDING")


def validate_docking_cfg(cfg: DockingVisionCfg):
    """Raise on a configuration that cannot work (called from `load_vision_config`)."""
    d = np.asarray(cfg.satellite_drift_direction, dtype=float)
    if d.shape != (3,) or not np.isfinite(d).all():
        raise ValueError(f"docking.satellite_drift_direction must be 3 finite values, got {cfg.satellite_drift_direction}")
    if cfg.satellite_velocity_mps < 0.0:
        raise ValueError("docking.satellite_velocity_mps must be >= 0")
    n = float(np.linalg.norm(d))
    if cfg.satellite_velocity_mps > 0.0 and n < 1e-9:
        raise ValueError("docking.satellite_drift_direction must be non-zero when satellite_velocity_mps > 0")
    cfg.satellite_drift_direction = (d / n).tolist() if n > 1e-9 else [0.0, 0.0, 0.0]
    if cfg.pre_dock_distance_m <= 0.0:
        raise ValueError("docking.pre_dock_distance_m must be > 0")
    if not 0.0 < cfg.insertion_zone_m <= cfg.slow_zone_m:
        raise ValueError("docking requires 0 < insertion_zone_m <= slow_zone_m")
    for name in ("approach_speed_mps", "near_speed_mps", "insertion_speed_mps"):
        if getattr(cfg, name) <= 0.0:
            raise ValueError(f"docking.{name} must be > 0")
    if not cfg.insertion_speed_mps <= cfg.near_speed_mps <= cfg.approach_speed_mps:
        raise ValueError("docking speeds must satisfy insertion <= near <= approach")
    if cfg.transport_speed_mps < cfg.approach_speed_mps:
        raise ValueError("docking.transport_speed_mps must be >= approach_speed_mps")
    if cfg.align_lateral_m <= 0.0 or cfg.approach_abort_lateral_m < cfg.align_lateral_m:
        raise ValueError("docking requires 0 < align_lateral_m <= approach_abort_lateral_m")
    if cfg.approach_abort_axis_deg < cfg.align_axis_deg:
        raise ValueError("docking requires align_axis_deg <= approach_abort_axis_deg")
    if cfg.depth_patch_px < 1 or cfg.depth_patch_px % 2 == 0:
        raise ValueError("docking.depth_patch_px must be a positive odd number of pixels")
    if not 0.0 < cfg.depth_min_m < cfg.depth_max_m:
        raise ValueError("docking requires 0 < depth_min_m < depth_max_m")
    if cfg.depth_calibration_max_lateral_m <= 0.0 or cfg.depth_calibration_max_axis_deg <= 0.0:
        raise ValueError("docking.depth_calibration_max_lateral_m / _max_axis_deg must be > 0")
    if int(cfg.depth_calibration_samples) < 1:
        raise ValueError("docking.depth_calibration_samples must be >= 1")


######################
### Relative pose ###
######################


def relative_pose(probe_w: Frame, dock_w: Frame) -> Frame:
    """Probe dock frame expressed in the satellite dock frame (`T_D_P`)."""
    return dock_w.inv() @ probe_w


def dock_errors(probe_w: Frame, dock_w: Frame) -> Dict[str, float]:
    """Position / orientation error of the probe tip w.r.t. SAT_DOCK_POINT.

    All lengths [m] and angles [deg]. `axial` (= e_z) is negative while the tip is
    still short of the dock point, so `-axial` is the remaining insertion distance.
    `roll` is about the docking axis (the rotationally free direction of a round
    probe; reported and checked, as in the existing demo).
    """
    rel = relative_pose(probe_w, dock_w)
    x, y, z = (float(v) for v in rel.pos)
    r = rel.rot
    # Tilt of the probe axis w.r.t. the docking axis: both +Z point into the nozzle
    axis_deg = math.degrees(axis_angle(r[:, 2], np.array([0.0, 0.0, 1.0])))
    # Roll: probe +X against dock +X, measured in the plane normal to the docking axis
    xp = r[:, 0] - r[2, 0] * np.array([0.0, 0.0, 1.0])
    roll_deg = math.degrees(axis_angle(xp, np.array([1.0, 0.0, 0.0]))) if np.linalg.norm(xp) > 1e-9 else math.nan
    return {
        "e_x": x, "e_y": y, "e_z": z,
        "axial": z,
        "lateral": float(math.hypot(x, y)),
        "axis_deg": axis_deg,
        "roll_deg": roll_deg,
        # Full 3-axis orientation error (roll included), for the log
        "orientation_deg": math.degrees(rotation_angle(rel.rot, np.eye(3))),
    }


def rpy_errors_deg(probe_w: Frame, dock_w: Frame) -> Tuple[float, float, float]:
    """(roll, pitch, yaw) of `T_D_P` [deg] -- a human-readable view of the same error.

    Not used for control: the gates use `dock_errors` (axis / roll / lateral), which
    are computed from the rotation matrix without any Euler accumulation.
    """
    r = relative_pose(probe_w, dock_w).rot
    pitch = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    if abs(math.cos(pitch)) < 1e-9:  # gimbal lock: put everything in roll
        return math.degrees(math.atan2(-r[1, 2], r[1, 1])), math.degrees(pitch), 0.0
    roll = math.atan2(r[2, 1], r[2, 2])
    yaw = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def insertion_depth(probe_w: Frame, exit_w: Frame) -> float:
    """Probe tip depth inside the nozzle [m] (negative while still outside)."""
    return float((probe_w.pos - exit_w.pos) @ exit_w.rot[:, 2])


def tip_goal(dock_w: Frame, axial: float) -> Frame:
    """Probe-tip goal pose on the docking axis, `axial` from the dock point (< 0: outside).

    The orientation is the dock frame itself, i.e. the probe is asked to be coaxial and
    roll-aligned; the position is on the axis, so lateral error is commanded to zero.
    """
    return Frame(dock_w.pos + float(axial) * dock_w.rot[:, 2], dock_w.rot.copy())


def ramped(cfg: DockingVisionCfg, current: float, target: float, dt: float) -> float:
    """`current` speed moved towards `target` under the acceleration limit [m/s].

    Slowing down is immediate (a fail-safe must be able to stop at once); speeding up is
    limited to `accel_mps2`.
    """
    if target <= current:
        return float(target)
    return float(min(target, current + cfg.accel_mps2 * dt))


def decelerated(cfg: DockingVisionCfg, speed: float, distance: float, gain_hz: float = None) -> float:
    """`speed`, capped by a constant-deceleration profile towards `distance` [m].

    Keeps the commanded velocity continuous down to zero distance, so the arm is not
    stepped at the end of a motion (which rings the lightly damped arm+payload mode).
    """
    return float(min(speed, max(cfg.creep_speed_mps, (cfg.decel_gain_hz if gain_hz is None else gain_hz) * max(0.0, distance))))


def settled(history: Sequence[Tuple[float, np.ndarray]], now: float, window_s: float, window_m: float) -> bool:
    """Has the probe tip stayed within `window_m` over the last `window_s` seconds?"""
    recent = [p for t, p in history if t >= now - window_s]
    if len(recent) < 2 or history[0][0] > now - window_s:
        return False
    p = np.asarray(recent)
    return float(np.linalg.norm(p.max(axis=0) - p.min(axis=0))) <= window_m


def approach_speed(cfg: DockingVisionCfg, remaining: float) -> float:
    """Commanded docking-axis speed [m/s] for `remaining` insertion distance [m].

    Three zones (the "fast far / controlled near / slow insertion" requirement):
    beyond `slow_zone_m` the full approach speed, inside `insertion_zone_m` the
    insertion speed, and a linear ramp in between so the payload is never stepped.
    """
    r = max(0.0, float(remaining))
    if r >= cfg.slow_zone_m:
        return cfg.approach_speed_mps
    if r <= cfg.insertion_zone_m:
        return cfg.insertion_speed_mps
    span = cfg.slow_zone_m - cfg.insertion_zone_m
    f = (r - cfg.insertion_zone_m) / span
    return cfg.insertion_speed_mps + (cfg.near_speed_mps - cfg.insertion_speed_mps) * f


####################
### Depth camera ###
####################


def sample_depth(depth: np.ndarray, k: np.ndarray, patch_px: int, min_m: float, max_m: float) -> Tuple[float, int]:
    """Median depth [m] of a `patch_px` square at the principal point of `k`.

    `depth` is a `distance_to_image_plane` image (metres along the optical axis), which
    is exactly the docking-axis distance when the camera looks along the probe axis.
    Returns (nan, 0) when nothing in the patch is a finite reading inside
    [`min_m`, `max_m`] -- Isaac Sim writes `inf` (and RTX can write 0 or NaN) where no
    surface is hit, and a blind advance on such a reading is never allowed.
    """
    if depth is None or depth.ndim != 2 or patch_px < 1:
        return math.nan, 0
    h, w = depth.shape
    cx, cy = int(round(float(k[0, 2]))), int(round(float(k[1, 2])))
    r = patch_px // 2
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x0 >= x1 or y0 >= y1:
        return math.nan, 0
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=float).reshape(-1)
    ok = np.isfinite(patch) & (patch >= min_m) & (patch <= max_m)
    if not ok.any():
        return math.nan, 0
    return float(np.median(patch[ok])), int(ok.sum())


def depth_to_dock_distance(depth_m: float, cam_to_tip_m: float, backstop_gap_m: float) -> float:
    """Remaining insertion distance [m] implied by a depth reading on the back plate.

    The camera sits `cam_to_tip_m` behind the probe tip on the docking axis and looks
    along it, so the ray through the principal point hits the nozzle back plate, which
    is `backstop_gap_m` behind SAT_DOCK_POINT:

        depth = cam_to_tip + (tip -> dock point) + backstop_gap
        => tip -> dock point = depth - cam_to_tip - backstop_gap

    The result is comparable with `-dock_errors(...)["axial"]` computed from the frames.
    """
    if not math.isfinite(depth_m):
        return math.nan
    return float(depth_m) - float(cam_to_tip_m) - float(backstop_gap_m)


def depth_valid(cfg: DockingVisionCfg, depth_distance_m: float, geometry_distance_m: float) -> Tuple[bool, str]:
    """Is the depth reading usable for the approach? (fail-safe, never assume good)."""
    if not cfg.depth_enabled:
        return False, "depth disabled"
    if not math.isfinite(depth_distance_m):
        return False, "no finite depth reading at the principal point"
    d = abs(depth_distance_m - geometry_distance_m)
    if d > cfg.depth_agreement_m:
        return False, f"depth {depth_distance_m:.3f} m disagrees with geometry {geometry_distance_m:.3f} m by {d*1000:.0f} mm"
    return True, "ok"


##################
### Conditions ###
##################


def alignment_ok(cfg: DockingVisionCfg, m: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Lateral + orientation gate (`dock_errors` output). Checked before and during
    the docking-axis approach, with the wider `approach_abort_*` used by the caller."""
    bad = []
    if not m["lateral"] <= cfg.align_lateral_m:
        bad.append(f"lateral {m['lateral']*1000:.2f} mm > {cfg.align_lateral_m*1000:.0f} mm")
    if not m["axis_deg"] <= cfg.align_axis_deg:
        bad.append(f"axis {m['axis_deg']:.3f} deg > {cfg.align_axis_deg}")
    if not (math.isfinite(m["roll_deg"]) and m["roll_deg"] <= cfg.align_roll_deg):
        bad.append(f"roll {m['roll_deg']:.3f} deg > {cfg.align_roll_deg}")
    return not bad, bad


def depth_calibration_sample_ok(cfg: DockingVisionCfg, m: Dict[str, float]) -> Tuple[bool, str]:
    """May this instant's depth reading be used for the surface-offset calibration?

    Only on the docking axis (`depth_calibration_max_lateral_m`, `_max_axis_deg`) with a
    valid reading; the tighter-than-alignment limits keep the ray on the same surface
    it sees during the on-axis approach."""
    if not (math.isfinite(m["depth_raw"]) and m["depth_pixels"] >= 4):
        return False, "no valid depth reading"
    if not m["lateral"] <= cfg.depth_calibration_max_lateral_m:
        return False, f"lateral {m['lateral']*1000:.1f} mm > {cfg.depth_calibration_max_lateral_m*1000:.0f} mm"
    if not m["axis_deg"] <= cfg.depth_calibration_max_axis_deg:
        return False, f"axis {m['axis_deg']:.2f} deg > {cfg.depth_calibration_max_axis_deg}"
    return True, "ok"


def approach_still_aligned(cfg: DockingVisionCfg, m: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Closed-loop check while advancing along the docking axis (wider than the gate:
    the approach is only aborted when the alignment really leaves the corridor)."""
    bad = []
    if not m["lateral"] <= cfg.approach_abort_lateral_m:
        bad.append(f"lateral {m['lateral']*1000:.2f} mm > {cfg.approach_abort_lateral_m*1000:.0f} mm")
    if not m["axis_deg"] <= cfg.approach_abort_axis_deg:
        bad.append(f"axis {m['axis_deg']:.3f} deg > {cfg.approach_abort_axis_deg}")
    return not bad, bad


def wall_clearance(inner_radius_m: float, probe_tip_radius_m: float, lateral_m: float,
                   insertion_depth_m: float = 1.0) -> float:
    """Gap between the probe surface and the nozzle inner wall at the current depth [m].

    Only meaningful once the tip is inside the nozzle; outside it there is no wall to
    hit on the approach axis, so `inf` is returned (the lateral gate guards the rim).
    """
    if insertion_depth_m <= 0.0:
        return math.inf
    return float(inner_radius_m) - float(probe_tip_radius_m) - float(lateral_m)


def dock_ready(
    cfg: DockingVisionCfg,
    m: Dict[str, float],
    rel_speed_mps: float,
    depth_ok: bool,
    tracking_valid: bool,
    clearance_m: float,
) -> Tuple[bool, List[str]]:
    """Every docking condition (Section 10). A FixedJoint is only created when this
    returns True: proximity alone is never enough."""
    bad = []
    if not abs(m["axial"]) <= cfg.dock_axial_m:
        bad.append(f"axial {m['axial']*1000:.2f} mm > {cfg.dock_axial_m*1000:.0f} mm")
    if not m["lateral"] <= cfg.dock_radial_m:
        bad.append(f"radial {m['lateral']*1000:.2f} mm > {cfg.dock_radial_m*1000:.0f} mm")
    if not m["axis_deg"] <= cfg.dock_axis_deg:
        bad.append(f"axis {m['axis_deg']:.3f} deg > {cfg.dock_axis_deg}")
    if not (math.isfinite(m["roll_deg"]) and m["roll_deg"] <= cfg.dock_roll_deg):
        bad.append(f"roll {m['roll_deg']:.3f} deg > {cfg.dock_roll_deg}")
    if not (math.isfinite(rel_speed_mps) and rel_speed_mps <= cfg.max_relative_velocity_mps):
        bad.append(f"relative velocity {rel_speed_mps*1000:.1f} mm/s > {cfg.max_relative_velocity_mps*1000:.0f} mm/s")
    if not tracking_valid:
        bad.append("docking target / probe pose not valid")
    if cfg.require_depth and not depth_ok:
        bad.append("no valid depth reading")
    if not (math.isfinite(clearance_m) and clearance_m >= cfg.min_wall_clearance_m):
        bad.append(f"wall clearance {clearance_m*1000:.0f} mm < {cfg.min_wall_clearance_m*1000:.0f} mm")
    return not bad, bad


CSV_COLUMNS: Tuple[str, ...] = (
    "run_id", "timestamp", "state",
    "probe_tip_x", "probe_tip_y", "probe_tip_z",
    "probe_tip_qw", "probe_tip_qx", "probe_tip_qy", "probe_tip_qz",
    "sat_dock_x", "sat_dock_y", "sat_dock_z",
    "sat_dock_qw", "sat_dock_qx", "sat_dock_qy", "sat_dock_qz",
    "relative_x", "relative_y", "relative_z",
    "roll_error_deg", "pitch_error_deg", "yaw_error_deg", "orientation_error_deg",
    "axis_error_deg", "roll_about_axis_deg", "lateral_error_m",
    "geometry_distance_m", "depth_distance_m", "depth_raw_m", "depth_pixels",
    "insertion_depth_m", "wall_clearance_m",
    "relative_vx", "relative_vy", "relative_vz", "relative_speed_mps",
    "commanded_speed_mps", "alignment_valid", "depth_valid", "dock_ready", "dock_success",
)
