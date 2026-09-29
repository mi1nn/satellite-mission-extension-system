"""MRV Phase 1: AprilTag-based predictive capture of a linearly drifting 3 t MEP.

Runs on `srb/debris_capture_vision`. The controller never reads the ground truth:

    cam_wrist image -> 4 AprilTags -> 16-point PnP -> Cylinder_01 pose (vision)
    -> linear velocity -> Cylinder_01 at t + horizon -> EE target (robot/world frame)
    -> Canadarm3 differential IK (joint position targets) -> existing magnet capture
       (`CaptureManager.attach`, UsdPhysics.FixedJoint) -> holding -> slow retreat

Ground truth (the simulated Cylinder_01 / MEP state) is only used for logging and for
judging the tests.

`mep.motion_mode: six_dof` adds a combined roll/pitch/yaw rate: the prediction becomes
the constant-twist `ConstantTwistPredictor` (orientation integrated on SO(3), Cylinder_01
through a fixed transform), the EE reference and joint feed-forward follow the MEP
angular velocity, and the capture also requires a small relative angular velocity.
`translation_only` (default) keeps the Phase 1 behaviour.

`mrv.enabled` puts the MRV rendezvous phase (`mrv_approach.py`) in front of all of
this: the arm starts folded, the MRV (Canadarm3 base + spacecraft hull) translates to
its nominal pose in two legs with a thruster plume, and the arm is then deployed into
the pose the capture pipeline starts from. The MEP drifts and turns the whole time --
nothing in that phase touches it, the capture, the transport or the docking.

States (Section 26 of the task spec):
    [mrv.enabled: INIT -> MRV_MOVE_STEP_1 -> MRV_STEP_1_REACHED -> MRV_MOVE_STEP_2
     -> MRV_STEP_2_REACHED -> ARM_DEPLOY ->] SEARCH ...
    INIT -> SEARCH -> TAG_DETECTED -> POSE_ESTIMATED -> PREDICTING -> APPROACHING
    -> SLOW_APPROACH -> CAPTURE_ATTEMPT -> CAPTURED -> HOLDING -> RETREAT -> SUCCESS
    static scenario: ... PREDICTING -> STATIC_MEASURE -> SUCCESS
    failures: TAG_LOST, POSE_INVALID, PREDICTION_INVALID, APPROACH_TIMEOUT,
              CAPTURE_FAILED, PHYSICS_ERROR (arm holds its joints = safe stop)
    moving client (`client.release_enabled`, `moving_dock.py`):
    ... HOLDING (or --dock_only) -> CLIENT_RELEASE -> CLIENT_CRUISE -> CHASE
    -> VELOCITY_MATCHING -> RENDEZVOUS -> DOCK_TARGET_ACQUIRE ... DOCKED (unchanged)
    -> STABILIZING -> STOPPING -> ROBOT_RELEASE -> ARM_RETREAT -> MRV_SEPARATION -> SUCCESS
"""

import csv
import ctypes
import json
import math
import signal
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .docking_demo import ArmKinematics, interp_frame
from . import coupled_dock, moving_dock, probe_dock
from .frames import Frame, axis_angle, rotation_angle
from .moving_dock import (
    MrvVelocityController,
    StableTimer,
    VelocityEstimator,
    bounded_accel,
    matched,
    relative,
    separation_direction,
    stopped,
)
from .mrv_approach import (
    MrvTransit,
    ThrusterVfx,
    TransitLeg,
    back_propagate_free_body,
    joint_lerp,
    joint_span_deg,
    smoothstep,
    wrap_joint_target,
)
from .vision import (
    yaw_about_base,
    PROJECT_DIR,
    AprilTagDetector,
    ConstantTwistPredictor,
    Estimate,
    LinearDriftPredictor,
    TagDetection,
    VisionCaptureConfig,
    camera_in_link,
    downsample,
    estimate_constellation_pose,
    intrinsics_from_usd,
    pose_errors,
    project_points,
    rotmat_to_rpy_deg,
    scale_intrinsics,
    so3_exp,
    so3_log,
    tag_centres,
)

if TYPE_CHECKING:
    from isaacsim.simulation_app import SimulationApp

    from .vision_task import VisionCaptureTask


# The dashboard consumes ROS image topics; separate Isaac camera windows add
# render/UI load without providing any required functionality.
ENABLE_EXTRA_CAMERA_WINDOWS = False


class State(Enum):
    INIT = "INIT"
    ## MRV rendezvous phase (`mrv.enabled`, `mrv_approach.py`). Purely in front of the
    ## capture pipeline: SEARCH onwards is reached in exactly the state the verified
    ## pipeline starts in (MRV at its nominal pose, arm at the observation / yawed pose).
    MRV_MOVE_STEP_1 = "MRV_MOVE_STEP_1"
    MRV_STEP_1_REACHED = "MRV_STEP_1_REACHED"
    MRV_MOVE_STEP_2 = "MRV_MOVE_STEP_2"
    MRV_STEP_2_REACHED = "MRV_STEP_2_REACHED"
    ARM_DEPLOY = "ARM_DEPLOY"
    SEARCH = "SEARCH"
    TAG_DETECTED = "TAG_DETECTED"
    POSE_ESTIMATED = "POSE_ESTIMATED"
    PREDICTING = "PREDICTING"
    APPROACHING = "APPROACHING"
    SLOW_APPROACH = "SLOW_APPROACH"
    CAPTURE_ATTEMPT = "CAPTURE_ATTEMPT"
    CAPTURED = "CAPTURED"
    HOLDING = "HOLDING"
    RETREAT = "RETREAT"
    SUCCESS = "SUCCESS"
    STATIC_MEASURE = "STATIC_MEASURE"
    # failures
    TAG_LOST = "TAG_LOST"
    POSE_INVALID = "POSE_INVALID"
    PREDICTION_INVALID = "PREDICTION_INVALID"
    APPROACH_TIMEOUT = "APPROACH_TIMEOUT"
    CAPTURE_FAILED = "CAPTURE_FAILED"
    PHYSICS_ERROR = "PHYSICS_ERROR"
    ABORTED = "ABORTED"  # ROS cmd/abort
    MRV_APPROACH_FAILED = "MRV_APPROACH_FAILED"
    ## Phase 2 -- Ares1 probe -> satellite thruster docking (`probe_dock.py`).
    ## Enabled by `docking.enabled` (`vision_capture.py --dock`); the capture states
    ## above are unchanged and hand over after HOLDING / RETREAT.
    DOCK_TARGET_ACQUIRE = "DOCK_TARGET_ACQUIRE"
    PRE_DOCK_APPROACH = "PRE_DOCK_APPROACH"
    XY_ALIGN = "XY_ALIGN"
    # `docking_control.mode: coupled_predictive` only: lateral + attitude together
    # (replaces XY_ALIGN -> ORIENTATION_ALIGN; ALIGNMENT_CHECK follows as before)
    POSITION_ATTITUDE_ALIGN = "POSITION_ATTITUDE_ALIGN"
    ORIENTATION_ALIGN = "ORIENTATION_ALIGN"
    ALIGNMENT_CHECK = "ALIGNMENT_CHECK"
    Z_APPROACH = "Z_APPROACH"
    FINAL_INSERTION = "FINAL_INSERTION"
    DOCK_READY = "DOCK_READY"
    DOCKED = "DOCKED"
    DOCK_HOLDING = "DOCK_HOLDING"
    DOCK_FAILED = "DOCK_FAILED"
    ## Moving client (`client.release_enabled`, `moving_dock.py`). Wraps the docking
    ## states above: the capture hands over to CLIENT_RELEASE instead of
    ## DOCK_TARGET_ACQUIRE, RENDEZVOUS hands over to the unchanged docking states, and
    ## DOCKED continues with STABILIZING instead of DOCK_HOLDING.
    CLIENT_RELEASE = "CLIENT_RELEASE"
    CLIENT_CRUISE = "CLIENT_CRUISE"
    CHASE = "CHASE"
    VELOCITY_MATCHING = "VELOCITY_MATCHING"
    RENDEZVOUS = "RENDEZVOUS"
    STABILIZING = "STABILIZING"
    STOPPING = "STOPPING"
    ROBOT_RELEASE = "ROBOT_RELEASE"
    ARM_RETREAT = "ARM_RETREAT"
    MRV_SEPARATION = "MRV_SEPARATION"
    # failures (DOCKING_FAILED = DOCK_FAILED, PHYSICS_UNSTABLE = PHYSICS_ERROR)
    CLIENT_RELEASE_FAILED = "CLIENT_RELEASE_FAILED"
    VELOCITY_MATCH_TIMEOUT = "VELOCITY_MATCH_TIMEOUT"
    RENDEZVOUS_TIMEOUT = "RENDEZVOUS_TIMEOUT"
    STOP_FAILED = "STOP_FAILED"
    ROBOT_RELEASE_FAILED = "ROBOT_RELEASE_FAILED"
    ARM_RETREAT_FAILED = "ARM_RETREAT_FAILED"
    SEPARATION_COLLISION = "SEPARATION_COLLISION"
    SEPARATION_FAILED = "SEPARATION_FAILED"


FAILURES = {State.TAG_LOST, State.POSE_INVALID, State.PREDICTION_INVALID, State.APPROACH_TIMEOUT, State.CAPTURE_FAILED, State.PHYSICS_ERROR,
            State.ABORTED, State.DOCK_FAILED, State.MRV_APPROACH_FAILED,
            State.CLIENT_RELEASE_FAILED, State.VELOCITY_MATCH_TIMEOUT, State.RENDEZVOUS_TIMEOUT, State.STOP_FAILED,
            State.ROBOT_RELEASE_FAILED, State.ARM_RETREAT_FAILED, State.SEPARATION_COLLISION, State.SEPARATION_FAILED}
# MRV rendezvous phase: no vision, no capture, no docking runs in any of them
MRV_STATES = {State.MRV_MOVE_STEP_1, State.MRV_STEP_1_REACHED, State.MRV_MOVE_STEP_2,
              State.MRV_STEP_2_REACHED, State.ARM_DEPLOY}
# ... of which these translate the MRV (the only states with the thruster plume on)
MRV_MOTION = {State.MRV_MOVE_STEP_1, State.MRV_MOVE_STEP_2}
# States of the docking phase (the probe tip is the controlled frame in all of them)
DOCKING_STATES = {State.DOCK_TARGET_ACQUIRE, State.PRE_DOCK_APPROACH, State.XY_ALIGN, State.ORIENTATION_ALIGN,
                  State.POSITION_ATTITUDE_ALIGN, State.ALIGNMENT_CHECK, State.Z_APPROACH, State.FINAL_INSERTION, State.DOCK_READY,
                  State.DOCKED, State.DOCK_HOLDING}
# ... of which these actively command a motion towards the docking axis
DOCKING_MOTION = {State.PRE_DOCK_APPROACH, State.XY_ALIGN, State.ORIENTATION_ALIGN, State.POSITION_ATTITUDE_ALIGN,
                  State.Z_APPROACH, State.FINAL_INSERTION}
TERMINAL = FAILURES | {State.SUCCESS}
TRACKING = {State.APPROACHING, State.SLOW_APPROACH, State.CAPTURE_ATTEMPT, State.STATIC_MEASURE}
# Moving-client states before / after the (unchanged) docking states
MOVING_PRE_DOCK = {State.CLIENT_RELEASE, State.CLIENT_CRUISE, State.CHASE, State.VELOCITY_MATCHING, State.RENDEZVOUS}
MOVING_POST_DOCK = {State.STABILIZING, State.STOPPING, State.ROBOT_RELEASE, State.ARM_RETREAT, State.MRV_SEPARATION}
MOVING_STATES = MOVING_PRE_DOCK | MOVING_POST_DOCK
# ... in which the robot <-> MEP capture joint must exist
MOVING_ATTACHED = MOVING_PRE_DOCK | {State.STABILIZING, State.STOPPING}
# ... in which the MEP <-> client docking joint must exist
MOVING_DOCKED = MOVING_POST_DOCK
# ... in which the MRV holds its station relative to the client (v = v_client + Kp e)
MRV_TRACKS_CLIENT = {State.VELOCITY_MATCHING, State.RENDEZVOUS, State.STABILIZING, State.STOPPING} | DOCKING_STATES

# EE contact frame relative to the tag frame when facing it: half turn about X
FLIP = Frame(np.zeros(3), np.diag([1.0, -1.0, -1.0]))

CSV_COLUMNS = [
    # required (task spec Section 25)
    "timestamp",
    "gt_x", "gt_y", "gt_z", "gt_roll", "gt_pitch", "gt_yaw",
    "est_x", "est_y", "est_z", "est_roll", "est_pitch", "est_yaw",
    "position_error_mm", "angle_error_deg",
    "pred_x", "pred_y", "pred_z",
    "ee_x", "ee_y", "ee_z",
    "ee_target_x", "ee_target_y", "ee_target_z",
    "ee_target_distance_m",
    "relative_velocity_mps",
    "mep_linear_velocity_x", "mep_linear_velocity_y", "mep_linear_velocity_z",
    "mep_angular_velocity_x", "mep_angular_velocity_y", "mep_angular_velocity_z",
    "num_detected_tags",
    "capture_state", "capture_success",
    "total_elapsed_time",
    # extra
    "scenario", "est_age_s", "reproj_rms_px", "pnp_inliers", "raw_position_error_mm", "raw_angle_error_deg",
    "est_base_x", "est_base_y", "est_base_z",
    "gt_mep_linear_velocity_x", "gt_mep_linear_velocity_y", "gt_mep_linear_velocity_z",
    "gt_mep_angular_velocity_deg_s", "gt_relative_velocity_mps",
    "ee_to_cylinder_distance_m", "ee_lateral_error_m", "ee_gap_m", "ee_normal_angle_deg",
    "standoff_cmd_m", "fixed_joint_valid", "mep_ee_relative_drift_mm", "mep_ee_relative_drift_deg",
    # 6-DoF extension (appended so the Phase 1 columns keep their order).
    # Quaternions are (w, x, y, z), world frame; angular velocities rad/s, world frame.
    "motion_mode", "prediction_horizon_s",
    "gt_qw", "gt_qx", "gt_qy", "gt_qz",
    "est_qw", "est_qx", "est_qy", "est_qz",
    "pred_qw", "pred_qx", "pred_qy", "pred_qz",
    "ee_qw", "ee_qx", "ee_qy", "ee_qz",
    "ee_target_qw", "ee_target_qx", "ee_target_qy", "ee_target_qz",
    "gt_mep_angular_velocity_x", "gt_mep_angular_velocity_y", "gt_mep_angular_velocity_z",
    "ee_angular_velocity_x", "ee_angular_velocity_y", "ee_angular_velocity_z",
    # capture-condition quantities w.r.t. the vision estimate (the gt_* / ee_* ones above are GT)
    "est_ee_to_cylinder_distance_m", "est_ee_lateral_error_m", "est_ee_normal_angle_deg",
    "est_ee_orientation_error_deg", "est_relative_angular_velocity_rad_s",
    "gt_ee_orientation_error_deg", "gt_relative_angular_velocity_rad_s",
    "pnp_valid", "tracking_valid",
]


@dataclass
class VisionFrame:
    """Result of processing one `cam_wrist` image."""

    t: float
    detections: Dict[int, TagDetection]
    image: Optional[np.ndarray] = None
    cam_w: Optional[Frame] = None
    t_w_y: Optional[Frame] = None  # estimated Cylinder_01 (world)
    reproj_rms: float = math.nan
    inliers: int = 0
    valid: bool = False
    reason: str = ""


@dataclass
class Results:
    scenario: str
    checks: Dict[str, dict] = field(default_factory=dict)
    metrics: Dict[str, object] = field(default_factory=dict)
    failure: Optional[str] = None
    final_state: str = ""

    def check(self, name: str, ok: bool, detail: str = "", **values) -> bool:
        self.checks[name] = {"pass": bool(ok), "detail": detail, **values}
        print(f"[CHECK] {name}: {'PASS' if ok else 'FAIL'}{'  (' + detail + ')' if detail else ''}", flush=True)
        return ok


class Visualizer:
    """3D debug draw (`isaacsim.util.debug_draw`).

    Debug draw is rendered into every render product, including `cam_wrist`, where it
    would hide the tags. It is therefore cleared right before each render whose image
    is used for detection and redrawn right after.
    """

    COLORS = {
        "tag": (0.1, 0.5, 1.0, 1.0),
        "gt": (0.1, 1.0, 0.2, 1.0),
        "est": (1.0, 0.55, 0.0, 1.0),
        "pred": (1.0, 0.1, 1.0, 1.0),
        "ee": (0.0, 1.0, 1.0, 1.0),
        "target": (1.0, 1.0, 1.0, 1.0),
        "velocity": (1.0, 1.0, 0.0, 1.0),
    }
    # Linear velocity arrow = v * this [s] (the real 0.3 s displacement is only mm long)
    VELOCITY_ARROW_SEC = 10.0

    def __init__(self, enabled: bool):
        self.draw = None
        self.num_drawn = 0
        if not enabled:
            return
        try:
            from isaacsim.util.debug_draw import _debug_draw

            self.draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as e:  # extension missing -> 2D overlay only
            print(f"[VIS] debug draw unavailable: {e}", flush=True)
        self._items: Tuple[list, list, list, list, list, list, list] = ([], [], [], [], [], [], [])

    @property
    def available(self) -> bool:
        return self.draw is not None

    # Axis triads: X red, Y green, Z blue, scaled by `brightness`
    AXIS_COLORS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    def set_scene(self, tags_w: List[np.ndarray], centre_w, est_w, gt_w, pred_w, ee_w, target_w,
                  axes: Sequence[Tuple[Frame, float, float, float]] = (),
                  vectors: Sequence[Tuple[np.ndarray, np.ndarray, str, float]] = ()):
        """`axes`: (frame, axis length [m], line width, brightness) triads.
        `vectors`: (start, end, colour name, line width) segments."""
        pts, cols, sizes, a, b, lcol, lw = [], [], [], [], [], [], []

        def point(p, c, s):
            if p is not None:
                pts.append(tuple(float(x) for x in p))
                cols.append(self.COLORS[c] if isinstance(c, str) else c)
                sizes.append(s)

        def line(p, q, c, w):
            if p is not None and q is not None:
                a.append(tuple(float(x) for x in p))
                b.append(tuple(float(x) for x in q))
                lcol.append(self.COLORS[c] if isinstance(c, str) else c)
                lw.append(w)

        for t in tags_w:
            point(t, "tag", 10)
        for i in range(len(tags_w)):
            line(tags_w[i], tags_w[(i + 1) % len(tags_w)], "tag", 2)
        point(centre_w, "tag", 14)
        point(gt_w, "gt", 14)
        point(est_w, "est", 12)
        point(pred_w, "pred", 12)
        point(ee_w, "ee", 14)
        point(target_w, "target", 10)
        line(ee_w, target_w, "target", 1)
        for frame, length, width, bright in axes:
            if frame is None:
                continue
            for i, rgb in enumerate(self.AXIS_COLORS):
                line(frame.pos, frame.pos + length * frame.rot[:, i], (*(bright * c for c in rgb), 1.0), width)
        for p, q, c, width in vectors:
            line(p, q, c, width)
        self._items = (pts, cols, sizes, a, b, lcol, lw)

    def clear(self):
        if self.draw is not None:
            self.draw.clear_points()
            self.draw.clear_lines()

    def show(self):
        if self.draw is None:
            return
        self.clear()
        pts, cols, sizes, a, b, lcol, lw = self._items
        if pts:
            self.draw.draw_points(pts, cols, sizes)
        if a:
            self.draw.draw_lines(a, b, lcol, lw)
        self.num_drawn = max(self.num_drawn, len(pts) + len(a))


class VisionCaptureDemo:
    def __init__(self, env, sim_app: "SimulationApp", scenario: str, headless: bool, out_dir: Path, csv_path: Path, label: Optional[str] = None):
        assert scenario in ("static", "dynamic")
        self.env = env
        self.task: "VisionCaptureTask" = env.unwrapped
        t = self.task
        self.cfg: VisionCaptureConfig = t.vision_cfg
        self.scenario = scenario
        self.label = label or scenario  # run name (output file names, CSV `scenario` column)
        self.sim_app = sim_app
        self.headless = headless
        self.sim = t.sim
        self.scene = t.scene
        self.stage = t.scene.stage
        self.geo = t.docking_geometry
        self.capture = t._capture
        self.mep = t._obj
        self.camera = t.scene[self.cfg.camera.name]
        self.contacts = self.scene.sensors.get("contacts_robot")
        a = self.cfg.approach
        self.arm = ArmKinematics(t, self.geo.ee_contact_link, a.ik_lambda)
        self.dt = self.sim.get_physics_dt()
        self.render_interval = max(1, t.cfg.sim.render_interval)

        ## Fixed transforms
        self.t_l_c = camera_in_link(self.cfg.camera)  # link -> camera (mount)
        self.t_t_y = t.t_t_y  # tag constellation -> Cylinder_01 (design time)
        self.t_y_t = t.t_t_y.inv()
        self.t_m_y = t.t_m_y  # MEP body -> Cylinder_01 (ground truth, logging only)
        self.t_m_t = t.t_m_t
        self.face_offset = self.cfg.apriltag.face_offset_m

        ## Intrinsics (from the Isaac Sim camera) + cross-check against the USD prim.
        ## With supersampling the camera renders `ss` x larger; the processed image is
        ## area-downsampled to the configured resolution and K scaled accordingly.
        self.ss = int(self.cfg.camera.supersample)
        self.k_render = self.camera.data.intrinsic_matrices[0].cpu().numpy().astype(float)
        self.k = scale_intrinsics(self.k_render, self.ss)
        prim = self.stage.GetPrimAtPath(f"{self.geo.link_path}/{self.cfg.camera.name}")
        from pxr import UsdGeom

        cam = UsdGeom.Camera(prim)
        self.usd_camera = {
            "prim": str(prim.GetPath()),
            "focal_length_mm": float(cam.GetFocalLengthAttr().Get()),
            "horizontal_aperture_mm": float(cam.GetHorizontalApertureAttr().Get()),
            "vertical_aperture_mm": float(cam.GetVerticalApertureAttr().Get()),
        }
        self.k_usd = intrinsics_from_usd(self.usd_camera["focal_length_mm"], self.usd_camera["horizontal_aperture_mm"], self.cfg.camera.width, self.cfg.camera.height)
        self.k_usd_render = intrinsics_from_usd(self.usd_camera["focal_length_mm"], self.usd_camera["horizontal_aperture_mm"], self.cfg.camera.width * self.ss, self.cfg.camera.height * self.ss)

        self.detector = AprilTagDetector(self.cfg.apriltag)
        ## translation_only: Phase 1 predictor (unchanged). six_dof: constant twist of a
        ## reference frame R rigid on the MEP, mapped to Cylinder_01 with a fixed
        ## transform: T_W_Y(t+h) = T_W_R(t+h) @ T_R_Y
        ##   tag:     R = tag constellation T (the PnP-measured frame), T_R_Y = T_T_Y
        ##   mep_com: R = MEP centre of mass with the MEP body axes, T_R_Y = T_COM_M @ T_M_T @ T_T_Y.
        ##            The COM offset in the MEP body is a mass property of the model, read
        ##            once here (like T_T_Y is measured once from the USD) -- never the live pose.
        self.six_dof = self.cfg.mep.six_dof
        root_rot = Frame.from_pos_quat(np.zeros(3), self.mep.data.root_quat_w[0].tolist()).rot
        self.com_in_mep = root_rot.T @ (self.mep.data.root_com_pos_w[0] - self.mep.data.root_pos_w[0]).cpu().numpy()
        if self.cfg.prediction.reference_frame == "mep_com":
            self.t_ref_y = Frame(self.com_in_mep, np.eye(3)).inv() @ self.t_m_t @ self.t_t_y
        else:
            self.t_ref_y = self.t_t_y
        if self.six_dof:
            self.predictor = ConstantTwistPredictor(self.cfg.prediction, self.t_ref_y)
        else:
            self.predictor = LinearDriftPredictor(self.cfg.prediction)
        self.vis = Visualizer(self.cfg.logging.debug_draw)
        ## Optional ROS 2 interface (telemetry out, start / abort / capture_enable in)
        self.ros = None
        self._next_ros_t = 0.0
        self._next_camera_stream_t = 0.0
        self._camera_stream_period_s = 1.0 / min(float(self.cfg.ros.publish_rate_hz), 5.0)
        self._live_capture_warmup_frames = 10
        self._rendered_frame_count = 0
        self._start_wait_logged = False
        self._capture_wait_logged = False
        ## Astrobee observation camera (`astrobee.py`): flies and streams its camera only;
        ## nothing of the mission reads it. Its one input is the existing latched state
        ## topic `/<ros.namespace>/state` (docking complete -> follow the MRV retreat).
        ## Created before the vision ROS node so the camera feed (which keeps going in
        ## `idle()`) owns the rclpy context.
        self.astrobee = None
        if self.cfg.astrobee.enabled:
            from .astrobee import AstrobeeObserver

            self.astrobee = AstrobeeObserver(t, self.cfg.astrobee, self.cfg.ros.enabled, self.cfg.ros.distro, headless,
                                             out_dir, self.label, mission_state_topic=f"/{self.cfg.ros.namespace}/state")
        if self.cfg.ros.enabled:
            from .ros_interface import VisionRosInterface

            self.ros = VisionRosInterface(self.cfg.ros, (self.cfg.camera.width, self.cfg.camera.height), self.k, self.cfg.camera.name)
        ## Main interactive viewport (API capture, not a screen share): GUI only.
        self._next_viewport_t = 0.0
        self._viewport_period_s = 1.0 / 10.0  # Viewport-only target; does not change simulation dt.
        self._viewport_api = None
        self._viewport_capture_pending = False
        if not headless and self.ros is not None:
            try:
                from omni.kit.viewport.utility import get_active_viewport

                self._viewport_api = get_active_viewport()
            except Exception as e:
                print(f"[VIEWPORT] could not get the active viewport: {e} -- viewport publishing disabled", flush=True)

        ## State
        self.state = State.INIT
        self.state_time = 0.0
        self.sim_time = 0.0
        self.wall_start = time.time()
        self.done = False
        self.results = Results(scenario)
        self.ref: Optional[Frame] = None  # EE contact reference pose (rate limited)
        self.goal: Optional[Frame] = None  # EE contact target pose (from prediction)
        self.standoff: float = a.approach_standoff_m
        self.q_hold: Optional[torch.Tensor] = None
        self.last_vision: Optional[VisionFrame] = None
        self.last_valid: Optional[VisionFrame] = None
        self._next_vision_t = 0.0
        self._next_log_t = 0.0
        self._next_overlay_t = 0.0
        self._next_status_t = 0.0
        self._settled_since: Optional[float] = None
        self._capture_range_logged = False
        self._capture_info: Dict[str, float] = {}
        self._rel_at_capture: Optional[Frame] = None
        self._hold: Dict[str, float] = {}
        self._static_idx = 0
        self._static_samples: List[Tuple[float, float, float]] = []
        self._static_stats: List[dict] = []
        self._retreat: Optional[dict] = None
        self._max_contact = 0.0
        self._ambiguous = 0
        self._cylinder_color = None
        self.rows: List[dict] = []
        ## Phase 2 (docking): bookkeeping, probe camera handles, CSV
        self._dock: Dict[str, object] = {}
        self._dock_m_latest: Optional[Dict[str, float]] = None
        self._dock_ref: Optional[Frame] = None
        self._dock_align_since: Optional[float] = None
        self._dock_axial_hold: Optional[float] = None
        self._dock_started: Optional[float] = None
        self._dock_speed = 0.0
        self._calib_rejected = False
        self._depth_calibrated_t: Optional[float] = None
        self._tip_history: List[Tuple[float, np.ndarray]] = []
        self._depth_offset: Optional[float] = None
        self._depth_reason: str = "not read yet"
        self._dock_rows: List[dict] = []
        self._dock_csv = self._dock_csv_file = None
        ## Docking controller (`docking_control.mode`); the metrics / CSV run in both modes
        self.coupled = self.cfg.docking_control.mode == "coupled_predictive"
        self._cd_tracker = coupled_dock.TRACKERS.get(self.cfg.docking_control.mode, coupled_dock.CoupledPoseTracker)(self.cfg.docking_control)
        self._cd_gate = coupled_dock.AlignmentGate(self.cfg.docking_alignment)
        self._cd_metrics = coupled_dock.DockingControlMetrics()
        self._cd_tip_twist = coupled_dock.TwistEstimator()
        self._cd_active = False  # the coupled tracker has taken over the reference
        self._cd_axial: Optional[float] = None  # axial set point of the desired tip [m]
        self._cd_axial_rate = 0.0  # its rate [m/s] (fed forward along the docking axis)
        self._cd_last: Optional[Dict[str, object]] = None
        self._cd_scale = 1.0
        self._cd_out: Optional[coupled_dock.TrackerOutput] = None
        self._cd_csv = self._cd_csv_file = None
        self._probe_frames: List[Path] = []
        self._next_dock_log_t = 0.0
        self._next_probe_rgb_t = 0.0
        self._probe_cam = None
        self._probe_k = np.eye(3)
        self._cam_to_tip = 0.0
        if self.cfg.docking.enabled and self.cfg.probe_camera.enabled:
            self._probe_cam = t.scene[self.cfg.probe_camera.name]
            self._probe_k = self._probe_cam.data.intrinsic_matrices[0].cpu().numpy().astype(float)
            # Negative: the camera sits in *front* of the tip (see `probe_dock.depth_to_dock_distance`)
            self._cam_to_tip = -float(self.cfg.probe_camera.offset_from_tip_m)
        # six_dof: GT pose at t = 0 for the runtime angular-velocity frame check
        self._w_frame_ref: Optional[Tuple[float, Frame]] = None
        self._w_frame_result: Optional[dict] = None
        ## MRV rendezvous phase (`mrv_approach.py`). All of it is inactive -- and every
        ## handle below stays None -- when `mrv.enabled` is false, so the verified
        ## capture / transport / docking pipeline runs exactly as before.
        self.mrv: Optional[MrvTransit] = None
        self.vfx: Optional[ThrusterVfx] = None
        self._leg: Optional[TransitLeg] = None
        self._leg_targets: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._q_folded: Optional[torch.Tensor] = None
        self._q_deployed: Optional[torch.Tensor] = None
        self._deploy_alpha = 0.0
        self._mrv_log: Dict[str, object] = {}
        # sim_time at which the capture pipeline takes over (SEARCH). Every capture-phase
        # deadline is measured from here, so `test.dynamic_timeout_sec` keeps its meaning
        # whether or not the approach ran ahead of it.
        self._capture_t0 = 0.0
        ## Moving-client scenario (`moving_dock.py`). Inactive -- nothing below is used
        ## and every existing path runs unchanged -- unless `client.release_enabled`.
        self.moving = bool(self.cfg.client.release_enabled)
        self._mv_active = False  # set once the capture hands over to CLIENT_RELEASE
        self._client_live = False  # client released at t = 0, before the docking phase
        rv = self.cfg.rendezvous
        self._mv_ctrl = MrvVelocityController(rv.mrv_max_accel_mps2, rv.position_gain_hz, rv.max_correction_speed_mps)
        self._mv_est = {k: VelocityEstimator(rv.velocity_window_s) for k in ("client", "mep", "mrv")}
        self._mv_timer = StableTimer()
        self._mv_d_ref: Optional[np.ndarray] = None  # verified client -> MRV offset (world)
        self._mv_sep_dir: Optional[np.ndarray] = None
        self._mv_force: Dict[str, np.ndarray] = {}
        # Arm reference twist (v, w) [m/s, rad/s, world] last sent as joint velocity target,
        # at the IK tool point (`set_velocity_feedforward`); v_ref of the Astrobee assist
        self._ee_ref_twist: Tuple[np.ndarray, np.ndarray] = (np.zeros(3), np.zeros(3))
        # Astrobee damping assist thrust currently on the MEP (None: off)
        self._assist_force: Optional[np.ndarray] = None
        self._mvm: Dict[str, object] = {}  # measurements of the current step
        self._mv: Dict[str, object] = {}  # bookkeeping / metrics
        self._arm_stow: Optional[dict] = None  # MRV_SEPARATION arm return (`arm_stow_step`)
        self._mv_rows: List[dict] = []
        self._next_mv_log_t = 0.0
        # Motion trail (world-fixed start cross + a point every `motion_trail_period_s`)
        # and the GUI speed window -- visual only (`logging.motion_trail` / `motion_hud`)
        self._trail: Dict[str, List[np.ndarray]] = {"client": [], "mep": [], "mrv": []}
        self._next_trail_t = 0.0
        self._hud = None
        self._hud_labels: List[object] = []

        ## Output
        self.out_dir = out_dir
        self.overlay_dir = out_dir / f"{self.label}_overlay"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        for old in self.overlay_dir.glob("*.png"):  # overlays of a previous run
            old.unlink()
        self.probe_dir_name = f"{self.label}_probe_rgb"
        self._probe_dir = out_dir / self.probe_dir_name
        if self.cfg.docking.enabled:
            self._probe_dir.mkdir(parents=True, exist_ok=True)
            for old in self._probe_dir.glob("*.png"):
                old.unlink()
            self.dock_csv_path = out_dir / f"{self.label}_docking.csv"
            if self.cfg.logging.csv_enabled:
                self._dock_csv_file = open(self.dock_csv_path, "w", newline="")
                self._dock_csv = csv.DictWriter(self._dock_csv_file, fieldnames=list(probe_dock.CSV_COLUMNS))
                self._dock_csv.writeheader()
            self.dock_control_csv_path = out_dir / f"{self.label}_docking_control.csv"
            if self.cfg.logging.csv_enabled:
                self._cd_csv_file = open(self.dock_control_csv_path, "w", newline="")
                self._cd_csv = csv.DictWriter(self._cd_csv_file, fieldnames=list(coupled_dock.CSV_COLUMNS))
                self._cd_csv.writeheader()
        self._mv_csv = self._mv_csv_file = None
        if self.moving:
            self.moving_csv_path = out_dir / f"{self.label}_moving.csv"
            if self.cfg.logging.csv_enabled:
                self._mv_csv_file = open(self.moving_csv_path, "w", newline="")
                self._mv_csv = csv.DictWriter(self._mv_csv_file, fieldnames=list(moving_dock.CSV_COLUMNS))
                self._mv_csv.writeheader()
        self.csv_path = csv_path
        self._csv_file = self._csv = None
        if self.cfg.logging.csv_enabled:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(csv_path, "w", newline="")
            self._csv = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
            self._csv.writeheader()
        self._overlays: List[Path] = []

    ##############
    ### Frames ###
    ##############

    def mep_frame(self) -> Frame:
        return Frame.from_pos_quat(self.mep.data.root_pos_w[0].tolist(), self.mep.data.root_quat_w[0].tolist())

    def gt_cylinder(self) -> Frame:
        """Ground truth Cylinder_01 pose (logging / test judgement only)."""
        return self.mep_frame() @ self.t_m_y

    def ee_pose(self) -> Frame:
        return self.arm.link_pose() @ self.geo.ee_contact_link

    def cam_pose(self) -> Frame:
        return self.arm.link_pose() @ self.t_l_c

    def base_frame(self) -> Frame:
        return self.arm.base

    def ee_velocity(self) -> np.ndarray:
        """Linear velocity of the EE contact point (proprioception)."""
        lid = self.arm.link_id
        v = self.task._robot.data.body_lin_vel_w[0, lid].cpu().numpy()
        w = self.task._robot.data.body_ang_vel_w[0, lid].cpu().numpy()
        r = self.arm.link_pose().rot @ self.geo.ee_contact_link.pos
        return v + np.cross(w, r)

    def ee_angular_velocity(self) -> np.ndarray:
        """Angular velocity [rad/s] (world) of the EE (= last link, rigid tool)."""
        return self.task._robot.data.body_ang_vel_w[0, self.arm.link_id].cpu().numpy()

    def est_mep_twist_at(self, p: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Vision-estimated MEP velocity at world point `p` and angular velocity (world).

        translation_only: (v, None) -- exactly the Phase 1 velocity. six_dof: the rigid
        velocity field v + w x (p - p_ref) and w."""
        v, w = self.predictor.twist()
        if v is None or w is None:
            return v, w
        return self.predictor.velocity_at(p, self.sim_time), w

    def feedforward(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Feed-forward (v, w) for the EE reference: MEP motion at the EE contact point."""
        return self.est_mep_twist_at(self.ee_pose().pos)

    def gt_mep_velocity_at(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = self.mep.data
        v = d.root_com_lin_vel_w[0].cpu().numpy()
        w = d.root_com_ang_vel_w[0].cpu().numpy()
        c = d.root_com_pos_w[0].cpu().numpy()
        return v + np.cross(w, p - c), w

    def ee_goal(self, t_w_t: Frame, standoff: float) -> Frame:
        """EE contact pose facing the tag frame, `standoff` from the MEP face."""
        return t_w_t @ Frame(np.array([0.0, 0.0, standoff - self.face_offset]), np.eye(3)) @ FLIP

    ##############
    ### Vision ###
    ##############

    def estimate_now(self) -> Optional[Frame]:
        """System estimate of Cylinder_01 now: fitted position, window-mean orientation."""
        if self.last_valid is None:
            return None
        return self.predictor.estimate(self.sim_time)

    def predicted(self) -> Optional[Frame]:
        return self.predictor.predict(self.sim_time)

    def estimate_age(self) -> float:
        return math.inf if self.last_valid is None else self.sim_time - self.last_valid.t

    def process_image(self):
        """Detect + PnP on the image rendered at the current physics state."""
        rgb = self.camera.data.output["rgb"][0].cpu().numpy()
        img = np.ascontiguousarray(downsample(rgb[..., :3], self.ss))
        dets = self.detector.detect(img)
        vf = VisionFrame(self.sim_time, dets, image=img, cam_w=self.cam_pose())
        # Prior for the planar ambiguity: fused orientation of the constellation seen
        # from the current camera pose
        prior = None
        # (only once the window holds enough images, so a single early wrong solution
        # cannot become its own prior)
        # (six_dof: the fitted orientation propagated to this image's time)
        fused = self.predictor.orientation_at(vf.t) if len(self.predictor.history) >= self.cfg.prediction.min_samples else None
        if fused is not None:
            prior = vf.cam_w.rot.T @ fused @ self.t_y_t.rot
        pose = estimate_constellation_pose(dets, self.cfg.apriltag, self.k, prior_rot_c=prior)
        if pose is not None and pose.ambiguous:
            self._ambiguous += 1
        if pose is None:
            vf.reason = f"{len(dets)}/4 tags" if len(dets) < self.cfg.apriltag.min_tags_for_pose else "PnP failed"
        elif pose.reproj_rms_px > self.cfg.apriltag.max_reprojection_rms_px:
            vf.reason = f"reprojection RMS {pose.reproj_rms_px:.2f} px"
            vf.reproj_rms = pose.reproj_rms_px
        else:
            vf.t_w_y = vf.cam_w @ pose.t_c_t @ self.t_t_y
            vf.reproj_rms = pose.reproj_rms_px
            vf.inliers = pose.num_inliers
            vf.valid = True
            self.predictor.add(Estimate(vf.t, vf.t_w_y))
            self.last_valid = vf
        self.last_vision = vf
        if self.sim_time >= self._next_overlay_t:
            self._next_overlay_t = self.sim_time + self.cfg.logging.overlay_every_sec
            self.save_overlay(vf)

    ###############
    ### Control ###
    ###############

    def track(self, goal: Frame, speed: float, v_ff: Optional[np.ndarray], max_step: Optional[float] = None,
              w_ff: Optional[np.ndarray] = None, lag_limit: float = 0.05):
        """Move the reference towards `goal` (rate limited + feed-forward), then IK.

        `v_ff` [m/s] / `w_ff` [rad/s] (world): MEP motion the reference follows every
        step; `w_ff` is only given in six_dof (rotating target)."""
        a = self.cfg.approach
        if self.ref is None:
            self.ref = self.ee_pose()
        ee = self.ee_pose()
        lag_p, lag_a = pose_errors(ee, self.ref)
        pos = self.ref.pos.copy()
        if v_ff is not None:
            pos = pos + v_ff * self.dt
        rot = self.ref.rot
        if w_ff is not None:
            rot = so3_exp(w_ff * self.dt) @ rot  # world-frame w: left multiplication
        # The reference only advances while the arm keeps up with it
        if lag_p < lag_limit and lag_a < 3.0:
            d = goal.pos - pos
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                pos = pos + d / n * min(n, speed * self.dt)
            ang = rotation_angle(rot, goal.rot)
            if ang > 1e-9:
                s = min(1.0, math.radians(a.speed_ang_deg_s) * self.dt / ang)
                rot = interp_frame(Frame(pos, rot), Frame(pos, goal.rot), s).rot
        self.ref = Frame(pos, rot)
        q = self.arm.ik_joint_target(self.ref, a.max_joint_step_rad if max_step is None else max_step)
        self.task._robot.set_joint_position_target(q, joint_ids=self.arm.joint_ids)
        self.set_velocity_feedforward(v_ff, w_ff)

    def set_velocity_feedforward(self, v_ff: Optional[np.ndarray], w_ff: Optional[np.ndarray] = None):
        """Joint velocity targets that move the EE with the estimated MEP velocity.

        The implicit joint drives are PD on position *and* velocity. With position
        targets only, a target moving at v is trailed by v * damping / stiffness
        (0.625 s for the Canadarm3); q_dot* = J^+ [v; w] removes that lag (w = 0 unless
        six_dof gives the MEP angular velocity).
        """
        n = len(self.arm.joint_ids)
        self._ee_ref_twist = (self.mrv_base_velocity() if v_ff is None else np.asarray(v_ff, dtype=float).copy(),
                              np.zeros(3) if w_ff is None else np.asarray(w_ff, dtype=float).copy())
        if v_ff is None:
            qd = torch.zeros((1, n), device=self.arm.device)
        else:
            # The joints only produce the EE motion RELATIVE to the base. The base is
            # fixed in every scenario except the moving client, where the MRV translates
            # (`mrv_base_velocity` is zero otherwise, so this is the previous formula).
            v_ff = np.asarray(v_ff, dtype=float) - self.mrv_base_velocity()
            j = self.arm.jacobian_base()[0]  # (6, n), base frame, EE contact point
            twist = torch.zeros(6, dtype=j.dtype, device=j.device)
            twist[:3] = torch.tensor(self.arm.base.rot.T @ v_ff, dtype=j.dtype, device=j.device)
            if w_ff is not None:
                twist[3:] = torch.tensor(self.arm.base.rot.T @ w_ff, dtype=j.dtype, device=j.device)
            lam = self.cfg.approach.ik_lambda
            jjt = j @ j.T + (lam ** 2) * torch.eye(6, dtype=j.dtype, device=j.device)
            qd = (j.T @ torch.linalg.solve(jjt, twist)).unsqueeze(0)
        self.task._robot.set_joint_velocity_target(qd, joint_ids=self.arm.joint_ids)

    def hold(self):
        if self.q_hold is None:
            self.q_hold = self.arm.joint_pos().clone()
        self.task._robot.set_joint_position_target(self.q_hold, joint_ids=self.arm.joint_ids)
        self.set_velocity_feedforward(None)

    def goto(self, state: State, reason: str = ""):
        print(f"[STATE] {self.state.value} -> {state.value}  (t={self.sim_time:.2f} s){'  ' + reason if reason else ''}", flush=True)
        if state in FAILURES:
            self.results.failure = f"{state.value}: {reason}"
            print(f"[FAIL] {state.value}: {reason}", flush=True)
            self.q_hold = None  # safe stop: hold the joints where they are
        if state in TERMINAL:
            # Moving client: no thrust stays on after the end (the wrench buffers would keep
            # being applied); the MRV keeps its station on the client while the MEP is held,
            # and stops otherwise (`moving_after_step`)
            self.clear_wrenches()
        if getattr(self, "_cd_metrics", None) is not None:
            self._cd_metrics.transition(self.state.value, state.value)
        self.state = state
        self.state_time = 0.0
        self._settled_since = None
        self._mv_timer.reset()

    ##################
    ### Conditions ###
    ##################

    def capture_metrics(self, y: Optional[Frame], v_mep: Optional[np.ndarray], w_mep: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Capture conditions w.r.t. a Cylinder_01 pose `y` (estimated or GT).

        `v_mep`: MEP velocity at the EE contact point [m/s]; `w_mep`: MEP angular
        velocity [rad/s] (world), only needed for `rel_ang_vel`."""
        ee = self.ee_pose()
        if y is None:
            return {k: math.nan for k in ("distance", "gap", "lateral", "angle", "orientation", "rel_vel", "rel_ang_vel")}
        t = y @ self.t_y_t  # tag frame (face normal = +Z)
        n = t.rot[:, 2]
        rel = ee.pos - t.pos
        gap_t = float(rel @ n)
        return {
            "distance": float(np.linalg.norm(ee.pos - y.pos)),
            "gap": gap_t + self.face_offset,  # EE face above the MEP face
            "lateral": float(np.linalg.norm(rel - gap_t * n)),
            # tilt of the EE axis w.r.t. the face normal (the capture condition)
            "angle": math.degrees(axis_angle(ee.rot[:, 2], -n)),
            # full 3-axis orientation error to the EE capture orientation (incl. roll about
            # the axis; logged, the rotationally symmetric magnet does not need it)
            "orientation": math.degrees(rotation_angle(ee.rot, (t @ FLIP).rot)),
            "rel_vel": float(np.linalg.norm(self.ee_velocity() - v_mep)) if v_mep is not None else math.nan,
            "rel_ang_vel": float(np.linalg.norm(self.ee_angular_velocity() - w_mep)) if w_mep is not None else math.nan,
        }

    def est_capture_metrics(self) -> Dict[str, float]:
        """Capture metrics w.r.t. the vision estimate (what the controller uses)."""
        v, w = self.est_mep_twist_at(self.ee_pose().pos)
        return self.capture_metrics(self.estimate_now(), v, w)

    def gt_capture_metrics(self) -> Dict[str, float]:
        """Capture metrics w.r.t. the ground truth (logging / judgement only)."""
        v_gt, w_gt = self.gt_mep_velocity_at(self.ee_pose().pos)
        return self.capture_metrics(self.gt_cylinder(), v_gt, w_gt)

    def tracking_ok(self) -> Tuple[bool, str]:
        lv = self.last_vision
        if self.estimate_age() > self.cfg.vision.tag_loss_timeout_s:
            return False, f"no valid pose for {self.estimate_age():.2f} s"
        if lv is None or not lv.valid:
            return False, f"latest image invalid ({lv.reason if lv else 'none'})"
        if self.predictor.velocity() is None:
            return False, "no velocity estimate"
        return True, "ok"

    def capture_conditions(self) -> Tuple[bool, Dict[str, float], List[str]]:
        c = self.cfg.capture
        # translation_only: MEP velocity v (as in Phase 1); six_dof: v + w x r at the EE
        m = self.est_capture_metrics()
        failed = []
        if not m["distance"] <= c.max_distance_m:
            failed.append(f"distance {m['distance']:.3f} m > {c.max_distance_m}")
        if not m["angle"] <= c.max_angle_deg:
            failed.append(f"angle {m['angle']:.2f} deg > {c.max_angle_deg}")
        if not m["rel_vel"] <= c.max_relative_velocity_mps:
            failed.append(f"relative velocity {m['rel_vel']:.4f} m/s > {c.max_relative_velocity_mps}")
        if self.six_dof and not m["rel_ang_vel"] <= c.max_relative_angular_velocity_rad_s:
            failed.append(f"relative angular velocity {m['rel_ang_vel']:.4f} rad/s > {c.max_relative_angular_velocity_rad_s}")
        if not (m["gap"] > 0.0 and m["lateral"] <= c.max_lateral_m):
            failed.append(f"approach direction (gap {m['gap']*1000:.1f} mm, lateral {m['lateral']*1000:.1f} mm)")
        ok, why = self.tracking_ok()
        if not ok:
            failed.append(f"tracking: {why}")
        return not failed, m, failed

    def check_physics(self) -> Optional[str]:
        d = self.mep.data
        vals = [d.root_pos_w, d.root_quat_w, d.root_com_lin_vel_w, d.root_com_ang_vel_w, self.task._robot.data.joint_pos, self.task._robot.data.joint_vel]
        if not all(bool(torch.isfinite(x).all()) for x in vals):
            return "NaN/Inf in the MEP or robot state"
        v = float(torch.norm(d.root_com_lin_vel_w[0]))
        w = math.degrees(float(torch.norm(d.root_com_ang_vel_w[0])))
        if v > 0.5:
            return f"MEP speed {v:.3f} m/s (explosion)"
        # Phase 1 limit 5 deg/s on top of the commanded rate (0 in translation_only)
        w_limit = 5.0 + math.degrees(float(np.linalg.norm(self.cfg.mep.angular_velocity_w())))
        if w > w_limit:
            return f"MEP angular rate {w:.2f} deg/s > {w_limit:.2f} (unintended rotation)"
        qd = float(torch.abs(self.task._robot.data.joint_vel[0]).max())
        if qd > 1.0:
            return f"robot joint speed {qd:.2f} rad/s (joint explosion)"
        if (self.state in (State.CAPTURED, State.HOLDING, State.RETREAT) or self.state in DOCKING_STATES
                or self.state in MOVING_ATTACHED) and not self.capture.is_attached(0):
            return "FixedJoint missing after capture"
        if self._mv_active or self._client_live:
            err = self.check_client_physics()
            if err is not None:
                return err
        if self.contacts is not None and not self.capture.is_attached(0):
            f = float(torch.norm(self.contacts.data.net_forces_w[0], dim=-1).max())
            self._max_contact = max(self._max_contact, f)
            if f > 200.0:
                return f"unexpected contact on the arm: {f:.0f} N"
        return None

    ###############
    ### Stepping ###
    ###############

    def start(self):
        """t = 0: report the setup and (optionally) place the arm at the observation pose."""
        c = self.cfg
        r = self.results
        t = self.task
        print("[INIT] ---- MRV Phase 1 vision capture ----", flush=True)
        self.open_motion_hud()  # monitor window for the whole run (GUI only)
        print(f"[INIT] scenario {self.scenario}, motion mode {c.mep.motion_mode}, MEP mass {float(self.mep.root_physx_view.get_masses().sum()):.1f} kg, gravity {tuple(t.cfg.sim.gravity)}", flush=True)
        v0 = self.mep.data.root_com_lin_vel_w[0].cpu().numpy()
        w0 = self.mep.data.root_com_ang_vel_w[0].cpu().numpy()
        v_cmd, w_cmd = c.mep.linear_velocity_w(), c.mep.angular_velocity_w()
        print(f"[INIT] MEP initial linear velocity {np.round(v0, 4).tolist()} m/s (|v| {np.linalg.norm(v0):.4f}), angular {np.round(w0, 5).tolist()} rad/s "
              f"= {np.round(np.degrees(w0), 4).tolist()} deg/s (world)", flush=True)
        if not self.six_dof and np.linalg.norm(c.mep.angular_velocity_rad_s) > 0.0:
            print(f"[INIT] translation_only: mep.angular_velocity_rad_s {c.mep.angular_velocity_rad_s} is not applied", flush=True)
        # MEP centre of mass relative to the root origin (world): the point the linear
        # velocity refers to, and the lever arm of the rotation
        com_offset = (self.mep.data.root_com_pos_w[0] - self.mep.data.root_pos_w[0]).cpu().numpy()
        self._w_frame_ref = (self.sim_time, self.mep_frame())
        print(f"[INIT] K (cam_wrist) = {np.round(self.k, 3).tolist()}", flush=True)
        print(f"[INIT] USD camera {self.usd_camera}", flush=True)
        print(f"[INIT] T_link_cam pos {self.t_l_c.pos.tolist()} quat(wxyz) {np.round(self.t_l_c.quat, 4).tolist()}", flush=True)
        print(f"[INIT] T_mep_tags pos {np.round(self.t_m_t.pos, 4).tolist()} quat {np.round(self.t_m_t.quat, 4).tolist()}", flush=True)
        print(f"[INIT] T_tags_cylinder01 pos {np.round(self.t_t_y.pos, 5).tolist()} quat {np.round(self.t_t_y.quat, 5).tolist()}", flush=True)
        print(f"[INIT] Cylinder_01 radius {t.cylinder_radius*1000:.1f} mm, height {t.cylinder_height*1000:.1f} mm", flush=True)
        mass = float(self.mep.root_physx_view.get_masses().sum())
        r.metrics["setup"] = {
            "mep_mass_kg": mass,
            "gravity": list(t.cfg.sim.gravity),
            "mep_initial_linear_velocity_mps": v0.tolist(),
            "mep_initial_angular_velocity_deg_s": np.degrees(w0).tolist(),
            "mep_initial_angular_velocity_rad_s": w0.tolist(),
            "motion_mode": c.mep.motion_mode,
            "commanded_linear_velocity_mps": v_cmd.tolist(),
            "commanded_angular_velocity_rad_s_world": w_cmd.tolist(),
            "mep_com_offset_from_root_w_m": com_offset.tolist(),
            "mep_com_in_body_m": self.com_in_mep.tolist(),
            "prediction_reference_frame": c.prediction.reference_frame,
            "t_reference_cylinder01": {"pos": self.t_ref_y.pos.tolist(), "quat_wxyz": list(self.t_ref_y.quat)},
            "prediction_horizon_s": c.prediction.horizon_sec,
            "physics_dt": self.dt,
            "render_interval": self.render_interval,
            "K": self.k.tolist(),
            "K_render": self.k_render.tolist(),
            "supersample": self.ss,
            "K_from_usd_attributes": self.k_usd.tolist(),
            "usd_camera": self.usd_camera,
            "t_link_cam": {"pos": self.t_l_c.pos.tolist(), "quat_wxyz": list(self.t_l_c.quat)},
            "t_mep_tags": {"pos": self.t_m_t.pos.tolist(), "quat_wxyz": list(self.t_m_t.quat)},
            "t_tags_cylinder01": {"pos": self.t_t_y.pos.tolist(), "quat_wxyz": list(self.t_t_y.quat)},
            "cylinder01_radius_m": t.cylinder_radius,
            "cylinder01_height_m": t.cylinder_height,
            "tag_layout": t.layout_detail,
            "ee_contact_in_link": {"pos": self.geo.ee_contact_link.pos.tolist(), "quat_wxyz": list(self.geo.ee_contact_link.quat)},
            "config": c.to_dict(),
        }
        r.check("MEP mass is 3000 kg", abs(mass - c.mep.mass_kg) < 1.0 and abs(c.mep.mass_kg - 3000.0) < 1e-6, f"{mass:.1f} kg")
        r.check("Zero gravity", all(abs(g) < 1e-9 for g in t.cfg.sim.gravity), f"gravity {tuple(t.cfg.sim.gravity)}")
        if not self.six_dof:
            r.check("MEP linear drift, no rotation", float(np.linalg.norm(w0)) < 1e-6 and abs(float(np.linalg.norm(v0)) - c.mep.linear_velocity_mps) < 1e-4,
                    f"|v| {np.linalg.norm(v0):.4f} m/s, |w| {np.degrees(np.linalg.norm(w0)):.4f} deg/s")
        else:
            # Isaac Lab writes init_state lin_vel / ang_vel as the root COM velocity in
            # the world frame; reading back the configured vectors confirms that at runtime
            r.check("MEP 6-DoF initial velocity (linear + angular, world)",
                    float(np.linalg.norm(v0 - v_cmd)) < 1e-4 and float(np.linalg.norm(w0 - w_cmd)) < 1e-5,
                    f"v {np.round(v0, 5).tolist()} (cmd {np.round(v_cmd, 5).tolist()}) m/s, w {np.round(w0, 6).tolist()} (cmd {np.round(w_cmd, 6).tolist()}) rad/s, "
                    f"COM offset from root {np.round(com_offset, 3).tolist()} m")
        r.check("Tag layout clear of Cylinder_01 and inside the face", t.layout_ok, t.layout_detail)
        r_ee = float(t.ee_cylinder_radius)
        r.metrics["setup"]["ee_cylinder_radius_m"] = r_ee
        r.check("EE capture cylinder and Cylinder_01 marker share one radius", abs(r_ee - t.cylinder_radius) < 5e-4 and abs(r_ee - c.capture.cylinder_radius_m) < 5e-4,
                f"measured in the stage: EE cylinder r {r_ee*1000:.1f} mm, Cylinder_01 r {t.cylinder_radius*1000:.1f} mm (config {c.capture.cylinder_radius_m*1000:.0f} mm), "
                f"Cylinder_01 height {t.cylinder_height*1000:.1f} mm, ior {c.capture.cylinder_ior}")
        r.check("Intrinsics consistent with the USD camera", float(np.abs(self.k_render - self.k_usd_render).max()) < 1e-3 and float(np.abs(self.k - self.k_usd).max()) < 1e-3,
                f"max |K - K_usd| {np.abs(self.k - self.k_usd).max():.2e} (render {self.ss}x: {np.abs(self.k_render - self.k_usd_render).max():.2e})")

        a = c.approach
        # Configured search pose: nominal (planned) MEP pose, not the live one
        self.observe_pose = self.ee_goal(t.mep_nominal @ self.t_m_t, a.observe_distance_m)
        if a.start_at_observe_pose:
            ok, q, pe, ae = self.arm.solve_kinematic(self.observe_pose, self.arm.joint_pos().clone())
            r.check("Observation pose reachable (IK)", ok, f"IK err {pe*1000:.3f} mm / {math.degrees(ae):.4f} deg")
            if a.start_yaw_offset_deg != 0.0:
                # Start swung about the base axis: the camera does not see the MEP and
                # SEARCH has to find it (the observation pose stays the configured target)
                start = yaw_about_base(self.observe_pose, self.arm.base, a.start_yaw_offset_deg)
                ok, q, pe, ae = self.arm.solve_kinematic(start, q)
                r.check("Yawed start pose reachable (IK)", ok, f"{a.start_yaw_offset_deg:+.1f} deg about the base axis, IK err {pe*1000:.3f} mm / {math.degrees(ae):.4f} deg")
                print(f"[SEARCH] arm starts swung {a.start_yaw_offset_deg:+.1f} deg in azimuth from the observation pose", flush=True)
            self.task._robot.set_joint_position_target(q, joint_ids=self.arm.joint_ids)
            self.q_hold = q.clone()
            # The pose the capture pipeline starts from is also the deployment target of
            # the MRV phase, so the arm arrives in exactly the verified start state.
            self._q_deployed = q.clone()
        self.ref = self.ee_pose()
        ## MRV rendezvous phase: fold the arm and back the MRV off to its start pose
        if c.mrv.enabled:
            self.begin_mrv_approach()
        ## Moving client from t = 0 (`client.release_at_start`): released now, the MRV
        ## joins at the docking handover
        if self.moving and c.client.release_at_start and not c.docking.skip_capture:
            self.begin_client_drift()
        ## Docking phase reported next to the capture setup
        if c.docking.enabled:
            geo = self.geo
            r.metrics["setup"]["docking"] = {
                "probe_tip_in_mep_m": geo.probe_dock.pos.tolist(),
                "probe_axis_in_mep": geo.probe_dock.rot[:, 2].tolist(),
                "probe_tip_radius_m": geo.probe.tip_radius,
                "sat_dock_in_satellite_m": geo.sat_dock.pos.tolist(),
                "sat_dock_axis_in_satellite": geo.sat_dock.rot[:, 2].tolist(),
                "dock_depth_m": geo.dock_depth,
                "backstop_gap_m": geo.backstop_gap,
                "satellite_length_scale": geo.length_scale,
                "pre_dock_distance_m": float(c.docking.pre_dock_distance_m) * geo.length_scale,
                "nozzle_exit_radius_m": geo.nozzle.exit_radius,
                "nozzle_inner_radius_at_dock_m": geo.nozzle.inner_radius(geo.dock_depth),
                "satellite_velocity_mps": (np.asarray(c.docking.satellite_drift_direction) * c.docking.satellite_velocity_mps).tolist(),
                "probe_camera": ({"pos_in_mep_m": t.probe_cam_in_mep.pos.tolist(),
                                  "axis_in_mep": t.probe_cam_in_mep.rot[:, 2].tolist(),
                                  "K": self._probe_k.tolist(),
                                  "camera_to_tip_m": self._cam_to_tip}
                                 if getattr(t, "probe_cam_in_mep", None) is not None else None),
            }
            sat_v0 = self.task._satellite.data.root_com_lin_vel_w[0].cpu().numpy()
            sat_cmd = np.asarray(c.docking.satellite_drift_direction) * c.docking.satellite_velocity_mps
            r.check("[DOCK0] Satellite free-floating, no rotation",
                    float(np.linalg.norm(sat_v0 - sat_cmd)) < 1e-4
                    and float(np.linalg.norm(self.task._satellite.data.root_com_ang_vel_w[0].cpu().numpy())) < 1e-6,
                    f"satellite v {np.round(sat_v0, 5).tolist()} m/s (commanded {np.round(sat_cmd, 5).tolist()}), w = 0")
            r.check("[DOCK0] Probe fits the thruster", geo.probe.tip_radius < geo.nozzle.inner_radius(geo.dock_depth) - c.docking.min_wall_clearance_m,
                    f"probe tip radius {geo.probe.tip_radius:.3f} m, nozzle inner radius at the dock depth {geo.nozzle.inner_radius(geo.dock_depth):.3f} m")
            print(f"[INIT] PROBE_DOCK_POINT (MEP body): tip {np.round(geo.probe_dock.pos, 4).tolist()}, axis {np.round(geo.probe_dock.rot[:, 2], 4).tolist()}, tip radius {geo.probe.tip_radius:.3f} m", flush=True)
            print(f"[INIT] SAT_DOCK_POINT (satellite body): pos {np.round(geo.sat_dock.pos, 4).tolist()}, axis {np.round(geo.sat_dock.rot[:, 2], 4).tolist()}, dock_depth {geo.dock_depth:.3f} m (x{geo.length_scale:.3f} of the scale-{t.cfg.docking.reference_scale:g} values)", flush=True)
            if c.docking.skip_capture:
                if self.skip_capture_to_docking():
                    if self.moving:
                        self.goto(self.begin_moving_client())
                    else:
                        self.begin_docking()
                        self.goto(State.DOCK_TARGET_ACQUIRE)
                else:
                    self.goto(State.DOCK_FAILED, "skip_capture could not attach the MEP at the nominal grasp pose")

    ############################
    ### MRV rendezvous phase ###
    ############################

    def begin_mrv_approach(self):
        """Fold the arm, displace the MRV to its start pose and prepare the plume layer.

        Runs inside `start()`, before the first physics step, and only when
        `mrv.enabled`. It changes nothing about the MEP, the capture, the transport or
        the docking: the two legs cancel the displacement exactly, so the capture
        pipeline takes over in the configuration it was verified in.
        """
        m = self.cfg.mrv
        r = self.results
        if self.cfg.docking.enabled and self.cfg.docking.skip_capture:
            # `skip_capture` closes the arm on the MEP at its nominal grasp pose right
            # here in `start()`; there is no capture to approach, so the phase is skipped
            print("[MRV] rendezvous phase skipped: docking.skip_capture attaches the MEP at its nominal grasp pose", flush=True)
            return
        if self._q_deployed is None:
            # `approach.start_at_observe_pose: false` leaves no planned start pose for
            # the arm, so there is nothing to deploy into
            r.check("[MRV] Deployment target available", False,
                    "mrv.enabled requires approach.start_at_observe_pose: true (the deployed pose is the pipeline's start pose)")
            self.goto(State.MRV_APPROACH_FAILED, "no deployment target (approach.start_at_observe_pose is false)")
            return

        ## Folded arm -- commanded through the same joint targets the rest of the demo uses
        q_folded = torch.tensor(m.folded_joint_rad(), dtype=self._q_deployed.dtype,
                                device=self._q_deployed.device).reshape(self._q_deployed.shape)
        self._q_folded = q_folded
        # The IK that planned the deployed pose returns whatever multiple of a turn it
        # happened to converge to (the Canadarm3 joints are continuous). Interpolating to
        # that number would spin the joints for thousands of degrees, so the target is
        # re-expressed within +-180 deg of the folded pose -- the same arm configuration.
        q_raw_deg = np.degrees(self._q_deployed[0].cpu().numpy())
        self._q_deployed = wrap_joint_target(q_folded, self._q_deployed)
        q_dep_deg = np.degrees(self._q_deployed[0].cpu().numpy())
        if float(np.abs(q_dep_deg - q_raw_deg).max()) > 1.0:
            print(f"[MRV] deployed joint target unwound to the shortest turn: "
                  f"{np.round(q_raw_deg, 1).tolist()} -> {np.round(q_dep_deg, 1).tolist()} deg", flush=True)
        self.arm.set_kinematic(q_folded)
        self.task._robot.set_joint_position_target(q_folded, joint_ids=self.arm.joint_ids)
        self.q_hold = q_folded.clone()
        ee_folded = self.ee_pose()
        self.ref = ee_folded
        base = self.arm.base
        reach_folded = float(np.linalg.norm(ee_folded.pos - base.pos))
        self.arm.set_kinematic(self._q_deployed)
        reach_deployed = float(np.linalg.norm(self.ee_pose().pos - base.pos))
        self.arm.set_kinematic(q_folded)
        span = joint_span_deg(m.folded_joint_deg, np.degrees(self._q_deployed[0].cpu().numpy()).tolist())
        r.check("[MRV] Arm starts folded", reach_folded < 0.6 * reach_deployed,
                f"EE contact point {reach_folded:.2f} m from the base folded vs {reach_deployed:.2f} m deployed, "
                f"largest joint travel to deploy {span:.1f} deg",
                folded_reach_m=reach_folded, deployed_reach_m=reach_deployed)

        ## MRV start pose: nominal + start_offset (translation only, both legs cancel it)
        try:
            self.mrv = MrvTransit(self.task, self.arm)
            self.mrv.apply(m.offset())
        except Exception as e:  # a broken transit must not silently run the old pipeline
            self.mrv = None
            r.check("[MRV] Two-step approach available", False, f"the MRV could not be displaced: {e}")
            self.goto(State.MRV_APPROACH_FAILED, f"MRV placement failed: {e}")
            return
        self._leg_targets = m.leg_targets()
        d1, d2 = m.leg_distances()
        r.check("[MRV] Two translation legs are separately visible", min(d1, d2) > 0.5,
                f"leg 1 {d1:.2f} m, leg 2 {d2:.2f} m (start offset {m.start_offset_m} m, "
                f"planned {m.leg_time_s(d1):.1f} s + {m.leg_time_s(d2):.1f} s at {m.speed_mps} m/s)",
                leg1_m=d1, leg2_m=d2)
        self.roll_back_free_bodies(m.roll_back_s())
        self._mrv_log = {
            "start_offset_m": m.offset().tolist(),
            "leg1_target_offset_m": self._leg_targets[0].tolist(),
            "leg1_distance_m": d1,
            "leg2_distance_m": d2,
            "folded_joint_deg": list(m.folded_joint_deg),
            "deployed_joint_deg": np.degrees(self._q_deployed[0].cpu().numpy()).tolist(),
            "folded_reach_m": reach_folded,
            "deployed_reach_m": reach_deployed,
            "planned_duration_s": m.planned_duration_s(),
            "roll_back_s": m.roll_back_s(),
        }
        r.metrics["setup"]["mrv_approach"] = self._mrv_log
        print(f"[MRV] start offset {np.round(m.offset(), 2).tolist()} m from the nominal pose; "
              f"leg 1 {d1:.2f} m -> offset {np.round(self._leg_targets[0], 2).tolist()}, leg 2 {d2:.2f} m -> nominal. "
              f"Arm folded at {np.round(m.folded_joint_deg, 1).tolist()} deg", flush=True)

        ## Thruster plume: separate prims, visual only, never gates the pipeline
        if m.vfx_enabled:
            try:
                self.vfx = ThrusterVfx(self.task, m, hull_nominal=self.mrv.hull_nominal)
            except Exception as e:
                self.vfx = None
                print(f"[MRV-VFX] disabled: the plume layer could not be created ({e}); "
                      "the approach and the rest of the pipeline continue unchanged", flush=True)
        self._mrv_log["vfx"] = self.vfx is not None

    def roll_back_free_bodies(self, dt: float):
        """Roll the free-flying MEP (and a drifting satellite) back by `dt` seconds.

        The scene places them so the capture pipeline meets the MEP at its nominal
        rendezvous pose `mep.rendezvous_time_s` after the pipeline starts. The rendezvous
        phase adds `dt` seconds in front of it, so the bodies are put where free flight
        would have brought them from `dt` seconds earlier: at the handover they are then
        in exactly the state the verified run has at its own t = 0, and the placement
        itself (`vision_task`) is untouched.

        This is the only pose written here, it happens in `start()` before the first
        physics step (as `skip_capture` does), and the velocities are not touched -- the
        MEP keeps the `six_dof` twist it was configured with.
        """
        if dt <= 0.0:
            return
        c = self.cfg
        v_mep, w_mep = c.mep.linear_velocity_w(), c.mep.angular_velocity_w()
        nominal = self.mep_frame()
        start = back_propagate_free_body(nominal, v_mep, w_mep, self.com_in_mep, dt)
        self.mep.write_root_pose_to_sim(
            torch.tensor([[*start.pos, *start.quat]], dtype=torch.float32, device=self.mep.device)
        )
        moved = [f"MEP {float(np.linalg.norm(start.pos - nominal.pos)):.3f} m / "
                 f"{math.degrees(rotation_angle(nominal.rot, start.rot)):.2f} deg"]
        ## The satellite is translation-only in this phase (`docking.satellite_velocity_mps`,
        ## 0 by default, in which case nothing below changes its pose)
        sat = getattr(self.task, "_satellite", None)
        v_sat = np.asarray(c.docking.satellite_drift_direction, dtype=float) * float(c.docking.satellite_velocity_mps)
        if sat is not None and float(np.linalg.norm(v_sat)) > 0.0:
            s_now = Frame.from_pos_quat(sat.data.root_pos_w[0].tolist(), sat.data.root_quat_w[0].tolist())
            s_start = Frame(s_now.pos - v_sat * dt, s_now.rot)
            sat.write_root_pose_to_sim(
                torch.tensor([[*s_start.pos, *s_start.quat]], dtype=torch.float32, device=sat.device)
            )
            moved.append(f"satellite {float(np.linalg.norm(v_sat)) * dt:.3f} m")
        self.sim.forward()
        self.mep.update(1e-6)
        if sat is not None:
            sat.update(1e-6)
        self._w_frame_ref = (self.sim_time, self.mep_frame())  # the 6-DoF frame check restarts here
        print(f"[MRV] free bodies rolled back {dt:.1f} s (the planned approach + deployment time) so the capture "
              f"starts from the verified rendezvous state: {', '.join(moved)}", flush=True)

    def start_leg(self, index: int):
        """Begin translation leg 1 or 2 towards its target offset."""
        m = self.cfg.mrv
        target = self._leg_targets[index - 1]
        self._leg = TransitLeg(self.mrv.offset, target, m.speed_mps, m.accel_mps2)
        print(f"[MRV] leg {index}: {self._leg.length:.2f} m along {np.round(self._leg.dir, 3).tolist()} "
              f"at up to {m.speed_mps:.2f} m/s (planned {m.leg_time_s(self._leg.length):.1f} s)", flush=True)

    def step_mrv_leg(self, index: int, reached: State):
        """One control step of a translation leg (the arm holds its folded joints)."""
        m = self.cfg.mrv
        self.hold()
        if self._leg is None:
            self.start_leg(index)
        offset = self._leg.step(self.dt)
        self.mrv.apply(offset)
        self.update_thruster_vfx()
        err = self.mrv.error_m(self._leg.target)
        if self._leg.done(1e-6) and err <= m.position_tolerance_m:
            print(f"[MRV] leg {index} complete: offset {np.round(self.mrv.offset, 3).tolist()} m, "
                  f"measured base error {err*1000:.1f} mm (tolerance {m.position_tolerance_m*1000:.0f} mm), "
                  f"t={self.sim_time:.1f} s", flush=True)
            self._mrv_log[f"leg{index}_end_error_mm"] = err * 1000.0
            self._mrv_log[f"leg{index}_end_time_s"] = self.sim_time
            self._leg = None
            self.goto(reached)
        elif self.state_time > m.stage_timeout_s:
            self.goto(State.MRV_APPROACH_FAILED,
                      f"translation leg {index} did not reach its target within {m.stage_timeout_s:.0f} s "
                      f"(remaining {self._leg.remaining:.3f} m, base error {err*1000:.1f} mm)")

    def step_mrv_reached(self, nxt: State):
        """Hold still between the two legs (the plume is off: the MRV is not thrusting)."""
        self.hold()
        self.update_thruster_vfx()
        if self.state_time >= self.cfg.mrv.settle_time_s:
            self.goto(nxt)

    def step_arm_deploy(self):
        """Unfold the arm into the pose the capture pipeline starts from.

        Joint-space interpolation with an ease-in/ease-out profile, so no joint is
        stepped: q(t) = q_folded + smoothstep(t / duration) * (q_deployed - q_folded).
        """
        m = self.cfg.mrv
        self.update_thruster_vfx()  # off: the MRV is not translating any more
        u = self.state_time / max(1e-6, m.deploy_duration_s)
        self._deploy_alpha = smoothstep(u)
        q = joint_lerp(self._q_folded, self._q_deployed, self._deploy_alpha)
        self.task._robot.set_joint_position_target(q, joint_ids=self.arm.joint_ids)
        self.q_hold = q.clone()
        self.set_velocity_feedforward(None)
        settled = self.state_time >= m.deploy_duration_s and (
            math.degrees(float(torch.max(torch.abs(self.arm.joint_pos() - self._q_deployed)))) < m.deploy_tolerance_deg
        )
        if settled or self.state_time >= m.deploy_duration_s + m.deploy_settle_s:
            q_err = float(torch.max(torch.abs(self.arm.joint_pos() - self._q_deployed)))
            pe, ae = pose_errors(self.ee_pose(), self._deployed_ee_pose())
            self.results.check("[MRV] Arm deployed to the pipeline start pose", math.degrees(q_err) < 1.0,
                               f"max joint error {math.degrees(q_err):.3f} deg, EE {pe*1000:.1f} mm / {ae:.3f} deg from the planned start pose",
                               max_joint_error_deg=math.degrees(q_err))
            self._mrv_log["deploy_end_time_s"] = self.sim_time
            self._mrv_log["deploy_joint_error_deg"] = math.degrees(q_err)
            self.enter_capture_pipeline()
        elif self.state_time > m.stage_timeout_s:
            self.goto(State.MRV_APPROACH_FAILED, f"the arm was not deployed within {m.stage_timeout_s:.0f} s")

    def _deployed_ee_pose(self) -> Frame:
        """EE contact pose of the planned deployed joint configuration (no teleporting)."""
        return self.observe_pose if self.cfg.approach.start_yaw_offset_deg == 0.0 else yaw_about_base(
            self.observe_pose, self.arm.base, self.cfg.approach.start_yaw_offset_deg
        )

    def enter_capture_pipeline(self):
        """Hand over to the existing capture pipeline, which starts its own clock here."""
        err = self.mrv.error_m(np.zeros(3)) if self.mrv is not None else 0.0
        self.results.check("[MRV] MRV arrived at its nominal pose", err <= self.cfg.mrv.position_tolerance_m,
                           f"articulation root {err*1000:.1f} mm from the nominal (verified) pose "
                           f"(tolerance {self.cfg.mrv.position_tolerance_m*1000:.0f} mm)", mrv_arrival_error_mm=err * 1000.0)
        self.thruster_vfx_off()
        self._capture_t0 = self.sim_time
        self._mrv_log["capture_start_time_s"] = self._capture_t0
        self.results.metrics["mrv_approach"] = dict(self._mrv_log)
        gt = self.gt_cylinder()
        nominal = (self.task.mep_nominal @ self.t_m_y).pos
        print(f"[MRV] approach complete at t={self.sim_time:.1f} s -- handing over to the capture pipeline. "
              f"MEP Cylinder_01 is {float(np.linalg.norm(gt.pos - nominal))*1000:.0f} mm from its nominal rendezvous pose "
              f"(it kept drifting and turning throughout)", flush=True)
        self.ref = self.ee_pose()
        self.goto(State.SEARCH)

    def update_thruster_vfx(self):
        """Plume on only while a leg is running, pointing opposite the commanded motion."""
        if self.vfx is None:
            return
        direction = None
        if self.state in MRV_MOTION and self._leg is not None and self._leg.v > 1e-4:
            direction = -self._leg.dir  # exhaust leaves opposite the direction of travel
        try:
            self.vfx.update(direction, self.mrv.offset if self.mrv is not None else np.zeros(3))
        except Exception as e:
            print(f"[MRV-VFX] update failed ({e}); the plume is switched off and the pipeline continues", flush=True)
            self.vfx = None

    def thruster_vfx_off(self):
        if self.vfx is None:
            return
        try:
            self.vfx.off()
        except Exception:
            pass

    def capture_clock(self) -> float:
        """Simulated time since the capture pipeline took over (0 without the MRV phase)."""
        return self.sim_time - self._capture_t0

    def idle(self):
        """GUI after a successful run: keep simulating (MEP held) until the window closes."""
        sim, scene = self.sim, self.scene
        self.q_hold = self.arm.joint_pos().clone()
        print("[DEMO] Scenario finished -- the simulation keeps running with the MEP held. Close the Isaac Sim window to exit.", flush=True)
        self.assist_thrust_off()
        n = 0
        with torch.no_grad():
            while self.sim_app.is_running():
                if not sim.is_playing():
                    sim.render()
                    continue
                self.hold()
                # Astrobee clock only (`sim_time` stays at the end of the scenario)
                t_obs = self.sim_time + n * self.dt
                if self.astrobee is not None:
                    self.astrobee.step(t_obs, self.state.value)
                scene.write_data_to_sim()
                sim.step(render=False)
                n += 1
                if n % self.render_interval == 0:
                    sim.render()
                scene.update(dt=self.dt)
                if n % self.render_interval == 0 and self.astrobee is not None:
                    self.astrobee.after_render(t_obs + self.dt)

    ## Astrobee docking damping assist (`astrobee_assist.py`, `astrobee.assist.enabled`)
    def assist_context(self):
        """MEP data for the assist (None when it is off). Also in failure / terminal
        states, so an engaged assist sees them and lets go of the probe."""
        if self.astrobee is None or self.astrobee.assist is None:
            return None
        from .astrobee_assist import MepContext, grip_point_in_mep

        pr = self.geo.probe
        grip_in_mep = grip_point_in_mep(pr.tip, pr.direction, pr.length, self.cfg.astrobee.assist.grip_fraction_from_root)
        mep = self.mep_frame()
        grip = mep.point(grip_in_mep)
        v_ref, w_ref = self._ee_ref_twist
        return MepContext(mep=mep, grip_in_mep=grip_in_mep, axis_in_mep=np.asarray(pr.direction, dtype=float),
                          grip_velocity=self.gt_mep_velocity_at(grip)[0],
                          reference_velocity=v_ref + np.cross(w_ref, grip - self.arm.tool_pose().pos))

    def apply_assist_thrust(self):
        """Astrobee thrust on the MEP at the grip point: force at the COM + r x F torque."""
        f = None if self.astrobee is None else self.astrobee.assist_force_w
        if f is None:
            return self.assist_thrust_off()
        com = self.mep.data.root_com_pos_w[0].cpu().numpy().astype(float)
        self.apply_wrench("mep", f, np.cross(self.astrobee.assist_grip_w - com, f))
        self._assist_force = np.asarray(f, dtype=float).copy()

    def assist_thrust_off(self):
        """Remove the assist wrench, unless another controller has set the MEP wrench since."""
        if self._assist_force is None:
            return
        if np.allclose(self._mv_force.get("mep", np.zeros(3)), self._assist_force):
            self.apply_wrench("mep", np.zeros(3), np.zeros(3))
        self._assist_force = None

    def close_astrobee(self):
        """End of the run (after `idle()` in the GUI): stop the Astrobee camera feed."""
        if self.astrobee is not None:
            self.astrobee.complete()
            self.astrobee.close()
            self.astrobee = None

    def step(self):
        """One control step (called before every physics step)."""
        c = self.cfg
        a = c.approach
        self.state_time += self.dt
        s = self.state

        if self.ros is not None:
            self.ros.poll()
            if self.ros.abort_requested and s not in TERMINAL:
                self.goto(State.ABORTED, "ROS cmd/abort")
                s = self.state

        if self._mv_active:
            self.moving_measure()
            ## The arm must not hit the client / MEP once it has let go (checked before
            ## the generic 200 N contact limit of `check_physics`, with its own limit)
            if s in (State.ARM_RETREAT, State.MRV_SEPARATION):
                f = self._mvm.get("contact_n", 0.0)
                self._mv["max_separation_contact_n"] = max(float(self._mv.get("max_separation_contact_n", 0.0)), f)
                if f > self.cfg.separation.max_contact_force_n:
                    self.goto(State.SEPARATION_COLLISION, f"arm contact force {f:.1f} N > {self.cfg.separation.max_contact_force_n:.0f} N in {s.value}")
                    s = self.state

        if s not in TERMINAL:
            err = self.check_physics()
            if err is not None:
                self.goto(State.PHYSICS_ERROR, err)
                s = self.state

        ## Tag loss in the tracking states: keep the last valid pose, stop after timeout
        if s in TRACKING | {State.PREDICTING} and self.estimate_age() > c.vision.tag_loss_timeout_s:
            self.goto(State.TAG_LOST, f"no valid constellation pose for {self.estimate_age():.2f} s (last: {self.last_vision.reason if self.last_vision else 'none'})")
            s = self.state
        if self.scenario == "dynamic" and s in TRACKING and self.capture_clock() > c.test.dynamic_timeout_sec:
            self.goto(State.APPROACH_TIMEOUT, f"no capture within {c.test.dynamic_timeout_sec:.0f} s")
            s = self.state

        if self.six_dof and self._w_frame_result is None and self._w_frame_ref is not None and self.sim_time - self._w_frame_ref[0] >= 1.0:
            self.check_angular_velocity_frame()

        v_est = self.predictor.velocity()
        if s == State.INIT:
            self.hold()
            if self.state_time >= 0.5:
                self.goto(State.MRV_MOVE_STEP_1 if self.mrv is not None else State.SEARCH)
        elif s == State.MRV_MOVE_STEP_1:
            self.step_mrv_leg(1, State.MRV_STEP_1_REACHED)
        elif s == State.MRV_STEP_1_REACHED:
            self.step_mrv_reached(State.MRV_MOVE_STEP_2)
        elif s == State.MRV_MOVE_STEP_2:
            self.step_mrv_leg(2, State.MRV_STEP_2_REACHED)
        elif s == State.MRV_STEP_2_REACHED:
            self.step_mrv_reached(State.ARM_DEPLOY)
        elif s == State.ARM_DEPLOY:
            self.step_arm_deploy()
        elif s == State.SEARCH:
            if a.start_at_observe_pose and a.start_yaw_offset_deg == 0.0:
                self.hold()
            else:
                # swing towards the configured observation pose until tags are seen
                self.q_hold = None
                self.track(self.observe_pose, a.search_speed_mps, None,
                           max_step=a.search_max_joint_step_rad, lag_limit=a.search_lag_m)
            # A swung start: the first tags appear at the edge of the image while the arm is
            # still mid-swing. Ending the search there stops the arm with only 1-2 of the 4
            # tags in view, so no pose can be estimated (measured: tags [0, 1] seen 600 mm off
            # the axis). The search therefore only ends once the arm has reached the
            # observation pose, where the whole constellation is in view.
            arrived = True
            if a.start_at_observe_pose and a.start_yaw_offset_deg != 0.0:
                pe, ae = pose_errors(self.ee_pose(), self.observe_pose)
                arrived = pe < 0.03 and ae < 2.0
            if self.last_vision is not None and self.last_vision.detections and arrived:
                self.goto(State.TAG_DETECTED, f"tags {sorted(self.last_vision.detections)}")
            elif self.state_time > a.stage_timeout_s:
                self.goto(State.TAG_LOST, "no AprilTag detected while searching")
        elif s == State.TAG_DETECTED:
            self.hold()
            if self.last_valid is not None:
                self.goto(State.POSE_ESTIMATED, f"{self.last_valid.inliers} PnP inliers, RMS {self.last_valid.reproj_rms:.3f} px")
            elif self.state_time > 3.0:
                self.goto(State.POSE_INVALID, f"no valid constellation pose ({self.last_vision.reason if self.last_vision else 'no image'})")
        elif s == State.POSE_ESTIMATED:
            self.hold()
            pe, ae = pose_errors(self.gt_cylinder(), self.last_valid.t_w_y)
            print(f"[VISION] first Cylinder_01 estimate: pos {np.round(self.last_valid.t_w_y.pos, 4).tolist()} (GT error {pe*1000:.2f} mm / {ae:.3f} deg)", flush=True)
            self.goto(State.PREDICTING)
        elif s == State.PREDICTING:
            self.hold()
            if v_est is not None and self.ros is not None and not self.ros.start_received:
                if not self._start_wait_logged:
                    self._start_wait_logged = True
                    print("[ROS] prediction ready, waiting for cmd/start (the arm holds the observation pose)", flush=True)
            elif v_est is not None:
                print(f"[PREDICT] velocity estimate {np.round(v_est, 5).tolist()} m/s from {len(self.predictor.history)} samples", flush=True)
                self.ref = self.ee_pose()
                self.q_hold = None
                self.goto(State.STATIC_MEASURE if self.scenario == "static" else State.APPROACHING)
            elif self.state_time > self.cfg.prediction.velocity_window_sec + 3.0:
                self.goto(State.PREDICTION_INVALID, f"no plausible velocity from {len(self.predictor.history)} samples")
        elif s == State.APPROACHING:
            self.step_approaching(v_est)
        elif s == State.SLOW_APPROACH:
            self.step_slow_approach(v_est)
        elif s == State.CAPTURE_ATTEMPT:
            self.step_capture_attempt(v_est)
        elif s == State.CAPTURED:
            self.hold()
            self.goto(State.HOLDING)
        elif s == State.HOLDING:
            self.step_holding()
        elif s == State.RETREAT:
            self.step_retreat()
        elif s == State.STATIC_MEASURE:
            self.step_static(v_est)
        elif s in MOVING_STATES:
            self.step_moving(s)
        elif s in DOCKING_STATES:
            m = self.dock_metrics()
            self._dock_m_latest = m  # telemetry only (publish_ros)
            if self._dock_started is None:
                self._dock_started = self.sim_time
            elif (self.sim_time - self._dock_started > c.docking.phase_timeout_s
                  and s not in (State.DOCKED, State.DOCK_HOLDING)):
                return self.goto(State.DOCK_FAILED,
                                 f"the docking phase did not finish within {c.docking.phase_timeout_s:.0f} s "
                                 f"(state {s.value}, remaining {m['geometry_distance']:.3f} m, lateral {m['lateral']*1000:.1f} mm)")
            # "At rest" means at rest RELATIVE to the client: with a drifting client the
            # tip moves in the world while it holds its pose on the docking axis
            tip = self.probe_world().pos.copy()
            self._tip_history.append((self.sim_time, self.sat_frame().inv().point(tip) if self._mv_active else tip))
            self._tip_history = [x for x in self._tip_history if x[0] >= self.sim_time - 3.0 * self.cfg.docking.settle_window_s]
            cm = self.control_measure(m)  # both modes: metrics + control log
            if s == State.DOCK_TARGET_ACQUIRE:
                self.step_dock_target_acquire(m)
            elif s == State.PRE_DOCK_APPROACH:
                self.step_pre_dock(m)
            elif s == State.POSITION_ATTITUDE_ALIGN:
                self.step_position_attitude_align(m, cm)
            elif self.coupled and s == State.ALIGNMENT_CHECK:
                self.step_coupled_alignment_check(m, cm)
            elif self.coupled and s in (State.Z_APPROACH, State.FINAL_INSERTION):
                self.step_coupled_approach(m, cm, final=s == State.FINAL_INSERTION)
            elif s == State.XY_ALIGN:
                self.step_xy_align(m)
            elif s == State.ORIENTATION_ALIGN:
                self.step_orientation_align(m)
            elif s == State.ALIGNMENT_CHECK:
                self.step_alignment_check(m)
            elif s == State.Z_APPROACH:
                self.step_z_approach(m, final=False)
            elif s == State.FINAL_INSERTION:
                self.step_z_approach(m, final=True)
            elif s == State.DOCK_READY:
                self.step_dock_ready(m)
            elif s == State.DOCKED:
                self.step_docked(m)
            else:
                self.step_dock_holding(m)
            if self.sim_time >= self._next_dock_log_t:
                self._next_dock_log_t = self.sim_time + 1.0 / self.cfg.logging.rate_hz - 1e-9
                self.log_dock_row(m)
                self.log_control_row(m, cm)
                if not self._mv_active:
                    self.update_motion_hud(m)  # the moving client refreshes it in `moving_after_step`
                if self.cfg.logging.debug_draw:
                    self.update_dock_visuals(m)
            if self.sim_time >= self._next_status_t:
                self._next_status_t = self.sim_time + 2.0
                self.dock_status(m)
            if self.sim_time >= self._next_probe_rgb_t and self.cfg.probe_camera.rgb_every_sec > 0.0:
                self._next_probe_rgb_t = self.sim_time + self.cfg.probe_camera.rgb_every_sec
                self.save_probe_rgb()
        else:  # SUCCESS / failures: hold the joints
            if s == State.SUCCESS and self._mv_active:
                # The MRV keeps departing (`moving_after_step`); stop the run manually (Ctrl+C).
                # The arm finishes returning to its start pose if it has not yet.
                self.arm_stow_step()
                return
            self.hold()
            if self.state_time >= (0.5 if s == State.SUCCESS else 1.0):
                self.done = True

    def skip_capture_to_docking(self) -> bool:
        """`docking.skip_capture`: put the MEP at its nominal pose, close the arm on it
        kinematically and attach, so a docking run does not re-run the 30 s capture.

        Nothing is teleported once the simulation is running: this happens in `start()`,
        before the first physics step, and only when the docking phase is enabled.
        """
        mep_nominal = self.task.mep_nominal
        drift = self.cfg.mep.linear_velocity_w()
        root = torch.zeros((1, 13), device=self.mep.device)
        root[0, :3] = torch.tensor(mep_nominal.pos, device=self.mep.device)
        root[0, 3:7] = torch.tensor(mep_nominal.quat, device=self.mep.device)
        root[0, 7:10] = torch.tensor(drift, device=self.mep.device)
        self.mep.write_root_state_to_sim(root)
        self.sim.forward()
        self.mep.update(1e-6)
        goal = self.mep_frame() @ self.geo.ee_contact_to_mep.inv()
        ok, q, pe, ae = self.arm.solve_kinematic(goal, self.arm.joint_pos().clone())
        self.results.check("[DOCK0] Grasp pose reachable (skip_capture)", ok,
                           f"IK err {pe*1000:.3f} mm / {math.degrees(ae):.4f} deg at the nominal MEP grasp face")
        if not ok:
            return False
        self.task._robot.set_joint_position_target(q, joint_ids=self.arm.joint_ids)
        self.q_hold = q.clone()
        self.capture.attach(0)
        self.results.check("[DOCK0] MEP attached (skip_capture)", self.capture.is_attached(0),
                           "capture FixedJoint created at the nominal grasp pose")
        if not self.capture.is_attached(0):
            return False
        self._rel_at_capture = self.ee_pose().inv() @ self.mep_frame()
        self._mep_at_capture = self.mep_frame()
        self._normal_at_capture = self.mep_grasp_normal()
        print(f"[DOCK] skip_capture: MEP attached at the nominal grasp pose (IK err {pe*1000:.2f} mm), "
              f"capture phase skipped", flush=True)
        return True

    def mep_grasp_normal(self) -> np.ndarray:
        return (self.mep_frame() @ self.geo.mep_grasp).rot[:, 2]

    def after_capture_state(self) -> State:
        """Where the capture phase hands over: SUCCESS, or the docking phase."""
        if not self.cfg.docking.enabled:
            return State.SUCCESS
        if self.moving:
            # Moving client: rendezvous first, the docking starts from RENDEZVOUS
            return self.begin_moving_client()
        self.begin_docking()
        return State.DOCK_TARGET_ACQUIRE

    def check_angular_velocity_frame(self):
        """six_dof, ~1 s after the start (GT, judgement only): is the configured /
        reported angular velocity a WORLD-frame vector? Compares Log(R1 R0^T) / dt
        (world) and Log(R0^T R1) / dt (body) with the commanded w."""
        t0, f0 = self._w_frame_ref
        f1 = self.mep_frame()
        dt = self.sim_time - t0
        w_cmd = self.cfg.mep.angular_velocity_w()
        w_world = so3_log(f1.rot @ f0.rot.T) / dt
        w_body = so3_log(f0.rot.T @ f1.rot) / dt
        w_meas = self.mep.data.root_com_ang_vel_w[0].cpu().numpy()
        err_w, err_b = float(np.linalg.norm(w_world - w_cmd)), float(np.linalg.norm(w_body - w_cmd))
        tol = 0.05 * float(np.linalg.norm(w_cmd)) + 1e-5
        distinct = float(np.linalg.norm(w_world - w_body)) > 2.0 * tol
        # Frame question = DIRECTION: which of the two readings of the commanded vector
        # (world / body) is the one the body actually rotates about. The rate is judged
        # separately (`rate_ratio`): PhysX may report a different |w| than the pose change.
        n_cmd = float(np.linalg.norm(w_cmd))
        n_world = float(np.linalg.norm(w_world))
        cos = float(w_world @ w_cmd) / max(1e-12, n_world * n_cmd)
        angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
        ratio = n_world / max(1e-12, n_cmd)
        self._w_frame_result = {
            "dt_s": dt, "w_cmd": w_cmd.tolist(), "w_from_pose_world": w_world.tolist(), "w_from_pose_body": w_body.tolist(),
            "w_root_com_ang_vel_w": w_meas.tolist(), "err_world": err_w, "err_body": err_b, "tolerance": tol,
            "world_and_body_distinguishable": distinct,
            "direction_error_deg": angle_deg, "rate_ratio_pose_over_cmd": ratio,
            "rate_ratio_pose_over_root_com": n_world / max(1e-12, float(np.linalg.norm(w_meas))),
            "pass": angle_deg < 5.0 and (err_w < err_b or not distinct),
        }
        note = "" if abs(ratio - 1.0) <= 0.05 else f"; WARNING: rotation rate from poses is {ratio:.3f} x commanded (root_com_ang_vel_w {float(np.linalg.norm(w_meas)) / n_cmd:.3f} x)"
        print(f"[FRAME] angular velocity over {dt:.2f} s: from poses (world) {np.round(w_world, 5).tolist()}, (body) {np.round(w_body, 5).tolist()}, "
              f"root_com_ang_vel_w {np.round(w_meas, 5).tolist()}, commanded {np.round(w_cmd, 5).tolist()} rad/s -> "
              f"{'WORLD frame confirmed' if self._w_frame_result['pass'] else 'MISMATCH'} (direction error {angle_deg:.2f} deg){note}"
              f"{'' if distinct else ' (world/body not distinguishable for this start orientation)'}", flush=True)

    ## Stage A/B: far approach, then track the predicted target at the approach standoff
    def step_approaching(self, v_est):
        a = self.cfg.approach
        pred = self.predicted()
        if pred is None:
            return self.goto(State.PREDICTION_INVALID, "prediction unavailable while approaching")
        self.standoff = a.approach_standoff_m
        self.goal = self.ee_goal(pred @ self.t_y_t, self.standoff)
        v_ff, w_ff = self.feedforward()
        self.track(self.goal, a.speed_far_mps, v_ff, w_ff=w_ff)
        if self._settled(a.settle_lateral_m, a.settle_angle_deg, 0.02, 0.5):
            m = self.est_capture_metrics()
            print(f"[APPROACH] Stage B: tracking the predicted Cylinder_01 at {self.standoff:.2f} m (lateral {m['lateral']*1000:.1f} mm, angle {m['angle']:.2f} deg)", flush=True)
            self.goto(State.SLOW_APPROACH)
        elif self.state_time > a.stage_timeout_s:
            self.goto(State.APPROACH_TIMEOUT, "approach standoff not reached")

    ## Stage C/D: close the standoff slowly while staying on the predicted axis
    def step_slow_approach(self, v_est):
        a = self.cfg.approach
        c = self.cfg.capture
        pred = self.predicted()
        if pred is None:
            return self.goto(State.PREDICTION_INVALID, "prediction unavailable in slow approach")
        # Gate on the EE pose w.r.t. the current Cylinder_01 estimate (as the capture
        # conditions are): on the axis, aligned, and not lagging the commanded standoff
        m = self.est_capture_metrics()
        gate = m["lateral"] < a.gate_lateral_m and m["angle"] < a.gate_angle_deg and abs(m["gap"] - self.standoff) < 0.02
        # Relative velocity guard: do not close in while the EE moves too fast relative to the MEP
        if not math.isnan(m["rel_vel"]) and m["rel_vel"] > 0.8 * c.max_relative_velocity_mps:
            gate = False
        if gate and self.standoff > a.final_gap_m:
            span = max(1e-6, a.near_distance_m - a.capture_range_m)
            f = min(1.0, max(0.0, (self.standoff - a.capture_range_m) / span))
            v = a.speed_capture_range_mps + (a.speed_near_mps - a.speed_capture_range_mps) * f
            if self.standoff > a.near_distance_m:
                v = a.speed_near_mps
            self.standoff = max(a.final_gap_m, self.standoff - v * self.dt)
        if self.standoff <= a.capture_range_m and not self._capture_range_logged:
            self._capture_range_logged = True
            print(f"[APPROACH] Stage D: capture range entered (standoff {self.standoff:.3f} m, distance to Cylinder_01 {m['distance']:.3f} m)", flush=True)
        self.goal = self.ee_goal(pred @ self.t_y_t, self.standoff)
        v_ff, w_ff = self.feedforward()
        self.track(self.goal, a.speed_near_mps, v_ff, w_ff=w_ff)
        # Stage D -> E once the EE holds the final gap on the Cylinder_01 axis. Judged
        # against the current estimate (as the capture conditions are), not against the
        # 0.3 s-ahead goal: the stiff, overdamped joint drives trail their targets by
        # ~0.6 s, so the EE stays a few mm behind the leading goal while on the axis.
        on_axis = (m["lateral"] < a.gate_lateral_m and m["angle"] < a.gate_angle_deg
                   and abs(m["gap"] - a.final_gap_m) < 0.005)
        if self.standoff <= a.final_gap_m + 1e-9 and on_axis:
            self._settled_since = self._settled_since if self._settled_since is not None else self.sim_time
        else:
            self._settled_since = None
        if self._settled_since is not None and self.sim_time - self._settled_since >= 0.3:
            self.goto(State.CAPTURE_ATTEMPT)
        elif self.state_time > a.stage_timeout_s:
            self.goto(State.APPROACH_TIMEOUT, f"slow approach stalled at standoff {self.standoff:.3f} m")

    ## Stage E/F: check every condition, then the existing magnet attach (FixedJoint)
    def step_capture_attempt(self, v_est):
        a = self.cfg.approach
        pred = self.predicted()
        if pred is not None:
            self.goal = self.ee_goal(pred @ self.t_y_t, a.final_gap_m)
            v_ff, w_ff = self.feedforward()
            self.track(self.goal, a.speed_capture_range_mps, v_ff, w_ff=w_ff)
        if self.ros is not None and not self.ros.capture_enabled:
            if not self._capture_wait_logged:
                self._capture_wait_logged = True
                print("[ROS] capture disabled (cmd/capture_enable false): tracking without attaching", flush=True)
            return
        ok, m, failed = self.capture_conditions()
        if not ok:
            if self.state_time > 5.0:
                self.goto(State.CAPTURE_FAILED, "; ".join(failed))
            return
        v_gt, _ = self.gt_mep_velocity_at(self.ee_pose().pos)
        m_gt = self.gt_capture_metrics()
        print(f"[CAPTURE] ATTEMPT  t={self.sim_time:.2f} s | vision: distance {m['distance']:.4f} m, gap {m['gap']*1000:.1f} mm, lateral {m['lateral']*1000:.2f} mm, "
              f"angle {m['angle']:.3f} deg, rel. velocity {m['rel_vel']:.4f} m/s, rel. angular velocity {m['rel_ang_vel']:.4f} rad/s | GT: distance {m_gt['distance']:.4f} m, "
              f"lateral {m_gt['lateral']*1000:.2f} mm, angle {m_gt['angle']:.3f} deg, rel. velocity {m_gt['rel_vel']:.4f} m/s, "
              f"rel. angular velocity {m_gt['rel_ang_vel']:.4f} rad/s", flush=True)
        self.capture.attach(0)  # existing magnet-style capture -> UsdPhysics.FixedJoint
        if self.capture.is_attached(0):
            print(f"[CAPTURE] CAPTURED  t={self.sim_time:.2f} s  (FixedJoint {self.capture._joint_prim_paths[0]})", flush=True)
            self._capture_info = {
                # Measured from the start of the capture pipeline (= sim_time unless the
                # MRV rendezvous phase ran ahead of it), which is what
                # `test.dynamic_timeout_sec` has always meant
                "time_s": self.capture_clock(),
                "sim_time_s": self.sim_time,
                "vision": m,
                "gt": m_gt,
                "ee_speed_mps": float(np.linalg.norm(self.ee_velocity())),
                "mep_speed_mps": float(np.linalg.norm(v_gt)),
            }
            self._rel_at_capture = self.ee_pose().inv() @ self.mep_frame()
            self._mep_at_capture = self.mep_frame()
            self._normal_at_capture = (self.estimate_now() @ self.t_y_t).rot[:, 2]
            self.q_hold = self.arm.joint_pos().clone()
            self.goto(State.CAPTURED)
        else:
            self.goto(State.CAPTURE_FAILED, "FixedJoint was not created")

    def relative_drift(self) -> Tuple[float, float]:
        now = self.ee_pose().inv() @ self.mep_frame()
        pe, ae = pose_errors(self._rel_at_capture, now)
        return pe, ae

    def step_holding(self):
        self.hold()
        h = self._hold
        pe, ae = self.relative_drift()
        h["max_drift_mm"] = max(h.get("max_drift_mm", 0.0), pe * 1000.0)
        h["max_drift_deg"] = max(h.get("max_drift_deg", 0.0), ae)
        v = float(torch.norm(self.mep.data.root_com_lin_vel_w[0]))
        w = math.degrees(float(torch.norm(self.mep.data.root_com_ang_vel_w[0])))
        h["max_mep_speed"] = max(h.get("max_mep_speed", 0.0), v)
        if self.state_time >= 0.5 * self.cfg.test.holding_duration_sec:
            h["max_mep_speed_second_half"] = max(h.get("max_mep_speed_second_half", 0.0), v)
        h["max_mep_rate_deg_s"] = max(h.get("max_mep_rate_deg_s", 0.0), w)
        ee = self.ee_pose()
        h.setdefault("ee_start", ee.pos.copy())
        h["max_ee_motion_mm"] = max(h.get("max_ee_motion_mm", 0.0), 1000.0 * float(np.linalg.norm(ee.pos - h["ee_start"])))
        if self.state_time >= self.cfg.test.holding_duration_sec:
            h["duration_s"] = self.state_time
            h["final_mep_speed"] = v
            h["joint_valid"] = self.capture.is_attached(0)
            h["mep_displacement_mm"] = 1000.0 * float(np.linalg.norm(self.mep_frame().pos - self._mep_at_capture.pos))
            print(f"[HOLD] {self.state_time:.1f} s held: max MEP-EE drift {h['max_drift_mm']:.3f} mm / {h['max_drift_deg']:.4f} deg, "
                  f"MEP speed now {v*1000:.2f} mm/s, FixedJoint valid {h['joint_valid']}", flush=True)
            if self.cfg.test.retreat_distance_m > 0.0:
                self.goto(State.RETREAT)
            else:
                self.goto(self.after_capture_state())

    def step_retreat(self):
        """Slowly back the EE off along the face normal: the MEP must come along."""
        c = self.cfg.test
        if self._retreat is None:
            start = self.ee_pose()
            self._retreat = {"start": start, "mep_start": self.mep_frame().pos.copy(), "dist": 0.0, "max_drift_mm": 0.0, "max_drift_deg": 0.0}
            self.ref = start
            self.q_hold = None
        rt = self._retreat
        rt["dist"] = min(c.retreat_distance_m, rt["dist"] + c.retreat_speed_mps * self.dt)
        goal = Frame(rt["start"].pos + rt["dist"] * self._normal_at_capture, rt["start"].rot)
        self.goal = goal
        self.ref = goal
        q = self.arm.ik_joint_target(goal, 0.01)
        self.task._robot.set_joint_position_target(q, joint_ids=self.arm.joint_ids)
        self.set_velocity_feedforward(None)
        pe, ae = self.relative_drift()
        rt["max_drift_mm"] = max(rt["max_drift_mm"], pe * 1000.0)
        rt["max_drift_deg"] = max(rt["max_drift_deg"], ae)
        if rt["dist"] >= c.retreat_distance_m and self.state_time > c.retreat_distance_m / c.retreat_speed_mps + 3.0:
            ee_moved = float((self.ee_pose().pos - rt["start"].pos) @ self._normal_at_capture)
            mep_moved = float((self.mep_frame().pos - rt["mep_start"]) @ self._normal_at_capture)
            rt["ee_moved_m"], rt["mep_moved_m"] = ee_moved, mep_moved
            print(f"[RETREAT] EE moved {ee_moved*1000:.1f} mm, MEP moved {mep_moved*1000:.1f} mm along the face normal, "
                  f"max MEP-EE drift {rt['max_drift_mm']:.3f} mm / {rt['max_drift_deg']:.4f} deg", flush=True)
            self.q_hold = None
            self.goto(self.after_capture_state())

    def step_static(self, v_est):
        """Test 1: stationary MEP observed from each configured standoff."""
        c = self.cfg
        standoffs = c.test.static_standoffs_m
        if self._static_idx >= len(standoffs):
            self.hold()
            self.goto(State.SUCCESS)
            return
        y = self.estimate_now()
        s = standoffs[self._static_idx]
        if self.goal is None or self._settled_since is None:
            self.goal = self.ee_goal(y @ self.t_y_t, s)
        self.track(self.goal, c.approach.speed_far_mps, None)
        # Sample only once the whole estimation window was observed from this standoff
        if not self._settled(0.005, 0.3, 0.005, max(c.prediction.velocity_window_sec, c.prediction.orientation_window_sec) + 0.2):
            if self.state_time > c.approach.stage_timeout_s:
                self.goto(State.APPROACH_TIMEOUT, f"static pose at {s} m not reached")
            return
        lv = self.last_vision
        if lv is not None and lv.valid and lv.t > self._static_last_t:
            self._static_last_t = lv.t
            gt = self.gt_cylinder()
            pe, ae = pose_errors(gt, self.estimate_now())  # system estimate (window-fused)
            rpe, rae = pose_errors(gt, lv.t_w_y)  # this image alone
            self._static_samples.append((pe * 1000.0, ae, lv.reproj_rms, rpe * 1000.0, rae))
        if len(self._static_samples) >= c.test.static_samples:
            arr = np.array(self._static_samples)
            dist = float(np.linalg.norm(self.cam_pose().pos - self.gt_cylinder().pos))
            stat = {
                "standoff_m": s,
                "camera_to_cylinder01_m": dist,
                "samples": len(arr),
                "position_error_mm_mean": float(arr[:, 0].mean()),
                "position_error_mm_max": float(arr[:, 0].max()),
                "angle_error_deg_mean": float(arr[:, 1].mean()),
                "angle_error_deg_max": float(arr[:, 1].max()),
                "reproj_rms_px_mean": float(arr[:, 2].mean()),
                "single_image_position_error_mm_mean": float(arr[:, 3].mean()),
                "single_image_position_error_mm_max": float(arr[:, 3].max()),
                "single_image_angle_error_deg_mean": float(arr[:, 4].mean()),
                "single_image_angle_error_deg_max": float(arr[:, 4].max()),
            }
            self._static_stats.append(stat)
            print(f"[STATIC] standoff {s:.2f} m (camera-Cylinder_01 {dist:.3f} m): position error mean {stat['position_error_mm_mean']:.3f} / max "
                  f"{stat['position_error_mm_max']:.3f} mm, angle error mean {stat['angle_error_deg_mean']:.4f} / max {stat['angle_error_deg_max']:.4f} deg "
                  f"| single image: {stat['single_image_position_error_mm_max']:.3f} mm / {stat['single_image_angle_error_deg_max']:.4f} deg max", flush=True)
            self._static_samples = []
            self._static_idx += 1
            self.goal = None
            self._settled_since = None

    _static_last_t = -1.0

    ###############################################
    ### Phase 2: Ares1 probe -> satellite dock ###
    ###############################################
    ## The controlled frame is the probe tip (PROBE_DOCK_POINT, measured from the USD by
    ## `docking.py`). The relative pose comes from the *simulated body poses* of the MEP
    ## and the satellite, i.e. it is transform/ground-truth based, NOT a vision estimate:
    ## no camera measures the satellite here. `cam_probe` depth is an independent
    ## cross-check of the docking-axis distance (`probe_dock.depth_valid`), never the
    ## only source, and the probe never advances on an invalid depth reading.

    def probe_world(self) -> Frame:
        """Live PROBE_DOCK_POINT: tip + insertion axis, from the MEP body pose."""
        return self.mep_frame() @ self.geo.probe_dock

    def dock_world(self) -> Frame:
        """Live SAT_DOCK_POINT: `dock_depth` inside the nozzle, +Z into the nozzle."""
        return self.sat_frame() @ self.geo.sat_dock

    def sat_frame(self) -> Frame:
        d = self.task._satellite.data
        return Frame.from_pos_quat(d.root_pos_w[0].tolist(), d.root_quat_w[0].tolist())

    def sat_exit_world(self) -> Frame:
        return self.sat_frame() @ self.geo.sat_exit

    def sat_velocity_at(self, p) -> np.ndarray:
        d = self.task._satellite.data
        v = d.root_com_lin_vel_w[0].cpu().numpy()
        w = d.root_com_ang_vel_w[0].cpu().numpy()
        c = d.root_com_pos_w[0].cpu().numpy()
        return v + np.cross(w, np.asarray(p, dtype=float) - c)

    def tool_velocity_vec(self) -> np.ndarray:
        """Velocity of the IK tool point [m/s] (world), from proprioception."""
        lid = self.arm.link_id
        v = self.task._robot.data.body_lin_vel_w[0, lid].cpu().numpy()
        w = self.task._robot.data.body_ang_vel_w[0, lid].cpu().numpy()
        return v + np.cross(w, self.arm.link_pose().rot @ self.arm.tool.pos)

    def probe_velocity(self) -> np.ndarray:
        """Velocity of the probe tip [m/s] (the MEP is rigid on the arm once captured)."""
        return self.gt_mep_velocity_at(self.probe_world().pos)[0]

    def read_probe_depth(self) -> Tuple[float, int]:
        """(median depth [m], valid pixels) at the principal point of `cam_probe`."""
        d = self.cfg.docking
        if self._probe_cam is None or not d.depth_enabled:
            return math.nan, 0
        out = self._probe_cam.data.output
        if "distance_to_image_plane" not in out:
            return math.nan, 0
        img = out["distance_to_image_plane"][0].cpu().numpy()
        return probe_dock.sample_depth(np.squeeze(img), self._probe_k, d.depth_patch_px, d.depth_min_m, d.depth_max_m)

    def dock_metrics(self) -> Dict[str, float]:
        """Relative pose, depth cross-check and clearance at the current instant."""
        d = self.cfg.docking
        probe, dock = self.probe_world(), self.dock_world()
        m = probe_dock.dock_errors(probe, dock)
        m["roll_rpy_deg"], m["pitch_rpy_deg"], m["yaw_rpy_deg"] = probe_dock.rpy_errors_deg(probe, dock)
        m["insertion_depth"] = probe_dock.insertion_depth(probe, self.sat_exit_world())
        m["geometry_distance"] = -m["axial"]  # remaining insertion distance [m]
        raw, px = self.read_probe_depth()
        m["depth_raw"], m["depth_pixels"] = raw, float(px)
        # (the configured fallback is a scale-reference length like `dock_depth`)
        offset = self._depth_offset if self._depth_offset is not None else d.depth_surface_offset_m * self.geo.length_scale
        m["depth_distance"] = probe_dock.depth_to_dock_distance(raw, self._cam_to_tip, offset)
        ok, why = probe_dock.depth_valid(d, m["depth_distance"], m["geometry_distance"])
        m["depth_ok"], self._depth_reason = float(ok), why
        rel_v = self.probe_velocity() - self.sat_velocity_at(probe.pos)
        m["rel_vx"], m["rel_vy"], m["rel_vz"] = (float(x) for x in rel_v)
        m["rel_speed"] = float(np.linalg.norm(rel_v))
        # Probe-to-wall gap at the current depth (only meaningful inside the nozzle)
        inner = self.geo.nozzle.inner_radius(max(0.0, m["insertion_depth"]))
        m["clearance"] = probe_dock.wall_clearance(inner, self.geo.probe.tip_radius, m["lateral"], m["insertion_depth"])
        return m

    def docking_tracking_valid(self) -> Tuple[bool, str]:
        """Fail-safe: the probe/dock frames must be finite and the MEP still held."""
        if not self.capture.is_attached(0):
            return False, "the MEP is no longer attached to the arm"
        probe, dock = self.probe_world(), self.dock_world()
        if not (np.isfinite(probe.pos).all() and np.isfinite(dock.pos).all()):
            return False, "probe tip or docking target pose is not finite"
        prev = self._dock.get("prev_dock_pos")
        if prev is not None:
            # Compared with where the target's own motion takes it since the last check
            # (this runs only in some states, so the previous sample can be many seconds
            # old). A drifting client moves metres between them; a static one does not
            # (v = 0 -> the previous check, unchanged).
            prev_t, prev_pos = prev
            expected = prev_pos + self.sat_velocity_at(prev_pos) * (self.sim_time - prev_t)
            jump = float(np.linalg.norm(dock.pos - expected))
            if jump > 0.25:  # the target cannot move that far in one control step
                return False, f"docking target jumped {jump*1000:.0f} mm in one step"
        self._dock["prev_dock_pos"] = (self.sim_time, dock.pos.copy())
        return True, "ok"

    def begin_docking(self):
        """Hand over from the capture phase: the IK tool becomes the probe tip."""
        d = self.cfg.docking
        link = self.arm.link_pose()
        tip_in_link = link.inv() @ self.probe_world()
        self.arm.set_tool(tip_in_link)
        self._dock_ref = self.arm.tool_pose()
        self.q_hold = None
        m = self.dock_metrics()
        print(f"[DOCK] probe tip is now the controlled frame (in link '{self.cfg.camera.name}' parent): "
              f"pos {np.round(tip_in_link.pos, 4).tolist()} m", flush=True)
        print(f"[DOCK] target acquired: remaining insertion {m['geometry_distance']:.3f} m, lateral {m['lateral']*1000:.1f} mm, "
              f"axis {m['axis_deg']:.2f} deg, roll {m['roll_deg']:.2f} deg, depth raw {m['depth_raw']:.3f} m ({int(m['depth_pixels'])} px)", flush=True)
        self.frame_docking_view(close=False)
        self.open_motion_hud()  # static docking too (the moving client opened it already)
        self.results.metrics["docking"] = {
            "probe_tip_in_link": {"pos": tip_in_link.pos.tolist(), "quat_wxyz": list(tip_in_link.quat)},
            "at_acquire": {k: v for k, v in m.items() if isinstance(v, float)},
            "pre_dock_axial_m": self._pre_dock_axial(),
        }

    def _pre_dock_axial(self) -> float:
        """Axial coordinate of the pre-dock pose: `pre_dock_distance_m` in front of the
        nozzle *exit*, i.e. `dock_depth + pre_dock_distance` before SAT_DOCK_POINT (both
        scaled with the satellite, `DockingGeometry.length_scale`)."""
        return -(self.geo.dock_depth + float(self.cfg.docking.pre_dock_distance_m) * self.geo.length_scale)

    def track_probe(self, goal: Frame, speed: float, max_step: Optional[float] = None):
        """Rate-limited probe-tip tracking with the satellite's motion fed forward.

        Same structure as the capture-phase `track()`, but the controlled frame is the
        probe tip and the feed-forward is the docking target's velocity, so a drifting
        satellite does not leave a standing tracking error.
        """
        d = self.cfg.docking
        if self._dock_ref is None:
            self._dock_ref = self.arm.tool_pose()
        prev = self._dock_ref
        pos = prev.pos + self.sat_velocity_at(self.dock_world().pos) * self.dt
        rot = prev.rot
        # The reference advances continuously towards the goal and then stops there.
        # It deliberately does NOT gate on the measured tracking error: an on/off gate
        # advances the reference on the half-swing that closes the error and freezes it
        # on the other, which rectifies the arm + 3 t payload mode (period ~20 s) and
        # pumps it -- measured as a steady +-150 mm lateral swing that never decayed.
        self._dock_speed = probe_dock.ramped(d, self._dock_speed, speed, self.dt)
        delta = goal.pos - pos
        n = float(np.linalg.norm(delta))
        if n > 1e-9:
            pos = pos + delta / n * min(n, self._dock_speed * self.dt)
        ang = rotation_angle(rot, goal.rot)
        if ang > 1e-9:
            f = min(1.0, math.radians(d.align_speed_deg_s) * self.dt / ang)
            rot = interp_frame(Frame(pos, rot), Frame(pos, goal.rot), f).rot
        self._dock_ref = Frame(pos, rot)
        n = float(np.linalg.norm(goal.pos - pos))  # distance still to run after this step
        # Once the reference has arrived, latch the joint target instead of re-deriving
        # it from the measured joints every step: `ik_joint_target` returns
        # `q_measured + lead`, so with the tip at its goal (lead -> 0) the target would
        # follow the joints and the arm would hold no pose at all, leaving the payload
        # swing undamped. Latching is the same mechanism the capture phase uses (`hold`).
        # (Integrating the target instead was tried and destabilised the arm: with a
        # joint stiffness of 40000 N m/rad even a 0.2 rad accumulated offset is huge.)
        moved = n > 1e-9 or ang > 1e-9
        if not moved:
            return self.hold()
        self.q_hold = None
        step = self.cfg.approach.max_joint_step_rad if max_step is None else max_step
        q = self.arm.ik_joint_target(self._dock_ref, step)
        self.task._robot.set_joint_position_target(q, joint_ids=self.arm.joint_ids)
        # Feed the reference's own velocity forward. The joint drives are PD on position
        # *and* velocity, so a zero velocity target brakes against the moving position
        # target (measured: the probe tip crept at ~20 mm/s with a 100 mm/s reference).
        v_cmd = (pos - prev.pos) / self.dt
        # Active damping: bias the joint velocity target against the measured tip
        # velocity error so the drives brake the lightly damped arm + payload swing
        # instead of only holding a position target.
        v_ff = v_cmd - d.swing_damping * (self.tool_velocity_vec() - v_cmd)
        self.set_velocity_feedforward(v_ff, so3_log(rot @ prev.rot.T) / self.dt)

    def dock_hold(self):
        """Stop the probe where it is (fail-safe / waiting states)."""
        self._dock_ref = self.arm.tool_pose()
        self.hold()

    def _dock_stage_timeout(self, m) -> bool:
        if self.state_time <= self.cfg.docking.stage_timeout_s:
            return False
        self.goto(State.DOCK_FAILED, f"{self.state.value} timed out after {self.state_time:.0f} s "
                                     f"(remaining {m['geometry_distance']:.3f} m, lateral {m['lateral']*1000:.1f} mm, axis {m['axis_deg']:.2f} deg)")
        return True

    ## States
    def step_dock_target_acquire(self, m):
        """Confirm the docking target and the probe pose, then plan the pre-dock pose."""
        self.dock_hold()
        if self.state_time < self.cfg.docking.settle_time_s:
            return
        ok, why = self.docking_tracking_valid()
        if not ok:
            return self.goto(State.DOCK_FAILED, f"docking target invalid: {why}")
        self.results.check("[DOCK1] Docking target and probe tip acquired", True,
                           f"probe tip {np.round(self.probe_world().pos, 3).tolist()}, SAT_DOCK_POINT {np.round(self.dock_world().pos, 3).tolist()}, "
                           f"remaining insertion {m['geometry_distance']:.3f} m")
        self.goto(State.PRE_DOCK_APPROACH)

    def step_pre_dock(self, m):
        """Move the probe tip onto the docking axis, `pre_dock_distance_m` in front of
        the nozzle exit (never a world-Z offset: the axis comes from the dock frame)."""
        d = self.cfg.docking
        goal = probe_dock.tip_goal(self.dock_world(), self._pre_dock_axial())
        pe, ae = pose_errors(self.arm.tool_pose(), goal)
        self._dock["commanded_speed"] = probe_dock.decelerated(d, d.transport_speed_mps, pe)
        self.track_probe(goal, self._dock["commanded_speed"])
        # Hand over only once the swing left by the transport has died out, otherwise the
        # alignment chases a tip that is still moving 100+ mm per half period
        forced = self.forced_insertion()
        # coupled_predictive: no settle window, ALIGNMENT_CHECK gates the relative velocity
        if pe < 0.05 and (forced or self.coupled or self.tip_settled()):
            self._dock_axial_hold = m["axial"]
            self.frame_docking_view(close=True)
            print(f"[DOCK] pre-dock reached{' (forced insertion: no settle wait)' if forced else ' and settled'}: tip {-m['insertion_depth']:.3f} m in front of the nozzle exit, "
                  f"lateral {m['lateral']*1000:.1f} mm, axis {m['axis_deg']:.2f} deg, goal error {pe*1000:.0f} mm", flush=True)
            if self.coupled:
                self.goto(State.POSITION_ATTITUDE_ALIGN)
            else:
                self.goto(State.ALIGNMENT_CHECK if forced else State.XY_ALIGN)
        else:
            self._dock_stage_timeout(m)

    def align_goal(self, m) -> Frame:
        """Tip goal for the alignment states: on the docking axis at a *frozen* axial
        distance. Using the live axial error would make the goal follow the arm's own
        oscillation, which keeps the swing alive instead of damping it."""
        if self._dock_axial_hold is None:
            self._dock_axial_hold = m["axial"]
        return probe_dock.tip_goal(self.dock_world(), self._dock_axial_hold)

    def align_track(self, m):
        """Decelerating lateral/angular correction with a reduced joint step."""
        d = self.cfg.docking
        goal = self.align_goal(m)
        pe, _ = pose_errors(self.arm.tool_pose(), goal)
        self._dock["commanded_speed"] = probe_dock.decelerated(d, d.align_speed_mps, pe)
        self.track_probe(goal, self._dock["commanded_speed"],
                         max_step=0.5 * self.cfg.approach.max_joint_step_rad)

    def tip_settled(self) -> bool:
        d = self.cfg.docking
        return probe_dock.settled(self._tip_history, self.sim_time, d.settle_window_s, d.settle_window_m)

    def step_xy_align(self, m):
        """Lateral (dock-frame X/Y) alignment at a constant docking-axis distance."""
        d = self.cfg.docking
        self.align_track(m)
        if m["lateral"] <= d.align_lateral_m:
            print(f"[DOCK] X/Y aligned: e_x {m['e_x']*1000:.2f} mm, e_y {m['e_y']*1000:.2f} mm "
                  f"(lateral {m['lateral']*1000:.2f} mm <= {d.align_lateral_m*1000:.0f} mm)", flush=True)
            self.goto(State.ORIENTATION_ALIGN)
        else:
            self._dock_stage_timeout(m)

    def step_orientation_align(self, m):
        """Roll/pitch/yaw alignment, still at a constant docking-axis distance."""
        d = self.cfg.docking
        self.align_track(m)
        if m["axis_deg"] <= d.align_axis_deg and math.isfinite(m["roll_deg"]) and m["roll_deg"] <= d.align_roll_deg:
            print(f"[DOCK] orientation aligned: axis {m['axis_deg']:.3f} deg, roll {m['roll_deg']:.3f} deg "
                  f"(rpy {m['roll_rpy_deg']:.2f} / {m['pitch_rpy_deg']:.2f} / {m['yaw_rpy_deg']:.2f} deg)", flush=True)
            self.goto(State.ALIGNMENT_CHECK)
        else:
            self._dock_stage_timeout(m)

    def step_alignment_check(self, m):
        """Both gates must hold together for `align_hold_s`, with the tip at rest,
        before any insertion."""
        d = self.cfg.docking
        self.align_track(m)
        if self.forced_insertion():
            return self.forced_alignment_check(m)
        ok, bad = probe_dock.alignment_ok(d, m)
        if ok and not self.tip_settled():
            ok, bad = False, [f"probe tip not at rest (> {d.settle_window_m*1000:.0f} mm over {d.settle_window_s:.0f} s)"]
            self._dock_align_since = None
            return self._dock_stage_timeout(m)
        if not ok:
            self._dock_align_since = None
            if m["lateral"] > d.align_lateral_m:
                return self.goto(State.XY_ALIGN, "; ".join(bad))
            return self.goto(State.ORIENTATION_ALIGN, "; ".join(bad))
        # Aligned and at rest: collect on-axis calibration samples (`_calibrate_depth`
        # keeps only those within the tighter calibration limits)
        needs_calibration = d.auto_calibrate and d.depth_enabled and self._depth_offset is None
        if needs_calibration:
            self._calibrate_depth(m)
            needs_calibration = self._depth_offset is None
        self._dock_align_since = self._dock_align_since if self._dock_align_since is not None else self.sim_time
        # With `require_depth` the approach cannot advance on an uncalibrated depth, so it
        # does not start before the on-axis calibration exists (the alignment keeps
        # converging meanwhile; `stage_timeout_s` still applies)
        if needs_calibration and d.require_depth:
            return self._dock_stage_timeout(m)
        if self._mv_active and self.cfg.rendezvous.concurrent_docking:
            ok_mv, _ = self.insertion_gates()
            if not ok_mv:
                return self._dock_stage_timeout(m)
        if self.sim_time - self._dock_align_since >= d.align_hold_s:
            self.results.check("[DOCK2] Alignment before the docking-axis approach", True,
                               f"lateral {m['lateral']*1000:.2f} mm (<= {d.align_lateral_m*1000:.0f}), axis {m['axis_deg']:.3f} deg (<= {d.align_axis_deg}), "
                               f"roll {m['roll_deg']:.3f} deg (<= {d.align_roll_deg}), held {d.align_hold_s:.1f} s")
            print(f"[DOCK] alignment check passed; starting the docking-axis approach "
                  f"(remaining {m['geometry_distance']:.3f} m, depth {m['depth_distance']:.3f} m)", flush=True)
            self.goto(State.Z_APPROACH)
        else:
            self._dock_stage_timeout(m)

    def step_z_approach(self, m, final: bool):
        """Closed-loop approach along the docking axis.

        Every step re-reads both body poses, so a drifting satellite is tracked; the
        forward motion stops and the demo returns to aligning as soon as the corridor
        is left, and it never advances without a valid depth reading.
        """
        d = self.cfg.docking
        forced = self.forced_insertion()
        ok, bad = self.forced_corridor(m) if forced else probe_dock.approach_still_aligned(d, m)
        if not ok:
            self._dock_align_since = None
            self._dock["realigns"] = int(self._dock.get("realigns", 0)) + 1
            print(f"[DOCK] approach stopped, re-aligning: {'; '.join(bad)}", flush=True)
            return self.goto(State.ALIGNMENT_CHECK if forced else State.XY_ALIGN)
        if forced and self._depth_offset is None and d.auto_calibrate and d.depth_enabled:
            self._calibrate_depth(m)  # on-axis samples only; the reading is advisory here
        valid, why = self.docking_tracking_valid()
        if not valid:
            return self.goto(State.DOCK_FAILED, f"docking target invalid: {why}")
        self._dock_axial_hold = None
        remaining = m["geometry_distance"]
        # Fail-safe: hold position (do not advance) while the depth is not trusted
        # Forced insertion: the depth is logged, it does not hold the approach
        depth_blocked = d.require_depth and not m["depth_ok"] and not forced
        if depth_blocked:
            self._dock["depth_blocked_s"] = float(self._dock.get("depth_blocked_s", 0.0)) + self.dt
            if not self._dock.get("depth_warned"):
                self._dock["depth_warned"] = True
                print(f"[DOCK] holding: {self._depth_reason}", flush=True)
            speed = 0.0
            axial_cmd = m["axial"]
        else:
            self._dock["depth_warned"] = False
            speed = probe_dock.approach_speed(d, remaining)
            axial_cmd = 0.0  # the dock point itself; the rate limit does the staging
        self._dock["commanded_speed"] = self._dock_speed
        goal = probe_dock.tip_goal(self.dock_world(), axial_cmd)
        step = self.cfg.approach.max_joint_step_rad if not final else 0.5 * self.cfg.approach.max_joint_step_rad
        v_ax = probe_dock.decelerated(d, speed, remaining, d.z_decel_gain_hz) if speed > 0.0 else 1e-9
        if forced:  # the MEP always closes on the client (relative speed along the axis)
            v_ax = max(v_ax, self.cfg.rendezvous.insertion_min_relative_speed_mps)
        self.track_probe(goal, v_ax, max_step=step)
        self._dock["min_clearance"] = min(float(self._dock.get("min_clearance", math.inf)), m["clearance"]) if m["insertion_depth"] > 0.0 else self._dock.get("min_clearance", math.inf)
        self._dock["max_lateral_inserting"] = max(float(self._dock.get("max_lateral_inserting", 0.0)), m["lateral"]) if m["insertion_depth"] > 0.0 else self._dock.get("max_lateral_inserting", 0.0)
        if not final and remaining <= d.insertion_zone_m:
            print(f"[DOCK] final insertion zone ({remaining*1000:.0f} mm remaining, depth {m['depth_distance']:.3f} m, "
                  f"wall clearance {m['clearance']*1000:.0f} mm)", flush=True)
            return self.goto(State.FINAL_INSERTION)
        if final and abs(m["axial"]) <= d.dock_axial_m:
            return self.goto(State.DOCK_READY)
        self._dock_stage_timeout(m)

    def step_dock_ready(self, m):
        """Every docking condition is re-checked here; only then is the joint created."""
        d = self.cfg.docking
        if self.coupled:
            # The docking conditions already hold the tip at the dock point: keep it there
            # relative to the client (axial set point frozen at the tip on entry) instead
            # of pressing on towards 0 -- pressing pushed the client and the MRV followed
            cm = self._cd_last
            if not self._dock.get("cd_ready_axial_frozen"):
                self._dock["cd_ready_axial_frozen"] = True
                self._cd_axial = cm["axial"]
            self.coupled_track(cm, self._hold_axial(cm), max_step=0.5 * self.cfg.approach.max_joint_step_rad)
        else:
            self.track_probe(probe_dock.tip_goal(self.dock_world(), 0.0), d.insertion_speed_mps,
                             max_step=0.5 * self.cfg.approach.max_joint_step_rad)
        valid, why = self.docking_tracking_valid()
        ready, bad = probe_dock.dock_ready(d, m, m["rel_speed"], bool(m["depth_ok"]) or self.forced_insertion(), valid, m["clearance"])
        if self._mv_active:
            # Moving client: the same conditions, plus MEP-client and MRV-client RELATIVE
            # velocity and the MRV station error. The world velocity is not a condition.
            ok_mv, bad_mv = self.moving_dock_gates()
            ready, bad = ready and ok_mv, bad + bad_mv
        if ready:
            detail = (f"axial {m['axial']*1000:.2f} mm, radial {m['lateral']*1000:.2f} mm, axis {m['axis_deg']:.3f} deg, roll {m['roll_deg']:.3f} deg, "
                      f"depth {m['insertion_depth']:.3f} m inside the nozzle, relative velocity {m['rel_speed']*1000:.1f} mm/s, "
                      f"wall clearance {m['clearance']*1000:.0f} mm, depth reading {m['depth_distance']:.3f} m")
            print(f"[DOCK] DOCK_READY  t={self.sim_time:.2f} s | {detail}", flush=True)
            cm = self._cd_last
            self._cd_metrics.docked(self.sim_time, {
                "position_error_m": math.hypot(cm["lateral"], cm["axial"]),
                "orientation_error_deg": cm["orientation_deg"], "relative_speed_mps": cm["rel_speed"]})
            self.results.check("[DOCK4] Docking conditions", True, detail)
            sat_before = self.sat_frame()
            self.task.docking.dock()
            self._dock["docked"] = self.task.docking.is_docked
            self._dock["at_dock"] = {k: v for k, v in m.items() if isinstance(v, float)}
            self._dock["sat_at_dock"] = sat_before
            self._dock["rel_at_dock"] = self.probe_world().inv() @ self.sat_frame()
            self.results.check("[DOCK5] Docking FixedJoint created", bool(self.task.docking.is_docked),
                               "UsdPhysics.FixedJoint MEP <-> satellite anchored at the probe tip")
            if self._mv_active:
                self.record_moving_dock()
            self.q_hold = self.arm.joint_pos().clone()
            self.goto(State.DOCKED if self.task.docking.is_docked else State.DOCK_FAILED,
                      "" if self.task.docking.is_docked else "the docking joint was not created")
        elif self.state_time > d.dock_ready_timeout_s:
            self.results.check("[DOCK4] Docking conditions", False, "; ".join(bad))
            self.goto(State.DOCK_FAILED, "docking conditions not met: " + "; ".join(bad))

    def step_docked(self, m):
        self.dock_hold()
        # Moving client: the docked stack is still drifting -> stabilise, then stop it
        if self._mv_active:
            # immediate_release: let go at once, the stack stops in the background and the
            # MRV departs; otherwise stabilise and stop everything first
            self.goto(State.ROBOT_RELEASE if self.cfg.post_docking.immediate_release else State.STABILIZING)
        else:
            self.goto(State.DOCK_HOLDING)

    def step_dock_holding(self, m):
        """Hold the docked state and watch for jumps / penetration / joint loss."""
        d = self.cfg.docking
        self.hold()
        h = self._dock
        rel_now = self.probe_world().inv() @ self.sat_frame()
        pe, ae = pose_errors(rel_now, h["rel_at_dock"])
        h["hold_max_drift_mm"] = max(float(h.get("hold_max_drift_mm", 0.0)), pe * 1000.0)
        h["hold_max_drift_deg"] = max(float(h.get("hold_max_drift_deg", 0.0)), ae)
        h["hold_joint_valid"] = bool(h.get("hold_joint_valid", True)) and self.task.docking.is_docked
        h["hold_min_clearance_mm"] = min(float(h.get("hold_min_clearance_mm", math.inf)), m["clearance"] * 1000.0)
        if self.state_time >= d.hold_duration_s:
            h["hold_duration_s"] = self.state_time
            self.results.check("[DOCK6] Docked state held", h["hold_joint_valid"] and h["hold_max_drift_mm"] < 5.0 and h["hold_max_drift_deg"] < 0.5,
                               f"{self.state_time:.1f} s: max MEP-satellite relative drift {h['hold_max_drift_mm']:.3f} mm / {h['hold_max_drift_deg']:.4f} deg, "
                               f"joint present the whole time: {h['hold_joint_valid']}")
            self.results.check("[DOCK7] No penetration while docked", h["hold_min_clearance_mm"] > 0.0,
                               f"min probe-to-wall clearance {h['hold_min_clearance_mm']:.0f} mm")
            self.goto(State.SUCCESS)

    #####################################################################
    ### Coupled predictive docking control (`docking_control.mode`) ###
    #####################################################################
    ## `coupled_dock.py` has the math. Only POSITION_ATTITUDE_ALIGN and, with
    ## `mode: coupled_predictive`, ALIGNMENT_CHECK / Z_APPROACH / FINAL_INSERTION /
    ## DOCK_READY's tracking run here; `control_measure` and `log_control_row` run in
    ## both modes (read-only) so the two controllers are logged the same way.

    def client_twist(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(CoM [m], CoM velocity [m/s], angular velocity [rad/s]) of the client, world."""
        d = self.task._satellite.data
        return (d.root_com_pos_w[0].cpu().numpy().astype(float), d.root_com_lin_vel_w[0].cpu().numpy().astype(float),
                d.root_com_ang_vel_w[0].cpu().numpy().astype(float))

    def control_measure(self, m) -> Dict[str, object]:
        """Client motion, predicted dock frame and measured-tip gate errors (both modes)."""
        c = self.cfg.docking_control
        tip, dock = self.probe_world(), self.dock_world()
        com, v_com, w_c = self.client_twist()
        T = float(c.prediction_horizon_sec)
        v_tip, w_tip = self._cd_tip_twist.update(self.sim_time, tip)
        v_port = coupled_dock.point_velocity(dock.pos, com, v_com, w_c)
        ge = coupled_dock.dock_frame_errors(tip, dock)
        rel_v = v_tip - v_port if v_tip is not None else np.full(3, math.nan)
        rel_w = w_tip - w_c if w_tip is not None else np.full(3, math.nan)
        cm = {
            "tip": tip, "dock": dock, "com": com, "v_com": v_com, "w_c": w_c, "T": T,
            "dock_pred": coupled_dock.predict_body_frame(dock, com, v_com, w_c, T),
            "v_port": v_port, "v_tip": v_tip, "w_tip": w_tip, "rel_v": rel_v, "rel_w": rel_w,
            "rel_speed": float(np.linalg.norm(rel_v)), "rel_rate_deg_s": math.degrees(float(np.linalg.norm(rel_w))),
            "insertion_depth": m["insertion_depth"], **ge,
        }
        cm["scale"] = coupled_dock.alignment_scale(c, cm["lateral"], cm["orientation_deg"])
        self._cd_metrics.update(self.sim_time, self.state.value, cm["lateral"], cm["orientation_deg"], cm["rel_speed"],
                                m["insertion_depth"])
        self._cd_last = cm
        return cm

    def _hold_axial(self, cm) -> float:
        """Alignment states: the desired tip stays at a frozen axial distance (as the
        legacy `align_goal`: a live axial goal would follow the arm's own swing)."""
        if self._cd_axial is None:
            self._cd_axial = cm["axial"]
        self._cd_axial_rate = 0.0
        return self._cd_axial

    def _advance_axial(self, cm, speed: float) -> float:
        """Axial set point moved towards the dock point at `speed` [m/s] RELATIVE to the
        client, never more than `max_axial_lead_m` (inside the nozzle
        `max_axial_lead_inside_m`) ahead of the measured tip."""
        c = self.cfg.docking_control
        old = self._hold_axial(cm)
        lead = c.max_axial_lead_inside_m if cm["insertion_depth"] > 0.0 else c.max_axial_lead_m
        new = min(0.0, old + max(0.0, speed) * self.dt, cm["axial"] + lead)
        self._cd_axial, self._cd_axial_rate = new, (new - old) / self.dt
        return new

    def coupled_track(self, cm, axial_cmd: float, max_step: Optional[float] = None):
        """One coupled position + attitude tracking step of the probe-tip reference
        towards the desired tip pose on the PREDICTED docking axis (`coupled_dock`)."""
        if self._dock_ref is None:
            self._dock_ref = self.arm.tool_pose()
        if not self._cd_active:  # take over from the transport without a velocity step
            self._cd_active = True
            self._cd_tracker.reset(cm["v_tip"] if cm["v_tip"] is not None else cm["v_port"],
                                   cm["w_tip"] if cm["w_tip"] is not None else np.zeros(3))
        prev = self._dock_ref
        desired = coupled_dock.desired_tip(cm["dock_pred"], axial_cmd)
        com_pred = cm["com"] + cm["v_com"] * cm["T"]
        v_des = (coupled_dock.point_velocity(desired.pos, com_pred, cm["v_com"], cm["w_c"])
                 + self._cd_axial_rate * desired.rot[:, 2])
        v_tip = cm["v_tip"] if cm["v_tip"] is not None else v_des
        w_tip = cm["w_tip"] if cm["w_tip"] is not None else cm["w_c"]
        out = self._cd_tracker.step(prev, cm["tip"], v_tip, w_tip, desired, v_des, cm["w_c"], self.dt)
        self._cd_out = out
        ref = out.ref
        if float(np.linalg.norm(ref.pos - prev.pos)) < 1e-7 and rotation_angle(prev.rot, ref.rot) < 1e-7:
            return self.hold()  # at rest on a static goal: latch the joints (see `track_probe`)
        self._dock_ref = ref
        self.q_hold = None
        q = self.arm.ik_joint_target(ref, self.cfg.approach.max_joint_step_rad if max_step is None else max_step)
        self.task._robot.set_joint_position_target(q, joint_ids=self.arm.joint_ids)
        # The reference's own twist as joint velocity target (drives are PD on q and q_dot)
        self.set_velocity_feedforward((ref.pos - prev.pos) / self.dt, so3_log(ref.rot @ prev.rot.T) / self.dt)

    def step_position_attitude_align(self, m, cm):
        """Lateral and attitude errors reduced together at a frozen axial distance."""
        self.coupled_track(cm, self._hold_axial(cm), max_step=0.5 * self.cfg.approach.max_joint_step_rad)
        ok, bad = self._cd_gate.entry_conditions(cm["lateral"], cm["orientation_deg"], cm["rel_speed"], cm["rel_rate_deg_s"])
        if ok:
            print(f"[DOCK] position + attitude aligned: lateral {cm['lateral']*1000:.1f} mm, orientation "
                  f"{cm['orientation_deg']:.2f} deg, relative {cm['rel_speed']*1000:.1f} mm/s / {cm['rel_rate_deg_s']:.2f} deg/s", flush=True)
            self._cd_gate.reset()
            self.goto(State.ALIGNMENT_CHECK)
        else:
            self._dock_stage_timeout(m)

    def step_coupled_alignment_check(self, m, cm):
        """Every condition held for `docking_alignment.stable_duration_sec` (no timer-only
        wait); back to POSITION_ATTITUDE_ALIGN only past the wider rollback thresholds."""
        d, a = self.cfg.docking, self.cfg.docking_alignment
        self.coupled_track(cm, self._hold_axial(cm), max_step=0.5 * self.cfg.approach.max_joint_step_rad)
        back, bad_rb = self._cd_gate.rollback(self.sim_time, cm["lateral"], cm["orientation_deg"])
        if back:
            return self.goto(State.POSITION_ATTITUDE_ALIGN, "; ".join(bad_rb))
        forced = self.forced_insertion()
        blocked = []
        if d.auto_calibrate and d.depth_enabled and self._depth_offset is None:
            self._calibrate_depth(m)  # on-axis samples only (same rule as legacy)
            if self._depth_offset is None and d.require_depth and not forced:
                blocked.append("depth not calibrated on the docking axis yet")
        if self._mv_active and self.cfg.rendezvous.concurrent_docking:
            ok_mv, bad_mv = self.insertion_gates(mep_speed=not forced)
            blocked += [] if ok_mv else bad_mv
        passed, held, bad = self._cd_gate.entry(self.sim_time, cm["lateral"], cm["orientation_deg"], cm["rel_speed"],
                                                cm["rel_rate_deg_s"], blocked)
        if passed:
            self.results.check("[DOCK2] Alignment before the docking-axis approach", True,
                               f"coupled: lateral {cm['lateral']*1000:.2f} mm (< {a.position_threshold_m*1000:.0f}), orientation "
                               f"{cm['orientation_deg']:.3f} deg (< {a.orientation_threshold_deg:g}), relative {cm['rel_speed']*1000:.1f} mm/s, "
                               f"{cm['rel_rate_deg_s']:.2f} deg/s, held {held:.2f} s")
            print(f"[DOCK] coupled alignment held {held:.2f} s; starting the docking-axis approach "
                  f"(remaining {m['geometry_distance']:.3f} m)", flush=True)
            self._cd_gate.reset()
            self.goto(State.Z_APPROACH)
        else:
            self._dock_stage_timeout(m)

    def step_coupled_approach(self, m, cm, final: bool):
        """Docking-axis approach at a soft-gated speed while position and attitude keep
        being tracked; emergency hard stop / hysteresis rollback back to aligning."""
        d, c = self.cfg.docking, self.cfg.docking_control
        forced = self.forced_insertion()
        # Relative speed is a contact hazard only inside the nozzle (the far approach
        # itself closes at up to `approach_speed_mps`)
        inside = m["insertion_depth"] > 0.0
        rel = cm["rel_speed"] if inside and math.isfinite(cm["rel_speed"]) else 0.0
        stop, bad = coupled_dock.emergency_stop(c, cm["lateral"], cm["orientation_deg"], rel, m["clearance"], d.min_wall_clearance_m)
        back = stop
        if not stop:
            back, bad = self._cd_gate.rollback(self.sim_time, cm["lateral"], cm["orientation_deg"])
        if back:
            self._dock["realigns"] = int(self._dock.get("realigns", 0)) + 1
            self._cd_axial, self._cd_axial_rate, self._dock_speed = cm["axial"], 0.0, 0.0
            if stop:  # hard stop: no relative motion from here, the reference is the tip
                self._cd_metrics.emergency_stops += 1
                self._cd_tracker.reset(cm["v_port"], cm["w_c"])
                self.dock_hold()
            print(f"[DOCK] {'EMERGENCY STOP' if stop else 'approach rolled back'}, re-aligning: {'; '.join(bad)}", flush=True)
            return self.goto(State.POSITION_ATTITUDE_ALIGN, "; ".join(bad))
        if forced and self._depth_offset is None and d.auto_calibrate and d.depth_enabled:
            self._calibrate_depth(m)  # on-axis samples only; the reading is advisory here
        valid, why = self.docking_tracking_valid()
        if not valid:
            return self.goto(State.DOCK_FAILED, f"docking target invalid: {why}")
        remaining = m["geometry_distance"]
        if d.require_depth and not m["depth_ok"] and not forced:  # same fail-safe as legacy
            self._dock["depth_blocked_s"] = float(self._dock.get("depth_blocked_s", 0.0)) + self.dt
            if not self._dock.get("depth_warned"):
                self._dock["depth_warned"] = True
                print(f"[DOCK] holding: {self._depth_reason}", flush=True)
            speed = 0.0
        else:
            self._dock["depth_warned"] = False
            # Soft gate: the approach profile scaled by the alignment (never a 0/1 switch)
            speed = probe_dock.decelerated(d, probe_dock.approach_speed(d, remaining), remaining, d.z_decel_gain_hz) * cm["scale"]
        self._cd_scale = cm["scale"]
        self._dock_speed = probe_dock.ramped(d, self._dock_speed, speed, self.dt)
        self._dock["commanded_speed"] = self._dock_speed
        step = self.cfg.approach.max_joint_step_rad if not final else 0.5 * self.cfg.approach.max_joint_step_rad
        self.coupled_track(cm, self._advance_axial(cm, self._dock_speed), max_step=step)
        if m["insertion_depth"] > 0.0:
            self._dock["min_clearance"] = min(float(self._dock.get("min_clearance", math.inf)), m["clearance"])
            self._dock["max_lateral_inserting"] = max(float(self._dock.get("max_lateral_inserting", 0.0)), m["lateral"])
        if not final and remaining <= d.insertion_zone_m:
            print(f"[DOCK] final insertion zone ({remaining*1000:.0f} mm remaining, alignment scale {cm['scale']:.2f}, "
                  f"wall clearance {m['clearance']*1000:.0f} mm)", flush=True)
            return self.goto(State.FINAL_INSERTION)
        if final and abs(m["axial"]) <= d.dock_axial_m:
            return self.goto(State.DOCK_READY)
        self._dock_stage_timeout(m)

    def log_control_row(self, m, cm):
        """`<run>_docking_control.csv` row (both modes; see `coupled_dock.CSV_COLUMNS`)."""
        if self._cd_csv is None:
            return
        axial = self._cd_axial if (self.coupled and self._cd_axial is not None) else cm["axial"]
        desired = coupled_dock.desired_tip(cm["dock_pred"], axial)
        e_p, e_r = coupled_dock.pose_error(desired, cm["tip"])
        out = self._cd_out if self.coupled else None
        row = {"run_id": self.label, "timestamp": round(self.sim_time, 4), "mode": self.cfg.docking_control.mode,
               "fsm_state": self.state.value}

        def put(prefix, v, keys="xyz", q=""):
            row.update({f"{prefix}_{q}{k}": ("" if v is None else float(x)) for k, x in zip(keys, v if v is not None else [None] * len(keys))})

        put("client_dock_position", cm["dock"].pos)
        put("client_dock_orientation", cm["dock"].quat, "wxyz", "q")
        put("client_linear_velocity", cm["v_com"])
        put("client_angular_velocity", cm["w_c"])
        put("desired_dock_position", cm["dock_pred"].pos)
        put("desired_dock_orientation", cm["dock_pred"].quat, "wxyz", "q")
        put("probe_tip_position", cm["tip"].pos)
        put("probe_tip_orientation", cm["tip"].quat, "wxyz", "q")
        put("position_error", e_p)
        put("orientation_error_rotvec", e_r)
        put("relative_linear_velocity", cm["rel_v"])
        put("relative_angular_velocity", cm["rel_w"])
        put("command_linear_velocity", None if out is None else out.v_cmd)
        put("command_angular_velocity", None if out is None else out.w_cmd)
        row.update({
            "position_error_m": float(np.linalg.norm(e_p)), "lateral_error_m": cm["lateral"], "axial_error_m": cm["axial"],
            "orientation_error_deg": cm["orientation_deg"], "relative_speed_mps": cm["rel_speed"],
            "relative_angular_rate_deg_s": cm["rel_rate_deg_s"], "prediction_horizon": cm["T"], "axial_command_m": axial,
            "approach_speed": float(self._dock_speed), "alignment_scale": cm["scale"],
            "rollback_count": self._cd_metrics.rollbacks, "emergency_stop_count": self._cd_metrics.emergency_stops,
            "docking_success": int(self.task.docking.is_docked),
        })
        self._cd_csv.writerow({k: row.get(k, "") for k in coupled_dock.CSV_COLUMNS})

    ###############################################################
    ### Moving client: rendezvous -> dock -> stop -> release -> depart ###
    ###############################################################
    ## Frames: world, SI units. Everything gated is RELATIVE to the client:
    ##   v_rel_mrv = v_mrv - v_client,  v_rel_mep = v_mep - v_client,
    ##   e_mrv = (p_client - p_mrv) - d_ref  (d_ref: the verified client -> MRV offset).
    ## Joints: robot <-> MEP = `CaptureManager` `{ENV}/capture_joint` (released in
    ## ROBOT_RELEASE); MEP <-> client = `DockingManager` `{ENV}/docking_joint` (kept).

    def mrv_base_velocity(self) -> np.ndarray:
        """Commanded world velocity of the MRV (arm base) [m/s]; zero unless it translates."""
        if not self._mv_active:
            return np.zeros(3)
        return self._mv_ctrl.v.copy()

    def _body_mass_inertia(self, body) -> Tuple[float, np.ndarray]:
        """Total mass [kg] and world-frame inertia about the COM [kg m^2] of a rigid object."""
        mass = float(body.root_physx_view.get_masses().sum())
        inertia = body.root_physx_view.get_inertias().reshape(-1)[:9].cpu().numpy().astype(float).reshape(3, 3)
        rot = Frame.from_pos_quat(np.zeros(3), body.data.root_com_quat_w[0].tolist()).rot
        return mass, rot @ inertia @ rot.T

    def apply_wrench(self, name: str, force_w, torque_w):
        """World-frame force [N] / torque [N m] at the COM, applied every physics step
        until changed (`RigidObject.set_external_force_and_torque`, written by
        `scene.write_data_to_sim`). Passed in the body (link) frame, the asset's default
        wrench frame: switching an asset to `is_global=True` makes Isaac Lab warn once,
        and every caller here re-sets the wrench each step anyway."""
        body = self.task._satellite if name == "client" else self.mep
        rot = Frame.from_pos_quat(np.zeros(3), body.data.root_quat_w[0].tolist()).rot
        f_b = rot.T @ np.asarray(force_w, dtype=float)
        t_b = rot.T @ np.asarray(torque_w, dtype=float)
        f = torch.tensor(f_b, dtype=torch.float32, device=body.device).reshape(1, 1, 3)
        tq = torch.tensor(t_b, dtype=torch.float32, device=body.device).reshape(1, 1, 3)
        body.set_external_force_and_torque(f, tq, is_global=False)
        self._mv_force[name] = np.asarray(force_w, dtype=float).copy()

    def clear_wrenches(self):
        if not self._mv_active:
            return
        for name in ("client", "mep"):
            self.apply_wrench(name, np.zeros(3), np.zeros(3))

    def client_constraints_report(self) -> dict:
        """PXR stage traversal: is the client a free dynamic body, and what is jointed to it?

        Joints between two prims *inside* the client are its own structure and are never
        touched. A joint with the client on one side and nothing (the world) on the other
        would hold the whole client in place: such a joint is disabled here. None exists
        in `satellite_v3.usd` (measured), so this only reports.
        """
        from pxr import UsdPhysics

        stage = self.stage
        root, body = self.geo.sat_prim_path, self.geo.sat_body_path
        prim = stage.GetPrimAtPath(body)
        rb = UsdPhysics.RigidBodyAPI(prim)
        out = {
            "body": body,
            "rigid_body_api": bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
            "rigid_body_enabled": bool(rb.GetRigidBodyEnabledAttr().Get()) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else False,
            "kinematic": bool(rb.GetKinematicEnabledAttr().Get()) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else None,
            "disable_gravity_attr": prim.GetAttribute("physxRigidBody:disableGravity").Get() if prim.HasAttribute("physxRigidBody:disableGravity") else None,
            "scene_gravity": list(self.task.cfg.sim.gravity),
            "internal_joints": [], "external_joints": [], "world_joints_disabled": [],
        }

        def inside(path: str) -> bool:
            return path == root or path.startswith(root + "/")

        for p in stage.Traverse():
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            if not any(inside(t) for t in b0 + b1):
                continue
            entry = {"path": str(p.GetPath()), "body0": b0, "body1": b1, "enabled": bool(j.GetJointEnabledAttr().Get())}
            if b0 and b1 and all(inside(t) for t in b0 + b1):
                out["internal_joints"].append(entry)  # the client's own structure: kept
            elif not b0 or not b1:
                if entry["enabled"]:
                    j.GetJointEnabledAttr().Set(False)
                    out["world_joints_disabled"].append(entry)
            else:
                out["external_joints"].append(entry)
        return out

    def begin_client_drift(self):
        """t = 0: release the client while the MRV approach / capture run as verified.

        The MRV station `d_ref` is fixed here from the NOMINAL poses (the MRV approach
        displaces the MRV only temporarily), i.e. the relative geometry the docking was
        verified in. When the docking starts the client has drifted ahead; the MRV then
        closes that gap next to the docking transport and the insertion waits for it
        (`insertion_gates`)."""
        c = self.cfg
        if self.mrv is None:
            self.mrv = MrvTransit(self.task, self.arm)
        p_client = self.task._satellite.data.root_com_pos_w[0].cpu().numpy().astype(float)
        self._mv_d_ref = p_client - self.mrv.nominal_root.cpu().numpy().astype(float)
        self._mv["start_time_s"] = self.sim_time
        self._client_live = True
        self.client_live_step(apply=False)
        print(f"[MOVE] client released at t = {self.sim_time:.1f} s (before the capture): drift "
              f"{np.round(c.client.velocity(), 4).tolist()} m/s; MRV station d_ref {np.round(self._mv_d_ref, 3).tolist()} m "
              f"(nominal poses), the MRV joins at the docking handover", flush=True)

    def client_live_step(self, apply: bool = True):
        """Before the docking phase: client thrust ramp, trail, window (MRV untouched)."""
        sat = self.task._satellite.data
        p_c = sat.root_com_pos_w[0].cpu().numpy().astype(float)
        self._mvm.update({"client_p": p_c, "client_v_px": sat.root_com_lin_vel_w[0].cpu().numpy().astype(float),
                          "client_w": sat.root_com_ang_vel_w[0].cpu().numpy().astype(float)})
        if apply:
            self.concurrent_release_step()
        lg = self.cfg.logging
        if lg.motion_trail and self.sim_time >= self._next_trail_t:
            self._next_trail_t = self.sim_time + lg.motion_trail_period_s - 1e-9
            pts = self._trail["client"]
            pts.append(p_c.copy())
            if len(pts) > int(lg.motion_trail_max_points):
                del pts[1]

    def begin_moving_client(self) -> State:
        """Hand over from the capture (MEP held by the arm) to the moving-client scenario."""
        c = self.cfg
        r = self.results
        report = self.client_constraints_report()
        self._mv["client_report"] = report
        g0 = all(abs(g) < 1e-9 for g in report["scene_gravity"]) or bool(report["disable_gravity_attr"])
        free = report["rigid_body_api"] and report["rigid_body_enabled"] and report["kinematic"] is False
        r.check("[MOVE0] Client is a free dynamic rigid body", free and g0 and not report["external_joints"],
                f"{report['body']}: RigidBodyAPI {report['rigid_body_api']}, enabled {report['rigid_body_enabled']}, "
                f"kinematic {report['kinematic']}, scene gravity {report['scene_gravity']}; joints to the world disabled: "
                f"{[j['path'] for j in report['world_joints_disabled']] or 'none found'}; internal joints kept: "
                f"{len(report['internal_joints'])}; joints to other bodies: {[j['path'] for j in report['external_joints']] or 'none'}")
        print(f"[MOVE] client constraints (PXR traversal): {report}", flush=True)
        if not (free and g0):
            return State.CLIENT_RELEASE_FAILED
        ## MRV handle: the one the approach phase used (at its nominal pose by now), or a
        ## new one at the current (nominal) pose
        try:
            if self.mrv is None:
                self.mrv = MrvTransit(self.task, self.arm)
            if self.vfx is None and c.mrv.vfx_enabled:
                try:
                    self.vfx = ThrusterVfx(self.task, c.mrv, hull_nominal=self.mrv.hull_nominal, quiet=True)
                except Exception as e:
                    print(f"[MRV-VFX] plume layer unavailable ({e}); the scenario continues without it", flush=True)
        except Exception as e:
            r.check("[MOVE0] MRV can be translated", False, f"{e}")
            return State.CLIENT_RELEASE_FAILED
        sat = self.task._satellite
        m_c, _ = self._body_mass_inertia(sat)
        m_m, _ = self._body_mass_inertia(self.mep)
        p_client = sat.data.root_com_pos_w[0].cpu().numpy().astype(float)
        if self._mv_d_ref is None:  # else: fixed at t = 0 from the nominal poses (`begin_client_drift`)
            self._mv_d_ref = p_client - self.mrv.root_pos()
        self._mv_sep_dir = separation_direction(c.client.velocity(), c.separation.direction)
        self._mv_ctrl.reset()
        for est in self._mv_est.values():
            est.reset()
        self._mv_active = True
        self._mv.update({
            "client_mass_kg": m_c, "mep_mass_kg": m_m, "d_ref_m": self._mv_d_ref.tolist(),
            "separation_direction": self._mv_sep_dir.tolist(),
            "start_time_s": float(self._mv.get("start_time_s", self.sim_time)),
            "docking_start_time_s": self.sim_time,
            "capture_joint": self.capture._joint_prim_paths[0], "docking_joint": self.task.docking._joint_path,
        })
        self.results.metrics["moving"] = self._mv
        self.moving_measure()
        self.open_motion_hud()
        if float(np.linalg.norm(c.client.angular_velocity())) > 0.0:
            print("[MOVE] WARNING: client.angular_velocity_rad_s is non-zero; the MRV only translates, "
                  "so docking a rotating client is not a verified configuration", flush=True)
        print(f"[MOVE] moving-client scenario: client {m_c:.0f} kg, MEP {m_m:.0f} kg; client velocity "
              f"{np.round(c.client.velocity(), 4).tolist()} m/s; MRV station d_ref = p_client - p_mrv = "
              f"{np.round(self._mv_d_ref, 3).tolist()} m; departure direction {np.round(self._mv_sep_dir, 3).tolist()}; "
              f"robot<->MEP joint {self._mv['capture_joint']} (released later), MEP<->client joint {self._mv['docking_joint']} (kept)",
              flush=True)
        if c.rendezvous.concurrent_docking:
            # No waiting: the docking transport starts now; the release thrust and the MRV
            # station keeping run next to it, the relative gates are checked before insertion
            self.begin_docking()
            return State.DOCK_TARGET_ACQUIRE
        return State.CLIENT_RELEASE

    def moving_measure(self):
        """All client / MEP / MRV quantities of this control step (world frame)."""
        sat, mep = self.task._satellite.data, self.mep.data
        t = self.sim_time
        p_c = sat.root_com_pos_w[0].cpu().numpy().astype(float)
        p_m = mep.root_com_pos_w[0].cpu().numpy().astype(float)
        p_r = self.mrv.root_pos()
        for key, p in (("client", p_c), ("mep", p_m), ("mrv", p_r)):
            self._mv_est[key].add(t, p)
        v_c_px = sat.root_com_lin_vel_w[0].cpu().numpy().astype(float)
        v_m_px = mep.root_com_lin_vel_w[0].cpu().numpy().astype(float)
        v_c = self._mv_est["client"].velocity()
        v_m = self._mv_est["mep"].velocity()
        v_r = self._mv_ctrl.v.copy()
        v_c_gate = v_c if v_c is not None else v_c_px
        contact = 0.0
        if self.contacts is not None:
            contact = float(torch.norm(self.contacts.data.net_forces_w[0], dim=-1).max())
        probe, dock = self.probe_world(), self.dock_world()
        e_dock = probe_dock.dock_errors(probe, dock)
        self._mvm = {
            "client_p": p_c, "client_v": v_c_gate, "client_v_fd": v_c, "client_v_px": v_c_px,
            "client_w": sat.root_com_ang_vel_w[0].cpu().numpy().astype(float),
            "mep_p": p_m, "mep_v": v_m, "mep_v_px": v_m_px,
            "mep_w": mep.root_com_ang_vel_w[0].cpu().numpy().astype(float),
            "mrv_p": p_r, "mrv_v": v_r, "mrv_v_fd": self._mv_est["mrv"].velocity(), "mrv_a": self._mv_ctrl.a.copy(),
            "v_rel_mrv": relative(v_r, v_c) if v_c is not None else None,
            "v_rel_mep": relative(v_m, v_c) if (v_m is not None and v_c is not None) else None,
            "e_mrv": self._mv_ctrl.position_error(p_r, p_c, self._mv_d_ref),
            "rel_mep_client": relative(probe.pos, dock.pos),  # MEP docking point - client docking point
            "dock_position_error": float(np.linalg.norm(probe.pos - dock.pos)),
            "dock_orientation_error": e_dock["orientation_deg"],
            # probe axis vs nozzle axis (dock frame +Z): the angle the insertion aborts on
            "dock_axis_deg": e_dock["axis_deg"], "dock_lateral_m": e_dock["lateral"],
            "dock_roll_deg": e_dock["roll_deg"],
            "contact_n": contact,
        }

    @staticmethod
    def _speed(v) -> float:
        return math.nan if v is None else float(np.linalg.norm(v))

    def check_client_physics(self) -> Optional[str]:
        """PHYSICS_UNSTABLE (reported as PHYSICS_ERROR): NaN / runaway client."""
        cc = self.cfg.client
        d = self.task._satellite.data
        if not all(bool(torch.isfinite(x).all()) for x in (d.root_pos_w, d.root_quat_w, d.root_com_lin_vel_w, d.root_com_ang_vel_w)):
            return "PHYSICS_UNSTABLE: NaN/Inf in the client state"
        v = float(torch.norm(d.root_com_lin_vel_w[0]))
        w = float(torch.norm(d.root_com_ang_vel_w[0]))
        if v > cc.max_speed_mps:
            return f"PHYSICS_UNSTABLE: client speed {v:.3f} m/s > {cc.max_speed_mps} (explosion)"
        if w > cc.max_angular_rate_rad_s:
            return f"PHYSICS_UNSTABLE: client angular rate {w:.4f} rad/s > {cc.max_angular_rate_rad_s}"
        if self.state in MOVING_DOCKED and not self.task.docking.is_docked:
            return "PHYSICS_UNSTABLE: MEP <-> client docking joint lost"
        return None

    ## States
    def step_moving(self, s: State):
        if s == State.CLIENT_RELEASE:
            self.step_client_release()
        elif s == State.CLIENT_CRUISE:
            self.step_client_cruise()
        elif s == State.CHASE:
            self.step_chase()
        elif s == State.VELOCITY_MATCHING:
            self.step_velocity_matching()
        elif s == State.RENDEZVOUS:
            self.step_rendezvous()
        elif s == State.STABILIZING:
            self.step_stabilizing()
        elif s == State.STOPPING:
            self.step_stopping()
        elif s == State.ROBOT_RELEASE:
            self.step_robot_release()
        elif s == State.ARM_RETREAT:
            self.step_arm_retreat()
        elif s == State.MRV_SEPARATION:
            self.step_mrv_separation()

    def client_release_thrust(self, elapsed_s: float) -> Optional[bool]:
        """One step of the client release: thrust (F = m a, tau = I alpha) towards the
        drift velocity. True once released (the thrust is off from then on), False while
        thrusting, None after `release_timeout_s` (the caller fails the run)."""
        cc = self.cfg.client
        m = self._mvm
        v_err = cc.velocity() - m["client_v_px"]  # free body: the PhysX velocity is exact
        w_err = cc.angular_velocity() - m["client_w"]
        if float(np.linalg.norm(v_err)) <= cc.velocity_tolerance_mps and float(np.linalg.norm(w_err)) <= cc.angular_tolerance_rad_s:
            self.apply_wrench("client", np.zeros(3), np.zeros(3))
            self._mv["release"] = {"time_s": elapsed_s, "sim_time_s": self.sim_time, "velocity_mps": m["client_v_px"].tolist()}
            print(f"[MOVE] client released: v {np.round(m['client_v_px'], 5).tolist()} m/s after {elapsed_s:.1f} s of thrust, "
                  f"now coasting (no force)", flush=True)
            return True
        if elapsed_s > cc.release_timeout_s:
            return None
        mass, inertia = self._body_mass_inertia(self.task._satellite)
        a = bounded_accel(v_err, cc.release_time_constant_s, cc.release_accel_mps2)
        alpha = bounded_accel(w_err, cc.release_time_constant_s, cc.release_ang_accel_rad_s2)
        self.apply_wrench("client", mass * a, inertia @ alpha)
        return False

    def step_client_release(self):
        """Sequential mode (`rendezvous.concurrent_docking: false`): thrust the client up to
        its drift velocity with the arm holding, then CLIENT_CRUISE / CHASE / matching.
        Timeout: CLIENT_RELEASE_FAILED."""
        cc = self.cfg.client
        self.hold()
        done = self.client_release_thrust(self.state_time)
        if done:
            return self.goto(State.CLIENT_CRUISE if cc.cruise_check_s > 0.0 else self.after_cruise_state())
        if done is None:
            self.goto(State.CLIENT_RELEASE_FAILED, f"client velocity {np.round(self._mvm['client_v_px'], 5).tolist()} m/s did not reach "
                                                   f"{cc.linear_velocity_mps} within {cc.release_timeout_s:.0f} s")

    def concurrent_release_step(self):
        """Concurrent mode: the release runs next to the docking transport (no waiting)."""
        if "release" in self._mv or self.state in TERMINAL:
            return
        done = self.client_release_thrust(self.sim_time - float(self._mv["start_time_s"]))
        if done is None:
            cc = self.cfg.client
            self.goto(State.CLIENT_RELEASE_FAILED, f"client velocity {np.round(self._mvm['client_v_px'], 5).tolist()} m/s did not reach "
                                                   f"{cc.linear_velocity_mps} within {cc.release_timeout_s:.0f} s")

    def insertion_gates(self, mep_speed: bool = True) -> Tuple[bool, List[str]]:
        """Concurrent mode: what VELOCITY_MATCHING + RENDEZVOUS checked, now checked right
        before the docking-axis approach (the transport does not need it, the insertion
        does): client released, both relative speeds and the MRV station error small."""
        rv = self.cfg.rendezvous
        m = self._mvm
        if "release" not in self._mv:
            return False, ["client release still thrusting"]
        ok, bad = matched(m["v_rel_mrv"], m["v_rel_mep"] if mep_speed else np.zeros(3),
                          rv.max_mrv_client_relative_velocity_mps, rv.max_mep_client_relative_velocity_mps)
        e = float(np.linalg.norm(m["e_mrv"]))
        if e > rv.max_relative_position_m:
            ok = False
            bad.append(f"MRV station error {e*1000:.0f} mm > {rv.max_relative_position_m*1000:.0f} mm")
        if ok and "matched" not in self._mv:
            rec = {"time_s": self.sim_time - float(self._mv["start_time_s"]), "sim_time_s": self.sim_time,
                   "v_rel_mrv_mps": self._speed(m["v_rel_mrv"]), "v_rel_mep_mps": self._speed(m["v_rel_mep"]),
                   "client_speed_mps": self._speed(m["client_v"]), "mrv_speed_mps": self._speed(m["mrv_v"]),
                   "mep_speed_mps": self._speed(m["mep_v"]), "station_error_m": e, "mep_relative_speed_gated": mep_speed}
            self._mv["matched"] = rec
            self._mv["rendezvous"] = dict(rec)
            print(f"[MOVE] relative velocity / station gates passed before insertion: MRV-client {rec['v_rel_mrv_mps']*1000:.2f}, "
                  f"MEP-client {rec['v_rel_mep_mps']*1000:.2f} mm/s, station error {e*1000:.1f} mm "
                  f"(world speed {rec['client_speed_mps']*1000:.1f} mm/s)", flush=True)
        return ok, bad

    def step_client_cruise(self):
        """Watch the free drift (Test 1): constant velocity, no gravity, no instability."""
        cc = self.cfg.client
        self.hold()
        m = self._mvm
        cr = self._mv.setdefault("cruise", {"t0": self.sim_time, "p0": m["client_p"].tolist(), "v0": m["client_v_px"].tolist(),
                                            "max_dv_mps": 0.0, "max_dvz_mps": 0.0})
        dv = m["client_v_px"] - np.asarray(cr["v0"])
        cr["max_dv_mps"] = max(cr["max_dv_mps"], float(np.linalg.norm(dv)))
        cr["max_dvz_mps"] = max(cr["max_dvz_mps"], abs(float(dv[2])))
        if self.state_time < cc.cruise_check_s:
            return
        T = self.sim_time - cr["t0"]
        disp = m["client_p"] - np.asarray(cr["p0"])
        expected = np.asarray(cr["v0"]) * T
        cr.update({"duration_s": T, "displacement_m": disp.tolist(), "expected_displacement_m": expected.tolist(),
                   "displacement_error_m": float(np.linalg.norm(disp - expected)),
                   "final_velocity_mps": m["client_v_px"].tolist()})
        ok = (cr["max_dv_mps"] <= cc.cruise_max_velocity_change_mps
              and float(np.linalg.norm(m["client_v_px"] - cc.velocity())) <= 2.0 * cc.velocity_tolerance_mps
              and cr["displacement_error_m"] <= 0.01 * max(1e-3, float(np.linalg.norm(expected))) + 1e-3)
        cr["pass"] = ok
        print(f"[MOVE] client cruise {T:.1f} s: moved {np.round(disp, 4).tolist()} m (expected {np.round(expected, 4).tolist()}), "
              f"max |dv| {cr['max_dv_mps']*1000:.3f} mm/s, max |dv_z| {cr['max_dvz_mps']*1000:.3f} mm/s", flush=True)
        if ok:
            self.goto(self.after_cruise_state())
        else:
            self.goto(State.CLIENT_RELEASE_FAILED, f"client drift not constant: max |dv| {cr['max_dv_mps']*1000:.3f} mm/s, "
                                                   f"displacement error {cr['displacement_error_m']*1000:.1f} mm")

    def after_cruise_state(self) -> State:
        """CHASE only with `rendezvous.mrv_initial_velocity_mps` set (the spec's Test 2 setup)."""
        return State.CHASE if self.cfg.rendezvous.chase_enabled else State.VELOCITY_MATCHING

    def step_chase(self):
        """MRV accelerates (open loop) to `rendezvous.mrv_initial_velocity_mps`."""
        rv = self.cfg.rendezvous
        self.hold()
        target = rv.mrv_initial_velocity()
        if float(np.linalg.norm(self._mv_ctrl.v - target)) < 1e-6:
            m = self._mvm
            self._mv["chase_end"] = {"time_s": self.sim_time, "client_v": m["client_v"].tolist(), "mrv_v": m["mrv_v"].tolist(),
                                     "mep_v": None if m["mep_v"] is None else m["mep_v"].tolist(),
                                     "station_error_m": float(np.linalg.norm(m["e_mrv"]))}
            print(f"[MOVE] chase: MRV at {np.round(m['mrv_v'], 4).tolist()} m/s, client {np.round(m['client_v'], 4).tolist()} m/s, "
                  f"MEP {np.round(m['mep_v'], 4).tolist() if m['mep_v'] is not None else 'n/a'} m/s -> closed-loop matching", flush=True)
            self.goto(State.VELOCITY_MATCHING)
        elif self.state_time > rv.chase_timeout_s:
            self.goto(State.VELOCITY_MATCH_TIMEOUT, f"the MRV did not reach its chase velocity within {rv.chase_timeout_s:.0f} s")

    def step_velocity_matching(self):
        """|v_mrv - v_client| and |v_mep - v_client| below their limits for `stable_duration_sec`."""
        rv = self.cfg.rendezvous
        self.hold()
        m = self._mvm
        ok, bad = matched(m["v_rel_mrv"], m["v_rel_mep"], rv.max_mrv_client_relative_velocity_mps, rv.max_mep_client_relative_velocity_mps)
        if self._mv_timer.update(ok, self.sim_time) >= rv.stable_duration_sec:
            self._mv["matched"] = {"time_s": self.state_time, "sim_time_s": self.sim_time,
                                   "v_rel_mrv_mps": self._speed(m["v_rel_mrv"]), "v_rel_mep_mps": self._speed(m["v_rel_mep"]),
                                   "client_speed_mps": self._speed(m["client_v"]), "mrv_speed_mps": self._speed(m["mrv_v"]),
                                   "mep_speed_mps": self._speed(m["mep_v"])}
            print(f"[MOVE] velocity matched after {self.state_time:.1f} s: |v_mrv - v_client| {self._speed(m['v_rel_mrv'])*1000:.2f} mm/s, "
                  f"|v_mep - v_client| {self._speed(m['v_rel_mep'])*1000:.2f} mm/s (world speeds: client {self._speed(m['client_v'])*1000:.1f}, "
                  f"MRV {self._speed(m['mrv_v'])*1000:.1f}, MEP {self._speed(m['mep_v'])*1000:.1f} mm/s)", flush=True)
            self.goto(State.RENDEZVOUS)
        elif self.state_time > rv.velocity_match_timeout_s:
            self.goto(State.VELOCITY_MATCH_TIMEOUT, "; ".join(bad) or "not held long enough")

    def step_rendezvous(self):
        """Station reached: |e_mrv| and both relative speeds within limits, held."""
        rv = self.cfg.rendezvous
        self.hold()
        m = self._mvm
        ok, bad = matched(m["v_rel_mrv"], m["v_rel_mep"], rv.max_mrv_client_relative_velocity_mps, rv.max_mep_client_relative_velocity_mps)
        e = float(np.linalg.norm(m["e_mrv"]))
        if e > rv.max_relative_position_m:
            ok = False
            bad.append(f"MRV station error {e*1000:.0f} mm > {rv.max_relative_position_m*1000:.0f} mm")
        if self._mv_timer.update(ok, self.sim_time) >= rv.stable_duration_sec:
            self._mv["rendezvous"] = {"time_s": self.state_time, "sim_time_s": self.sim_time, "station_error_m": e,
                                      "v_rel_mrv_mps": self._speed(m["v_rel_mrv"]), "v_rel_mep_mps": self._speed(m["v_rel_mep"])}
            print(f"[MOVE] rendezvous: MRV station error {e*1000:.1f} mm, relative speeds MRV {self._speed(m['v_rel_mrv'])*1000:.2f} / "
                  f"MEP {self._speed(m['v_rel_mep'])*1000:.2f} mm/s -> docking (existing states, client moving at "
                  f"{self._speed(m['client_v'])*1000:.1f} mm/s)", flush=True)
            self.begin_docking()
            self.goto(State.DOCK_TARGET_ACQUIRE)
        elif self.state_time > rv.rendezvous_timeout_s:
            self.goto(State.RENDEZVOUS_TIMEOUT, "; ".join(bad) or "not held long enough")

    def forced_insertion(self) -> bool:
        return self._mv_active and self.cfg.rendezvous.forced_insertion

    def forced_corridor(self, m) -> Tuple[bool, List[str]]:
        """Wider insertion corridor of the forced insertion + the nozzle-wall limit."""
        rv, d = self.cfg.rendezvous, self.cfg.docking
        if m["insertion_depth"] > 0.0:
            clearance = m["clearance"]
        else:  # before the exit plane: the gap to the exit rim
            clearance = self.geo.nozzle.exit_radius - self.geo.probe.tip_radius - m["lateral"]
        return moving_dock.insertion_corridor_ok(m["lateral"], m["axis_deg"], clearance, rv.insertion_max_lateral_m,
                                                 rv.insertion_max_axis_deg, d.min_wall_clearance_m)

    def forced_alignment_check(self, m):
        """Forced insertion: start the axis approach as soon as the probe is inside the
        wider corridor and the client / MRV side is ready -- no settle wait, no depth
        calibration wait, no MEP relative-speed wait (the arm + payload swing is what
        the corridor tolerates; DOCK_READY still checks the relative speed)."""
        d = self.cfg.docking
        ok, bad = self.forced_corridor(m)
        if ok and self.cfg.rendezvous.concurrent_docking:
            ok_mv, bad_mv = self.insertion_gates(mep_speed=False)
            ok, bad = ok and ok_mv, bad + bad_mv
        if self._depth_offset is None and d.auto_calibrate and d.depth_enabled:
            self._calibrate_depth(m)  # opportunistic, on-axis samples only
        if not ok:
            self._dock_align_since = None
            return self._dock_stage_timeout(m)
        self._dock_align_since = self._dock_align_since if self._dock_align_since is not None else self.sim_time
        if self.sim_time - self._dock_align_since >= d.align_hold_s:
            self.results.check("[DOCK2] Alignment before the docking-axis approach", True,
                               f"forced insertion corridor: lateral {m['lateral']*1000:.2f} mm (<= {self.cfg.rendezvous.insertion_max_lateral_m*1000:.0f}), "
                               f"axis {m['axis_deg']:.3f} deg (<= {self.cfg.rendezvous.insertion_max_axis_deg:g}), held {d.align_hold_s:.1f} s")
            print(f"[DOCK] forced insertion: starting the docking-axis approach (lateral {m['lateral']*1000:.1f} mm, "
                  f"axis {m['axis_deg']:.2f} deg, remaining {m['geometry_distance']:.3f} m)", flush=True)
            self.goto(State.Z_APPROACH)
        else:
            self._dock_stage_timeout(m)

    def moving_dock_gates(self) -> Tuple[bool, List[str]]:
        """Extra DOCK_READY conditions while the client moves (relative quantities only)."""
        rv = self.cfg.rendezvous
        m = self._mvm
        bad = []
        s_mrv, s_mep = self._speed(m["v_rel_mrv"]), self._speed(m["v_rel_mep"])
        e = float(np.linalg.norm(m["e_mrv"]))
        if not s_mrv <= rv.dock_max_mrv_client_relative_velocity_mps:
            bad.append(f"MRV-client relative velocity {s_mrv*1000:.2f} mm/s > {rv.dock_max_mrv_client_relative_velocity_mps*1000:.1f}")
        if not s_mep <= rv.dock_max_mep_client_relative_velocity_mps:
            bad.append(f"MEP-client relative velocity {s_mep*1000:.2f} mm/s > {rv.dock_max_mep_client_relative_velocity_mps*1000:.1f}")
        if not e <= rv.dock_max_relative_position_m:
            bad.append(f"MRV station error {e*1000:.0f} mm > {rv.dock_max_relative_position_m*1000:.0f}")
        return not bad, bad

    def record_moving_dock(self):
        """Test 3 evidence at the instant the docking joint is created."""
        m = self._mvm
        self._mv["dock"] = {
            "sim_time_s": self.sim_time, "docked": bool(self.task.docking.is_docked),
            "client_speed_mps": self._speed(m["client_v"]), "mep_speed_mps": self._speed(m["mep_v"]),
            "mrv_speed_mps": self._speed(m["mrv_v"]),
            "v_rel_mep_mps": self._speed(m["v_rel_mep"]), "v_rel_mrv_mps": self._speed(m["v_rel_mrv"]),
            "station_error_m": float(np.linalg.norm(m["e_mrv"])),
            "dock_position_error_m": m["dock_position_error"], "dock_orientation_error_deg": m["dock_orientation_error"],
        }
        print(f"[MOVE] docked while moving: world speeds client {self._mv['dock']['client_speed_mps']*1000:.1f} / MEP "
              f"{self._mv['dock']['mep_speed_mps']*1000:.1f} / MRV {self._mv['dock']['mrv_speed_mps']*1000:.1f} mm/s, relative "
              f"MEP-client {self._mv['dock']['v_rel_mep_mps']*1000:.2f} / MRV-client {self._mv['dock']['v_rel_mrv_mps']*1000:.2f} mm/s", flush=True)

    def _docked_drift(self) -> Tuple[float, float]:
        """MEP-client relative pose change since docking [m], [deg]."""
        rel_now = self.probe_world().inv() @ self.sat_frame()
        return pose_errors(rel_now, self._dock["rel_at_dock"])

    def step_stabilizing(self):
        """DOCKED -> the stack keeps drifting for `stabilization_sec`; the joint must hold."""
        pd = self.cfg.post_docking
        self.hold()
        st = self._mv.setdefault("stabilizing", {"max_drift_mm": 0.0, "max_drift_deg": 0.0})
        pe, ae = self._docked_drift()
        st["max_drift_mm"] = max(st["max_drift_mm"], pe * 1000.0)
        st["max_drift_deg"] = max(st["max_drift_deg"], ae)
        if self.state_time < pd.stabilization_sec:
            return
        ok = self.task.docking.is_docked and st["max_drift_mm"] < pd.max_drift_mm and st["max_drift_deg"] < pd.max_drift_deg
        st["pass"] = ok
        if ok:
            self.goto(State.STOPPING)
        else:
            self.goto(State.DOCK_FAILED, f"docked stack not stable: drift {st['max_drift_mm']:.2f} mm / {st['max_drift_deg']:.3f} deg")

    def step_stopping(self):
        """Common deceleration of client and MEP (F = m a), the MRV follows its station."""
        pd = self.cfg.post_docking
        self.hold()
        m = self._mvm
        a = bounded_accel(-m["client_v"], pd.stop_time_constant_s, pd.stop_accel_mps2)
        alpha = bounded_accel(-m["client_w"], pd.stop_ang_time_constant_s, pd.stop_ang_accel_rad_s2)
        m_c, i_c = self._body_mass_inertia(self.task._satellite)
        m_m, i_m = self._body_mass_inertia(self.mep)
        self.apply_wrench("client", m_c * a, i_c @ alpha)
        self.apply_wrench("mep", m_m * a, i_m @ alpha)
        speeds = [self._speed(m["client_v"]), self._speed(m["mep_v"]), self._speed(m["mrv_v"])]
        rates = [float(np.linalg.norm(m["client_w"]))]
        ok = stopped(speeds, rates, pd.stop_velocity_tolerance_mps, pd.stop_angular_tolerance_rad_s)
        if self._mv_timer.update(ok, self.sim_time) >= pd.stop_hold_sec:
            self.clear_wrenches()
            pe, ae = self._docked_drift()
            self._mv["stop"] = {"time_s": self.state_time, "client_speed_mps": speeds[0], "mep_speed_mps": speeds[1],
                                "mrv_speed_mps": speeds[2], "client_rate_rad_s": rates[0],
                                "mep_rate_physx_rad_s": float(np.linalg.norm(m["mep_w"])),
                                "docked": bool(self.task.docking.is_docked), "drift_mm": pe * 1000.0, "drift_deg": ae}
            print(f"[MOVE] stopped after {self.state_time:.1f} s: client {speeds[0]*1000:.3f}, MEP {speeds[1]*1000:.3f}, "
                  f"MRV {speeds[2]*1000:.3f} mm/s, client rate {rates[0]:.5f} rad/s; docked {self.task.docking.is_docked}", flush=True)
            self.goto(State.ROBOT_RELEASE)
        elif self.state_time > pd.stop_timeout_s:
            self.goto(State.STOP_FAILED, f"speeds {[round(s * 1000, 3) for s in speeds]} mm/s, client rate {rates[0]:.5f} rad/s "
                                         f"after {pd.stop_timeout_s:.0f} s")

    def step_robot_release(self):
        """Remove ONLY the robot <-> MEP capture joint; the MEP <-> client joint stays."""
        pd = self.cfg.post_docking
        self.hold()
        if "released_at" not in self._mv:
            self._mv["released_at"] = self.sim_time
            self.capture.release([0])  # removes `{ENV}/capture_joint` (CaptureManager)
            print(f"[MOVE] robot <-> MEP attachment released ({self._mv['capture_joint']}); "
                  f"MEP <-> client joint {self._mv['docking_joint']} kept (present: {self.task.docking.is_docked})", flush=True)
        if self.state_time < pd.release_wait_sec:
            return
        attached, docked = self.capture.is_attached(0), bool(self.task.docking.is_docked)
        pe, ae = self._docked_drift()
        ok = not attached and docked
        self._mv["robot_release"] = {"robot_mep_attached": attached, "mep_client_docked": docked,
                                     "drift_mm": pe * 1000.0, "drift_deg": ae, "wait_s": self.state_time}
        if ok:
            if self.cfg.post_docking.immediate_release:
                self.start_stack_stop()
                self.goto(State.MRV_SEPARATION)
            else:
                self.goto(State.ARM_RETREAT)
        elif self.state_time > pd.release_timeout_s:
            self.goto(State.ROBOT_RELEASE_FAILED, f"robot<->MEP attached {attached}, MEP<->client docked {docked}")

    def step_arm_retreat(self):
        """Back the EE off the MEP face along its normal before the MRV moves at all."""
        sp = self.cfg.separation
        rt = self._mv.get("retreat_state")
        if rt is None:
            # The EE contact face is the controlled frame again (it was the probe tip)
            self.arm.set_tool(self.geo.ee_contact_link)
            start = self.ee_pose()
            normal = self.mep_grasp_normal()  # out of the MEP face, towards the arm
            rt = {"start": start, "normal": normal, "gap0": self._ee_face_gap(normal)}
            self._mv["retreat_state"] = rt
            self.ref = start
            self.q_hold = None
            print(f"[MOVE] arm retreat: {sp.arm_retreat_distance_m:.2f} m along the MEP face normal "
                  f"{np.round(normal, 3).tolist()} (EE-face gap now {rt['gap0']*1000:.0f} mm)", flush=True)
        goal = Frame(rt["start"].pos + sp.arm_retreat_distance_m * rt["normal"], rt["start"].rot)
        self.goal = goal
        self.track(goal, sp.arm_retreat_speed_mps, None)
        pe, ae = pose_errors(self.ee_pose(), goal)
        gap = self._ee_face_gap(rt["normal"])
        if pe < 0.01 and ae < 1.0 and self.state_time > sp.arm_retreat_distance_m / sp.arm_retreat_speed_mps:
            self.q_hold = None
            self.hold()
            self._mv["retreat"] = {"time_s": self.state_time, "gap_start_m": rt["gap0"], "gap_end_m": gap,
                                   "docked": bool(self.task.docking.is_docked)}
            print(f"[MOVE] arm retreated: EE-face gap {rt['gap0']*1000:.0f} -> {gap*1000:.0f} mm, MEP still docked "
                  f"{self.task.docking.is_docked}", flush=True)
            self.goto(State.MRV_SEPARATION)
        elif self.state_time > sp.arm_retreat_timeout_s:
            self.goto(State.ARM_RETREAT_FAILED, f"EE {pe*1000:.0f} mm / {ae:.2f} deg from the retreat pose after {sp.arm_retreat_timeout_s:.0f} s")

    def _ee_face_gap(self, normal: np.ndarray) -> float:
        """EE contact face distance in front of the MEP grasp face, along its normal [m]."""
        face = self.mep_frame() @ self.geo.mep_grasp
        return float((self.ee_pose().pos - face.pos) @ normal)

    def step_mrv_separation(self):
        """The MRV departs opposite the client's original drift.

        Velocity-controlled from whatever the MRV is doing when it starts: with the
        immediate release it is still moving with the client, so it first decelerates and
        reverses (acceleration-limited), cruises `duration_sec` at `velocity_mps` along
        the departure direction, then stops. The EE leaves the MEP face along its normal
        (the departure direction is the approach axis reversed)."""
        sp = self.cfg.separation
        m = self._mvm
        dist = float(np.linalg.norm(m["mrv_p"] - m["client_p"]))
        sep = self._mv.get("separation")
        if sep is None:
            sep = {"dist0_m": dist, "t0": self.sim_time, "max_increase_m": 0.0, "max_decrease_m": 0.0, "phase": "accelerate",
                   "direction": self._mv_sep_dir.tolist(), "max_drift_mm": 0.0,
                   "mrv_velocity_at_start_mps": self._mv_ctrl.v.tolist()}
            self._mv["separation"] = sep
            print(f"[MOVE] MRV separation along {np.round(self._mv_sep_dir, 3).tolist()} (= -client drift direction): "
                  f"from {np.round(self._mv_ctrl.v * 1000, 1).tolist()} mm/s to {sp.velocity_mps*1000:.0f} mm/s, "
                  f"{sp.duration_sec:.0f} s cruise, then stop", flush=True)
        self.arm_stow_step()
        v_cruise = self._mv_sep_dir * sp.velocity_mps
        if sep["phase"] == "accelerate" and float(np.linalg.norm(self._mv_ctrl.v - v_cruise)) < 1e-6:
            sep["phase"], sep["cruise_t0"] = "cruise", self.sim_time
        if sep["phase"] == "cruise" and self.sim_time - sep["cruise_t0"] >= sp.duration_sec:
            sep["phase"] = "stop"
        self._mv["sep_v_target"] = np.zeros(3) if sep["phase"] == "stop" else v_cruise
        inc = dist - sep["dist0_m"]
        sep["max_increase_m"] = max(sep["max_increase_m"], inc)
        sep["max_decrease_m"] = max(sep["max_decrease_m"], sep["max_increase_m"] - inc)  # any approach back
        pe, _ = self._docked_drift()
        sep["max_drift_mm"] = max(sep["max_drift_mm"], pe * 1000.0)
        sep["increase_m"] = inc
        mrv_stopped = sep["phase"] == "stop" and float(np.linalg.norm(self._mv_ctrl.v)) < 1e-9
        stack_done = not self._mv.get("stack_stop_active", False)
        if mrv_stopped and stack_done:
            ok = (inc >= sp.min_distance_increase_m and sep["max_decrease_m"] <= 0.005 and self.task.docking.is_docked
                  and sep["max_drift_mm"] < self.cfg.post_docking.max_drift_mm)
            sep.update({"time_s": self.state_time, "docked": bool(self.task.docking.is_docked), "pass": ok,
                        "max_contact_n": float(self._mv.get("max_separation_contact_n", 0.0))})
            if ok:
                print(f"[MOVE] MISSION COMPLETE: MRV-client distance +{inc:.3f} m, MEP still docked (drift {sep['max_drift_mm']:.2f} mm)", flush=True)
                self.goto(State.SUCCESS)
            else:
                self.goto(State.SEPARATION_FAILED, f"distance +{inc:.3f} m (>= {sp.min_distance_increase_m}), came back "
                                                   f"{sep['max_decrease_m']*1000:.1f} mm, docked {self.task.docking.is_docked}, "
                                                   f"drift {sep['max_drift_mm']:.2f} mm")
        elif self.state_time > sp.timeout_s:
            self.goto(State.SEPARATION_FAILED, f"departure not finished within {sp.timeout_s:.0f} s "
                                               f"(phase {sep['phase']}, stack still stopping: {not stack_done})")

    def arm_stow_step(self):
        """MRV_SEPARATION (and the SUCCESS cruise after it): return the arm to the pose it
        started the run in -- folded with `mrv.enabled`, else the pipeline start pose.

        Joint space, q(t) = q0 + smoothstep((t - t0) / arm_stow_duration_s) (q_start - q0),
        t0 = separation start + `arm_stow_delay_s` (the EE first clears the MEP face while
        the MRV reverses). q_start is re-expressed within +-180 deg of q0 (continuous joints)."""
        sp = self.cfg.separation
        q_start = self._q_folded if self._q_folded is not None else self._q_deployed
        sep = self._mv.get("separation")
        if not sp.arm_stow or q_start is None or sep is None or self.sim_time - sep["t0"] < sp.arm_stow_delay_s:
            self.hold()
            return
        stow = self._arm_stow
        if stow is None:
            q0 = self.arm.joint_pos().clone()
            stow = {"t0": self.sim_time, "q0": q0, "q1": wrap_joint_target(q0, q_start), "done": False}
            self._arm_stow = stow
            span = math.degrees(float(torch.max(torch.abs(stow["q1"] - q0))))
            print(f"[MOVE] arm returning to its start pose over {sp.arm_stow_duration_s:.1f} s "
                  f"(largest joint travel {span:.1f} deg)", flush=True)
        u = (self.sim_time - stow["t0"]) / sp.arm_stow_duration_s
        q = joint_lerp(stow["q0"], stow["q1"], smoothstep(u))
        self.q_hold = q.clone()
        self.hold()
        if u >= 1.0 and not stow["done"]:
            stow["done"] = True
            err = math.degrees(float(torch.max(torch.abs(self.arm.joint_pos() - stow["q1"]))))
            self._mv["arm_stow"] = {"time_s": self.sim_time - stow["t0"], "max_joint_error_deg": err}
            print(f"[MOVE] arm back at its start pose (max joint error {err:.2f} deg)", flush=True)

    def start_stack_stop(self):
        """After the release: the docked MEP + client stack stops by thrust (background)."""
        self._mv["stack_stop_active"] = True
        self._mv["stack_stop_t0"] = self.sim_time
        print(f"[MOVE] MEP + client stack stopping (common deceleration, background) from "
              f"{self._speed(self._mvm['client_v'])*1000:.1f} mm/s", flush=True)

    def stack_stop_step(self):
        """Common deceleration a = clip(-v / tau, a_max) as F = m a on client AND MEP (docked,
        no longer held by the arm), until both are below the stop tolerance for `stop_hold_sec`."""
        pd = self.cfg.post_docking
        m = self._mvm
        a = bounded_accel(-m["client_v"], pd.stop_time_constant_s, pd.stop_accel_mps2)
        alpha = bounded_accel(-m["client_w"], pd.stop_ang_time_constant_s, pd.stop_ang_accel_rad_s2)
        m_c, i_c = self._body_mass_inertia(self.task._satellite)
        m_m, i_m = self._body_mass_inertia(self.mep)
        self.apply_wrench("client", m_c * a, i_c @ alpha)
        self.apply_wrench("mep", m_m * a, i_m @ alpha)
        speeds = [self._speed(m["client_v"]), self._speed(m["mep_v"])]
        rate = float(np.linalg.norm(m["client_w"]))
        ok = stopped(speeds, [rate], pd.stop_velocity_tolerance_mps, pd.stop_angular_tolerance_rad_s)
        held = self._mv.get("stack_stop_since")
        if not ok:
            self._mv["stack_stop_since"] = None
        elif held is None:
            self._mv["stack_stop_since"] = self.sim_time
        elif self.sim_time - held >= pd.stop_hold_sec:
            self.clear_wrenches()
            self._mv["stack_stop_active"] = False
            pe, ae = self._docked_drift()
            self._mv["stop"] = {"time_s": self.sim_time - self._mv["stack_stop_t0"], "client_speed_mps": speeds[0],
                                "mep_speed_mps": speeds[1], "mrv_speed_mps": math.nan, "client_rate_rad_s": rate,
                                "mep_rate_physx_rad_s": float(np.linalg.norm(m["mep_w"])),
                                "docked": bool(self.task.docking.is_docked), "drift_mm": pe * 1000.0, "drift_deg": ae}
            print(f"[MOVE] MEP + client stack stopped after {self._mv['stop']['time_s']:.1f} s: client {speeds[0]*1000:.3f}, "
                  f"MEP {speeds[1]*1000:.3f} mm/s (thrust off)", flush=True)
            return
        if self.sim_time - self._mv["stack_stop_t0"] > pd.stop_timeout_s:
            self.goto(State.STOP_FAILED, f"stack speeds {[round(x * 1000, 3) for x in speeds]} mm/s after {pd.stop_timeout_s:.0f} s")

    ## After every control step: MRV translation, thrust, telemetry
    def moving_after_step(self):
        if self.cfg.rendezvous.concurrent_docking:
            self.concurrent_release_step()
        s = self.state
        dt = self.dt
        ctrl = self._mv_ctrl
        m = self._mvm
        if self._mv.get("stack_stop_active") and s not in TERMINAL:
            self.stack_stop_step()
        immediate = self.cfg.post_docking.immediate_release
        if s == State.MRV_SEPARATION:
            ctrl.toward(self._mv.get("sep_v_target", np.zeros(3)), dt, a_max=self.cfg.separation.accel_mps2)
            self.mrv.apply(self.mrv.offset + ctrl.v * dt)
        elif s == State.SUCCESS:
            # MISSION COMPLETE required the MRV briefly at rest (`step_mrv_separation`,
            # verifies the departure distance / no drift back); it now resumes the same
            # cruise and keeps departing indefinitely -- SUCCESS no longer auto-ends the
            # run (see the state-hold branch below) so the Astrobee has a moving MRV to
            # follow (`AstrobeeCfg.follow_mrv_after_dock`). Ends only on a manual stop.
            ctrl.toward(self._mv_sep_dir * self.cfg.separation.velocity_mps, dt, a_max=self.cfg.separation.accel_mps2)
            self.mrv.apply(self.mrv.offset + ctrl.v * dt)
        else:
            if s == State.CHASE:
                ctrl.toward(self.cfg.rendezvous.mrv_initial_velocity(), dt)
            elif (s in MRV_TRACKS_CLIENT or (s in FAILURES and self.capture.is_attached(0))
                  or (s == State.ROBOT_RELEASE and immediate)
                  or (s in (State.CLIENT_RELEASE, State.CLIENT_CRUISE) and self.cfg.rendezvous.mrv_follow_during_release)):
                # Station keeping on the client: v = v_client + Kp e (relative position)
                ctrl.track(m["mrv_p"], m["client_p"], m["client_v"], self._mv_d_ref, dt)
            else:
                ctrl.toward(np.zeros(3), dt)
            self.mrv.apply(self.mrv.offset + ctrl.v * dt)
        if self.vfx is not None:
            # Plume only while the MRV accelerates, exhausting against the acceleration
            thrust = ctrl.a if float(np.linalg.norm(ctrl.a)) > 1e-4 else None
            try:
                self.vfx.update(None if thrust is None else -thrust, self.mrv.offset)
            except Exception as e:
                print(f"[MRV-VFX] update failed ({e}); the plume is switched off", flush=True)
                self.vfx = None
        lg = self.cfg.logging
        if lg.motion_trail and self.sim_time >= self._next_trail_t:
            self._next_trail_t = self.sim_time + lg.motion_trail_period_s - 1e-9
            for key, p in (("client", m["client_p"]), ("mep", m["mep_p"]), ("mrv", self.mrv.root_pos())):
                pts = self._trail[key]
                pts.append(np.asarray(p, dtype=float).copy())
                if len(pts) > int(lg.motion_trail_max_points):
                    del pts[1]  # keep the start point: it is the world-fixed reference
        if self.sim_time >= self._next_mv_log_t:
            self._next_mv_log_t = self.sim_time + 1.0 / self.cfg.logging.rate_hz - 1e-9
            self.log_moving_row()
            self.update_motion_hud()
            # (the docking states draw their own scene, `update_dock_visuals`, with these vectors)
            if self.cfg.logging.debug_draw and s not in DOCKING_STATES:
                self.update_moving_visuals()

    ## Telemetry / visualisation
    VELOCITY_ARROW_S = 100.0  # arrow length = v * this [s]: 0.02 m/s -> 2 m

    def moving_vectors(self) -> List[Tuple[np.ndarray, np.ndarray, tuple, float]]:
        """Debug-draw segments: body velocities, relative position, departure direction."""
        m = self._mvm
        if not m:
            return []
        k = self.VELOCITY_ARROW_S
        probe, dock = self.probe_world().pos, self.dock_world().pos
        out = []
        for p, v, col in ((dock, m["client_v"], (0.2, 1.0, 0.2, 1.0)),
                          (probe, m["mep_v"], (1.0, 0.55, 0.0, 1.0)),
                          (m["mrv_p"], m["mrv_v"], (0.2, 0.6, 1.0, 1.0))):
            if v is not None:
                out.append((p, p + k * np.asarray(v), col, 4))
        out.append((probe, dock, (1.0, 1.0, 1.0, 1.0), 2))  # MEP docking point -> client docking point
        if self._mv_sep_dir is not None and self.state in (State.ARM_RETREAT, State.MRV_SEPARATION, State.SUCCESS):
            out.append((m["mrv_p"], m["mrv_p"] + 3.0 * self._mv_sep_dir, (1.0, 0.2, 0.9, 1.0), 3))
        if self.cfg.logging.motion_trail:
            out += self.trail_segments()
        return out

    TRAIL_COLORS = {"client": (0.2, 1.0, 0.2), "mep": (1.0, 0.55, 0.0), "mrv": (0.2, 0.6, 1.0)}

    def trail_segments(self) -> List[Tuple[np.ndarray, np.ndarray, tuple, float]]:
        """Path of each body since the release (a tick every trail point) and a
        world-fixed cross at its start: the drift is read against the cross."""
        out = []
        for key, pts in self._trail.items():
            if not pts:
                continue
            rgb = self.TRAIL_COLORS[key]
            line, tick, cross = (*rgb, 0.8), (*rgb, 1.0), (*rgb, 1.0)
            for a, b in zip(pts[:-1], pts[1:]):
                out.append((a, b, line, 2))
            for p in pts[1:]:
                out.append((p - np.array([0.0, 0.0, 0.08]), p + np.array([0.0, 0.0, 0.08]), tick, 2))
            s = pts[0]
            for axis in np.eye(3):
                out.append((s - 0.5 * axis, s + 0.5 * axis, cross, 4))
            if len(pts) > 1:
                out.append((s, pts[-1], (*rgb, 0.35), 1))  # start -> now (total displacement)
        return out

    def open_motion_hud(self):
        """GUI window with the live speeds (visual only, never gates the run)."""
        if self.headless or not self.cfg.logging.motion_hud or self._hud is not None:
            return
        try:
            import omni.ui as ui

            self._hud = ui.Window("Docking monitor", width=560, height=360, position_x=700, position_y=60)
            with self._hud.frame:
                with ui.VStack(spacing=4):
                    self._hud_labels = [ui.Label("", height=24, style={"font_size": 18}) for _ in range(12)]
            print("[VIS] opened the 'Docking monitor' window", flush=True)
        except Exception as e:
            self._hud = None
            print(f"[VIS] speed window unavailable ({e}); the trail and the CSV still show the motion", flush=True)

    def update_motion_hud(self, dm: Optional[Dict[str, float]] = None):
        """Refresh the monitor window. Moving client: speeds + docking lines; static
        docking: `dm` (the `dock_metrics` of this step) only."""
        if self._hud is None or not self._hud_labels:
            return
        try:
            d = self.cfg.docking
            lines = [f"state      {self.state.value}   t = {self.sim_time:6.1f} s"]
            if self.astrobee is not None:
                lines.append(f"Astrobee |v| {self.astrobee.speed_mps:6.3f} m/s   "
                             f"rel. satellite {self.astrobee.rel_speed_mps:6.3f} m/s")
                if self.astrobee.assist is not None and self.astrobee.assist.phase is not None:
                    f = self._assist_force
                    lines.append(f"Astrobee assist {self.astrobee.assist.phase.replace('ASTROBEE_', '')}   "
                                 f"|F| {0.0 if f is None else float(np.linalg.norm(f)):5.2f} N")
            if self._mv_active and self._mvm:
                m = self._mvm
                moved = {k: (float(np.linalg.norm(v[-1] - v[0])) if len(v) > 1 else 0.0) for k, v in self._trail.items()}
                sep = self._mv.get("separation", {})
                lines += [
                    f"client  |v| {self._speed(m['client_v'])*1000:6.1f} mm/s   moved {moved['client']:6.2f} m",
                    f"MEP     |v| {self._speed(m['mep_v'])*1000:6.1f} mm/s   moved {moved['mep']:6.2f} m",
                    f"MRV     |v| {self._speed(m['mrv_v'])*1000:6.1f} mm/s   moved {moved['mrv']:6.2f} m",
                    f"relative  MEP-client {self._speed(m['v_rel_mep'])*1000:5.2f}   MRV-client {self._speed(m['v_rel_mrv'])*1000:5.2f} mm/s",
                    f"MRV station error {float(np.linalg.norm(m['e_mrv']))*1000:6.1f} mm   separation {float(sep.get('increase_m', 0.0)):+.3f} m",
                ]
                axis, lateral, roll, dist = m["dock_axis_deg"], m["dock_lateral_m"], m["dock_roll_deg"], m["dock_position_error"]
            elif dm is not None:
                sat = self.task._satellite.data
                lines += [
                    f"satellite |v| {float(torch.norm(sat.root_com_lin_vel_w[0]))*1000:6.1f} mm/s   "
                    f"MEP |v| {float(torch.norm(self.mep.data.root_com_lin_vel_w[0]))*1000:6.1f} mm/s",
                    f"relative probe-satellite {dm['rel_speed']*1000:6.1f} mm/s",
                    f"remaining insertion {dm['geometry_distance']:.3f} m   depth {dm['depth_distance']:.3f} m "
                    f"({'ok' if dm['depth_ok'] else 'HOLD'})",
                ]
                axis, lateral, roll = dm["axis_deg"], dm["lateral"], dm["roll_deg"]
                dist = float(math.hypot(dm["lateral"], dm["axial"]))
            else:
                ## MRV approach / capture phase (ground truth, display only)
                v_mep = float(torch.norm(self.mep.data.root_com_lin_vel_w[0]))
                w_mep = math.degrees(float(torch.norm(self.mep.data.root_com_ang_vel_w[0])))
                g = self.gt_capture_metrics()
                leg_v = self._leg.v if self._leg is not None else 0.0
                off = self.mrv.offset if self.mrv is not None else np.zeros(3)
                lines += [
                    f"MRV  |v| {leg_v*1000:6.1f} mm/s   offset from nominal {float(np.linalg.norm(off)):5.2f} m",
                    f"MEP  |v| {v_mep*1000:6.1f} mm/s   |w| {w_mep:5.2f} deg/s",
                    *([f"client |v| {self._speed(self._mvm.get('client_v_px'))*1000:6.1f} mm/s   moved "
                       f"{(float(np.linalg.norm(self._trail['client'][-1] - self._trail['client'][0])) if len(self._trail['client']) > 1 else 0.0):6.2f} m"]
                      if self._client_live else []),
                    f"EE -> Cylinder_01 {g['distance']:.3f} m   gap {g['gap']*1000:7.1f} mm",
                    f"lateral {g['lateral']*1000:6.1f} mm   angle {g['angle']:5.2f} deg   (capture <= {self.cfg.capture.max_lateral_m*1000:.0f} mm, "
                    f"{self.cfg.capture.max_angle_deg:g} deg)",
                    f"EE-MEP relative velocity {g['rel_vel']*1000:6.1f} mm/s   (capture <= {self.cfg.capture.max_relative_velocity_mps*1000:.0f})",
                    f"tags {len(self.last_vision.detections) if self.last_vision else 0}/4   "
                    f"robot-MEP {'ATTACHED' if self.capture.is_attached(0) else 'free'}",
                ]
                for k, label in enumerate(self._hud_labels):
                    label.text = lines[k] if k < len(lines) else ""
                return
            forced = self.forced_insertion()
            ab_ax = self.cfg.rendezvous.insertion_max_axis_deg if forced else d.approach_abort_axis_deg
            ab_lat = self.cfg.rendezvous.insertion_max_lateral_m if forced else d.approach_abort_lateral_m
            lines += [
                f"probe-nozzle axis {axis:5.2f} deg   (align <= {d.align_axis_deg:g}, abort > {ab_ax:g})",
                f"lateral {lateral*1000:6.1f} mm   (align <= {d.align_lateral_m*1000:.0f}, abort > {ab_lat*1000:.0f})   roll {roll:5.2f} deg",
                f"probe tip -> dock point {dist:.3f} m{'   [forced insertion]' if forced else ''}",
                f"robot-MEP {'ATTACHED' if self.capture.is_attached(0) else 'released'}   "
                f"MEP-client {'DOCKED' if self.task.docking.is_docked else 'free'}",
            ]
            for k, label in enumerate(self._hud_labels):
                label.text = lines[k] if k < len(lines) else ""
        except Exception as e:
            print(f"[VIS] monitor window update failed ({e}); closing it", flush=True)
            self._hud = None

    def update_moving_visuals(self):
        probe, dock = self.probe_world(), self.dock_world()
        self.vis.set_scene([], None, probe.pos, dock.pos, None, self.ee_pose().pos, None,
                           axes=[(probe, 0.5, 3, 1.0), (dock, 0.5, 3, 0.6)], vectors=self.moving_vectors())
        self.vis.show()

    def log_moving_row(self):
        m = self._mvm
        if not m:
            return

        def xyz(prefix, v):
            v = np.full(3, math.nan) if v is None else np.asarray(v, dtype=float)
            return {f"{prefix}_{a}": float(x) for a, x in zip("xyz", v)}

        rel_mrv = relative(m["mrv_p"], m["client_p"])
        dist = float(np.linalg.norm(rel_mrv))
        sep = self._mv.get("separation")
        row = {"run_id": self.label, "timestamp": round(self.sim_time, 4), "state": self.state.value,
               **xyz("client_position", m["client_p"]), **xyz("client_velocity", m["client_v"]),
               **xyz("client_physx_velocity", m["client_v_px"]), **xyz("client_angular_velocity", m["client_w"]),
               **xyz("mep_position", m["mep_p"]), **xyz("mep_velocity", m["mep_v"]),
               **xyz("mep_physx_velocity", m["mep_v_px"]), **xyz("mep_angular_velocity", m["mep_w"]),
               **xyz("mrv_position", m["mrv_p"]), **xyz("mrv_velocity", m["mrv_v"]),
               **xyz("mrv_measured_velocity", m["mrv_v_fd"]), **xyz("mrv_acceleration", m["mrv_a"]),
               **xyz("mep_client_relative_position", m["rel_mep_client"]),
               "mep_client_relative_position": float(np.linalg.norm(m["rel_mep_client"])),
               **xyz("mep_client_relative_velocity", m["v_rel_mep"]), "mep_client_relative_velocity": self._speed(m["v_rel_mep"]),
               **xyz("mrv_client_relative_position", rel_mrv), "mrv_client_relative_position": dist,
               **xyz("mrv_station_error", m["e_mrv"]), "mrv_station_error": float(np.linalg.norm(m["e_mrv"])),
               **xyz("mrv_client_relative_velocity", m["v_rel_mrv"]), "mrv_client_relative_velocity": self._speed(m["v_rel_mrv"]),
               "docking_position_error": m["dock_position_error"], "docking_orientation_error": m["dock_orientation_error"],
               "robot_mep_attached": int(self.capture.is_attached(0)), "mep_client_docked": int(self.task.docking.is_docked),
               "mrv_client_distance": dist,
               "separation_distance": (dist - sep["dist0_m"]) if sep else 0.0,
               **xyz("client_force", self._mv_force.get("client", np.zeros(3))), **xyz("mep_force", self._mv_force.get("mep", np.zeros(3))),
               "arm_contact_force": m["contact_n"]}
        row = {k: (round(float(v), 7) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
        self._mv_rows.append(row)
        if self._mv_csv is not None:
            self._mv_csv.writerow({k: row.get(k, "") for k in moving_dock.CSV_COLUMNS})

    def moving_status(self):
        m = self._mvm
        if not m:
            return
        print(f"[MOVE] t={self.sim_time:6.1f}s {self.state.value:<18} |v| client {self._speed(m['client_v'])*1000:5.1f} "
              f"MEP {self._speed(m['mep_v'])*1000:5.1f} MRV {self._speed(m['mrv_v'])*1000:5.1f} mm/s | rel.v MEP "
              f"{self._speed(m['v_rel_mep'])*1000:5.2f} MRV {self._speed(m['v_rel_mrv'])*1000:5.2f} mm/s | station err "
              f"{float(np.linalg.norm(m['e_mrv']))*1000:6.1f} mm | robot-MEP {'Y' if self.capture.is_attached(0) else 'n'} "
              f"MEP-client {'Y' if self.task.docking.is_docked else 'n'} | contact {m['contact_n']:.1f} N | "
              f"wall {time.time() - self.wall_start:.0f} s", flush=True)

    def _finish_moving(self):
        """Tests 1-6 of the moving-client scenario, judged on what was measured."""
        c = self.cfg
        r = self.results
        mv = self._mv
        r.metrics["moving"] = {k: v for k, v in mv.items() if k not in ("retreat_state",)}
        cr = mv.get("cruise")
        if c.client.cruise_check_s > 0.0:
            r.check("[MOVE1] Client free flight", bool(cr and cr.get("pass")),
                    (f"{cr['duration_s']:.1f} s coasting: max |dv| {cr['max_dv_mps']*1000:.3f} mm/s (<= {c.client.cruise_max_velocity_change_mps*1000:.1f}), "
                     f"max |dv_z| {cr['max_dvz_mps']*1000:.3f} mm/s, displacement error {cr['displacement_error_m']*1000:.2f} mm")
                    if cr and "duration_s" in cr else "not reached")
        else:  # not a failure and not a pass: the test did not run
            r.metrics["moving"]["test1"] = "skipped (client.cruise_check_s = 0); run with --set client.cruise_check_s=5.0"
        ce, mt = mv.get("chase_end"), mv.get("matched")
        if ce and mt:
            detail = (f"from client {self._speed(ce['client_v'])*1000:.1f} / MRV {self._speed(ce['mrv_v'])*1000:.1f} / "
                      f"MEP {self._speed(ce['mep_v'])*1000:.1f} mm/s to |v_mrv - v_client| {mt['v_rel_mrv_mps']*1000:.2f}, "
                      f"|v_mep - v_client| {mt['v_rel_mep_mps']*1000:.2f} mm/s in {mt['time_s']:.1f} s")
        elif mt:
            detail = (f"MRV kept its station from the release (no chase): |v_mrv - v_client| {mt['v_rel_mrv_mps']*1000:.2f}, "
                      f"|v_mep - v_client| {mt['v_rel_mep_mps']*1000:.2f} mm/s at world speed {mt['client_speed_mps']*1000:.1f} mm/s")
        else:
            detail = "not reached"
        r.check("[MOVE2] Velocity matching", bool(mt), detail)
        dk = mv.get("dock")
        v_cmd = float(np.linalg.norm(c.client.velocity()))
        ok3 = bool(dk and dk["docked"] and dk["client_speed_mps"] > 0.5 * v_cmd
                   and dk["v_rel_mep_mps"] <= c.rendezvous.dock_max_mep_client_relative_velocity_mps
                   and dk["v_rel_mrv_mps"] <= c.rendezvous.dock_max_mrv_client_relative_velocity_mps)
        r.check("[MOVE3] Moving docking", ok3,
                (f"FixedJoint created with the client moving at {dk['client_speed_mps']*1000:.1f} mm/s (world); relative "
                 f"MEP-client {dk['v_rel_mep_mps']*1000:.2f}, MRV-client {dk['v_rel_mrv_mps']*1000:.2f} mm/s" if dk else "not reached"))
        st = mv.get("stop")
        r.check("[MOVE4] Post-docking stop", bool(st and st["docked"]),
                (f"client {st['client_speed_mps']*1000:.3f}, MEP {st['mep_speed_mps']*1000:.3f}, MRV {st['mrv_speed_mps']*1000:.3f} mm/s "
                 f"(<= {c.post_docking.stop_velocity_tolerance_mps*1000:.1f}), client rate {st['client_rate_rad_s']:.5f} rad/s, "
                 f"MEP-client drift {st['drift_mm']:.2f} mm" if st else "not reached"))
        rr = mv.get("robot_release")
        r.check("[MOVE5] Robot release (robot<->MEP released, MEP<->client kept)",
                bool(rr and not rr["robot_mep_attached"] and rr["mep_client_docked"]),
                (f"capture joint present {rr['robot_mep_attached']}, docking joint present {rr['mep_client_docked']}, "
                 f"MEP-client drift {rr['drift_mm']:.2f} mm" if rr else "not reached"))
        sep = mv.get("separation")
        r.check("[MOVE6] MRV separation", bool(sep and sep.get("pass")),
                (f"MRV-client distance +{sep['increase_m']:.3f} m (>= {c.separation.min_distance_increase_m}), MEP docked "
                 f"{sep.get('docked')}, max arm contact {float(mv.get('max_separation_contact_n', 0.0)):.1f} N "
                 f"(<= {c.separation.max_contact_force_n:.0f}), MEP-client drift {sep['max_drift_mm']:.2f} mm" if sep and "increase_m" in sep else "not reached"))
        if self._mv_csv is not None:
            r.check("[MOVE] CSV generated", len(self._mv_rows) > 0, f"{len(self._mv_rows)} rows -> {self.moving_csv_path}")

    ## Depth calibration / logging
    def set_dock_ring_color(self, rgb):
        """Colour of the nozzle rim ring (state colour), only when the colour changes."""
        path = getattr(self.task, "dock_ring_path", None)
        if path is None or getattr(self, "_ring_rgb", None) == tuple(rgb):
            return
        self._ring_rgb = tuple(rgb)
        from pxr import Gf, Sdf, Usd, UsdShade

        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(path)):
            if prim.IsA(UsdShade.Shader):
                sh = UsdShade.Shader(prim)
                for name in ("diffuseColor", "emissiveColor"):
                    sh.CreateInput(name, Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))

    def frame_docking_view(self, close: bool):
        """GUI: point the viewport at the docking area. The startup view is aimed at the
        capture point (x ~ 5 m) while the docking happens around x ~ 15 m, y ~ 3 m, so
        without this every docking marker is outside the picture."""
        if self.headless:
            return
        try:
            probe, dock = self.probe_world(), self.dock_world()
            axis = dock.rot[:, 2]
            up = np.array([0.0, 0.0, 1.0])
            side = np.cross(axis, up)
            side = side / max(1e-9, float(np.linalg.norm(side)))
            if close:  # the last metres: nozzle and probe tip
                target = dock.pos - 0.6 * axis
                eye = target + 5.0 * side + 2.0 * up - 1.5 * axis
            else:      # the whole transport: from the tip to the nozzle
                target = 0.5 * (probe.pos + dock.pos)
                eye = target + (0.9 * float(np.linalg.norm(probe.pos - dock.pos)) + 4.0) * side + 3.0 * up
            self.sim.set_camera_view(eye=tuple(eye.tolist()), target=tuple(target.tolist()))
            print(f"[VIS] viewport moved to the docking area ({'close' if close else 'wide'} view)", flush=True)
        except Exception as e:
            print(f"[VIS] could not move the viewport: {e}", flush=True)

    def update_dock_visuals(self, m):
        """Debug draw for the docking phase: probe tip, SAT_DOCK_POINT, pre-dock point,
        both coordinate frames and the probe -> dock vector, colour-coded by state."""
        probe, dock = self.probe_world(), self.dock_world()
        pre = probe_dock.tip_goal(dock, self._pre_dock_axial())
        aligned = probe_dock.alignment_ok(self.cfg.docking, m)[0]
        if self.task.docking.is_docked:
            col = (0.1, 1.0, 0.2, 1.0)        # DOCKED
        elif self.state in (State.Z_APPROACH, State.FINAL_INSERTION, State.DOCK_READY):
            col = (1.0, 0.55, 0.0, 1.0)       # INSERTING
        elif aligned:
            col = (1.0, 1.0, 0.2, 1.0)        # READY
        else:
            col = (1.0, 0.2, 0.2, 1.0)        # NOT ALIGNED
        self.set_dock_ring_color(col[:3])
        axes = [(probe, 0.8, 5, 1.0), (dock, 0.8, 5, 0.6), (pre, 0.4, 2, 0.4)]
        vectors = [(probe.pos, dock.pos, col, 3)]
        if self._mv_active:
            vectors += self.moving_vectors()
        self.vis.set_scene([], pre.pos, probe.pos, dock.pos, None, self.ee_pose().pos, None,
                           axes=axes, vectors=vectors)
        self.vis.show()

    def dock_status(self, m):
        """HUD line: state, the dock-frame errors, both distances and the dock flags."""
        print(f"[DOCK] t={self.sim_time:6.1f}s {self.state.value:<20} "
              f"e_xyz {m['e_x']*1000:7.1f} {m['e_y']*1000:7.1f} {m['e_z']*1000:8.1f} mm | "
              f"rpy {m['roll_rpy_deg']:6.2f} {m['pitch_rpy_deg']:6.2f} {m['yaw_rpy_deg']:6.2f} deg | "
              f"axis {m['axis_deg']:5.2f} deg | geo {m['geometry_distance']:.3f} m depth {m['depth_distance']:.3f} m "
              f"({int(m['depth_pixels'])} px) | rel.v {m['rel_speed']*1000:5.1f} mm/s | "
              f"align {'Y' if probe_dock.alignment_ok(self.cfg.docking, m)[0] else 'n'} "
              f"depth {'Y' if m['depth_ok'] else 'n'} docked {'Y' if self.task.docking.is_docked else 'n'} | "
              f"wall {('--' if not math.isfinite(m['clearance']) else format(m['clearance']*1000, '.0f')):>5} mm | wall-clock {time.time() - self.wall_start:.0f} s", flush=True)

    def _calibrate_depth(self, m):
        """Measure the depth offset once, where the geometric distance is known.

        `docking.depth_surface_offset_m` assumes the ray lands on the nozzle back plate,
        but that plate is a collision-only prim, so the real surface is the thruster mesh
        behind it. Both values are logged so the assumption is visible.

        Only on-axis samples count (`probe_dock.depth_calibration_sample_ok`): the offset
        is the median of `depth_calibration_samples` of them.
        """
        d = self.cfg.docking
        ok, why = probe_dock.depth_calibration_sample_ok(d, m)
        if not ok:
            if self._dock.get("calib_wait_reason") != why.split(" ")[0]:
                self._dock["calib_wait_reason"] = why.split(" ")[0]
                print(f"[DOCK] depth calibration waiting for the docking axis: {why}", flush=True)
            return
        measured = m["depth_raw"] - self._cam_to_tip - m["geometry_distance"]
        # Fail-safe: a ray that missed the nozzle gives a nonsense offset. The surface
        # can only be between the dock point and the far end of the thruster.
        if not -0.5 <= measured <= 3.0:
            if not self._calib_rejected:
                self._calib_rejected = True
                print(f"[DOCK] depth calibration rejected: implied surface offset {measured:.2f} m is outside "
                      f"[-0.5, 3.0] m (raw {m['depth_raw']:.3f} m, geometry {m['geometry_distance']:.3f} m); "
                      f"the ray is not landing inside the nozzle", flush=True)
            return
        samples = self._dock.setdefault("calib_samples", [])
        samples.append(float(measured))
        if len(samples) < int(d.depth_calibration_samples):
            return
        self._depth_offset = float(np.median(samples))
        self._depth_calibrated_t = self.sim_time
        cfgd = self.cfg.docking.depth_surface_offset_m
        self.results.metrics.setdefault("docking", {})["depth_calibration"] = {
            "measured_surface_offset_m": self._depth_offset,
            "samples": len(samples),
            "sample_spread_m": float(np.ptp(samples)),
            "lateral_at_calibration_m": m["lateral"],
            "axis_at_calibration_deg": m["axis_deg"],
            "configured_surface_offset_m": cfgd,
            "backstop_gap_m": self.geo.backstop_gap,
            "geometry_distance_m": m["geometry_distance"],
            "depth_raw_m": m["depth_raw"],
            "camera_to_tip_m": self._cam_to_tip,
        }
        print(f"[DOCK] depth calibration at the pre-dock pose (on the axis: lateral {m['lateral']*1000:.1f} mm, axis {m['axis_deg']:.2f} deg; "
              f"median of {len(samples)}, spread {np.ptp(samples)*1000:.1f} mm): raw {m['depth_raw']:.3f} m, geometry {m['geometry_distance']:.3f} m "
              f"-> surface offset {self._depth_offset:.3f} m (configured {cfgd:.3f} m, back plate at {self.geo.backstop_gap:.3f} m)", flush=True)

    def log_dock_row(self, m):
        row = {
            "run_id": self.label, "timestamp": round(self.sim_time, 4), "state": self.state.value,
            "commanded_speed_mps": float(self._dock.get("commanded_speed", 0.0)),
            "alignment_valid": int(probe_dock.alignment_ok(self.cfg.docking, m)[0]),
            "depth_valid": int(m["depth_ok"]), "dock_ready": int(self.state == State.DOCK_READY),
            "dock_success": int(self.task.docking.is_docked),
        }
        probe, dock = self.probe_world(), self.dock_world()
        for name, f in (("probe_tip", probe), ("sat_dock", dock)):
            row.update({f"{name}_{a}": float(v) for a, v in zip("xyz", f.pos)})
            row.update({f"{name}_q{a}": float(v) for a, v in zip("wxyz", f.quat)})
        row.update({
            "relative_x": m["e_x"], "relative_y": m["e_y"], "relative_z": m["e_z"],
            "roll_error_deg": m["roll_rpy_deg"], "pitch_error_deg": m["pitch_rpy_deg"], "yaw_error_deg": m["yaw_rpy_deg"],
            "orientation_error_deg": m["orientation_deg"], "axis_error_deg": m["axis_deg"],
            "roll_about_axis_deg": m["roll_deg"], "lateral_error_m": m["lateral"],
            "geometry_distance_m": m["geometry_distance"], "depth_distance_m": m["depth_distance"],
            "depth_raw_m": m["depth_raw"], "depth_pixels": m["depth_pixels"],
            "insertion_depth_m": m["insertion_depth"], "wall_clearance_m": m["clearance"],
            "relative_vx": m["rel_vx"], "relative_vy": m["rel_vy"], "relative_vz": m["rel_vz"],
            "relative_speed_mps": m["rel_speed"],
        })
        self._dock_rows.append(row)
        if self._dock_csv is not None:
            self._dock_csv.writerow({k: row.get(k, "") for k in probe_dock.CSV_COLUMNS})

    def publish_live_camera_images(self):
        """Publish wrist and probe RGB feeds independently of mission state."""
        if self.ros is None:
            return

        wrist_out = self.camera.data.output
        if self.cfg.ros.publish_image and "rgb" in wrist_out:
            wrist_rgb = wrist_out["rgb"][0].cpu().numpy()[..., :3]
            wrist_img = np.ascontiguousarray(downsample(wrist_rgb, self.ss))
            self.ros.publish_image(self.sim_time, wrist_img)

        if self._probe_cam is not None:
            probe_out = self._probe_cam.data.output
            if "rgb" in probe_out:
                probe_img = np.ascontiguousarray(
                    probe_out["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
                )
                self.ros.publish_probe_image(self.sim_time, probe_img)

    def save_probe_rgb(self):
        """One `cam_probe` RGB frame (the operator's view of the approach)."""
        pc = self.cfg.probe_camera
        if self._probe_cam is None or pc.rgb_every_sec <= 0.0:
            return
        out = self._probe_cam.data.output
        if "rgb" not in out:
            return
        import cv2

        img = np.ascontiguousarray(out["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))

        path = self._probe_dir / f"{self.sim_time:07.2f}s_{self.state.value}.png"
        cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self._probe_frames.append(path)

    def _goal_errors(self, goal: Frame) -> Tuple[float, float, float]:
        ee = self.ee_pose()
        n = goal.rot[:, 2]
        d = ee.pos - goal.pos
        axial = float(d @ n)
        return float(np.linalg.norm(d - axial * n)), math.degrees(rotation_angle(ee.rot, goal.rot)), axial

    def _settled(self, lat_tol: float, ang_tol: float, axial_tol: float, hold_s: float) -> bool:
        if self.goal is None:
            return False
        lat, ang, axial = self._goal_errors(self.goal)
        if lat < lat_tol and ang < ang_tol and abs(axial) < axial_tol:
            if self._settled_since is None:
                self._settled_since = self.sim_time
            return self.sim_time - self._settled_since >= hold_s
        self._settled_since = None
        return False

    ###############
    ### Logging ###
    ###############

    def log_row(self):
        gt = self.gt_cylinder()
        est = self.estimate_now()
        pred = self.predicted()
        ee = self.ee_pose()
        # MEP velocity at the EE contact point, as the GT columns (= v in translation_only)
        v_est, w_twist = self.est_mep_twist_at(ee.pos)
        # translation_only: Phase 1 monitoring rate (not used by the controller)
        w_est = w_twist if self.six_dof else self.predictor.angular_velocity()
        if w_est is None:
            w_est = np.full(3, math.nan)
        v_gt, w_gt = self.gt_mep_velocity_at(ee.pos)
        m = self.capture_metrics(est, v_est, w_twist)
        m_gt = self.capture_metrics(gt, v_gt, w_gt)
        lv = self.last_vision
        ee_w = self.ee_angular_velocity()
        tracking_ok, _ = self.tracking_ok()

        def quat(f: Optional[Frame], prefix: str) -> dict:
            q = f.quat if f is not None else (math.nan,) * 4
            return {f"{prefix}_q{k}": v for k, v in zip("wxyz", q)}
        base = self.base_frame()
        est_b = (base.inv() @ est).pos if est is not None else None
        g_rpy = rotmat_to_rpy_deg(gt.rot)
        e_rpy = rotmat_to_rpy_deg(est.rot) if est is not None else (math.nan,) * 3
        pe, ae = pose_errors(gt, est) if est is not None else (math.nan, math.nan)
        rpe, rae = pose_errors(gt, lv.t_w_y) if lv is not None and lv.valid else (math.nan, math.nan)
        drift = self.relative_drift() if self._rel_at_capture is not None else (math.nan, math.nan)
        row = {
            "timestamp": round(self.sim_time, 4),
            "gt_x": gt.pos[0], "gt_y": gt.pos[1], "gt_z": gt.pos[2],
            "gt_roll": g_rpy[0], "gt_pitch": g_rpy[1], "gt_yaw": g_rpy[2],
            "est_x": est.pos[0] if est else math.nan, "est_y": est.pos[1] if est else math.nan, "est_z": est.pos[2] if est else math.nan,
            "est_roll": e_rpy[0], "est_pitch": e_rpy[1], "est_yaw": e_rpy[2],
            # error of the system estimate (est_*) vs the GT at the same time
            "position_error_mm": pe * 1000.0, "angle_error_deg": ae,
            "pred_x": pred.pos[0] if pred else math.nan, "pred_y": pred.pos[1] if pred else math.nan, "pred_z": pred.pos[2] if pred else math.nan,
            "ee_x": ee.pos[0], "ee_y": ee.pos[1], "ee_z": ee.pos[2],
            "ee_target_x": self.goal.pos[0] if self.goal else math.nan,
            "ee_target_y": self.goal.pos[1] if self.goal else math.nan,
            "ee_target_z": self.goal.pos[2] if self.goal else math.nan,
            "ee_target_distance_m": float(np.linalg.norm(ee.pos - self.goal.pos)) if self.goal else math.nan,
            "relative_velocity_mps": m["rel_vel"],
            "mep_linear_velocity_x": v_est[0] if v_est is not None else math.nan,
            "mep_linear_velocity_y": v_est[1] if v_est is not None else math.nan,
            "mep_linear_velocity_z": v_est[2] if v_est is not None else math.nan,
            "mep_angular_velocity_x": w_est[0], "mep_angular_velocity_y": w_est[1], "mep_angular_velocity_z": w_est[2],
            "num_detected_tags": len(lv.detections) if lv else 0,
            "capture_state": self.state.value,
            "capture_success": int(self.capture.is_attached(0)),
            "total_elapsed_time": round(time.time() - self.wall_start, 3),
            "scenario": self.label,
            "est_age_s": self.estimate_age(),
            "reproj_rms_px": lv.reproj_rms if lv else math.nan,
            "pnp_inliers": lv.inliers if lv else 0,
            "raw_position_error_mm": rpe * 1000.0, "raw_angle_error_deg": rae,
            "est_base_x": est_b[0] if est_b is not None else math.nan,
            "est_base_y": est_b[1] if est_b is not None else math.nan,
            "est_base_z": est_b[2] if est_b is not None else math.nan,
            "gt_mep_linear_velocity_x": v_gt[0], "gt_mep_linear_velocity_y": v_gt[1], "gt_mep_linear_velocity_z": v_gt[2],
            "gt_mep_angular_velocity_deg_s": math.degrees(float(np.linalg.norm(w_gt))),
            "gt_relative_velocity_mps": m_gt["rel_vel"],
            "ee_to_cylinder_distance_m": m_gt["distance"],
            "ee_lateral_error_m": m_gt["lateral"],
            "ee_gap_m": m_gt["gap"],
            "ee_normal_angle_deg": m_gt["angle"],
            "standoff_cmd_m": self.standoff,
            "fixed_joint_valid": int(self.capture.is_attached(0)),
            "mep_ee_relative_drift_mm": drift[0] * 1000.0,
            "mep_ee_relative_drift_deg": drift[1],
            "motion_mode": self.cfg.mep.motion_mode,
            "prediction_horizon_s": self.cfg.prediction.horizon_sec,
            **quat(gt, "gt"), **quat(est, "est"), **quat(pred, "pred"), **quat(ee, "ee"), **quat(self.goal, "ee_target"),
            "gt_mep_angular_velocity_x": w_gt[0], "gt_mep_angular_velocity_y": w_gt[1], "gt_mep_angular_velocity_z": w_gt[2],
            "ee_angular_velocity_x": ee_w[0], "ee_angular_velocity_y": ee_w[1], "ee_angular_velocity_z": ee_w[2],
            "est_ee_to_cylinder_distance_m": m["distance"],
            "est_ee_lateral_error_m": m["lateral"],
            "est_ee_normal_angle_deg": m["angle"],
            "est_ee_orientation_error_deg": m["orientation"],
            "est_relative_angular_velocity_rad_s": m["rel_ang_vel"],
            "gt_ee_orientation_error_deg": m_gt["orientation"],
            "gt_relative_angular_velocity_rad_s": m_gt["rel_ang_vel"],
            "pnp_valid": int(bool(lv is not None and lv.valid)),
            "tracking_valid": int(tracking_ok),
        }
        row = {k: (round(float(v), 6) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()}
        if self._csv is not None:
            self._csv.writerow(row)
        self.rows.append(row)

    def update_visuals(self):
        lv = self.last_valid
        tags_w: List[np.ndarray] = []
        centre = est = None
        if lv is not None:
            t_w_t = lv.t_w_y @ self.t_y_t
            tags_w = [t_w_t.point(c) for c in tag_centres(self.cfg.apriltag).values()]
            centre, est = t_w_t.pos, lv.t_w_y.pos
        pred = self.predicted()
        ee = self.ee_pose()
        cc = self.task.cfg.capture
        attached = self.capture.is_attached(0)
        # Display only: once the MEP is held there is no target any more (`self.goal` stays at
        # its last approach value), so the target marker follows the EE instead of lagging it
        goal = ee if attached else self.goal
        if attached:
            col = cc.color_captured
        elif self.state in (State.SLOW_APPROACH, State.CAPTURE_ATTEMPT):
            col = cc.color_approach
        else:
            col = cc.color_idle
        if col != self._cylinder_color:  # capture cylinder: blue -> yellow (closing in) -> green (captured)
            self._cylinder_color = col
            self.capture.set_color(0, col)
        ## Orientation triads (X red, Y green, Z blue): estimated current Cylinder_01
        ## (bright, long), predicted (dim, long), GT (thin), EE and EE target (short)
        est_now = self.estimate_now()
        gt = self.gt_cylinder()
        axes = [(est_now, 0.35, 4, 1.0), (pred, 0.35, 4, 0.45), (gt, 0.25, 1, 1.0), (ee, 0.2, 2, 1.0), (goal, 0.2, 1, 0.6)]
        vectors = []
        v = self.predictor.velocity()
        if est_now is not None and v is not None:
            vectors.append((est_now.pos, est_now.pos + v * self.vis.VELOCITY_ARROW_SEC, "velocity", 3))
        if est_now is not None and pred is not None:
            vectors.append((est_now.pos, pred.pos, "pred", 2))  # future position prediction
        if self._client_live and self.cfg.logging.motion_trail:
            vectors += self.trail_segments()  # cleared before every image the tags are detected in
        self.vis.set_scene(tags_w, centre, est, gt.pos, pred.pos if pred else None, ee.pos, goal.pos if goal else None,
                           axes=axes, vectors=vectors)

    def save_overlay(self, vf: VisionFrame):
        """2D overlay on the `cam_wrist` image (OpenCV)."""
        import cv2

        img = cv2.cvtColor(vf.image, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        centers = []
        for tid in self.cfg.apriltag.ids:
            d = vf.detections.get(tid)
            if d is None:
                continue
            pts = d.corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
            cv2.circle(img, tuple(int(x) for x in d.corners[0]), 4, (0, 0, 255), -1)  # corner 0
            centers.append(d.center)
            cv2.putText(img, f"Tag {tid}", tuple(int(x) for x in d.corners[0] + np.array([0, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(centers) >= 2:
            cv2.polylines(img, [np.array(centers, dtype=np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        cv2.drawMarker(img, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 40, 1)
        cam_inv = vf.cam_w.inv()

        def proj(p_w):
            pc = cam_inv.point(p_w)
            if pc[2] <= 0.01:
                return None
            uv = project_points(pc, self.k)[0]
            return int(round(uv[0])), int(round(uv[1]))

        gt = proj(self.gt_cylinder().pos)
        if gt:
            cv2.circle(img, gt, 9, (0, 255, 0), 2)
        if vf.t_w_y is not None:
            e = proj(vf.t_w_y.pos)
            if e:
                cv2.drawMarker(img, e, (0, 140, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
        pred = self.predicted()
        if pred is not None:
            p = proj(pred.pos)
            if p:
                cv2.drawMarker(img, p, (255, 0, 255), cv2.MARKER_DIAMOND, 18, 2)

        def draw_axes(f: Optional[Frame], length: float, thickness: int, bright: float):
            """Local XYZ axes of `f` (X red, Y green, Z blue in BGR)."""
            if f is None:
                return
            o = proj(f.pos)
            for i, bgr in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                e = proj(f.pos + length * f.rot[:, i])
                if o and e:
                    cv2.line(img, o, e, tuple(int(bright * c) for c in bgr), thickness, cv2.LINE_AA)

        # current estimate (bright, thick) vs prediction (dim, thin): the rotation over
        # the horizon shows as the offset between the two triads
        draw_axes(self.estimate_now(), 0.12, 3, 1.0)
        draw_axes(pred, 0.12, 1, 0.6)
        err = ""
        if vf.valid:
            pe, ae = pose_errors(self.gt_cylinder(), vf.t_w_y)
            err = f" | err {pe*1000:.2f} mm {ae:.3f} deg | rms {vf.reproj_rms:.2f} px"
        lines = [
            f"t={vf.t:6.2f}s  {self.state.value}  tags {len(vf.detections)}/4{err}",
            "green O: Cylinder_01 GT   orange X: estimate   magenta <>: prediction (+%.1fs)" % self.cfg.prediction.horizon_sec,
            "axes XYZ=RGB: thick = estimate now, thin/dim = prediction   [%s]" % self.cfg.mep.motion_mode,
        ]
        for i, text in enumerate(lines):
            cv2.putText(img, text, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(img, text, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        path = self.overlay_dir / f"{self.label}_{vf.t:07.2f}s.png"
        cv2.imwrite(str(path), img)
        self._overlays.append(path)

    def publish_ros(self):
        """Telemetry on the ROS topics (`ros_interface.py`). Ground truth goes to `gt/` only."""
        ee = self.ee_pose()
        est, pred = self.estimate_now(), self.predicted()
        v, w = self.predictor.twist()
        gt = self.gt_cylinder()
        gt_v, gt_w = self.gt_mep_velocity_at(gt.pos)
        docking = self.state in DOCKING_STATES and self._dock_m_latest is not None
        probe, dock = (self.probe_world(), self.dock_world()) if docking else (None, None)
        self.ros.publish_poses(self.sim_time, est, pred, ee, self.goal, self.cam_pose(), v=v, w=w, gt=gt, gt_v=gt_v, gt_w=gt_w,
                               probe=probe, dock=dock)
        m = self.est_capture_metrics()
        # GT metrics are evaluation/monitoring values only.
        # They must never be used by the capture controller.
        m_gt = self.gt_capture_metrics()
        capture_target_gap = float(self.cfg.approach.final_gap_m)
        gt_capture_position_error = math.hypot(
            m_gt["gap"] - capture_target_gap,
            m_gt["lateral"],
        )
        gt_capture_remaining_distance = max(
            0.0,
            m_gt["gap"] - capture_target_gap,
        )
        dm = self._dock_m_latest if docking else None
        lv = self.last_vision
        ok, why = self.tracking_ok()
        self.ros.publish_state(self.sim_time, self.state.value, self.capture.is_attached(0), {
            "scenario": self.scenario, "motion_mode": self.cfg.mep.motion_mode,
            "tags": len(lv.detections) if lv else 0, "pose_valid": bool(lv.valid) if lv else False,
            "tracking_ok": ok, "tracking_reason": why, "standoff_m": self.standoff,
            "est_age_s": self.estimate_age(), "capture_enabled": self.ros.capture_enabled,
            "start_received": self.ros.start_received, "failure": self.results.failure,
            **{f"est_{k}": val for k, val in m.items()},

            # Ground-truth capture evaluation metrics.
            # Controller decisions continue to use only the est_* metrics above.
            "capture_target_gap_m": capture_target_gap,
            "gt_capture_position_error_m": gt_capture_position_error,
            "gt_capture_gap_m": m_gt["gap"],
            "gt_capture_lateral_error_m": m_gt["lateral"],
            "gt_capture_orientation_error_deg": m_gt["orientation"],
            "gt_relative_velocity_mps": m_gt["rel_vel"],
            "gt_relative_angular_velocity_rad_s": m_gt["rel_ang_vel"],
            "gt_capture_remaining_distance_m": gt_capture_remaining_distance,

            # Docking phase (probe tip -> SAT_DOCK_POINT); the est_* capture metrics above no longer apply then
            "dock_enabled": bool(self.cfg.docking.enabled),  # false: a capture-only run, mission success = capture success
            "dock_active": docking,
            **({"dock_distance": dm["geometry_distance"], "dock_lateral": dm["lateral"], "dock_axis_deg": dm["axis_deg"],
                "dock_roll_deg": dm["roll_deg"], "dock_rel_speed": dm["rel_speed"], "dock_insertion_depth": dm["insertion_depth"],
                "dock_clearance": dm["clearance"], "dock_docked": bool(self.task.docking.is_docked),
                # probe tip -> SAT_DOCK_POINT error in the dock frame [m] and the full orientation error
                "dock_relative_x": dm["e_x"], "dock_relative_y": dm["e_y"], "dock_relative_z": dm["e_z"],
                "dock_orientation_deg": dm["orientation_deg"]} if dm is not None else {}),
        })

    def status(self):
        if self.state in MRV_STATES:
            return self.mrv_status()
        if self._mv_active:
            return self.moving_status()
        lv = self.last_vision
        m = self.capture_metrics(self.gt_cylinder(), self.gt_mep_velocity_at(self.ee_pose().pos)[0])
        print(f"[STATUS] t={self.sim_time:6.1f}s {self.state.value:<16} tags {len(lv.detections) if lv else 0}/4 "
              f"standoff {self.standoff:.3f} m | GT: dist {m['distance']:.3f} m gap {m['gap']*1000:7.1f} mm lat {m['lateral']*1000:6.1f} mm "
              f"ang {m['angle']:.2f} deg rel.v {m['rel_vel']*1000:.1f} mm/s | wall {time.time() - self.wall_start:.0f} s", flush=True)

    def mrv_status(self):
        """Progress line of the rendezvous phase (no vision runs in it)."""
        off = self.mrv.offset if self.mrv is not None else np.zeros(3)
        leg = f"{self._leg.remaining:5.2f} m left at {self._leg.v:.2f} m/s" if self._leg is not None else "-"
        mep = self.gt_cylinder().pos - (self.task.mep_nominal @ self.t_m_y).pos
        print(f"[STATUS] t={self.sim_time:6.1f}s {self.state.value:<18} MRV offset {np.round(off, 2).tolist()} m | {leg} | "
              f"arm {'folded' if self.state is not State.ARM_DEPLOY else f'deploying {100.0 * self._deploy_alpha:3.0f} %'} | "
              f"plume {'ON' if (self.vfx is not None and self.vfx.visible) else 'off'} | "
              f"MEP {float(np.linalg.norm(mep)):.2f} m from its rendezvous pose (still drifting) | "
              f"wall {time.time() - self.wall_start:.0f} s", flush=True)

    ###################
    ### Main loop ###
    ###################

    def open_wrist_view(self):
        """GUI: a live `cam_wrist` viewport window (the view the vision system uses)."""
        if self.headless:
            return
        try:
            from omni.kit.viewport.utility import create_viewport_window
            from pxr import Sdf

            self._wrist_window = create_viewport_window(
                "cam_wrist", width=640, height=360, position_x=40, position_y=60,
                camera_path=Sdf.Path(f"{self.geo.link_path}/{self.cfg.camera.name}"),
            )
            print("[VIS] opened the 'cam_wrist' viewport window", flush=True)
            # Docking phase: the probe's own view, under the wrist window
            pc = self.cfg.probe_camera
            if self.cfg.docking.enabled and pc.enabled:
                self._probe_window = create_viewport_window(
                    pc.name, width=640, height=480, position_x=40, position_y=460,
                    camera_path=Sdf.Path(f"{self.geo.mep_path}/{pc.name}"),
                )
                print(f"[VIS] opened the '{pc.name}' viewport window", flush=True)
        except Exception as e:  # never let a GUI convenience stop the run
            print(f"[VIS] could not open the camera window(s): {e} -- pick the camera in the viewport camera menu", flush=True)

    def _capture_viewport(self):
        """Fire-and-forget capture of the main interactive viewport for the
        dashboard's Isaac Sim viewport panel: only the RTX render buffer
        (`omni.kit.viewport.utility.capture_viewport_to_buffer`), never a
        screen share, so no OS window chrome / Isaac Sim UI ever leaks in."""
        if self._viewport_api is None or self.ros is None or self._viewport_capture_pending:
            return
        try:
            from omni.kit.viewport.utility import capture_viewport_to_buffer

            def on_capture(buffer, buffer_size, width, height, byte_format):
                self._viewport_capture_pending = False
                try:
                    if not buffer or width <= 0 or height <= 0 or buffer_size <= 0:
                        return
                    channels = buffer_size // (width * height)
                    if channels < 3:
                        return
                    # `buffer` is a PyCapsule (GPU readback landed in a plain C buffer,
                    # not a Python bytes object) -- same extraction ctypes uses in
                    # Kit's own viewport thumbnail generator.
                    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(ctypes.c_byte * buffer_size)
                    ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
                    content = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
                    arr = np.frombuffer(content.contents, dtype=np.uint8)[: width * height * channels]
                    arr = arr.reshape(height, width, channels)
                    self.ros.publish_viewport_image(self.sim_time, arr[:, :, :3])
                except Exception as e:
                    print(f"[VIEWPORT] capture callback error: {e}", flush=True)

            self._viewport_capture_pending = True
            capture_viewport_to_buffer(self._viewport_api, on_capture)
        except Exception as e:
            self._viewport_capture_pending = False
            print(f"[VIEWPORT] capture request failed: {e}", flush=True)

    def run(self) -> Results:
        sim, scene = self.sim, self.scene
        if ENABLE_EXTRA_CAMERA_WINDOWS:
            self.open_wrist_view()
            if self.astrobee is not None:
                self.astrobee.open_window()
        vision_period = 1.0 / self.cfg.vision.rate_hz
        n = 0
        self.start()
        # SUCCESS on the moving-client pipeline no longer auto-finishes (the MRV keeps
        # departing for the Astrobee to follow). Ctrl+C is then the way to end the run;
        # this still exits the loop cleanly so `finish()` below writes the CSV / video /
        # checks instead of the process dying mid-run.
        prev_sigint = signal.signal(signal.SIGINT, lambda *_: setattr(self, "done", True))
        try:
            with torch.no_grad():
                while self.sim_app.is_running() and not self.done:

                    # Mission control commands must remain responsive even while
                    # the simulation itself is paused.
                    if self.ros is not None:
                        self.ros.poll()

                        # STOP must also work while PAUSED.
                        if self.ros.abort_requested and self.state not in TERMINAL:
                            self.goto(State.ABORTED, "ROS cmd/abort")

                        # PAUSE freezes all mission/physics progression but keeps
                        # ROS callbacks and the GUI responsive. RESUME simply
                        # clears pause_requested and execution continues here.
                        if self.ros.pause_requested and self.state not in TERMINAL:
                            sim.render()
                            time.sleep(0.02)
                            continue

                    # Isaac Timeline is not used as the web PAUSE mechanism.
                    # If it is stopped manually, keep the GUI responsive without
                    # advancing mission time or physics.
                    if not sim.is_playing():
                        sim.render()
                        time.sleep(0.02)
                        continue

                    self.step()
                    if self._mv_active:
                        # MRV translation (after the arm's IK used this step's base frame),
                        # client/MEP thrust, telemetry
                        self.moving_after_step()
                    elif self._client_live:
                        self.client_live_step()  # client drifting since t = 0 (MRV still static)
                    if self.astrobee is not None:
                        self.astrobee.step(self.sim_time, self.state.value, self.assist_context())
                        self.apply_assist_thrust()
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    n += 1
                    self.sim_time += self.dt
                    rendered = n % self.render_interval == 0
                    if rendered:
                        self._rendered_frame_count += 1
                    # Neither the docking phase nor the MRV rendezvous phase uses the wrist
                    # camera / AprilTags (the tag search starts once the arm is deployed)
                    vision_due = (rendered and self.sim_time >= self._next_vision_t - 1e-9
                                  and self.state not in DOCKING_STATES and self.state not in MRV_STATES
                                  and not self._mv_active)
                    if rendered:
                        if vision_due:
                            self.vis.clear()  # keep debug draw out of the image used for detection
                        # headless + camera.render_on_vision_only: nothing looks at the other frames
                        if vision_due or not (self.headless and self.cfg.camera.render_on_vision_only):
                            sim.render()
                    scene.update(dt=self.dt)
                    capture_warm = self._rendered_frame_count >= self._live_capture_warmup_frames
                    mission_started = self.ros is None or self.ros.start_received
                    if (rendered and capture_warm and self.ros is not None
                            and self.sim_time >= self._next_camera_stream_t):
                        self._next_camera_stream_t = (
                            self.sim_time + self._camera_stream_period_s - 1e-9
                        )
                        self.publish_live_camera_images()
                    if rendered and capture_warm and self.astrobee is not None:
                        self.astrobee.after_render(self.sim_time)
                    if (rendered and capture_warm and mission_started and self._viewport_api is not None
                            and self.sim_time >= self._next_viewport_t):
                        self._next_viewport_t = self.sim_time + self._viewport_period_s
                        self._capture_viewport()
                    if vision_due:
                        self._next_vision_t = self.sim_time + vision_period - 1e-9
                        self.process_image()
                        self.update_visuals()
                        self.vis.show()
                    if self.ros is not None and self.sim_time >= self._next_ros_t:
                        self._next_ros_t = self.sim_time + 1.0 / self.cfg.ros.publish_rate_hz - 1e-9
                        self.publish_ros()
                    if self.sim_time >= self._next_log_t:
                        self._next_log_t = self.sim_time + 1.0 / self.cfg.logging.rate_hz - 1e-9
                        self.log_row()
                        if not self._mv_active and self.state not in DOCKING_STATES:
                            self.update_motion_hud()  # MRV approach / capture lines
                    if self.sim_time >= self._next_status_t and self.state not in DOCKING_STATES:
                        self._next_status_t = self.sim_time + 2.0
                        self.status()
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
        self.vis.clear()
        if self._csv_file is not None:
            self._csv_file.close()
        if self._dock_csv_file is not None:
            self._dock_csv_file.close()
        if self._cd_csv_file is not None:
            self._cd_csv_file.close()
        if self._mv_csv_file is not None:
            self._mv_csv_file.close()
        if self.ros is not None:
            self.publish_ros()  # final state (latched topics keep it for late subscribers)
            self.ros.close()
        return self.finish()

    ###############
    ### Results ###
    ###############

    def finish(self) -> Results:
        c = self.cfg
        r = self.results
        r.final_state = self.state.value
        r.metrics["sim_time_s"] = self.sim_time
        if self._mrv_log:
            r.metrics.setdefault("mrv_approach", dict(self._mrv_log))
            r.metrics["capture_phase_start_s"] = self._capture_t0
        r.metrics["wall_time_s"] = time.time() - self.wall_start
        r.metrics["max_arm_contact_force_n"] = self._max_contact
        if self.astrobee is not None and self.astrobee.assist is not None:
            r.metrics["astrobee_assist"] = self.astrobee.assist.summary()
            print("[ASTROBEE] assist: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                   for k, v in r.metrics["astrobee_assist"].items()), flush=True)
        rows = self.rows
        valid = [x for x in rows if not math.isnan(x["position_error_mm"])]
        r.metrics["csv"] = str(self.csv_path)
        r.metrics["csv_rows"] = len(rows)
        r.metrics["overlay_images"] = len(self._overlays)
        r.metrics["overlay_dir"] = str(self.overlay_dir)
        r.metrics["debug_draw_primitives"] = self.vis.num_drawn
        r.metrics["pnp_ambiguous_images"] = self._ambiguous
        if valid:
            pe = np.array([x["position_error_mm"] for x in valid])
            ae = np.array([x["angle_error_deg"] for x in valid])
            r.metrics["tracking_error"] = {
                "position_error_mm_mean": float(pe.mean()), "position_error_mm_max": float(pe.max()),
                "angle_error_deg_mean": float(ae.mean()), "angle_error_deg_max": float(ae.max()),
                "samples": len(valid),
            }
            raw = [x for x in valid if not math.isnan(x["raw_position_error_mm"])]
            if raw:
                r.metrics["tracking_error"].update({
                    "single_image_position_error_mm_max": float(max(x["raw_position_error_mm"] for x in raw)),
                    "single_image_angle_error_deg_max": float(max(x["raw_angle_error_deg"] for x in raw)),
                })

        if c.logging.csv_enabled:
            with open(self.csv_path) as f:
                header = next(csv.reader(f))
            missing = [col for col in CSV_COLUMNS[:36] if col not in header]
            csv_ok = len(rows) > 0 and not missing
        else:  # CSV switched off: the "CSV generated" checks are skipped
            missing, csv_ok = [], None
            r.metrics["csv"] = "disabled (logging.csv_enabled: false)"

        if c.docking.enabled and c.docking.skip_capture:
            # The capture phase never ran, so its checks would only report their own
            # absence; the capture is covered by the other scenarios.
            r.metrics["capture_phase"] = "skipped (docking.skip_capture)"
        elif self.scenario == "static":
            self._finish_static(csv_ok, missing)
        else:
            self._finish_dynamic(csv_ok, missing)
        if c.docking.enabled:
            self._finish_docking()
        if self.moving:
            self._finish_moving()
        self._write_video()
        return r

    def _finish_static(self, csv_ok, missing):
        c = self.cfg
        r = self.results
        stats = self._static_stats
        r.metrics["static"] = stats
        det_ok = bool(stats) and all(s["samples"] >= c.test.static_samples for s in stats)
        r.check("[TEST1] AprilTag detection", det_ok and self.state == State.SUCCESS,
                f"4/4 tags with a valid PnP pose in {sum(s['samples'] for s in stats)} images at standoffs {[s['standoff_m'] for s in stats]}")
        if stats:
            worst_p = max(s["position_error_mm_max"] for s in stats)
            worst_a = max(s["angle_error_deg_max"] for s in stats)
        else:
            worst_p = worst_a = math.nan
        rms = max((s["reproj_rms_px_mean"] for s in stats), default=math.nan)
        r.check("[TEST1] PnP pose estimation", bool(stats) and rms <= c.apriltag.max_reprojection_rms_px, f"mean reprojection RMS <= {rms:.3f} px (16 points)")
        r.check("[TEST1] Cylinder_01 center estimation", bool(stats) and worst_p < c.test.static_position_threshold_mm and worst_a < c.test.static_angle_threshold_deg,
                f"worst position error {worst_p:.3f} mm (< {c.test.static_position_threshold_mm}), worst angle error {worst_a:.4f} deg (< {c.test.static_angle_threshold_deg})",
                position_error_mm=worst_p, angle_error_deg=worst_a)
        r.check("[TEST1] Debug visualization", self.vis.available and self.vis.num_drawn > 0 and len(self._overlays) > 0,
                f"debug draw primitives {self.vis.num_drawn}, overlay images {len(self._overlays)} in {self.overlay_dir}")
        if csv_ok is not None:
            r.check("[TEST1] CSV generated", csv_ok, f"{len(self.rows)} rows, {'all required columns' if not missing else 'missing ' + str(missing)} -> {self.csv_path}")
        passed = all(v["pass"] for k, v in r.checks.items() if k.startswith("[TEST1]")) and self.state == State.SUCCESS
        r.metrics["test1_pass"] = passed

    def _finish_dynamic(self, csv_ok, missing):
        c = self.cfg
        r = self.results
        ci = self._capture_info
        captured = bool(ci)
        cap_t = ci.get("time_s", math.nan)
        rel_gt = ci["gt"]["rel_vel"] if captured else math.nan
        rel_est = ci["vision"]["rel_vel"] if captured else math.nan
        r.metrics["capture"] = ci
        pre = [x for x in self.rows if x["capture_state"] in (State.APPROACHING.value, State.SLOW_APPROACH.value, State.CAPTURE_ATTEMPT.value)]
        tags_ok = bool(pre) and all(x["num_detected_tags"] == 4 for x in pre)
        r.check("[TEST2] AprilTag detection during approach", tags_ok, f"{sum(1 for x in pre if x['num_detected_tags'] == 4)}/{len(pre)} logged frames with 4/4 tags")
        pred_err = self._prediction_error()
        r.metrics["prediction_error"] = pred_err
        r.check("[TEST2] Future position prediction", pred_err.get("samples", 0) > 0,
                f"{c.prediction.horizon_sec:.1f} s prediction vs GT at t+h: mean {pred_err.get('mean_mm', math.nan):.2f} mm, max {pred_err.get('max_mm', math.nan):.2f} mm ({pred_err.get('samples', 0)} samples)")
        r.check("[TEST2] FixedJoint within timeout", captured and cap_t <= c.test.dynamic_timeout_sec,
                f"captured at t={cap_t:.2f} s (limit {c.test.dynamic_timeout_sec:.0f} s)", capture_time_s=cap_t)
        r.check("[TEST2] Relative velocity at capture", captured and rel_gt < c.capture.max_relative_velocity_mps,
                f"GT {rel_gt:.4f} m/s, vision estimate {rel_est:.4f} m/s (< {c.capture.max_relative_velocity_mps})", relative_velocity_mps=rel_gt)
        if captured:
            g = ci["gt"]
            r.check("[TEST2] Capture geometry (GT)", g["distance"] <= c.capture.max_distance_m and g["angle"] <= c.capture.max_angle_deg and g["lateral"] <= c.capture.max_lateral_m,
                    f"distance {g['distance']:.4f} m (<= {c.capture.max_distance_m}), angle {g['angle']:.3f} deg (<= {c.capture.max_angle_deg}), lateral {g['lateral']*1000:.2f} mm (<= {c.capture.max_lateral_m*1000:.0f})")
        if self.six_dof:
            self._finish_six_dof(captured)
        h = self._hold
        held = "duration_s" in h
        r.metrics["holding"] = {k: v for k, v in h.items() if k != "ee_start"}
        r.check("[TEST3] Holding duration", held and h["duration_s"] >= c.test.holding_duration_sec - 1e-6,
                f"{h.get('duration_s', 0.0):.2f} s (>= {c.test.holding_duration_sec})", holding_duration_s=h.get("duration_s", 0.0))
        r.check("[TEST3] FixedJoint validity", held and h.get("joint_valid", False), "joint prim present during the whole holding" if held else "not reached")
        stable = held and h["max_drift_mm"] < c.test.holding_max_relative_drift_mm and h["max_drift_deg"] < c.test.holding_max_relative_drift_deg and self.results.failure is None
        r.check("[TEST3] Physics stability (MEP-EE relative pose)", stable,
                f"max drift {h.get('max_drift_mm', math.nan):.3f} mm / {h.get('max_drift_deg', math.nan):.4f} deg, max MEP rate {h.get('max_mep_rate_deg_s', math.nan):.4f} deg/s, no NaN/explosion" if held else "not reached")
        if held:
            v0 = ci.get("mep_speed_mps", math.nan)
            v2 = h.get("max_mep_speed_second_half", math.nan)
            r.check("[TEST3] Robot EE stability (motion bounded, not diverging)", h["max_ee_motion_mm"] < 50.0 and v2 < v0,
                    f"max EE excursion {h['max_ee_motion_mm']:.1f} mm (< 50), MEP speed {v0*1000:.2f} mm/s at capture -> max {v2*1000:.2f} mm/s in the 2nd half of holding, "
                    f"{h['final_mep_speed']*1000:.2f} mm/s at the end (residual arm+payload oscillation), MEP moved {h['mep_displacement_mm']:.1f} mm after capture")
        rt = self._retreat
        if c.test.retreat_distance_m > 0.0:
            ok = rt is not None and "mep_moved_m" in rt and abs(rt["mep_moved_m"] - rt["ee_moved_m"]) < 0.005 and rt["max_drift_mm"] < c.test.holding_max_relative_drift_mm and rt["ee_moved_m"] > 0.8 * c.test.retreat_distance_m
            r.metrics["retreat"] = {k: v for k, v in (rt or {}).items() if k not in ("start", "mep_start")}
            r.check("[TEST3] Retreat with the MEP attached", ok,
                    f"EE {rt.get('ee_moved_m', math.nan)*1000:.1f} mm, MEP {rt.get('mep_moved_m', math.nan)*1000:.1f} mm, max drift {rt.get('max_drift_mm', math.nan):.3f} mm" if rt else "not reached")
        if csv_ok is not None:
            r.check("[DYN] CSV generated", csv_ok, f"{len(self.rows)} rows -> {self.csv_path}")
        r.metrics["test2_pass"] = all(v["pass"] for k, v in r.checks.items() if k.startswith("[TEST2]"))
        r.metrics["test3_pass"] = all(v["pass"] for k, v in r.checks.items() if k.startswith("[TEST3]")) and self.state == State.SUCCESS

    def _prediction_error(self) -> dict:
        """Compare each logged prediction (made at t for t+h) with the GT at t+h
        (position: linear interpolation, orientation: slerp between the logged GT rows)."""
        h = self.cfg.prediction.horizon_sec
        rows = [x for x in self.rows if not math.isnan(x["pred_x"]) and x["capture_success"] == 0]
        t = np.array([x["timestamp"] for x in self.rows])
        gt = np.array([[x["gt_x"], x["gt_y"], x["gt_z"]] for x in self.rows])
        errs, ang_errs, hold_errs = [], [], []
        for x in rows:
            tq = x["timestamp"] + h
            if tq > t[-1]:
                continue
            g = np.array([np.interp(tq, t, gt[:, i]) for i in range(3)])
            errs.append(1000.0 * float(np.linalg.norm(np.array([x["pred_x"], x["pred_y"], x["pred_z"]]) - g)))
            if math.isnan(x["pred_qw"]) or math.isnan(x["est_qw"]):
                continue
            i = int(np.clip(np.searchsorted(t, tq), 1, len(t) - 1))
            a, b = self.rows[i - 1], self.rows[i]
            s = 0.0 if b["timestamp"] <= a["timestamp"] else (tq - a["timestamp"]) / (b["timestamp"] - a["timestamp"])
            g_rot = interp_frame(*(Frame.from_pos_quat(np.zeros(3), [r[f"gt_q{k}"] for k in "wxyz"]) for r in (a, b)), float(np.clip(s, 0.0, 1.0))).rot
            pred_rot = Frame.from_pos_quat(np.zeros(3), [x[f"pred_q{k}"] for k in "wxyz"]).rot
            est_rot = Frame.from_pos_quat(np.zeros(3), [x[f"est_q{k}"] for k in "wxyz"]).rot
            ang_errs.append(math.degrees(rotation_angle(pred_rot, g_rot)))
            # baseline: holding the current estimate (no rotation model) for the horizon
            hold_errs.append(math.degrees(rotation_angle(est_rot, g_rot)))
        if not errs:
            return {"samples": 0}
        out = {"samples": len(errs), "mean_mm": float(np.mean(errs)), "max_mm": float(np.max(errs))}
        if ang_errs:
            out.update({"orientation_samples": len(ang_errs), "orientation_mean_deg": float(np.mean(ang_errs)), "orientation_max_deg": float(np.max(ang_errs)),
                        "hold_orientation_mean_deg": float(np.mean(hold_errs)), "hold_orientation_max_deg": float(np.max(hold_errs))})
        return out

    def _finish_six_dof(self, captured: bool):
        """Additional six_dof checks. The '(report)' checks only require the data to exist
        (no accuracy threshold has been agreed yet); the values are in the detail/metrics."""
        c = self.cfg
        r = self.results
        fr = self._w_frame_result
        r.metrics["angular_velocity_frame"] = fr
        r.check("[6DOF] Angular velocity is a world-frame vector", bool(fr and fr["pass"]),
                (f"direction error {fr['direction_error_deg']:.2f} deg (< 5); from poses (world) error {fr['err_world']:.2e}, (body) error {fr['err_body']:.2e} rad/s"
                 f"{'' if fr['world_and_body_distinguishable'] else ', world/body not distinguishable for this start orientation'}") if fr else "not measured")
        r.check("[6DOF] Rotation rate vs commanded (report)", bool(fr),
                (f"rate from poses / commanded = {fr['rate_ratio_pose_over_cmd']:.3f}, / root_com_ang_vel_w = {fr['rate_ratio_pose_over_root_com']:.3f} "
                 "(1.0 = the body turns at the reported rate)") if fr else "not measured")
        rows = [x for x in self.rows if x["capture_success"] == 0 and not math.isnan(x["mep_angular_velocity_x"])]
        if rows:
            e = np.array([[x[f"mep_angular_velocity_{k}"] - x[f"gt_mep_angular_velocity_{k}"] for k in "xyz"] for x in rows])
            n = np.linalg.norm(e, axis=1)
            w_gt = np.array([[x[f"gt_mep_angular_velocity_{k}"] for k in "xyz"] for x in rows])
            stats = {"samples": len(rows), "error_mean_rad_s": float(n.mean()), "error_max_rad_s": float(n.max()),
                     "gt_rate_mean_rad_s": float(np.linalg.norm(w_gt, axis=1).mean())}
        else:
            stats = {"samples": 0}
        r.metrics["angular_velocity_error"] = stats
        r.check("[6DOF] Angular velocity estimation (report)", stats["samples"] > 0,
                f"|w_est - w_gt| mean {stats.get('error_mean_rad_s', math.nan):.5f}, max {stats.get('error_max_rad_s', math.nan):.5f} rad/s "
                f"(|w_gt| mean {stats.get('gt_rate_mean_rad_s', math.nan):.5f} rad/s, {stats['samples']} samples)")
        pe = r.metrics.get("prediction_error", {})
        r.check("[6DOF] Future orientation prediction (report)", pe.get("orientation_samples", 0) > 0,
                f"{c.prediction.horizon_sec:.1f} s prediction vs GT at t+h: mean {pe.get('orientation_mean_deg', math.nan):.4f} deg, max {pe.get('orientation_max_deg', math.nan):.4f} deg "
                f"(holding the current estimate instead: mean {pe.get('hold_orientation_mean_deg', math.nan):.4f} deg)")
        ci = self._capture_info
        rel = ci["gt"]["rel_ang_vel"] if captured else math.nan
        r.check("[6DOF] Relative angular velocity at capture", captured and rel <= c.capture.max_relative_angular_velocity_rad_s,
                f"GT {rel:.5f} rad/s, vision estimate {ci['vision']['rel_ang_vel'] if captured else math.nan:.5f} rad/s (<= {c.capture.max_relative_angular_velocity_rad_s})",
                relative_angular_velocity_rad_s=rel)

    def _finish_docking(self):
        """Docking-phase results: what actually happened, plus the depth cross-check."""
        d = self.cfg.docking
        r = self.results
        dk = self._dock
        docked = bool(self.task.docking.is_docked)
        r.metrics.setdefault("docking", {}).update({
            "docked": docked,
            "rows": len(self._dock_rows),
            "csv": str(getattr(self, "dock_csv_path", "")) if self.cfg.logging.csv_enabled else "disabled",
            "probe_rgb_frames": len(self._probe_frames),
            "probe_rgb_dir": str(self._probe_dir),
            "realign_events": int(dk.get("realigns", 0)),
            "depth_blocked_s": float(dk.get("depth_blocked_s", 0.0)),
            "min_wall_clearance_m": (None if not math.isfinite(float(dk.get("min_clearance", math.inf))) else float(dk["min_clearance"])),
            "max_lateral_while_inserting_m": float(dk.get("max_lateral_inserting", 0.0)),
            "at_dock": dk.get("at_dock"),
        })
        # Controller comparison metrics (same keys for both `docking_control.mode`s)
        cmp_ = self._cd_metrics.summary(self.cfg.docking_control.mode, docked)
        cmp_["csv"] = str(getattr(self, "dock_control_csv_path", "")) if self.cfg.logging.csv_enabled else "disabled"
        r.metrics["docking_control"] = cmp_
        print("[DOCK] controller metrics: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                                        for k, v in cmp_.items() if k != "csv"), flush=True)
        if self._dock_csv is not None:
            r.check("[DOCK] CSV generated", len(self._dock_rows) > 0,
                    f"{len(self._dock_rows)} rows -> {getattr(self, 'dock_csv_path', '')}")
        # Depth vs geometry, from the moment the camera is on the docking axis. Before
        # that the probe is still being swung 5.5 m sideways and the ray simply misses
        # the nozzle, so those readings say nothing about the sensor.
        since = self._depth_calibrated_t if self._depth_calibrated_t is not None else math.inf
        rows = [x for x in self._dock_rows
                if x["timestamp"] >= since and math.isfinite(x["depth_distance_m"]) and math.isfinite(x["geometry_distance_m"])]
        if rows:
            diff = np.array([abs(x["depth_distance_m"] - x["geometry_distance_m"]) for x in rows])
            geo = np.array([x["geometry_distance_m"] for x in rows])
            dep = np.array([x["depth_distance_m"] for x in rows])
            rejected = int((diff > d.depth_agreement_m).sum())
            stats = {"samples": len(rows), "mean_abs_difference_m": float(diff.mean()), "max_abs_difference_m": float(diff.max()),
                     "rejected_frames": rejected, "rejected_fraction": rejected / len(rows),
                     "geometry_first_m": float(geo[0]), "geometry_last_m": float(geo[-1]),
                     "depth_first_m": float(dep[0]), "depth_last_m": float(dep[-1]),
                     "correlation": float(np.corrcoef(geo, dep)[0, 1]) if len(rows) > 2 and geo.std() > 1e-9 and dep.std() > 1e-9 else None}
            r.metrics["docking"]["depth_vs_geometry"] = stats
            # Judged on what the controller was allowed to use: frames outside the
            # agreement band are rejected by `depth_valid` and the probe holds instead
            # of advancing, so a few of them are handled, not ignored.
            r.check("[DOCK3] Depth agrees with the geometric docking distance",
                    float(diff.mean()) <= d.depth_agreement_m and rejected <= 0.05 * len(rows),
                    f"|depth - geometry| mean {diff.mean()*1000:.0f} mm, max {diff.max()*1000:.0f} mm (band {d.depth_agreement_m*1000:.0f} mm), "
                    f"{rejected}/{len(rows)} frames outside the band were rejected and held the approach, correlation {stats['correlation']:.4f}")
            r.check("[DOCK3] Depth decreases during the approach",
                    float(dep[-1]) < float(dep[0]) and float(geo[-1]) < float(geo[0]),
                    f"depth {dep[0]:.3f} -> {dep[-1]:.3f} m, geometry {geo[0]:.3f} -> {geo[-1]:.3f} m")
        else:
            r.check("[DOCK3] Depth agrees with the geometric docking distance", False,
                    "the probe never reached the docking axis, so the depth camera was never on the nozzle"
                    if self._depth_calibrated_t is None else f"no usable depth reading ({self._depth_reason})")
        r.check("[DOCK5] Docking FixedJoint created", docked,
                r.checks.get("[DOCK5] Docking FixedJoint created", {}).get("detail", "the probe never reached the docking conditions"))
        if not docked:
            r.failure = r.failure or f"docking did not complete (final state {self.state.value})"

    def _write_video(self):
        import cv2

        if not self._overlays:
            return
        first = cv2.imread(str(self._overlays[0]))
        h, w = first.shape[:2]
        path = self.out_dir / f"{self.label}_overlay.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
        if vw.isOpened():
            for p in self._overlays:
                vw.write(cv2.imread(str(p)))
            vw.release()
            self.results.metrics["overlay_video"] = str(path)

    def write_json(self, path: Path):
        r = self.results
        payload = {"scenario": r.scenario, "run": self.label, "final_state": r.final_state, "failure": r.failure, "checks": r.checks, "metrics": r.metrics}
        path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def default_out_dir() -> Path:
    return PROJECT_DIR.joinpath("logs", "vision_capture")
