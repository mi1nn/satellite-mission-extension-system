"""AprilTag constellation -> Cylinder_01 pose estimation and motion prediction
(`LinearDriftPredictor`: translation only, `ConstantTwistPredictor`: 6-DoF).

Pure numpy / OpenCV (plus a few USD helpers for spawning the tags), so the vision
math can be tested without Isaac Sim (`project/tests/test_vision_math.py`).

Frames (all rigid, right-handed, rotation matrices have the axes as columns):

- ``W``     world
- ``B``     robot base (``canadarm3_large_0``, fixed; the MRV frame for Phase 1)
- ``L``     Canadarm3 last link (``canadarm3_large_7``)
- ``C``     ``cam_wrist`` optical frame, OpenCV/ROS convention: +X right, +Y down,
            +Z forward (the image plane looks along +Z)
- ``M``     MEP rigid-body frame (``debris``)
- ``T``     tag constellation: origin at the constellation centre on the tag plane,
            +Z = outward normal of the MEP attachment face, +X = MEP grasp +X
- ``Y``     ``Cylinder_01`` prim frame (unscaled), the ground-truth docking point

Estimation chain:  image -> (PnP) T_C_T -> T_W_T = T_W_L @ T_L_C @ T_C_T
                   -> T_W_Y = T_W_T @ T_T_Y   (T_T_Y is a fixed, design-time transform)

Conventions: quaternions are (w, x, y, z) as in Isaac Lab / USD Gf.Quat; `A @ B` of
`Frame`s is T_A_B composition (child on the right); angular velocities are rad/s.
"""

import copy
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .astrobee import AstrobeeCfg, validate_astrobee_cfg
from .coupled_dock import DockingAlignmentCfg, DockingControlCfg, validate_docking_control_cfg
from .frames import Frame, rotation_angle
from .moving_dock import (
    ClientMotionCfg,
    PostDockingCfg,
    RendezvousCfg,
    SeparationCfg,
    validate_moving_cfg,
)
from .mrv_approach import MrvApproachCfg, validate_mrv_cfg
from .probe_dock import (
    DockingVisionCfg,
    ProbeCameraCfg,
    validate_docking_cfg,
    validate_probe_camera_cfg,
)

##############
### Config ###
##############


MOTION_MODES = ("translation_only", "six_dof")
# six_dof prediction reference: the tag constellation T (vision + the design-time
# T_T_Y only) or the MEP centre of mass (moves in a straight line in free flight)
REFERENCE_FRAMES = ("tag", "mep_com")
# six_dof start orientation: "converge" = the MEP turns INTO the nominal (parallel) orientation
# at `rendezvous_time_s` and away from it afterwards; "diverge" = it starts in the nominal
# orientation and turns away from it, so the arm has to chase the rotation
ROTATION_STARTS = ("converge", "diverge")


@dataclass
class MepVisionCfg:
    mass_kg: float = 3000.0
    # "translation_only" (Phase 1: no rotation, angular_velocity_rad_s is not applied)
    # or "six_dof" (XYZ drift + combined roll/pitch/yaw rate)
    motion_mode: str = "six_dof"
    # Linear velocity [m/s] (world) = drift_direction (unit) * linear_velocity_mps
    linear_velocity_mps: float = 0.01
    drift_direction: List[float] = field(default_factory=lambda: [0.0, -0.8, 0.6])
    # six_dof only: [wx, wy, wz] [rad/s] in the WORLD frame (Isaac Lab
    # `RigidObjectCfg.InitialStateCfg.ang_vel` / `root_com_ang_vel_w` convention)
    angular_velocity_rad_s: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rendezvous_time_s: float = 15.0
    # six_dof only, see ROTATION_STARTS
    rotation_start: str = "converge"

    @property
    def six_dof(self) -> bool:
        return self.motion_mode == "six_dof"

    def linear_velocity_w(self) -> np.ndarray:
        """Initial MEP (centre of mass) linear velocity [m/s], world frame."""
        return np.asarray(self.drift_direction, dtype=float) * float(self.linear_velocity_mps)

    def angular_velocity_w(self) -> np.ndarray:
        """Initial MEP angular velocity [rad/s], world frame (zero in translation_only)."""
        if not self.six_dof:
            return np.zeros(3)
        return np.asarray(self.angular_velocity_rad_s, dtype=float).reshape(3)


@dataclass
class CameraVisionCfg:
    name: str = "cam_wrist"
    width: int = 1280
    height: int = 720
    horizontal_fov_deg: float = 90.0
    horizontal_aperture_mm: float = 20.955
    clipping_range_m: List[float] = field(default_factory=lambda: [0.01, 100.0])
    mount_link_z_m: float = -0.45
    # Roll of the camera about its viewing axis [deg], right-handed about the optical +Z
    # (clockwise as seen from behind the camera). 90: image upright (was 0 = rolled 90 deg CCW)
    image_roll_deg: float = 90.0
    antialiasing: str = "Off"
    # Render at `supersample` x the resolution and area-average down to width x height
    supersample: int = 1
    # headless only: render the scene (and so `cam_wrist`) only for the frames the vision
    # system uses (`vision.rate_hz`) instead of every `render_interval` physics steps
    render_on_vision_only: bool = False

    @property
    def focal_length_mm(self) -> float:
        return self.horizontal_aperture_mm / (2.0 * math.tan(math.radians(self.horizontal_fov_deg) / 2.0))


@dataclass
class AprilTagCfg:
    family: str = "tag36h11"
    ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    tag_size_m: float = 0.06
    constellation_half_width_m: float = 0.19
    quiet_zone_cells: int = 1
    face_offset_m: float = 0.001
    texture_px_per_cell: int = 40
    min_tags_for_pose: int = 4
    use_ransac: bool = True
    ransac_reprojection_px: float = 3.0
    max_reprojection_rms_px: float = 1.5
    # Planar ambiguity: if the 2nd IPPE solution's RMS is within this factor of the
    # best one, pick the solution closer to the fused orientation (prior)
    ambiguity_ratio: float = 1.5


@dataclass
class VisionRateCfg:
    rate_hz: float = 10.0
    tag_loss_timeout_s: float = 1.0


@dataclass
class PredictionCfg:
    horizon_sec: float = 0.3
    velocity_window_sec: float = 2.0
    # Orientation is averaged over a longer window (the MEP does not rotate in Phase 1)
    orientation_window_sec: float = 5.0
    min_samples: int = 10
    max_plausible_speed_mps: float = 0.2
    # six_dof: an estimated angular rate above this is rejected as a bad estimate [rad/s]
    max_plausible_angular_rate_rad_s: float = 0.2
    # six_dof: frame whose origin is fitted with constant velocity ("tag" | "mep_com")
    reference_frame: str = "tag"


@dataclass
class ApproachCfg:
    start_at_observe_pose: bool = True
    # Start the arm swung about its base axis (azimuth) by this angle [deg] instead of
    # at the observation pose: the wrist camera then does not look at the MEP, and the
    # SEARCH state has to swing the arm back until the AprilTags come into view.
    # 0: start exactly at the observation pose.
    start_yaw_offset_deg: float = 15.0
    # SEARCH swing (only when the arm starts off the observation pose). The normal
    # tracking gate stops the reference as soon as the arm lags it by 5 cm, which with
    # the overdamped Canadarm3 drives (lag = v * 0.625 s) caps the speed at ~80 mm/s.
    # The arm carries no payload here, so the gate and the joint step can be wider.
    search_speed_mps: float = 0.25
    search_lag_m: float = 0.20
    search_max_joint_step_rad: float = 0.08
    observe_distance_m: float = 0.6
    approach_standoff_m: float = 0.5
    final_gap_m: float = 0.05
    speed_far_mps: float = 0.11
    speed_near_mps: float = 0.04
    speed_capture_range_mps: float = 0.01
    near_distance_m: float = 0.30
    capture_range_m: float = 0.12
    speed_ang_deg_s: float = 6.0
    gate_lateral_m: float = 0.01
    gate_angle_deg: float = 1.0
    settle_lateral_m: float = 0.02
    settle_angle_deg: float = 1.0
    max_joint_step_rad: float = 0.04
    ik_lambda: float = 0.05
    stage_timeout_s: float = 60.0


@dataclass
class CaptureVisionCfg:
    max_distance_m: float = 0.15
    max_angle_deg: float = 5.0
    max_relative_velocity_mps: float = 0.05
    # six_dof only: |w_EE - w_MEP| [rad/s] (not checked in translation_only)
    max_relative_angular_velocity_rad_s: float = 0.01
    max_lateral_m: float = 0.02
    # Radius of the (translucent) EE capture cylinder, sized to the Canadarm3 flange.
    # The MEP marker Cylinder_01 is scaled to the same radius (position, orientation
    # and height unchanged), so their overlap shows the alignment error directly.
    cylinder_radius_m: float = 0.20
    # Index of refraction of the capture cylinder the wrist camera looks through
    # (1.0 = no refraction, so the tags are not optically displaced)
    cylinder_ior: float = 1.0


@dataclass
class TestVisionCfg:
    static_position_threshold_mm: float = 5.0
    static_angle_threshold_deg: float = 0.5
    static_standoffs_m: List[float] = field(default_factory=lambda: [0.7, 0.4, 0.2])
    static_samples: int = 30
    dynamic_timeout_sec: float = 60.0
    holding_duration_sec: float = 10.0
    holding_max_relative_drift_mm: float = 5.0
    holding_max_relative_drift_deg: float = 0.5
    retreat_distance_m: float = 0.2
    retreat_speed_mps: float = 0.01


@dataclass
class LoggingVisionCfg:
    dir: str = "logs/vision_capture"
    csv: str = "logs/vision_capture_metrics.csv"
    rate_hz: float = 10.0
    # false: no per-run metrics CSV is written (rows are still kept for the result checks)
    csv_enabled: bool = True
    overlay_every_sec: float = 1.0
    debug_draw: bool = True
    # Moving client only (visual, no effect on physics or control): the path of the
    # client / MEP / MRV as a trail of points, with a fixed cross at each start position,
    # so a 20 mm/s drift is visible; and a GUI window with the live speeds (not headless)
    motion_trail: bool = True
    motion_trail_period_s: float = 1.0
    motion_trail_max_points: int = 600
    motion_hud: bool = True


@dataclass
class RosVisionCfg:
    """Optional ROS 2 interface (`ros_interface.py`): telemetry out, start / abort /
    capture-enable commands in. Off by default; `vision_capture.py --ros` switches it on."""

    enabled: bool = True
    node_name: str = "mrv_vision_capture"
    # Every topic lives under /<namespace>/ (no leading slash)
    namespace: str = "mrv"
    world_frame: str = "world"
    # Pose / twist / status topics [Hz] (simulation time); the camera topics follow vision.rate_hz
    publish_rate_hz: float = 10.0
    publish_image: bool = True
    publish_tf: bool = True
    # Ground-truth topics (`gt/...`): evaluation only, never used by the controller
    publish_ground_truth: bool = True
    # true: the arm holds at the observation pose until `cmd/start` arrives (the MEP keeps
    # drifting meanwhile, and `test.dynamic_timeout_sec` keeps counting)
    require_start_cmd: bool = False
    # Isaac Sim ROS 2 bridge distro used when rclpy is not already sourced
    distro: str = "jazzy"


@dataclass
class VisionCaptureConfig:
    mep: MepVisionCfg = field(default_factory=MepVisionCfg)
    camera: CameraVisionCfg = field(default_factory=CameraVisionCfg)
    apriltag: AprilTagCfg = field(default_factory=AprilTagCfg)
    vision: VisionRateCfg = field(default_factory=VisionRateCfg)
    prediction: PredictionCfg = field(default_factory=PredictionCfg)
    approach: ApproachCfg = field(default_factory=ApproachCfg)
    capture: CaptureVisionCfg = field(default_factory=CaptureVisionCfg)
    test: TestVisionCfg = field(default_factory=TestVisionCfg)
    logging: LoggingVisionCfg = field(default_factory=LoggingVisionCfg)
    ros: RosVisionCfg = field(default_factory=RosVisionCfg)
    # MRV rendezvous phase in front of the capture (`mrv_approach.py`): folded arm,
    # two translation legs, arm deployment, thruster plume VFX
    mrv: MrvApproachCfg = field(default_factory=MrvApproachCfg)
    # Ares1 probe -> satellite thruster docking phase (`probe_dock.py`)
    docking: DockingVisionCfg = field(default_factory=DockingVisionCfg)
    # Docking controller selection (`coupled_dock.py`): legacy (default, the verified
    # sequential alignment) or coupled_predictive, and the latter's alignment gate
    docking_control: DockingControlCfg = field(default_factory=DockingControlCfg)
    docking_alignment: DockingAlignmentCfg = field(default_factory=DockingAlignmentCfg)
    # RGB-D camera on the Ares1 probe (docking phase)
    probe_camera: ProbeCameraCfg = field(default_factory=ProbeCameraCfg)
    # Moving-client scenario (`moving_dock.py`): client release -> chase -> velocity
    # matching -> rendezvous -> docking -> stop -> release -> arm retreat -> departure.
    # Off unless `client.release_enabled` (`vision_capture.py --moving_dock`).
    client: ClientMotionCfg = field(default_factory=ClientMotionCfg)
    rendezvous: RendezvousCfg = field(default_factory=RendezvousCfg)
    post_docking: PostDockingCfg = field(default_factory=PostDockingCfg)
    separation: SeparationCfg = field(default_factory=SeparationCfg)
    # Astrobee satellite observation camera (`astrobee.py`): flies around the satellite
    # and streams its camera image only; independent of (and invisible to) the mission
    astrobee: AstrobeeCfg = field(default_factory=AstrobeeCfg)

    def to_dict(self) -> dict:
        def conv(obj):
            if is_dataclass(obj):
                return {f.name: conv(getattr(obj, f.name)) for f in fields(obj)}
            return copy.deepcopy(obj)

        return conv(self)


PROJECT_DIR = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = PROJECT_DIR.joinpath("config", "vision_capture.yaml")


def _apply(obj, data: dict, where: str):
    for key, value in data.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown vision config key '{where}{key}'")
        current = getattr(obj, key)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"'{where}{key}' must be a mapping")
            _apply(current, value, f"{where}{key}.")
        else:
            setattr(obj, key, value)


def load_vision_config(path: Optional[str | Path] = None, overrides: Sequence[str] = ()) -> VisionCaptureConfig:
    """Load the YAML config; `overrides` are `section.key=value` strings (YAML values)."""
    import yaml

    cfg = VisionCaptureConfig()
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        _apply(cfg, yaml.safe_load(f) or {}, "")
    for item in overrides:
        key, _, raw = item.partition("=")
        section, _, name = key.strip().partition(".")
        if not name:
            raise ValueError(f"Override must look like 'section.key=value', got '{item}'")
        # Nested sections: `astrobee.assist.enabled=true` -> {astrobee: {assist: {enabled: true}}}
        nested = yaml.safe_load(raw)
        for part in reversed(name.split(".")):
            nested = {part: nested}
        _apply(cfg, {section: nested}, "")
    d = np.asarray(cfg.mep.drift_direction, dtype=float)
    if np.linalg.norm(d) < 1e-9:
        raise ValueError("mep.drift_direction must be non-zero")
    cfg.mep.drift_direction = (d / np.linalg.norm(d)).tolist()
    if cfg.mep.motion_mode not in MOTION_MODES:
        raise ValueError(f"mep.motion_mode must be one of {MOTION_MODES}, got '{cfg.mep.motion_mode}'")
    w = np.asarray(cfg.mep.angular_velocity_rad_s, dtype=float)
    if w.shape != (3,) or not np.isfinite(w).all():
        raise ValueError(f"mep.angular_velocity_rad_s must be 3 finite values [rad/s], got {cfg.mep.angular_velocity_rad_s}")
    cfg.mep.angular_velocity_rad_s = w.tolist()
    if cfg.mep.rotation_start not in ROTATION_STARTS:
        raise ValueError(f"mep.rotation_start must be one of {ROTATION_STARTS}, got '{cfg.mep.rotation_start}'")
    if cfg.prediction.reference_frame not in REFERENCE_FRAMES:
        raise ValueError(f"prediction.reference_frame must be one of {REFERENCE_FRAMES}, got '{cfg.prediction.reference_frame}'")
    if cfg.mep.six_dof and np.linalg.norm(w) >= cfg.prediction.max_plausible_angular_rate_rad_s:
        raise ValueError("mep.angular_velocity_rad_s exceeds prediction.max_plausible_angular_rate_rad_s "
                         "(the estimate would always be rejected)")
    validate_mrv_cfg(cfg.mrv)
    validate_docking_cfg(cfg.docking)
    validate_docking_control_cfg(cfg.docking_control, cfg.docking_alignment)
    validate_probe_camera_cfg(cfg.probe_camera)
    validate_moving_cfg(cfg.client, cfg.rendezvous, cfg.post_docking, cfg.separation)
    validate_astrobee_cfg(cfg.astrobee)
    if cfg.logging.motion_trail_period_s <= 0.0 or int(cfg.logging.motion_trail_max_points) < 2:
        raise ValueError("logging.motion_trail_period_s must be > 0 and motion_trail_max_points >= 2")
    if cfg.client.release_enabled and not cfg.docking.enabled:
        raise ValueError("client.release_enabled (moving-client docking) requires docking.enabled")
    if cfg.ros.publish_rate_hz <= 0.0 or not cfg.ros.namespace.strip("/"):
        raise ValueError("ros.publish_rate_hz must be > 0 and ros.namespace must not be empty")
    cfg.ros.namespace = cfg.ros.namespace.strip("/")
    if cfg.apriltag.min_tags_for_pose < 2:
        raise ValueError("apriltag.min_tags_for_pose must be >= 2 (constellation PnP)")
    if len(set(cfg.apriltag.ids)) != 4:
        raise ValueError("apriltag.ids must hold 4 distinct IDs")
    return cfg


####################
### Math helpers ###
####################


def rotmat_to_rpy_deg(r: np.ndarray) -> Tuple[float, float, float]:
    """Roll/pitch/yaw [deg] of R = Rz(yaw) Ry(pitch) Rx(roll), each wrapped to (-180, 180]."""
    pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
    return tuple(math.degrees(a) for a in (roll, pitch, yaw))  # type: ignore


def wrap_deg(a: float) -> float:
    """Wrap an angle difference to (-180, 180] (e.g. 179 - (-179) -> -2)."""
    r = (a + 180.0) % 360.0 - 180.0
    return 180.0 if r == -180.0 else r


def rotvec_from_rotmat(r: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(r, dtype=float))[0].reshape(3)


def rotmat_from_rotvec(v) -> np.ndarray:
    import cv2

    return cv2.Rodrigues(np.asarray(v, dtype=float).reshape(3, 1))[0]


def pose_errors(gt: Frame, est: Frame) -> Tuple[float, float]:
    """(|p_gt - p_est| [m], minimal rotation angle of R_gt^-1 R_est [deg])."""
    return float(np.linalg.norm(gt.pos - est.pos)), math.degrees(rotation_angle(gt.rot, est.rot))


# OpenCV/ROS optical frame of `cam_wrist` in the Canadarm3 last link: on the link axis at
# `mount_link_z_m`, looking along link -Z (the capture direction). A half turn about X
# gives +X_C = +X_L, +Y_C = -Y_L, +Z_C = -Z_L, i.e. the same axes as the EE contact frame.
# `camera.image_roll_deg` then rolls the camera about +Z_C (so the image axes are no longer
# the EE axes; the estimator uses this same transform, so the pose chain is unchanged).
CAM_MOUNT_QUAT_ROS = (0.0, 1.0, 0.0, 0.0)


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def camera_in_link(cfg: CameraVisionCfg) -> Frame:
    """`cam_wrist` optical frame (OpenCV/ROS) in the Canadarm3 last link."""
    mount = Frame.from_pos_quat((0.0, 0.0, cfg.mount_link_z_m), CAM_MOUNT_QUAT_ROS)
    return Frame(mount.pos, mount.rot @ rot_z(cfg.image_roll_deg))


def downsample(image: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter (area) downsampling by an integer factor.

    In continuous image coordinates (pixel centres at +0.5) a point at u in the large
    image lands at u / factor, so K of the output is K_large with fx, fy, cx, cy / factor.
    """
    if factor == 1:
        return image
    h, w = image.shape[:2]
    h2, w2 = h // factor, w // factor
    x = image[: h2 * factor, : w2 * factor].astype(np.float32)
    x = x.reshape(h2, factor, w2, factor, *image.shape[2:]).mean(axis=(1, 3))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def scale_intrinsics(k: np.ndarray, factor: int) -> np.ndarray:
    out = np.array(k, dtype=float)
    out[:2, :] /= factor
    return out


def intrinsics_from_usd(focal_length_mm: float, horizontal_aperture_mm: float, width: int, height: int) -> np.ndarray:
    """Pinhole K from USD camera attributes (square pixels, centred principal point).

    Same model as `isaaclab.sensors.Camera._update_intrinsic_matrices`; used to
    cross-check the K reported by the sensor.
    """
    fx = width * focal_length_mm / horizontal_aperture_mm
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])


##########################
### Tag constellation ###
##########################

_FAMILIES = {
    "tag36h11": "DICT_APRILTAG_36h11",
    "tag36h10": "DICT_APRILTAG_36h10",
    "tag25h9": "DICT_APRILTAG_25h9",
    "tag16h5": "DICT_APRILTAG_16h5",
}
# Width of a tag's black square in cells (data bits + 1-cell black border)
_FAMILY_CELLS = {"tag36h11": 8, "tag36h10": 8, "tag25h9": 7, "tag16h5": 6}


def aruco_dictionary(family: str):
    import cv2

    if family not in _FAMILIES:
        raise ValueError(f"Unsupported AprilTag family '{family}', use one of {sorted(_FAMILIES)}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, _FAMILIES[family]))


def tag_centres(cfg: AprilTagCfg) -> Dict[int, np.ndarray]:
    """Tag centre in the constellation frame T: Tag0..3 at (-d,+d), (+d,+d), (+d,-d), (-d,-d)."""
    d = cfg.constellation_half_width_m
    signs = [(-1, 1), (1, 1), (1, -1), (-1, -1)]
    return {tid: np.array([sx * d, sy * d, 0.0]) for tid, (sx, sy) in zip(cfg.ids, signs)}


def tag_object_corners(cfg: AprilTagCfg) -> Dict[int, np.ndarray]:
    """(4, 3) corners of each tag's black square in frame T, in OpenCV detection order.

    OpenCV returns corners clockwise from the tag image's top-left. The tag texture is
    applied with image-up = +Y_T and image-right = +X_T seen from outside the face
    (from +Z_T), so the order is (-,+), (+,+), (+,-), (-,-) around the tag centre.
    """
    h = cfg.tag_size_m / 2.0
    offs = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])
    return {tid: c + offs for tid, c in tag_centres(cfg).items()}


def check_constellation_layout(cfg: AprilTagCfg, keep_out_radius: float, face_half_extent: float) -> Tuple[bool, str]:
    """Tags (incl. quiet zone) must not cover the Cylinder_01 disc and must fit the face."""
    cells = _FAMILY_CELLS[cfg.family]
    half_quad = cfg.tag_size_m / 2.0 * (cells + 2 * cfg.quiet_zone_cells) / cells
    d = cfg.constellation_half_width_m
    inner = math.sqrt(2.0) * (d - half_quad)  # nearest quad corner to the centre
    outer = math.sqrt(2.0) * (d + half_quad)
    ok = inner > keep_out_radius and outer < face_half_extent and d > half_quad
    return ok, (
        f"tag quad (with quiet zone) {2 * half_quad * 1000:.0f} mm, nearest point to the "
        f"Cylinder_01 axis {inner * 1000:.1f} mm (> Cylinder_01 radius {keep_out_radius * 1000:.1f} mm), "
        f"farthest {outer * 1000:.0f} mm (< face half-size {face_half_extent * 1000:.0f} mm)"
    )


def make_tag_texture(cfg: AprilTagCfg, tag_id: int, path: Path) -> Path:
    """PNG of the tag with a white quiet zone; the black square spans the inner cells."""
    import cv2

    cells = _FAMILY_CELLS[cfg.family]
    ppc = cfg.texture_px_per_cell
    marker = cv2.aruco.generateImageMarker(aruco_dictionary(cfg.family), int(tag_id), cells * ppc, borderBits=1)
    q = cfg.quiet_zone_cells * ppc
    img = cv2.copyMakeBorder(marker, q, q, q, q, cv2.BORDER_CONSTANT, value=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def spawn_tag_constellation(stage, mep_prim_path: str, mep_scale: float, t_m_t: Frame, cfg: AprilTagCfg, texture_dir: Path) -> List[str]:
    """Visual-only textured quads (no collider, no mass) under the MEP body.

    `{mep}/vision_tags` undoes the MEP's uniform USD scale so everything below it is
    authored in metres in the MEP body frame; `constellation` carries T_M_T.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    root = f"{mep_prim_path}/vision_tags"
    xf = UsdGeom.Xform.Define(stage, root)
    xf.AddScaleOp().Set(Gf.Vec3d(1.0 / mep_scale, 1.0 / mep_scale, 1.0 / mep_scale))
    const = UsdGeom.Xform.Define(stage, f"{root}/constellation")
    cx = UsdGeom.Xformable(const)
    cx.AddTranslateOp().Set(Gf.Vec3d(*t_m_t.pos.tolist()))
    cx.AddOrientOp().Set(Gf.Quatf(*t_m_t.quat))

    cells = _FAMILY_CELLS[cfg.family]
    half = cfg.tag_size_m / 2.0 * (cells + 2 * cfg.quiet_zone_cells) / cells
    paths = []
    for tid, c in tag_centres(cfg).items():
        tex = make_tag_texture(cfg, tid, texture_dir / f"{cfg.family}_{tid:03d}.png")
        path = f"{root}/constellation/tag_{tid}"
        mesh = UsdGeom.Mesh.Define(stage, path)
        pts = [(c[0] - half, c[1] - half, 0.0), (c[0] + half, c[1] - half, 0.0), (c[0] + half, c[1] + half, 0.0), (c[0] - half, c[1] + half, 0.0)]
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.CreateExtentAttr([Gf.Vec3f(*pts[0]), Gf.Vec3f(*pts[2])])
        mesh.CreateDoubleSidedAttr(False)
        # st (0,0) = bottom-left of the image, so image-up = +Y_T and image-right = +X_T
        st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])

        mat_path = f"{path}_material"
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        reader = UsdShade.Shader.Define(stage, f"{mat_path}/stReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        texture = UsdShade.Shader.Define(stage, f"{mat_path}/texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        rgb = texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        ## NOTE: Purely self-lit (black diffuse, texture as emission). The pattern then
        ## does not depend on the sun direction, and the ray-traced lighting noise of
        ## a lit diffuse surface (which moved the detected edges by ~0.3 px from frame
        ## to frame while the arm moved) never reaches the tag.
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        paths.append(path)
    return paths


#################
### Detection ###
#################


@dataclass
class TagDetection:
    tag_id: int
    corners: np.ndarray  # (4, 2) px, OpenCV order (TL, TR, BR, BL of the tag image)

    @property
    def center(self) -> np.ndarray:
        return self.corners.mean(axis=0)


class AprilTagDetector:
    def __init__(self, cfg: AprilTagCfg):
        import cv2

        self.cfg = cfg
        params = cv2.aruco.DetectorParameters()
        ## NOTE: Measured on rendered `cam_wrist` frames against ground-truth corner
        ## projections: the AprilTag edge refinement gives the most accurate pose
        ## (0.002 deg vs 0.14 deg with SUBPIX at 1 m) and returns corners in the
        ## continuous image convention (pixel centres at +0.5) that Isaac Sim's
        ## K (cx = W/2, cy = H/2) uses, so K needs no half-pixel correction.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        # Small tags at the observation distance (~40 px): sample 8 px per cell when
        # decoding and allow small perimeters, otherwise decoding fails intermittently
        params.perspectiveRemovePixelPerCell = 8
        params.perspectiveRemoveIgnoredMarginPerCell = 0.25
        params.minMarkerPerimeterRate = 0.01
        self._detector = cv2.aruco.ArucoDetector(aruco_dictionary(cfg.family), params)

    def detect(self, image: np.ndarray) -> Dict[int, TagDetection]:
        import cv2

        gray = image if image.ndim == 2 else cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        out: Dict[int, TagDetection] = {}
        if ids is None:
            return out
        for c, i in zip(corners, ids.reshape(-1)):
            i = int(i)
            if i in self.cfg.ids and i not in out:  # ignore foreign IDs and duplicates
                out[i] = TagDetection(i, c.reshape(4, 2).astype(float))
        return out


###########
### PnP ###
###########


@dataclass
class ConstellationPose:
    t_c_t: Frame  # constellation in the camera frame
    num_tags: int
    num_points: int
    num_inliers: int
    reproj_rms_px: float
    tag_ids: List[int]
    ambiguous: bool = False  # both IPPE solutions fit about equally well


def estimate_constellation_pose(
    detections: Dict[int, TagDetection],
    cfg: AprilTagCfg,
    k: np.ndarray,
    dist: Optional[np.ndarray] = None,
    prior_rot_c: Optional[np.ndarray] = None,
) -> Optional[ConstellationPose]:
    """16-point (4 tags x 4 corners) planar PnP of the tag constellation.

    1. RANSAC (seeded with the best IPPE pose) removes corner outliers.
    2. IPPE returns *both* poses of the planar ambiguity; each is refined with
       Levenberg-Marquardt on the inliers.
    3. The lower reprojection error wins, unless the two are within
       `ambiguity_ratio` of each other: then the one closer to `prior_rot_c` (the
       expected rotation of the constellation in the camera frame, from the fused
       estimate) is used. Nearly face-on, the two solutions differ by a ~1 deg tilt
       with almost equal error, and image noise alone would flip between them.
    """
    import cv2

    ids = sorted(i for i in detections if i in cfg.ids)
    if len(ids) < cfg.min_tags_for_pose:
        return None
    obj_map = tag_object_corners(cfg)
    obj = np.vstack([obj_map[i] for i in ids]).astype(np.float64)
    img = np.vstack([detections[i].corners for i in ids]).astype(np.float64)
    dist = np.zeros(5) if dist is None else dist

    def ippe(idx):
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj[idx], img[idx], k, dist, flags=cv2.SOLVEPNP_IPPE)
        return list(zip(rvecs, tvecs))[:n]

    def rms_of(idx, rvec, tvec):
        proj, _ = cv2.projectPoints(obj[idx], rvec, tvec, k, dist)
        return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img[idx]) ** 2, axis=1))))

    inliers = np.arange(len(obj))
    cands = ippe(inliers)
    if not cands:
        return None
    if cfg.use_ransac:
        r0, t0 = min(cands, key=lambda c: rms_of(inliers, *c))
        ok_r, _, _, inl = cv2.solvePnPRansac(
            obj, img, k, dist, r0.copy(), t0.copy(), useExtrinsicGuess=True,
            iterationsCount=100, reprojectionError=cfg.ransac_reprojection_px, confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok_r and inl is not None and len(inl) >= 4 * cfg.min_tags_for_pose - 2 and len(inl) < len(obj):
            inliers = np.sort(inl.reshape(-1))
            cands = ippe(inliers) or cands
    refined = []
    for rvec, tvec in cands:
        rvec, tvec = cv2.solvePnPRefineLM(obj[inliers], img[inliers], k, dist, rvec.copy(), tvec.copy())
        f = Frame(tvec.reshape(3).astype(float), rotmat_from_rotvec(rvec))
        # The constellation must face the camera (its +Z towards the camera, in front of it)
        if f.pos[2] > 0.0 and f.rot[:, 2] @ (-f.pos) > 0.0:
            refined.append((rms_of(inliers, rvec, tvec), f))
    if not refined:
        return None
    refined.sort(key=lambda x: x[0])
    rms, t_c_t = refined[0]
    ambiguous = len(refined) > 1 and refined[1][0] <= cfg.ambiguity_ratio * max(refined[0][0], 1e-9)
    if ambiguous and prior_rot_c is not None:
        rms, t_c_t = min(refined, key=lambda x: rotation_angle(x[1].rot, prior_rot_c))
    return ConstellationPose(t_c_t, len(ids), len(obj), len(inliers), rms, ids, ambiguous)


def project_points(points_c: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Project (N, 3) camera-frame points to pixels (no distortion)."""
    p = np.asarray(points_c, dtype=float).reshape(-1, 3)
    uv = (k @ p.T).T
    return uv[:, :2] / uv[:, 2:3]


##################
### Prediction ###
##################


@dataclass
class Estimate:
    t: float  # simulation time of the image [s]
    t_w_y: Frame  # estimated Cylinder_01 pose (world)


class LinearDriftPredictor:
    """Constant-velocity model of the Cylinder_01 position (Phase 1: no rotation model).

    v is the least-squares slope of the estimated positions over the most recent
    `velocity_window_sec` (a noise-robust form of (p_t - p_{t-dt}) / dt); the current
    position is the fitted line at the query time and the future one is
    p(t + horizon) = p(t) + v * horizon. Orientation is the window mean (constant, not
    extrapolated); an angular rate is only computed for monitoring.
    """

    def __init__(self, cfg: PredictionCfg):
        self.cfg = cfg
        self.history: List[Estimate] = []  # velocity window
        self._rot_history: List[Estimate] = []  # orientation window

    def reset(self):
        self.history.clear()
        self._rot_history.clear()

    def add(self, est: Estimate):
        self.history.append(est)
        self.history = [e for e in self.history if e.t >= est.t - self.cfg.velocity_window_sec]
        self._rot_history.append(est)
        self._rot_history = [e for e in self._rot_history if e.t >= est.t - self.cfg.orientation_window_sec]

    @property
    def latest(self) -> Optional[Estimate]:
        return self.history[-1] if self.history else None

    def _fit(self) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        if len(self.history) < self.cfg.min_samples:
            return None
        t = np.array([e.t for e in self.history])
        p = np.array([e.t_w_y.pos for e in self.history])
        t0 = float(t.mean())
        a = np.column_stack((np.ones_like(t), t - t0))
        coef, *_ = np.linalg.lstsq(a, p, rcond=None)
        return t0, coef[0], coef[1]

    def velocity(self) -> Optional[np.ndarray]:
        fit = self._fit()
        if fit is None:
            return None
        v = fit[2]
        return v if np.linalg.norm(v) <= self.cfg.max_plausible_speed_mps else None

    def angular_velocity(self) -> np.ndarray:
        """Monitoring only: mean angular rate [rad/s] (world) over the window."""
        if len(self.history) < 2:
            return np.zeros(3)
        a, b = self.history[0], self.history[-1]
        dt = b.t - a.t
        if dt <= 0.0:
            return np.zeros(3)
        return rotvec_from_rotmat(b.t_w_y.rot @ a.t_w_y.rot.T) / dt

    def orientation(self) -> Optional[np.ndarray]:
        """Mean orientation over `orientation_window_sec` (chordal L2 mean on SO(3)).

        Phase 1 has no rotation, so this is the least-squares estimate of a constant
        orientation. Single-image planar PnP has an unbiased but noisy out-of-plane
        tilt (measured: mean |tilt| ~0.25 deg, max ~1.8 deg at ~1.4 m, bias < 0.07 deg)
        that stays correlated for ~1 s while the arm creeps; a 5 s mean removes it. With too few samples the latest estimate is used.
        """
        hist = self._rot_history
        if not hist:
            return None
        if len(hist) < self.cfg.min_samples:
            return hist[-1].t_w_y.rot.copy()
        m = np.sum([e.t_w_y.rot for e in hist], axis=0)
        u, _, vt = np.linalg.svd(m)
        d = np.sign(np.linalg.det(u @ vt))
        return u @ np.diag([1.0, 1.0, d]) @ vt

    def position_at(self, t_query: float) -> Optional[np.ndarray]:
        fit = self._fit()
        if fit is None or self.velocity() is None:
            return None
        t0, p0, v = fit
        return p0 + v * (t_query - t0)

    def estimate(self, t_now: float) -> Optional[Frame]:
        """Filtered Cylinder_01 pose now (fitted position, window-mean orientation)."""
        if self.latest is None:
            return None
        p = self.position_at(t_now)
        return Frame(self.latest.t_w_y.pos.copy() if p is None else p, self.orientation())

    def predict(self, t_now: float, horizon: Optional[float] = None) -> Optional[Frame]:
        """Cylinder_01 pose at t_now + horizon (position extrapolated, orientation held)."""
        h = self.cfg.horizon_sec if horizon is None else horizon
        p = self.position_at(t_now + h)
        if p is None or self.latest is None:
            return None
        return Frame(p, self.orientation())

    ## Same interface as `ConstantTwistPredictor` (Phase 1 behaviour unchanged)
    def orientation_at(self, t: float) -> Optional[np.ndarray]:
        return self.orientation()

    def velocity_at(self, p_w, t: float) -> Optional[np.ndarray]:
        """Velocity of a point rigidly attached to the MEP: no rotation model -> v."""
        return self.velocity()

    def twist(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """(v [m/s], w [rad/s]) used by the controller; no rotation model -> w = None."""
        return self.velocity(), None


def so3_exp(w) -> np.ndarray:
    """Exp([w]x): rotation matrix of the rotation vector w [rad]."""
    return rotmat_from_rotvec(w)


def so3_log(r: np.ndarray) -> np.ndarray:
    """Log of a rotation matrix as a rotation vector [rad] (|result| <= pi)."""
    return rotvec_from_rotmat(r)


def yaw_about_base(pose: Frame, base: Frame, deg: float) -> Frame:
    """`pose` rotated by `deg` about the base +Z axis through the base origin (azimuth).

    Pure rigid rotation of the whole tool pose about the arm's vertical axis: position
    and orientation both turn, so the camera on it looks at a different bearing.
    """
    axis = base.rot[:, 2]
    r = rotmat_from_rotvec(axis * math.radians(deg))
    return Frame(base.pos + r @ (pose.pos - base.pos), r @ pose.rot)


def rendezvous_start_pose(mep_nominal: Frame, t_m_y: Frame, v_w, w_w, t_r: float) -> Frame:
    """MEP start pose so that it drifts (v_w) and turns (world-frame w_w) into
    `mep_nominal` after `t_r` seconds, with the back-rotation Q = Exp(-[w] t_r) taken
    about Cylinder_01 (`t_m_y`): Cylinder_01 starts exactly at nominal - v t_r.
    With w = 0 this is exactly the Phase 1 pose (nominal - v t_r, nominal rotation)."""
    q = so3_exp(-np.asarray(w_w, dtype=float) * t_r)
    pivot = (mep_nominal @ t_m_y).pos
    pos = mep_nominal.pos + (q - np.eye(3)) @ (mep_nominal.pos - pivot) - np.asarray(v_w, dtype=float) * t_r
    return Frame(pos, q @ mep_nominal.rot)


class ConstantTwistPredictor:
    """6-DoF constant linear + angular velocity model (no learning).

    Everything is fitted on the pose of a *reference frame* rigidly attached to the MEP
    (the tag constellation T, i.e. what the PnP measures, or the MEP centre of mass) and
    mapped to Cylinder_01 with the fixed transform T_ref_Y, so a Cylinder_01 offset from
    the rotation centre moves on the correct arc:

        T_W_ref(t) = (p(t), R(t))              fitted over the recent estimates
        p(t + h)   = p(t) + v * h
        R(t + h)   = R(t) @ Exp([w_B] h),   w_B = R(t)^T w_W   (= Exp([w_W] h) @ R(t))
        T_W_Y(t + h) = T_W_ref(t + h) @ T_ref_Y

    Estimation (noise-robust forms of (p_t - p_{t-dt}) / dt and Log(R_{t-dt}^T R_t) / dt):
    - v: least-squares slope of the reference origin over `velocity_window_sec`
    - w: least-squares slope of phi_i = Log(R_i @ R_bar^T) over `orientation_window_sec`,
      where R_bar is the chordal mean of the window, so every phi_i is small. phi is a
      rotation vector of a *left* (world-frame) perturbation, so its slope is the angular
      velocity in the WORLD frame (the frame of Isaac Lab `root_com_ang_vel_w`).
      R(t) = Exp([w_W] (t - t0)) @ Exp([phi0]) @ R_bar.

    Model limits: constant w_W is exact for a spin about a principal axis; a torque-free
    tumble about a non-principal axis precesses (w_W slowly changes) -- the sliding window
    tracks that. In free flight only the centre of mass moves in a straight line; any other
    reference origin at distance r from it accelerates centripetally (a = |w|^2 r), which
    the constant-velocity fit misses by about a * (T^2/12 + T h/2 + h^2/2) with the
    velocity window T (0.1 mm for |w| = 0.5 deg/s, r = 2 m, T = 2 s, h = 0.3 s; 7 mm for
    |w| = 5 deg/s, r = 1.8 m). `prediction.reference_frame: mep_com` removes this bias.
    """

    def __init__(self, cfg: PredictionCfg, t_ref_y: Frame):
        self.cfg = cfg
        self.t_ref_y = t_ref_y
        self._t_y_ref = t_ref_y.inv()
        self.history: List[Estimate] = []  # velocity window (Cylinder_01 estimates, as received)
        self._rot_history: List[Estimate] = []  # orientation / angular-velocity window
        self._cache: Dict[str, object] = {}

    def reset(self):
        self.history.clear()
        self._rot_history.clear()
        self._cache.clear()

    def add(self, est: Estimate):
        self.history.append(est)
        self.history = [e for e in self.history if e.t >= est.t - self.cfg.velocity_window_sec]
        self._rot_history.append(est)
        self._rot_history = [e for e in self._rot_history if e.t >= est.t - self.cfg.orientation_window_sec]
        self._cache.clear()

    @property
    def latest(self) -> Optional[Estimate]:
        return self.history[-1] if self.history else None

    def _ref(self, e: Estimate) -> Frame:
        return e.t_w_y @ self._t_y_ref

    ## Fits (cached until the next estimate arrives)
    def _fit_position(self) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        if "pos" not in self._cache:
            fit = None
            if len(self.history) >= self.cfg.min_samples:
                t = np.array([e.t for e in self.history])
                p = np.array([self._ref(e).pos for e in self.history])
                t0 = float(t.mean())
                a = np.column_stack((np.ones_like(t), t - t0))
                coef, *_ = np.linalg.lstsq(a, p, rcond=None)
                fit = (t0, coef[0], coef[1])
            self._cache["pos"] = fit
        return self._cache["pos"]  # type: ignore

    def _fit_rotation(self) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """(t0, R(t0), w_W) of the reference frame, or None."""
        if "rot" not in self._cache:
            fit = None
            hist = self._rot_history
            if len(hist) >= self.cfg.min_samples:
                rots = [self._ref(e).rot for e in hist]
                u, _, vt = np.linalg.svd(np.sum(rots, axis=0))
                r_bar = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
                t = np.array([e.t for e in hist])
                phi = np.array([so3_log(r @ r_bar.T) for r in rots])
                t0 = float(t.mean())
                a = np.column_stack((np.ones_like(t), t - t0))
                coef, *_ = np.linalg.lstsq(a, phi, rcond=None)
                fit = (t0, so3_exp(coef[0]) @ r_bar, coef[1])
            self._cache["rot"] = fit
        return self._cache["rot"]  # type: ignore

    ## Twist
    def velocity(self) -> Optional[np.ndarray]:
        """Linear velocity [m/s] (world) of the reference origin; None unless the whole
        twist (linear and angular) is available and plausible."""
        fit = self._fit_position()
        if fit is None or self.angular_velocity() is None:
            return None
        v = fit[2]
        return v if np.linalg.norm(v) <= self.cfg.max_plausible_speed_mps else None

    def angular_velocity(self) -> Optional[np.ndarray]:
        """Angular velocity [rad/s] in the WORLD frame, or None (too few / implausible)."""
        fit = self._fit_rotation()
        if fit is None:
            return None
        w = fit[2]
        return w if np.linalg.norm(w) <= self.cfg.max_plausible_angular_rate_rad_s else None

    def twist(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        v = self.velocity()
        return v, (self.angular_velocity() if v is not None else None)

    ## Poses
    def reference_at(self, t: float) -> Optional[Frame]:
        """Fitted reference-frame pose at time t (None until the twist is available)."""
        fp, fr = self._fit_position(), self._fit_rotation()
        if fp is None or fr is None or self.velocity() is None:
            return None
        t0p, p0, v = fp
        t0r, r0, w = fr
        return Frame(p0 + v * (t - t0p), so3_exp(w * (t - t0r)) @ r0)

    def orientation_at(self, t: float) -> Optional[np.ndarray]:
        """Cylinder_01 orientation at time t (latest measurement until the fit exists)."""
        ref = self.reference_at(t)
        if ref is not None:
            return ref.rot @ self.t_ref_y.rot
        return None if self.latest is None else self.latest.t_w_y.rot.copy()

    def orientation(self) -> Optional[np.ndarray]:
        return None if self.latest is None else self.orientation_at(self.latest.t)

    def estimate(self, t_now: float) -> Optional[Frame]:
        """Filtered Cylinder_01 pose now (latest measurement until the fit exists)."""
        if self.latest is None:
            return None
        ref = self.reference_at(t_now)
        if ref is None:
            y = self.latest.t_w_y
            return Frame(y.pos.copy(), y.rot.copy())
        return ref @ self.t_ref_y

    def predict_reference(self, t_now: float, horizon: Optional[float] = None) -> Optional[Frame]:
        """Reference frame at t_now + horizon: p + v h, R Exp([R^T w_W] h)."""
        h = self.cfg.horizon_sec if horizon is None else horizon
        cur = self.reference_at(t_now)
        v, w = self.twist()
        if cur is None or v is None or w is None:
            return None
        w_body = cur.rot.T @ w
        return Frame(cur.pos + v * h, cur.rot @ so3_exp(w_body * h))

    def predict(self, t_now: float, horizon: Optional[float] = None) -> Optional[Frame]:
        """Cylinder_01 pose at t_now + horizon: T_W_ref(t + h) @ T_ref_Y."""
        ref = self.predict_reference(t_now, horizon)
        return None if ref is None else ref @ self.t_ref_y

    def velocity_at(self, p_w, t: float) -> Optional[np.ndarray]:
        """Velocity [m/s] of the MEP material point at `p_w` (world): v + w x (p - p_ref)."""
        ref = self.reference_at(t)
        v, w = self.twist()
        if ref is None or v is None or w is None:
            return None
        return v + np.cross(w, np.asarray(p_w, dtype=float) - ref.pos)
