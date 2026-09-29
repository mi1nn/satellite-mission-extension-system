"""Astrobee free-flyer: satellite observation / monitoring camera platform.

The Astrobee flies around the target satellite and keeps it -- and what happens around
it (MRV, Canadarm3, MEP, docking) -- in the view of its camera. That camera feed is its
only output (ROS 2 `sensor_msgs/Image`, `/<astrobee.ros_namespace>/<astrobee.image_topic>`).

What it does NOT do: no MEP search / capture, no AprilTag or marker detection, no pose
estimation or vision tracking, no satellite state / angular-velocity estimate, no docking
feasibility (no DOCKING_AVAILABLE / DOCKING_UNAVAILABLE), no mission decision, no command
to the Canadarm3 / MEP / docking pipeline. Nothing it does feeds back into the mission --
except the optional docking damping assist (`astrobee.assist.enabled`, off by default,
`astrobee_assist.py`): it grabs the probe root and damps the arm + payload swing with
thrust on the MEP during the aligning states, and lets go before the probe reaches the
nozzle. While engaged it replaces the observation flight below.

The one input it reacts to (`follow_mrv_after_dock`): the mission state on the existing
latched MRV topic `/<ros.namespace>/state` (std_msgs/String, `ros_interface.py`; with
ROS off, the same state string handed over in-process). Once it reports a docked state
(`DOCK_COMPLETE_STATES`), the observation loop stops for good and the Astrobee follows
the MRV's retreat: it moves by the MRV root displacement since that moment and turns
its camera onto the MRV / docking port (ASTROBEE_FOLLOW_MRV).

Motion: a simplified, kinematic 6-DoF flight (no propulsion, drag or gravity model; the
model is visual only -- no rigid body, no collider, so it cannot touch the scene). The
path is anchored at the satellite's *simulation* pose (ground truth, only to place the
path; it is not measured, estimated or published):

    ASTROBEE_IDLE (start_delay_s) -> ASTROBEE_APPROACH (straight in to point 1)
    -> ASTROBEE_OBSERVATION_START -> ASTROBEE_OBSERVING (dwell at each inspection point,
    smooth arc to the next one, `loops` times; 0 = until the run ends)
    -> ASTROBEE_OBSERVATION_COMPLETE (holds at the last point, camera still on)
    any of these -> ASTROBEE_FOLLOW_MRV (docking complete: follows the MRV retreat)

The states are internal (console log only): they are not published and not shown in the UI.

Frames: world W (Z up, metres). Astrobee body frame B as in the NASA description: +X
forward, +Y starboard, +Z down, origin at the body centre. The camera sits at the SciCam
mount of the NASA geometry config (`sci_cam_transform`, B frame, scaled with the model)
and looks along +X_B. Quaternions are (w, x, y, z).

The planner below is pure numpy (`project/tests/test_astrobee_observer.py`); the Isaac Sim
side (`AstrobeeObserver`) imports Isaac Lab / rclpy lazily.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .astrobee_assist import AstrobeeAssistCfg, DampingAssist, MepContext, validate_assist_cfg
from .frames import Frame

ASTROBEE_STATES = (
    "ASTROBEE_IDLE",
    "ASTROBEE_APPROACH",
    "ASTROBEE_OBSERVATION_START",
    "ASTROBEE_OBSERVING",
    "ASTROBEE_OBSERVATION_COMPLETE",
    "ASTROBEE_FOLLOW_MRV",
)

# Mission states (`vision_capture_demo.State` values on `/<ros.namespace>/state`) that mean
# "the MEP is docked": DOCKED itself lasts one control step and the topic is published at
# `ros.publish_rate_hz`, so the states that follow it count as well
DOCK_COMPLETE_STATES = frozenset({
    "DOCKED", "DOCK_HOLDING",  # static client
    "STABILIZING", "STOPPING", "ROBOT_RELEASE", "ARM_RETREAT", "MRV_SEPARATION",  # moving client
})

# Mission states in which the Astrobee closes in on the departing MRV (faster than it, so
# the shrinking distance is visible); before them it only keeps its offset to the MRV
CLOSE_IN_STATES = frozenset({"MRV_SEPARATION", "SUCCESS"})

# Planner phases (the observer turns the first "observe" step into ASTROBEE_OBSERVATION_START)
_PHASE_STATE = {
    "idle": "ASTROBEE_IDLE",
    "approach": "ASTROBEE_APPROACH",
    "observe": "ASTROBEE_OBSERVING",
    "complete": "ASTROBEE_OBSERVATION_COMPLETE",
}

# Camera optical axis in B: forward +X_B, image up = -Z_B (B is Z-down). Isaac Lab
# "world" camera convention is forward +X, up +Z, so the camera frame in B is Rx(pi).
CAM_IN_BODY_QUAT = (0.0, 1.0, 0.0, 0.0)

##############
### Config ###
##############


@dataclass
class AstrobeeCameraCfg:
    """RGB observation camera on the Astrobee (monitoring feed only, never processed)."""

    name: str = "cam_astrobee"
    width: int = 640
    height: int = 480
    horizontal_fov_deg: float = 70.0
    horizontal_aperture_mm: float = 20.955
    clipping_range_m: List[float] = field(default_factory=lambda: [0.1, 1000.0])
    # NASA `sci_cam_transform` translation (B frame, unscaled model metres)
    mount_pos_body_m: List[float] = field(default_factory=lambda: [0.118, 0.0, -0.096])
    # ROS 2 image rate [Hz] (simulation time)
    publish_rate_hz: float = 5.0
    # Save a PNG frame every this many seconds (0: off) under logs/<tag>_astrobee/
    save_every_s: float = 0.0

    @property
    def focal_length_mm(self) -> float:
        return self.horizontal_aperture_mm / (2.0 * math.tan(math.radians(self.horizontal_fov_deg) / 2.0))


@dataclass
class AstrobeeCfg:
    """`astrobee:` section of `vision_capture.yaml`."""

    enabled: bool = True
    # Under `assets/space_asset/` (built by `project/scripts/build_astrobee_usd.py`)
    usd_relpath: str = "astrobee/astrobee.usd"
    prim_name: str = "astrobee"
    # Display scale of the 0.32 m NASA model, like the other scaled assets of this scene
    # (satellite x2.4, MRV hull x3.4); the camera mount scales with it
    scale: float = 3.0
    # Path around the satellite. The horizontal ring radius is the satellite's
    # horizontal AABB half-diagonal + `orbit_margin_m`; the ring sits `elevation_deg`
    # above the satellite centre as seen from it
    start_delay_s: float = 0.0
    approach_distance_m: float = 20.0
    approach_speed_mps: float = 1.0
    orbit_margin_m: float = 25.0
    elevation_deg: float = 25.0
    # Inspection points, azimuth [deg] about world +Z measured from the horizontal
    # direction satellite centre -> docking port (0 = straight in front of the port,
    # where the MRV / MEP come in). Visited in list order, always turning the same way.
    inspection_azimuths_deg: List[float] = field(default_factory=lambda: [45.0, 135.0, 225.0, 315.0])
    dwell_s: float = 5.0
    transit_speed_mps: float = 2.0
    # Camera aim: 0 = satellite centre, 1 = docking port; in between keeps the satellite
    # and the docking area (MEP / MRV side) in the same view
    look_at_dock_weight: float = 0.5
    # Full loops over the inspection points before ASTROBEE_OBSERVATION_COMPLETE (0: forever)
    loops: int = 0
    # Docking complete (see the module docstring): stop observing and follow the MRV retreat.
    # The camera aim moves from the observation aim point to the midpoint MRV root <->
    # docking port within `follow_aim_blend_s` [s]
    follow_mrv_after_dock: bool = True
    follow_aim_blend_s: float = 2.0
    # MRV_SEPARATION (`CLOSE_IN_STATES`): on top of the MRV motion, the Astrobee closes
    # its distance to the MRV root at `follow_close_speed_mps` [m/s] (trapezoid profile,
    # `follow_close_accel_mps2` [m/s^2]) along the line towards the MRV, down to
    # `follow_standoff_m` [m]. Its world speed is then ~ MRV speed + closing speed.
    follow_close_speed_mps: float = 3.0
    follow_close_accel_mps2: float = 1.0
    follow_standoff_m: float = 10.0
    # ROS 2 (only when `ros.enabled`): camera image only
    ros_namespace: str = "astrobee"
    ros_node_name: str = "astrobee_observer"
    image_topic: str = "camera/image_raw"
    camera: AstrobeeCameraCfg = field(default_factory=AstrobeeCameraCfg)
    # Docking damping assist (`astrobee_assist.py`, off by default)
    assist: AstrobeeAssistCfg = field(default_factory=AstrobeeAssistCfg)


def validate_astrobee_cfg(cfg: AstrobeeCfg):
    c = cfg.camera
    if cfg.scale <= 0.0:
        raise ValueError("astrobee.scale must be > 0")
    if cfg.approach_speed_mps <= 0.0 or cfg.transit_speed_mps <= 0.0:
        raise ValueError("astrobee.approach_speed_mps / transit_speed_mps must be > 0")
    if cfg.approach_distance_m < 0.0 or cfg.orbit_margin_m < 0.0 or cfg.dwell_s < 0.0 or cfg.start_delay_s < 0.0:
        raise ValueError("astrobee distances / times must be >= 0")
    if not 0.0 <= cfg.look_at_dock_weight <= 1.0:
        raise ValueError("astrobee.look_at_dock_weight must be in [0, 1]")
    if not -80.0 <= cfg.elevation_deg <= 80.0:
        raise ValueError("astrobee.elevation_deg must be in [-80, 80]")
    if len(cfg.inspection_azimuths_deg) < 1:
        raise ValueError("astrobee.inspection_azimuths_deg needs at least one point")
    if cfg.follow_aim_blend_s < 0.0:
        raise ValueError("astrobee.follow_aim_blend_s must be >= 0")
    if cfg.follow_close_speed_mps < 0.0 or cfg.follow_close_accel_mps2 <= 0.0 or cfg.follow_standoff_m < 0.0:
        raise ValueError("astrobee.follow_close_speed_mps / follow_standoff_m must be >= 0, follow_close_accel_mps2 > 0")
    if int(cfg.loops) < 0:
        raise ValueError("astrobee.loops must be >= 0 (0 = forever)")
    if not cfg.ros_namespace.strip("/") or not cfg.image_topic.strip("/"):
        raise ValueError("astrobee.ros_namespace / image_topic must not be empty")
    cfg.ros_namespace = cfg.ros_namespace.strip("/")
    cfg.image_topic = cfg.image_topic.strip("/")
    if c.width < 16 or c.height < 16:
        raise ValueError("astrobee.camera.width / .height must be >= 16 px")
    if not 1.0 < c.horizontal_fov_deg < 179.0:
        raise ValueError("astrobee.camera.horizontal_fov_deg must be in (1, 179)")
    lo, hi = c.clipping_range_m
    if not 0.0 < lo < hi:
        raise ValueError("astrobee.camera.clipping_range_m must be [near, far] with 0 < near < far")
    if c.publish_rate_hz <= 0.0 or c.save_every_s < 0.0:
        raise ValueError("astrobee.camera.publish_rate_hz must be > 0 and save_every_s >= 0")
    if len(c.mount_pos_body_m) != 3:
        raise ValueError("astrobee.camera.mount_pos_body_m must be [x, y, z]")
    validate_assist_cfg(cfg.assist)


###############
### Planner ###
###############


def smoothstep(x: float) -> float:
    """C1 ease-in / ease-out on [0, 1] (zero velocity at both ends)."""
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def look_at_rotation(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation of the Astrobee body B (columns = axes in W): +X_B towards `target`,
    +Z_B (down) as close as possible to -`up`, +Y_B = Z_B x X_B (right-handed)."""
    x = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    n = np.linalg.norm(x)
    if n < 1e-9:
        return np.eye(3)
    x /= n
    down = -np.asarray(up, dtype=float)
    z = down - float(down @ x) * x
    if np.linalg.norm(z) < 1e-6:  # looking straight up / down: any horizontal "down"
        z = np.array([1.0, 0.0, 0.0]) - x[0] * x
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))


class ObservationPath:
    """Offsets from the satellite centre (world axes) as a function of time since start.

    Pure function of time, so the flight is smooth and repeatable; the satellite's own
    translation / rotation is added by the caller (`center_w`), never predicted here.
    """

    def __init__(self, cfg: AstrobeeCfg, ring_radius_m: float, ref_dir_w):
        self.cfg = cfg
        self.r = float(ring_radius_m)
        u = np.asarray(ref_dir_w, dtype=float).copy()
        u[2] = 0.0
        if np.linalg.norm(u) < 1e-9:
            u = np.array([1.0, 0.0, 0.0])
        self.u = u / np.linalg.norm(u)
        self.v = np.cross([0.0, 0.0, 1.0], self.u)
        self.h = self.r * math.tan(math.radians(cfg.elevation_deg))
        az = [math.radians(a) for a in cfg.inspection_azimuths_deg]
        self.az = az
        # Signed steps between consecutive points, always turning the same way (+)
        self.steps = [((az[(i + 1) % len(az)] - az[i]) % (2.0 * math.pi)) or (2.0 * math.pi if len(az) == 1 else 0.0)
                      for i in range(len(az))]
        self.transit_s = [self.r * s / cfg.transit_speed_mps for s in self.steps]
        self.leg_s = [cfg.dwell_s + t for t in self.transit_s]
        self.loop_s = sum(self.leg_s)
        self.approach_s = cfg.approach_distance_m / cfg.approach_speed_mps
        self.observe_t0 = cfg.start_delay_s + self.approach_s

    def ring_point(self, azimuth_rad: float) -> np.ndarray:
        return self.r * (math.cos(azimuth_rad) * self.u + math.sin(azimuth_rad) * self.v) + np.array([0.0, 0.0, self.h])

    def inspection_points(self) -> List[np.ndarray]:
        return [self.ring_point(a) for a in self.az]

    def approach_start(self) -> np.ndarray:
        p0 = self.ring_point(self.az[0])
        return p0 + self.cfg.approach_distance_m * p0 / np.linalg.norm(p0)

    def sample(self, t: float) -> Tuple[str, np.ndarray, int]:
        """(phase, offset from the satellite centre [m, world axes], inspection point index)."""
        c = self.cfg
        if t < c.start_delay_s:
            return "idle", self.approach_start(), 0
        if t < self.observe_t0 and self.approach_s > 0.0:
            s = smoothstep((t - c.start_delay_s) / self.approach_s)
            p0 = self.ring_point(self.az[0])
            return "approach", (1.0 - s) * self.approach_start() + s * p0, 0
        tau = t - self.observe_t0
        n = len(self.az)
        if self.loop_s <= 0.0:
            return ("complete" if c.loops > 0 else "observe"), self.ring_point(self.az[0]), 0
        loop = int(tau // self.loop_s)
        if c.loops > 0 and loop >= c.loops:
            return "complete", self.ring_point(self.az[0]), 0
        tau -= loop * self.loop_s
        for i in range(n):
            if tau < c.dwell_s:
                return "observe", self.ring_point(self.az[i]), i
            tau -= c.dwell_s
            if tau < self.transit_s[i]:
                a = self.az[i] + self.steps[i] * smoothstep(tau / self.transit_s[i])
                return "observe", self.ring_point(a), i
            tau -= self.transit_s[i]
        return "observe", self.ring_point(self.az[0]), 0


def ring_radius_from_aabb(aabb_min, aabb_max, margin_m: float) -> float:
    """Horizontal half-diagonal of a world AABB + margin [m]."""
    size = np.asarray(aabb_max, dtype=float) - np.asarray(aabb_min, dtype=float)
    return float(0.5 * math.hypot(size[0], size[1]) + margin_m)


def closing_distance(elapsed_s: float, total_m: float, speed_mps: float, accel_mps2: float) -> float:
    """Distance covered after `elapsed_s` [s] of a rest-to-rest trapezoid move of
    `total_m` [m] (cruise `speed_mps`, accel = decel `accel_mps2`; triangular if short)."""
    t, d = float(elapsed_s), float(total_m)
    if t <= 0.0 or d <= 0.0 or speed_mps <= 0.0:
        return 0.0
    a = float(accel_mps2)
    v = min(float(speed_mps), math.sqrt(a * d))  # peak speed
    ta = v / a
    tc = (d - v * ta) / v  # cruise time (0 for the triangle)
    if t < ta:
        return 0.5 * a * t * t
    if t < ta + tc:
        return 0.5 * v * ta + v * (t - ta)
    td = t - ta - tc
    if td < ta:
        return d - 0.5 * a * (ta - td) ** 2
    return d


def follow_mrv_pose(pos0, aim0, mrv0, mrv, dock, elapsed_s: float, blend_s: float,
                    closed_m: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """ASTROBEE_FOLLOW_MRV: (position, aim point) [m, world].

    The Astrobee keeps its offset to the MRV from the moment the docking completed
    (`pos0`, `mrv0`), i.e. it moves by the MRV displacement since then, minus `closed_m`
    [m] along that offset (closing in on the MRV, see `closing_distance`); the aim point
    blends (smoothstep over `blend_s`) from the observation aim `aim0` to the midpoint of
    the MRV root and the docking port, so the widening gap stays in view."""
    offset0 = np.asarray(pos0, dtype=float) - np.asarray(mrv0, dtype=float)
    n = float(np.linalg.norm(offset0))
    if closed_m > 0.0 and n > 1e-9:
        offset0 = offset0 * max(0.0, n - float(closed_m)) / n
    pos = np.asarray(mrv, dtype=float) + offset0
    target = 0.5 * (np.asarray(mrv, dtype=float) + np.asarray(dock, dtype=float))
    s = smoothstep(elapsed_s / blend_s) if blend_s > 0.0 else 1.0
    return pos, (1.0 - s) * np.asarray(aim0, dtype=float) + s * target


def camera_world_pose(body: Frame, cfg: AstrobeeCfg) -> Frame:
    """Camera frame in W (Isaac Lab "world" convention: +X forward, +Z up)."""
    mount = Frame.from_pos_quat(np.asarray(cfg.camera.mount_pos_body_m, dtype=float) * cfg.scale, CAM_IN_BODY_QUAT)
    return body @ mount


################
### Isaac Sim ###
################


class AstrobeeCameraPublisher:
    """ROS 2 publisher of the Astrobee camera image. Nothing else is published; the only
    subscription is the existing MRV mission state topic (docking-complete signal)."""

    def __init__(self, cfg: AstrobeeCfg, distro: str, mission_state_topic: Optional[str] = None):
        from .ros_interface import _import_rclpy

        rclpy = self.rclpy = _import_rclpy(distro)
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        self._image = Image
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        from rclpy.node import Node

        self.node = Node(cfg.ros_node_name, namespace=f"/{cfg.ros_namespace}", start_parameter_services=False)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        # Same QoS as the existing camera topics (`cam_wrist/image_raw`)
        self.pub = self.node.create_publisher(Image, cfg.image_topic, QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.frame_id = f"{cfg.ros_namespace}/{cfg.camera.name}"
        self.topic = f"/{cfg.ros_namespace}/{cfg.image_topic}"
        print(f"[ASTROBEE] ROS 2 camera feed on {self.topic} (sensor_msgs/Image rgb8)", flush=True)
        self.mission_state: Optional[str] = None
        if mission_state_topic:
            # Same QoS as the publisher (`ros_interface.py` `state`: latched)
            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.node.create_subscription(String, mission_state_topic, self._on_mission_state, latched)
            print(f"[ASTROBEE] docking-complete signal: {mission_state_topic} (std_msgs/String)", flush=True)

    def _on_mission_state(self, msg):
        self.mission_state = str(msg.data)

    def poll(self):
        """Deliver a pending mission-state message (non-blocking)."""
        self._executor.spin_once(timeout_sec=0.0)

    def publish(self, t: float, image: np.ndarray):
        from builtin_interfaces.msg import Time

        img = np.ascontiguousarray(image, dtype=np.uint8)
        m = self._image()
        sec = int(math.floor(t))
        m.header.stamp = Time(sec=sec, nanosec=int((t - sec) * 1e9))
        m.header.frame_id = self.frame_id
        m.height, m.width, m.encoding, m.is_bigendian = int(img.shape[0]), int(img.shape[1]), "rgb8", 0
        m.step = int(img.shape[1]) * 3
        m.data = img.tobytes()
        self.pub.publish(m)

    def close(self):
        try:
            self._executor.remove_node(self.node)
            self.node.destroy_node()
            if self._owns_context and self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception as e:  # never let ROS teardown hide the run result
            print(f"[ASTROBEE] ROS shutdown: {e}", flush=True)


class AstrobeeObserver:
    """Flies the Astrobee model + camera along `ObservationPath` and streams the image.

    `step(t)` before each physics step (poses for the next render), `after_render(t)`
    after `scene.update` (image grab / publish at `camera.publish_rate_hz`).
    """

    def __init__(self, task, cfg: AstrobeeCfg, ros_enabled: bool, ros_distro: str, headless: bool,
                 out_dir: Optional[Path] = None, label: str = "run", mission_state_topic: Optional[str] = None):
        import torch
        from pxr import Usd, UsdGeom

        from .docking import prim_frame

        self._torch = torch
        self.cfg = cfg
        self.task = task
        self.headless = headless
        self.xform = task.scene.extras[cfg.prim_name]
        self._xform_device = self.xform.get_world_poses()[0].device
        self.camera = task.scene[cfg.camera.name]
        self.sat = task._satellite
        self.mrv = task._robot  # MRV articulation root (moved by `MrvTransit`)
        env_path = task.scene.env_prim_paths[0]
        self.camera_path = f"{env_path}/{cfg.camera.name}"
        stage = task.scene.stage

        ## Satellite centre (AABB centre) in the satellite root frame, measured once on the
        ## USD stage; the live position then follows the simulated root pose
        geo = task.docking_geometry
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        box = cache.ComputeWorldBound(stage.GetPrimAtPath(geo.sat_prim_path)).ComputeAlignedRange()
        lo, hi = np.array(box.GetMin(), dtype=float), np.array(box.GetMax(), dtype=float)
        sat_usd = prim_frame(stage, geo.sat_prim_path)
        self.center_in_sat = sat_usd.inv().point(0.5 * (lo + hi))
        radius = ring_radius_from_aabb(lo, hi, cfg.orbit_margin_m)
        self.sat_dock = geo.sat_dock  # satellite root frame (docking.py SAT_DOCK_POINT)
        center_w = self.center_w()
        dock_w = (self.sat_frame() @ geo.sat_dock).pos
        self.path = ObservationPath(cfg, radius, dock_w - center_w)
        self.ros: Optional[AstrobeeCameraPublisher] = None
        if ros_enabled:
            self.ros = AstrobeeCameraPublisher(cfg, ros_distro, mission_state_topic if cfg.follow_mrv_after_dock else None)
        self.state: Optional[str] = None
        self.frames = 0
        self.saved = 0
        self._next_pub_t = 0.0
        self._next_save_t = 0.0
        self._point = -1
        self._window = None
        self._aim: Optional[np.ndarray] = None
        # ASTROBEE_FOLLOW_MRV: set once, when the docking-complete signal arrives
        self._follow: Optional[dict] = None
        # Flight speed for the GUI "Docking monitor" (display only): from consecutive
        # commanded poses, world frame and relative to the satellite centre [m/s]
        self.speed_mps = 0.0
        self.rel_speed_mps = 0.0
        self._last: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        self._vel = np.zeros(3)
        ## Docking damping assist (`astrobee_assist.py`, `assist.enabled`): the caller reads
        ## `assist_force_w` / `assist_grip_w` after `step` and applies the thrust to the MEP
        self.assist = DampingAssist(cfg.assist) if cfg.assist.enabled else None
        self.assist_force_w: Optional[np.ndarray] = None
        self.assist_grip_w: Optional[np.ndarray] = None
        self.save_dir: Optional[Path] = None
        if cfg.camera.save_every_s > 0.0 and out_dir is not None:
            self.save_dir = Path(out_dir) / f"{label}_astrobee"
            self.save_dir.mkdir(parents=True, exist_ok=True)
            for old in self.save_dir.glob("*.png"):
                old.unlink()
        print(f"[ASTROBEE] observation ring: radius {radius:.1f} m, {self.path.h:.1f} m above the satellite centre "
              f"{np.round(center_w, 2).tolist()} (satellite AABB {np.round(hi - lo, 1).tolist()} m), "
              f"{len(cfg.inspection_azimuths_deg)} inspection points, approach {self.path.approach_s:.0f} s, "
              f"one loop {self.path.loop_s:.0f} s, camera {cfg.camera.width}x{cfg.camera.height} "
              f"FOV {cfg.camera.horizontal_fov_deg:g} deg", flush=True)

    def sat_frame(self) -> Frame:
        d = self.sat.data
        return Frame.from_pos_quat(d.root_pos_w[0].tolist(), d.root_quat_w[0].tolist())

    def center_w(self) -> np.ndarray:
        return self.sat_frame().point(self.center_in_sat)

    def _set_state(self, state: str, detail: str = ""):
        if state != self.state:
            self.state = state
            print(f"[ASTROBEE] {state}{'  ' + detail if detail else ''}", flush=True)

    def mrv_pos(self) -> np.ndarray:
        return self.mrv.data.root_pos_w[0].cpu().numpy().astype(float)

    def _check_dock_signal(self, t: float, mission_state: Optional[str]):
        """Docking-complete signal: the MRV state topic when ROS is on, else `mission_state`."""
        if self._follow is not None or not self.cfg.follow_mrv_after_dock or self._last is None:
            return
        if self.ros is not None:
            self.ros.poll()
            mission_state = self.ros.mission_state
        if mission_state in DOCK_COMPLETE_STATES:
            self._follow = {"t0": t, "pos0": self._last[1].copy(), "aim0": self._aim.copy(), "mrv0": self.mrv_pos()}
            print(f"[ASTROBEE] docking complete ({mission_state}"
                  f"{' on ROS' if self.ros is not None else ''}): observation loop stopped at "
                  f"{np.round(self._last[1], 2).tolist()}, following the MRV from {np.round(self._follow['mrv0'], 2).tolist()}", flush=True)

    def step(self, t: float, mission_state: Optional[str] = None, mep_ctx: Optional[MepContext] = None):
        """Place the Astrobee and its camera for time `t` [s, simulation].

        `mission_state`: the mission state in-process, used as the docking-complete
        signal only when ROS is off (with ROS the MRV state topic is). The damping
        assist always uses the in-process state: it acts on the in-process MEP
        (`mep_ctx`, None = no MEP data this step: an engaged assist holds its pose)."""
        torch = self._torch
        mission_now = mission_state
        self._check_dock_signal(t, mission_state)
        if self.ros is not None and self._follow is not None and self._follow.get("close_t0") is None:
            # Close-in signal (`CLOSE_IN_STATES`): same source as the docking-complete one
            self.ros.poll()
            mission_state = self.ros.mission_state
        sat = self.sat_frame()
        center = sat.point(self.center_in_sat)
        dock = (sat @ self.sat_dock).pos
        if self._follow is not None:
            f = self._follow
            phase, idx = "follow", self._point
            if f.get("close_t0") is None and mission_state in CLOSE_IN_STATES:
                f["close_t0"] = t
                f["close_total_m"] = max(0.0, float(np.linalg.norm(f["pos0"] - f["mrv0"])) - self.cfg.follow_standoff_m)
                print(f"[ASTROBEE] {mission_state}: closing in on the MRV, {f['close_total_m']:.1f} m at "
                      f"{self.cfg.follow_close_speed_mps:.1f} m/s relative (standoff {self.cfg.follow_standoff_m:.1f} m)", flush=True)
            closed = 0.0
            if f.get("close_t0") is not None:
                closed = closing_distance(t - f["close_t0"], f["close_total_m"],
                                          self.cfg.follow_close_speed_mps, self.cfg.follow_close_accel_mps2)
            pos, aim = follow_mrv_pose(f["pos0"], f["aim0"], f["mrv0"], self.mrv_pos(), dock,
                                       t - f["t0"], self.cfg.follow_aim_blend_s, closed)
            offset = pos - center
        else:
            phase, offset, idx = self.path.sample(t)
            pos = center + offset
            aim = center + self.cfg.look_at_dock_weight * (dock - center)
        self.assist_force_w = None
        if self.assist is not None and self._follow is None and self._last is not None:
            dt = max(t - self._last[0], 0.0)
            out = self.assist.step(t, dt, mission_now, self._last[1], self._vel, mep_ctx) if dt > 0.0 and mep_ctx is not None else None
            if out is not None:
                phase, pos, aim = "assist", out.pos, out.aim
                self.assist_force_w, self.assist_grip_w = out.force_w, out.grip_w
            elif self.assist.phase is not None:  # engaged but no MEP data: hold still
                phase, pos, aim = "assist", self._last[1].copy(), self._aim
            if phase == "assist":
                offset = pos - center
        self._aim = np.asarray(aim, dtype=float)
        body = Frame(pos, look_at_rotation(pos, aim))
        if self._last is not None and t > self._last[0]:
            dt = t - self._last[0]
            self.speed_mps = float(np.linalg.norm(pos - self._last[1])) / dt
            self.rel_speed_mps = float(np.linalg.norm(offset - self._last[2])) / dt
            self._vel = (pos - self._last[1]) / dt
        self._last = (t, pos.copy(), np.asarray(offset, dtype=float).copy())
        if phase == "follow":
            self._set_state("ASTROBEE_FOLLOW_MRV")
        elif phase == "assist":
            self.state = self.assist.phase  # `DampingAssist` logs its own transitions
        elif phase == "observe" and self.state in (None, "ASTROBEE_IDLE", "ASTROBEE_APPROACH"):
            self._set_state("ASTROBEE_OBSERVATION_START", "at inspection point 1")
        else:
            self._set_state(_PHASE_STATE[phase])
        if phase == "observe" and idx != self._point:
            self._point = idx
            print(f"[ASTROBEE] inspection point {idx + 1}/{len(self.path.az)} "
                  f"(azimuth {self.cfg.inspection_azimuths_deg[idx]:g} deg)", flush=True)
        dev = self._xform_device
        self.xform.set_world_poses(
            positions=torch.tensor([pos.tolist()], dtype=torch.float32, device=dev),
            orientations=torch.tensor([list(body.quat)], dtype=torch.float32, device=dev),
        )
        cam = camera_world_pose(body, self.cfg)
        cdev = self.camera.device
        self.camera.set_world_poses(
            positions=torch.tensor([cam.pos.tolist()], dtype=torch.float32, device=cdev),
            orientations=torch.tensor([list(cam.quat)], dtype=torch.float32, device=cdev),
            convention="world",
        )

    def after_render(self, t: float):
        """Grab the rendered image and publish it (rate limited)."""
        c = self.cfg.camera
        publish_due = self.ros is not None and t >= self._next_pub_t
        save_due = self.save_dir is not None and t >= self._next_save_t
        if not (publish_due or save_due):
            return
        rgb = self.camera.data.output["rgb"][0].cpu().numpy()[..., :3]
        if publish_due:
            self._next_pub_t = t + 1.0 / min(float(c.publish_rate_hz), 5.0) - 1e-9
            self.ros.publish(t, rgb)
            self.frames += 1
        if save_due:
            import cv2

            self._next_save_t = t + c.save_every_s - 1e-9
            cv2.imwrite(str(self.save_dir / f"astrobee_{t:08.2f}s.png"), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
            self.saved += 1

    def open_window(self):
        """GUI: live viewport window of the Astrobee camera."""
        if self.headless:
            return
        try:
            from omni.kit.viewport.utility import create_viewport_window
            from pxr import Sdf

            self._window = create_viewport_window(
                self.cfg.camera.name, width=640, height=480, position_x=700, position_y=60,
                camera_path=Sdf.Path(self.camera_path),
            )
            print(f"[ASTROBEE] opened the '{self.cfg.camera.name}' viewport window", flush=True)
        except Exception as e:  # never let a GUI convenience stop the run
            print(f"[ASTROBEE] could not open the camera window: {e}", flush=True)

    def complete(self):
        """End of the run: the observation stops with it."""
        self._set_state("ASTROBEE_OBSERVATION_COMPLETE")

    def close(self):
        print(f"[ASTROBEE] {self.frames} camera frames published"
              f"{f', {self.saved} saved to {self.save_dir}' if self.save_dir is not None else ''}", flush=True)
        if self.ros is not None:
            self.ros.close()
            self.ros = None
