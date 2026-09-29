"""MEP Dashboard backend: Firestore validation + live ROS2 telemetry."""
import asyncio
import io
import json
import math
import os
import re
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles


# firebase_admin pulls in `cryptography`, which aborts at import time when the
# OpenSSL 3 legacy provider is unavailable on this host. None of the Firestore
# code paths need legacy algorithms, so opt out before the import happens.
os.environ.setdefault("CRYPTOGRAPHY_OPENSSL_NO_LEGACY", "1")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
REPO_ROOT = Path(__file__).resolve().parents[2]
SESSIONS = "simulation_sessions"
TELEMETRY = "session_telemetry"
DEFAULT_CREDENTIALS = REPO_ROOT / "serviceAccount.json"
RUN_VIDEO_ROOT = Path(
    os.environ.get(
        "MRV_RUN_VIDEO_DIR",
        REPO_ROOT / "project" / "logs" / "vision_capture",
    )
).resolve()
CACHE_ROOT = Path(
    os.environ.get(
        "MEP_DASHBOARD_CACHE_DIR",
        Path(__file__).resolve().parent.parent / ".cache" / "validation",
    )
).resolve()
# The run list grows while the bridge records, so it is only cached briefly;
# a finished run's telemetry never changes and is cached until deleted.
RUNS_CACHE_TTL_S = float(os.environ.get("MEP_DASHBOARD_RUNS_TTL_S", "60"))
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
IMAGE_STALE_TIMEOUT_S = 3.0
# Astrobee satellite map (/astrobee/map/points, /astrobee/map/clearance): latched,
# low-rate topics. The web gets the snapshot as one binary WS frame (see
# `encode_map_frame`); larger maps are thinned (obstruction voxels always kept).
ASTROBEE_MAP_POINTS_TOPIC = "/astrobee/map/points"
ASTROBEE_MAP_CLEARANCE_TOPIC = "/astrobee/map/clearance"
MAP_MAX_POINTS = 250_000
MAP_FRAME_MAGIC = b"ABM1"

app = FastAPI(
    title="MEP Dashboard · Firestore Validation + ROS2 Live",
    docs_url=None,
    redoc_url=None,
)

app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


# ============================================================
# Firestore
# ============================================================

def json_value(value: Any):
    """Serialize Firestore types and turn NaN/inf into null for the browser."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


# ============================================================
# Validation cache
# ============================================================
#
# Firestore bills per document read: the run list is one read per session and a
# run's telemetry is one read per row (800-2000). Without a cache every click
# re-reads the whole run, which exhausts the project's daily read quota
# (`ResourceExhausted: 429 Quota exceeded`) after a few dozen clicks.
#
# A finished run is immutable, so its telemetry is cached on disk and served
# with zero reads afterwards. A cached copy is also the fallback whenever
# Firestore is unreachable or out of quota, so the dashboard degrades to stale
# data instead of an error.


def cache_read(name: str):
    try:
        with (CACHE_ROOT / name).open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def cache_write(name: str, payload: dict):
    try:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        partial = CACHE_ROOT / f"{name}.part"
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        partial.replace(CACHE_ROOT / name)
    except (OSError, ValueError) as exc:
        print(f"[cache] write failed for {name}: {type(exc).__name__}: {exc}", flush=True)


def telemetry_cache_name(session_id: str) -> str:
    return f"telemetry_{session_id}.json"


def session_is_finished(db, session_id: str) -> bool:
    """A run still being recorded must not be cached as final.

    Answered from the cached run list when possible; only an unknown session
    costs a single document read.
    """
    cached = cache_read("runs.json") or {}

    for run in cached.get("runs", []):
        if run.get("session_id") == session_id:
            return not run.get("is_running")

    try:
        document = db.collection(SESSIONS).document(session_id).get()
    except Exception as exc:
        print(f"[cache] run state unknown for {session_id}: {type(exc).__name__}: {exc}", flush=True)
        return False

    # Runs recorded before `is_running` existed carry no flag and are finished.
    return not (document.to_dict() or {}).get("is_running")


def firestore_client():
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            credential_path = (
                os.environ.get("FIREBASE_CREDENTIALS")
                or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            )

            # The repo ships the same service account the pipeline writes with,
            # so the dashboard works without Application Default Credentials.
            if not credential_path and DEFAULT_CREDENTIALS.is_file():
                credential_path = str(DEFAULT_CREDENTIALS)

            if credential_path:
                credential = credentials.Certificate(credential_path)
            elif os.environ.get("FIRESTORE_EMULATOR_HOST"):
                credential = None
            else:
                credential = credentials.ApplicationDefault()

            options = (
                {"projectId": os.environ["FIREBASE_PROJECT_ID"]}
                if os.environ.get("FIREBASE_PROJECT_ID")
                else None
            )

            firebase_admin.initialize_app(credential, options)

        return firestore.client()

    except Exception as exc:
        # Surface the cause: a swallowed message here looks identical to an
        # empty database from the browser.
        print(f"[firestore] client init failed: {type(exc).__name__}: {exc}", flush=True)
        raise HTTPException(
            status_code=503,
            detail=f"Firebase Connection Error: {type(exc).__name__}: {exc}",
        ) from exc


# ============================================================
# LIVE ROS2 mapping
# ============================================================

# The UI contract always contains seven stages. Stage 7 is intentionally not
# mapped from any current simulator state because orbit transfer is not yet
# implemented; completed docking runs therefore stop at stage 6.
STAGE_LABELS = {
    1: "MRV MOVE + ASTROBEE",
    2: "MEP SEARCH",
    3: "MEP ATTACH",
    4: "DOCK PREP",
    5: "DOCKING",
    6: "DOCK COMPLETE",
    7: "ORBIT TRANSFER",
}
TOTAL_STAGES = 7

STAGE1_STATES = {
    "INIT", "MRV_MOVE_STEP_1", "MRV_STEP_1_REACHED",
    "MRV_MOVE_STEP_2", "MRV_STEP_2_REACHED", "ARM_DEPLOY",
}

STAGE2_STATES = {
    "SEARCH", "TAG_DETECTED", "POSE_ESTIMATED", "PREDICTING",
    "APPROACHING", "SLOW_APPROACH", "CAPTURE_ATTEMPT",
}

STAGE3_STATES = {"CAPTURED", "HOLDING", "RETREAT"}

# Moving-client states before the docking-axis approach (chasing / matching
# velocity with the drifting satellite while still holding the MEP) count as
# "dock prep", same as the stationary-target DOCK_TARGET_ACQUIRE / PRE_DOCK_APPROACH.
STAGE4_STATES = {
    "DOCK_TARGET_ACQUIRE", "PRE_DOCK_APPROACH", "XY_ALIGN",
    "ORIENTATION_ALIGN", "POSITION_ATTITUDE_ALIGN", "ALIGNMENT_CHECK",
    "CLIENT_RELEASE", "CLIENT_CRUISE", "CHASE", "VELOCITY_MATCHING", "RENDEZVOUS",
}

STAGE5_STATES = {"Z_APPROACH", "FINAL_INSERTION", "DOCK_READY", "DOCKED"}

# Moving-client states after the docking joint exists (stabilise / stop / release
# the robot / MRV departs) are still "dock complete", same as DOCK_HOLDING.
STAGE6_STATES = {
    "DOCK_HOLDING", "STABILIZING", "STOPPING",
    "ROBOT_RELEASE", "ARM_RETREAT", "MRV_SEPARATION",
}

# Failure states keep the stage they failed in -- never reset the UI to stage 1.
FAILURE_STAGE = {
    "TAG_LOST": 2, "POSE_INVALID": 2, "PREDICTION_INVALID": 2,
    "APPROACH_TIMEOUT": 2, "CAPTURE_FAILED": 2, "PHYSICS_ERROR": 2,
    "MRV_APPROACH_FAILED": 1,
    "DOCK_FAILED": 5,
    "CLIENT_RELEASE_FAILED": 4, "VELOCITY_MATCH_TIMEOUT": 4, "RENDEZVOUS_TIMEOUT": 4,
    "STOP_FAILED": 6, "ROBOT_RELEASE_FAILED": 6, "ARM_RETREAT_FAILED": 6,
    "SEPARATION_COLLISION": 6, "SEPARATION_FAILED": 6,
}
# ABORTED carries no stage of its own (manual stop from any state): the
# caller passes the previous stage through `last_progress`.

# Any state where the dock_* fields (rather than the capture gt_* fields)
# are the meaningful telemetry to show.
DOCKING_PHASE_STATES = STAGE4_STATES | STAGE5_STATES | STAGE6_STATES | {
    "DOCK_FAILED", "DOCKING_UNAVAILABLE",
    "CLIENT_RELEASE_FAILED", "VELOCITY_MATCH_TIMEOUT", "RENDEZVOUS_TIMEOUT",
    "STOP_FAILED", "ROBOT_RELEASE_FAILED", "ARM_RETREAT_FAILED",
    "SEPARATION_COLLISION", "SEPARATION_FAILED",
}


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stage_for_state(state: str, dock_enabled: bool, last_progress: "int | None"):
    """Returns (stage 1-6, label, is_failure); stage 7 is not implemented."""
    if state in STAGE1_STATES:
        return 1, STAGE_LABELS[1], False
    if state in STAGE2_STATES:
        return 2, STAGE_LABELS[2], False
    if state in STAGE3_STATES:
        return 3, STAGE_LABELS[3], False
    if state in STAGE4_STATES:
        return 4, STAGE_LABELS[4], False
    if state in STAGE5_STATES:
        return 5, STAGE_LABELS[5], False
    if state in STAGE6_STATES:
        return 6, STAGE_LABELS[6], False
    if state == "SUCCESS":
        # Capture-only run (docking.enabled=false) ends at stage 3;
        # a docking run ends at stage 6.
        stage = 6 if dock_enabled else 3
        return stage, STAGE_LABELS[stage], False
    # Stopped wherever the mission was: a manual abort, or the Astrobee finding the
    # docking port blocked (it can stop any phase) -- keep the stage it stopped in
    if state in ("ABORTED", "DOCKING_UNAVAILABLE"):
        stage = last_progress or 1
        return stage, STAGE_LABELS.get(stage, STAGE_LABELS[1]), True
    if state in FAILURE_STAGE:
        stage = FAILURE_STAGE[state]
        return stage, STAGE_LABELS[stage], True
    # Unknown / not-yet-seen state: do not guess.
    return 1, STAGE_LABELS[1], False


def normalize_live_status(raw: dict, last_progress: "int | None" = None):
    state = str(raw.get("state") or "UNKNOWN")
    dock_enabled = bool(raw.get("dock_enabled"))
    progress, phase, is_failure = stage_for_state(state, dock_enabled, last_progress)

    docking = bool(raw.get("dock_active")) or state in DOCKING_PHASE_STATES

    if docking:
        rx = finite_number(raw.get("dock_relative_x"))
        ry = finite_number(raw.get("dock_relative_y"))
        rz = finite_number(raw.get("dock_relative_z"))

        if None not in (rx, ry, rz):
            position_error = math.sqrt(rx * rx + ry * ry + rz * rz)
        else:
            position_error = None

        orientation_error = finite_number(raw.get("dock_orientation_deg"))
        total_velocity = finite_number(raw.get("dock_rel_speed"))

        # Current /mrv/status does not publish a docking relative angular velocity.
        # Do not substitute an unrelated capture angular metric.
        total_angular_velocity = None

        remaining_distance = finite_number(raw.get("dock_distance"))

    else:
        position_error = finite_number(
            raw.get("gt_capture_position_error_m")
        )

        orientation_error = finite_number(
            raw.get("gt_capture_orientation_error_deg")
        )

        total_velocity = finite_number(
            raw.get("gt_relative_velocity_mps")
        )

        angular_rad = finite_number(
            raw.get("gt_relative_angular_velocity_rad_s")
        )

        total_angular_velocity = (
            math.degrees(angular_rad)
            if angular_rad is not None
            else None
        )

        remaining_distance = finite_number(
            raw.get("gt_capture_remaining_distance_m")
        )

    return {
        "connected": True,
        "source": "ros2:/mrv/status",
        "sim_time_s": finite_number(raw.get("sim_time_s")),
        "state": state,
        "phase": phase,
        "progress": progress,
        "total_steps": TOTAL_STAGES,
        "is_failure": is_failure,
        "position_error": position_error,
        "orientation_error_deg": orientation_error,
        "total_velocity_mps": total_velocity,
        "total_angular_velocity_deg_s": total_angular_velocity,
        "remaining_distance_m": remaining_distance,
        "captured": bool(raw.get("captured")),
        "docked": bool(raw.get("dock_docked")),
    }


def cloud_to_arrays(msg):
    """(N, 3) float32 xyz and (N,) bool obstruction flags of a PointCloud2 with float32
    x, y, z (+ optional `obstruction`) fields."""
    offsets = {f.name: f.offset for f in msg.fields}
    n = int(msg.width) * int(msg.height)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)[: n * msg.point_step].reshape(n, msg.point_step)

    def column(name):
        return raw[:, offsets[name]:offsets[name] + 4].copy().view("<f4")[:, 0]

    xyz = np.column_stack([column(k) for k in ("x", "y", "z")]).astype("<f4")
    flags = column("obstruction") > 0.5 if "obstruction" in offsets else np.zeros(n, dtype=bool)
    ok = np.isfinite(xyz).all(axis=1)
    return xyz[ok], flags[ok]


def encode_map_frame(seq: int, xyz, flags, max_points: int = MAP_MAX_POINTS) -> bytes:
    """Binary WS frame of a map snapshot (little endian):
    b"ABM1", uint32 seq, uint32 n, float32 xyz[n * 3], uint8 obstruction[n].
    Above `max_points` the free voxels are thinned evenly; obstruction voxels stay."""
    xyz = np.asarray(xyz, dtype="<f4").reshape(-1, 3)
    flags = np.asarray(flags, dtype=bool).reshape(-1)
    if len(xyz) > max_points:
        hit = np.flatnonzero(flags)
        free = np.flatnonzero(~flags)
        keep = max(0, max_points - len(hit))
        free = free[np.linspace(0, len(free) - 1, keep).astype(np.int64)] if keep and len(free) else free[:0]
        idx = np.sort(np.concatenate((hit, free)))
        xyz, flags = xyz[idx], flags[idx]
    header = MAP_FRAME_MAGIC + np.array([seq, len(xyz)], dtype="<u4").tobytes()
    return header + xyz.tobytes() + flags.astype(np.uint8).tobytes()


class RosLiveBridge:
    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import Empty, String
        from sensor_msgs.msg import Image, PointCloud2
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

        self.rclpy = rclpy
        self.Empty = Empty
        self.String = String

        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0

        # Camera 1: real Isaac Sim wrist camera.
        self._camera1_jpeg = None
        self._camera1_seq = 0
        self._camera1_last_rx = 0.0
        self._camera1_last_encode = 0.0

        self._camera2_jpeg = None
        self._camera2_seq = 0
        self._camera2_last_rx = 0.0
        self._camera2_last_encode = 0.0

        # Camera 3: Astrobee observation camera (separate /astrobee/ namespace,
        # publish_rate_hz 5 in vision_capture.yaml -- so the offline timeout is longer).
        self._camera3_jpeg = None
        self._camera3_seq = 0
        self._camera3_last_rx = 0.0
        self._camera3_last_encode = 0.0

        # Astrobee satellite map: latest snapshot (binary WS frame) and clearance.
        self._map_frame = None
        self._map_seq = 0
        self._map_points = 0
        self._map_status = None
        self._map_status_seq = 0
        self._last_sim_time = None
        self._map_last_rx = 0.0

        # Main Isaac Sim 3D viewport (API capture, not screen share).
        self._viewport_jpeg = None
        self._viewport_seq = 0
        self._viewport_last_rx = 0.0
        self._viewport_last_encode = 0.0

        self._running = True

        self._owns_context = not rclpy.ok()

        if self._owns_context:
            rclpy.init()

        self.node = Node("mep_dashboard_live_bridge")
        viewport_sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.node.create_subscription(
            String,
            "/mrv/status",
            self._status_callback,
            10,
        )

        self.node.create_subscription(
            Image,
            "/mrv/cam_wrist/image_raw",
            self._camera1_callback,
            qos_profile_sensor_data,
        )

        self.node.create_subscription(
            Image,
            "/mrv/cam_probe/image_raw",
            self._camera2_callback,
            qos_profile_sensor_data,
        )

        self.node.create_subscription(
            Image,
            "/astrobee/camera/image_raw",
            self._camera3_callback,
            qos_profile_sensor_data,
        )

        self.node.create_subscription(
            Image,
            "/mrv/viewport/image_raw",
            self._viewport_callback,
            viewport_sensor_qos,
        )

        # Same QoS as the Astrobee publishers (latched): a late start gets the latest.
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # The map snapshot is a large message: a node + executor thread of its own, so
        # taking / converting it never holds up the camera and viewport image callbacks
        self.map_node = Node("mep_dashboard_map_bridge")

        self.map_node.create_subscription(
            PointCloud2,
            ASTROBEE_MAP_POINTS_TOPIC,
            self._map_points_callback,
            map_qos,
        )

        self.map_node.create_subscription(
            String,
            ASTROBEE_MAP_CLEARANCE_TOPIC,
            self._map_clearance_callback,
            map_qos,
        )

        self.start_pub = self.node.create_publisher(
            Empty,
            "/mrv/cmd/start",
            10,
        )

        self.pause_pub = self.node.create_publisher(
            Empty,
            "/mrv/cmd/pause",
            10,
        )

        self.resume_pub = self.node.create_publisher(
            Empty,
            "/mrv/cmd/resume",
            10,
        )

        self.abort_pub = self.node.create_publisher(
            Empty,
            "/mrv/cmd/abort",
            10,
        )

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self.thread = threading.Thread(
            target=self._spin,
            name="mep-dashboard-ros2",
            daemon=True,
        )

        self.thread.start()

        self.map_executor = SingleThreadedExecutor()
        self.map_executor.add_node(self.map_node)

        self.map_thread = threading.Thread(
            target=self._spin_map,
            name="mep-dashboard-ros2-map",
            daemon=True,
        )

        self.map_thread.start()

        print(
            "[LIVE] ROS2 bridge started: "
            "/mrv/status -> /ws/live",
            flush=True,
        )

    def _status_callback(self, msg):
        try:
            raw = json.loads(msg.data)
            last_progress = self._latest.get("progress") if self._latest else None
            payload = normalize_live_status(raw, last_progress)
        except Exception as exc:
            print(f"[LIVE] invalid /mrv/status: {exc}", flush=True)
            return

        sim_time = finite_number(raw.get("sim_time_s"))

        with self._lock:
            self._seq += 1
            self._latest = payload
            # A new run (the simulator clock went back): the previous run's map must not
            # be shown as if it were already scanned -- start the map view empty
            if sim_time is not None and self._last_sim_time is not None and sim_time < self._last_sim_time - 1.0:
                self._reset_map_locked()
            if sim_time is not None:
                self._last_sim_time = sim_time


    def _map_points_callback(self, msg):
        """Astrobee map snapshot (a few seconds apart) -> one binary WS frame."""
        try:
            xyz, flags = cloud_to_arrays(msg)
        except Exception as exc:
            print(f"[LIVE] invalid {ASTROBEE_MAP_POINTS_TOPIC}: {exc}", flush=True)
            return

        with self._lock:
            self._map_last_rx = time.monotonic()
            self._map_seq += 1
            self._map_points = int(len(xyz))
            self._map_frame = encode_map_frame(self._map_seq, xyz, flags)

    def _reset_map_locked(self):
        """Empty map snapshot + no clearance (caller holds the lock)."""
        self._map_seq += 1
        self._map_points = 0
        self._map_frame = encode_map_frame(self._map_seq, np.zeros((0, 3)), np.zeros(0, dtype=bool))
        self._map_status_seq += 1
        self._map_status = {"reset": True}
        print("[LIVE] new run: Astrobee map view cleared", flush=True)

    def _map_clearance_callback(self, msg):
        try:
            status = json.loads(msg.data)
        except ValueError as exc:
            print(f"[LIVE] invalid {ASTROBEE_MAP_CLEARANCE_TOPIC}: {exc}", flush=True)
            return

        with self._lock:
            self._map_last_rx = time.monotonic()
            self._map_status_seq += 1
            self._map_status = status

    def map_status(self):
        """(seq, `astrobee_map` field of the WS payload or None)."""
        with self._lock:
            if self._map_status is None and self._map_frame is None:
                return self._map_status_seq, None
            return self._map_status_seq, {
                **(self._map_status or {}),
                "points": self._map_points,
                "map_seq": self._map_seq,
            }

    def astrobee_alive(self) -> bool:
        """The Astrobee is streaming: its camera image (5 Hz) or a map message arrived
        within `IMAGE_STALE_TIMEOUT_S` (the map alone can be quiet for a few seconds)."""
        last = max(self._camera3_last_rx, self._map_last_rx)
        return last > 0.0 and time.monotonic() - last <= IMAGE_STALE_TIMEOUT_S

    def map_frame(self):
        with self._lock:
            return self._map_seq, self._map_frame

    @staticmethod
    def _decode_ros_image_to_jpeg(msg, tag: str):
        """Convert a real sensor_msgs/Image to a JPEG byte string, or None
        (unsupported encoding / malformed message -- logged, never raised)."""
        height = int(msg.height)
        width = int(msg.width)
        step = int(msg.step)

        if height <= 0 or width <= 0 or step <= 0:
            return None

        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if raw.size < height * step:
            return None

        rows = raw[:height * step].reshape(height, step)
        encoding = str(msg.encoding or "").lower()

        if encoding == "rgb8":
            frame = rows[:, :width * 3].reshape(height, width, 3)

        elif encoding == "bgr8":
            frame = rows[:, :width * 3].reshape(height, width, 3)
            frame = frame[:, :, ::-1].copy()

        elif encoding == "rgba8":
            frame = rows[:, :width * 4].reshape(height, width, 4)
            frame = frame[:, :, :3].copy()

        elif encoding == "bgra8":
            frame = rows[:, :width * 4].reshape(height, width, 4)
            frame = frame[:, :, [2, 1, 0]].copy()

        elif encoding in {"mono8", "8uc1"}:
            frame = rows[:, :width].reshape(height, width)

        else:
            print(f"[{tag}] unsupported encoding: {msg.encoding}", flush=True)
            return None

        image = PILImage.fromarray(frame)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=80, optimize=False)
        return output.getvalue()

    def _camera1_callback(self, msg):
        """Real Isaac Sim wrist camera (cam_wrist)."""
        now = time.monotonic()
        self._camera1_last_rx = now

        # Limit JPEG encoding to about 5 FPS.
        if now - self._camera1_last_encode < 0.20:
            return
        self._camera1_last_encode = now

        try:
            jpeg = self._decode_ros_image_to_jpeg(msg, "CAM1")
            if jpeg is None:
                return
            with self._lock:
                self._camera1_jpeg = jpeg
                self._camera1_seq += 1
        except Exception as exc:
            print(f"[CAM1] conversion error: {type(exc).__name__}: {exc}", flush=True)

    def camera1_snapshot(self):
        with self._lock:
            jpeg = self._camera1_jpeg

        if time.monotonic() - self._camera1_last_rx > IMAGE_STALE_TIMEOUT_S:
            return None

        return jpeg

    def _camera2_callback(self, msg):
        """Real probe-tip camera (cam_probe), used by the docking phase."""
        now = time.monotonic()
        self._camera2_last_rx = now

        # About 5 FPS max for browser streaming.
        if now - self._camera2_last_encode < 0.20:
            return
        self._camera2_last_encode = now

        try:
            jpeg = self._decode_ros_image_to_jpeg(msg, "CAM2")
            if jpeg is None:
                return
            with self._lock:
                self._camera2_jpeg = jpeg
                self._camera2_seq += 1
        except Exception as exc:
            print(f"[CAM2] conversion error: {type(exc).__name__}: {exc}", flush=True)

    def camera2_snapshot(self):
        with self._lock:
            jpeg = self._camera2_jpeg

        if time.monotonic() - self._camera2_last_rx > IMAGE_STALE_TIMEOUT_S:
            return None

        return jpeg

    def _camera3_callback(self, msg):
        """Astrobee observation camera (/astrobee/camera/image_raw, separate
        namespace from /mrv/). publish_rate_hz is 5 in vision_capture.yaml."""
        now = time.monotonic()
        self._camera3_last_rx = now

        # About 5 FPS max for browser streaming (matches the publish rate).
        if now - self._camera3_last_encode < 0.20:
            return
        self._camera3_last_encode = now

        try:
            jpeg = self._decode_ros_image_to_jpeg(msg, "CAM3")
            if jpeg is None:
                return
            with self._lock:
                self._camera3_jpeg = jpeg
                self._camera3_seq += 1
        except Exception as exc:
            print(f"[CAM3] conversion error: {type(exc).__name__}: {exc}", flush=True)

    def camera3_snapshot(self):
        with self._lock:
            jpeg = self._camera3_jpeg

        if time.monotonic() - self._camera3_last_rx > IMAGE_STALE_TIMEOUT_S:
            return None

        return jpeg

    def _viewport_callback(self, msg):
        """Main Isaac Sim 3D viewport, captured on the Isaac side via the
        omni.kit.viewport.utility API (not a screen share)."""
        now = time.monotonic()
        self._viewport_last_rx = now

        # Accept the 15 FPS source without jitter-driven frame drops; cap encode work at 20 FPS.
        if now - self._viewport_last_encode < 0.05:
            return
        self._viewport_last_encode = now

        try:
            jpeg = self._decode_ros_image_to_jpeg(msg, "VIEWPORT")
            if jpeg is None:
                return
            with self._lock:
                self._viewport_jpeg = jpeg
                self._viewport_seq += 1
        except Exception as exc:
            print(f"[VIEWPORT] conversion error: {type(exc).__name__}: {exc}", flush=True)

    def viewport_snapshot(self):
        with self._lock:
            jpeg = self._viewport_jpeg

        if time.monotonic() - self._viewport_last_rx > 2.0:
            return None

        return jpeg

    def _spin(self):
        try:
            while self._running and self.rclpy.ok():
                self.executor.spin_once(timeout_sec=0.2)
        except Exception as exc:
            # SIGINT may shut the shared rclpy context down before FastAPI's
            # shutdown hook reaches this thread. That is a normal exit path.
            if self._running and self.rclpy.ok():
                print(f"[LIVE] ROS2 executor stopped: {type(exc).__name__}: {exc}", flush=True)

    def _spin_map(self):
        try:
            while self._running and self.rclpy.ok():
                self.map_executor.spin_once(timeout_sec=0.2)
        except Exception as exc:
            if self._running and self.rclpy.ok():
                print(f"[LIVE] ROS2 map executor stopped: {type(exc).__name__}: {exc}", flush=True)

    def snapshot(self):
        _, astrobee_map = self.map_status()

        with self._lock:
            payload = dict(self._latest) if self._latest is not None else None

            if payload is not None and astrobee_map is not None:
                payload["astrobee_map"] = astrobee_map

            return self._seq, payload

    def command(self, command: str):
        if command == "start":
            publisher = self.start_pub
        elif command == "pause":
            publisher = self.pause_pub
        elif command == "resume":
            publisher = self.resume_pub
        elif command == "abort":
            publisher = self.abort_pub
        else:
            raise ValueError(command)

        subscribers = publisher.get_subscription_count()

        if subscribers < 1:
            raise RuntimeError(
                f"No ROS2 subscriber for command '{command}'"
            )

        publisher.publish(self.Empty())

        return subscribers

    def shutdown(self):
        self._running = False

        for thread in (self.thread, self.map_thread):
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass

        for executor, node in ((self.executor, self.node), (self.map_executor, self.map_node)):
            try:
                executor.remove_node(node)
            except Exception:
                pass

            try:
                node.destroy_node()
            except Exception:
                pass

        if self._owns_context and self.rclpy.ok():
            self.rclpy.shutdown()


live_ros = None
live_ros_error = None


@app.on_event("startup")
def start_live_ros():
    global live_ros, live_ros_error

    try:
        live_ros = RosLiveBridge()
        live_ros_error = None
    except Exception as exc:
        live_ros = None
        live_ros_error = f"{type(exc).__name__}: {exc}"

        print(
            f"[LIVE] ROS2 bridge unavailable: {live_ros_error}",
            flush=True,
        )


@app.on_event("shutdown")
def stop_live_ros():
    global live_ros

    if live_ros is not None:
        live_ros.shutdown()
        live_ros = None


# ============================================================
# HTTP
# ============================================================

@app.get("/", include_in_schema=False)
def index():
    # Always revalidated: the page pins its CSS/JS versions (`?v=`), so a stale copy of
    # it would keep serving old assets after a frontend change
    return FileResponse(FRONTEND / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/health")
def health():
    live_connected = False

    if live_ros is not None:
        _, latest = live_ros.snapshot()
        live_connected = latest is not None

    return {
        "status": "ok",
        "mode": "firestore-validation+ros2-live",
        "ros_bridge": live_ros is not None,
        "ros_live_data": live_connected,
        "ros_error": live_ros_error,
    }


@app.get("/api/validation/runs")
def validation_runs():
    cached = cache_read("runs.json")

    if cached and time.time() - cached.get("cached_at", 0.0) < RUNS_CACHE_TTL_S:
        return {"runs": cached.get("runs", []), "cache": "fresh"}

    try:
        db = firestore_client()
        runs = []

        for document in db.collection(SESSIONS).stream():
            data = json_value(document.to_dict() or {})
            data.setdefault("session_id", document.id)
            runs.append(data)

    except Exception as exc:
        reason = exc.detail if isinstance(exc, HTTPException) else f"{type(exc).__name__}: {exc}"
        print(f"[firestore] run list failed: {reason}", flush=True)

        if cached:
            print("[cache] serving the stale run list", flush=True)
            return {"runs": cached.get("runs", []), "cache": "stale", "cache_error": reason}

        raise HTTPException(
            status_code=503,
            detail=reason if isinstance(exc, HTTPException) else f"Firebase Connection Error: {reason}",
        ) from exc

    cache_write("runs.json", {"cached_at": time.time(), "runs": runs})
    return {"runs": runs, "cache": "live"}


@app.get("/api/validation/runs/{session_id}/telemetry")
def validation_telemetry(session_id: str):
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid session ID",
        )

    name = telemetry_cache_name(session_id)
    cached = cache_read(name)

    # A finished run never changes: serve it without touching Firestore.
    if cached and cached.get("complete"):
        return {
            "session_id": session_id,
            "telemetry": cached.get("telemetry", []),
            "cache": "complete",
        }

    try:
        db = firestore_client()
        telemetry = []

        query = (
            db.collection(SESSIONS)
            .document(session_id)
            .collection(TELEMETRY)
            .order_by("id")
        )

        for document in query.stream():
            data = json_value(document.to_dict() or {})
            data.setdefault("id", document.id)
            telemetry.append(data)

        complete = session_is_finished(db, session_id)

    except Exception as exc:
        reason = exc.detail if isinstance(exc, HTTPException) else f"{type(exc).__name__}: {exc}"
        print(f"[firestore] telemetry failed for {session_id}: {reason}", flush=True)

        if cached:
            print(f"[cache] serving stale telemetry for {session_id}", flush=True)
            return {
                "session_id": session_id,
                "telemetry": cached.get("telemetry", []),
                "cache": "stale",
                "cache_error": reason,
            }

        raise HTTPException(
            status_code=503,
            detail=reason if isinstance(exc, HTTPException) else f"Firebase Connection Error: {reason}",
        ) from exc

    cache_write(name, {
        "cached_at": time.time(),
        "session_id": session_id,
        "complete": complete,
        "telemetry": telemetry,
    })

    return {
        "session_id": session_id,
        "telemetry": telemetry,
        "cache": "live" if complete else "live-running",
    }



def find_run_video(session_id: str):
    """Resolve only an exact run-named MP4 below the configured video root."""
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        return None

    # `<id>.mp4` is the sim-time aligned session video of firebase_bridge.py; `<id>_overlay.mp4` the legacy one
    for filename in (f"{session_id}.mp4", f"{session_id}_overlay.mp4"):
        candidate = (RUN_VIDEO_ROOT / filename).resolve()
        if candidate.parent == RUN_VIDEO_ROOT and candidate.is_file():
            return candidate

    return None


@app.get("/api/validation/runs/{session_id}/video-metadata")
def validation_video_metadata(session_id: str):
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")

    video = find_run_video(session_id)

    # Sidecar of the session video: video time 0 = sim time `t0_sim_s` (telemetry `sim_time`)
    timing = {}
    if video is not None and video.name == f"{session_id}.mp4":
        try:
            timing = json.loads((RUN_VIDEO_ROOT / f"{session_id}.video.json").read_text())
        except (OSError, ValueError):
            timing = {}

    return {
        "session_id": session_id,
        "available": video is not None,
        "video_t0_s": timing.get("t0_sim_s"),
        "fps": timing.get("fps"),
        "duration_s": timing.get("duration_s"),
        "url": (
            f"/api/validation/runs/{session_id}/video"
            if video is not None
            else None
        ),
        "size_bytes": video.stat().st_size if video is not None else None,
    }


@app.get("/api/validation/runs/{session_id}/video")
def validation_video(session_id: str):
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")

    video = find_run_video(session_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Run video not available")

    return FileResponse(
        video,
        media_type="video/mp4",
        headers={"Cache-Control": "private, max-age=0, must-revalidate"},
    )


def find_run_pointcloud(session_id: str):
    """`<id>.ply` (firebase_bridge.py session map), next to the session video."""
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        return None

    candidate = (RUN_VIDEO_ROOT / f"{session_id}.ply").resolve()
    if candidate.parent == RUN_VIDEO_ROOT and candidate.is_file():
        return candidate

    return None


@app.api_route("/api/validation/runs/{session_id}/pointcloud", methods=["GET", "HEAD"])
def validation_pointcloud(session_id: str):
    if not SESSION_ID_PATTERN.fullmatch(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID")

    ply = find_run_pointcloud(session_id)
    if ply is None:
        raise HTTPException(status_code=404, detail="Run point cloud not available")

    return FileResponse(
        ply,
        media_type="application/octet-stream",
        filename=f"{session_id}_astrobee_map.ply",
        headers={"Cache-Control": "private, max-age=0, must-revalidate"},
    )


@app.get("/api/live/camera2.jpg")
def camera2_jpeg():
    if live_ros is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    jpeg = live_ros.camera2_snapshot()

    if jpeg is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/live/camera1.jpg")
def camera1_jpeg():
    if live_ros is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    jpeg = live_ros.camera1_snapshot()

    if jpeg is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/live/camera3.jpg")
def camera3_jpeg():
    if live_ros is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    jpeg = live_ros.camera3_snapshot()

    if jpeg is None:
        return Response(status_code=204, headers={"Cache-Control": "no-store"})

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/live/viewport.jpg")
def viewport_jpeg():
    if live_ros is None:
        return Response(status_code=204, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        })

    jpeg = live_ros.viewport_snapshot()

    if jpeg is None:
        return Response(status_code=204, headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        })

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.post("/api/live/command/{command}")
def live_command(command: str):
    if command not in {"start", "pause", "resume", "abort"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported live command",
        )

    if live_ros is None:
        raise HTTPException(
            status_code=503,
            detail=live_ros_error or "ROS2 bridge unavailable",
        )

    try:
        count = live_ros.command(command)

        return {
            "ok": True,
            "command": command,
            "subscriber_count": count,
        }

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


# ============================================================
# WebSocket
# ============================================================

@app.websocket("/ws/live")
async def live_websocket(websocket: WebSocket):
    await websocket.accept()

    last_seq = -1
    last_map_seq = 0
    last_map_status_seq = 0
    last_alive = None
    heartbeat = 0

    try:
        while True:
            heartbeat += 1

            if live_ros is None:
                payload = {
                    "connected": False,
                    "state": "ROS2 OFFLINE",
                    "phase": "LIVE TELEMETRY",
                    "progress": 0,
                    "total_steps": TOTAL_STAGES,
                    "error": live_ros_error,
                }

                if heartbeat % 10 == 1:
                    await websocket.send_json(payload)

            else:
                seq, payload = live_ros.snapshot()

                if payload is None:
                    if heartbeat % 10 == 1:
                        await websocket.send_json({
                            "connected": False,
                            "state": "WAITING FOR MISSION",
                            "phase": "ROS2 CONNECTED",
                            "progress": 0,
                            "total_steps": TOTAL_STAGES,
                        })

                elif seq != last_seq:
                    last_seq = seq
                    await websocket.send_json(payload)

                # Astrobee map: the clearance as a JSON message of its own (it changes
                # without /mrv/status too), the snapshot as one binary frame per update
                map_status_seq, astrobee_map = live_ros.map_status()

                if astrobee_map is not None and map_status_seq != last_map_status_seq:
                    last_map_status_seq = map_status_seq
                    await websocket.send_json({"type": "astrobee_map", "astrobee_map": astrobee_map})

                # CAMERA 4 liveness (LIVE / WAITING), sent when it changes
                alive = live_ros.astrobee_alive()

                if alive != last_alive:
                    last_alive = alive
                    await websocket.send_json({"type": "astrobee_map", "astrobee_map": {"alive": alive}})

                map_seq, frame = live_ros.map_frame()

                if frame is not None and map_seq != last_map_seq:
                    last_map_seq = map_seq
                    await websocket.send_bytes(frame)

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        return
