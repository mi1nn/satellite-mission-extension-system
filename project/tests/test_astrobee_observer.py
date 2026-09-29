"""Offline tests for the Astrobee observation path / camera mount (`astrobee.py`).

No Isaac Sim needed:

    cd ~/space_robotics_bench/project && ~/isaac-sim/python.sh -m pytest tests/test_astrobee_observer.py -q
"""

import ast
import importlib
import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# The `debris_capture` package `__init__` imports Isaac Lab, so the pure-numpy modules
# are loaded as members of a stand-in package instead (as in `test_moving_dock.py`).
_PKG_DIR = Path(__file__).resolve().parents[1].joinpath("srb", "tasks", "manipulation", "debris_capture")
_PKG = "_debris_capture_offline"
if _PKG not in sys.modules:
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_PKG_DIR)]
    sys.modules[_PKG] = pkg
ab = importlib.import_module(f"{_PKG}.astrobee")
load_vision_config = importlib.import_module(f"{_PKG}.vision").load_vision_config
Frame = importlib.import_module(f"{_PKG}.frames").Frame

DT = 1.0 / 60.0
SEED = 0


def _path(**kw):
    cfg = ab.AstrobeeCfg(**kw)
    ab.validate_astrobee_cfg(cfg)
    return cfg, ab.ObservationPath(cfg, ring_radius_m=30.0, ref_dir_w=[1.0, 1.0, 0.3])


def test_look_at_rotation_is_proper_and_points_at_target():
    rng = np.random.default_rng(SEED)
    for _ in range(50):
        eye, target = rng.normal(size=3) * 20.0, rng.normal(size=3) * 20.0
        r = ab.look_at_rotation(eye, target)
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-9)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-9)
        d = (target - eye) / np.linalg.norm(target - eye)
        assert np.allclose(r[:, 0], d, atol=1e-9)
        # Body +Z is "down": never pointing up
        assert r[2, 2] <= 1e-9


def test_look_at_rotation_straight_down_is_defined():
    r = ab.look_at_rotation([0.0, 0.0, 10.0], [0.0, 0.0, 0.0])
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-9)
    assert np.allclose(r[:, 0], [0.0, 0.0, -1.0])


def test_camera_looks_along_body_x_with_image_up_minus_body_z():
    body = Frame(np.array([1.0, 2.0, 3.0]), ab.look_at_rotation([1.0, 2.0, 3.0], [10.0, -4.0, 0.0]))
    cfg = ab.AstrobeeCfg(scale=3.0)
    cam = ab.camera_world_pose(body, cfg)
    assert np.allclose(cam.rot[:, 0], body.rot[:, 0])  # forward (Isaac "world" convention +X)
    assert np.allclose(cam.rot[:, 2], -body.rot[:, 2])  # image up = -Z_B
    assert np.allclose(cam.pos, body.point(np.array(cfg.camera.mount_pos_body_m) * 3.0))


def test_path_idle_then_approach_ends_at_point_1():
    cfg, p = _path(start_delay_s=2.0, approach_distance_m=20.0, approach_speed_mps=1.0)
    phase, off, _ = p.sample(0.0)
    assert phase == "idle" and np.allclose(off, p.approach_start())
    assert np.linalg.norm(p.approach_start() - p.inspection_points()[0]) == pytest.approx(20.0)
    assert p.sample(2.0 + 10.0)[0] == "approach"
    phase, off, idx = p.sample(2.0 + 20.0 + 1e-6)
    assert phase == "observe" and idx == 0 and np.allclose(off, p.inspection_points()[0], atol=1e-6)


def test_path_visits_every_point_on_the_ring_and_is_continuous():
    cfg, p = _path(approach_distance_m=0.0, dwell_s=2.0, transit_speed_mps=1.5)
    seen = set()
    prev = None
    max_speed = 0.0
    for k in range(int(p.loop_s / DT) + 1):
        phase, off, idx = p.sample(k * DT)
        assert phase == "observe"
        seen.add(idx)
        # Every observation position is on the ring, at the ring height
        assert math.hypot(off[0], off[1]) == pytest.approx(p.r, rel=1e-9)
        assert off[2] == pytest.approx(p.h)
        if prev is not None:
            max_speed = max(max_speed, float(np.linalg.norm(off - prev)) / DT)
        prev = off
    assert seen == set(range(len(cfg.inspection_azimuths_deg)))
    # smoothstep peaks at 1.5x the mean transit speed; no jumps anywhere
    assert max_speed <= 1.5 * cfg.transit_speed_mps + 1e-3


def test_path_dwells_at_each_inspection_point():
    cfg, p = _path(approach_distance_m=0.0, dwell_s=3.0)
    t = 0.0
    for i, point in enumerate(p.inspection_points()):
        for dt in (1e-6, 1.5, 2.99):  # (just after the leg boundary: float rounding)
            phase, off, idx = p.sample(t + dt)
            assert idx == i and np.allclose(off, point, atol=1e-9)
        t += p.leg_s[i]


def test_path_inspection_point_azimuths_are_relative_to_the_dock_direction():
    cfg, p = _path(inspection_azimuths_deg=[0.0, 90.0])
    p0, p1 = p.inspection_points()
    u = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    assert np.allclose(p0[:2] / p.r, u[:2])
    assert np.allclose(p1[:2] / p.r, np.cross([0.0, 0.0, 1.0], u)[:2])


def test_path_completes_after_loops():
    cfg, p = _path(loops=2, approach_distance_m=10.0)
    t_end = p.observe_t0 + 2 * p.loop_s
    assert p.sample(t_end - 0.1)[0] == "observe"
    phase, off, _ = p.sample(t_end + 0.1)
    assert phase == "complete" and np.allclose(off, p.inspection_points()[0])


def test_ring_radius_from_aabb():
    assert ab.ring_radius_from_aabb([0.0, 0.0, 0.0], [6.0, 8.0, 100.0], 2.0) == pytest.approx(7.0)


def test_config_loads_and_overrides():
    cfg = load_vision_config()
    a = cfg.astrobee
    assert a.enabled and a.ros_namespace == "astrobee" and a.image_topic == "camera/image_raw"
    assert a.camera.name == "cam_astrobee"
    assert not load_vision_config(overrides=["astrobee.enabled=false"]).astrobee.enabled


@pytest.mark.parametrize("override", [
    "astrobee.scale=0.0",
    "astrobee.transit_speed_mps=0.0",
    "astrobee.inspection_azimuths_deg=[]",
    "astrobee.loops=-1",
    "astrobee.elevation_deg=90.0",
])
def test_config_rejects_bad_values(override):
    with pytest.raises(ValueError):
        load_vision_config(overrides=[override])


def _calls(tree, attr):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr]


def test_astrobee_publishes_image_and_map_only():
    """Role separation: publishers for the image and the map (clearance + snapshot), one
    subscription (the existing MRV state topic, docking-complete signal), and no import
    of the vision / capture / docking-control code (the only package imports are frame
    math, the pure-numpy map, the USD prim-frame helper and the rclpy loader). Its one
    feedback to the mission is the in-process `docking_clearance()`."""
    tree = ast.parse((_PKG_DIR / "astrobee.py").read_text())
    pubs = _calls(tree, "create_publisher")
    assert sorted(ast.unparse(p.args[0]) for p in pubs) == ["Image", "PointCloud2", "String"]
    subs = _calls(tree, "create_subscription")
    assert len(subs) == 1 and ast.unparse(subs[0].args[0]) == "String"
    local = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1 for a in n.names}
    assert {m for m, _ in local} == {"frames", "astrobee_map", "docking", "ros_interface"}
    assert {a for m, a in local if m in ("frames", "docking", "ros_interface")} == {
        "Frame", "prim_frame", "mesh_points_and_triangles", "_import_rclpy"}
    # the map module stays Isaac-free
    map_tree = ast.parse((_PKG_DIR / "astrobee_map.py").read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(map_tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(map_tree) if isinstance(n, ast.ImportFrom) and n.level == 0}
    assert imported <= {"math", "dataclasses", "pathlib", "typing", "numpy"}


def test_db_bridge_reads_only_the_astrobee_map():
    """The DB bridge writes no Astrobee telemetry; its one Astrobee input is the map
    snapshot topic, saved as the session point cloud."""
    bridge = Path(__file__).resolve().parents[1].joinpath("scripts", "firebase_bridge.py").read_text()
    lines = [x for x in bridge.splitlines() if "astrobee" in x.lower()]
    assert lines and all("/astrobee/map/points" in x for x in lines)
    cfg = load_vision_config().astrobee
    assert f"/{cfg.ros_namespace}/{cfg.map.points_topic}" == "/astrobee/map/points"


def test_db_bridge_saves_the_session_point_cloud(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "_firebase_bridge", Path(__file__).resolve().parents[1].joinpath("scripts", "firebase_bridge.py"))
    fb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fb)
    am = importlib.import_module(f"{_PKG}.astrobee_map")
    rec = fb.SessionRecorder(fb.StdoutSink(), session_id="run_test", map_dir=str(tmp_path))
    rec.on_status({"sim_time_s": 0.0, "state": "SEARCH", "start_received": True})
    pts = np.random.default_rng(3).normal(size=(40, 3))
    rec.on_map(pts, np.arange(40) < 5)
    rec.finish()
    assert np.allclose(am.read_ply_xyz(tmp_path / "run_test.ply"), pts, atol=1e-6)
    # PointCloud2 decoding (x, y, z, obstruction float32, 16-byte points)
    data = np.column_stack((pts, (np.arange(40) < 5).astype(float))).astype("<f4")
    msg = types.SimpleNamespace(fields=[types.SimpleNamespace(name=n, offset=4 * i) for i, n in enumerate("xyz")]
                                + [types.SimpleNamespace(name="obstruction", offset=12)],
                                width=40, height=1, point_step=16, data=data.tobytes())
    xyz, flags = fb.cloud_to_arrays(msg)
    assert np.allclose(xyz, pts, atol=1e-6) and flags.sum() == 5 and flags[:5].all()


def _mission_states():
    """`State` enum values of the demo, read from its source (it imports Isaac Lab)."""
    tree = ast.parse((_PKG_DIR / "vision_capture_demo.py").read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "State")
    return {n.value.value for n in cls.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}


def test_dock_complete_states_are_real_mission_states():
    states = _mission_states()
    assert ab.DOCK_COMPLETE_STATES <= states
    # nothing before the docking joint exists, and no failure, counts as "docked"
    assert not ab.DOCK_COMPLETE_STATES & {"DOCK_READY", "FINAL_INSERTION", "DOCK_FAILED", "SUCCESS", "CAPTURED"}


def test_follow_moves_with_the_mrv_and_blends_the_aim():
    pos0, aim0 = np.array([10.0, 20.0, 5.0]), np.array([0.0, 0.0, 0.0])
    mrv0, dock = np.array([-5.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])
    # at the switch: no jump
    pos, aim = ab.follow_mrv_pose(pos0, aim0, mrv0, mrv0, dock, 0.0, 2.0)
    assert np.allclose(pos, pos0) and np.allclose(aim, aim0)
    # MRV retreats 7 m along -X: the Astrobee moves the same way, same distance
    mrv = mrv0 + np.array([-7.0, 0.0, 0.0])
    pos, aim = ab.follow_mrv_pose(pos0, aim0, mrv0, mrv, dock, 5.0, 2.0)
    assert np.allclose(pos - pos0, mrv - mrv0)
    assert np.allclose(aim, 0.5 * (mrv + dock))
    # aim blend is continuous (smoothstep), half way at half the blend time
    _, mid = ab.follow_mrv_pose(pos0, aim0, mrv0, mrv0, dock, 1.0, 2.0)
    assert np.allclose(mid, 0.5 * (aim0 + 0.5 * (mrv0 + dock)))


def test_follow_config_defaults_and_validation():
    cfg = load_vision_config()
    assert cfg.astrobee.follow_mrv_after_dock is True
    assert cfg.astrobee.follow_aim_blend_s == pytest.approx(2.0)
    with pytest.raises(ValueError):
        load_vision_config(overrides=["astrobee.follow_aim_blend_s=-1.0"])


def test_closing_distance_is_a_rest_to_rest_trapezoid():
    total, v, a = 20.0, 3.0, 1.0
    assert ab.closing_distance(0.0, total, v, a) == pytest.approx(0.0)
    assert ab.closing_distance(1.0, total, v, a) == pytest.approx(0.5)  # accelerating
    # cruise at 3 m/s after 3 s: 4.5 m ramp, (20 - 9) / 3 s cruise, 3 s ramp down
    assert ab.closing_distance(4.0, total, v, a) - ab.closing_distance(3.0, total, v, a) == pytest.approx(3.0)
    t_end = 3.0 + 11.0 / 3.0 + 3.0
    assert ab.closing_distance(t_end, total, v, a) == pytest.approx(total)
    assert ab.closing_distance(t_end + 100.0, total, v, a) == pytest.approx(total)
    # short move: triangular profile still reaches the total
    assert ab.closing_distance(10.0, 2.0, v, a) == pytest.approx(2.0)
    ts = np.linspace(0.0, t_end, 200)
    d = [ab.closing_distance(t, total, v, a) for t in ts]
    assert all(b >= c - 1e-12 for b, c in zip(d[1:], d[:-1]))


def test_follow_closes_in_along_the_offset_to_the_mrv():
    pos0, aim0 = np.array([30.0, 0.0, 0.0]), np.zeros(3)
    mrv0, dock = np.zeros(3), np.array([1.0, 0.0, 0.0])
    mrv = np.array([-4.0, 0.0, 0.0])
    pos, _ = ab.follow_mrv_pose(pos0, aim0, mrv0, mrv, dock, 5.0, 2.0, closed_m=20.0)
    assert np.allclose(pos, mrv + np.array([10.0, 0.0, 0.0]))


def test_close_in_config_defaults_and_validation():
    cfg = load_vision_config()
    assert cfg.astrobee.follow_close_speed_mps >= 3.0
    assert cfg.astrobee.follow_close_speed_mps > cfg.separation.velocity_mps
    with pytest.raises(ValueError):
        load_vision_config(overrides=["astrobee.follow_close_accel_mps2=0.0"])
    with pytest.raises(ValueError):
        load_vision_config(overrides=["separation.arm_stow_duration_s=0.0"])


def _set_of(tree, name):
    """Names of `State.X` members in the set assigned to `name` (module level)."""
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets))
    return {a.attr for a in ast.walk(node.value) if isinstance(a, ast.Attribute) and isinstance(a.value, ast.Name) and a.value.id == "State"}


def test_docking_unavailable_is_a_real_terminal_verdict():
    assert "DOCKING_UNAVAILABLE" in _mission_states()
    tree = ast.parse((_PKG_DIR / "vision_capture_demo.py").read_text())
    # the final verdict of the run: a failure (terminal), not a docking state that waits
    assert "DOCKING_UNAVAILABLE" in _set_of(tree, "FAILURES")
    assert "DOCKING_UNAVAILABLE" not in _set_of(tree, "DOCKING_STATES")
    assert "DOCKING_UNAVAILABLE" not in ab.DOCK_COMPLETE_STATES
    # the gate: before the probe starts along the docking axis (static and forced
    # alignment checks), on the way into DOCK_READY and inside it
    fn = {n.name: ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("step_alignment_check", "forced_alignment_check", "step_z_approach", "step_dock_ready"):
        assert "self.corridor_blocked(" in fn[name]
    gate = fn["corridor_blocked"]
    assert "docking_clearance()" in gate and "State.DOCKING_UNAVAILABLE" in gate
    assert "step_docking_unavailable" not in fn


def test_docking_unavailable_is_known_downstream():
    root = Path(__file__).resolve().parents[2]
    bridge = ast.parse(root.joinpath("project", "scripts", "firebase_bridge.py").read_text())
    sets = {t.id: ast.literal_eval(n.value) for n in bridge.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name) and t.id in ("TERMINAL_STATES", "MISSION_STATES", "FAILURE_STAGES")}
    # ends the session as a docking-stage failure
    assert "DOCKING_UNAVAILABLE" in sets["TERMINAL_STATES"] and sets["FAILURE_STAGES"]["DOCKING_UNAVAILABLE"] == "DOCKING"
    assert "DOCKING_UNAVAILABLE" not in sets["MISSION_STATES"]
    app = root.joinpath("mep_dashboard", "backend", "app.py").read_text()
    assert '"DOCKING_UNAVAILABLE"' in app
    assert "'DOCKING_UNAVAILABLE'" in root.joinpath("mep_dashboard", "frontend", "js", "validation.js").read_text()


def test_depth_samples_are_taken_inside_each_dwell_only():
    cfg, p = _path(approach_distance_m=0.0, dwell_s=5.0, loops=1)
    slots = {}
    for k in range(int((p.loop_s + 20.0) / DT)):
        t = k * DT
        phase, _, _ = p.sample(t)
        key = ab.depth_sample_slot(p, t, 2, phase)
        if key is not None and key not in slots:
            slots[key] = t
    obs = {k: t for k, t in slots.items() if k[0] != "complete"}
    # 2 per inspection point, at 1/3 and 2/3 of the dwell
    assert sorted(obs) == [(0, i, s) for i in range(4) for s in (1, 2)]
    t0 = 0.0
    for i in range(4):
        assert obs[(0, i, 1)] == pytest.approx(t0 + 5.0 / 3.0, abs=2 * DT)
        assert obs[(0, i, 2)] == pytest.approx(t0 + 10.0 / 3.0, abs=2 * DT)
        t0 += p.leg_s[i]
    # never while flying between points
    for t in obs.values():
        assert p.dwell_at(t) is not None
    # after the last loop: one sample per dwell_s
    assert len([k for k in slots if k[0] == "complete"]) in (4, 5)


def test_clearance_record_is_json_and_names_the_docking_status():
    import json

    am = importlib.import_module(f"{_PKG}.astrobee_map")
    blocked = am.DockingClearance(is_clear=False, obstruction_voxels=7, nearest_m=0.31, corridor_views=3)
    rec = ab.clearance_record(12.34567, blocked, frames=8, voxels=900)
    assert json.loads(json.dumps(rec)) == rec
    assert rec["status"] == "DOCKING_UNAVAILABLE" and rec["obstruction_voxels"] == 7 and rec["nearest_m"] == pytest.approx(0.31)
    clear = am.DockingClearance(is_clear=True, obstruction_voxels=0, nearest_m=math.inf, corridor_views=2)
    rec = ab.clearance_record(1.0, clear, frames=2, voxels=10)
    assert rec["status"] == "DOCKING_AVAILABLE" and rec["nearest_m"] is None


def test_map_topics_config():
    cfg = load_vision_config()
    m = cfg.astrobee.map
    assert (m.clearance_topic, m.points_topic) == ("map/clearance", "map/points")
    assert m.points_publish_period_s >= 1.0  # a snapshot every few seconds, never per frame


def test_dock_view_offset_is_the_requested_close_up():
    center = np.array([22.34, 12.09, 4.24])
    exit_w = np.array([15.24, 0.29, 3.0])
    open_dir = np.array([-1.0, 0.02, 0.0])
    open_dir /= np.linalg.norm(open_dir)
    u = np.array([-0.515, -0.857, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross([0.0, 0.0, 1.0], u)
    az, h, r = ab.dock_view_offset(u, v, center, exit_w, open_dir, 12.0, 45.0, 90.0)
    p = center + r * (math.cos(az) * u + math.sin(az) * v) + np.array([0.0, 0.0, h])
    rel = p - exit_w
    assert np.linalg.norm(rel) == pytest.approx(12.0)
    assert math.degrees(math.acos(rel @ open_dir / 12.0)) == pytest.approx(45.0)
    assert rel[2] == pytest.approx(12.0 * math.sin(math.radians(45.0)))  # roll 90: straight above the axis


def test_dock_view_is_the_first_point_aimed_at_the_port_and_the_flight_stays_continuous():
    cfg = ab.AstrobeeCfg(approach_distance_m=0.0, dwell_s=2.0, look_at_dock_weight=0.5)
    ab.validate_astrobee_cfg(cfg)
    p = ab.ObservationPath(cfg, 30.0, [1.0, 0.0, 0.0], dock_view=(math.radians(20.0), 6.0, 12.0))
    assert len(p.az) == 5 and p.labels[0] == "docking port close-up"
    assert np.allclose(p.sample(0.5)[1], p.ring_point(math.radians(20.0), 6.0, 12.0))
    assert p.aim_weight(0.5) == pytest.approx(1.0)  # looks at the docking port there
    assert p.aim_weight(p.leg_s[0] + 0.5) == pytest.approx(0.5)  # the ring points keep theirs
    prev, vmax = None, 0.0
    for k in range(int(p.loop_s / DT) + 2):
        off = p.sample(k * DT)[1]
        if prev is not None:
            vmax = max(vmax, float(np.linalg.norm(off - prev)) / DT)
        prev = off
    assert vmax <= 1.6 * cfg.transit_speed_mps  # smoothstep peak 1.5x (+ curved arc)
    assert len(ab.ObservationPath(cfg, 30.0, [1.0, 0.0, 0.0]).az) == 4


def test_depth_samples_while_flying_fill_the_map_gradually():
    cfg, p = _path(approach_distance_m=10.0, approach_speed_mps=1.0, dwell_s=5.0, loops=1)
    keys = set()
    for k in range(int((p.observe_t0 + p.loop_s) / DT)):
        t = k * DT
        key = ab.depth_sample_slot(p, t, 2, p.sample(t)[0], scan_period_s=1.0)
        if key is not None:
            keys.add(key)
    scans = [k for k in keys if k[0] == "scan"]
    dwells = [k for k in keys if k[0] != "scan"]
    assert len(dwells) == 2 * len(p.az)
    # ~1 per second between inspection points, none while dwelling
    assert abs(len(scans) - sum(p.transit_s)) <= len(p.az) + 2
    # nothing on the approach: the scan starts once the Astrobee is close
    assert all(k[1] * 1.0 >= p.observe_t0 - 1.0 for k in scans)
    assert ab.depth_sample_slot(p, 5.0, 2, "approach", scan_period_s=1.0) is None
    assert ab.depth_sample_slot(p, p.observe_t0 + p.leg_s[0] - 1.0, 2, "observe", scan_period_s=0.0) is None  # off


def test_scan_is_close_up_and_slow():
    cfg = load_vision_config().astrobee
    assert cfg.map.max_range_m <= 20.0  # nothing mapped from far away
    assert cfg.orbit_margin_m <= 5.0 and cfg.transit_speed_mps <= 1.0
    assert cfg.map.voxel_size_m <= 0.03 and cfg.map.grid_stride_px <= 4  # dense map


def test_confirmed_obstruction_stops_the_mission_in_any_phase():
    tree = ast.parse((_PKG_DIR / "vision_capture_demo.py").read_text())
    fn = {n.name: ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    stop = fn["astrobee_stop"]
    # checked every control step, right after the ROS abort, before the phase logic
    step = fn["step"]
    assert step.index("self.astrobee_stop()") < step.index("self.check_physics()")
    assert "State.DOCKING_UNAVAILABLE" in stop and "clear.observed" in stop
    # never before the start, after the end / once docked, or on a run without docking
    for guard in ("State.INIT", "TERMINAL", "State.DOCKED", "MOVING_POST_DOCK", "docking.enabled", "stop_mission_on_obstruction"):
        assert guard in stop


def test_dashboard_keeps_the_stage_the_mission_stopped_in():
    root = Path(__file__).resolve().parents[2]
    app = root.joinpath("mep_dashboard", "backend", "app.py").read_text()
    assert 'if state in ("ABORTED", "DOCKING_UNAVAILABLE"):' in app
    live = root.joinpath("mep_dashboard", "frontend", "js", "live.js").read_text()
    assert "DOCKING UNAVAILABLE" in live and "ASTROBEE: DOCKING PORT BLOCKED" in live


def test_isaac_keeps_running_after_docking_unavailable():
    script = Path(__file__).resolve().parents[1].joinpath("scripts", "vision_capture.py").read_text()
    assert 'results.final_state in ("SUCCESS", "DOCKING_UNAVAILABLE")' in script


def test_scan_rings_cover_above_and_below_continuously():
    cfg, p = _path(approach_distance_m=0.0, dwell_s=2.0, scan_elevations_deg=[25.0, -20.0])
    assert len(p.az) == 2 * len(cfg.inspection_azimuths_deg)
    hs = [pt[2] for pt in p.inspection_points()]
    assert hs[:4] == pytest.approx([30.0 * math.tan(math.radians(25.0))] * 4)
    assert hs[4:] == pytest.approx([30.0 * math.tan(math.radians(-20.0))] * 4)
    prev, vmax = None, 0.0
    for k in range(int(p.loop_s / DT) + 2):
        off = p.sample(k * DT)[1]
        assert math.hypot(off[0], off[1]) == pytest.approx(p.r, rel=1e-9)  # always on the ring radius
        if prev is not None:
            vmax = max(vmax, float(np.linalg.norm(off - prev)) / DT)
        prev = off
    assert vmax <= 1.6 * cfg.transit_speed_mps  # no jump between the rings
    with pytest.raises(ValueError):
        load_vision_config(overrides=["astrobee.scan_elevations_deg=[25.0, -95.0]"])
    assert load_vision_config().astrobee.scan_elevations_deg == [25.0, -20.0]
