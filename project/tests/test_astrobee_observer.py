"""Offline tests for the Astrobee observation path / camera mount (`astrobee.py`).

No Isaac Sim needed:

    cd ~/space_robotics_bench/project && ~/isaac-sim/python.sh -m pytest tests/test_astrobee_observer.py -q
"""

import ast
import importlib
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


def test_astrobee_publishes_the_camera_image_only():
    """Role separation: one publisher (the image), one subscription (the existing MRV
    state topic, docking-complete signal), and no import of the vision / capture /
    docking-control code (the only package imports are frame math, the USD prim-frame
    helper, the rclpy loader and the pure-numpy damping assist, whose thrust the demo
    applies)."""
    tree = ast.parse((_PKG_DIR / "astrobee.py").read_text())
    pubs = _calls(tree, "create_publisher")
    assert len(pubs) == 1 and ast.unparse(pubs[0].args[0]) == "Image"
    subs = _calls(tree, "create_subscription")
    assert len(subs) == 1 and ast.unparse(subs[0].args[0]) == "String"
    local = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level == 1 for a in n.names}
    assert local == {("frames", "Frame"), ("docking", "prim_frame"), ("ros_interface", "_import_rclpy"),
                     ("astrobee_assist", "AstrobeeAssistCfg"), ("astrobee_assist", "DampingAssist"),
                     ("astrobee_assist", "MepContext"), ("astrobee_assist", "validate_assist_cfg")}


def test_db_bridge_has_no_astrobee_entries():
    bridge = Path(__file__).resolve().parents[1].joinpath("scripts", "firebase_bridge.py").read_text()
    assert "astrobee" not in bridge.lower()


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
