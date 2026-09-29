"""Offline tests for the predictive coupled docking control (`coupled_dock.py`).

No Isaac Sim needed:

    cd ~/isaac_space/project && python3 -m pytest tests/test_coupled_dock.py -q

The tracker tests close the loop over a model of the arm + 3 t payload: the probe tip
follows the commanded pose through a lightly damped second-order mode (period 20 s,
damping ratio 0.02), the dominant dynamics the existing docking notes describe.
Deterministic: no randomness.
"""

import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_PKG_DIR = Path(__file__).resolve().parents[1].joinpath("srb", "tasks", "manipulation", "debris_capture")
_PKG = "_debris_capture_offline"
if _PKG not in sys.modules:
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_PKG_DIR)]
    sys.modules[_PKG] = pkg
cd = importlib.import_module(f"{_PKG}.coupled_dock")
Frame = importlib.import_module(f"{_PKG}.frames").Frame
load_vision_config = importlib.import_module(f"{_PKG}.vision").load_vision_config

DT = 1.0 / 60.0


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


################
### SO(3) ###
################


@pytest.mark.parametrize("phi", [[0, 0, 0], [1e-9, 0, 0], [0.3, -0.2, 0.1], [0, 0, math.pi - 1e-6], [2.0, 1.0, -1.5]])
def test_so3_exp_log_round_trip(phi):
    phi = np.asarray(phi, dtype=float)
    r = cd.so3_exp(phi)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(cd.so3_exp(cd.so3_log(r)), r, atol=1e-6)


def test_orientation_error_is_a_rotation_vector_not_an_euler_difference():
    cur = Frame(np.zeros(3), rot_z(170.0))
    des = Frame(np.zeros(3), rot_z(-170.0))
    _, e_r = cd.pose_error(des, cur)
    # Euler subtraction would say -340 deg; the true error is +20 deg about +Z
    assert np.allclose(e_r, [0.0, 0.0, math.radians(20.0)], atol=1e-9)


##################
### Prediction ###
##################


def test_prediction_without_rotation_is_linear():
    f = Frame(np.array([1.0, 2.0, 3.0]), rot_x(30.0))
    p = cd.predict_body_frame(f, com=[0.0, 0.0, 0.0], v_com=[0.02, 0.0, -0.01], w=[0, 0, 0], horizon=0.3)
    assert np.allclose(p.pos, f.pos + np.array([0.02, 0.0, -0.01]) * 0.3)
    assert np.allclose(p.rot, f.rot)


def test_prediction_of_a_tumbling_port_follows_the_circle_about_the_com():
    # Port 2 m from the CoM on +X, the client turns at 0.5 rad/s about world +Z for 1 s
    com = np.array([10.0, 0.0, 0.0])
    port = Frame(com + [2.0, 0.0, 0.0], np.eye(3))
    w = np.array([0.0, 0.0, 0.5])
    p = cd.predict_body_frame(port, com, v_com=[0, 0, 0], w=w, horizon=1.0)
    assert np.allclose(p.pos, com + 2.0 * np.array([math.cos(0.5), math.sin(0.5), 0.0]))
    assert np.isclose(np.linalg.norm(p.pos - com), 2.0)  # the tangent p + v T would drift out
    assert np.allclose(p.rot, rot_z(math.degrees(0.5)))  # world rate: R' = Exp(w T) R
    v_port = cd.point_velocity(port.pos, com, [0, 0, 0], w)
    assert np.allclose(v_port, [0.0, 1.0, 0.0])


def test_zero_horizon_is_the_current_pose():
    f = Frame(np.array([1.0, 0.0, 0.0]), rot_x(10.0))
    p = cd.predict_body_frame(f, [0, 0, 0], [1, 2, 3], [0.1, 0.2, 0.3], 0.0)
    assert np.allclose(p.pos, f.pos) and np.allclose(p.rot, f.rot)


def test_desired_tip_is_on_the_docking_axis():
    dock = Frame(np.array([1.0, 2.0, 3.0]), rot_x(90.0))  # +Z of the dock = world -Y
    d = cd.desired_tip(dock, -0.5)
    assert np.allclose(d.pos, [1.0, 2.5, 3.0]) and np.allclose(d.rot, dock.rot)


def test_dock_frame_errors_split_lateral_axial_and_full_orientation():
    dock = Frame(np.zeros(3), np.eye(3))
    tip = Frame(np.array([0.03, 0.04, -0.2]), rot_z(3.0))  # roll about the axis only
    e = cd.dock_frame_errors(tip, dock)
    assert math.isclose(e["lateral"], 0.05) and math.isclose(e["axial"], -0.2)
    assert math.isclose(e["orientation_deg"], 3.0, abs_tol=1e-9)  # roll is included


#################
### Soft gate ###
#################


def test_soft_gate_is_continuous_and_monotonic():
    c = cd.DockingControlCfg()
    xs = np.linspace(0.0, 0.1, 401)
    s = np.array([cd.alignment_scale(c, x, 0.0) for x in xs])
    assert s[0] == 1.0 and s[-1] == 0.0
    assert np.all(np.diff(s) <= 1e-12)  # never increases with the error
    assert np.max(np.abs(np.diff(s))) < 0.02  # no step (the approach speed is never switched)
    mid = cd.alignment_scale(c, 0.5 * (c.soft_gate_full_position_m + c.soft_gate_zero_position_m), 0.0)
    assert 0.0 < mid < 1.0
    # Both errors reduce the speed together
    assert cd.alignment_scale(c, 0.03, 3.0) < min(cd.alignment_scale(c, 0.03, 0.0), cd.alignment_scale(c, 0.0, 3.0))
    assert cd.alignment_scale(c, math.nan, 0.0) == 0.0


def test_emergency_stop_only_on_unsafe_states():
    c = cd.DockingControlCfg()
    assert cd.emergency_stop(c, 0.07, 6.0, 0.05, math.inf, 0.02) == (False, [])
    for args in ((0.2, 0.0, 0.0, math.inf), (0.0, 15.0, 0.0, math.inf), (0.0, 0.0, 0.2, math.inf), (0.0, 0.0, 0.0, 0.01)):
        stop, bad = cd.emergency_stop(c, *args, 0.02)
        assert stop and len(bad) == 1


######################
### Alignment gate ###
######################


def test_alignment_needs_the_stable_duration_not_one_step():
    a = cd.DockingAlignmentCfg(stable_duration_sec=0.5)
    g = cd.AlignmentGate(a)
    t = 0.0
    passed, _, _ = g.entry(t, 0.01, 1.0, 0.001, 0.1)
    assert not passed  # a single step is never enough
    while t < 0.49:
        t += DT
        passed, held, _ = g.entry(t, 0.01, 1.0, 0.001, 0.1)
        assert not passed
    t += 0.02
    passed, held, _ = g.entry(t, 0.01, 1.0, 0.001, 0.1)
    assert passed and held >= 0.5


def test_alignment_stable_interval_restarts_when_a_condition_breaks():
    g = cd.AlignmentGate(cd.DockingAlignmentCfg(stable_duration_sec=0.5))
    for i in range(20):
        g.entry(i * DT, 0.01, 1.0, 0.001, 0.1)
    passed, held, bad = g.entry(20 * DT, 0.01, 1.0, 0.05, 0.1)  # relative velocity too high
    assert not passed and held == 0.0 and "relative speed" in bad[0]
    passed, _, bad = g.entry(21 * DT, 0.01, 1.0, 0.001, 0.1, blocked=["MRV station error"])
    assert not passed and bad == ["MRV station error"]
    passed, held, _ = g.entry(22 * DT, 0.01, 1.0, 0.001, 0.1)
    assert not passed and held == 0.0


def test_rollback_hysteresis_ignores_noise_between_the_thresholds():
    a = cd.DockingAlignmentCfg(position_threshold_m=0.05, rollback_position_m=0.08, rollback_hold_sec=0.2)
    g = cd.AlignmentGate(a)
    # Noise around the ENTRY threshold never rolls back
    for i in range(300):
        back, _ = g.rollback(i * DT, 0.05 + 0.02 * math.sin(i), 2.0)
        assert not back
    # A single spike past the rollback threshold does not either
    g.reset()
    assert not g.rollback(0.0, 0.09, 2.0)[0]
    assert not g.rollback(DT, 0.06, 2.0)[0]
    # Held past it for rollback_hold_sec: rollback
    t, back = 1.0, False
    while t < 1.3 and not back:
        back, bad = g.rollback(t, 0.09, 2.0)
        t += DT
    assert back and "lateral" in bad[0]


def test_config_defaults_keep_legacy_and_validate_thresholds():
    cfg = load_vision_config()
    assert cfg.docking_control.mode == "legacy"  # the verified path stays the default
    assert cfg.docking_alignment.position_threshold_m < cfg.docking_alignment.rollback_position_m
    assert cfg.docking_alignment.orientation_threshold_deg < cfg.docking_alignment.rollback_orientation_deg
    cfg = load_vision_config(overrides=["docking_control.mode=coupled_predictive", "docking_control.prediction_horizon_sec=0.5"])
    assert cfg.docking_control.mode == "coupled_predictive" and cfg.docking_control.prediction_horizon_sec == 0.5
    for bad in (["docking_control.mode=smc"],
                ["docking_alignment.rollback_position_m=0.04"],        # rollback inside the entry band
                ["docking_alignment.rollback_orientation_deg=20.0"],   # rollback beyond the hard stop
                ["docking_alignment.stable_duration_sec=0.0"],
                ["docking_control.tip_feedback_weight=1.5"],
                ["docking_control.soft_gate_zero_position_m=0.005"]):
        with pytest.raises(ValueError):
            load_vision_config(overrides=bad)


###############
### Metrics ###
###############


def test_metrics_count_rollbacks_and_switches():
    m = cd.DockingControlMetrics()
    for old, new in (("PRE_DOCK_APPROACH", "POSITION_ATTITUDE_ALIGN"), ("POSITION_ATTITUDE_ALIGN", "ALIGNMENT_CHECK"),
                     ("ALIGNMENT_CHECK", "Z_APPROACH"), ("Z_APPROACH", "POSITION_ATTITUDE_ALIGN"),
                     ("POSITION_ATTITUDE_ALIGN", "ALIGNMENT_CHECK"), ("ALIGNMENT_CHECK", "Z_APPROACH"),
                     ("Z_APPROACH", "FINAL_INSERTION"), ("FINAL_INSERTION", "DOCK_READY")):
        m.transition(old, new)
    assert m.rollbacks == 1 and m.switches == 3
    m.update(0.0, "PRE_DOCK_APPROACH", 3.0, 90.0, 0.1, -1.0)  # transport errors are not alignment errors
    m.update(1.0, "Z_APPROACH", 0.02, 1.5, 0.03, 0.05)
    m.update(2.0, "FINAL_INSERTION", 0.01, 1.0, 0.01, 0.1)
    m.docked(5.0, {"position_error_m": 0.01, "orientation_error_deg": 0.5, "relative_speed_mps": 0.004})
    s = m.summary("coupled_predictive", True)
    assert s["docking_time_s"] == 5.0 and s["max_position_error_m"] == 0.02 and s["max_orientation_error_deg"] == 1.5
    assert s["max_contact_velocity_mps"] == 0.03 and s["relative_velocity_at_dock_mps"] == 0.004
    assert s["rollback_count"] == 1 and s["alignment_state_switching_count"] == 3


def test_twist_estimator_from_poses():
    est = cd.TwistEstimator(tau_s=0.05)
    v_true, w_true = np.array([0.02, -0.01, 0.0]), np.array([0.0, 0.0, 0.05])
    v = w = None
    for i in range(120):
        t = i * DT
        v, w = est.update(t, Frame(v_true * t, cd.so3_exp(w_true * t)))
    assert np.allclose(v, v_true, atol=1e-6) and np.allclose(w, w_true, atol=1e-6)


###########################
### Closed-loop tracker ###
###########################

WN, ZETA = 2.0 * math.pi / 20.0, 0.02  # arm + 3 t payload mode


def simulate(cfg, client_v, client_w, t_end, dock_offset=(0.0, 2.0, 0.0), tip0_err=(0.05, -0.04, 0.0), rot0_deg=4.0):
    """Probe tip (pose + twist) driven by the commanded pose through the payload mode;
    the client translates with `client_v` and turns with `client_w` about its CoM."""
    tracker = cd.CoupledPoseTracker(cfg)
    com0 = np.array([5.0, 0.0, 0.0])
    port0 = Frame(com0 + np.asarray(dock_offset, dtype=float), rot_x(-90.0))
    v_c, w_c = np.asarray(client_v, dtype=float), np.asarray(client_w, dtype=float)
    axial = -0.5
    start = cd.desired_tip(port0, axial)
    tip = Frame(start.pos + np.asarray(tip0_err), rot_z(rot0_deg) @ start.rot)
    ref = Frame(tip.pos.copy(), tip.rot.copy())
    v_tip, w_tip = v_c.copy(), w_c.copy()
    tracker.reset(v_tip, w_tip)
    lat, ori = [], []
    for i in range(int(t_end / DT)):
        t = i * DT
        com = com0 + v_c * t
        q = cd.so3_exp(w_c * t)
        port = Frame(com + q @ (port0.pos - com0), q @ port0.rot)
        pred = cd.predict_body_frame(port, com, v_c, w_c, cfg.prediction_horizon_sec)
        des = cd.desired_tip(pred, axial)
        v_des = cd.point_velocity(des.pos, com + v_c * cfg.prediction_horizon_sec, v_c, w_c)
        out = tracker.step(ref, tip, v_tip, w_tip, des, v_des, w_c, DT)
        v_ref = (out.ref.pos - ref.pos) / DT
        w_ref = cd.so3_log(out.ref.rot @ ref.rot.T) / DT
        ref = out.ref
        # Payload mode: the joint drives are PD on position and velocity (feed-forward)
        a = WN * WN * (ref.pos - tip.pos) + 2.0 * ZETA * WN * (v_ref - v_tip)
        alpha = WN * WN * cd.so3_log(ref.rot @ tip.rot.T) + 2.0 * ZETA * WN * (w_ref - w_tip)
        v_tip = v_tip + a * DT
        w_tip = w_tip + alpha * DT
        tip = Frame(tip.pos + v_tip * DT, cd.so3_exp(w_tip * DT) @ tip.rot)
        e = cd.dock_frame_errors(tip, port)
        lat.append(math.hypot(e["lateral"], e["axial"] - axial))
        ori.append(e["orientation_deg"])
    return np.asarray(lat), np.asarray(ori)


def test_tracker_converges_on_a_static_client():
    lat, ori = simulate(cd.DockingControlCfg(), [0, 0, 0], [0, 0, 0], 120.0)
    assert lat[-600:].max() < 2e-3 and ori[-600:].max() < 0.1
    assert lat.max() < 0.1  # no large overshoot of the payload


def test_tracker_follows_a_drifting_and_tumbling_client_without_lag():
    cfg = cd.DockingControlCfg()
    lat, ori = simulate(cfg, [0.02, 0.0, 0.0], [0.0, 0.0, 0.01], 150.0)
    # The port swings 2 m around the CoM at 0.01 rad/s (20 mm/s) and drifts at 20 mm/s:
    # position and attitude both tracked, both errors small at once
    assert lat[-600:].max() < 5e-3 and ori[-600:].max() < 0.2


def test_prediction_horizon_reduces_the_tracking_error_of_a_tumbling_client():
    base = cd.DockingControlCfg(prediction_horizon_sec=0.0)
    pred = cd.DockingControlCfg(prediction_horizon_sec=0.3)
    kw = dict(client_v=[0.0, 0.0, 0.0], client_w=[0.0, 0.0, 0.02], t_end=150.0)
    lat0, _ = simulate(base, **kw)
    lat1, _ = simulate(pred, **kw)
    # measured on this model: 6.2 mm without prediction, 4.3 mm with T = 0.3 s
    assert lat1[-600:].max() < 0.8 * lat0[-600:].max()


def test_measured_tip_feedback_alone_limit_cycles_on_the_payload_mode():
    """Why the proportional term uses the COMMANDED tip pose (module docstring)."""
    good, _ = simulate(cd.DockingControlCfg(), [0, 0, 0], [0, 0, 0], 120.0)
    bad, _ = simulate(cd.DockingControlCfg(tip_feedback_weight=1.0, kd_position=0.0, kd_attitude=0.0), [0, 0, 0], [0, 0, 0], 120.0)
    assert bad[-600:].max() > 10.0 * good[-600:].max()


def test_reference_never_runs_away_from_the_measured_tip():
    cfg = cd.DockingControlCfg(max_reference_lead_m=0.02, max_reference_lead_deg=1.0)
    tr = cd.CoupledPoseTracker(cfg)
    tip = Frame(np.zeros(3), np.eye(3))  # the arm is stuck
    ref = Frame(np.zeros(3), np.eye(3))
    des = Frame(np.array([1.0, 0.0, 0.0]), rot_z(30.0))
    for _ in range(3000):
        ref = tr.step(ref, tip, np.zeros(3), np.zeros(3), des, np.zeros(3), np.zeros(3), DT).ref
    assert np.linalg.norm(ref.pos - tip.pos) <= 0.02 + 1e-12
    assert math.degrees(np.linalg.norm(cd.so3_log(ref.rot @ tip.rot.T))) <= 1.0 + 1e-9


######################################
### MRV plume scan (post-docking) ###
######################################

_mrv = importlib.import_module(f"{_PKG}.mrv_approach")


def _box_triangles(half):
    """Closed box mesh (12 triangles) centred at the origin, half extents `half`."""
    hx, hy, hz = half
    v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for a, b, c, d in faces for a, b, c in ((a, b, c), (a, c, d))])


def _vfx():
    vfx = object.__new__(_mrv.ThrusterVfx)
    vfx.cfg = _mrv.MrvApproachCfg()
    vfx.cluster_centre = np.zeros(3)
    vfx.cluster_half = np.array([2.0, 1.5, 1.5])
    vfx._fire_anchor_cache = {}
    vfx._hull_tris = _box_triangles([2.5, 2.0, 2.0])
    return vfx


def test_plume_scan_runs_once_per_face_not_per_direction(monkeypatch):
    """Station keeping after the docking turns the thrust by fractions of a degree every
    step; that must not re-scan the hull (it stalled the simulation at ROBOT_RELEASE)."""
    calls = []
    real = _mrv._wall_anchors
    monkeypatch.setattr(_mrv, "_wall_anchors", lambda noz, d, tris, *a, **k: calls.append(np.array(d)) or real(noz, d, tris, *a, **k))
    vfx, hull = _vfx(), Frame(np.zeros(3), np.eye(3))
    ref = vfx.fire_anchors(np.array([1.0, 0.0, 0.0]), hull)
    for i in range(50):  # the measured post-docking wobble: up to ~0.03 off-axis
        tilted = vfx.fire_anchors(np.array([1.0, -0.028 * math.sin(i), 0.01 * math.cos(i)]), hull)
        assert np.allclose([p for p, _ in tilted], [p for p, _ in ref])
    assert len(calls) == 1 and np.allclose(calls[0], [1.0, 0.0, 0.0])  # scanned along the axis (fast path)
    vfx.fire_anchors(np.array([-1.0, 0.02, 0.0]), hull)  # the opposite face: its own single scan
    assert len(calls) == 2
