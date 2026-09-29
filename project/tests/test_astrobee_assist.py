"""Offline tests for the Astrobee docking damping assist (`astrobee_assist.py`).

No Isaac Sim needed:

    cd project && uv run pytest tests/test_astrobee_assist.py -q
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
# are loaded as members of a stand-in package instead (as in `test_astrobee_observer.py`).
_PKG_DIR = Path(__file__).resolve().parents[1].joinpath("srb", "tasks", "manipulation", "debris_capture")
_PKG = "_debris_capture_offline"
if _PKG not in sys.modules:
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_PKG_DIR)]
    sys.modules[_PKG] = pkg
aa = importlib.import_module(f"{_PKG}.astrobee_assist")
load_vision_config = importlib.import_module(f"{_PKG}.vision").load_vision_config
Frame = importlib.import_module(f"{_PKG}.frames").Frame

DT = 1.0 / 60.0
SEED = 0
# States in which the probe is at / inside the nozzle: the Astrobee must never hold it there
CONTACT_STATES = {"Z_APPROACH", "FINAL_INSERTION", "DOCK_READY", "DOCKED", "DOCK_HOLDING"}


def _mission_states():
    """`State` enum values of the demo, read from its source (it imports Isaac Lab)."""
    tree = ast.parse((_PKG_DIR / "vision_capture_demo.py").read_text())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "State")
    return {n.value.value for n in cls.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)}


def _cfg(**kw):
    cfg = aa.AstrobeeAssistCfg(**{"enabled": True, **kw})
    aa.validate_assist_cfg(cfg)
    return cfg


## Damping law


def test_damping_force_opposes_the_velocity_error():
    f = aa.damping_force([0.01, 0.0, 0.0], [0.0, 0.0, 0.0], damping=300.0, max_force=10.0)
    assert np.allclose(f, [-3.0, 0.0, 0.0])


def test_damping_force_is_zero_on_the_reference_motion():
    f = aa.damping_force([0.02, -0.01, 0.005], [0.02, -0.01, 0.005], damping=300.0, max_force=10.0)
    assert np.allclose(f, 0.0)


def test_damping_force_is_clipped_to_the_thrust_limit_keeping_its_direction():
    f = aa.damping_force([0.3, 0.4, 0.0], [0.0, 0.0, 0.0], damping=300.0, max_force=10.0)
    assert math.isclose(float(np.linalg.norm(f)), 10.0, rel_tol=1e-9)
    assert np.allclose(f / np.linalg.norm(f), [-0.6, -0.8, 0.0])


def _swing(max_force, periods=4.0, amp=0.1, mass=3000.0, period_s=20.0, zeta=0.01):
    """1-D arm + payload mode (mass on a spring, light structural damping) with the
    Astrobee damper on it; returns the final energy / initial energy."""
    w = 2.0 * math.pi / period_s
    k, c0 = mass * w * w, 2.0 * zeta * mass * w
    x, v = amp, 0.0
    e0 = 0.5 * k * x * x
    for _ in range(int(periods * period_s / DT)):
        f = float(aa.damping_force([v, 0, 0], [0, 0, 0], damping=300.0, max_force=max_force)[0])
        v += (-k * x - c0 * v + f) / mass * DT
        x += v * DT
    return (0.5 * k * x * x + 0.5 * mass * v * v) / e0


def test_scaled_thrust_damps_the_measured_swing():
    """Measured in full_6dof: ~0.1 m lateral swing, period ~20 s, 3 t MEP. The default
    5 N removes nearly all of it within the ~80 s alignment window; a real Astrobee's
    ~0.6 N only part of it (4 F / k ~ 8 mm of amplitude per cycle)."""
    assisted = _swing(max_force=aa.AstrobeeAssistCfg().max_force_n)
    real_astrobee = _swing(max_force=0.6)
    none = _swing(max_force=0.0)
    assert assisted < 0.01 * none
    assert 0.2 * none < real_astrobee < 0.8 * none


## Lateral centering


def test_lateral_error_is_perpendicular_to_the_docking_axis():
    e = aa.lateral_error_to_axis(tip=[0.3, -0.05, -1.0], axis_point=[0.0, 0.0, 0.0], axis_dir=[0.0, 0.0, 2.0])
    assert np.allclose(e, [-0.3, 0.05, 0.0])


def test_centering_force_points_at_the_axis_and_fades_out_far_away():
    f = aa.centering_force([0.0, 0.02, 0.0], gain=200.0, radius=0.3)
    assert np.allclose(f, [0.0, 4.0, 0.0])
    half = aa.centering_force([0.0, 0.45, 0.0], gain=200.0, radius=0.3)
    assert np.allclose(half, 0.5 * 200.0 * np.array([0.0, 0.45, 0.0]))
    assert np.allclose(aa.centering_force([0.0, 4.8, 0.0], gain=200.0, radius=0.3), 0.0)  # pre-dock transport


def test_damping_keeps_priority_on_the_thrust_budget():
    cfg = _cfg()
    # damper alone saturates: no centering left
    f = aa.assist_force([0.0, 0.0, 0.1], [0.0, 0.0, 0.0], [0.0, 0.1, 0.0], cfg)
    assert np.allclose(f, [0.0, 0.0, -cfg.max_force_n])
    # at rest: centering only, clipped to F_max, towards the axis
    f = aa.assist_force(np.zeros(3), np.zeros(3), [0.0, 0.1, 0.0], cfg)
    assert np.allclose(f, [0.0, cfg.max_force_n, 0.0])
    # no docking axis: damping only
    assert np.allclose(aa.assist_force(np.zeros(3), np.zeros(3), None, cfg), 0.0)


def _static_offset(cfg, disturbance_n=-14.8, mass=3000.0, period_s=20.0, zeta=0.01, t_end=400.0):
    """1-D lateral (world Y) arm + payload mode pushed off the axis by a constant force
    (the -Y lean: 14.8 N = 50 mm on the ~296 N/m mode), with the assist on it."""
    w = 2.0 * math.pi / period_s
    k, c0 = mass * w * w, 2.0 * zeta * mass * w
    y, v = 0.0, 0.0
    for _ in range(int(t_end / DT)):
        f = float(aa.assist_force([0, v, 0], [0, 0, 0], [0, -y, 0], cfg)[1])
        v += (-k * y - c0 * v + disturbance_n + f) / mass * DT
        y += v * DT
    return y


def test_centering_reduces_a_static_minus_y_offset():
    none = -14.8 / (3000.0 * (2.0 * math.pi / 20.0) ** 2)  # static offset without the assist
    damping_only = _static_offset(_cfg(centering_gain_n_per_m=0.0))
    centered = _static_offset(_cfg())
    assert none == pytest.approx(-0.05, abs=1e-3)
    assert damping_only == pytest.approx(none, abs=1e-3)  # a damper cannot move a static offset
    # F_max = 5 N on k ~ 296 N/m: the offset shrinks by ~17 mm, towards the axis
    assert centered - none == pytest.approx(aa.AstrobeeAssistCfg().max_force_n / 296.1, abs=2e-3)
    assert none < centered < 0.0


def test_planner_centering_never_pushes_along_the_docking_axis():
    mep = _Mep()
    assist = aa.DampingAssist(_cfg())
    pos, vel = np.array([25.0, 15.0, 10.0]), np.zeros(3)
    grip_in_mep = aa.grip_point_in_mep(mep.tip, mep.dir, mep.length, assist.cfg.grip_fraction_from_root)
    axis = np.array([0.0, 0.0, 1.0])  # docking axis along the probe (world Z here)
    t, pushed = 0.0, False
    while t < 130.0:
        state = "HOLDING" if t < 60.0 else "POSITION_ATTITUDE_ALIGN"
        tip = mep.frame(t).point(mep.tip)
        # docking axis 30 mm to -Y of the swing centre, drifting along with the MEP (client)
        lateral = aa.lateral_error_to_axis(tip, mep.v * t + np.array([0.0, -0.03, 0.0]), axis)
        ctx = aa.MepContext(mep=mep.frame(t), grip_in_mep=grip_in_mep, axis_in_mep=mep.dir,
                            grip_velocity=mep.velocity(t), reference_velocity=mep.velocity(t),
                            tip_lateral_error=lateral)
        out = assist.step(t, DT, state, pos, vel, ctx)
        if out is not None:
            vel, pos = (out.pos - pos) / DT, out.pos
            if out.force_w is not None:
                assert abs(float(out.force_w @ axis)) < 1e-9
                assert float(np.linalg.norm(out.force_w)) <= assist.cfg.max_force_n + 1e-9
                assert float(out.force_w @ lateral) >= 0.0
                pushed = pushed or float(np.linalg.norm(out.force_w)) > 1.0
        t += DT
    assert assist.summary()["damping_time_s"] > 60.0 and pushed


## Config


def test_assist_is_off_by_default_and_loads_from_yaml():
    assert not aa.AstrobeeAssistCfg().enabled
    cfg = load_vision_config()
    assert not cfg.astrobee.assist.enabled
    on = load_vision_config(overrides=["astrobee.assist.enabled=true"])
    assert on.astrobee.assist.enabled


@pytest.mark.parametrize("override", [
    "astrobee.assist.max_force_n=0.0",
    "astrobee.assist.damping_n_s_per_m=-1.0",
    "astrobee.assist.centering_gain_n_per_m=-1.0",
    "astrobee.assist.centering_radius_m=0.0",
    "astrobee.assist.grip_fraction_from_root=1.5",
    "astrobee.assist.final_approach_speed_mps=0.0",
    "astrobee.assist.max_accel_mps2=0.0",
    "astrobee.assist.damping_states=[Z_APPROACH]",
    "astrobee.assist.engage_states=[NOT_A_STATE]",
])
def test_assist_config_rejects_bad_values(override):
    with pytest.raises(ValueError):
        load_vision_config(overrides=["astrobee.assist.enabled=true", override])


def test_default_states_are_real_and_release_before_contact():
    cfg = aa.AstrobeeAssistCfg()
    states = _mission_states()
    assert set(cfg.engage_states) <= states and set(cfg.damping_states) <= states
    assert not (set(cfg.engage_states) | set(cfg.damping_states)) & CONTACT_STATES


## Grip point


def test_grip_point_is_near_the_probe_root_far_from_the_tip():
    tip, d, length = np.array([0.0, 0.0, 5.0]), np.array([0.0, 0.0, 1.0]), 4.0
    g = aa.grip_point_in_mep(tip, d, length, fraction_from_root=0.15)
    assert np.allclose(g, [0.0, 0.0, 1.6])
    assert float(np.linalg.norm(tip - g)) >= 0.85 * length - 1e-9


## Flight


def test_flyer_respects_acceleration_and_relative_speed_limits():
    p, v = np.zeros(3), np.zeros(3)
    target_v = np.array([0.02, 0.0, 0.0])
    for i in range(int(200 / DT)):
        target = np.array([30.0, 5.0, 0.0]) + target_v * i * DT
        p2, v2 = aa.fly_towards(p, v, target, target_v, DT, max_rel_speed=1.0, max_accel=0.1)
        assert float(np.linalg.norm(v2 - v)) / DT <= 0.1 + 1e-9
        assert float(np.linalg.norm(v2 - target_v)) <= 1.0 + 0.1 * DT + 1e-9
        p, v = p2, v2
    assert float(np.linalg.norm(p - target)) < 0.01
    assert float(np.linalg.norm(v - target_v)) < 0.005


## Planner


class _Mep:
    """MEP drifting at constant velocity (with the client) plus a lateral swing."""

    def __init__(self, v=(0.02, 0.0, 0.0), swing=0.05, period=20.0):
        self.v = np.asarray(v, dtype=float)
        self.swing, self.w = swing, 2.0 * math.pi / period
        self.tip, self.dir, self.length = np.array([0.0, 0.0, 6.0]), np.array([0.0, 0.0, 1.0]), 4.0

    def frame(self, t):
        pos = self.v * t + np.array([0.0, self.swing * math.sin(self.w * t), 0.0])
        return Frame(pos, np.eye(3))

    def velocity(self, t):
        return self.v + np.array([0.0, self.swing * self.w * math.cos(self.w * t), 0.0])


def _run(assist, mep, schedule, t_end, start=(25.0, 15.0, 10.0)):
    """Drive the planner with a mission-state schedule [(t_from, state), ...]."""
    pos, vel = np.asarray(start, dtype=float), np.zeros(3)
    log = []
    t = 0.0
    while t < t_end:
        state = [s for t0, s in schedule if t >= t0][-1]
        grip_in_mep = aa.grip_point_in_mep(mep.tip, mep.dir, mep.length, assist.cfg.grip_fraction_from_root)
        ctx = aa.MepContext(mep=mep.frame(t), grip_in_mep=grip_in_mep, axis_in_mep=mep.dir,
                            grip_velocity=mep.velocity(t), reference_velocity=mep.v)
        out = assist.step(t, DT, state, pos, vel, ctx)
        if out is not None:
            vel = (out.pos - pos) / DT
            pos = out.pos
        log.append((t, state, out))
        t += DT
    return log


SCHEDULE = [(0.0, "HOLDING"), (60.0, "PRE_DOCK_APPROACH"), (120.0, "POSITION_ATTITUDE_ALIGN"),
            (150.0, "Z_APPROACH"), (160.0, "POSITION_ATTITUDE_ALIGN"), (170.0, "Z_APPROACH")]


def test_planner_is_passive_when_disabled_or_before_engaging():
    mep = _Mep()
    off = aa.DampingAssist(aa.AstrobeeAssistCfg())
    assert all(o is None for _, _, o in _run(off, mep, SCHEDULE, 20.0))
    early = aa.DampingAssist(_cfg())
    assert all(o is None for _, _, o in _run(early, mep, [(0.0, "MRV_MOVE_STEP_1")], 20.0))


def test_planner_grasps_velocity_matched_damps_then_releases_before_contact():
    mep = _Mep()
    assist = aa.DampingAssist(_cfg())
    log = _run(assist, mep, SCHEDULE, 200.0)
    phases = [o.phase for _, _, o in log if o is not None]
    order = list(dict.fromkeys(phases))
    assert order == ["ASTROBEE_ASSIST_RENDEZVOUS", "ASTROBEE_ASSIST_FINAL_APPROACH", "ASTROBEE_ASSIST_GRASPED",
                     "ASTROBEE_DAMPING_ASSIST", "ASTROBEE_ASSIST_RELEASE", "ASTROBEE_ASSIST_STANDBY"]
    s = assist.summary()
    assert s["grasped"] and s["grasp_relative_speed_mps"] <= assist.cfg.grasp_speed_tol_mps + 1e-9
    assert s["grasp_time_s"] < 60.0  # attached before the damping window opens
    for t, state, o in log:
        if o is None:
            continue
        if o.force_w is not None and float(np.linalg.norm(o.force_w)) > 0.0:
            # thrust only while holding the probe in a damping state, never near contact
            assert state in assist.cfg.damping_states and o.phase == "ASTROBEE_DAMPING_ASSIST"
            assert float(np.linalg.norm(o.force_w)) <= assist.cfg.max_force_n + 1e-9
        if state in CONTACT_STATES:
            assert o.phase in ("ASTROBEE_ASSIST_RELEASE", "ASTROBEE_ASSIST_STANDBY")
    # one-shot: the rollback to aligning at t = 160 s does not re-grasp
    assert "ASTROBEE_DAMPING_ASSIST" not in [o.phase for t, _, o in log if o is not None and t >= 150.0]
    assert s["damping_time_s"] == pytest.approx(90.0, abs=0.1)
    assert s["energy_removed_j"] > 0.0 and s["max_force_n"] <= assist.cfg.max_force_n + 1e-9


def test_attached_astrobee_rides_with_the_mep_at_the_standoff():
    mep = _Mep()
    assist = aa.DampingAssist(_cfg())
    log = _run(assist, mep, SCHEDULE, 140.0)
    grip_in_mep = aa.grip_point_in_mep(mep.tip, mep.dir, mep.length, assist.cfg.grip_fraction_from_root)
    for t, _, o in log:
        if o is not None and o.phase in ("ASTROBEE_ASSIST_GRASPED", "ASTROBEE_DAMPING_ASSIST"):
            grip = mep.frame(t).point(grip_in_mep)
            r = o.pos - grip
            assert float(np.linalg.norm(r)) == pytest.approx(assist.cfg.grip_standoff_m, abs=1e-6)
            assert abs(float(r @ mep.dir)) < 1e-6  # beside the rod, never in front of the tip
            assert np.allclose(o.aim, grip)


def test_release_backs_off_from_the_probe():
    mep = _Mep()
    assist = aa.DampingAssist(_cfg())
    log = _run(assist, mep, SCHEDULE, 200.0)
    grip_in_mep = aa.grip_point_in_mep(mep.tip, mep.dir, mep.length, assist.cfg.grip_fraction_from_root)
    last_t, _, last = log[-1]
    dist = float(np.linalg.norm(last.pos - mep.frame(last_t).point(grip_in_mep)))
    assert dist == pytest.approx(assist.cfg.standby_distance_m, abs=0.05)
    assert last.force_w is None


def test_failure_state_releases_immediately():
    mep = _Mep()
    assist = aa.DampingAssist(_cfg())
    log = _run(assist, mep, [(0.0, "HOLDING"), (60.0, "PRE_DOCK_APPROACH"), (80.0, "DOCK_FAILED")], 90.0)
    after = [o for t, _, o in log if t >= 80.0]
    assert after[0].phase == "ASTROBEE_ASSIST_RELEASE" and after[0].force_w is None


def test_standby_states_fly_in_early_but_grasp_only_once_the_arm_holds_the_mep():
    """The MEP is free-floating until CAPTURED: the Astrobee may wait beside it, never grab
    it. Starting ~80 m out (the observation ring), it must be attached shortly after the
    damping window opens (the real run has only ~2 s of HOLDING)."""
    mep = _Mep()
    assist = aa.DampingAssist(_cfg())
    schedule = [(0.0, "ARM_DEPLOY"), (12.0, "SEARCH"), (44.0, "HOLDING"), (46.0, "PRE_DOCK_APPROACH"),
                (100.0, "POSITION_ATTITUDE_ALIGN"), (130.0, "Z_APPROACH")]
    log = _run(assist, mep, schedule, 140.0, start=(-50.0, 55.0, 25.0))
    for t, state, o in log:
        if o is not None and o.phase != "ASTROBEE_ASSIST_RENDEZVOUS" and t < 44.0:
            pytest.fail(f"{o.phase} at t = {t:.1f} s while the MEP is still free ({state})")
    first = log[0][2]
    assert first is not None and first.phase == "ASTROBEE_ASSIST_RENDEZVOUS"
    s = assist.summary()
    assert s["grasped"] and s["grasp_time_s"] < 46.0 + 30.0


def test_default_standby_states_are_real_pre_capture_states():
    cfg = aa.AstrobeeAssistCfg()
    assert set(cfg.standby_states) <= _mission_states()
    assert not set(cfg.standby_states) & (set(cfg.engage_states) | set(cfg.damping_states) | CONTACT_STATES)
