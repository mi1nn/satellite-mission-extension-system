"""Offline tests for the Astrobee perching arm kinematics (`astrobee_arm.py`).

No Isaac Sim needed:

    cd project && ~/isaac-sim/python.sh -m pytest tests/test_astrobee_arm.py -q
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
arm = importlib.import_module(f"{_PKG}.astrobee_arm")

# NASA `body` collision box (model.urdf.xacro): the stowed arm tucks into it
BODY_HALF = np.array([0.290513, 0.151942, 0.281129]) / 2.0 + 0.02


def test_stowed_arm_is_inside_the_body():
    f = arm.link_frames(*arm.STOWED)
    for name in ("proximal", "distal"):
        assert np.all(np.abs(f[name][:3, 3]) < BODY_HALF), name
    tip = arm.grip_center_body(*arm.STOWED)
    assert np.all(np.abs(tip) < BODY_HALF)


def test_deployed_arm_reaches_out_behind_the_body():
    tip = arm.grip_center_body(*arm.DEPLOYED)
    assert tip[0] < -BODY_HALF[0]  # aft of the body (-X_B)
    assert 0.3 < float(np.linalg.norm(tip)) < 0.45
    assert abs(tip[1]) < 1e-6  # in the arm plane


def test_joint_limits_hold_for_the_named_poses():
    for q1, q2, g in (arm.STOWED, arm.DEPLOYED):
        assert arm.PROXIMAL_LIMITS[0] <= q1 <= arm.PROXIMAL_LIMITS[1]
        assert arm.DISTAL_LIMITS[0] <= q2 <= arm.DISTAL_LIMITS[1]
        assert 0.0 <= g <= 1.0


def test_gripper_opens_with_g():
    closed, opened = arm.finger_gap(*arm.DEPLOYED[:2], arm.GRIPPER_CLOSED), arm.finger_gap(*arm.DEPLOYED[:2], 1.0)
    assert opened > closed + 0.04


def test_joint_quaternions_are_axis_angle_about_the_urdf_axes():
    q = arm.joint_quats(0.3, -0.2, 1.0)
    w, x, y, z = q["proximal_joint"]
    assert math.isclose(w, math.cos(0.15), abs_tol=1e-9) and math.isclose(y, -math.sin(0.15), abs_tol=1e-9)
    assert abs(x) < 1e-12 and abs(z) < 1e-12
    w, x, y, z = q["distal_joint"]
    assert math.isclose(z, math.sin(-0.1), abs_tol=1e-9)
    # the fingers mirror each other
    assert q["gripper_left_proximal_joint"][3] == pytest.approx(-q["gripper_right_proximal_joint"][3])
    for v in q.values():
        assert math.isclose(sum(c * c for c in v), 1.0, abs_tol=1e-9)


def test_step_joints_is_rate_limited_and_arrives():
    cur = np.array(arm.STOWED, dtype=float)
    target = np.array(arm.DEPLOYED, dtype=float)
    dt = 1.0 / 60.0
    for _ in range(int(20.0 / dt)):
        nxt = arm.step_joints(cur, target, joint_speed=0.6, gripper_speed=1.0, dt=dt)
        assert np.all(np.abs(nxt[:2] - cur[:2]) <= 0.6 * dt + 1e-12)
        assert abs(nxt[2] - cur[2]) <= 1.0 * dt + 1e-12
        cur = nxt
    assert np.allclose(cur, target)


def test_align_body_maps_the_arm_onto_the_world_direction():
    d = np.array([0.0, 1.0, 0.0])
    axis = np.array([0.0, 0.0, 1.0])
    r = arm.align_body(arm.grip_center_body(*arm.DEPLOYED), d, arm.ROD_AXIS_BODY, axis)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9) and np.linalg.det(r) > 0.0
    tip = arm.grip_center_body(*arm.DEPLOYED)
    assert np.allclose(r @ (tip / np.linalg.norm(tip)), d, atol=1e-9)
    assert abs(float((r @ arm.ROD_AXIS_BODY) @ axis)) > 0.99
