"""MRV rendezvous phase in front of the verified vision capture (`vision_capture_demo`).

The capture / transport / docking pipeline is not touched by anything in this module.
What is added ahead of it:

    folded arm -> MRV translation step 1 -> step 2 -> arm deploy -> (existing SEARCH ...)

The "MRV" in this scene is the Canadarm3 base together with the spacecraft hull it is
mounted on (`{ENV}/robot`, a fixed-base articulation, and `{ENV}/scenery`, a static
`XFormPrim`); neither is a free-flying rigid body, so the transit is a commanded rigid
translation of both, not a thruster-driven trajectory. The MRV starts displaced by
`start_offset_m` from its nominal pose and the two legs cancel that offset exactly, so
from ARM_DEPLOY onwards every frame of the scene is the one the existing pipeline was
verified in. The MEP keeps its `six_dof` motion throughout: nothing here touches it (it
only has to start further upstream, see `MrvApproachCfg.planned_duration_s`).

`ThrusterVfx` is a purely visual plume layer on separate prims. It never applies a
force, never carries a collider and never gates the pipeline: a failure to create or
update it is logged and the run continues.

Frames follow the rest of the package: world-frame vectors, quaternions (w, x, y, z).
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from .frames import Frame

if TYPE_CHECKING:
    import torch

    from .vision_task import VisionCaptureTask


##############
### Config ###
##############


@dataclass
class MrvApproachCfg:
    """`mrv:` section of `project/config/vision_capture.yaml`."""

    # false: the demo starts exactly as before (arm at the observation pose, MRV at its
    # nominal pose) -- the configuration the capture/docking pipeline was verified in
    enabled: bool = True

    ## Initial placement ------------------------------------------------------------
    # World-frame offset of the MRV *start* pose from its nominal pose [m]. The two
    # translation legs cancel it, so the MRV arrives exactly at the nominal pose.
    # Negative X = further back from the MEP (the approach axis is world +X), negative
    # Z = below it, so leg 1 is a horizontal run and leg 2 a climb, as in the spec.
    start_offset_m: List[float] = field(default_factory=lambda: [-5.0, 0.0, -3.0])
    # Per-axis fraction of `start_offset_m` cancelled by leg 1; leg 2 cancels the rest.
    # [1, 1, 0]: leg 1 is the horizontal move, leg 2 the vertical one.
    step1_axis_mask: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.0])

    ## Transit ----------------------------------------------------------------------
    speed_mps: float = 0.4
    accel_mps2: float = 0.15
    # Arrival is judged on the distance to the leg target, never on elapsed time
    position_tolerance_m: float = 0.02
    # Pause at the end of each leg (visible separation of the two steps)
    settle_time_s: float = 1.0

    ## Folded arm -------------------------------------------------------------------
    # `canadarm3_large_joint_1..7` [deg]. The asset's rest pose (joints 2-5 at zero)
    # already folds the two 3.66 m booms back onto each other (measured from the USD
    # joint frames: link 3 -> 4 is +3.658 m along Z, link 4 -> 5 is -3.663 m; the elbow
    # itself still swings up to 4.25 m before folding back -- that excursion is
    # unavoidable given the boom length and stays clear of the hull either way).
    # Joint 1 keeps the azimuth of the arm's own `init_state` so the deployment only has
    # to unfold the booms. Joint 6 (wrist pitch) sets where the folded stub ends up: an
    # FK search over joint 6 found 135 deg tucks the wrist close ABOVE the shoulder
    # (EE contact point 1.32 m above the base, only 0.72 m sideways -- a vertical stow,
    # base reach 1.50 m) instead of out to the side (-90 deg: 0.59 m up but 2.17 m
    # sideways, base reach 2.24 m, reads as a horizontal fold).
    folded_joint_deg: List[float] = field(default_factory=lambda: [50.0, 0.0, 0.0, 0.0, 0.0, 135.0, 0.0])
    # Joint-space interpolation folded -> the pose the existing pipeline starts from
    deploy_duration_s: float = 10.0
    # Upper bound on the time the drives get to settle on the deployed pose before
    # SEARCH takes over; the state ends as soon as every joint is within
    # `deploy_tolerance_deg` of the target. 1.5 s left 0.36 deg / 57 mm at the EE
    # (measured), which is more than the 30 mm SEARCH treats as "arrived".
    deploy_settle_s: float = 4.0
    deploy_tolerance_deg: float = 0.1
    # Abort the phase if a leg or the deployment does not finish within this time [s]
    stage_timeout_s: float = 120.0

    ## MEP rendezvous ---------------------------------------------------------------
    # The MEP drifts and turns during the whole approach. To make the capture start from
    # exactly the state the pipeline was verified in, the free-flying bodies are rolled
    # BACK by this many seconds at t = 0 (`back_propagate_free_body`), so that after the
    # approach they are where the verified run has them at its own t = 0.
    # < 0: auto = `planned_duration_s()`, which is what it should normally be.
    mep_rendezvous_delay_s: float = -1.0

    ## Thruster plume (visual only) -------------------------------------------------
    vfx_enabled: bool = True
    # Plume cone size [m], sized to the 3.4x scaled hull
    plume_length_m: float = 2.5
    plume_radius_m: float = 0.45
    # Per-step random variation of the plume length (0 = steady)
    plume_flicker: float = 0.18
    plume_color: List[float] = field(default_factory=lambda: [0.55, 0.78, 1.0])
    # Blazing near-nozzle core (white-hot with a blue tint) and the cooled, diffuse tail
    # colour the plume fades to (warm amber). `plume_color` sits between the two as the
    # mid-plume tone -- the three together give the hot-core -> cooling-tail gradient a
    # real thruster exhaust shows instead of one flat-tinted wedge.
    plume_core_color: List[float] = field(default_factory=lambda: [0.85, 0.93, 1.0])
    plume_tail_color: List[float] = field(default_factory=lambda: [1.0, 0.55, 0.15])
    # How far apart the four plumes sit on the exhaust face, as a fraction of the
    # thruster box half extent (1.0 = its corners). The box spans the full thruster
    # layout (6.4 m vertically on this hull), so the corners read as four jets floating
    # away from the bus; 0.45 keeps the quad against it.
    plume_spread: float = 0.45
    # Apex position along the exhaust axis, as a fraction of the box half extent. < 1
    # tucks the nozzle slightly inside the hull silhouette instead of hanging off it.
    plume_inset: float = 0.85
    # "flow": volumetric fire (Omniverse Flow combustion sim, rendered by RTX);
    # "cones": the translucent cone meshes. "flow" falls back to "cones" if the
    # omni.flowusd extension cannot be loaded.
    vfx_style: str = "flow"
    # Exhaust speed along the plume axis [m/s]; with `fire_cooling_rate` it sets how
    # far the flame reaches before it cools below the visible part of the colormap.
    fire_speed_mps: float = 25.0
    fire_cooling_rate: float = 1.5
    # Exhaust density fade [1/s]: sets the visible plume length. Keep it above
    # `fire_cooling_rate`, or the gas turns into a dark trail once it has cooled.
    fire_fade_rate: float = 6.0
    # How far from the nozzle the Flow grid is kept open [m]: the longest the plume can
    # get (its actual reach is set by the speed and fade rate)
    fire_length_m: float = 5.0
    # Flow voxel size [m]: smaller = finer flame detail, more GPU memory/time
    fire_cell_size_m: float = 0.07

    ## Derived ----------------------------------------------------------------------
    def offset(self) -> np.ndarray:
        return np.asarray(self.start_offset_m, dtype=float).reshape(3)

    def mask(self) -> np.ndarray:
        return np.clip(np.asarray(self.step1_axis_mask, dtype=float).reshape(3), 0.0, 1.0)

    def leg_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """Offsets (relative to the nominal pose) at the end of leg 1 and leg 2."""
        d0 = self.offset()
        after_1 = d0 * (1.0 - self.mask())  # the masked components are cancelled
        return after_1, np.zeros(3)

    def leg_distances(self) -> Tuple[float, float]:
        after_1, _ = self.leg_targets()
        return float(np.linalg.norm(after_1 - self.offset())), float(np.linalg.norm(after_1))

    def leg_time_s(self, distance: float) -> float:
        """Duration of a trapezoidal leg (`speed_mps`, `accel_mps2`) of `distance` [m]."""
        if distance <= 0.0:
            return 0.0
        v, a = max(self.speed_mps, 1e-6), max(self.accel_mps2, 1e-6)
        if distance <= v * v / a:  # triangular profile: never reaches `speed_mps`
            return 2.0 * math.sqrt(distance / a)
        return distance / v + v / a

    def planned_duration_s(self) -> float:
        """Simulated time from t = 0 until the capture pipeline takes over (SEARCH).

        Used for the MEP rendezvous offset, so the MEP reaches its nominal pose when the
        capture starts instead of while the MRV is still in transit.
        """
        d1, d2 = self.leg_distances()
        return (
            0.5  # INIT
            + self.leg_time_s(d1)
            + self.settle_time_s
            + self.leg_time_s(d2)
            + self.settle_time_s
            + self.deploy_duration_s
            + self.deploy_settle_s
        )

    def roll_back_s(self) -> float:
        """Seconds the free-flying bodies are rolled back at t = 0 (0 when disabled)."""
        return self.rendezvous_delay_s()

    def rendezvous_delay_s(self) -> float:
        if not self.enabled:
            return 0.0
        return self.planned_duration_s() if self.mep_rendezvous_delay_s < 0.0 else float(self.mep_rendezvous_delay_s)

    def folded_joint_rad(self) -> np.ndarray:
        return np.radians(np.asarray(self.folded_joint_deg, dtype=float))


def validate_mrv_cfg(cfg: MrvApproachCfg):
    if len(cfg.start_offset_m) != 3 or not np.isfinite(cfg.start_offset_m).all():
        raise ValueError(f"mrv.start_offset_m must be 3 finite values [m], got {cfg.start_offset_m}")
    if len(cfg.step1_axis_mask) != 3:
        raise ValueError(f"mrv.step1_axis_mask must be 3 values, got {cfg.step1_axis_mask}")
    if len(cfg.folded_joint_deg) != 7:
        raise ValueError(f"mrv.folded_joint_deg must hold 7 values (canadarm3_large_joint_1..7), got {cfg.folded_joint_deg}")
    if len(cfg.plume_color) != 3:
        raise ValueError(f"mrv.plume_color must be 3 values (r, g, b), got {cfg.plume_color}")
    if len(cfg.plume_core_color) != 3:
        raise ValueError(f"mrv.plume_core_color must be 3 values (r, g, b), got {cfg.plume_core_color}")
    if len(cfg.plume_tail_color) != 3:
        raise ValueError(f"mrv.plume_tail_color must be 3 values (r, g, b), got {cfg.plume_tail_color}")
    if cfg.vfx_style not in ("flow", "cones"):
        raise ValueError(f"mrv.vfx_style must be 'flow' or 'cones', got {cfg.vfx_style!r}")
    if cfg.speed_mps <= 0.0 or cfg.accel_mps2 <= 0.0:
        raise ValueError("mrv.speed_mps and mrv.accel_mps2 must be > 0")
    if cfg.position_tolerance_m <= 0.0:
        raise ValueError("mrv.position_tolerance_m must be > 0")
    if cfg.deploy_duration_s <= 0.0:
        raise ValueError("mrv.deploy_duration_s must be > 0")
    if not cfg.enabled:
        return
    d1, d2 = cfg.leg_distances()
    if min(d1, d2) < 1e-3:
        raise ValueError(
            "mrv: both translation legs must have a non-zero length "
            f"(leg 1 {d1:.3f} m, leg 2 {d2:.3f} m from start_offset_m {cfg.start_offset_m} "
            f"and step1_axis_mask {cfg.step1_axis_mask}); the demo requires two visible steps"
        )


####################
### Transit legs ###
####################


class TransitLeg:
    """One straight, velocity-profiled translation between two MRV offsets.

    The reference accelerates at `accel_mps2` up to `speed_mps` and decelerates on the
    same ramp into the target, so the (heavy, visually large) MRV never steps.
    """

    def __init__(self, start: np.ndarray, target: np.ndarray, speed: float, accel: float):
        self.start = np.asarray(start, dtype=float).copy()
        self.target = np.asarray(target, dtype=float).copy()
        delta = self.target - self.start
        self.length = float(np.linalg.norm(delta))
        self.dir = delta / self.length if self.length > 1e-9 else np.zeros(3)
        self.speed = float(speed)
        self.accel = float(accel)
        self.travelled = 0.0
        self.v = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.length - self.travelled)

    def step(self, dt: float) -> np.ndarray:
        """Advance the reference by one control step and return the new MRV offset."""
        # Decelerate so the reference arrives with v -> 0 (v^2 = 2 a s_remaining)
        v_stop = math.sqrt(max(0.0, 2.0 * self.accel * self.remaining))
        self.v = min(self.speed, self.v + self.accel * dt, v_stop)
        self.travelled = min(self.length, self.travelled + self.v * dt)
        return self.start + self.travelled * self.dir

    def done(self, tolerance: float) -> bool:
        return self.remaining <= tolerance


class MrvTransit:
    """Rigid translation of the MRV: Canadarm3 articulation root + spacecraft hull.

    The robot is a fixed-base articulation (`canadarm3_large_0/root_joint`), so its pose
    is commanded through `write_root_pose_to_sim` (which moves the whole articulation and
    leaves the joint positions untouched) and the hull, a static `XFormPrim`, follows with
    the same offset. Orientation is never changed -- this phase is pure translation.
    """

    def __init__(self, task: "VisionCaptureTask", arm):
        import torch

        self._torch = torch
        self.task = task
        self.arm = arm
        self.robot = task._robot
        self.device = self.robot.device
        self.nominal_root = self.robot.data.root_pos_w[0].clone()
        self.root_quat = self.robot.data.root_quat_w[0].clone()
        self.scenery = task.scene.extras.get("scenery")
        self.scenery_nominal = None
        self.scenery_quat = None
        self.hull_nominal: Optional[Frame] = None
        if self.scenery is not None:
            pos, quat = self.scenery.get_world_poses()
            self.scenery_nominal = pos[0].clone()
            self.scenery_quat = quat[0].clone()
            # Hull pose at zero offset, for the plume layer (which is built after this)
            self.hull_nominal = Frame.from_pos_quat(
                self.scenery_nominal.cpu().numpy().tolist(), self.scenery_quat.cpu().numpy().tolist()
            )
        self._offset = np.zeros(3)

    @property
    def offset(self) -> np.ndarray:
        return self._offset.copy()

    def root_pos(self) -> np.ndarray:
        return self.robot.data.root_pos_w[0].cpu().numpy().astype(float)

    def apply(self, offset) -> None:
        """Place the MRV at `nominal + offset` (world, translation only)."""
        torch = self._torch
        d = np.asarray(offset, dtype=float).reshape(3)
        self._offset = d.copy()
        delta = torch.tensor(d, dtype=self.nominal_root.dtype, device=self.nominal_root.device)
        pos = (self.nominal_root + delta).reshape(1, 3)
        pose = torch.cat((pos, self.root_quat.reshape(1, 4)), dim=1)
        self.robot.write_root_pose_to_sim(pose)
        if self.scenery is not None:
            s_delta = delta.to(self.scenery_nominal.device).to(self.scenery_nominal.dtype)
            self.scenery.set_world_poses(
                positions=(self.scenery_nominal + s_delta).reshape(1, 3),
                orientations=self.scenery_quat.reshape(1, 4),
            )
        # The IK of the capture pipeline expresses its goals in the arm base frame
        self.arm.refresh_base()

    def error_m(self, target) -> float:
        """Distance of the *measured* articulation root from `nominal + target` [m]."""
        return float(np.linalg.norm(self.root_pos() - (self.nominal_root.cpu().numpy() + np.asarray(target, dtype=float))))


def back_propagate_free_body(pose: Frame, v_w, w_w, com_body, dt: float) -> Frame:
    """Pose a free-flying rigid body must START from to be at `pose` after `dt` seconds.

    Free flight as PhysX integrates it: the centre of mass travels in a straight line and
    the body turns about it at a constant world-frame rate,

        COM(t) = COM(0) + v t,   R(t) = Exp([w] t) R(0),   p(t) = COM(t) - R(t) c_body

    so the inverse is R(0) = Exp(-[w] dt) R(dt) and COM(0) = COM(dt) - v dt. `com_body`
    is the centre of mass in the body frame, a mass property that is only known once the
    simulation is up -- which is why this is applied in the demo's `start()` and not when
    the scene is built.
    """
    # Imported here, not at module import time: `vision.py` imports this module.
    from .vision import so3_exp

    v = np.asarray(v_w, dtype=float).reshape(3)
    w = np.asarray(w_w, dtype=float).reshape(3)
    c = np.asarray(com_body, dtype=float).reshape(3)
    com_end = pose.pos + pose.rot @ c
    rot_start = so3_exp(-w * float(dt)) @ pose.rot
    com_start = com_end - v * float(dt)
    return Frame(com_start - rot_start @ c, rot_start)


####################
### Thruster VFX ###
####################


def thruster_box(task: "VisionCaptureTask") -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Half extents and centre (hull body frame, metres) of the MRV's thruster cluster.

    Read from the thruster positions the spacecraft asset itself declares
    (`ThrustActionGroup` -> `ThrusterCfg.offset`, in the unscaled body frame) and scaled
    by the spawn scale of the hull, so nothing about the nozzle layout is hard-coded
    here. Returns None when the asset declares no thrusters.
    """
    scenery = getattr(task.cfg, "_scenery", None)
    if scenery is None:
        return None
    actions = getattr(scenery, "actions", None)
    thrust = getattr(actions, "thrust", None)
    thrusters = getattr(thrust, "thrusters", None)
    if not thrusters:
        return None
    pts = np.array([list(t.offset) for t in thrusters], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    scale = np.asarray(scenery.asset_cfg.spawn.scale, dtype=float)
    pts = pts * scale
    centre = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    half = 0.5 * (pts.max(axis=0) - pts.min(axis=0))
    return centre, np.maximum(half, 1e-3)


FLOW_LAYER = 7301  # Flow layer id of the plume; keeps it from coupling with any other Flow prim


def _flow_set(prim, name: str, value, type_name):
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, type_name)
    attr.Set(value)


def enable_flow_rendering():
    """Load omni.flowusd and switch on its RTX volume pass (both idempotent)."""
    import carb.settings
    import omni.kit.app

    mgr = omni.kit.app.get_app().get_extension_manager()
    if not mgr.is_extension_enabled("omni.flowusd"):
        mgr.set_extension_enabled_immediate("omni.flowusd", True)
    if not mgr.is_extension_enabled("omni.flowusd"):
        raise RuntimeError("the omni.flowusd extension could not be enabled")
    settings = carb.settings.get_settings()
    settings.set("/rtx/flow/enabled", True)
    settings.set("/rtx/flow/rayTracedTranslucencyEnabled", True)
    settings.set("/rtx/flow/rayTracedReflectionsEnabled", True)
    settings.set("/rtx/flow/pathTracingEnabled", True)


class FlowFire:
    """Volumetric thruster plume: one Flow sim, one sphere emitter per nozzle.

    Each emitter sits under a nozzle Xform (whose +Z is the exhaust direction) and
    injects hot fuel at high speed along its local +Z. The fuel is burnt fast and close
    to the nozzle with no expansion and little swirl -- slow combustion with vorticity
    billows like a campfire, a thruster jet leaves the nozzle already hot and fast --
    then cooled, and RTX ray-marches its temperature through a colormap -- a white-hot core at the
    nozzle fading through blue to nothing (no gravity, no buoyancy: vacuum). Visual only:
    Flow does not touch PhysX here (`physicsCollisionEnabled` off, no body coupling).
    """

    def __init__(self, stage, root_path: str, nozzle_paths: Sequence[str], cfg: MrvApproachCfg):
        from pxr import Gf, Sdf, UsdGeom, Vt

        enable_flow_rendering()
        T = Sdf.ValueTypeNames
        L = FLOW_LAYER
        UsdGeom.Xform.Define(stage, root_path)

        sim = stage.DefinePrim(f"{root_path}/flowSimulate", "FlowSimulate")
        if not sim.IsValid() or sim.GetTypeName() != "FlowSimulate":
            raise RuntimeError("the FlowSimulate schema is not registered")
        _flow_set(sim, "layer", L, T.Int)
        _flow_set(sim, "densityCellSize", float(cfg.fire_cell_size_m), T.Float)
        _flow_set(sim, "stepsPerSecond", 60.0, T.Float)
        _flow_set(sim, "physicsCollisionEnabled", False, T.Bool)
        # Isaac Lab steps the app with the timeline's auto-update off, so Flow sees a zero
        # delta time and would never step on its own
        _flow_set(sim, "simulateWhenPaused", True, T.Bool)
        _flow_set(sim, "forceSimulate", True, T.Bool)

        adv = stage.DefinePrim(f"{root_path}/flowSimulate/advection", "FlowAdvectionCombustionParams")
        _flow_set(adv, "combustionEnabled", True, T.Bool)
        _flow_set(adv, "gravity", Gf.Vec3f(0.0, 0.0, 0.0), T.Float3)
        _flow_set(adv, "buoyancyPerTemp", 0.0, T.Float)
        _flow_set(adv, "buoyancyPerSmoke", 0.0, T.Float)
        _flow_set(adv, "coolingRate", float(cfg.fire_cooling_rate), T.Float)
        _flow_set(adv, "ignitionTemp", 0.02, T.Float)
        _flow_set(adv, "burnPerTemp", 3.0, T.Float)
        _flow_set(adv, "fuelPerBurn", 0.3, T.Float)
        _flow_set(adv, "tempPerBurn", 8.0, T.Float)
        _flow_set(adv, "smokePerBurn", 1.0, T.Float)
        _flow_set(adv, "divergencePerBurn", 0.0, T.Float)
        for channel, damping, fade in (("smoke", 0.0, float(cfg.fire_fade_rate)), ("velocity", 0.0, 0.2),
                                       ("temperature", 0.0, 0.0), ("fuel", 0.0, 4.0), ("burn", 0.0, 4.0)):
            ch = stage.DefinePrim(f"{root_path}/flowSimulate/advection/{channel}", "FlowAdvectionChannelParams")
            _flow_set(ch, "damping", damping, T.Float)
            _flow_set(ch, "fade", fade, T.Float)
        vort = stage.DefinePrim(f"{root_path}/flowSimulate/vorticity", "FlowVorticityParams")
        _flow_set(vort, "enabled", True, T.Bool)
        # just enough swirl to keep the jet edge alive; more reads as a billowing fire
        _flow_set(vort, "forceScale", 0.12, T.Float)
        _flow_set(vort, "velocityMask", 1.0, T.Float)
        stage.DefinePrim(f"{root_path}/flowSimulate/pressure", "FlowPressureParams")
        alloc = stage.DefinePrim(f"{root_path}/flowSimulate/summaryAllocate", "FlowSummaryAllocateParams")
        _flow_set(alloc, "smokeThreshold", 0.02, T.Float)

        off = stage.DefinePrim(f"{root_path}/flowOffscreen", "FlowOffscreen")
        _flow_set(off, "layer", L, T.Int)
        cmap = stage.DefinePrim(f"{root_path}/flowOffscreen/colormap", "FlowRayMarchColormapParams")
        # temperature -> (HDR rgb, alpha): transparent when cool, a faint blue haze,
        # bright blue, a blue-white core at the nozzle
        _flow_set(cmap, "xPoints", Vt.FloatArray([0.0, 0.08, 0.25, 0.5, 0.8, 1.0]), T.FloatArray)
        _flow_set(cmap, "rgbaPoints", Vt.Vec4fArray([
            Gf.Vec4f(0.0, 0.0, 0.0, 0.0),
            Gf.Vec4f(0.3, 0.45, 1.4, 0.02),
            Gf.Vec4f(1.2, 2.0, 6.0, 0.10),
            Gf.Vec4f(2.0, 4.5, 14.0, 0.28),
            Gf.Vec4f(6.0, 12.0, 30.0, 0.55),
            Gf.Vec4f(16.0, 24.0, 40.0, 0.75),
        ]), T.Float4Array)
        _flow_set(cmap, "colorScale", 8.0, T.Float)
        shadow = stage.DefinePrim(f"{root_path}/flowOffscreen/shadow", "FlowShadowParams")
        _flow_set(shadow, "enabled", False, T.Bool)  # a flame is self-luminous
        stage.DefinePrim(f"{root_path}/flowOffscreen/debugVolume", "FlowDebugVolumeParams")

        render = stage.DefinePrim(f"{root_path}/flowRender", "FlowRender")
        _flow_set(render, "layer", L, T.Int)
        march = stage.DefinePrim(f"{root_path}/flowRender/rayMarch", "FlowRayMarchParams")
        _flow_set(march, "attenuation", 2.0, T.Float)
        _flow_set(march, "stepSizeScale", 0.75, T.Float)
        stage.DefinePrim(f"{root_path}/flowRender/rayMarch/cloud", "FlowRayMarchCloudParams")

        # One emitter per nozzle injects the hot fuel. Flow only simulates inside grid
        # blocks it has allocated, and it allocates them around emitters: gas that runs
        # past them is dropped, which would cut every plume off about a metre from the
        # nozzle. Allocation-only spheres (zero coupling: they inject nothing) keep the
        # grid open along the exhaust axis up to `fire_length_m`.
        self.emitters = []
        r0 = 1.4 * float(cfg.plume_radius_m)
        length = float(cfg.fire_length_m)
        n_alloc = max(1, int(math.ceil(length / 1.0)))
        for path in nozzle_paths:
            em = stage.DefinePrim(f"{path}/fire_emitter", "FlowEmitterSphere")
            _flow_set(em, "layer", L, T.Int)
            _flow_set(em, "enabled", False, T.Bool)
            _flow_set(em, "position", Gf.Vec3f(0.0, 0.0, 0.6 * r0), T.Float3)
            _flow_set(em, "radius", 0.6 * r0, T.Float)
            _flow_set(em, "radiusIsWorldSpace", True, T.Bool)
            _flow_set(em, "velocity", Gf.Vec3f(0.0, 0.0, float(cfg.fire_speed_mps)), T.Float3)
            _flow_set(em, "velocityIsWorldSpace", False, T.Bool)
            _flow_set(em, "fuel", 1.0, T.Float)
            _flow_set(em, "temperature", 2.0, T.Float)
            _flow_set(em, "smoke", 1.0, T.Float)
            _flow_set(em, "burn", 1.0, T.Float)  # already burning at the nozzle: no dark ignition gap
            for name in ("coupleRateFuel", "coupleRateTemperature", "coupleRateVelocity", "coupleRateSmoke", "coupleRateBurn"):
                _flow_set(em, name, 20.0, T.Float)
            _flow_set(em, "physicsVelocityScale", 0.0, T.Float)
            self.emitters.append(em)
            for k in range(n_alloc):
                al = stage.DefinePrim(f"{path}/fire_alloc_{k}", "FlowEmitterSphere")
                _flow_set(al, "layer", L, T.Int)
                _flow_set(al, "enabled", False, T.Bool)
                _flow_set(al, "position", Gf.Vec3f(0.0, 0.0, (k + 1) * length / n_alloc), T.Float3)
                _flow_set(al, "radius", 2.0 * r0, T.Float)
                _flow_set(al, "radiusIsWorldSpace", True, T.Bool)
                for name in ("coupleRateFuel", "coupleRateTemperature", "coupleRateVelocity",
                             "coupleRateSmoke", "coupleRateBurn", "coupleRateDivergence"):
                    _flow_set(al, name, 0.0, T.Float)
                _flow_set(al, "physicsVelocityScale", 0.0, T.Float)
                self.emitters.append(al)

        self._saved_translucency: Optional[bool] = None
        self._burnout_sub = None
        self._frames_off = 0

    # App updates after the emitters stop until the flame has burnt out (~1 s at 60 Hz)
    BURNOUT_FRAMES = 120
    TRANSLUCENCY = "/rtx/translucency/enabled"

    def set_enabled(self, on: bool):
        """Emitters on/off. RTX real-time only composites Flow volumes with translucency
        on, which the balanced/performance render modes switch off; it is switched on
        only while a flame is burning and restored once it has burnt out, so every other
        frame renders with the app's own settings."""
        import carb.settings
        import omni.kit.app

        for em in self.emitters:
            em.GetAttribute("enabled").Set(bool(on))
        settings = carb.settings.get_settings()
        if on:
            self._stop_burnout()
            if self._saved_translucency is None:
                self._saved_translucency = bool(settings.get(self.TRANSLUCENCY))
                settings.set(self.TRANSLUCENCY, True)
        elif self._saved_translucency is not None and self._burnout_sub is None:
            self._frames_off = 0
            self._burnout_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="mrv_thruster_fire_burnout")

    def _on_update(self, _event):
        self._frames_off += 1
        if self._frames_off >= self.BURNOUT_FRAMES:
            self.restore_render_settings()

    def _stop_burnout(self):
        if self._burnout_sub is not None:
            self._burnout_sub.unsubscribe()
            self._burnout_sub = None

    def restore_render_settings(self):
        import carb.settings

        self._stop_burnout()
        if self._saved_translucency is not None:
            carb.settings.get_settings().set(self.TRANSLUCENCY, self._saved_translucency)
            self._saved_translucency = None


class ThrusterVfx:
    """Translucent plume cones on their own prims, switched on while the MRV translates.

    Deliberately kept outside the MRV asset: the prims live under
    `{ENV}/mrv_thruster_vfx`, are re-posed from the MRV offset every control step and
    carry no collider, no rigid body and no physics force. The MRV moves because
    `MrvTransit` commands it to; the plume only says so visually.
    """

    def __init__(self, task: "VisionCaptureTask", cfg: MrvApproachCfg,
                 hull_nominal: Optional[Frame] = None, quiet: bool = False):
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        self._gf, self._usdgeom = Gf, UsdGeom
        self.cfg = cfg
        self.stage = task.scene.stage
        self.task = task
        self.visible = False
        self.num_nozzles = 4
        self._rng = np.random.default_rng(7)

        box = thruster_box(task)
        if box is None:
            raise ValueError("the MRV hull asset declares no thrusters (ThrustActionGroup)")
        self.cluster_centre, self.cluster_half = box
        # Hull body frame -> world at the nominal pose (the plume anchors are expressed
        # in the hull frame and re-posed with the MRV offset, never re-measured)
        # `hull_nominal` is the hull pose at zero MRV offset. It is passed in (from
        # `MrvTransit`, which measured it before anything moved) because the MRV may
        # already be displaced by the time the plume layer is built; reading the prim
        # here would then double-count the offset in `update`.
        if hull_nominal is None:
            scenery = task.scene.extras.get("scenery")
            if scenery is None:
                raise ValueError("no 'scenery' prim: the MRV hull is not in the scene")
            pos, quat = scenery.get_world_poses()
            hull_nominal = Frame.from_pos_quat(pos[0].cpu().numpy().tolist(), quat[0].cpu().numpy().tolist())
        self.hull_nominal = hull_nominal
        self._hull_tris: Optional[np.ndarray] = None
        self._fire_anchor_cache = {}

        env = task.scene.env_prim_paths[0]
        self.hull_path = f"{env}/scenery"
        self.root_path = f"{env}/mrv_thruster_vfx"
        root = UsdGeom.Xform.Define(self.stage, self.root_path)
        UsdGeom.Imageable(root.GetPrim()).MakeInvisible()

        # Six nested cones (length factor, radius factor, opacity, colour, emissive
        # intensity): a blazing, near-white throat glow at the nozzle mouth, a tight
        # blue-hot supersonic core, the configured mid-plume colour, and a wider, dimmer
        # amber tail as the exhaust cools and expands -- the colour-temperature gradient
        # and HDR bloom a real thruster plume shows, instead of one flat-tinted wedge.
        core = np.asarray(cfg.plume_core_color, dtype=float)
        mid = np.asarray(cfg.plume_color, dtype=float)
        tail = np.asarray(cfg.plume_tail_color, dtype=float)
        self.shells = (
            (0.05, 0.28, 1.00, core, 8.0),
            (0.18, 0.22, 0.92, core, 5.0),
            (0.38, 0.40, 0.65, core, 2.6),
            (0.65, 0.62, 0.38, mid, 1.4),
            (0.90, 0.88, 0.16, 0.5 * mid + 0.5 * tail, 0.7),
            (1.05, 1.05, 0.06, tail, 0.35),
        )
        self.nozzles = []
        for i in range(self.num_nozzles):
            path = f"{self.root_path}/nozzle_{i}"
            xf = UsdGeom.Xform.Define(self.stage, path)
            xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
            xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            self.nozzles.append(xf)

        self.fire: Optional[FlowFire] = None
        if cfg.vfx_style == "flow":
            try:
                self.fire = FlowFire(self.stage, f"{self.root_path}/flow", [xf.GetPath().pathString for xf in self.nozzles], cfg)
                # Flow volumes are hidden with their ancestors, so the root stays visible
                # and the flame is switched through the emitters instead
                UsdGeom.Imageable(root.GetPrim()).MakeVisible()
            except Exception as e:
                self.fire = None
                self.stage.RemovePrim(f"{self.root_path}/flow")
                print(f"[MRV-VFX] Flow fire unavailable ({e}); using the cone plume instead", flush=True)
        if self.fire is None:
            for xf in self.nozzles:
                path = xf.GetPath().pathString
                for j, (f_len, f_rad, opacity, tint, intensity) in enumerate(self.shells):
                    mesh_path = f"{path}/shell_{j}"
                    _cone_mesh(self.stage, mesh_path, cfg.plume_radius_m * f_rad, cfg.plume_length_m * f_len, Gf, UsdGeom, Vt)
                    _emissive_material(self.stage, f"{mesh_path}/Material", np.clip(tint, 0.0, 1.0), opacity, intensity, Gf, Sdf, UsdShade)
        if not quiet:
            print(f"[MRV-VFX] {self.num_nozzles} {'Flow fire' if self.fire else 'cone'} plumes on separate prims under {self.root_path} "
                  f"(thruster cluster half extents {np.round(self.cluster_half, 2).tolist()} m, hull body frame; "
                  "visual only: no collider, no rigid body, no force)", flush=True)

    def anchors(self, plume_dir_w: np.ndarray, hull: Frame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Nozzle (position, plume direction) pairs in the world frame.

        The plumes sit on the face of the declared thruster box that the exhaust leaves
        through, i.e. the trailing face for the commanded motion, spread over that face
        so they read as an RCS quad rather than one jet.
        """
        d = np.asarray(plume_dir_w, dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-9:
            return []
        d = d / n
        d_body = hull.rot.T @ d  # plume direction in the hull body frame
        axis = int(np.argmax(np.abs(d_body)))
        lat = [i for i in range(3) if i != axis]
        sign = math.copysign(1.0, d_body[axis])
        spread = float(np.clip(self.cfg.plume_spread, 0.0, 1.0))
        out = []
        for s0 in (-1.0, 1.0):
            for s1 in (-1.0, 1.0):
                p = self.cluster_centre.copy()
                p[axis] += sign * self.cluster_half[axis] * float(np.clip(self.cfg.plume_inset, 0.0, 1.0))
                p[lat[0]] += s0 * spread * self.cluster_half[lat[0]]
                p[lat[1]] += s1 * spread * self.cluster_half[lat[1]]
                out.append((hull.point(p), d))
        return out

    def hull_triangles(self, hull: Frame) -> Optional[np.ndarray]:
        """Hull render mesh as triangles (N, 3, 3) in the hull body frame, read once."""
        if self._hull_tris is None:
            from pxr import Usd, UsdGeom

            prim = self.stage.GetPrimAtPath(self.hull_path)
            if not prim:
                return None
            cache = UsdGeom.XformCache()
            tris = []
            for p in Usd.PrimRange(prim):
                mesh = UsdGeom.Mesh(p)
                if not mesh:
                    continue
                pts = mesh.GetPointsAttr().Get()
                counts = mesh.GetFaceVertexCountsAttr().Get()
                idx = mesh.GetFaceVertexIndicesAttr().Get()
                if not pts or not counts or not idx:
                    continue
                m = np.array(cache.GetLocalToWorldTransform(p), dtype=float)  # row-vector convention
                w = np.c_[np.asarray(pts, dtype=float), np.ones(len(pts))] @ m
                body = (w[:, :3] - hull.pos) @ hull.rot  # rows: hull.rot.T @ (p - pos)
                idx = np.asarray(idx)
                counts = np.asarray(counts)
                if (counts == 3).all():
                    fans = idx.reshape(-1, 3)
                else:
                    start, fans = 0, []
                    for c in counts:
                        for k in range(1, c - 1):  # fan triangulation
                            fans.append((idx[start], idx[start + k], idx[start + k + 1]))
                        start += c
                    fans = np.asarray(fans, dtype=int).reshape(-1, 3)
                if len(fans):
                    tris.append(body[fans])
            self._hull_tris = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        return self._hull_tris if len(self._hull_tris) else None

    def fire_anchors(self, plume_dir_w: np.ndarray, hull: Frame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """`anchors`, moved along the exhaust axis onto the hull's outer surface.

        The cone anchors sit inside the hull (`plume_inset`) and rely on the 2.5 m cone
        poking out; a Flow flame starting there burns out inside the bus. Instead the
        exhaust face is scanned with a grid of rays cast from outside, opposite the
        exhaust, and the most common hit depth is taken as the bus wall. Each nozzle then
        moves to the nearest grid point on that wall, so it sits on the flat bus face and
        not inside an appendage in front of it (on this hull the -X face carries a 5 m
        high-gain dish). A nozzle with no wall point nearby keeps its own first hit, or
        the cone anchor if its line misses the hull. Cached per exhaust direction (body
        frame), so this runs once per leg.
        """
        pairs = self.anchors(plume_dir_w, hull)
        if not pairs:
            return pairs
        d_body = hull.rot.T @ pairs[0][1]
        key = tuple(np.round(d_body, 3))
        if key not in self._fire_anchor_cache:
            noz = [hull.rot.T @ (pos - hull.pos) for pos, _ in pairs]
            tris = self.hull_triangles(hull)
            self._fire_anchor_cache[key] = noz if tris is None else _wall_anchors(noz, d_body, tris)
        return [(hull.point(p), d) for p, (_, d) in zip(self._fire_anchor_cache[key], pairs)]

    def update(self, plume_dir_w: Optional[np.ndarray], mrv_offset: np.ndarray):
        """Show the plumes along `plume_dir_w` (None / zero = off) at the current MRV pose."""
        Gf, UsdGeom = self._gf, self._usdgeom
        root = self.stage.GetPrimAtPath(self.root_path)
        on = plume_dir_w is not None and float(np.linalg.norm(plume_dir_w)) > 1e-9 and self.cfg.vfx_enabled
        if self.fire is not None:
            # Emitters off = no new fuel; the flame already in flight burns out on its own
            if on:
                hull = Frame(self.hull_nominal.pos + np.asarray(mrv_offset, dtype=float), self.hull_nominal.rot)
                for xf, (pos, direction) in zip(self.nozzles, self.fire_anchors(np.asarray(plume_dir_w, dtype=float), hull)):
                    frame = _frame_towards(pos, direction)
                    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
                    ops["xformOp:translate"].Set(Gf.Vec3d(*frame.pos.tolist()))
                    ops["xformOp:orient"].Set(Gf.Quatf(*frame.quat))
            if on != self.visible:
                self.fire.set_enabled(on)
                self.visible = on
            return
        if not on:
            if self.visible:
                UsdGeom.Imageable(root).MakeInvisible()
                self.visible = False
            return
        hull = Frame(self.hull_nominal.pos + np.asarray(mrv_offset, dtype=float), self.hull_nominal.rot)
        pairs = self.anchors(np.asarray(plume_dir_w, dtype=float), hull)
        for xf, (pos, direction) in zip(self.nozzles, pairs):
            frame = _frame_towards(pos, direction)
            ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
            ops["xformOp:translate"].Set(Gf.Vec3d(*frame.pos.tolist()))
            ops["xformOp:orient"].Set(Gf.Quatf(*frame.quat))
        if self.cfg.plume_flicker > 0.0:
            for xf in self.nozzles:
                for prim in xf.GetPrim().GetChildren():
                    mesh = UsdGeom.Mesh(prim)
                    if not mesh:
                        continue
                    # Length flickers more than radius: a plume that visibly pulses in
                    # and out along its axis while only breathing slightly in width
                    # reads as turbulent exhaust, not a scaling triangle.
                    f_len = 1.0 + self.cfg.plume_flicker * float(self._rng.uniform(-1.0, 1.0))
                    f_rad = 1.0 + 0.5 * self.cfg.plume_flicker * float(self._rng.uniform(-1.0, 1.0))
                    attr = prim.GetAttribute("xformOp:scale")
                    if attr:
                        attr.Set(Gf.Vec3f(float(f_rad), float(f_rad), float(f_len)))
        if not self.visible:
            UsdGeom.Imageable(root).MakeVisible()
            self.visible = True

    def off(self):
        self.update(None, np.zeros(3))


def _wall_anchors(noz: List[np.ndarray], d: np.ndarray, tris: np.ndarray,
                  margin: float = 2.5, n: int = 13, min_gap: float = 0.8,
                  min_proud: float = 0.25) -> List[np.ndarray]:
    """Nozzle points (body frame) moved onto the dominant flat wall facing along `d`.

    Rays parallel to `d` are cast from outside over an n x n grid spanning the nozzles
    plus `margin`; the most common hit depth (1 cm bins, so a trim panel a couple of
    centimetres proud of the bus face is not mistaken for it) is the wall. Each nozzle takes
    the nearest free wall point (at least `min_gap` from the others).
    """
    axis = int(np.argmax(np.abs(d)))
    lat = [i for i in range(3) if i != axis]
    nz = np.asarray(noz, dtype=float)
    lo = nz[:, lat].min(axis=0) - margin
    hi = nz[:, lat].max(axis=0) + margin
    tl, th = tris[:, :, lat].min(axis=1), tris[:, :, lat].max(axis=1)
    sub = tris[np.all(th >= lo, axis=1) & np.all(tl <= hi, axis=1)]
    if not len(sub):
        return list(nz)

    def depth(a: float, b: float) -> Optional[float]:
        o = np.zeros(3)
        o[lat[0]], o[lat[1]] = a, b
        t = _ray_mesh_first_hit(o + d * 1.0e3, -d, sub)
        return None if t is None else float(d[axis] * (1.0e3 - t))

    pts = np.array([(a, b) for a in np.linspace(lo[0], hi[0], n) for b in np.linspace(lo[1], hi[1], n)])
    if abs(abs(d[axis]) - 1.0) < 1e-6:
        # axis-aligned rays: one projected pass over the triangles instead of n*n casts
        z = _outer_surface(sub, axis, lat, math.copysign(1.0, d[axis]), pts)
        hits = [(a, b, float(zz)) for (a, b), zz in zip(pts, z) if np.isfinite(zz)]
    else:
        hits = [(a, b, zz) for a, b in pts if (zz := depth(a, b)) is not None]
    out = []
    if hits:
        bins, counts = np.unique(np.round(np.array([z for _, _, z in hits]) / 0.01), return_counts=True)
        wall = bins[np.argmax(counts)] * 0.01
        free = [(a, b, z) for a, b, z in hits if abs(z - wall) < 0.006]
    else:
        free = []
        wall = None
    # An appendage standing proud of the wall (a dish, an instrument) hides the bus face
    # behind it: the nozzles then ring it, one per quadrant as closely as possible, so
    # they sit on the face right next to the body instead of off in its corners.
    proud = np.array([(a, b) for a, b, z in hits if wall is not None and z > wall + 0.1]).reshape(-1, 2)
    frac = len(proud) / max(1, len(hits))
    centre = proud.mean(axis=0) if frac >= min_proud else None
    print(f"[MRV-VFX] exhaust face: wall at {wall if wall is None else round(float(wall), 2)} m, "
          f"{frac:.0%} of it covered by an appendage{' (nozzles placed beside it)' if centre is not None else ''}",
          flush=True)
    if centre is not None:
        # beside it: outside its extent along one lateral axis, within it along the other
        half = 0.5 * (proud.max(axis=0) - proud.min(axis=0))
        for i, j in ((0, 1), (1, 0)):
            beside = [c for c in free if abs(c[i] - centre[i]) > 0.9 * half[i] and abs(c[j] - centre[j]) <= half[j]]
            if len(beside) >= len(nz):
                free = beside
                break
    mid = nz[:, lat].mean(axis=0)
    for p in nz:
        p = p.copy()
        cands = [c for c in free if all(np.hypot(c[0] - q[lat[0]], c[1] - q[lat[1]]) >= min_gap for q in out)]
        target = (p[lat[0]], p[lat[1]])
        if centre is not None:
            s = np.sign(p[lat] - mid)
            quad = [c for c in cands if np.sign(c[0] - centre[0]) == s[0] and np.sign(c[1] - centre[1]) == s[1]]
            if quad:
                cands, target = quad, (centre[0], centre[1])
        if cands:
            a, b, z = min(cands, key=lambda c: np.hypot(c[0] - target[0], c[1] - target[1]))
            p[lat[0]], p[lat[1]], p[axis] = a, b, z
        else:
            z = depth(p[lat[0]], p[lat[1]])
            if z is not None:
                p[axis] = z
        out.append(p)
    return out


def _outer_surface(tris: np.ndarray, axis: int, lat: List[int], sign: float, pts: np.ndarray,
                   chunk: int = 4096) -> np.ndarray:
    """Outermost hull coordinate along `axis` (in the `sign` direction) under each
    lateral point `pts` (G, 2) -- the first hit of a ray cast from outside along
    -sign * axis -- or -inf/+inf where the mesh is missed (never finite)."""
    best = np.full(len(pts), -np.inf)
    px, py = pts[:, 0:1], pts[:, 1:2]
    for s in range(0, len(tris), chunk):
        t = tris[s:s + chunk]
        a, b, c = t[:, 0, lat], t[:, 1, lat], t[:, 2, lat]
        v0, v1 = b - a, c - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        ok = np.abs(den) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, den, 1.0), 0.0)
        w0, w1 = px - a[:, 0], py - a[:, 1]
        u = (w0 * v1[:, 1] - v1[:, 0] * w1) * inv
        v = (v0[:, 0] * w1 - w0 * v0[:, 1]) * inv
        inside = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        za = t[:, 0, axis]
        z = za + u * (t[:, 1, axis] - za) + v * (t[:, 2, axis] - za)
        best = np.maximum(best, np.where(inside, sign * z, -np.inf).max(axis=1))
    return sign * best


def _ray_mesh_first_hit(origin: np.ndarray, direction: np.ndarray, tris: np.ndarray) -> Optional[float]:
    """Distance along a unit ray to the nearest triangle it hits (Moller-Trumbore), or None."""
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    h = np.cross(direction, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-12
    f = np.zeros_like(a)
    f[ok] = 1.0 / a[ok]
    s = origin - v0
    u = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = f * (q @ direction)
    t = f * np.einsum("ij,ij->i", e2, q)
    hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 0.0)
    return float(t[hit].min()) if hit.any() else None


def _frame_towards(origin: np.ndarray, z_dir: np.ndarray) -> Frame:
    """Frame at `origin` with +Z along `z_dir` (the cone opens along its own +Z)."""
    from .frames import frame_from_axes

    z = np.asarray(z_dir, dtype=float)
    hint = [1.0, 0.0, 0.0] if abs(z[0] / max(1e-9, np.linalg.norm(z))) < 0.9 else [0.0, 0.0, 1.0]
    return frame_from_axes(origin, z, hint)


def _cone_mesh(stage, path: str, radius: float, length: float, Gf, UsdGeom, Vt, segments: int = 24):
    """Open cone: apex at the nozzle (origin), base circle of `radius` at +Z * `length`.

    Visual only -- no collision API and no physics API is applied anywhere on it.
    """
    ang = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    pts = [Gf.Vec3f(0.0, 0.0, 0.0)]
    pts += [Gf.Vec3f(float(radius * np.cos(a)), float(radius * np.sin(a)), float(length)) for a in ang]
    counts, idx = [], []
    for i in range(segments):
        j = (i + 1) % segments
        counts.append(3)
        # (apex, ring j, ring i): this winding puts the face normals on the OUTSIDE.
        # The other order points them inward, and the cone then renders with a dark
        # interior wedge wherever it is seen against a bright background (measured).
        idx += [0, 1 + j, 1 + i]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(-radius, -radius, 0.0), Gf.Vec3f(radius, radius, float(length))])
    mesh.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
    return mesh


def _emissive_material(stage, path: str, colour: np.ndarray, opacity: float, intensity: float, Gf, Sdf, UsdShade):
    """`colour` is the physical (0-1) tint; `emissiveColor` is scaled by `intensity`
    beyond that range so the hot inner shells bloom under the RTX renderer instead of
    reading as flat opaque paint."""
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    c = Gf.Vec3f(*[float(x) for x in colour])
    glow = Gf.Vec3f(*[float(x) * intensity for x in colour])
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(c)
    sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(glow)
    sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path.rsplit("/", 1)[0])).Bind(mat)
    return mat


############################
### Arm deploy (joints)  ###
############################


def wrap_joint_target(q_from: "torch.Tensor", q_to: "torch.Tensor") -> "torch.Tensor":
    """Shortest-turn equivalent of `q_to`, seen from `q_from`.

    The Canadarm3 joints are continuous (`physics:lowerLimit/upperLimit` are -inf/+inf
    in `canadarm3_large.usdz`), so q and q + 2*pi*k are the same arm configuration. The
    IK that plans the deployed pose is iterative and happily returns a solution many
    turns away from the folded pose -- interpolating to *that* number would spin the
    joints for thousands of degrees. Re-expressing the target within +-180 deg of the
    folded pose gives the identical link poses over the shortest path.
    """
    import torch

    two_pi = 2.0 * math.pi
    delta = q_to - q_from
    return q_from + torch.remainder(delta + math.pi, two_pi) - math.pi


def joint_lerp(q_start: "torch.Tensor", q_goal: "torch.Tensor", alpha: float) -> "torch.Tensor":
    """q(t) = q_start + alpha * (q_goal - q_start), alpha clamped to [0, 1]."""
    a = float(min(1.0, max(0.0, alpha)))
    return q_start + a * (q_goal - q_start)


def smoothstep(u: float) -> float:
    """C1 ease-in/ease-out on [0, 1]: the deployment starts and ends at zero rate."""
    u = float(min(1.0, max(0.0, u)))
    return u * u * (3.0 - 2.0 * u)


def joint_span_deg(q_a_deg: Sequence[float], q_b_deg: Sequence[float]) -> float:
    """Largest single-joint travel between two configurations, both in degrees."""
    a = np.asarray(q_a_deg, dtype=float)
    b = np.asarray(q_b_deg, dtype=float)
    return float(np.abs(b - a).max()) if a.size else 0.0
