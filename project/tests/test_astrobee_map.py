"""Offline tests for the Astrobee satellite map / docking clearance (`astrobee_map.py`).

No Isaac Sim needed:

    cd ~/space_robotics_bench/project && ~/isaac-sim/python.sh -m pytest tests/test_astrobee_map.py -q
"""

import importlib
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Same stand-in package as `test_astrobee_observer.py` (the real `__init__` imports Isaac Lab)
_PKG_DIR = Path(__file__).resolve().parents[1].joinpath("srb", "tasks", "manipulation", "debris_capture")
_PKG = "_debris_capture_offline"
if _PKG not in sys.modules:
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(_PKG_DIR)]
    sys.modules[_PKG] = pkg
am = importlib.import_module(f"{_PKG}.astrobee_map")
ab = importlib.import_module(f"{_PKG}.astrobee")
load_vision_config = importlib.import_module(f"{_PKG}.vision").load_vision_config
Frame = importlib.import_module(f"{_PKG}.frames").Frame

W, H = 160, 120
K = am.intrinsics(W, H, focal_length_mm=14.96, horizontal_aperture_mm=20.955)


def _nozzle(exit_radius=0.54, throat_radius=0.40, length=1.2):
    """Straight-tapered nozzle along satellite +Y, exit at (0, 2, 0)."""
    depths = np.linspace(0.0, length, 13)
    profile = [(float(d), float(exit_radius + (throat_radius - exit_radius) * d / length)) for d in depths]
    return types.SimpleNamespace(exit_centre=np.array([0.0, 2.0, 0.0]), direction=np.array([0.0, 1.0, 0.0]),
                                 exit_radius=exit_radius, profile=profile)


def _corridor(**kw):
    args = dict(end_depth=0.85, start_depth=0.10, margin=0.08)
    args.update(kw)
    return am.DockingCorridor.from_nozzle(_nozzle(), **args)


def _wall_points(nozzle, n_depth=40, n_phi=72, max_depth=1.2):
    """Points on the nozzle's inner wall."""
    c = am.DockingCorridor.from_nozzle(nozzle, max_depth, 0.0, 0.0)
    out = []
    for d in np.linspace(0.0, max_depth, n_depth):
        r = float(c.free_radius(d))
        for phi in np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False):
            out.append(nozzle.exit_centre + d * nozzle.direction + r * np.array([math.cos(phi), 0.0, math.sin(phi)]))
    return np.array(out)


def _cube(centre, size, step=0.02):
    g = np.arange(-0.5 * size, 0.5 * size + 1e-9, step)
    x, y, z = np.meshgrid(g, g, g)
    return np.column_stack((x.ravel(), y.ravel(), z.ravel())) + np.asarray(centre)


def _map(*clouds, frames=2):
    m = am.SatelliteMap(0.05, max_score=4, min_hits=2)
    for _ in range(frames):
        m.integrate(np.vstack(clouds))
    return m


## Voxel accumulation


def test_voxels_deduplicate_and_accumulate_score():
    m = am.SatelliteMap(0.05, max_score=3, min_hits=2)
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.03, 0.04], [0.26, 0.0, 0.0], [-0.01, 0.0, 0.0]])
    assert m.integrate(pts) == 3  # first two share a voxel; -0.01 floors to index -1
    assert len(m) == 3 and m.frames == 1
    assert len(m.confirmed()) == 0  # one frame is not confirmation
    assert m.integrate(pts[:1]) == 0
    assert len(m.confirmed()) == 1
    for _ in range(5):
        m.integrate(pts)
    assert m.score.max() == 3  # capped
    centres = m.centers()
    assert np.allclose(np.sort(centres[:, 0]), [-0.025, 0.025, 0.275])


def test_voxel_keys_round_trip_negative_and_far_coordinates():
    m = am.SatelliteMap(0.05)
    pts = np.array([[-123.456, 78.9, -0.001], [4000.0, -4000.0, 12.34]])
    m.integrate(pts)
    c = m.centers()
    assert np.all(np.abs(np.sort(c, axis=0) - np.sort(pts, axis=0)) <= 0.025 + 1e-9)


## Projection


def test_backproject_project_round_trip():
    rng = np.random.default_rng(0)
    depth = rng.uniform(5.0, 40.0, size=(H, W))
    u, v = am.pixel_grid(W, H, 7)
    pc = am.backproject(depth, K, u, v, 100.0)
    assert len(pc) == len(u)
    pu, pv, z = am.project(pc, K)
    assert np.allclose(pu, u + 0.5) and np.allclose(pv, v + 0.5) and np.allclose(z, depth[v, u])
    # camera convention: +X forward, +Y left, +Z up -> a pixel right/below the centre is -Y/-Z
    p = am.backproject(np.full((H, W), 10.0), K, np.array([W - 1]), np.array([H - 1]), 100.0)[0]
    assert p[0] == pytest.approx(10.0) and p[1] < 0.0 and p[2] < 0.0


def test_backproject_drops_missing_and_far_depth():
    depth = np.full((H, W), 10.0)
    depth[0, 0], depth[0, 1], depth[0, 2] = np.inf, np.nan, 500.0
    pc = am.backproject(depth, K, np.array([0, 1, 2, 3]), np.array([0, 0, 0, 0]), 200.0)
    assert len(pc) == 1


def test_pixel_grid_is_sparse_but_dense_in_the_roi():
    u, v = am.pixel_grid(640, 480, 12)
    assert len(u) <= 5000
    u2, v2 = am.pixel_grid(640, 480, 12, roi=(300, 320, 200, 220), roi_stride=1)
    in_roi = (u2 >= 300) & (u2 < 320) & (v2 >= 200) & (v2 < 220)
    assert in_roi.sum() == 400
    assert len(np.unique(v2 * 640 + u2)) == len(u2)


def test_sphere_roi_contains_the_projected_sphere():
    centre = np.array([20.0, 1.0, -0.5])
    roi = am.sphere_roi(centre, 0.6, K, W, H)
    assert roi is not None
    rng = np.random.default_rng(1)
    d = rng.normal(size=(500, 3))
    pts = centre + 0.6 * d / np.linalg.norm(d, axis=1, keepdims=True)
    u, v, _ = am.project(pts, K)
    assert np.all((u >= roi[0]) & (u < roi[1]) & (v >= roi[2]) & (v < roi[3]))
    assert am.sphere_roi(np.array([-5.0, 0.0, 0.0]), 0.6, K, W, H) is None  # behind


## Corridor / clearance


def test_corridor_excludes_the_nozzle_wall_and_the_exit_ring():
    nz = _nozzle()
    c = _corridor()
    wall = _wall_points(nz)
    assert not c.contains(wall).any()
    # map of the wall alone (voxel-quantised): still clear
    clear = am.evaluate_clearance(_map(wall), c)
    assert clear.is_clear and clear.obstruction_voxels == 0 and math.isinf(clear.nearest_m)
    # the dock ring at the exit plane (0.75..1.0 x exit radius, depth 0) is not an obstruction
    ring = [nz.exit_centre + r * np.array([math.cos(a), 0.0, math.sin(a)])
            for r in np.linspace(0.75 * 0.54, 0.54, 5) for a in np.linspace(0, 2 * math.pi, 60)]
    assert am.evaluate_clearance(_map(wall, np.array(ring)), c).is_clear


def test_corridor_radius_follows_the_nozzle_profile():
    c = _corridor(margin=0.05)
    assert c.free_radius(0.0) == pytest.approx(0.54 - 0.05)
    assert c.free_radius(1.2) == pytest.approx(0.40 - 0.05)
    axis = c.exit_centre + np.outer([0.05, 0.11, 0.5, 0.84, 0.95], c.direction)
    assert c.contains(axis).tolist() == [False, True, True, True, False]


def test_object_in_the_corridor_is_an_obstruction():
    nz = _nozzle()
    c = _corridor()
    debris = _cube(nz.exit_centre + 0.4 * nz.direction + np.array([0.1, 0.0, 0.0]), 0.2)
    r = am.evaluate_clearance(_map(_wall_points(nz), debris), c)
    assert not r.is_clear and r.obstruction_voxels >= 8
    # nearest voxel centre: the cube's front face (depth 0.3) is at most one voxel off
    assert 0.25 <= r.nearest_m <= 0.40
    assert np.all(c.contains(r.obstruction_points))


def test_object_outside_the_corridor_is_not_an_obstruction():
    nz = _nozzle()
    c = _corridor()
    in_front = _cube(nz.exit_centre - 0.5 * nz.direction, 0.3)  # outside the nozzle exit
    past_plate = _cube(nz.exit_centre + 1.1 * nz.direction, 0.1)  # beyond the back plate
    assert am.evaluate_clearance(_map(in_front, past_plate), c).is_clear


def test_unconfirmed_voxels_do_not_block():
    nz = _nozzle()
    debris = _cube(nz.exit_centre + 0.4 * nz.direction, 0.15)
    assert am.evaluate_clearance(_map(debris, frames=1), _corridor()).is_clear
    assert not am.evaluate_clearance(_map(debris, frames=2), _corridor()).is_clear
    assert am.evaluate_clearance(_map(debris, frames=2), _corridor(), min_voxels=10_000).is_clear


def test_capsule_filter_removes_the_probe():
    nz = _nozzle()
    tip = nz.exit_centre + 0.5 * nz.direction
    back = tip - 3.0 * nz.direction
    rod = np.array([tip + t * (back - tip) + 0.07 * np.array([math.cos(a), 0.0, math.sin(a)])
                    for t in np.linspace(0, 1, 60) for a in np.linspace(0, 2 * math.pi, 16)])
    keep = ~am.capsule_mask(rod, tip, back, 0.08 + 0.1)
    assert not keep.any()
    assert am.capsule_mask(np.array([tip + np.array([0.4, 0.0, 0.0])]), tip, back, 0.18).tolist() == [False]


## Full depth-frame pipeline (synthetic renderer)


def _render(cam_c_from_w: Frame, planes, w=W, h=H):
    """Depth (distance to image plane) of axis-aligned squares [(centre, normal_axis, half)]."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    rays_c = np.stack((np.ones_like(u), -(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1]), axis=-1)
    cam_w = cam_c_from_w
    rays_w = rays_c @ cam_w.rot.T
    depth = np.full((h, w), np.inf)
    for centre, ax, half in planes:
        centre = np.asarray(centre, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (centre[ax] - cam_w.pos[ax]) / rays_w[..., ax]
        hit = cam_w.pos + s[..., None] * rays_w
        others = [i for i in range(3) if i != ax]
        ok = (s > 0) & np.all(np.abs(hit[..., others] - centre[others]) <= half, axis=-1)
        depth = np.where(ok & (s < depth), s, depth)  # ray x-component is 1: s = image-plane depth
    return depth


def _looking_along_y(pos):
    """Camera at `pos` looking along world +Y (C: +X fwd -> W +Y, +Y left -> W -X, +Z up)."""
    return Frame(np.asarray(pos, dtype=float), np.column_stack(([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0])))


def test_depth_frame_points_land_on_the_surface_in_the_satellite_frame():
    cfg = am.AstrobeeMapCfg(grid_stride_px=4, crop_margin_m=1.0)
    cam = _looking_along_y([0.0, -10.0, 0.0])
    sat = Frame(np.array([3.0, -1.0, 2.0]), ab.look_at_rotation([0, 0, 0], [1.0, 0.3, 0.2]))
    wall_w = [np.array([0.0, 5.0, 0.0]), 1, 2.0]  # square plane y = 5 in the world
    depth = _render(cam, [wall_w])
    pts_s, cam_s, _ = am.depth_frame_points(depth, K, cam, sat, cfg)
    assert len(pts_s) > 50
    pts_w = am.transform(sat, pts_s)
    assert np.allclose(pts_w[:, 1], 5.0, atol=1e-6)
    assert np.allclose(am.transform(sat, cam_s.pos[None, :])[0], cam.pos)
    # crop to a satellite box that excludes the wall: nothing left
    far = (np.array([100.0, 100.0, 100.0]), np.array([101.0, 101.0, 101.0]))
    assert len(am.depth_frame_points(depth, K, cam, sat, cfg, crop=far)[0]) == 0


def test_carving_removes_an_object_that_went_away():
    cfg = am.AstrobeeMapCfg(grid_stride_px=2)
    cam = _looking_along_y([0.0, -10.0, 0.0])
    sat = Frame.identity()
    back = [np.array([0.0, 5.0, 0.0]), 1, 3.0]
    box = [np.array([0.0, 0.0, 0.0]), 1, 0.5]  # a square in front of the back plane
    m = am.SatelliteMap(0.05, max_score=2, min_hits=2)
    for _ in range(2):
        pts, cam_s, _ = am.depth_frame_points(_render(cam, [back, box]), K, cam, sat, cfg)
        m.carve(cam_s, _render(cam, [back, box]), K, cfg.carve_tolerance_m)
        m.integrate(pts)
    near_box = lambda: np.abs(m.confirmed()[:, 1]) < 0.1  # noqa: E731
    assert near_box().sum() > 10
    # the box is gone: two frames seeing through it remove its voxels, the back plane stays
    depth = _render(cam, [back])
    for _ in range(2):
        m.carve(cam, depth, K, cfg.carve_tolerance_m)
        m.integrate(am.depth_frame_points(depth, K, cam, sat, cfg)[0])
    assert near_box().sum() == 0
    assert (np.abs(m.confirmed()[:, 1] - 5.0) < 0.1).sum() > 10


def test_carving_keeps_occluded_voxels():
    cam = _looking_along_y([0.0, -10.0, 0.0])
    m = _map(_cube([0.0, 3.0, 0.0], 0.2))
    n = len(m)
    occluder = _render(cam, [[np.array([0.0, 0.0, 0.0]), 1, 2.0]])  # everything hidden behind y = 0
    assert m.carve(cam, occluder, K, 0.15) == 0 and len(m) == n


## Config / PLY


def test_map_config_defaults_and_validation():
    cfg = load_vision_config()
    mc = cfg.astrobee.map
    assert mc.enabled and mc.voxel_size_m <= 0.05 and not mc.test_obstruction
    assert load_vision_config(overrides=["astrobee.map={'test_obstruction': true}"]).astrobee.map.test_obstruction
    assert mc.min_corridor_views >= mc.min_hits
    for bad in ("{'voxel_size_m': 0.0}", "{'min_hits': 9}", "{'grid_stride_px': 0}", "{'min_obstruction_voxels': 0}",
                "{'min_corridor_views': 1}"):
        with pytest.raises(ValueError):
            load_vision_config(overrides=[f"astrobee.map={bad}"])


def test_ply_round_trip(tmp_path):
    pts = np.random.default_rng(2).normal(size=(50, 3))
    cols = np.tile([255, 0, 0], (50, 1))
    p = am.write_ply(tmp_path / "a.ply", pts, cols)
    assert np.allclose(am.read_ply_xyz(p), pts, atol=1e-6)
    assert np.allclose(am.read_ply_xyz(am.write_ply(tmp_path / "b.ply", pts)), pts, atol=1e-6)
    assert p.read_bytes().startswith(b"ply\nformat binary_little_endian 1.0\nelement vertex 50\n")


def test_grid_offsets_cover_the_whole_stride_cell():
    s = 8
    seen = {am.grid_offset(k, s) for k in range(s * s)}
    assert len(seen) == s * s
    a = am.pixel_grid(64, 48, s, offset=am.grid_offset(0, s))
    b = am.pixel_grid(64, 48, s, offset=am.grid_offset(1, s))
    assert len(a[0]) == len(b[0]) == 8 * 6
    assert not set(zip(*a)) & set(zip(*b))


def test_unobserved_corridor_is_never_clear():
    empty = am.SatelliteMap(0.05)
    r = am.evaluate_clearance(empty, _corridor(), corridor_views=0, min_views=1)
    assert not r.is_clear and not r.observed and r.obstruction_voxels == 0
    assert "not observed" in r.summary()
    r = am.evaluate_clearance(empty, _corridor(), corridor_views=1, min_views=1)
    assert r.is_clear and r.observed


def test_corridor_view_counts_returns_near_the_nozzle():
    nz = _nozzle()
    c = _corridor()
    cfg = am.AstrobeeMapCfg(grid_stride_px=8)
    cam = _looking_along_y([0.0, -10.0, 0.0])
    sat = Frame.identity()
    # a plane right behind the corridor (inside its bounding sphere) vs. far behind it
    centre, radius = c.bounding_sphere()
    near_plane = [centre, 1, 1.0]
    far_plane = [centre + np.array([0.0, 5.0, 0.0]), 1, 1.0]
    assert am.depth_frame_points(_render(cam, [near_plane]), K, cam, sat, cfg, corridor=c)[2] >= 5
    assert am.depth_frame_points(_render(cam, [far_plane]), K, cam, sat, cfg, corridor=c)[2] == 0


def test_approach_corridor_in_front_of_the_exit():
    nz = _nozzle()
    c = _corridor(approach_length=0.5, approach_radius=0.2)
    front = nz.exit_centre - np.outer([0.05, 0.3, 0.49], nz.direction)
    assert c.contains(front).all()
    assert not c.contains(nz.exit_centre - 0.6 * nz.direction[None, :]).any()  # beyond the approach
    assert not c.contains(nz.exit_centre - 0.3 * nz.direction + np.array([0.25, 0.0, 0.0])).any()  # off to the side
    # the rim ring (0.75..1.0 x exit radius at the exit plane) is still no obstruction
    ring = np.array([nz.exit_centre + r * np.array([math.cos(a), 0.0, math.sin(a)])
                     for r in np.linspace(0.75 * 0.54, 0.54, 5) for a in np.linspace(0, 2 * math.pi, 60)])
    assert am.evaluate_clearance(_map(ring), c).is_clear
    # floating debris in front of the exit blocks; the bounding sphere (dense sampling) covers it
    r = am.evaluate_clearance(_map(_cube(nz.exit_centre - 0.3 * nz.direction, 0.15)), c)
    assert not r.is_clear and r.nearest_m < 0.45
    centre, radius = c.bounding_sphere()
    assert np.linalg.norm(nz.exit_centre - 0.5 * nz.direction - centre) <= radius


def _plate_mesh(centre, normal_axis, half, n=4):
    """Square plate (two triangles per cell) as (points, triangles)."""
    g = np.linspace(-half, half, n + 1)
    u, v = np.meshgrid(g, g)
    pts = np.zeros((u.size, 3))
    others = [i for i in range(3) if i != normal_axis]
    pts[:, others[0]], pts[:, others[1]] = u.ravel(), v.ravel()
    pts += np.asarray(centre, dtype=float)
    tris = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, i * (n + 1) + j + 1, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1
            tris += [(a, b, d), (a, d, c)]
    return pts, np.array(tris)


def test_sample_triangles_covers_the_surface():
    pts, tris = _plate_mesh([0.0, 0.0, 0.0], 2, 0.5, n=1)
    s = am.sample_triangles(pts, tris, 0.02)
    assert np.allclose(s[:, 2], 0.0)
    rng = np.random.default_rng(4)
    probe = np.column_stack((rng.uniform(-0.49, 0.49, (200, 2)), np.zeros(200)))
    d = np.min(np.linalg.norm(probe[:, None] - s[None], axis=2), axis=1)
    assert d.max() <= 0.02
    # only triangles near the sphere
    far = am.sample_triangles(pts, tris, 0.02, centre=[10.0, 0.0, 0.0], radius=1.0)
    assert len(far) == 0


def test_model_mask_explains_the_satellite_but_not_debris():
    wall_pts, wall_tris = _plate_mesh([0.0, 2.0, 0.0], 1, 1.0, n=8)
    model = am.ModelMask(am.sample_triangles(wall_pts, wall_tris, 0.025), 0.05, dilate=1)
    rng = np.random.default_rng(5)
    on_wall = np.column_stack((rng.uniform(-0.9, 0.9, 100), np.full(100, 2.0) + rng.normal(0, 0.01, 100), rng.uniform(-0.9, 0.9, 100)))
    assert model.explains(on_wall).all()
    debris = _cube([0.0, 1.6, 0.0], 0.1)
    assert not model.explains(debris).any()


def test_clearance_subtracts_the_model_inside_a_wide_keep_out_zone():
    nz = _nozzle()
    # zone wider than the nozzle exit: the rim / structure around it is in the zone
    c = _corridor(start_depth=0.0, margin=0.02, approach_length=1.0, approach_radius=0.54 + 0.25)
    rim_pts, rim_tris = _plate_mesh(nz.exit_centre, 1, 0.7, n=10)  # a flange plate at the exit plane
    rim_samples = am.sample_triangles(rim_pts, rim_tris, 0.025)
    rim_samples = rim_samples[np.linalg.norm((rim_samples - nz.exit_centre)[:, [0, 2]], axis=1) > 0.54]  # hole = the nozzle
    wall = _wall_points(nz)
    model = am.ModelMask(np.vstack((rim_samples, wall)), 0.05, dilate=1)
    scan = np.vstack((rim_samples, wall))
    assert not am.evaluate_clearance(_map(scan), c).is_clear  # without the model the flange "blocks"
    assert am.evaluate_clearance(_map(scan), c, model=model).is_clear
    debris = _cube(nz.exit_centre - 0.6 * nz.direction + np.array([0.3, 0.0, 0.0]), 0.15)
    r = am.evaluate_clearance(_map(scan, debris), c, model=model)
    assert not r.is_clear and r.obstruction_voxels >= 4
    # beyond the zone (1.2 m in front): not an obstacle
    assert am.evaluate_clearance(_map(scan, _cube(nz.exit_centre - 1.3 * nz.direction, 0.15)), c, model=model).is_clear


def test_known_moving_model_is_never_mapped():
    cfg = am.AstrobeeMapCfg(grid_stride_px=2)
    cam = _looking_along_y([0.0, -10.0, 0.0])
    plate = [np.array([0.0, 0.0, 0.0]), 1, 0.5]
    depth = _render(cam, [plate])
    pts, tris = _plate_mesh([0.0, 0.0, 0.0], 1, 0.5, n=2)
    pose = Frame(np.array([0.0, 0.0, 0.0]), np.eye(3))
    mep = am.ModelMask(am.sample_triangles(pts, tris, 0.025), 0.05)
    assert len(am.depth_frame_points(depth, K, cam, Frame.identity(), cfg)[0]) > 20
    assert len(am.depth_frame_points(depth, K, cam, Frame.identity(), cfg, exclude_models=[(pose, mep)])[0]) == 0


def test_far_surfaces_are_not_mapped():
    cfg = am.AstrobeeMapCfg(grid_stride_px=4, max_range_m=30.0)
    cam = _looking_along_y([0.0, -10.0, 0.0])
    near = _render(cam, [[np.array([0.0, 10.0, 0.0]), 1, 3.0]])   # 20 m away
    far = _render(cam, [[np.array([0.0, 40.0, 0.0]), 1, 10.0]])   # 50 m away
    assert len(am.depth_frame_points(near, K, cam, Frame.identity(), cfg)[0]) > 0
    assert len(am.depth_frame_points(far, K, cam, Frame.identity(), cfg)[0]) == 0
