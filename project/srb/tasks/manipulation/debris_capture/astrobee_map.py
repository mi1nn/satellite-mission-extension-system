"""Astrobee satellite-shape map and docking-corridor clearance (pure numpy).

The Astrobee (`astrobee.py`) samples its camera depth image a few times per inspection
point dwell. Each sample is back-projected on a pixel grid (plus a dense window around
the docking port), moved into the *satellite root frame* (so the map stays put while the
satellite drifts) and accumulated in a voxel grid (`SatelliteMap`). Voxels the camera now
sees *through* lose score and disappear (free-space carving), so the map follows a scene
that changes.

The docking decision (`evaluate_clearance`): the free insertion corridor of the thruster
nozzle -- from just inside the exit to the back plate, radius = nozzle inner radius at that
depth minus a margin (`DockingCorridor`) -- must hold no confirmed voxel. The nozzle wall
lies outside that radius, so only foreign objects count as obstructions.

Frames: world W, satellite root S (`sat_frame()` of the demo, the frame the docking
geometry is expressed in), camera C in the Isaac Lab "world" convention (+X forward,
+Y left, +Z up). Image pixels: u right, v down. SI units.

No Isaac Sim / Isaac Lab imports (`project/tests/test_astrobee_map.py`).
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .frames import Frame

##############
### Config ###
##############


@dataclass
class AstrobeeMapCfg:
    """`astrobee.map:` section of `vision_capture.yaml`."""

    enabled: bool = True
    voxel_size_m: float = 0.03
    # Depth samples per inspection-point dwell (spread evenly over the dwell); after the
    # last loop (ASTROBEE_OBSERVATION_COMPLETE) one sample every `dwell_s`
    samples_per_dwell: int = 3
    # ... and one every this many seconds while flying between inspection points (not
    # on the approach): the map fills in as the Astrobee scans (0: dwell samples only)
    scan_period_s: float = 1.0
    # Back-projection: every `grid_stride_px` pixel over the whole image (the grid shifts
    # from sample to sample, so repeated samples fill in other pixels), every
    # `roi_stride_px` pixel in the (fixed) window around the docking corridor
    grid_stride_px: int = 4
    roi_stride_px: int = 1
    # Scan range: only returns closer than this are mapped, so the map grows as the
    # Astrobee flies close past the satellite (nothing from far away)
    max_range_m: float = 20.0
    # Only points inside the satellite's box (satellite frame) + this margin are mapped
    crop_margin_m: float = 2.0
    # Voxel score: +1 per sample that hits it (capped), -1 per sample that sees through it
    max_score: int = 4
    # Score from which a voxel counts (map export / obstruction)
    min_hits: int = 2
    # A voxel is seen through when the nearest depth around its pixel is this much farther
    carve_tolerance_m: float = 0.15
    ## Docking clearance: an obstruction is a confirmed voxel inside the docking keep-out
    ## zone that the satellite's own 3D model (its USD meshes) does not explain and that is
    ## not the servicer (probe / MEP model)
    # Keep-out zone, inside the nozzle: radius = inner radius - margin, from
    # `corridor_start_m` inside the exit to `corridor_end_margin_m` before the back plate
    corridor_margin_m: float = 0.02
    corridor_start_m: float = 0.0
    corridor_end_margin_m: float = 0.05
    # ... and in front of the exit: a cylinder `keepout_length_m` long (beyond the
    # pre-dock pose of the probe tip), radius = exit radius + `keepout_radius_extra_m`
    keepout_length_m: float = 1.0
    keepout_radius_extra_m: float = 0.25
    # Model subtraction: surface samples every `model_sample_spacing_m` of the satellite /
    # MEP meshes; a voxel within `model_dilate_voxels` of a sample is explained by the model
    model_sample_spacing_m: float = 0.015
    model_dilate_voxels: int = 1
    # An observed corridor with confirmed obstruction stops the mission at once
    # (DOCKING_UNAVAILABLE); an unobserved corridor waits for the insertion gate
    stop_mission_on_obstruction: bool = True
    # Obstruction voxels needed to declare DOCKING_UNAVAILABLE
    min_obstruction_voxels: int = 1
    # The corridor counts as clear only once this many depth frames have seen into it
    # (>= `min_view_points` returns inside its bounding sphere); before that it is
    # unknown. At least `min_hits`, or an obstruction could not be confirmed yet
    min_corridor_views: int = 2
    min_view_points: int = 5
    # Points on the servicer's own probe are not mapped (probe radius + this margin)
    probe_filter_margin_m: float = 0.10
    # Gate the mission's DOCK_READY on the clearance: not clear -> DOCKING_UNAVAILABLE,
    # the final verdict of the run (false: judge and log only)
    gate_docking: bool = True
    ## Test obstruction for the demo: floating debris (visual only, no collider) at the
    ## docking port, bobbing and tumbling slowly in the probe's approach corridor
    test_obstruction: bool = False
    test_obstruction_depth_m: float = -0.3  # centre along the nozzle axis (< 0: in front of the exit)
    test_obstruction_size_m: float = 0.2  # longest edge of the debris plate
    test_obstruction_lateral_m: float = 0.0  # offset from the nozzle axis
    test_obstruction_float_amplitude_m: float = 0.04  # drift about its centre
    test_obstruction_float_period_s: float = 20.0
    test_obstruction_tumble_deg_s: float = 6.0
    ## Outputs (Phase 5/6): ROS topics under `/<astrobee.ros_namespace>/`, run log
    clearance_topic: str = "map/clearance"
    points_topic: str = "map/points"
    points_publish_period_s: float = 2.0
    save_ply: bool = True


def validate_map_cfg(cfg: AstrobeeMapCfg):
    if cfg.voxel_size_m <= 0.0:
        raise ValueError("astrobee.map.voxel_size_m must be > 0")
    if cfg.scan_period_s < 0.0:
        raise ValueError("astrobee.map.scan_period_s must be >= 0 (0: off)")
    if int(cfg.samples_per_dwell) < 1:
        raise ValueError("astrobee.map.samples_per_dwell must be >= 1")
    if int(cfg.grid_stride_px) < 1 or int(cfg.roi_stride_px) < 1:
        raise ValueError("astrobee.map.grid_stride_px / roi_stride_px must be >= 1")
    if cfg.max_range_m <= 0.0 or cfg.crop_margin_m < 0.0 or cfg.carve_tolerance_m <= 0.0:
        raise ValueError("astrobee.map.max_range_m / carve_tolerance_m must be > 0, crop_margin_m >= 0")
    if not 1 <= int(cfg.min_hits) <= int(cfg.max_score):
        raise ValueError("astrobee.map requires 1 <= min_hits <= max_score")
    if cfg.corridor_margin_m < 0.0 or cfg.corridor_start_m < 0.0 or cfg.corridor_end_margin_m < 0.0:
        raise ValueError("astrobee.map.corridor_* must be >= 0")
    if int(cfg.min_corridor_views) < 0 or int(cfg.min_view_points) < 1:
        raise ValueError("astrobee.map.min_corridor_views must be >= 0 and min_view_points >= 1")
    if 0 < int(cfg.min_corridor_views) < int(cfg.min_hits):
        raise ValueError("astrobee.map.min_corridor_views must be >= min_hits (a clear verdict needs as many "
                         "views as confirming an obstruction does), or 0 to disable")
    if int(cfg.min_obstruction_voxels) < 1:
        raise ValueError("astrobee.map.min_obstruction_voxels must be >= 1")
    if cfg.keepout_length_m < 0.0 or cfg.keepout_radius_extra_m < 0.0:
        raise ValueError("astrobee.map.keepout_length_m / keepout_radius_extra_m must be >= 0")
    if cfg.model_sample_spacing_m <= 0.0 or int(cfg.model_dilate_voxels) < 0:
        raise ValueError("astrobee.map.model_sample_spacing_m must be > 0 and model_dilate_voxels >= 0")
    if cfg.test_obstruction_float_amplitude_m < 0.0 or cfg.test_obstruction_float_period_s <= 0.0:
        raise ValueError("astrobee.map.test_obstruction_float_amplitude_m must be >= 0, float_period_s > 0")
    if cfg.test_obstruction_size_m <= 0.0 or cfg.points_publish_period_s <= 0.0:
        raise ValueError("astrobee.map.test_obstruction_size_m / points_publish_period_s must be > 0")
    if not cfg.clearance_topic.strip("/") or not cfg.points_topic.strip("/"):
        raise ValueError("astrobee.map.clearance_topic / points_topic must not be empty")
    cfg.clearance_topic = cfg.clearance_topic.strip("/")
    cfg.points_topic = cfg.points_topic.strip("/")


##################
### Projection ###
##################


def intrinsics(width: int, height: int, focal_length_mm: float, horizontal_aperture_mm: float) -> np.ndarray:
    """Pinhole K of a USD camera with square pixels, principal point at the image centre."""
    f = focal_length_mm / horizontal_aperture_mm * width
    return np.array([[f, 0.0, 0.5 * width], [0.0, f, 0.5 * height], [0.0, 0.0, 1.0]])


def grid_offset(frame: int, stride: int) -> Tuple[int, int]:
    """Grid shift of the `frame`-th sample: cycles through the stride x stride cell."""
    s = int(stride)
    return (5 * frame) % s, (3 * frame + (frame // s)) % s


def pixel_grid(width: int, height: int, stride: int, roi: Optional[Tuple[int, int, int, int]] = None,
               roi_stride: int = 1, offset: Tuple[int, int] = (0, 0)) -> Tuple[np.ndarray, np.ndarray]:
    """Pixel (u, v) of a regular grid over the image (shifted by `offset` px) plus a
    denser one over `roi` (u0, u1, v0, v1, exclusive ends); duplicates removed."""
    s = int(stride)
    u, v = np.meshgrid(np.arange((s // 2 + offset[0]) % s, width, s), np.arange((s // 2 + offset[1]) % s, height, s))
    u, v = u.ravel(), v.ravel()
    if roi is not None:
        u0, u1, v0, v1 = roi
        ru, rv = np.meshgrid(np.arange(u0, u1, int(roi_stride)), np.arange(v0, v1, int(roi_stride)))
        u, v = np.concatenate((u, ru.ravel())), np.concatenate((v, rv.ravel()))
        flat = np.unique(v.astype(np.int64) * width + u)
        u, v = flat % width, flat // width
    return u.astype(np.int64), v.astype(np.int64)


def backproject(depth: np.ndarray, k: np.ndarray, u: np.ndarray, v: np.ndarray, max_range_m: float) -> np.ndarray:
    """Camera-frame points (C, "world" convention) of pixels (u, v) of a
    distance-to-image-plane image; pixels with no / out-of-range depth are dropped."""
    d = np.asarray(depth, dtype=float)[v, u]
    ok = np.isfinite(d) & (d > 0.0) & (d <= max_range_m)
    d, uu, vv = d[ok], u[ok] + 0.5, v[ok] + 0.5
    right = (uu - k[0, 2]) / k[0, 0] * d
    down = (vv - k[1, 2]) / k[1, 1] * d
    return np.column_stack((d, -right, -down))


def project(points_c: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v) continuous pixel coordinates and image-plane depth of camera-frame points."""
    p = np.asarray(points_c, dtype=float).reshape(-1, 3)
    z = p[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 0] * (-p[:, 1]) / z + k[0, 2]
        v = k[1, 1] * (-p[:, 2]) / z + k[1, 2]
    return u, v, z


def sphere_roi(centre_c, radius: float, k: np.ndarray, width: int, height: int, pad_px: int = 2) -> Optional[Tuple[int, int, int, int]]:
    """Pixel window (u0, u1, v0, v1) covering a sphere given in the camera frame, or None
    when it is behind the camera or outside the image."""
    c = np.asarray(centre_c, dtype=float)
    if c[0] <= radius:
        return None
    (u,), (v,), (z,) = project(c[None, :], k)
    r_px = k[0, 0] * radius / math.sqrt(max(z * z - radius * radius, 1e-12)) + pad_px
    u0, u1 = int(math.floor(u - r_px)), int(math.ceil(u + r_px)) + 1
    v0, v1 = int(math.floor(v - r_px)), int(math.ceil(v + r_px)) + 1
    u0, u1, v0, v1 = max(u0, 0), min(u1, width), max(v0, 0), min(v1, height)
    if u0 >= u1 or v0 >= v1:
        return None
    return u0, u1, v0, v1


def min_filter3(img: np.ndarray) -> np.ndarray:
    """3x3 minimum (edges replicated); NaN counts as +inf (no return)."""
    a = np.where(np.isfinite(img), img, np.inf)
    p = np.pad(a, 1, mode="edge")
    h, w = a.shape
    out = a.copy()
    for dv in range(3):
        for du in range(3):
            np.minimum(out, p[dv:dv + h, du:du + w], out=out)
    return out


def transform(frame: Frame, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float) @ frame.rot.T + frame.pos


def capsule_mask(points: np.ndarray, a, b, radius: float) -> np.ndarray:
    """Points within `radius` of the segment a-b."""
    p = np.asarray(points, dtype=float)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1) <= radius


def box_mask(points: np.ndarray, lo, hi) -> np.ndarray:
    p = np.asarray(points, dtype=float)
    return np.all((p >= np.asarray(lo)) & (p <= np.asarray(hi)), axis=1)


###########
### Map ###
###########

_BIAS = 1 << 20  # voxel index offset per axis (21 bits: +-52 km at 5 cm)


def _pack(idx: np.ndarray) -> np.ndarray:
    i = idx.astype(np.int64) + _BIAS
    return (i[:, 0] << 42) | (i[:, 1] << 21) | i[:, 2]


def _unpack(keys: np.ndarray) -> np.ndarray:
    m = (1 << 21) - 1
    return np.column_stack(((keys >> 42) & m, (keys >> 21) & m, keys & m)) - _BIAS


class SatelliteMap:
    """Voxel occupancy map in the satellite frame (sorted packed keys + scores)."""

    def __init__(self, voxel_size_m: float = 0.05, max_score: int = 4, min_hits: int = 2):
        self.voxel = float(voxel_size_m)
        self.max_score = int(max_score)
        self.min_hits = int(min_hits)
        self.keys = np.zeros(0, dtype=np.int64)
        self.score = np.zeros(0, dtype=np.int16)
        self.frames = 0

    def __len__(self) -> int:
        return int(len(self.keys))

    def keys_of(self, points: np.ndarray) -> np.ndarray:
        return _pack(np.floor(np.asarray(points, dtype=float) / self.voxel).astype(np.int64))

    def integrate(self, points: np.ndarray) -> int:
        """+1 to every voxel holding at least one of `points` (one frame); returns the
        number of new voxels."""
        self.frames += 1
        if len(points) == 0:
            return 0
        new = np.unique(self.keys_of(points))
        idx = np.searchsorted(self.keys, new)
        found = idx < len(self.keys)
        found[found] = self.keys[idx[found]] == new[found]
        hit = idx[found]
        self.score[hit] = np.minimum(self.score[hit] + 1, self.max_score)
        add = new[~found]
        if len(add):
            keys = np.concatenate((self.keys, add))
            score = np.concatenate((self.score, np.ones(len(add), dtype=np.int16)))
            order = np.argsort(keys, kind="stable")
            self.keys, self.score = keys[order], score[order]
        return int(len(add))

    def carve(self, cam_in_map: Frame, depth: np.ndarray, k: np.ndarray, tolerance_m: float,
              max_range_m: float = math.inf) -> int:
        """-1 to every voxel the camera sees through: in view, in front of the camera, and
        the nearest depth around its pixel lies more than `tolerance_m` beyond it (or
        there is no return at all). Voxels at 0 are removed; returns their number."""
        if not len(self.keys):
            return 0
        h, w = depth.shape
        c = transform(cam_in_map.inv(), self.centers())
        u, v, z = project(c, k)
        inside = (z > 0.0) & (z < max_range_m) & (u >= 0.0) & (u < w) & (v >= 0.0) & (v < h)
        if not inside.any():
            return 0
        near = min_filter3(depth)
        ui, vi = u[inside].astype(np.int64), v[inside].astype(np.int64)
        seen = near[vi, ui]
        through = seen > z[inside] + tolerance_m  # includes +inf (no return)
        sel = np.flatnonzero(inside)[through]
        self.score[sel] -= 1
        gone = self.score <= 0
        n = int(gone.sum())
        if n:
            self.keys, self.score = self.keys[~gone], self.score[~gone]
        return n

    def centers(self, min_score: int = 1) -> np.ndarray:
        sel = self.score >= min_score
        return (_unpack(self.keys[sel]).astype(float) + 0.5) * self.voxel

    def confirmed(self) -> np.ndarray:
        return self.centers(self.min_hits)


##########################
### Docking clearance ###
##########################


@dataclass
class DockingCorridor:
    """Free docking corridor of the thruster nozzle (satellite frame): inside the nozzle
    (`start_depth`..`end_depth`, radius = inner radius - `margin`) and, optionally, the
    probe's approach in front of the exit (`approach_length` long, `approach_radius`).

    `profile`: (depth, inner radius) samples along `direction` from `exit_centre`
    (`docking.NozzleGeometry.profile`); depth < 0 is in front of the exit."""

    exit_centre: np.ndarray
    direction: np.ndarray
    profile: List[Tuple[float, float]]
    start_depth: float
    end_depth: float
    margin: float
    approach_length: float = 0.0
    approach_radius: float = 0.0

    @staticmethod
    def from_nozzle(nozzle, end_depth: float, start_depth: float, margin: float,
                    approach_length: float = 0.0, approach_radius: float = 0.0) -> "DockingCorridor":
        d = np.asarray(nozzle.direction, dtype=float)
        return DockingCorridor(np.asarray(nozzle.exit_centre, dtype=float), d / np.linalg.norm(d),
                               [tuple(map(float, p)) for p in nozzle.profile], float(start_depth), float(end_depth),
                               float(margin), float(approach_length), float(approach_radius))

    def free_radius(self, depth) -> np.ndarray:
        d = np.array([p[0] for p in self.profile])
        r = np.array([p[1] for p in self.profile])
        return np.maximum(np.interp(depth, d, r) - self.margin, 0.0)

    def depth_radial(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rel = np.asarray(points, dtype=float).reshape(-1, 3) - self.exit_centre
        depth = rel @ self.direction
        radial = np.linalg.norm(rel - np.outer(depth, self.direction), axis=1)
        return depth, radial

    def contains(self, points: np.ndarray) -> np.ndarray:
        depth, radial = self.depth_radial(points)
        inside = (depth >= self.start_depth) & (depth <= self.end_depth) & (radial < self.free_radius(depth))
        if self.approach_length > 0.0:
            inside |= (depth >= -self.approach_length) & (depth < 0.0) & (radial < self.approach_radius)
        return inside

    def bounding_sphere(self) -> Tuple[np.ndarray, float]:
        lo = -self.approach_length if self.approach_length > 0.0 else self.start_depth
        mid = 0.5 * (lo + self.end_depth)
        r = float(max(self.free_radius(self.start_depth), self.free_radius(self.end_depth), self.free_radius(mid),
                      self.approach_radius))
        half = 0.5 * (self.end_depth - lo)
        return self.exit_centre + mid * self.direction, math.hypot(half, r + self.margin)


@dataclass
class DockingClearance:
    """Result of the corridor check. `nearest_m`: distance from the nozzle exit centre
    to the closest obstruction voxel centre (inf when there is none). `observed`: the
    corridor has been seen often enough to judge; unobserved is never clear."""

    is_clear: bool
    obstruction_voxels: int
    nearest_m: float
    observed_frames: int = 0
    corridor_views: int = 0
    observed: bool = True
    obstruction_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)), repr=False)

    def summary(self) -> str:
        frames = f"{self.observed_frames} depth frames, {self.corridor_views} into the nozzle"
        if not self.observed:
            return f"nozzle corridor not observed yet ({frames})"
        if self.is_clear:
            return f"corridor clear ({frames})"
        return (f"{self.obstruction_voxels} obstruction voxel(s) in the nozzle corridor, nearest "
                f"{self.nearest_m:.2f} m from the exit ({frames})")


def evaluate_clearance(smap: SatelliteMap, corridor: DockingCorridor, min_voxels: int = 1,
                       corridor_views: int = 0, min_views: int = 0,
                       model: Optional["ModelMask"] = None) -> DockingClearance:
    """Obstruction = confirmed voxel inside the keep-out zone `corridor` that the
    satellite model does not explain (`model`, same frame as the map)."""
    pts = smap.confirmed()
    inside = pts[corridor.contains(pts)] if len(pts) else pts
    if model is not None and len(inside):
        inside = inside[~model.explains(inside)]
    n = int(len(inside))
    nearest = float(np.linalg.norm(inside - corridor.exit_centre, axis=1).min()) if n else math.inf
    observed = int(corridor_views) >= int(min_views)
    return DockingClearance(is_clear=observed and n < int(min_voxels), obstruction_voxels=n, nearest_m=nearest,
                            observed_frames=smap.frames, corridor_views=int(corridor_views), observed=observed,
                            obstruction_points=inside)


###################
### Model mask ###
###################


def sample_triangles(points: np.ndarray, tris: np.ndarray, spacing: float,
                     centre=None, radius: float = math.inf) -> np.ndarray:
    """Surface samples of a triangle mesh about `spacing` apart (vertices included);
    only triangles with a vertex within `radius` of `centre` (if given)."""
    p = np.asarray(points, dtype=float)
    t = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
    if centre is not None and len(t):
        near = np.linalg.norm(p - np.asarray(centre, dtype=float), axis=1) <= radius
        t = t[near[t].any(axis=1)]
    if not len(t):
        return np.zeros((0, 3))
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    edge = np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1))
    n = np.clip(np.ceil(edge / spacing).astype(np.int64), 1, 400)
    out = [a, b, c]
    for k in np.unique(n):
        sel = n == k
        # barycentric grid with k steps per edge
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1))
        m = (i + j) <= k
        u, v = (i[m] / k)[None, :, None], (j[m] / k)[None, :, None]
        aa, bb, cc = a[sel][:, None], b[sel][:, None], c[sel][:, None]
        out.append((aa + u * (bb - aa) + v * (cc - aa)).reshape(-1, 3))
    return np.vstack(out)


class ModelMask:
    """Voxels a known 3D model occupies (dilated), in one fixed frame: what the scan is
    expected to see. `explains(points)` flags points on / next to the model surface."""

    def __init__(self, surface_points: np.ndarray, voxel_size_m: float, dilate: int = 1):
        self.voxel = float(voxel_size_m)
        idx = np.unique(np.floor(np.asarray(surface_points, dtype=float).reshape(-1, 3) / self.voxel).astype(np.int64), axis=0)
        r = np.arange(-int(dilate), int(dilate) + 1)
        off = np.stack(np.meshgrid(r, r, r), axis=-1).reshape(-1, 3)
        self.keys = np.unique(_pack((idx[:, None, :] + off[None]).reshape(-1, 3))) if len(idx) else np.zeros(0, dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.keys))

    def explains(self, points: np.ndarray) -> np.ndarray:
        k = _pack(np.floor(np.asarray(points, dtype=float).reshape(-1, 3) / self.voxel).astype(np.int64))
        if not len(self.keys):
            return np.zeros(len(k), dtype=bool)
        i = np.clip(np.searchsorted(self.keys, k), 0, len(self.keys) - 1)
        return self.keys[i] == k


####################
### Map building ###
####################


def depth_frame_points(depth: np.ndarray, k: np.ndarray, cam_w: Frame, sat_w: Frame, cfg: AstrobeeMapCfg,
                       corridor: Optional[DockingCorridor] = None,
                       crop: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       exclude_capsules: Sequence[Tuple[np.ndarray, np.ndarray, float]] = (),
                       frame_index: int = 0,
                       exclude_models: Sequence[Tuple[Frame, "ModelMask"]] = ()) -> Tuple[np.ndarray, Frame, int]:
    """Satellite-frame points of one depth image, the camera pose in that frame and the
    number of those points inside the corridor's bounding sphere (0 without `corridor`).

    Grid-subsampled over the image (grid shifted for the `frame_index`-th sample), dense
    over the corridor window (`corridor`), cropped
    to the satellite box `crop` = (lo, hi) (satellite frame), minus points inside any of
    `exclude_capsules` = (a, b, radius) (world frame) and minus points a known moving
    model explains (`exclude_models` = (model pose in the world, `ModelMask`))."""
    h, w = depth.shape
    cam_s = sat_w.inv() @ cam_w
    roi = None
    if corridor is not None:
        centre, radius = corridor.bounding_sphere()
        roi = sphere_roi(transform(cam_s.inv(), centre[None, :])[0], radius, k, w, h)
    u, v = pixel_grid(w, h, cfg.grid_stride_px, roi, cfg.roi_stride_px, grid_offset(frame_index, cfg.grid_stride_px))
    pc = backproject(depth, k, u, v, cfg.max_range_m)
    pw = transform(cam_w, pc)
    keep = np.ones(len(pw), dtype=bool)
    for a, b, r in exclude_capsules:
        keep &= ~capsule_mask(pw, a, b, r)
    for pose, model in exclude_models:
        keep &= ~model.explains(transform(pose.inv(), pw))
    ps = transform(sat_w.inv(), pw[keep])
    if crop is not None:
        ps = ps[box_mask(ps, crop[0] - cfg.crop_margin_m, crop[1] + cfg.crop_margin_m)]
    near = 0
    if corridor is not None and len(ps):
        near = int((np.linalg.norm(ps - centre, axis=1) <= radius).sum())
    return ps, cam_s, near


def write_ply(path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> Path:
    """Binary little-endian PLY (float32 xyz, optional uint8 rgb)."""
    path = Path(path)
    p = np.asarray(points, dtype="<f4").reshape(-1, 3)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(p)}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
        rec = np.zeros(len(p), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = p[:, 0], p[:, 1], p[:, 2]
        c = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        rec["r"], rec["g"], rec["b"] = c[:, 0], c[:, 1], c[:, 2]
        body = rec.tobytes()
    else:
        body = p.tobytes()
    header.append("end_header")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(header) + "\n").encode("ascii") + body)
    return path


def read_ply_xyz(path) -> np.ndarray:
    """xyz of a PLY written by `write_ply`."""
    data = Path(path).read_bytes()
    end = data.index(b"end_header\n") + len(b"end_header\n")
    lines = data[:end].decode("ascii").splitlines()
    n = int(next(x for x in lines if x.startswith("element vertex")).split()[-1])
    rgb = any(x.startswith("property uchar") for x in lines)
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if rgb else []))
    rec = np.frombuffer(data[end:end + n * dt.itemsize], dtype=dt)
    return np.column_stack((rec["x"], rec["y"], rec["z"])).astype(float)
