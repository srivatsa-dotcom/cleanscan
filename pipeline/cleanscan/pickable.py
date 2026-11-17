"""Pickable objects: capture session -> watertight mesh -> pose in the room.

Reconstruction (``reconstruct_pickable``):
  1. the object is where the camera's viewing rays converge (it was walked around); the
     support it stands on is the dominant horizontal surface around it
  2. space carving on a fine voxel grid above the support: a voxel is empty if some frame
     sees *through* it (it lies in front of the measured depth), or sees background there
     (outside the object mask, when the app wrote one). Voxels that are never seen empty
     are solid. This gives a closed solid by construction: the bottom, which stood on the
     support, is closed flat
  3. marching cubes on the smoothed occupancy -> watertight mesh, photo texture from the
     capture
The mesh is expressed in the object frame: origin at the bottom centre, z up.

Registration (``register_to_room``): the room pass saw the object where it belongs. The
mesh is placed on that spot (the support height under it), then a yaw search plus a 4-DoF
ICP (yaw, x, y, z) fits the mesh to the room's points of that object.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from . import carving
from .session import Session, load_label_image

log = logging.getLogger(__name__)

UP_Y = np.array([0.0, 1.0, 0.0])  # capture frames are gravity aligned, +Y up


@dataclass
class PickableConfig:
    voxel: float = 0.004  # m
    search_radius: float = 0.35  # object within this horizontal distance of the orbit centre
    max_height: float = 0.6
    depth_tol: float = 0.012  # m; a voxel must be this far in front of the surface to be "seen through"
    min_free_votes: int = 2
    mask_dilate_px: int = 1
    max_triangles: int = 6000
    texture_size: int = 512
    use_app_masks: bool = True  # object masks written by the app, when present
    noisy_free_frac: float = 0.2  # noisy depth (module, stereo): share of views that must agree


@dataclass
class PickableResult:
    V: np.ndarray  # object frame (origin bottom centre, z up)
    F: np.ndarray
    T_capture_object: np.ndarray  # object -> capture world (Y up)
    textured: object | None
    info: dict = field(default_factory=dict)


def _rays_center(session: Session) -> np.ndarray:
    """Least-squares point closest to all optical axes."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for fr in session.frames:
        c = fr.pose_cv[:3, 3]
        d = fr.pose_cv[:3, 2]
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A + 1e-6 * np.eye(3), b)


def _frame_points(session: Session, fr, mask: np.ndarray | None = None) -> np.ndarray:
    depth = session.load_depth(fr)
    h, w = depth.shape
    v, u = np.nonzero((depth > 0) if mask is None else (depth > 0) & mask)
    z = depth[v, u]
    K = fr.K_depth
    pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    P = fr.pose_cv
    return pc @ P[:3, :3].T + P[:3, 3]


def _object_mask(fr, size) -> np.ndarray | None:
    m = load_label_image(fr.extra_path("object_mask"), size)
    return None if m is None else m > 127


def find_support_and_extent(session: Session, cfg: PickableConfig, frames=None):
    """(support height y, object centre on the support (x, z), horizontal radius, height)."""
    frames = frames or session.frames
    c = _rays_center(session)
    pts = []
    masked = []
    step = max(1, len(frames) // 12)
    for fr in frames[::step]:
        p = _frame_points(session, fr)
        pts.append(p)
        m = _object_mask(fr, fr.depth_size) if cfg.use_app_masks else None
        if m is not None:
            masked.append(_frame_points(session, fr, m))
    P = np.concatenate(pts)
    r = np.hypot(P[:, 0] - c[0], P[:, 2] - c[2])
    near = P[r < cfg.search_radius]
    # support: the lowest strongly populated height near the object (an annulus excludes it)
    ring = near[np.hypot(near[:, 0] - c[0], near[:, 2] - c[2]) > 0.12]
    hist, edges = np.histogram(ring[:, 1], bins=np.arange(ring[:, 1].min(), ring[:, 1].max() + 0.01, 0.005))
    k = int(np.argmax(hist))
    sel = (ring[:, 1] >= edges[k] - 0.005) & (ring[:, 1] < edges[k + 1] + 0.005)
    support = float(np.median(ring[sel, 1]))
    if masked:
        obj = np.concatenate(masked)
    else:
        obj = near[near[:, 1] > support + 0.012]  # clear of floor noise
    obj = obj[(obj[:, 1] > support + 0.003) & (obj[:, 1] < support + cfg.max_height)]
    if len(obj) < 50:
        raise RuntimeError("no object found above the support surface in the capture")
    # keep the cluster closest to the ray centre (other things on the floor are ignored)
    q = np.round(obj / 0.02).astype(np.int64)
    keys, inv = np.unique(q, axis=0, return_inverse=True)
    grid = {tuple(kk): i for i, kk in enumerate(keys)}
    lab = -np.ones(len(keys), np.int64)
    n = 0
    for i in range(len(keys)):
        if lab[i] >= 0:
            continue
        stack = [i]
        lab[i] = n
        while stack:
            j = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nb = grid.get((keys[j][0] + dx, keys[j][1] + dy, keys[j][2] + dz))
                        if nb is not None and lab[nb] < 0:
                            lab[nb] = n
                            stack.append(nb)
        n += 1
    plab = lab[inv.ravel()]
    best = min(range(n), key=lambda L: np.hypot(*(np.median(obj[plab == L][:, [0, 2]], 0) - c[[0, 2]]))
               - 0.001 * (plab == L).sum())
    obj = obj[plab == best]
    lo, hi = np.percentile(obj, 1, axis=0), np.percentile(obj, 99, axis=0)
    centre = (lo[[0, 2]] + hi[[0, 2]]) / 2
    radius = 1.15 * float(np.percentile(np.hypot(obj[:, 0] - centre[0], obj[:, 2] - centre[1]), 99)) + 0.005
    height = float(hi[1] - support)
    return support, centre, radius, height, len(masked) > 0


def reconstruct_pickable(session: Session, cfg: PickableConfig | None = None, texture: bool = True) -> PickableResult:
    cfg = cfg or PickableConfig()
    support, centre, radius, height, has_masks = find_support_and_extent(session, cfg)
    margin = 3 * cfg.voxel + 0.01
    lo = np.array([centre[0] - radius - margin, support, centre[1] - radius - margin])
    hi = np.array([centre[0] + radius + margin, support + height + margin, centre[1] + radius + margin])
    mask_fn = (lambda fr, d: _object_mask(fr, fr.depth_size)) if has_masks else None
    views = carving.session_views(session, mask_fn=mask_fn)
    noisy = getattr(session, "depth_noise", (0, 0))[1] >= 0.005
    occ = carving.carve(views, lo, hi, cfg.voxel, depth_tol=cfg.depth_tol,
                        min_free_votes=cfg.min_free_votes, mask_dilate_px=cfg.mask_dilate_px,
                        min_free_frac=cfg.noisy_free_frac if noisy else 0.0)
    occ[:, 0, :] = occ[:, 1, :]  # the bottom layer touches the support
    if not occ.any():
        raise RuntimeError("space carving removed everything; check the capture")
    ci = np.round((np.array([centre[0], support + height / 2, centre[1]]) - lo) / cfg.voxel).astype(int)
    occ = carving.keep_component(occ, ci)
    Vg, Fg = carving.solid_to_mesh(occ, lo, cfg.voxel, cfg.max_triangles)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vg), o3d.utility.Vector3iVector(Fg))
    mesh.compute_vertex_normals()
    Vc = np.asarray(mesh.vertices)
    bottom = Vc[:, 1].min()
    # object frame: origin = bottom centre, z = up (capture +y), x = capture +x
    T_co = np.eye(4)
    T_co[:3, 0] = [1, 0, 0]
    T_co[:3, 1] = [0, 0, -1]
    T_co[:3, 2] = UP_Y
    T_co[:3, 3] = [centre[0], bottom, centre[1]]
    Tinv = np.linalg.inv(T_co)
    V = Vc @ Tinv[:3, :3].T + Tinv[:3, 3]
    F = np.asarray(mesh.triangles)
    textured = None
    if texture:
        from .texture import TextureConfig, bake_texture

        tm = bake_texture(session, mesh, TextureConfig(size=cfg.texture_size, max_keyframes=12), None)
        from .pipeline import transform_textured

        textured = transform_textured(tm, Tinv)
    from .physics import is_watertight

    info = {"support_height_capture": support, "height_m": round(float(V[:, 2].max()), 4),
            "radius_m": round(radius, 4), "voxel_m": cfg.voxel, "used_app_masks": has_masks,
            "watertight": bool(is_watertight(F)), "triangles": int(len(F))}
    log.info("pickable reconstructed: %.1f cm tall, %d triangles, watertight=%s",
             info["height_m"] * 100, len(F), info["watertight"])
    return PickableResult(V, F, T_co, textured, info)


# --------------------------------------------------------------------------- registration


def _yaw(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def register_to_room(V: np.ndarray, F: np.ndarray, region: np.ndarray, support_heights: np.ndarray,
                     n_yaw: int = 24, iters: int = 15) -> tuple[np.ndarray, dict]:
    """Pose (object -> scene) of a mesh (object frame, z up, origin at the bottom) that best
    explains the room-pass points of the object. Returns (4x4, info)."""
    region = np.asarray(region, float)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_uniformly(4000, use_triangle_normal=True)
    S, N = np.asarray(pcd.points), np.asarray(pcd.normals)
    tree = cKDTree(S)
    # coarse: centred on the region's footprint, standing on the support below it
    lo, hi = np.percentile(region, 2, axis=0), np.percentile(region, 98, axis=0)
    cxy = (lo[:2] + hi[:2]) / 2
    z0 = lo[2]
    on_support = False
    if len(support_heights):
        below = support_heights[(support_heights <= z0 + 0.03) & (support_heights >= z0 - 0.10)]
        if len(below):
            z0 = float(below[np.argmin(np.abs(below - z0))])
            on_support = True  # standing on a known surface: the height is fixed
    # rotationally symmetric objects (bottles, glasses) do not need a yaw search
    r = np.hypot(S[:, 0], S[:, 1])
    ang = np.arctan2(S[:, 1], S[:, 0])
    bins = np.digitize(ang, np.linspace(-np.pi, np.pi, 13))
    rad = [np.percentile(r[bins == b], 95) for b in range(1, 13) if (bins == b).sum() > 20]
    symmetric = len(rad) == 12 and (max(rad) - min(rad)) < 0.08 * max(rad)
    yaws = [0.0] if symmetric else np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)

    def score(R, t):
        # room points -> model surface (the room only saw part of the object)
        q = (region - t) @ R  # into the object frame
        d, i = tree.query(q)
        return float(np.mean(np.minimum(np.abs(np.einsum("ij,ij->i", q - S[i], N[i])) + 0.3 * d, 0.05))), q, i

    best = None
    for a in yaws:
        R = _yaw(a)
        t = np.array([cxy[0], cxy[1], z0])
        for _ in range(iters):
            _, q, i = score(R, t)
            d = np.linalg.norm(q - S[i], axis=1)
            w = (d < 0.05).astype(float)
            if w.sum() < 10:
                break
            # point-to-plane in the object frame: q = R^T (p - t), residual n . (q - s).
            # update R <- R Rz(a), t <- t + dt:  q' ~ q - a (z x q) - R^T dt
            n = N[i]
            Jy = np.cross(np.array([0, 0, 1.0]), q)
            A = np.stack([-np.einsum("ij,ij->i", n, Jy), *(-(n @ R.T).T)], 1)  # columns: a, dtx, dty, dtz
            res = np.einsum("ij,ij->i", n, q - S[i])
            sw = np.sqrt(w)
            cols = 3 if on_support else 4
            x = np.zeros(4)
            x[:cols], *_ = np.linalg.lstsq(A[:, :cols] * sw[:, None], -res * sw, rcond=None)
            x[0] = np.clip(x[0], -0.2, 0.2)
            x[1:] = np.clip(x[1:], -0.03, 0.03)
            R = R @ _yaw(x[0]) if not symmetric else R
            t = t + x[1:]
            if np.abs(x).max() < 1e-4:
                break
        s, _, _ = score(R, t)
        if best is None or s < best[0]:
            best = (s, R, t)
    s, R, t = best
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, {"fit_error_m": round(s, 4), "symmetric": bool(symmetric), "room_points": int(len(region)),
               "coarse_position": [round(float(cxy[0]), 4), round(float(cxy[1]), 4), round(z0, 4)],
               "on_support": on_support}


def settle(T: np.ndarray, V: np.ndarray, support_heights: np.ndarray, max_shift: float = 0.03,
           clearance: float = 0.001) -> tuple[np.ndarray, float]:
    """Put the object's lowest point on the nearest support surface (+1 mm)."""
    Vw = V @ T[:3, :3].T + T[:3, 3]
    bottom = Vw[:, 2].min()
    if not len(support_heights):
        return T, 0.0
    h = support_heights[np.argmin(np.abs(support_heights - bottom))]
    dz = float(h + clearance - bottom)
    if abs(dz) > max_shift:
        return T, 0.0
    T = T.copy()
    T[2, 3] += dz
    return T, dz


def fallback_primitive(region: np.ndarray, cls: str, support_heights: np.ndarray):
    """No usable capture: a cylinder (round classes) or box from the room-pass points."""
    lo, hi = np.percentile(region, 2, axis=0), np.percentile(region, 98, axis=0)
    z0 = lo[2]
    if len(support_heights):
        cand = support_heights[np.abs(support_heights - z0) < 0.05]
        if len(cand):
            z0 = float(cand[np.argmin(np.abs(cand - z0))])
    h = max(hi[2] - z0, 0.02)
    if cls in ("bottle", "cup", "mug", "can", "glass", "wine glass", "vase", "bowl", "jar"):
        r = max(hi[0] - lo[0], hi[1] - lo[1]) / 2
        m = o3d.geometry.TriangleMesh.create_cylinder(r, h, 32, 1).translate((0, 0, h / 2))
    else:
        m = o3d.geometry.TriangleMesh.create_box(hi[0] - lo[0], hi[1] - lo[1], h)
        m.translate((-(hi[0] - lo[0]) / 2, -(hi[1] - lo[1]) / 2, 0))
    m.remove_duplicated_vertices()
    T = np.eye(4)
    T[:3, 3] = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, z0]
    return np.asarray(m.vertices), np.asarray(m.triangles), T
