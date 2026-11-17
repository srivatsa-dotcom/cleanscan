"""Articulated objects (fridge, cupboard, drawer, oven, door): closed + open capture -> joint.

1. Change detection. A point of the *closed* capture that the *open* capture sees through
   (its measured depth lies clearly behind the point) belonged to the moving part: P_c, the
   part in its closed position. Likewise open-capture points that the closed capture sees
   through are the part in its open position: P_o. Occluded points never vote, so a closed
   door hiding the fridge's interior does not count as change.
2. Joint fit. Hypotheses: rotations about each edge of P_c's panel (hinges) and slides along
   its normal. The best one (P_c moved onto P_o) seeds an ICP; the resulting rigid motion M is
   decomposed: rotation > 5 deg -> revolute (axis, a point on it, opening angle), else
   prismatic (axis, distance). Limits are [0, observed opening].
3. Meshes. The moving part is carved in its own frame: closed-capture views directly,
   open-capture views moved back through M (the part is static in that frame). The body is
   carved from the room pass and both captures with the part's pixels ignored, so the
   interior seen through the open door becomes part of the body.

Everything here is in the capture frame (Y up); the caller converts to the scene.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import carving
from .session import Session

log = logging.getLogger(__name__)

UP = np.array([0.0, 1.0, 0.0])


@dataclass
class ArticulatedConfig:
    point_voxel: float = 0.01
    roi_radius: float = 1.3  # m around the object (horizontal)
    change_tol: float = 0.03  # m behind a point for it to count as seen through
    change_votes: int = 2
    part_voxel: float = 0.01
    body_voxel: float = 0.015
    part_pixel_dist: float = 0.025  # a pixel shows the part if within this of its points
    revolute_min_deg: float = 5.0
    max_room_views: int = 40
    part_floor_clearance: float = 0.01  # m; moving parts (doors) must not touch the floor
    snap_deg: float = 4.0  # snap joint axes this close to the object's main directions
    drawer_back_gap: float = 0.03  # inferred drawer depth stops this short of the body's back
    texture_size: int = 512


@dataclass
class JointFit:
    type: str  # revolute | prismatic
    axis: np.ndarray  # unit, capture frame
    origin: np.ndarray  # a point on the axis (revolute) / on the part (prismatic)
    value: float  # observed opening (rad or m)
    M: np.ndarray  # closed -> open motion of the part (capture frame)
    score: float
    info: dict = field(default_factory=dict)


@dataclass
class ArticulatedResult:
    A: np.ndarray  # object -> capture frame
    body: tuple  # (V, F) object frame
    part: tuple  # (V, F) object frame, closed configuration
    joint: dict  # name, type, axis, origin (object frame), lower, upper
    fit: JointFit
    textures: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- points


def _rays_center(session: Session) -> np.ndarray:
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for fr in session.frames:
        c, d = fr.pose_cv[:3, 3], fr.pose_cv[:3, 2]
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c
    return np.linalg.solve(A + 1e-6 * np.eye(3), b)


def backproject(session: Session, fr, depth=None) -> tuple[np.ndarray, np.ndarray]:
    """(world points (N,3), pixel indices (N,2) as (row, col)) of the valid depth pixels."""
    depth = session.load_depth(fr) if depth is None else depth
    v, u = np.nonzero(depth > 0)
    z = depth[v, u]
    K = fr.K_depth
    pc = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    P = fr.pose_cv
    return pc @ P[:3, :3].T + P[:3, 3], np.stack([v, u], 1)


def capture_points(session: Session, center, radius, floor_y, voxel, with_cameras: bool = False):
    """Depth points of a capture near the object, one per voxel. With ``with_cameras``, also
    the centre of the camera that observed each point (to orient normals)."""
    pts, cams = [], []
    for fr in session.frames:
        p, _ = backproject(session, fr)
        keep = (np.hypot(p[:, 0] - center[0], p[:, 2] - center[2]) < radius) & (p[:, 1] > floor_y + 0.02)
        pts.append(p[keep])
        cams.append(np.broadcast_to(fr.pose_cv[:3, 3], (int(keep.sum()), 3)))
    P = np.concatenate(pts)
    C = np.concatenate(cams)
    _, first = np.unique(np.floor(P / voxel).astype(np.int64), axis=0, return_index=True)
    return (P[first], C[first]) if with_cameras else P[first]


def seen_through(points: np.ndarray, views: list[carving.View], tol: float, window: int = 5,
                 min_frac: float = 0.5) -> np.ndarray:
    """Number of views that clearly see *through* each point (0 where most views that see the
    point's place agree it is there).

    The test uses the minimum depth over a small pixel window, so points on a silhouette are
    not 'seen through' just because a neighbouring pixel shows the background."""
    import cv2

    through = np.zeros(len(points), np.int32)
    agree = np.zeros(len(points), np.int32)
    k = np.ones((window, window), np.uint8)
    for v in views:
        h, w = v.depth.shape
        dmin = cv2.erode(v.depth.astype(np.float32), k)  # 0 (no data) wins -> never "through"
        pc = points @ v.E[:3, :3].T + v.E[:3, 3]
        z = pc[:, 2]
        ok = z > 0.1
        u = np.zeros(len(points), np.int64)
        r = np.zeros(len(points), np.int64)
        u[ok] = np.floor(v.K[0, 0] * pc[ok, 0] / z[ok] + v.K[0, 2]).astype(np.int64)
        r[ok] = np.floor(v.K[1, 1] * pc[ok, 1] / z[ok] + v.K[1, 2]).astype(np.int64)
        ok &= (u >= 0) & (u < w) & (r >= 0) & (r < h)
        idx = np.nonzero(ok)[0]
        d = v.depth[r[idx], u[idx]]
        dm = dmin[r[idx], u[idx]]
        through[idx[(dm > 0) & (dm > z[idx] + tol)]] += 1
        agree[idx[(d > 0) & (np.abs(d - z[idx]) < tol)]] += 1
    return np.where(through >= min_frac * (through + agree), through, 0)


def oriented_normals(P: np.ndarray, cams: np.ndarray, radius: float = 0.04) -> np.ndarray:
    """Normals facing the camera that observed each point (``cams``: one centre per point), so
    the two faces of a thin panel get opposite normals."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    N = np.asarray(pc.normals).copy()
    flip = np.einsum("ij,ij->i", N, cams - P) < 0
    N[flip] *= -1
    return N


class _NormalMatcher:
    """Nearest neighbours that also face the same way (front face never matches back face)."""

    def __init__(self, P, N, k=8):
        self.P, self.N, self.k = P, N, k
        self.tree = cKDTree(P)

    def query(self, q, nq, dmax):
        d, i = self.tree.query(q, k=self.k, distance_upper_bound=dmax)
        valid = np.isfinite(d)
        i = np.where(valid, i, 0)
        agree = valid & (np.einsum("nkj,nj->nk", self.N[i], nq) > 0.3)
        first = np.argmax(agree, axis=1)
        ok = agree[np.arange(len(q)), first]
        dist = np.where(ok, d[np.arange(len(q)), first], np.inf)
        return dist, i[np.arange(len(q)), first], ok


def largest_cluster(P: np.ndarray, extra: np.ndarray, eps: float = 0.04, min_points: int = 8,
                    keep_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """The biggest clusters of P (and the matching rows of ``extra``)."""
    if len(P) < min_points:
        return P, extra
    lab = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).cluster_dbscan(eps, min_points))
    if lab.max() < 0:
        return P, extra
    sizes = np.bincount(lab[lab >= 0])
    keep = np.isin(lab, np.nonzero(sizes >= keep_frac * sizes.max())[0])
    return P[keep], extra[keep]


# --------------------------------------------------------------------------- joint fit


def _motion_revolute(axis, point, angle) -> np.ndarray:
    R = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = point - R @ point
    return M


def _motion_prismatic(axis, dist) -> np.ndarray:
    M = np.eye(4)
    M[:3, 3] = axis * dist
    return M


def fit_joint(Pc: np.ndarray, Po: np.ndarray, cfg: ArticulatedConfig, rng=None,
              Nc: np.ndarray | None = None, No: np.ndarray | None = None) -> JointFit:
    rng = rng or np.random.default_rng(0)
    pick = rng.choice(len(Pc), min(len(Pc), 800), replace=False)
    sub = Pc[pick]
    nsub = None if Nc is None else Nc[pick]
    matcher = None if No is None else _NormalMatcher(Po, No)
    tree = cKDTree(Po)

    def score(M):
        q = sub @ M[:3, :3].T + M[:3, 3]
        if matcher is not None:
            d, _, _ = matcher.query(q, nsub @ M[:3, :3].T, 0.08)
        else:
            d, _ = tree.query(q, distance_upper_bound=0.08)
        return float(np.mean(np.minimum(d, 0.08)))

    c = Pc.mean(0)
    w, e = np.linalg.eigh(np.cov((Pc - c).T))
    n, v, u = e[:, 0], e[:, 1], e[:, 2]
    a, b = (Pc - c) @ u, (Pc - c) @ v
    a0, a1 = np.percentile(a, [1, 99])
    b0, b1 = np.percentile(b, [1, 99])
    # hinge lines along the panel's edges. Points right next to a hinge hardly move, so the
    # changed region stops short of it: also try lines up to 12 cm beyond each edge, and
    # slightly in front of / behind the panel (its thickness)
    hinges = []
    for off in (0.0, 0.03, 0.06, 0.09, 0.12):
        for dn in (-0.03, 0.0, 0.03):
            hinges += [(c + (a0 - off) * u + dn * n, v), (c + (a1 + off) * u + dn * n, v),
                       (c + (b0 - off) * v + dn * n, u), (c + (b1 + off) * v + dn * n, u)]
    hyps = []
    angles = np.radians(np.concatenate([np.arange(-175, -4, 5.0), np.arange(5, 176, 5.0)]))
    for p, d in hinges:
        for ang in angles:
            M = _motion_revolute(d, p, ang)
            hyps.append((score(M), "revolute", M))
    for d in (n, -n, u, -u, v, -v):
        for dist in np.arange(0.02, 0.9, 0.01):
            M = _motion_prismatic(d, dist)
            hyps.append((score(M), "prismatic", M))
    hyps.sort(key=lambda h: h[0])
    best_rev = next(h for h in hyps if h[1] == "revolute")
    best_pri = next(h for h in hyps if h[1] == "prismatic")
    s0, kind0, M0 = hyps[0]

    # ICP refinement of the full rigid motion
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pc))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Po))
    M = M0
    for dmax in (0.05, 0.02, 0.01):
        M = o3d.pipelines.registration.registration_icp(
            src, dst, dmax, M, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)).transformation
    if score(M) > s0:  # ICP (no normals) wandered off; keep the hypothesis
        M = M0
    R, t = M[:3, :3], M[:3, 3]
    rv = Rotation.from_matrix(R).as_rotvec()
    angle = float(np.linalg.norm(rv))
    info = {"hypothesis": kind0, "hypothesis_score_m": round(s0, 4),
            "best_revolute_score_m": round(best_rev[0], 4), "best_prismatic_score_m": round(best_pri[0], 4),
            "icp_score_m": round(score(M), 4), "part_points_closed": int(len(Pc)), "part_points_open": int(len(Po))}
    if np.degrees(angle) > cfg.revolute_min_deg:
        axis = rv / angle
        # point on the axis: (I - R) p = t, pinned along the axis at the part's centroid
        A = np.vstack([np.eye(3) - R, axis[None]])
        rhs = np.concatenate([t, [axis @ c]])
        p, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        return JointFit("revolute", axis, p, angle, M, score(M), info)
    dist = float(np.linalg.norm(t))
    axis = t / max(dist, 1e-9)
    return JointFit("prismatic", axis, c, dist, M, score(M), info)


def refine_with_axis(fit: JointFit, Pc: np.ndarray, Po: np.ndarray, axis: np.ndarray,
                     Nc: np.ndarray | None = None, No: np.ndarray | None = None) -> JointFit:
    """Re-fit the joint with a fixed axis direction by minimising the same robust,
    normal-aware distance as the hypothesis search: hinge position (2 dof) + opening angle,
    or the slide distance. A fine local grid, then Powell. (Nearest-point least squares
    slides along the door's own surface when parts of it were not observed.)"""
    from scipy.optimize import minimize

    axis = axis / np.linalg.norm(axis)
    if fit.axis @ axis < 0:
        axis = -axis
    rng = np.random.default_rng(0)
    pick = rng.choice(len(Pc), min(len(Pc), 1500), replace=False)
    sub = Pc[pick]
    nsub = None if Nc is None else Nc[pick]
    matcher = None if No is None else _NormalMatcher(Po, No)
    tree = cKDTree(Po)

    pick_o = rng.choice(len(Po), min(len(Po), 1500), replace=False)
    sub_o, nsub_o = Po[pick_o], None if No is None else No[pick_o]

    def dist(M):
        q = sub @ M[:3, :3].T + M[:3, 3]
        if matcher is not None:
            d, _, _ = matcher.query(q, nsub @ M[:3, :3].T, 0.08)
        else:
            d, _ = tree.query(q, distance_upper_bound=0.08)
        fwd = float(np.mean(np.minimum(d, 0.08)))
        if Nc is None:
            return fwd
        # and back: open points against the moved closed points (both faces disambiguated)
        moved = _NormalMatcher(sub @ M[:3, :3].T + M[:3, 3], nsub @ M[:3, :3].T)
        d2, _, _ = moved.query(sub_o, nsub_o, 0.08)
        return fwd + float(np.mean(np.minimum(d2, 0.08)))

    e1 = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    if fit.type == "revolute":
        # the fitted motion's rotation sign may flip with the axis
        ang0 = fit.value if fit.axis @ axis > 0 else -fit.value
        p0 = fit.origin - axis * ((fit.origin - Pc.mean(0)) @ axis)

        def motion(x):
            return _motion_revolute(axis, p0 + x[0] * e1 + x[1] * e2, x[2])
        best = (dist(motion([0, 0, ang0])), np.array([0.0, 0.0, ang0]))
        for da in np.radians(np.arange(-4, 4.1, 2.0)):
            for x0 in np.arange(-0.075, 0.076, 0.015):
                for x1 in np.arange(-0.075, 0.076, 0.015):
                    x = np.array([x0, x1, ang0 + da])
                    sc = dist(motion(x))
                    if sc < best[0]:
                        best = (sc, x)
        x = minimize(lambda xx: dist(motion(xx)), best[1], method="Powell",
                     options={"xtol": 1e-4, "ftol": 1e-6, "maxfev": 400}).x
        if dist(motion(x)) > best[0]:
            x = best[1]
        origin, value = p0 + x[0] * e1 + x[1] * e2, float(x[2])
    else:
        d0 = fit.value if fit.axis @ axis > 0 else -fit.value

        def motion(x):
            return _motion_prismatic(axis, x[0])
        grid = d0 + np.arange(-0.04, 0.0401, 0.002)
        x = np.array([grid[int(np.argmin([dist(motion([g])) for g in grid]))]])
        x = minimize(lambda xx: dist(motion(xx)), x, method="Powell", options={"xtol": 1e-4}).x
        origin, value = fit.origin, float(x[0])
    M = motion(x)
    if value < 0:
        axis, value = -axis, -value
    return JointFit(fit.type, axis, origin, value, M, dist(M), {**fit.info, "refined": True})


# --------------------------------------------------------------------------- reconstruction


def _object_frame(Pc: np.ndarray, cams: np.ndarray) -> np.ndarray:
    """object -> capture: z up, y pointing into the object (away from the user), x = y x z."""
    c = Pc.mean(0)
    w, e = np.linalg.eigh(np.cov((Pc - c).T))
    n = e[:, 0]
    if n @ (cams.mean(0) - c) < 0:
        n = -n  # n faces the user
    f = n - (n @ UP) * UP
    f /= np.linalg.norm(f)
    y = -f
    z = UP
    x = np.cross(y, z)
    A = np.eye(4)
    A[:3, 0], A[:3, 1], A[:3, 2] = x, y, z
    return A


def _to(A_inv, P):
    return P @ A_inv[:3, :3].T + A_inv[:3, 3]


def _part_mask_fn(session_or_none, tree: cKDTree, dist: float):
    def fn(fr, depth):
        mask = np.zeros(depth.shape, bool)
        p, pix = backproject(session_or_none, fr, depth)
        if len(p):
            d, _ = tree.query(p, distance_upper_bound=dist)
            sel = np.isfinite(d)
            mask[pix[sel, 0], pix[sel, 1]] = True
        return mask
    return fn


def _relevant_frames(session: Session, center: np.ndarray, n_max: int):
    out = []
    for fr in session.frames:
        pc = fr.extrinsic_cv @ np.append(center, 1)
        if pc[2] < 0.3:
            continue
        u = fr.K_depth[0, 0] * pc[0] / pc[2] + fr.K_depth[0, 2]
        v = fr.K_depth[1, 1] * pc[1] / pc[2] + fr.K_depth[1, 2]
        w, h = fr.depth_size
        if 0 <= u < w and 0 <= v < h:
            out.append(fr)
    step = max(1, len(out) // n_max)
    return out[::step]


def _keep_touching(occ, Qo, lo, voxel, min_frac=0.02):
    """Keep every connected piece that contains a fair share of the part's points."""
    from scipy.ndimage import label as cc_label

    lab, n = cc_label(occ)
    if n <= 1:
        return occ
    idx = np.floor((Qo - lo) / voxel).astype(int)
    ok = ((idx >= 0) & (idx < np.array(occ.shape))).all(1)
    hits = np.bincount(lab[idx[ok, 0], idx[ok, 1], idx[ok, 2]], minlength=n + 1)
    hits[0] = 0
    keep = np.nonzero(hits >= min_frac * max(hits.max(), 1))[0]
    return np.isin(lab, keep)


def _extend_drawer(occ, lo, voxel, axis_o, body_back_y):
    """Prior for drawers (only partly visible even when open): extrude the innermost observed
    cross-section back to just before the body's back panel and close it with a back panel.
    Only for slides along the object's depth axis (y)."""
    if abs(axis_o[1]) < 0.9:
        return occ, lo, None
    inward = 1 if axis_o[1] < 0 else -1  # moving out along -y means the drawer extends to +y
    ys = np.nonzero(occ.any(axis=(0, 2)))[0]
    if not len(ys):
        return occ, lo, None
    last = ys.max() if inward > 0 else ys.min()
    end_y = body_back_y
    n_extra = int(np.floor((end_y - (lo[1] + (last + 1) * voxel)) / voxel)) if inward > 0 else 0
    if n_extra <= 1:
        return occ, lo, {"extended_m": 0.0}
    slice_ = occ[:, last, :]
    pad = np.zeros((occ.shape[0], n_extra, occ.shape[2]), bool)
    pad[:] = slice_[:, None, :]
    xs, zs = np.nonzero(slice_)
    pad[xs.min():xs.max() + 1, -2:, zs.min():zs.max() + 1] = True  # back panel
    occ = np.concatenate([occ[:, :last + 1, :], pad], axis=1)
    return occ, lo, {"extended_m": round(n_extra * voxel, 3), "note": "drawer depth inferred, not observed"}


def reconstruct_articulated(closed: Session, opened: Session, room: Session | None, floor_y: float,
                            region: np.ndarray | None = None, cfg: ArticulatedConfig | None = None,
                            part_class_hint: str | None = None) -> ArticulatedResult:
    """``region``: room-pass points of the object (capture frame), when the detector found it."""
    cfg = cfg or ArticulatedConfig()
    center = _rays_center(closed)
    Pcl, Ccl = capture_points(closed, center, cfg.roi_radius, floor_y, cfg.point_voxel, with_cameras=True)
    Pop, Cop = capture_points(opened, center, cfg.roi_radius, floor_y, cfg.point_voxel, with_cameras=True)
    v_closed = carving.session_views(closed)
    v_open = carving.session_views(opened)
    sel_c = seen_through(Pcl, v_open, cfg.change_tol) >= cfg.change_votes
    sel_o = seen_through(Pop, v_closed, cfg.change_tol) >= cfg.change_votes
    Pc, Cc = largest_cluster(Pcl[sel_c], Ccl[sel_c])
    Po, Co = largest_cluster(Pop[sel_o], Cop[sel_o])
    if len(Pc) < 30 or len(Po) < 30:
        raise RuntimeError(f"no moving part found (changed points: closed {len(Pc)}, open {len(Po)}); "
                           "was it opened fully between the two captures?")
    Nc, No = oriented_normals(Pc, Cc), oriented_normals(Po, Co)
    fit = fit_joint(Pc, Po, cfg, Nc=Nc, No=No)
    M, Minv = fit.M, np.linalg.inv(fit.M)
    log.info("joint: %s, opening %.3f %s, fit %.1f mm", fit.type, fit.value,
             "rad" if fit.type == "revolute" else "m", fit.score * 1000)

    # ---- object frame and boxes ----------------------------------------------------------
    cams = np.stack([f.pose_cv[:3, 3] for f in closed.frames])
    A = _object_frame(Pc, cams)
    # hinges and slides are almost always along the object's main directions
    dirs = A[:3, :3].T
    cos = np.abs(dirs @ fit.axis)
    k = int(np.argmax(cos))
    off = float(np.degrees(np.arccos(min(cos[k], 1.0))))
    snapped = off < cfg.snap_deg
    fit = refine_with_axis(fit, Pc, Po, dirs[k] if snapped else fit.axis, Nc, No)
    fit.info["axis_snapped_deg"] = round(off, 2) if snapped else None
    M, Minv = fit.M, np.linalg.inv(fit.M)
    log.info("joint axis %s; opening %.3f, two-way fit score %.1f mm",
             f"snapped by {off:.2f} deg" if snapped else "free", fit.value, fit.score * 1000)
    Q = np.concatenate([Pc, _to(Minv, Po)])  # part, closed configuration
    tree_closed, tree_open = cKDTree(Q), cKDTree(_to(M, Q))
    Ainv = np.linalg.inv(A)
    Qo = _to(Ainv, Q)
    floor_o = _to(Ainv, np.array([[center[0], floor_y, center[2]]]))[0, 2]

    # body box: the room-pass region of the object if the detector found it, else a column
    # behind the moving part reaching back to the wall
    if region is not None and len(region) >= 200:
        R = _to(Ainv, region)
        lo_b, hi_b = np.percentile(R, 2, axis=0), np.percentile(R, 98, axis=0)
        lo_b, hi_b = np.minimum(lo_b, Qo.min(0)), np.maximum(hi_b, Qo.max(0))
        body_source = "room region"
    else:
        allp = _to(Ainv, Pcl)
        x0, x1 = Qo[:, 0].min() - 0.05, Qo[:, 0].max() + 0.05
        col = allp[(allp[:, 0] > x0) & (allp[:, 0] < x1) & (allp[:, 1] > Qo[:, 1].min() - 0.02)
                   & (allp[:, 2] > floor_o + 0.03) & (allp[:, 1] < Qo[:, 1].min() + 1.2)]
        # the wall: the farthest densely populated depth slice in the column
        hist, edges = np.histogram(col[:, 1], bins=np.arange(col[:, 1].min(), col[:, 1].max() + 0.02, 0.02))
        dense = np.nonzero(hist > 0.05 * hist.max())[0]
        wall_y = edges[dense.max()] if len(dense) else Qo[:, 1].max() + 0.6
        body_pts = col[col[:, 1] < wall_y - 0.01]
        top = np.percentile(body_pts[:, 2], 99) if len(body_pts) else Qo[:, 2].max()
        lo_b = np.array([x0 + 0.01, Qo[:, 1].min(), floor_o])
        hi_b = np.array([x1 - 0.01, wall_y, max(top, Qo[:, 2].max())])
        body_source = "column to wall"
    lo_b[2] = floor_o + 0.004
    lo_b[:2] -= 0.02
    hi_b += 0.02

    # ---- moving part ------------------------------------------------------------------------
    pmask_closed = _part_mask_fn(closed, tree_closed, cfg.part_pixel_dist)
    pmask_open = _part_mask_fn(opened, tree_open, cfg.part_pixel_dist)
    lo_p, hi_p = Qo.min(0) - 0.03, Qo.max(0) + 0.03
    lo_p[2] = max(lo_p[2], floor_o + cfg.part_floor_clearance)  # a moving part clears the floor
    drawer_info = None
    occ_c = carving.carve(carving.session_views(closed, mask_fn=pmask_closed), lo_p, hi_p, cfg.part_voxel,
                          grid_to_world=A, depth_tol=0.015, require_seen=False)
    occ_o = carving.carve(carving.session_views(opened, motion=M, mask_fn=pmask_open), lo_p, hi_p,
                          cfg.part_voxel, grid_to_world=A, depth_tol=0.03, require_seen=False)
    occ_p = occ_c & occ_o
    occ_p = _keep_touching(occ_p, Qo, lo_p, cfg.part_voxel)
    if fit.type == "prismatic":
        occ_p, lo_p, drawer_info = _extend_drawer(occ_p, lo_p, cfg.part_voxel, Ainv[:3, :3] @ fit.axis,
                                                  body_back_y=hi_b[1] - 0.02 - cfg.drawer_back_gap)
    part_V, part_F = carving.solid_to_mesh(occ_p, lo_p, cfg.part_voxel)

    # ---- body: everything else in the box; part pixels carry no information --------------
    def zero_part(tree, ses):
        def fn(fr):
            d = ses.load_depth(fr)
            p, pix = backproject(ses, fr, d)
            if len(p):
                dd, _ = tree.query(p, distance_upper_bound=cfg.part_pixel_dist)
                sel = np.isfinite(dd)
                d[pix[sel, 0], pix[sel, 1]] = 0
            return d
        return fn

    views_body = carving.session_views(closed, depth_fn=zero_part(tree_closed, closed)) + \
        carving.session_views(opened, depth_fn=zero_part(tree_open, opened))
    if room is not None:
        body_center = _to(A, ((lo_b + hi_b) / 2)[None])[0]
        frames = _relevant_frames(room, body_center, cfg.max_room_views)
        views_body += carving.session_views(room, frames, depth_fn=zero_part(tree_closed, room))
    occ_b = carving.carve(views_body, lo_b, hi_b, cfg.body_voxel, grid_to_world=A, depth_tol=0.02)
    # the part's own volume (closed) is not body
    Cb, _ = carving.grid_centers(lo_b, hi_b, cfg.body_voxel)
    in_part = np.zeros(len(Cb), bool)
    idx = np.floor((Cb - lo_p) / cfg.part_voxel).astype(int)
    ok = ((idx >= 0) & (idx < np.array(occ_p.shape))).all(1)
    in_part[ok] = occ_p[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
    occ_b &= ~in_part.reshape(occ_b.shape)
    occ_b = carving.keep_component(occ_b)
    body_V, body_F = carving.solid_to_mesh(occ_b, lo_b, cfg.body_voxel, max_triangles=12000)

    # ---- object frame origin: bottom centre of the body ------------------------------------
    o = np.array([(body_V[:, 0].min() + body_V[:, 0].max()) / 2, (body_V[:, 1].min() + body_V[:, 1].max()) / 2,
                  floor_o])
    shift = np.eye(4)
    shift[:3, 3] = o
    A = A @ shift
    body_V, part_V = body_V - o, part_V - o
    Ainv = np.linalg.inv(A)
    axis_o = Ainv[:3, :3] @ fit.axis
    origin_o = _to(Ainv, fit.origin[None])[0]
    if fit.type == "revolute":  # a point on the axis next to the part's middle
        pc = Qo.mean(0) - o
        origin_o = origin_o + axis_o * ((pc - origin_o) @ axis_o)
    joint = {"name": "hinge" if fit.type == "revolute" else "slide", "type": fit.type,
             "parent": "body", "child": "part", "axis": axis_o.tolist(), "origin": origin_o.tolist(),
             "lower": 0.0, "upper": float(fit.value)}
    info = {"joint_fit": fit.info, "joint_fit_score_m": round(fit.score, 4), "body_box": body_source,
            "observed_opening": round(fit.value, 4)}
    if drawer_info:
        info["inferred_drawer_depth"] = drawer_info
    return ArticulatedResult(A, (body_V, body_F), (part_V, part_F), joint, fit, info=info)
