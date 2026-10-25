"""External LiDAR mounted on the phone (the CleanScan rig, see rig/).

Every LiDAR reading carries a phone-clock timestamp. With the phone's 60 Hz ARKit/ARCore
poses and the fixed LiDAR->camera transform (the *extrinsic*), each reading becomes a
3D ray in the world. The extrinsic is found automatically from the scan itself: with the
wrong extrinsic, the same wall measured at different moments lands in different places,
so we pick the transform that makes the point cloud crispest (point-to-local-plane
distances between readings taken at different times), helped by ARKit's detected planes.

The rays are then turned into small "virtual camera" depth images so they go through the
same masked TSDF fusion and ghost removal as camera depth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from .fusion import DepthView
from .session import Session

log = logging.getLogger(__name__)


@dataclass
class LidarConfig:
    min_range: float = 0.05
    max_range: float = 12.0
    min_intensity: int = 20  # LD06/LD19 confidence; 1D sensors are filtered on the Pi
    max_pose_gap: float = 0.1  # s without a phone pose -> reading unusable (tracking lost)
    calib_samples: int = 6000
    view_size: int = 160  # virtual depth image width/height (px)
    view_window: float = 0.2  # s of 2D-LiDAR readings per group of virtual views
    view_window_1d: float = 1.0  # s of 1D-LiDAR readings per group
    splat_px: int = 2  # each reading covers a (2r+1)^2 disc in its virtual view
    mask_dilate_px: int = 3
    mask_max_dt: float = 0.3  # s: camera mask used for a reading must be this close in time


# --------------------------------------------------------------------------- poses


class PoseTrack:
    """Camera->world (OpenGL axes) interpolated at any time from the phone's pose stream."""

    def __init__(self, t: np.ndarray, M: np.ndarray):
        order = np.argsort(t)
        t, M = t[order], M[order]
        keep = np.concatenate([[True], np.diff(t) > 1e-6])
        self.t, M = t[keep], M[keep]
        self.slerp = Slerp(self.t, Rotation.from_matrix(M[:, :3, :3]))
        self.pos = M[:, :3, 3]

    @classmethod
    def from_session(cls, session: Session) -> "PoseTrack":
        if session.poses is not None:
            return cls(*session.poses)
        t = np.array([f.timestamp for f in session.frames])
        M = np.stack([f.pose_gl for f in session.frames])
        return cls(t, M)

    def valid(self, tq: np.ndarray, max_gap: float) -> np.ndarray:
        inside = (tq >= self.t[0]) & (tq <= self.t[-1])
        i = np.clip(np.searchsorted(self.t, tq), 1, len(self.t) - 1)
        gap = self.t[i] - self.t[i - 1]
        return inside & (gap <= max_gap)

    def __call__(self, tq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tq = np.clip(tq, self.t[0], self.t[-1])
        R = self.slerp(tq).as_matrix()
        p = np.stack([np.interp(tq, self.t, self.pos[:, k]) for k in range(3)], -1)
        return R, p


# --------------------------------------------------------------------------- geometry


def local_points(lidar: np.ndarray, kind: str) -> np.ndarray:
    """Points in the LiDAR frame: x = sensor front, y = left, z = up (2D scan plane = xy)."""
    r = lidar["range"].astype(np.float64)
    if kind == "1d":
        return np.stack([r, np.zeros_like(r), np.zeros_like(r)], -1)
    a = lidar["angle"].astype(np.float64)
    return np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], -1)


def to_world(track: PoseTrack, t: np.ndarray, local: np.ndarray, T_cam_lidar: np.ndarray):
    """Ray origins and end points in the world for readings at times t."""
    R, p = track(t)
    o_cam = T_cam_lidar[:3, 3]
    p_cam = local @ T_cam_lidar[:3, :3].T + o_cam
    origins = R @ o_cam + p
    points = np.einsum("nij,nj->ni", R, p_cam) + p
    return origins, points


def _T(x: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:]
    return T


def _plane_fit(nbr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(M, k, 3) neighbour sets -> centroid (M,3), unit normal (M,3)."""
    c = nbr.mean(1)
    d = nbr - c[:, None]
    cov = np.einsum("mki,mkj->mij", d, d)
    _, vec = np.linalg.eigh(cov)
    return c, vec[:, :, 0]


# --------------------------------------------------------------------------- calibration


class _Problem:
    """Crispness of the aggregated cloud as a function of the extrinsic."""

    def __init__(self, track, t, local, planes, k=8, radius=0.25, min_dt=0.4, cap=0.10,
                 max_offset=0.3, beam_prior=False):
        self.R, self.p = track(t)
        self.t, self.local = t, local
        self.planes = planes
        # the sensor is on the phone: the offset is bounded; a 1D beam roughly looks where
        # the camera looks (camera -z), which also fixes its unobservable roll
        self.bounds = ([-np.inf] * 3 + [-max_offset] * 3, [np.inf] * 3 + [max_offset] * 3)
        self.beam_prior = beam_prior
        self.k, self.radius, self.min_dt, self.cap = k, radius, min_dt, cap
        self.nbr = None

    def world(self, x):
        T = _T(x)
        p_cam = self.local @ T[:3, :3].T + T[:3, 3]
        return np.einsum("nij,nj->ni", self.R, p_cam) + self.p

    def associate(self, x) -> float:
        """Pick, for every point, neighbours measured at other moments; returns the
        fraction of points that have enough of them."""
        W = self.world(x)
        tree = cKDTree(W)
        d, idx = tree.query(W, k=4 * self.k, distance_upper_bound=self.radius)
        ok = np.isfinite(d)
        idx = np.where(ok, idx, 0)
        ok &= np.abs(self.t[idx] - self.t[:, None]) > self.min_dt
        nbr = np.full((len(W), self.k), -1)
        rows = []
        for i in range(len(W)):
            j = idx[i][ok[i]][: self.k]
            if len(j) >= 5:
                nbr[i, : len(j)] = j
                nbr[i, len(j):] = j[0]
                rows.append(i)
        self.rows = np.array(rows, dtype=np.int64)
        self.nbr = nbr[self.rows]
        self.plane_rows = []
        for pl in self.planes:  # ARKit plane anchors: points near and inside them
            M, n = pl["M"], pl["normal"]
            rel = W - M[:3, 3]
            dist = rel @ n
            u, v = rel @ M[:3, 0], rel @ M[:3, 2]
            inside = (np.abs(u) < pl["hx"]) & (np.abs(v) < pl["hz"]) & (np.abs(dist) < 0.15)
            self.plane_rows.append(np.nonzero(inside)[0])
        return len(self.rows) / max(len(W), 1)

    def residuals(self, x):
        W = self.world(x)
        c, n = _plane_fit(W[self.nbr])
        r = [np.einsum("ij,ij->i", W[self.rows] - c, n)]
        for pl, rows in zip(self.planes, self.plane_rows):
            if len(rows):
                r.append((W[rows] - pl["M"][:3, 3]) @ pl["normal"])
        if self.beam_prior:
            w = 0.02 * np.sqrt(max(sum(len(a) for a in r), 1))
            R = Rotation.from_rotvec(x[:3]).as_matrix()
            r.append(w * (R @ _BEAM_AXES - _NOMINAL_1D).ravel() / 0.35)  # ~20 deg tolerance
            r.append(w * x[3:] / 0.08)  # ~8 cm tolerance
        return np.concatenate(r)

    def score(self, x) -> float:
        """Lower is better; points without consistent neighbours count as bad."""
        frac = self.associate(x)
        res = np.abs(self.residuals(x)[: len(self.rows)]) if len(self.rows) else np.zeros(0)
        return float(np.minimum(res, self.cap).sum() + (len(self.t) - len(self.rows)) * self.cap) / len(self.t) \
            if frac > 0 else self.cap

    def refine(self, x, outer: int = 4) -> np.ndarray:
        for _ in range(outer):
            self.associate(x)
            if len(self.rows) < 50:
                break
            x = least_squares(self.residuals, x, loss="soft_l1", f_scale=0.02, max_nfev=60,
                              bounds=self.bounds).x
        return x


# LiDAR x (beam / front), y, z expressed in camera (OpenGL) axes for the usual mount:
# beam along the camera's view (-z), sensor top towards the phone's camera +y
_BEAM_AXES = np.eye(3)
_NOMINAL_1D = np.stack([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], 1).astype(float)


def _arkit_planes(session: Session) -> list[dict]:
    out = []
    for p in session.planes:
        M = np.array(p["transform"], dtype=np.float64).reshape(4, 4)
        c = np.array(p.get("center", [0, 0, 0]), dtype=np.float64)
        e = np.array(p.get("extent", [1, 0, 1]), dtype=np.float64)
        M = M.copy()
        M[:3, 3] = M[:3, :3] @ c + M[:3, 3]
        n = M[:3, 1] / np.linalg.norm(M[:3, 1])  # ARKit planes lie in their local x-z plane
        out.append({"M": M, "normal": n, "hx": e[0] / 2 + 0.05, "hz": e[2] / 2 + 0.05})
    return out


def calibrate(session: Session, track: PoseTrack, t: np.ndarray, local: np.ndarray,
              cfg: LidarConfig, seed: int = 0, kind: str = "2d") -> tuple[np.ndarray, dict]:
    """Find the LiDAR->camera transform. 2D sensors: no prior knowledge of the mount
    (all 24 axis-aligned orientations are tried). 1D sensors have far fewer readings, so
    they assume the beam points roughly where the camera looks."""
    rng = np.random.default_rng(seed)
    n = min(cfg.calib_samples, len(t))
    sel = np.sort(rng.choice(len(t), n, replace=False))
    planes = _arkit_planes(session)
    small = np.sort(rng.choice(sel, min(1500, n), replace=False))
    one_d = kind == "1d"
    opts = {"max_offset": 0.15, "beam_prior": True} if one_d else {}
    coarse = _Problem(track, t[small], local[small], planes, **opts)
    if one_d:  # nominal mount, tilted up to +-15 degrees
        tilts = [Rotation.from_euler("xy", [a, b], degrees=True) for a in (-15, 0, 15) for b in (-15, 0, 15)]
        rotations = [tl * Rotation.from_matrix(_NOMINAL_1D) for tl in tilts]
    else:  # every 90-degree orientation of the sensor relative to the phone (cube group)
        rotations = list(Rotation.create_group("O"))
    starts = []
    for R in rotations:
        x0 = np.concatenate([R.as_rotvec(), np.zeros(3)])
        x = coarse.refine(x0, outer=2)
        starts.append((coarse.score(x), x))
    starts.sort(key=lambda s: s[0])
    fine = _Problem(track, t[sel], local[sel], planes, **opts)
    best = None
    for _, x0 in starts[:3]:
        x = fine.refine(x0, outer=5)
        s = fine.score(x)
        if best is None or s < best[0]:
            best = (s, x)
    score, x = best
    fine.associate(x)
    res = np.abs(fine.residuals(x)[: len(fine.rows)])
    info = {
        "method": "auto (crispness" + (" + ARKit planes" if planes else "") + (" + 1D mount prior)" if one_d else ")"),
        "median_residual_m": float(np.median(res)) if len(res) else None,
        "consistent_fraction": float(len(fine.rows) / len(sel)),
        "score": score,
    }
    log.info("rig calibration: median residual %.1f mm, %.0f%% of points consistent",
             1000 * (info["median_residual_m"] or 0), 100 * info["consistent_fraction"])
    return _T(x), info


# --------------------------------------------------------------------------- the cloud


@dataclass
class LidarCloud:
    t: np.ndarray
    origins: np.ndarray
    points: np.ndarray
    keep: np.ndarray  # False: hit a person/animal or a transient object
    kind: str
    T_cam_lidar: np.ndarray
    calibration: dict
    normals: np.ndarray | None = None  # local surface normals (for exact splatting)

    def ensure_normals(self) -> np.ndarray:
        if self.normals is None:
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.12, max_nn=24))
            self.normals = np.asarray(pcd.normals).copy()
        return self.normals


def load_cloud(session: Session, cfg: LidarConfig) -> LidarCloud | None:
    if not session.has_lidar:
        return None
    kind = session.rig.get("kind", "2d")
    lid = session.lidar
    track = PoseTrack.from_session(session)
    ok = (lid["range"] >= cfg.min_range) & (lid["range"] <= cfg.max_range)
    if kind == "2d":
        ok &= lid["intensity"] >= cfg.min_intensity
    ok &= track.valid(lid["t"], cfg.max_pose_gap)
    lid = lid[ok]
    if len(lid) < 1000:
        log.warning("only %d usable LiDAR readings - ignoring the rig", len(lid))
        return None
    local = local_points(lid, kind)
    t = lid["t"].astype(np.float64)
    if session.rig.get("extrinsic"):
        T = np.array(session.rig["extrinsic"], dtype=np.float64).reshape(4, 4)
        info = {"method": "given in rig.json"}
    else:
        T, info = calibrate(session, track, t, local, cfg, kind=kind)
    origins, points = to_world(track, t, local, T)
    log.info("rig: %d LiDAR readings (%s)", len(t), kind)
    return LidarCloud(t, origins, points, np.ones(len(t), bool), kind, T, info)


def mask_living(session: Session, cloud: LidarCloud, cfg: LidarConfig,
                masks: dict | None = None) -> int:
    """Drop readings that hit a person/animal in the camera frame closest in time."""
    times = np.array([f.timestamp for f in session.frames])
    i = np.clip(np.searchsorted(times, cloud.t), 1, len(times) - 1)
    i -= (cloud.t - times[i - 1]) < (times[i] - cloud.t)
    near = np.abs(times[i] - cloud.t) <= cfg.mask_max_dt
    dropped = 0
    k = np.ones((2 * cfg.mask_dilate_px + 1,) * 2, np.uint8)
    for fi in np.unique(i[near]):
        fr = session.frames[fi]
        m = masks.get(fr.index) if masks else None
        if m is None:
            m = session.load_mask(fr)
        if m is None or not m.any():
            continue
        m = cv2.dilate(m.astype(np.uint8), k) > 0
        sel = np.nonzero(near & (i == fi) & cloud.keep)[0]
        E = fr.extrinsic_cv
        pc = cloud.points[sel] @ E[:3, :3].T + E[:3, 3]
        z = pc[:, 2]
        K = fr.K_depth
        w, h = fr.depth_size
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.floor(K[0, 0] * pc[:, 0] / z + K[0, 2])
            v = np.floor(K[1, 1] * pc[:, 1] / z + K[1, 2])
        inb = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        hit = np.zeros(len(sel), bool)
        hit[inb] = m[v[inb].astype(np.int64), u[inb].astype(np.int64)]
        cloud.keep[sel[hit]] = False
        dropped += int(hit.sum())
    log.info("rig: %d readings hit people/animals and were dropped", dropped)
    return dropped


def build_views(cloud: LidarCloud, cfg: LidarConfig) -> list[DepthView]:
    """Render the kept readings as small depth images from virtual pinhole cameras.

    Readings are grouped by time window (``cfg.view_window``) and, within a window, by
    viewing direction (world-fixed sectors), so each group fits one camera. Every reading
    is splatted as a small disc (``cfg.splat_px``) so neighbouring scan lines join into
    surfaces in the TSDF; the TSDF still only trusts surfaces seen repeatedly."""
    idx = np.nonzero(cloud.keep)[0]
    if len(idx) == 0:
        return []
    window = cfg.view_window if cloud.kind == "2d" else cfg.view_window_1d
    tbin = np.floor((cloud.t[idx] - cloud.t[idx[0]]) / window).astype(np.int64)
    d_all = cloud.points[idx] - cloud.origins[idx]
    dn_all = d_all / np.linalg.norm(d_all, axis=1, keepdims=True)
    az = np.floor((np.arctan2(dn_all[:, 0], dn_all[:, 2]) + np.pi) / (np.pi / 2)).astype(np.int64) % 4
    el = np.where(dn_all[:, 1] > 0.75, 1, np.where(dn_all[:, 1] < -0.75, 2, 0))
    sector = np.where(el > 0, 4 + el, az)
    key = tbin * 8 + sector
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    cuts = np.concatenate([[0], np.nonzero(np.diff(key_sorted))[0] + 1, [len(order)]])
    views: list[DepthView] = []
    S = cfg.view_size
    r = cfg.splat_px
    offsets = [(du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1) if du * du + dv * dv <= r * r + r]
    normals = cloud.ensure_normals()
    for a, b in zip(cuts[:-1], cuts[1:]):
        sel = idx[order[a:b]]
        if len(sel) < 3:
            continue
        o = cloud.origins[sel].mean(0)
        d = cloud.points[sel] - o
        dn = d / np.linalg.norm(d, axis=1, keepdims=True)
        z = dn.mean(0)
        if np.linalg.norm(z) < 1e-6:
            continue
        z /= np.linalg.norm(z)
        half = min(max(np.degrees(np.arccos(np.clip(dn @ z, -1, 1))).max() + 2.0, 5.0), 80.0)
        helper = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([x, y, z], 1)  # OpenCV camera axes
        pose[:3, 3] = o
        f = (S / 2) / np.tan(np.radians(half))
        K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1.0]])
        pc = d @ pose[:3, :3]
        zc = pc[:, 2]
        good = zc > 0.05
        u = np.floor(f * pc[good, 0] / zc[good] + S / 2).astype(np.int64)
        v = np.floor(f * pc[good, 1] / zc[good] + S / 2).astype(np.int64)
        zc = zc[good]
        inb = (u >= 0) & (u < S) & (v >= 0) & (v < S)
        u, v, zc = u[inb], v[inb], zc[inb]
        if len(zc) < 3:
            continue
        # Splat each return over a small disc. Each covered pixel gets the depth at which
        # ITS OWN ray meets the return's local surface plane - exact for flat surfaces,
        # unlike copying the return's depth (which pulls slanted walls towards the sensor).
        n_cam = (normals[sel] @ pose[:3, :3])[good][inb]
        p_cam = pc[good][inb]
        inv = np.zeros((S, S), np.float32)
        for du, dv in offsets:
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < S) & (vv >= 0) & (vv < S)
            ray = np.stack([(uu + 0.5 - S / 2) / f, (vv + 0.5 - S / 2) / f, np.ones_like(zc)], -1)
            denom = np.einsum("ij,ij->i", n_cam, ray)
            num = np.einsum("ij,ij->i", n_cam, p_cam)
            ray_n = np.linalg.norm(ray, axis=1)
            ok &= np.abs(denom) > 0.25 * ray_n  # skip pixels that graze the surface
            with np.errstate(divide="ignore", invalid="ignore"):
                zz = num / denom
            ok &= (zz > 0.05) & (np.abs(zz - zc) < 0.1 + 0.05 * zc)
            np.maximum.at(inv, (vv[ok], uu[ok]), (1.0 / zz[ok]).astype(np.float32))
        depth = np.where(inv > 0, 1.0 / np.maximum(inv, 1e-6), 0).astype(np.float32)
        views.append(DepthView(key=f"L{len(views)}", K=K, pose_cv=pose, depth=depth))
    log.info("rig: %d LiDAR views", len(views))
    return views
