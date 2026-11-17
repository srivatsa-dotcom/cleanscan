"""External depth module (MaixSense A010 on an ESP32-S3, or Arducam ToF on a Pi; module/).

The module streams small, noisy depth images (A010: 100 x 100, 0.2-2.5 m) with phone-clock
timestamps, recorded by the app as ``tof/*.png`` + ``tof.jsonl`` (docs/capture_format.md).
The phone itself has no depth (e.g. iPhone 11). This module turns them into depth maps for
the phone's camera frames, which then go through the normal masked TSDF fusion:

1. calibration: the module -> camera transform is found from the scan itself, exactly as for
   the LiDAR rig (lidar.calibrate: the extrinsic that makes the point cloud crispest, all 24
   axis-aligned mountings tried, helped by ARKit planes). ToF pixels are just dense rays.
2. per camera frame, the ToF frames closest in time are projected into the camera's depth
   image (pose interpolated at each ToF timestamp, so motion between them is compensated)
3. RGB-guided upsampling: a joint-bilateral normalised convolution fills the gaps between
   ToF samples using the colour image as the guide (no bleeding across colour edges); the
   ToF samples themselves are kept as metric anchors
4. outside the module's reach (beyond ~2.5 m) the camera stereo depth fills in, if enabled
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .session import Session

log = logging.getLogger(__name__)


@dataclass
class DepthModuleConfig:
    pixel_step: int = 3  # use every n-th ToF pixel for calibration
    min_range: float = 0.15
    max_range: float = 3.0
    edge_threshold: float = 0.05  # relative depth jump -> flying pixel, dropped
    frames_per_view: int = 3  # ToF frames (nearest in time) projected into each camera frame
    max_dt: float = 0.15  # s
    radius_px: int = 2  # upsampling window radius (depth-image pixels); larger smears edges
    sigma_color: float = 10.0  # 0..255 intensity units
    min_weight: float = 0.35  # less support than this -> no depth
    edge_spread: float = 0.03  # relative depth spread of the neighbours that marks a depth edge
    stereo_fill: bool = True  # stereo depth where the module sees nothing
    noise: tuple = (0.004, 0.006)  # 1-sigma (abs m, relative), MaixSense A010 class sensor
    calibrate_samples: int = 8000


@dataclass
class ToFFrame:
    t: float
    path: Path
    K: np.ndarray
    size: tuple[int, int]


@dataclass
class DepthModuleResult:
    T_cam_module: np.ndarray  # module (OpenCV axes) -> phone camera (OpenGL axes)
    calibration: dict = field(default_factory=dict)
    coverage: float = 0.0


def has_depth_module(session: Session) -> bool:
    return (session.root / "tof.jsonl").exists()


def load_tof(session: Session) -> list[ToFFrame]:
    out = []
    for line in (session.root / "tof.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        fx, fy, cx, cy = r["intrinsics"]
        out.append(ToFFrame(float(r["t"]), session.root / r["depth"],
                            np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]), tuple(r["size"])))
    out.sort(key=lambda f: f.t)
    return out


def module_info(session: Session) -> dict:
    p = session.root / "depth_module.json"
    return json.loads(p.read_text()) if p.exists() else {}


def read_tof(f: ToFFrame, cfg: DepthModuleConfig) -> np.ndarray:
    d = cv2.imread(str(f.path), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    d[(d < cfg.min_range) | (d > cfg.max_range)] = 0
    # flying pixels (mixed foreground/background on edges) are dropped
    valid = d > 0
    k = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(d, k) - cv2.erode(np.where(valid, d, 99).astype(np.float32), k)
    d[(spread > cfg.edge_threshold * np.maximum(d, 0.1)) & valid] = 0
    return d


def tof_points(f: ToFFrame, d: np.ndarray, step: int = 1) -> np.ndarray:
    """Points in the module frame (OpenCV axes: x right, y down, z forward)."""
    v, u = np.nonzero(d[::step, ::step] > 0)
    v, u = v * step, u * step
    z = d[v, u]
    K = f.K
    return np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)


# the clip holds the ToF lens next to the phone camera, looking the same way:
# module (OpenCV: z forward, y down) -> phone camera (OpenGL: -z forward, y up)
NOMINAL_MOUNT = np.diag([1.0, -1.0, -1.0])


def calibrate_module(session: Session, frames: list[ToFFrame], cfg: DepthModuleConfig,
                     seed: int = 0, extra_sessions: list[Session] = ()) -> tuple[np.ndarray, dict]:
    """module -> camera transform by the rig's crispness objective (lidar._Problem), started
    from the clip's nominal mounting and softly held near it (~20 deg, ~8 cm).

    Without that prior the problem is ambiguous for some walks: when the phone turns on a
    circle facing outwards, a module rotated about the circle's axis gives an equally crisp,
    just rotated, cloud. ARKit planes would resolve it too, but Android does not record them.

    ``extra_sessions`` (the guided object captures: same clock, same world frame) add
    viewpoints; walking around an object makes the module's offset well observable."""
    from scipy.spatial.transform import Rotation

    from .lidar import PoseTrack, _arkit_planes, _Problem, _T

    class _ToFProblem(_Problem):
        def residuals(self, x):
            r = super().residuals(x)
            w = 0.02 * np.sqrt(max(len(r), 1))
            R = Rotation.from_rotvec(x[:3]).as_matrix()
            # rotation held near the clip's mounting; the offset only bounded (+-12 cm)
            return np.concatenate([r, w * (R - NOMINAL_MOUNT).ravel() / 0.35, w * x[3:] / 1.0])

    rng = np.random.default_rng(seed)
    sessions = [(session, frames)] + [(s, load_tof(s)) for s in extra_sessions if has_depth_module(s)]
    pose_t, pose_m = [], []
    for s, _ in sessions:
        tr = PoseTrack.from_session(s)
        pose_t.append(tr.t)
        pose_m.append(np.concatenate([tr.slerp(tr.t).as_matrix(), tr.pos[:, :, None]], 2))
    Ms = np.concatenate(pose_m)
    full = np.zeros((len(Ms), 4, 4))
    full[:, :3, :4] = Ms
    full[:, 3, 3] = 1
    track = PoseTrack(np.concatenate(pose_t), full)  # one clock: sessions do not overlap in time
    ts, pts = [], []
    for _, frs in sessions:
        for f in frs:
            p = tof_points(f, read_tof(f, cfg), cfg.pixel_step)
            pts.append(p)
            ts.append(np.full(len(p), f.t))
    t = np.concatenate(ts)
    local = np.concatenate(pts)
    ok = track.valid(t, 0.1)
    t, local = t[ok], local[ok]
    n = min(cfg.calibrate_samples, len(t))
    sel = np.sort(rng.choice(len(t), n, replace=False))
    small = np.sort(rng.choice(sel, min(2000, n), replace=False))
    planes = _arkit_planes(session)
    coarse = _ToFProblem(track, t[small], local[small], planes, max_offset=0.12)
    starts = []
    for a in (-10, 0, 10):
        for b in (-10, 0, 10):
            R = Rotation.from_euler("xy", [a, b], degrees=True) * Rotation.from_matrix(NOMINAL_MOUNT)
            x = coarse.refine(np.concatenate([R.as_rotvec(), np.zeros(3)]), outer=2)
            starts.append((coarse.score(x), x))
    starts.sort(key=lambda s: s[0])
    fine = _ToFProblem(track, t[sel], local[sel], planes, max_offset=0.12)
    best = None
    for _, x0 in starts[:3]:
        x = fine.refine(x0, outer=5)
        sc = fine.score(x)
        if best is None or sc < best[0]:
            best = (sc, x)
    score, x = best
    fine.associate(x)
    res = np.abs(fine.residuals(x)[: len(fine.rows)])
    T = _T(x)
    off = float(np.degrees(Rotation.from_matrix(NOMINAL_MOUNT.T @ T[:3, :3]).magnitude()))
    info = {"method": "auto (crispness of ToF points" + (" + ARKit planes" if planes else "") + ", clip prior)",
            "median_residual_m": float(np.median(res)) if len(res) else None,
            "consistent_fraction": float(len(fine.rows) / len(sel)), "score": score,
            "deg_from_nominal_mount": round(off, 2), "offset_m": np.round(T[:3, 3], 4).tolist(),
            "sessions": len(sessions)}
    log.info("depth module calibration: median residual %.1f mm, %.1f deg from the nominal mount",
             1000 * (info["median_residual_m"] or 0), off)
    return T, info


def _interp_pose(track, t: float) -> np.ndarray:
    R, p = track(np.array([t]))
    M = np.eye(4)
    M[:3, :3] = R[0]
    M[:3, 3] = p[0]
    return M


def guided_upsample(sparse: np.ndarray, guide: np.ndarray, cfg: DepthModuleConfig) -> np.ndarray:
    """Joint-bilateral normalised convolution of sparse depth guided by a grey image.

    Where the nearby samples disagree (a depth edge: object in front of a wall), averaging
    would invent depths in between that belong to neither surface; there the single sample
    that matches best (closest in position and colour) is taken instead. Pixels that have a
    ToF sample keep it."""
    h, w = sparse.shape
    r = cfg.radius_px
    g = guide.astype(np.float32)
    has = (sparse > 0).astype(np.float32)
    num = np.zeros((h, w), np.float32)
    num2 = np.zeros((h, w), np.float32)
    den = np.zeros((h, w), np.float32)
    best_w = np.zeros((h, w), np.float32)
    best_d = np.zeros((h, w), np.float32)
    pad_d = np.pad(sparse, r)
    pad_m = np.pad(has, r)
    pad_g = np.pad(g, r, mode="edge")
    ss = (r / 1.5) ** 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ws = np.exp(-(dx * dx + dy * dy) / (2 * ss))
            sd = pad_d[r + dy:r + dy + h, r + dx:r + dx + w]
            sm = pad_m[r + dy:r + dy + h, r + dx:r + dx + w]
            sg = pad_g[r + dy:r + dy + h, r + dx:r + dx + w]
            wc = np.exp(-((g - sg) ** 2) / (2 * cfg.sigma_color ** 2))
            wgt = ws * wc * sm
            num += wgt * sd
            num2 += wgt * sd * sd
            den += wgt
            better = wgt > best_w
            best_w = np.where(better, wgt, best_w)
            best_d = np.where(better, sd, best_d)
    ok = den > cfg.min_weight * 0.25
    mean = np.where(ok, num / np.maximum(den, 1e-6), 0)
    var = np.where(ok, num2 / np.maximum(den, 1e-6) - mean * mean, 0)
    edge = np.sqrt(np.maximum(var, 0)) > cfg.edge_spread * np.maximum(mean, 0.1)
    out = np.where(edge, best_d, mean)
    out[~ok] = 0
    out[sparse > 0] = sparse[sparse > 0]
    return out.astype(np.float32)


def estimate_session_depth(session: Session, cfg: DepthModuleConfig | None = None,
                           T_cam_module: np.ndarray | None = None, stereo_cfg=None) -> DepthModuleResult:
    """Depth for every camera frame from the module (+ stereo fill). Writes depth_tof/ and
    points the frames at it, like stereo.estimate_session_depth."""
    from . import stereo
    from .lidar import PoseTrack
    from .session import GL_TO_CV

    cfg = cfg or DepthModuleConfig()
    frames = load_tof(session)
    if not frames:
        raise RuntimeError("tof.jsonl lists no frames")
    info = {}
    if T_cam_module is None:
        known = module_info(session).get("extrinsic")
        if known is not None:
            T_cam_module = np.array(known, float).reshape(4, 4)
            info = {"method": "given in depth_module.json"}
        else:
            T_cam_module, info = calibrate_module(session, frames, cfg)
    else:
        info = {"method": "from the room pass calibration"}
    track = PoseTrack.from_session(session)
    tof_t = np.array([f.t for f in frames])
    if cfg.stereo_fill:
        stereo_cfg = stereo_cfg or stereo.StereoConfig()
    out_dir = session.root / "depth_tof"
    out_dir.mkdir(exist_ok=True)
    cache = {}
    coverage = []
    for i, fr in enumerate(session.frames):
        w, h = fr.depth_size
        K = fr.K_depth
        E = np.linalg.inv(fr.pose_gl @ GL_TO_CV)  # world -> camera (OpenCV)
        sparse = np.zeros((h, w), np.float32)
        near = np.argsort(np.abs(tof_t - fr.timestamp))[: cfg.frames_per_view]
        for j in near:
            if abs(tof_t[j] - fr.timestamp) > cfg.max_dt:
                continue
            f = frames[j]
            if j not in cache:
                cache[j] = tof_points(f, read_tof(f, cfg))
            p = cache[j]
            if not len(p):
                continue
            # module -> camera (GL) -> world at the ToF time -> this frame's camera
            M = _interp_pose(track, f.t) @ T_cam_module
            pw = p @ M[:3, :3].T + M[:3, 3]
            pc = pw @ E[:3, :3].T + E[:3, 3]
            z = pc[:, 2]
            ok = z > 0.05
            u = np.floor(K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]).astype(int)
            v = np.floor(K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]).astype(int)
            zz = z[ok]
            inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            u, v, zz = u[inside], v[inside], zz[inside]
            # z-buffer: nearest sample wins (the frames are close in time)
            order = np.argsort(-zz)
            cur = sparse[v[order], u[order]]
            sparse[v[order], u[order]] = np.where((cur == 0) | (zz[order] < cur), zz[order], cur)
        grey = cv2.cvtColor(cv2.resize(session.load_color(fr), (w, h), interpolation=cv2.INTER_AREA),
                            cv2.COLOR_RGB2GRAY)
        dense = guided_upsample(sparse, grey, cfg)
        if cfg.stereo_fill and (dense == 0).mean() > 0.05:
            stereo_depth = stereo.estimate_frame_depth(session, i, stereo_cfg)
            fill = (dense == 0) & (stereo_depth > 0)
            dense[fill] = stereo_depth[fill]
        coverage.append(float((dense > 0).mean()))
        path = out_dir / f"{fr.index:06d}.png"
        cv2.imwrite(str(path), np.clip(dense * 1000, 0, 65535).astype(np.uint16))
        fr.depth_path = path
        if (i + 1) % 50 == 0 or i + 1 == len(session.frames):
            log.info("depth module: %d/%d frames", i + 1, len(session.frames))
    session.depth_unit = 0.001
    session.depth_noise = cfg.noise
    res = DepthModuleResult(T_cam_module, info, float(np.mean(coverage)) if coverage else 0.0)
    log.info("depth module: %.0f%% of pixels have depth on average", 100 * res.coverage)
    return res
