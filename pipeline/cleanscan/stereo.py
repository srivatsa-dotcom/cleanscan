"""Depth for camera-only captures (phones without LiDAR/ToF, e.g. iPhone 11).

ARKit/ARCore give a metric pose for every frame, so two frames a few centimetres
apart form a calibrated stereo pair. For each frame we pick up to two neighbours
with a mostly sideways baseline, rectify, run semi-global matching, and project
the triangulated points back into the frame. The result is metric depth at the
session's depth resolution, written next to the capture (``depth_est/``) and then
fused exactly like sensor depth. Texture-less surfaces give holes; the TSDF fills
them from other views where possible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .session import Frame, Session

log = logging.getLogger(__name__)


@dataclass
class StereoConfig:
    work_width: int = 640  # colour images are matched at this width
    min_baseline: float = 0.05  # metres
    max_baseline: float = 0.30
    target_baseline: float = 0.12
    max_angle_deg: float = 15.0
    min_lateral: float = 0.6  # |sideways part of the baseline| / |baseline|
    search_window: int = 40  # frames before/after considered as partners
    num_disparities: int = 128
    block_size: int = 5
    agree_rel: float = 0.05  # two pairs must agree within 5 % where both see a pixel


def _rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def pick_partners(frames: list[Frame], i: int, cfg: StereoConfig, k: int = 2) -> list[int]:
    """Neighbour frames giving a good, mostly sideways stereo baseline to frame i."""
    Ei = frames[i].extrinsic_cv
    cands = []
    for j in range(max(0, i - cfg.search_window), min(len(frames), i + cfg.search_window + 1)):
        if j == i:
            continue
        rel = frames[j].extrinsic_cv @ np.linalg.inv(Ei)  # camera i -> camera j
        T = rel[:3, 3]
        b = float(np.linalg.norm(T))
        if not cfg.min_baseline <= b <= cfg.max_baseline:
            continue
        if _rotation_angle_deg(rel[:3, :3]) > cfg.max_angle_deg:
            continue
        if np.hypot(T[0], T[1]) / b < cfg.min_lateral:
            continue
        cands.append((abs(b - cfg.target_baseline), j, np.sign(T[0])))
    cands.sort()
    # prefer one partner on each side: errors on the two sides are independent
    out, sides = [], set()
    for _, j, side in cands:
        if side not in sides:
            out.append(j)
            sides.add(side)
        if len(out) == k:
            break
    return out


def _gray(session: Session, fr: Frame, width: int) -> tuple[np.ndarray, np.ndarray]:
    rgb = session.load_color(fr)
    s = width / rgb.shape[1]
    img = cv2.resize(rgb, (width, int(round(rgb.shape[0] * s))), interpolation=cv2.INTER_AREA)
    K = fr.K_color.copy()
    K[:2] *= s
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), K


def pair_points(session: Session, a: Frame, b: Frame, cfg: StereoConfig) -> np.ndarray:
    """World points triangulated from the rectified pair (a, b); empty if unusable."""
    ga, Ka = _gray(session, a, cfg.work_width)
    gb, Kb = _gray(session, b, cfg.work_width)
    h, w = ga.shape
    rel = b.extrinsic_cv @ np.linalg.inv(a.extrinsic_cv)
    R, T = rel[:3, :3], rel[:3, 3]
    dist = np.zeros(5)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(Ka, dist, Kb, dist, (w, h), R.copy(), T.reshape(3, 1).copy(),
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if P2[0, 3] > 0:  # b is left of a: swap so the left image is always first
        return pair_points(session, b, a, cfg)
    m1 = cv2.initUndistortRectifyMap(Ka, dist, R1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(Kb, dist, R2, P2, (w, h), cv2.CV_32FC1)
    ra = cv2.remap(ga, m1[0], m1[1], cv2.INTER_LINEAR)
    rb = cv2.remap(gb, m2[0], m2[1], cv2.INTER_LINEAR)
    bs = cfg.block_size
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=cfg.num_disparities, blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs, disp12MaxDiff=1, uniquenessRatio=12,
        speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(ra, rb).astype(np.float32) / 16.0
    valid = (disp > 1.0) & (cv2.remap(np.ones_like(ga), m1[0], m1[1], cv2.INTER_NEAREST) > 0)
    if valid.sum() < 100:
        return np.zeros((0, 3))
    pts = cv2.reprojectImageTo3D(disp, Q)[valid]  # rectified camera a
    pts = pts[np.isfinite(pts).all(1) & (pts[:, 2] > 0.1)]
    pts_cam = pts @ R1  # R1^T applied to row vectors: rectified -> camera a
    pose = a.pose_cv
    return pts_cam @ pose[:3, :3].T + pose[:3, 3]


def project_depth(points_w: np.ndarray, fr: Frame) -> np.ndarray:
    """Z-buffer world points into the frame's depth image (0 = no data)."""
    w, h = fr.depth_size
    depth = np.zeros((h, w), np.float32)
    if len(points_w) == 0:
        return depth
    E = fr.extrinsic_cv
    pc = points_w @ E[:3, :3].T + E[:3, 3]
    z = pc[:, 2]
    K = fr.K_depth
    ok = z > 0.1
    u = np.floor(K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]).astype(np.int64)
    v = np.floor(K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]).astype(np.int64)
    z = z[ok]
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inb], v[inb], z[inb]
    order = np.argsort(-z)  # far first, near overwrites
    depth[v[order], u[order]] = z[order]
    return depth


def estimate_frame_depth(session: Session, i: int, cfg: StereoConfig) -> np.ndarray:
    frames = session.frames
    fr = frames[i]
    maps = [project_depth(pair_points(session, fr, frames[j], cfg), fr) for j in pick_partners(frames, i, cfg)]
    if not maps:
        w, h = fr.depth_size
        return np.zeros((h, w), np.float32)
    if len(maps) == 1:
        return maps[0]
    a, b = maps
    both = (a > 0) & (b > 0)
    out = np.where(a > 0, a, b)
    agree = both & (np.abs(a - b) <= cfg.agree_rel * np.maximum(a, b))
    out[both & ~agree] = 0
    out[agree] = 0.5 * (a[agree] + b[agree])
    return out


def estimate_session_depth(session: Session, cfg: StereoConfig | None = None) -> None:
    """Compute (or reuse cached) stereo depth for every frame and point the session at it."""
    cfg = cfg or StereoConfig()
    out_dir = session.root / "depth_est"
    out_dir.mkdir(exist_ok=True)
    n = len(session.frames)
    coverage = []
    for i, fr in enumerate(session.frames):
        path = out_dir / f"{fr.index:06d}.png"
        if not path.exists():
            d = estimate_frame_depth(session, i, cfg)
            cv2.imwrite(str(path), np.clip(d * 1000, 0, 65535).astype(np.uint16))
            coverage.append(float((d > 0).mean()))
        fr.depth_path = path
        if (i + 1) % 50 == 0 or i + 1 == n:
            log.info("stereo depth %d/%d frames", i + 1, n)
    session.depth_unit = 0.001
    if coverage:
        log.info("stereo depth: %.0f%% of pixels have depth on average", 100 * np.mean(coverage))
