"""Ghost removal: catch moving people/animals that the segmenter missed.

After a first fusion pass every triangle is projected into every frame:

  * measured depth clearly *behind* the triangle  -> that frame saw free space
    where the triangle is ("free" vote)
  * measured depth at the triangle                -> "occupied" vote
  * measured depth in front of it                 -> occluded, no information

Static geometry is occupied whenever it is visible. A surface that is often
seen through was transient - a person walking by, a pet, a door being opened.
The pixels that observed it are masked and the scene is fused again, so the
background is rebuilt from the views in which it was visible. Projection (not
rendering) is used on purpose: a person walking leaves a whole "curtain" of
ghost surfaces whose back layers would never be visible in a render.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import open3d as o3d

from .fusion import FusionConfig, prepare_depth
from .session import Session

log = logging.getLogger(__name__)


def _project(points: np.ndarray, fr) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w2c = fr.extrinsic_cv
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    K = fr.K_depth
    w, h = fr.depth_size
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(K[0, 0] * pc[:, 0] / z + K[0, 2])
        v = np.floor(K[1, 1] * pc[:, 1] / z + K[1, 2])
    inside = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx = np.nonzero(inside)[0]
    return idx, u[idx].astype(np.int64), v[idx].astype(np.int64), z[idx]


def _stable_depth(depth: np.ndarray) -> np.ndarray:
    """Pixels away from depth discontinuities (avoids silhouette false votes)."""
    valid = depth > 0
    k = np.ones((5, 5), np.uint8)
    hi = cv2.dilate(np.where(valid, depth, 0).astype(np.float32), k)
    lo = cv2.erode(np.where(valid, depth, 1e6).astype(np.float32), k)
    return valid & ((hi - lo) < 0.1 * np.maximum(depth, 0.3))


def detect_ghost_masks(session: Session, mesh: o3d.geometry.TriangleMesh, cfg: FusionConfig,
                       masks: dict[int, np.ndarray] | None = None,
                       min_votes: int = 3, ratio: float = 0.5, cloud=None,
                       ) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Returns (per-frame extra masks at depth resolution, transient flag per triangle).

    With an external-LiDAR ``cloud`` (lidar.LidarCloud), every kept ray also votes, and
    readings that measured a transient surface get ``cloud.keep`` set to False."""
    n_tri = len(mesh.triangles)
    if n_tri == 0:
        return {}, np.zeros(0, bool)
    V = np.asarray(mesh.vertices)
    C = V[np.asarray(mesh.triangles)].mean(1)
    free = np.zeros(n_tri, np.int32)
    occ = np.zeros(n_tri, np.int32)

    depths = {}
    for fr in session.frames:
        if fr.depth_path is None:
            continue  # LiDAR-rig scan fused without camera depth
        depth, _ = prepare_depth(session, fr, cfg,
                                 mask_override=None if masks is None else masks.get(fr.index))
        depth[~_stable_depth(depth)] = 0
        depths[fr.index] = depth
        idx, u, v, z = _project(C, fr)
        d = depth[v, u]
        tau = np.maximum(2.0 * cfg.sdf_trunc, 0.03 * z)
        good = d > 0
        free[idx[good & (d > z + tau)]] += 1
        occ[idx[good & (np.abs(d - z) <= tau)]] += 1

    ray_prim = ray_occ = ray_idx = None
    if cloud is not None:
        ray_idx, ray_prim, ray_occ = _lidar_votes(mesh, cloud, cfg, free, occ)

    transient = (free >= min_votes) & (free > ratio * occ)
    log.info("ghost removal: %d / %d triangles flagged transient", int(transient.sum()), n_tri)
    if cloud is not None and transient.any():
        drop = ray_occ & transient[np.clip(ray_prim, 0, None)] & (ray_prim >= 0)
        cloud.keep[ray_idx[drop]] = False
        log.info("ghost removal: %d LiDAR readings measured transient objects", int(drop.sum()))

    extra = {}
    if transient.any():
        Ct = C[transient]
        k = np.ones((3, 3), np.uint8)
        for fr in session.frames:
            if fr.index not in depths:
                continue
            depth = depths[fr.index]
            idx, u, v, z = _project(Ct, fr)
            d = depth[v, u]
            hit = (d > 0) & (np.abs(d - z) <= np.maximum(2.0 * cfg.sdf_trunc, 0.03 * z))
            if hit.any():
                w, h = fr.depth_size
                m = np.zeros((h, w), np.uint8)
                m[v[hit], u[hit]] = 1
                extra[fr.index] = cv2.dilate(m, k) > 0
    return extra, transient


def _lidar_votes(mesh, cloud, cfg: FusionConfig, free: np.ndarray, occ: np.ndarray):
    """Trace every kept LiDAR ray; a surface in front of the measured point was seen
    through (free vote), the surface at the measured point is confirmed (occupied vote)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    idx = np.nonzero(cloud.keep)[0]
    d = cloud.points[idx] - cloud.origins[idx]
    L = np.linalg.norm(d, axis=1)
    rays = np.concatenate([cloud.origins[idx], d / L[:, None]], 1).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = ans["t_hit"].numpy()
    prim = ans["primitive_ids"].numpy().astype(np.int64)
    hit = np.isfinite(t_hit)
    prim[~hit] = -1
    tau = np.maximum(2.0 * cfg.sdf_trunc, 0.02 * L)
    through = hit & (t_hit < L - tau)
    confirm = hit & (np.abs(t_hit - L) <= tau)
    free += np.bincount(prim[through], minlength=len(free)).astype(free.dtype)
    occ += np.bincount(prim[confirm], minlength=len(occ)).astype(occ.dtype)
    return idx, prim, confirm
