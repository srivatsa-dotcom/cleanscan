"""Space carving: a closed solid from depth images (and optional foreground masks).

A voxel is *empty* when a view sees through it (it lies in front of the measured depth), or
when a foreground mask says the pixel shows something else and the voxel is not behind that
surface. Voxels never proven empty stay solid, so the result is closed by construction:
unseen backs and bottoms are filled, which is what physics needs (a watertight mesh).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation, gaussian_filter, label as cc_label


@dataclass
class View:
    K: np.ndarray  # 3x3 at depth resolution
    E: np.ndarray  # 4x4 world -> camera (OpenCV axes)
    depth: np.ndarray  # (h, w) metres, 0 = no data
    mask: np.ndarray | None = None  # (h, w) bool foreground, or None


def session_views(session, frames=None, motion: np.ndarray | None = None, depth_fn=None, mask_fn=None,
                  step: int = 1) -> list[View]:
    """Views of a capture session. ``motion`` (4x4) re-expresses them in a moved frame
    (world point x is seen where the camera saw ``motion @ x``), e.g. an opened door."""
    out = []
    for fr in (frames or session.frames)[::step]:
        if fr.depth_path is None:
            continue
        d = session.load_depth(fr) if depth_fn is None else depth_fn(fr)
        E = fr.extrinsic_cv if motion is None else fr.extrinsic_cv @ motion
        out.append(View(fr.K_depth, E, d, None if mask_fn is None else mask_fn(fr, d)))
    return out


def grid_centers(lo, hi, voxel) -> tuple[np.ndarray, tuple]:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    shape = tuple(int(n) for n in np.ceil((hi - lo) / voxel).astype(int) + 1)
    gi = [lo[k] + voxel * (np.arange(shape[k]) + 0.5) for k in range(3)]
    X, Y, Z = np.meshgrid(*gi, indexing="ij")
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1), shape


def carve(views: list[View], lo, hi, voxel: float, grid_to_world: np.ndarray | None = None,
          depth_tol: float = 0.012, min_free_votes: int = 2, mask_dilate_px: int = 1,
          require_seen: bool = True) -> np.ndarray:
    """Occupancy (bool grid, shape from ``grid_centers``) of the box [lo, hi] in the grid frame."""
    C, shape = grid_centers(lo, hi, voxel)
    Cw = C if grid_to_world is None else C @ grid_to_world[:3, :3].T + grid_to_world[:3, 3]
    free = np.zeros(len(C), np.int32)
    seen = np.zeros(len(C), np.int32)
    for v in views:
        h, w = v.depth.shape
        pc = Cw @ v.E[:3, :3].T + v.E[:3, 3]
        z = pc[:, 2]
        ok = z > 0.05
        u = np.full(len(C), -1, np.int64)
        r = np.full(len(C), -1, np.int64)
        u[ok] = np.floor(v.K[0, 0] * pc[ok, 0] / z[ok] + v.K[0, 2]).astype(np.int64)
        r[ok] = np.floor(v.K[1, 1] * pc[ok, 1] / z[ok] + v.K[1, 2]).astype(np.int64)
        ok &= (u >= 0) & (u < w) & (r >= 0) & (r < h)
        idx = np.nonzero(ok)[0]
        if not len(idx):
            continue
        d = v.depth[r[idx], u[idx]]
        seen[idx] += 1
        empty = (d > 0) & (z[idx] < d - depth_tol)
        if v.mask is not None:
            m = binary_dilation(v.mask, iterations=mask_dilate_px) if mask_dilate_px else v.mask
            empty |= ~m[r[idx], u[idx]] & ((d == 0) | (z[idx] < d + depth_tol))
        free[idx[empty]] += 1
    occ = free < min_free_votes
    if require_seen:
        occ &= seen > 0
    return occ.reshape(shape)


def keep_component(occ: np.ndarray, near_index=None) -> np.ndarray:
    """The connected component containing ``near_index`` (grid index), else the largest."""
    lab, n = cc_label(occ)
    if n == 0:
        return occ
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    k = 0
    if near_index is not None:
        ci = tuple(np.clip(np.asarray(near_index), 0, np.array(occ.shape) - 1))
        k = lab[ci]
    if k == 0:
        k = int(np.argmax(sizes))
    return lab == k


def solid_to_mesh(occ: np.ndarray, lo, voxel: float, max_triangles: int = 8000,
                  sigma: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Watertight triangle mesh of a voxel solid (grid frame coordinates)."""
    from skimage.measure import marching_cubes

    field = np.pad(gaussian_filter(occ.astype(np.float32), sigma), 1)  # padding closes the solid
    verts, faces, _, _ = marching_cubes(field, 0.5)
    verts = (verts - 1 + 0.5) * voxel + np.asarray(lo, float)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts), o3d.utility.Vector3iVector(faces[:, ::-1]))
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    if len(mesh.triangles) > max_triangles:
        mesh = mesh.simplify_quadric_decimation(max_triangles)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)
