"""Render depth / triangle ids of a mesh from a pinhole camera (Open3D raycasting)."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def make_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def camera_rays(K: np.ndarray, pose_cv: np.ndarray, w: int, h: int) -> np.ndarray:
    """Rays (h, w, 6) whose direction has unit camera-z, so t_hit == z-depth."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    d_world = d_cam @ pose_cv[:3, :3].T
    origin = np.broadcast_to(pose_cv[:3, 3], d_world.shape)
    return np.concatenate([origin, d_world], -1).astype(np.float32)


def render(scene: o3d.t.geometry.RaycastingScene, K: np.ndarray, pose_cv: np.ndarray,
           w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (z-depth with inf for misses, primitive id with -1 for misses)."""
    ans = scene.cast_rays(o3d.core.Tensor(camera_rays(K, pose_cv, w, h)))
    depth = ans["t_hit"].numpy()
    prim = ans["primitive_ids"].numpy().astype(np.int64)
    prim[~np.isfinite(depth)] = -1
    return depth, prim
