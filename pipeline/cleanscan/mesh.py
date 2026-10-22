"""Mesh cleanup and conversion into the Gazebo / ROS frame (Z up, floor at z = 0)."""

from __future__ import annotations

import copy
import logging

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)

# ARKit/ARCore world (Y up) -> ROS/Gazebo world (Z up): x'=x, y'=-z, z'=y
Y_UP_TO_Z_UP = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])


def clean(mesh: o3d.geometry.TriangleMesh, min_component_frac: float = 0.002,
          min_component_tris: int = 100) -> o3d.geometry.TriangleMesh:
    """Remove floating fragments (noise, leftover bits of people) and bad geometry."""
    mesh = copy.deepcopy(mesh)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    if len(mesh.triangles) == 0:
        return mesh
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    counts = np.asarray(counts)
    keep_threshold = max(min_component_tris, min_component_frac * counts.max())
    drop = counts[labels] < keep_threshold
    log.info("cleanup: removing %d small fragments (%d triangles)",
             int((counts < keep_threshold).sum()), int(drop.sum()))
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def fill_small_holes(mesh: o3d.geometry.TriangleMesh, hole_size: float) -> o3d.geometry.TriangleMesh:
    """Close small holes (e.g. LiDAR dropouts). Big holes are left open on purpose:
    filling them would invent geometry that was never seen."""
    if hole_size <= 0:
        return mesh
    try:
        t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        filled = t.fill_holes(hole_size=hole_size).to_legacy()
        if len(filled.vertex_colors) != len(filled.vertices):
            return mesh
        filled.compute_vertex_normals()
        return filled
    except Exception as exc:  # fill_holes depends on Open3D's VTK build
        log.warning("hole filling skipped: %s", exc)
        return mesh


def floor_alignment(mesh_zup: o3d.geometry.TriangleMesh, center_xy: bool = True) -> np.ndarray:
    """Find the floor (lowest large horizontal plane) and return a 4x4 that makes
    it exactly z = 0 with +z normal, optionally centering the scan on the origin."""
    pcd = mesh_zup.sample_points_uniformly(number_of_points=200_000, use_triangle_normal=True)
    pts = np.asarray(pcd.points)
    nrm = np.asarray(pcd.normals)
    horiz = np.abs(nrm[:, 2]) > 0.9
    T = np.eye(4)
    if horiz.sum() < 500:
        log.warning("no floor found; keeping phone gravity alignment")
    else:
        cand = pts[horiz]
        best = None
        remaining = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand))
        for _ in range(6):  # extract a few horizontal planes, pick the lowest big one
            if len(remaining.points) < 500:
                break
            plane, inl = remaining.segment_plane(0.02, 3, 1000)
            p = np.asarray(remaining.points)[inl]
            a, b, c, d = plane
            if abs(c) > 0.95 and len(inl) > 0.05 * len(cand):
                if best is None or p[:, 2].mean() < best[1]:
                    best = (np.array(plane), p[:, 2].mean())
            remaining = remaining.select_by_index(inl, invert=True)
        if best is None:
            log.warning("no dominant floor plane; keeping phone gravity alignment")
        else:
            plane = best[0] / np.linalg.norm(best[0][:3])
            n = plane[:3] if plane[2] > 0 else -plane[:3]
            dist = plane[3] if plane[2] > 0 else -plane[3]
            # rotation taking n to +z
            z = np.array([0.0, 0.0, 1.0])
            v = np.cross(n, z)
            s, c = np.linalg.norm(v), float(n @ z)
            if s < 1e-9:
                R = np.eye(3)
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
            T[:3, :3] = R
            # plane: n.x + dist = 0  -> after rotation floor sits at z = -dist
            T[2, 3] = dist
            log.info("floor found, tilt correction %.2f deg", np.degrees(np.arccos(np.clip(c, -1, 1))))
    if center_xy:
        p = (T[:3, :3] @ pts.T).T + T[:3, 3]
        lo, hi = np.percentile(p[:, :2], 1, axis=0), np.percentile(p[:, :2], 99, axis=0)
        T[0, 3] -= (lo[0] + hi[0]) / 2
        T[1, 3] -= (lo[1] + hi[1]) / 2
    return T


def simplify(mesh: o3d.geometry.TriangleMesh, max_triangles: int) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) <= max_triangles:
        return mesh
    out = mesh.simplify_quadric_decimation(target_number_of_triangles=max_triangles)
    out.remove_degenerate_triangles()
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out
