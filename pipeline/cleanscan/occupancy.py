"""2D occupancy grid (ROS map_server / Nav2 format) sliced from the clean mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

FREE, OCCUPIED, UNKNOWN = 254, 0, 205


def occupancy_grid(mesh_zup: o3d.geometry.TriangleMesh, resolution: float = 0.05,
                   obstacle_min_z: float = 0.08, obstacle_max_z: float = 1.8,
                   floor_tol: float = 0.05, margin: float = 0.5):
    """Returns (grid uint8 with row 0 = max y, origin_xy)."""
    area = mesh_zup.get_surface_area()
    n = int(np.clip(area / resolution**2 * 16, 50_000, 5_000_000))
    pts = np.asarray(mesh_zup.sample_points_uniformly(number_of_points=n).points)
    lo = pts[:, :2].min(0) - margin
    hi = pts[:, :2].max(0) + margin
    w, h = np.ceil((hi - lo) / resolution).astype(int)
    ij = np.floor((pts[:, :2] - lo) / resolution).astype(int)
    ij = np.clip(ij, 0, [w - 1, h - 1])

    grid = np.full((h, w), UNKNOWN, np.uint8)
    z = pts[:, 2]
    floor = np.abs(z) < floor_tol
    grid[ij[floor, 1], ij[floor, 0]] = FREE
    obst = (z > obstacle_min_z) & (z < obstacle_max_z)
    counts = np.zeros((h, w), np.int32)
    np.add.at(counts, (ij[obst, 1], ij[obst, 0]), 1)
    grid[counts >= 2] = OCCUPIED
    return grid[::-1].copy(), lo


def write_map(path_stem: Path, grid: np.ndarray, origin_xy: np.ndarray, resolution: float) -> None:
    path_stem = Path(path_stem)
    h, w = grid.shape
    with open(path_stem.with_suffix(".pgm"), "wb") as fh:
        fh.write(f"P5\n# CleanScan occupancy map\n{w} {h}\n255\n".encode())
        fh.write(grid.tobytes())
    path_stem.with_suffix(".yaml").write_text(
        f"image: {path_stem.name}.pgm\n"
        f"mode: trinary\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin_xy[0]:.4f}, {origin_xy[1]:.4f}, 0.0]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.25\n"
    )
