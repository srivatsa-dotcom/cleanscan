"""TSDF fusion with dynamic objects removed.

Every pixel that belongs to a person/animal (or that the ghost detector flags)
is dropped *before* integration. Because the TSDF averages many views, the
wall/floor behind a person is reconstructed from the frames in which the person
was somewhere else - that is how the background behind living beings is
recovered without hallucinating it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from .masks import grow_mask
from .session import Frame, Session

log = logging.getLogger(__name__)


@dataclass
class DepthView:
    """A depth image from any source other than the session's camera frames, e.g. a
    virtual camera built from external-LiDAR rays (see lidar.py). Depth in metres, 0 = none,
    already free of people/animals."""

    key: str
    K: np.ndarray
    pose_cv: np.ndarray  # camera -> world, OpenCV axes
    depth: np.ndarray
    max_depth: float = 12.0

    @property
    def size(self) -> tuple[int, int]:
        return self.depth.shape[1], self.depth.shape[0]

    @property
    def extrinsic_cv(self) -> np.ndarray:
        return np.linalg.inv(self.pose_cv)


@dataclass
class FusionConfig:
    voxel_size: float = 0.02  # metres
    sdf_trunc_voxels: float = 4.0
    min_depth: float = 0.15
    max_depth: float = 5.0  # LiDAR/ToF quality drops quickly past ~5 m
    min_confidence: float | None = None  # 0..1; None = session's recommendation or 0.5
    mask_dilate_px: int = 4
    edge_threshold: float = 0.04  # relative depth jump that marks a "flying pixel"
    min_weight: float = 2.0  # a surface must be seen in >= 2 frames to be kept

    @property
    def sdf_trunc(self) -> float:
        return self.voxel_size * self.sdf_trunc_voxels

    @property
    def block_count(self) -> int:
        # initial hash capacity (8^3-voxel blocks); grows automatically if needed
        return int(40_000 * (0.02 / self.voxel_size) ** 2)


def remove_flying_pixels(depth: np.ndarray, rel_thresh: float) -> np.ndarray:
    """Invalidate pixels on depth discontinuities (mixed foreground/background)."""
    valid = depth > 0
    big = np.where(valid, depth, 0).astype(np.float32)
    small = np.where(valid, depth, 1e6).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(big, k) - cv2.erode(small, k)
    bad = spread > rel_thresh * np.maximum(depth, 1e-3)
    # A neighbour being invalid also makes the spread huge; only reject real jumps
    # between two valid surfaces.
    has_invalid_neighbour = cv2.erode(valid.astype(np.uint8), k) == 0
    bad &= ~has_invalid_neighbour
    out = depth.copy()
    out[bad] = 0
    return out


def prepare_depth(session: Session, frame: Frame, cfg: FusionConfig,
                  mask_override: np.ndarray | None = None,
                  extra_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (static depth in metres with invalid = 0, dynamic mask actually used)."""
    depth = session.load_depth(frame)
    conf = session.load_confidence(frame)
    if conf is not None:
        depth[conf < (0.5 if cfg.min_confidence is None else cfg.min_confidence) - 1e-6] = 0
    depth[(depth < cfg.min_depth) | (depth > cfg.max_depth)] = 0
    depth = remove_flying_pixels(depth, cfg.edge_threshold)

    mask = mask_override if mask_override is not None else session.load_mask(frame)
    if mask is None:
        mask = np.zeros(depth.shape, bool)
    mask = grow_mask(mask, depth, cfg.mask_dilate_px)
    if extra_mask is not None:
        mask = mask | extra_mask
    depth[mask] = 0
    return depth, mask


def integrate(session: Session, cfg: FusionConfig,
              masks: dict[int, np.ndarray] | None = None,
              extra_masks: dict[int, np.ndarray] | None = None,
              views: list[DepthView] | None = None,
              use_frames: bool = True) -> o3d.geometry.TriangleMesh:
    """Fuse all frames (and extra depth views) into a coloured triangle mesh (world, Y up)."""
    o3c = o3d.core
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=(1, 1, 3),
        voxel_size=cfg.voxel_size,
        block_resolution=8,
        block_count=cfg.block_count,
        device=o3c.Device("CPU:0"),
    )
    frames = session.frames if use_frames else []
    n = len(frames)
    for i, fr in enumerate(frames):
        depth, _ = prepare_depth(
            session, fr, cfg,
            mask_override=None if masks is None else masks.get(fr.index),
            extra_mask=None if extra_masks is None else extra_masks.get(fr.index),
        )
        if np.count_nonzero(depth) < 50:
            continue  # nothing static and valid in this frame
        w, h = fr.depth_size
        color = cv2.resize(session.load_color(fr), (w, h), interpolation=cv2.INTER_AREA)
        d_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(depth, np.float32)))
        c_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(color, np.float32) / 255.0))
        K = o3c.Tensor(fr.K_depth)
        E = o3c.Tensor(fr.extrinsic_cv)
        blocks = vbg.compute_unique_block_coordinates(d_img, K, E, 1.0, cfg.max_depth,
                                                      cfg.sdf_trunc_voxels)
        if blocks.shape[0] == 0:
            continue
        vbg.integrate(blocks, d_img, c_img, K, K, E, 1.0, cfg.max_depth, cfg.sdf_trunc_voxels)
        if (i + 1) % 100 == 0 or i + 1 == n:
            log.info("integrated %d/%d frames", i + 1, n)
    for v in views or []:
        if np.count_nonzero(v.depth) < 3:
            continue
        w, h = v.size
        d_img = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(v.depth, np.float32)))
        c_img = o3d.t.geometry.Image(o3c.Tensor(np.full((h, w, 3), 0.6, np.float32)))
        K = o3c.Tensor(v.K)
        E = o3c.Tensor(v.extrinsic_cv)
        try:
            blocks = vbg.compute_unique_block_coordinates(d_img, K, E, 1.0, v.max_depth, cfg.sdf_trunc_voxels)
        except RuntimeError:  # Open3D raises when no voxel block is touched
            continue
        if blocks.shape[0]:
            vbg.integrate(blocks, d_img, c_img, K, K, E, 1.0, v.max_depth, cfg.sdf_trunc_voxels)
    if views:
        log.info("integrated %d LiDAR views", len(views))
    if vbg.hashmap().size() == 0:
        return o3d.geometry.TriangleMesh()
    mesh = vbg.extract_triangle_mesh(weight_threshold=cfg.min_weight).to_legacy()
    mesh.compute_vertex_normals()
    return mesh
