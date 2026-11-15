"""End-to-end processing: session -> clean mesh -> Gazebo world + nav map."""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d

from . import gazebo, ghosts, lidar, mesh as meshops, occupancy, stereo
from .fusion import FusionConfig, integrate
from .masks import resegment_session
from .session import Session
from .texture import TextureConfig, TexturedMesh, bake_texture

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    name: str = "scan"
    fusion: FusionConfig = field(default_factory=FusionConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    stereo: stereo.StereoConfig = field(default_factory=stereo.StereoConfig)
    lidar: lidar.LidarConfig = field(default_factory=lidar.LidarConfig)
    # With an external LiDAR on a phone without depth sensor, also fuse camera stereo depth?
    rig_stereo: bool = True
    texture_mode: str = "photo"  # photo | none
    ghost_removal: bool = True
    ghost_min_votes: int = 3
    ghost_iterations: int = 3
    resegment_model: str | None = None  # e.g. "yolo11x-seg.pt"
    hole_size: float = 0.15  # metres; 0 disables
    max_visual_triangles: int = 250_000
    max_collision_triangles: int = 60_000
    map_resolution: float = 0.05
    map_obstacle_min_z: float = 0.08
    map_obstacle_max_z: float = 1.8
    center_xy: bool = True


def _merge_masks(a: dict | None, b: dict | None) -> dict | None:
    if not a:
        return dict(b) if b else a
    if not b:
        return dict(a)
    return {k: (a[k] | b[k]) if k in a and k in b else a.get(k, b.get(k)) for k in set(a) | set(b)}


def transform_textured(tm: TexturedMesh, T: np.ndarray) -> TexturedMesh:
    R = T[:3, :3]
    return TexturedMesh(tm.vertices @ R.T + T[:3, 3], tm.triangles, tm.uvs, tm.normals @ R.T, tm.texture)


@dataclass
class RoomResult:
    """The reconstructed static room, before anything is written to disk.

    ``aligned`` and ``textured`` are in the Gazebo/ROS frame (Z up, floor at z = 0);
    ``T`` maps the phone's capture frame (Y up) to it."""

    session: Session
    aligned: o3d.geometry.TriangleMesh
    textured: TexturedMesh | None
    T: np.ndarray
    cloud: object | None
    camera_only: bool
    use_frames: bool
    n_ghost: int
    dynamic_masks: dict | None
    t0: float


def reconstruct_room(session_dir: str | Path, cfg: PipelineConfig) -> RoomResult:
    """Masked fusion + ghost removal + cleanup + floor alignment + texture (no files written)."""
    t0 = time.time()
    session = Session.load(session_dir)
    log.info("loaded %d frames from %s (%s)", len(session.frames), session_dir,
             session.meta.get("platform", "?"))

    camera_only = not session.has_sensor_depth
    use_frames = True
    if camera_only and session.has_lidar and not cfg.rig_stereo and session.rig.get("kind", "2d") == "2d":
        use_frames = False
        log.info("external LiDAR only (camera stereo disabled)")
    elif camera_only:
        log.info("camera-only capture (no LiDAR/ToF): computing depth by multi-view stereo")
        stereo.estimate_session_depth(session, cfg.stereo)
    if cfg.fusion.min_confidence is None:
        cfg.fusion.min_confidence = float(session.meta.get("recommended_min_confidence", 0.5))

    masks = resegment_session(session, cfg.resegment_model) if cfg.resegment_model else None

    cloud = lidar.load_cloud(session, cfg.lidar) if session.has_lidar else None
    views = None
    if cloud is not None:
        lidar.mask_living(session, cloud, cfg.lidar, masks)
        views = lidar.build_views(cloud, cfg.lidar)

    log.info("pass 1: fusing with people/animal masks")
    fused = integrate(session, cfg.fusion, masks=masks, views=views, use_frames=use_frames)
    extra, n_ghost = None, 0
    if cfg.ghost_removal and len(fused.triangles):
        extra = {}
        for it in range(cfg.ghost_iterations):
            log.info("ghost check %d: looking for transient objects the masks missed", it + 1)
            combined = _merge_masks(masks, extra)
            kept = int(cloud.keep.sum()) if cloud is not None else 0
            new, transient = ghosts.detect_ghost_masks(session, fused, cfg.fusion, masks=combined,
                                                       min_votes=cfg.ghost_min_votes, cloud=cloud)
            lidar_changed = cloud is not None and int(cloud.keep.sum()) < kept
            if not new and not lidar_changed:
                break
            n_ghost += int(transient.sum())
            extra = _merge_masks(extra, new)
            if lidar_changed:
                views = lidar.build_views(cloud, cfg.lidar)
            log.info("re-fusing without %d transient triangles", int(transient.sum()))
            fused = integrate(session, cfg.fusion, masks=masks, extra_masks=extra, views=views,
                              use_frames=use_frames)
            if transient.sum() < 0.001 * len(transient):
                break

    world_mesh = meshops.clean(fused)
    world_mesh = meshops.fill_small_holes(world_mesh, cfg.hole_size)
    world_mesh = meshops.simplify(world_mesh, cfg.max_visual_triangles)
    if len(world_mesh.triangles) == 0:
        raise RuntimeError("reconstruction is empty - check depth data / poses")

    # Y-up phone world -> Z-up Gazebo world with the floor at z = 0
    zup = copy.deepcopy(world_mesh).transform(meshops.Y_UP_TO_Z_UP)
    T = meshops.floor_alignment(zup, center_xy=cfg.center_xy) @ meshops.Y_UP_TO_Z_UP
    aligned = copy.deepcopy(world_mesh).transform(T)
    aligned.compute_vertex_normals()

    dyn = _merge_masks(masks, extra)
    textured = None
    if cfg.texture_mode == "photo":
        textured = transform_textured(bake_texture(session, world_mesh, cfg.texture, dyn or None), T)
    return RoomResult(session, aligned, textured, T, cloud, camera_only, use_frames, n_ghost, dyn, t0)


def export_room(room: RoomResult, out_dir: str | Path, cfg: PipelineConfig) -> dict:
    """Write the Gazebo model/worlds, nav map, PLY and report.json (the `process` output)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session, aligned, cloud, T = room.session, room.aligned, room.cloud, room.T
    collision = meshops.simplify(copy.deepcopy(aligned), cfg.max_collision_triangles)
    files = gazebo.export_gazebo(out_dir, cfg.name, room.textured, aligned, collision)

    o3d.io.write_triangle_mesh(str(out_dir / f"{cfg.name}_mesh.ply"), aligned)
    (out_dir / "maps").mkdir(exist_ok=True)
    grid, origin = occupancy.occupancy_grid(aligned, cfg.map_resolution,
                                            cfg.map_obstacle_min_z, cfg.map_obstacle_max_z)
    occupancy.write_map(out_dir / "maps" / cfg.name, grid, origin, cfg.map_resolution)

    bbox = aligned.get_axis_aligned_bounding_box()
    report = {
        "name": cfg.name,
        "frames": len(session.frames),
        "depth_source": ("sensor" if not room.camera_only else "stereo" if room.use_frames else "none")
                        + (" + external LiDAR" if cloud is not None else ""),
        "rig": None if cloud is None else {
            "sensor": session.rig.get("sensor"), "kind": cloud.kind,
            "readings_used": int(cloud.keep.sum()), "readings_dropped": int((~cloud.keep).sum()),
            "calibration": cloud.calibration, "T_camera_lidar": cloud.T_cam_lidar.tolist(),
        },
        "platform": session.meta.get("platform"),
        "triangles_visual": len(aligned.triangles),
        "triangles_collision": len(collision.triangles),
        "ghost_triangles_removed": room.n_ghost,
        "extent_m": np.round(bbox.get_extent(), 3).tolist(),
        "transform_capture_to_gazebo": T.tolist(),
        "seconds": round(time.time() - room.t0, 1),
        "outputs": {k: str(v) for k, v in files.items()},
        "config": asdict(cfg),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("done in %.1fs -> %s", report["seconds"], out_dir)
    return report


def run(session_dir: str | Path, out_dir: str | Path, cfg: PipelineConfig) -> dict:
    """`cleanscan process`: session -> clean mesh, Gazebo world, nav map (unchanged output)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return export_room(reconstruct_room(session_dir, cfg), out_dir, cfg)
