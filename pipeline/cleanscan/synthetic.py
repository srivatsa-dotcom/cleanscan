"""Synthetic capture of a room with a walking person and a dog.

Used by the tests and for trying the pipeline without a phone:
    cleanscan synth demo_session && cleanscan process demo_session -o demo_out
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from .raycast import camera_rays
from .session import GL_TO_CV

ROOM = (6.0, 2.8, 5.0)  # x, y(up), z
# walking path of the person (x, z), passing ~0.8 m beside the camera
PERSON_A, PERSON_B = (-2.2, -1.9), (2.2, -0.9)


def _box(size, center):
    b = o3d.geometry.TriangleMesh.create_box(*size)
    b.translate(np.asarray(center) - np.asarray(size) / 2)
    return b


def static_scene() -> o3d.geometry.TriangleMesh:
    sx, sy, sz = ROOM
    room = _box((sx, sy, sz), (0, sy / 2, 0))  # rays from inside hit the inner faces
    table = _box((1.2, 0.75, 0.7), (1.6, 0.375, 1.2))
    cabinet = _box((0.6, 1.8, 0.5), (-2.4, 0.9, 1.9))
    return room + table + cabinet


def person_at(t: float) -> o3d.geometry.TriangleMesh:
    p = o3d.geometry.TriangleMesh.create_cylinder(radius=0.25, height=1.75, resolution=24)
    p.rotate(o3d.geometry.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)), center=(0, 0, 0))
    a = np.array([PERSON_A[0], 0.875, PERSON_A[1]])
    b = np.array([PERSON_B[0], 0.875, PERSON_B[1]])
    return p.translate(a + (b - a) * t)


def dog_at(t: float) -> o3d.geometry.TriangleMesh:
    ang = 2 * np.pi * t
    return _box((0.7, 0.5, 0.3), (1.8 * np.cos(ang), 0.25, 1.5 * np.sin(ang)))


def _shade(points: np.ndarray, geom: np.ndarray) -> np.ndarray:
    base = np.array([[0.85, 0.82, 0.75], [0.9, 0.1, 0.1], [0.55, 0.35, 0.15]])
    g = np.clip(geom, 0, 2)
    col = base[g].copy()
    checker = (np.floor((points + 0.125) / 0.25).astype(np.int64).sum(-1) % 2).astype(np.float32)
    stripes = 0.5 + 0.5 * np.sin(points[:, 0] * 3.0 + points[:, 2] * 2.0)
    col *= (0.65 + 0.25 * checker + 0.1 * stripes)[:, None]
    return col


def _pose(theta: float, phi: float, pos) -> np.ndarray:
    Ry = o3d.geometry.get_rotation_matrix_from_xyz((0, theta, 0))
    Rx = o3d.geometry.get_rotation_matrix_from_xyz((phi, 0, 0))
    T = np.eye(4)
    T[:3, :3] = Ry @ Rx
    T[:3, 3] = pos
    return T


def make_session(out: str | Path, n_frames: int = 160, with_masks: bool = True,
                 color_size=(640, 480), depth_size=(256, 192), noise: float = 0.004,
                 seed: int = 0, with_depth: bool = True, trajectory: str = "spin") -> Path:
    """``with_depth=False`` imitates a phone without LiDAR/ToF (e.g. iPhone 11).
    ``trajectory``: "spin" turns on the spot (fine with a depth sensor), "walk" walks a
    loop looking at the walls, which camera-only (stereo) capture needs."""
    out = Path(out)
    for d in ("color", "depth", "confidence", "mask"):
        (out / d).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    static = o3d.t.geometry.TriangleMesh.from_legacy(static_scene())
    fov = np.radians(60)

    def K_for(w, h):
        f = (w / 2) / np.tan(fov / 2)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])

    Kc, Kd = K_for(*color_size), K_for(*depth_size)
    lines = []
    for i in range(n_frames):
        s = i / max(n_frames - 1, 1)
        if trajectory == "walk":
            a = 2 * np.pi * s
            theta = np.arctan2(-np.cos(a), -np.sin(a))  # look outwards, at the walls
            phi = np.radians(-20 + 10 * np.sin(4 * np.pi * s))
            pos = (1.0 * np.cos(a), 1.4, 0.8 * np.sin(a))
        else:
            theta = 4 * np.pi * s  # two full turns
            phi = np.radians(-25 + 15 * np.sin(6 * np.pi * s))
            pos = (0.4 * np.sin(2 * np.pi * s), 1.4, 0.3 * np.cos(2 * np.pi * s))
        pose_gl = _pose(theta, phi, pos)
        pose_cv = pose_gl @ GL_TO_CV

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(static)
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(person_at(s)))
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(dog_at(s)))

        def cast(K, size):
            w, h = size
            rays = camera_rays(K, pose_cv, w, h)
            ans = scene.cast_rays(o3d.core.Tensor(rays))
            t = ans["t_hit"].numpy()
            g = ans["geometry_ids"].numpy().astype(np.int64)
            g[~np.isfinite(t)] = -1
            return rays, t, g

        rays, t, g = cast(Kc, color_size)
        pts = rays[..., :3] + rays[..., 3:] * np.where(np.isfinite(t), t, 0)[..., None]
        rgb = (_shade(pts.reshape(-1, 3), g.reshape(-1)).reshape(*t.shape, 3) * 255).astype(np.uint8)

        _, td, gd = cast(Kd, depth_size)
        depth = np.where(np.isfinite(td), td, 0) * (1 + noise * rng.standard_normal(td.shape))
        mask = ((gd == 1) | (gd == 2)).astype(np.uint8) * 255

        name = f"{i:06d}"
        cv2.imwrite(str(out / "color" / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        if with_depth:
            cv2.imwrite(str(out / "depth" / f"{name}.png"), np.clip(depth * 1000, 0, 65535).astype(np.uint16))
            cv2.imwrite(str(out / "confidence" / f"{name}.png"), np.full(td.shape, 2, np.uint8))
        rec = {
            "index": i, "timestamp": i / 10.0,
            "color": f"color/{name}.jpg",
            "color_intrinsics": [Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]],
            "color_size": list(color_size),
            "depth_intrinsics": [Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2]],
            "depth_size": list(depth_size),
            "camera_to_world": pose_gl.reshape(-1).tolist(),
        }
        if with_depth:
            rec["depth"] = f"depth/{name}.png"
            rec["confidence"] = f"confidence/{name}.png"
        if with_masks:
            cv2.imwrite(str(out / "mask" / f"{name}.png"), mask)
            rec["mask"] = f"mask/{name}.png"
        lines.append(json.dumps(rec))

    (out / "frames.jsonl").write_text("\n".join(lines) + "\n")
    (out / "session.json").write_text(json.dumps({
        "format": "cleanscan-capture", "version": 1, "platform": "synthetic",
        "device": "synthetic", "depth_unit": 0.001, "confidence_levels": 3,
        "world_up": "y", "camera_convention": "opengl",
        "depth_source": "synthetic" if with_depth else "none",
    }, indent=2))
    return out
