"""High-resolution texture from the phone's colour images.

The TSDF only stores colour at voxel resolution (~2 cm), which looks blurry in
Gazebo. Here the mesh gets a UV atlas and every texel is coloured from the
sharpest, most head-on keyframes that actually see it - skipping pixels that
show people/animals, so no "ghost" people get painted onto the walls.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np
import open3d as o3d

from .raycast import make_scene, render
from .session import Session

log = logging.getLogger(__name__)


@dataclass
class TextureConfig:
    size: int = 2048
    max_keyframes: int = 120
    visibility_width: int = 640  # resolution of the occlusion test render
    blend_power: float = 4.0  # higher = closer to "best single view", less blur
    min_cos: float = 0.25  # ignore views more grazing than ~75 deg


@dataclass
class TexturedMesh:
    vertices: np.ndarray  # (V, 3)
    triangles: np.ndarray  # (T, 3)
    uvs: np.ndarray  # (T, 3, 2), OBJ convention (v = 0 at image bottom)
    normals: np.ndarray  # (V, 3)
    texture: np.ndarray  # (S, S, 3) RGB uint8


def sharpness(rgb: np.ndarray) -> float:
    g = cv2.cvtColor(cv2.resize(rgb, (320, 240)), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def select_keyframes(session: Session, n: int) -> list:
    frames = session.frames
    if len(frames) <= n:
        return list(frames)
    chunks = np.array_split(np.arange(len(frames)), n)
    out = []
    for ch in chunks:
        best = max(ch, key=lambda i: sharpness(session.load_color(frames[i])))
        out.append(frames[best])
    return out


def _rasterize_uv(uvs: np.ndarray, size: int):
    """For every texel centre return (triangle id, barycentrics) by ray casting the
    flattened UV mesh - fully vectorised, no per-triangle Python loop."""
    n = len(uvs)
    verts = np.concatenate([uvs.reshape(-1, 2), np.zeros((3 * n, 1))], 1).astype(np.float32)
    tris = np.arange(3 * n, dtype=np.uint32).reshape(n, 3)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(verts), o3d.core.Tensor(tris))
    x, y = np.meshgrid((np.arange(size) + 0.5) / size, (np.arange(size) + 0.5) / size)
    rays = np.stack([x, y, np.ones_like(x), np.zeros_like(x), np.zeros_like(x), -np.ones_like(x)], -1)
    ans = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6).astype(np.float32)))
    hit = np.isfinite(ans["t_hit"].numpy())
    prim = ans["primitive_ids"].numpy().astype(np.int64)
    bary = ans["primitive_uvs"].numpy()
    return hit, prim, bary


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    x = np.clip(x - 0.5, 0, w - 1.001)
    y = np.clip(y - 0.5, 0, h - 1.001)
    x0, y0 = x.astype(np.int64), y.astype(np.int64)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    im = img.astype(np.float32)
    return (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x0 + 1] * fx * (1 - fy)
            + im[y0 + 1, x0] * (1 - fx) * fy + im[y0 + 1, x0 + 1] * fx * fy)


def bake_texture(session: Session, mesh: o3d.geometry.TriangleMesh, cfg: TextureConfig,
                 dynamic_masks: dict[int, np.ndarray] | None = None) -> TexturedMesh:
    """``mesh`` must be in the capture's world frame (the frame the poses use)."""
    mesh.compute_triangle_normals()
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    log.info("computing UV atlas for %d triangles", len(mesh.triangles))
    tmesh.compute_uvatlas(size=cfg.size, gutter=2.0, parallel_partitions=max(1, os.cpu_count() or 1))
    uvs = tmesh.triangle.texture_uvs.numpy().astype(np.float64)  # (T,3,2), row 0 = v 0

    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    Fn = np.asarray(mesh.triangle_normals)
    vcol = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.full((len(V), 3), 0.6)

    hit, prim, bary = _rasterize_uv(uvs, cfg.size)
    idx = np.nonzero(hit)[0]
    tri = prim[idx]
    b1, b2 = bary[idx, 0:1], bary[idx, 1:2]
    b0 = 1.0 - b1 - b2
    P = b0 * V[F[tri, 0]] + b1 * V[F[tri, 1]] + b2 * V[F[tri, 2]]
    N = Fn[tri]
    fallback = (b0 * vcol[F[tri, 0]] + b1 * vcol[F[tri, 1]] + b2 * vcol[F[tri, 2]]) * 255.0

    acc = np.zeros((len(idx), 3))
    wsum = np.zeros(len(idx))
    scene = make_scene(mesh)
    keyframes = select_keyframes(session, cfg.max_keyframes)
    log.info("projecting %d keyframes onto %d texels", len(keyframes), len(idx))

    for fr in keyframes:
        rgb = session.load_color(fr)
        ch, cw = rgb.shape[:2]
        sh = sharpness(rgb)
        pose = fr.pose_cv
        world_to_cam = np.linalg.inv(pose)
        pc = P @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        z = pc[:, 2]
        K = fr.K_color
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * pc[:, 0] / z + K[0, 2]
            v = K[1, 1] * pc[:, 1] / z + K[1, 2]
        view = pose[:3, 3] - P
        dist = np.linalg.norm(view, axis=1)
        cos = np.einsum("ij,ij->i", N, view) / np.maximum(dist, 1e-9)
        ok = (z > 0.1) & (u >= 1) & (u < cw - 1) & (v >= 1) & (v < ch - 1) & (cos > cfg.min_cos)
        if not ok.any():
            continue

        # occlusion test against a rendered depth map
        s = cfg.visibility_width / cw
        vw, vh = cfg.visibility_width, int(round(ch * s))
        Kv = K.copy()
        Kv[:2] *= s
        rd, _ = render(scene, Kv, pose, vw, vh)
        iu = np.clip((u[ok] * s).astype(np.int64), 0, vw - 1)
        iv = np.clip((v[ok] * s).astype(np.int64), 0, vh - 1)
        sel = np.nonzero(ok)[0]
        visible = np.abs(rd[iv, iu] - z[ok]) < 0.02 + 0.02 * z[ok]
        sel = sel[visible]

        # never take colour from pixels showing people / animals
        dm = session.load_mask(fr, size=(cw, ch))
        extra = None if dynamic_masks is None else dynamic_masks.get(fr.index)
        if extra is not None:
            up = cv2.resize(extra.astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST) > 0
            dm = up if dm is None else (dm | up)
        if dm is not None and dm.any():
            r = max(3, int(2 * cw / fr.depth_size[0]))
            dm = cv2.dilate(dm.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            sel = sel[~dm[v[sel].astype(np.int64), u[sel].astype(np.int64)]]
        if len(sel) == 0:
            continue

        w = (cos[sel] ** 2 / np.maximum(dist[sel], 0.2) ** 2) * (1.0 + np.log1p(sh))
        w = w ** cfg.blend_power
        acc[sel] += w[:, None] * _bilinear(rgb, u[sel], v[sel])
        wsum[sel] += w

    seen = wsum > 0
    log.info("texture coverage from photos: %.1f%% of texels", 100.0 * seen.mean() if len(seen) else 0)
    colors = fallback.copy()
    colors[seen] = acc[seen] / wsum[seen, None]

    tex = np.zeros((cfg.size * cfg.size, 3), np.uint8)
    tex[idx] = np.clip(colors, 0, 255).astype(np.uint8)
    tex = tex.reshape(cfg.size, cfg.size, 3)
    # pad charts into the gutter so mip-mapping in Gazebo shows no dark seams
    filled = hit.reshape(cfg.size, cfg.size)
    for _ in range(8):
        grown = cv2.dilate(tex, np.ones((3, 3), np.uint8))
        ring = cv2.dilate(filled.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ring &= ~filled
        tex[ring] = grown[ring]
        filled |= ring

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    uvs_obj = uvs.copy()
    uvs_obj[..., 1] = 1.0 - uvs_obj[..., 1]
    return TexturedMesh(V.copy(), F.copy(), uvs_obj, np.asarray(mesh.vertex_normals).copy(), tex)
