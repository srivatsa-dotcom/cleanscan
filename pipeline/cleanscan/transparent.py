"""Transparent and reflective objects (glasses, wine glasses, steel tumblers) from silhouettes.

Depth sensors and stereo see through or past glass, so the shape comes from the outline the
phone's segmentation gives (``object_mask``) in a few side views taken with the phone level,
plus one view from the top. The object is assumed to be a surface of revolution standing
upright on a support surface.

* support height: the surface just below the silhouette's base is opaque (a counter), so the
  depth there gives the plane the object stands on
* axis: the ray through the base meets that plane at the front of the base; the axis is one
  base radius behind it, in the direction that bisects the left and right edge rays.
  The axis positions of all side views are combined (median)
* profile: for every image row, the radius is the distance between the axis and the left /
  right edge rays (tangent rays of a surface of revolution); heights come from where those
  rays pass the axis. All views are binned by height (median)
* top view: the outline from above, projected onto the rim's height, gives an independent
  rim diameter (reported)
* fallback: a cylinder of the measured height and largest diameter when too few views are
  usable or the profile is inconsistent

All metric scale comes from the phone's pose (ARKit/ARCore, metres) and the support plane;
no depth on the object itself is needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .session import Session, load_label_image
from .shapes import lathe

log = logging.getLogger(__name__)

UP = np.array([0.0, 1.0, 0.0])


@dataclass
class TransparentConfig:
    bin_size: float = 0.003  # m, profile resolution
    min_views: int = 3
    max_profile_spread: float = 0.15  # relative spread of radii between views -> fallback
    level_tolerance_deg: float = 20.0  # side views must be roughly level
    segments: int = 48


@dataclass
class TransparentResult:
    V: np.ndarray  # object frame: origin = bottom centre, z up
    F: np.ndarray
    T_capture_object: np.ndarray
    profile: np.ndarray  # (n, 2) radius, height
    info: dict = field(default_factory=dict)


def _ray(fr, u: float, v: float, K) -> tuple[np.ndarray, np.ndarray]:
    """World ray (origin, unit direction) through pixel coordinates (u, v) (pixel edges, not centres)."""
    d = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
    P = fr.pose_cv
    d = P[:3, :3] @ d
    return P[:3, 3], d / np.linalg.norm(d)


def _closest_to_vertical(o, d, axis_xz):
    """Distance between ray (o, d) and the vertical line through axis_xz, and the height on
    the line of the closest approach."""
    a = np.array([axis_xz[0], 0.0, axis_xz[1]])
    w0 = o - a
    b = d @ UP
    dd, e = d @ w0, UP @ w0
    den = 1 - b * b
    if den < 1e-9:
        return np.inf, np.nan
    s = (b * e - dd) / den  # along the ray
    t = (e - b * dd) / den  # along the axis (height)
    p_ray, p_axis = o + s * d, a + t * UP
    return float(np.linalg.norm(p_ray - p_axis)), float(t)


def _support_y(session: Session, fr, mask: np.ndarray) -> float | None:
    """Height of the opaque surface right below the silhouette's base."""
    rows = np.nonzero(mask.any(1))[0]
    if not len(rows) or fr.depth_path is None:
        return None
    vb = rows.max()
    cols = np.nonzero(mask[vb])[0]
    depth = session.load_depth(fr)
    h, w = depth.shape
    ys = []
    for dv in range(2, 8):
        v = vb + dv
        if v >= h:
            break
        for u in range(max(cols.min() - 2, 0), min(cols.max() + 3, w)):
            z = depth[v, u]
            if z > 0:
                K = fr.K_depth
                pc = np.array([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z])
                ys.append((fr.pose_cv[:3, :3] @ pc + fr.pose_cv[:3, 3])[1])
    return float(np.median(ys)) if ys else None


def _view_axis(fr, mask, K, support_y):
    """(axis x, axis z, base radius) from one side view."""
    rows = np.nonzero(mask.any(1))[0]
    vb = rows.max()
    # base: a few rows above the very bottom (anti-aliased contact)
    band = mask[max(vb - 3, rows.min()):vb + 1]
    cols = np.nonzero(band.any(0))[0]
    ul, ur = cols.min(), cols.max() + 1
    uc = 0.5 * (ul + ur)
    o, dc = _ray(fr, uc, vb + 1.0, K)
    if abs(dc[1]) < 1e-6:
        return None
    s = (support_y - o[1]) / dc[1]
    if s <= 0:
        return None
    front = o + s * dc  # front of the base on the support
    _, dl = _ray(fr, ul, vb + 0.5, K)
    _, dr = _ray(fr, ur, vb + 0.5, K)
    bis = dl + dr
    bis[1] = 0
    bis /= np.linalg.norm(bis)
    half = 0.5 * np.arccos(np.clip(dl[[0, 2]] @ dr[[0, 2]] / (np.linalg.norm(dl[[0, 2]]) * np.linalg.norm(dr[[0, 2]])), -1, 1))
    d_front = np.linalg.norm((front - o)[[0, 2]])
    # the base circle (radius r) is tangent to both edge rays: r = D sin(half), D = d_front + r
    r = d_front * np.sin(half) / (1 - np.sin(half))
    D = d_front + r
    ax = o[[0, 2]] + D * bis[[0, 2]]
    return ax, r


def reconstruct_transparent(profile: Session, top: Session | None = None,
                            cfg: TransparentConfig | None = None) -> TransparentResult:
    cfg = cfg or TransparentConfig()
    views = []
    for fr in profile.frames:
        m = load_label_image(fr.extra_path("object_mask"), fr.depth_size)
        if m is None or (m > 127).sum() < 30:
            continue
        fwd = fr.pose_cv[:3, 2]
        if abs(np.degrees(np.arcsin(np.clip(fwd @ UP, -1, 1)))) > cfg.level_tolerance_deg:
            continue  # not level enough for a clean side profile
        views.append((fr, m > 127))
    if len(views) < 1:
        raise RuntimeError("no usable side views (the app's object mask is needed for glass)")
    sup = [y for fr, m in views if (y := _support_y(profile, fr, m)) is not None]
    if not sup:
        raise RuntimeError("support surface below the object not seen")
    support_y = float(np.median(sup))
    axes = [a for fr, m in views if (a := _view_axis(fr, m, fr.K_depth, support_y)) is not None]
    axis = np.median(np.array([a[0] for a in axes]), axis=0)

    # radius profile from all views against the common axis
    samples = []
    per_view_max = []
    for fr, m in views:
        K = fr.K_depth
        rows = np.nonzero(m.any(1))[0]
        rmax = 0.0
        for v in rows:
            cols = np.nonzero(m[v])[0]
            ul, ur = cols.min(), cols.max() + 1
            vals = []
            for u in (ul, ur):
                o, d = _ray(fr, float(u), v + 0.5, K)
                dist, t = _closest_to_vertical(o, d, axis)
                vals.append((dist, t))
            r = 0.5 * (vals[0][0] + vals[1][0])
            hgt = 0.5 * (vals[0][1] + vals[1][1]) - support_y
            samples.append((hgt, r))
            rmax = max(rmax, r)
        per_view_max.append(rmax)
    S = np.array(samples)
    S = S[(S[:, 0] > -0.01)]
    height = float(np.percentile(S[:, 0], 99.5))
    bins = np.arange(0.0, height + cfg.bin_size, cfg.bin_size)
    prof = []
    for b0 in bins:
        sel = (S[:, 0] >= b0) & (S[:, 0] < b0 + cfg.bin_size)
        if sel.sum() >= 2:
            prof.append((float(np.median(S[sel, 1])), b0 + cfg.bin_size / 2))
    prof = np.array(prof)
    spread = float(np.std(per_view_max) / max(np.mean(per_view_max), 1e-6))
    info = {"side_views": len(views), "support_y_capture": round(support_y, 4), "height_m": round(height, 4),
            "max_radius_m": round(float(prof[:, 0].max()), 4) if len(prof) else None,
            "view_spread": round(spread, 3), "axis_capture_xz": np.round(axis, 4).tolist()}

    # top view: rim radius from the outline seen from above (independent check)
    if top is not None:
        rims = []
        for fr in top.frames:
            m = load_label_image(fr.extra_path("object_mask"), fr.depth_size)
            if m is None or (m > 127).sum() < 20:
                continue
            vv, uu = np.nonzero(m > 127)
            pts = []
            for u, v in zip(uu, vv):
                o, d = _ray(fr, u + 0.5, v + 0.5, fr.K_depth)
                if abs(d[1]) > 1e-6:
                    pts.append(o + (support_y + height - o[1]) / d[1] * d)
            pts = np.array(pts)
            # area of the projected pixels -> equivalent radius
            K = fr.K_depth
            depth_to_rim = np.median(np.linalg.norm(pts - fr.pose_cv[:3, 3], axis=1))
            px_area = (depth_to_rim / K[0, 0]) * (depth_to_rim / K[1, 1])
            rims.append(float(np.sqrt(len(pts) * px_area / np.pi)))
        if rims:
            info["top_view_rim_radius_m"] = round(float(np.median(rims)), 4)
            if len(prof):
                info["rim_radius_side_m"] = round(float(prof[-5:, 0].max()), 4)

    use_fallback = len(views) < cfg.min_views or len(prof) < 5 or spread > cfg.max_profile_spread
    if use_fallback:
        r = float(np.median(per_view_max)) if per_view_max else 0.03
        prof = np.array([(r, 0.0), (r, height)])
        info["shape"] = "cylinder (fallback)"
        log.warning("transparent object: profile unreliable (%d views, spread %.2f) -> cylinder", len(views), spread)
    else:
        # light smoothing, flat bottom at z = 0 and flat top at the rim
        r = np.convolve(np.pad(prof[:, 0], 2, mode="edge"), np.ones(5) / 5, mode="valid")
        prof = np.column_stack([r, prof[:, 1]])
        prof = np.vstack([[prof[0, 0], 0.0], prof, [prof[-1, 0], height]])
        info["shape"] = "surface of revolution"
    mesh = lathe(prof, cfg.segments)
    mesh.remove_duplicated_vertices()
    V, F = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    T = np.eye(4)  # object (z up) -> capture (y up)
    T[:3, 0], T[:3, 1], T[:3, 2] = [1, 0, 0], [0, 0, -1], UP
    T[:3, 3] = [axis[0], support_y, axis[1]]
    log.info("transparent object: %s, %.1f cm tall, max radius %.1f cm, %d views", info["shape"],
             height * 100, prof[:, 0].max() * 100, len(views))
    return TransparentResult(V, F, T, prof, info)
