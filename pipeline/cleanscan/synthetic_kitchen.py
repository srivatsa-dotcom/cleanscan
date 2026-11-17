"""Synthetic kitchen capture for the RL-scene pipeline (tests and demos).

    cleanscan synth-kitchen kitchen_session
    cleanscan process-rl kitchen_session -o kitchen_scene && cleanscan validate kitchen_scene

What it writes (docs/capture_format.md + docs/RL_SCENES_PLAN.md §2):

* the room pass: a walk around the kitchen with a person walking through, sensor depth,
  people masks, per-frame instance labels (``instances/``) and ``tracks.json`` as the app's
  live detector would produce them (COCO classes: refrigerator, bottle, cup)
* ``selection.json``: what the user ticked, including the cupboard added by tapping
* one capture session per selected object under ``captures/<id>/<stage>/``, in the same
  world frame: bottles moved to the floor and orbited, fridge and cupboard scanned closed
  and open, the glass captured from the side (phone level) and from the top
* ``truth.json`` + ``truth/*.obj``: ground-truth meshes, poses, joints and masses (tests only)

The kitchen is authored in a Z-up frame K (x right, y towards the back wall, z up, floor at
z = 0) and converted to the phone's Y-up capture frame: (x, y, z)_K -> (x, z, -y).
Glass is invisible to depth (rays pass through), as for real LiDAR/ToF sensors, and only
tints the colour image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from . import physics
from .raycast import camera_rays
from .session import GL_TO_CV

ROOM_K = (5.0, 4.0, 2.6)  # x, y, z extent of the kitchen, centred on x/y, floor at z = 0
K_TO_Y = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, -1.0, 0, 0], [0, 0, 0, 1.0]])  # K (Z up) -> capture (Y up)
COUNTER = ((-1.0, 1.4, 0.0), (1.4, 2.0, 0.9))  # min, max in K
PERSON_PATH_K = ((-1.6, 0.2), (1.6, 0.9))


# --------------------------------------------------------------------------- meshes


def box(lo, hi) -> o3d.geometry.TriangleMesh:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    b = o3d.geometry.TriangleMesh.create_box(*(hi - lo))
    return b.translate(lo)


def boxes(*pairs) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh()
    for lo, hi in pairs:
        m += box(lo, hi)
    return m


def lathe(profile, segments: int = 40) -> o3d.geometry.TriangleMesh:
    """Closed surface of revolution about +z from a (radius, z) profile, bottom to top."""
    prof = np.asarray(profile, float)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    V = [[0.0, 0.0, prof[0, 1]]]
    for r, z in prof:
        V += [[r * np.cos(a), r * np.sin(a), z] for a in ang]
    V.append([0.0, 0.0, prof[-1, 1]])
    n, top = len(prof), len(V) - 1
    F = []
    ring = lambda i, k: 1 + i * segments + (k % segments)  # noqa: E731
    for k in range(segments):
        F.append([0, ring(0, k + 1), ring(0, k)])
        F.append([top, ring(n - 1, k), ring(n - 1, k + 1)])
        for i in range(n - 1):
            a, b, c, d = ring(i, k), ring(i, k + 1), ring(i + 1, k), ring(i + 1, k + 1)
            F += [[a, b, d], [a, d, c]]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.array(V)), o3d.utility.Vector3iVector(np.array(F)))
    vol, _, _ = physics._solid_integrals(np.asarray(m.vertices), np.asarray(m.triangles))
    if vol < 0:
        m.triangles = o3d.utility.Vector3iVector(np.asarray(m.triangles)[:, [0, 2, 1]])
    return m


def transformed(m: o3d.geometry.TriangleMesh, T: np.ndarray) -> o3d.geometry.TriangleMesh:
    out = o3d.geometry.TriangleMesh(m)
    return out.transform(T)


def rot_z(theta: float) -> np.ndarray:
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def translate(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = p
    return T


def joint_motion(j: dict, value: float) -> np.ndarray:
    """Transform (object frame) of the child link at joint value ``value``."""
    axis, origin = np.asarray(j["axis"], float), np.asarray(j["origin"], float)
    if j["type"] == "prismatic":
        return translate(axis * value)
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * value)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = origin - R @ origin
    return T


# --------------------------------------------------------------------------- the kitchen


@dataclass
class KObject:
    id: str
    cls: str  # class the user ends up with (selection)
    mode: str  # pickable | articulated | transparent
    pose_k: np.ndarray  # object -> K
    links: dict  # name -> mesh in the object frame (root first)
    color: tuple
    joints: list = field(default_factory=list)
    detected: tuple | None = None  # (coco class name, coco id) seen by the phone's detector
    track_id: int = 0
    link_classes: dict = field(default_factory=dict)
    open_value: float = 0.0


def _bottle(r, h):
    neck = 0.013
    return lathe([(r * 0.92, 0.0), (r, 0.006), (r, h * 0.62), (r * 0.85, h * 0.7), (neck * 1.3, h * 0.84),
                  (neck, h * 0.88), (neck, h * 0.98), (neck * 1.1, h)])


def kitchen_objects() -> list[KObject]:
    # fridge: body = steel shell open at the front (with a shelf), door hinged on its +x edge
    w, d, h = 0.70, 0.65, 1.80
    fx, fy = w / 2, d / 2
    body = boxes(((-fx, fy - 0.03, 0), (fx, fy, h)),  # back
                 ((-fx, -fy, 0), (-fx + 0.03, fy - 0.03, h)), ((fx - 0.03, -fy, 0), (fx, fy - 0.03, h)),  # sides
                 ((-fx + 0.03, -fy, h - 0.03), (fx - 0.03, fy - 0.03, h)),  # top
                 ((-fx + 0.03, -fy, 0), (fx - 0.03, fy - 0.03, 0.06)),  # bottom
                 ((-fx + 0.03, -fy + 0.05, 0.90), (fx - 0.03, fy - 0.03, 0.92)))  # shelf
    door = box((-fx, -fy - 0.05, 0.01), (fx, -fy - 0.005, h - 0.01))
    fridge = KObject("fridge", "refrigerator", "articulated", translate((-1.95, 2.0 - fy, 0.0)),
                     {"body": body, "door": door}, (0.92, 0.93, 0.95),
                     joints=[{"name": "door_hinge", "type": "revolute", "parent": "body", "child": "door",
                              "axis": [0.0, 0.0, 1.0], "origin": [fx, -fy - 0.05, 0.0],
                              "lower": 0.0, "upper": 1.75}],
                     detected=("refrigerator", 72), track_id=1, open_value=1.75,
                     link_classes={"door": "refrigerator"})

    # cupboard: carcass with a fixed lower front and a drawer at the top that slides out (-y)
    w, d, h = 0.80, 0.56, 0.85
    cx, cy = w / 2, d / 2
    carcass = boxes(((-cx, cy - 0.02, 0), (cx, cy, h)),
                    ((-cx, -cy, 0), (-cx + 0.02, cy - 0.02, h)), ((cx - 0.02, -cy, 0), (cx, cy - 0.02, h)),
                    ((-cx + 0.02, -cy, h - 0.02), (cx - 0.02, cy - 0.02, h)),
                    ((-cx + 0.02, -cy, 0), (cx - 0.02, cy - 0.02, 0.02)),
                    ((-cx + 0.02, -cy, 0.55), (cx - 0.02, cy - 0.02, 0.57)),
                    ((-cx + 0.02, -cy, 0.02), (cx - 0.02, -cy + 0.02, 0.55)))
    drawer = boxes(((-cx + 0.025, -cy - 0.02, 0.575), (cx - 0.025, -cy, 0.825)),  # front panel
                   ((-0.35, -cy, 0.58), (0.35, 0.22, 0.59)),  # tray bottom
                   ((-0.36, -cy, 0.58), (-0.35, 0.22, 0.78)), ((0.35, -cy, 0.58), (0.36, 0.22, 0.78)),
                   ((-0.35, 0.21, 0.59), (0.35, 0.22, 0.78)))
    cupboard = KObject("cupboard", "cupboard", "articulated", translate((1.95, 2.0 - cy, 0.0)),
                       {"body": carcass, "drawer": drawer}, (0.62, 0.45, 0.30),
                       joints=[{"name": "drawer_slide", "type": "prismatic", "parent": "body",
                                "child": "drawer", "axis": [0.0, -1.0, 0.0], "origin": [0.0, -cy, 0.7],
                                "lower": 0.0, "upper": 0.30}],
                       track_id=101, open_value=0.30, link_classes={"drawer": "drawer"})

    top = COUNTER[1][2]
    bottle_a = KObject("bottle_a", "bottle", "pickable", translate((0.1, 1.70, top)), {"base": _bottle(0.035, 0.25)},
                       (0.15, 0.55, 0.25), detected=("bottle", 39), track_id=2)
    bottle_b = KObject("bottle_b", "bottle", "pickable", translate((0.5, 1.62, top)) @ rot_z(0.4),
                       {"base": _bottle(0.042, 0.30)}, (0.2, 0.35, 0.75), detected=("bottle", 39), track_id=3)
    glass = KObject("glass", "glass", "transparent", translate((0.95, 1.68, top)),
                    {"base": lathe([(0.0285, 0.0), (0.030, 0.003), (0.038, 0.11)])},
                    (0.85, 0.92, 0.95), detected=("cup", 41), track_id=4)
    return [fridge, cupboard, bottle_a, bottle_b, glass]


def static_kitchen() -> o3d.geometry.TriangleMesh:
    sx, sy, sz = ROOM_K
    room = box((-sx / 2, -sy / 2, 0), (sx / 2, sy / 2, sz))  # seen from inside
    return room + box(*COUNTER)


def _static_placed(static_y, counter_y) -> list:
    return [Placed(static_y, 0, (0.85, 0.82, 0.75)), Placed(counter_y, 0, (0.45, 0.42, 0.40))]


def person_k(s: float) -> o3d.geometry.TriangleMesh:
    a, b = np.array(PERSON_PATH_K[0]), np.array(PERSON_PATH_K[1])
    p = a + (b - a) * s
    return o3d.geometry.TriangleMesh.create_cylinder(0.25, 1.75, 24).translate((p[0], p[1], 0.875))


# --------------------------------------------------------------------------- cameras


def look_at(eye_k, target_k, up_k=(0, 0, 1)) -> np.ndarray:
    """camera -> capture world (Y up), OpenGL camera axes, from K-frame points."""
    to_y = K_TO_Y[:3, :3]
    eye, tgt, up = to_y @ np.asarray(eye_k, float), to_y @ np.asarray(target_k, float), to_y @ np.asarray(up_k, float)
    f = tgt - eye
    f /= np.linalg.norm(f)
    if abs(f @ up) > 0.99:
        up = to_y @ np.array([0.0, 1.0, 0.0])  # looking straight down: phone top towards the back wall
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, -f, eye
    return T


def room_pass_pose(s: float) -> np.ndarray:
    """Walk a loop around the middle of the kitchen, phone facing the walls and counters."""
    a = 2 * np.pi * s + np.pi / 2  # start facing the back wall
    eye = np.array([0.9 * np.cos(a), 0.55 * np.sin(a) + 0.3, 1.45])
    look = np.array([np.cos(a), np.sin(a), 0.0])
    tgt = eye + look + np.array([0, 0, -0.45 + 0.15 * np.sin(6 * np.pi * s)])
    return look_at(eye, tgt)


def orbit_poses(center_k, radius, height, n, turns=1.0, wobble=0.1) -> list[np.ndarray]:
    out = []
    for i in range(n):
        a = 2 * np.pi * turns * i / n
        eye = np.asarray(center_k) + [radius * np.cos(a), radius * np.sin(a), height + wobble * np.sin(3 * a)]
        out.append(look_at(eye, center_k))
    return out


def arc_poses(center_k, front_dir_k, dist, n, half_angle=np.radians(40), heights=(1.0, 1.6)):
    out = []
    f = np.asarray(front_dir_k, float)
    for i in range(n):
        s = i / max(n - 1, 1)
        a = -half_angle + 2 * half_angle * s
        d = rot_z(a)[:3, :3] @ f
        eye = np.asarray(center_k) + d * dist
        eye[2] = heights[0] + (heights[1] - heights[0]) * (0.5 + 0.5 * np.sin(4 * np.pi * s))
        out.append(look_at(eye, center_k))
    return out


# --------------------------------------------------------------------------- rendering


@dataclass
class Placed:
    mesh_y: o3d.geometry.TriangleMesh  # in the capture frame
    label: int  # instance/track id (0 = room), -1 = person
    color: tuple
    transparent: bool = False


def _shade(points, base):
    checker = (np.floor((points + 0.06) / 0.12).astype(np.int64).sum(-1) % 2).astype(np.float32)
    stripes = 0.5 + 0.5 * np.sin(points[:, 0] * 9.0 + points[:, 2] * 7.0 + points[:, 1] * 5.0)
    return base * (0.7 + 0.18 * checker + 0.12 * stripes)[:, None]


def _render(placed: list[Placed], K, size, pose_gl):
    """(rgb uint8, depth m, label image) for one camera; glass only tints the colour."""
    w, h = size
    pose_cv = pose_gl @ GL_TO_CV
    rays = camera_rays(K, pose_cv, w, h)
    solid = o3d.t.geometry.RaycastingScene()
    ids = {}
    for p in placed:
        if not p.transparent:
            ids[solid.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(p.mesh_y))] = p
    ans = solid.cast_rays(o3d.core.Tensor(rays))
    t = ans["t_hit"].numpy()
    g = ans["geometry_ids"].numpy().astype(np.int64)
    hit = np.isfinite(t)
    pts = rays[..., :3] + rays[..., 3:] * np.where(hit, t, 0)[..., None]
    base = np.full((h, w, 3), 0.5, np.float32)
    label = np.zeros((h, w), np.int32)
    for gid, p in ids.items():
        sel = hit & (g == gid)
        base[sel] = p.color
        label[sel] = p.label
    # floor tiles a bit darker than walls
    rgb = _shade(pts.reshape(-1, 3), base.reshape(-1, 3)).reshape(h, w, 3)
    glass = [p for p in placed if p.transparent]
    if glass:
        gs = o3d.t.geometry.RaycastingScene()
        gids = {gs.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(p.mesh_y)): p for p in glass}
        ga = gs.cast_rays(o3d.core.Tensor(rays))
        gt = ga["t_hit"].numpy()
        gg = ga["geometry_ids"].numpy().astype(np.int64)
        for gid, p in gids.items():
            sel = np.isfinite(gt) & (gg == gid) & (~hit | (gt < t))
            rgb[sel] = 0.55 * rgb[sel] + 0.45 * np.asarray(p.color)  # tinted, edges show the silhouette
            label[sel] = p.label
    depth = np.where(hit, t, 0.0).astype(np.float32)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8), depth, label


def _K(size, fov=np.radians(60)):
    w, h = size
    f = (w / 2) / np.tan(fov / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def write_session(out: Path, poses: list[np.ndarray], scene_at, color_size, depth_size, rng, t0=0.0,
                  noise=0.004, with_depth=True, people_masks=True, instances=False,
                  object_mask_label: int | None = None, dt=0.1) -> dict:
    """Render a capture session. ``scene_at(i) -> list[Placed]``. Returns per-label stats."""
    out = Path(out)
    dirs = ["color", "depth", "confidence", "mask"] + (["instances"] if instances else []) \
        + (["object_mask"] if object_mask_label is not None else [])
    for d in dirs:
        (out / d).mkdir(parents=True, exist_ok=True)
    Kc, Kd = _K(color_size), _K(depth_size)
    lines, stats = [], {}
    for i, pose in enumerate(poses):
        placed = scene_at(i)
        rgb, _, _ = _render(placed, Kc, color_size, pose)
        _, depth, lab = _render(placed, Kd, depth_size, pose)
        depth = depth * (1 + noise * rng.standard_normal(depth.shape)).astype(np.float32)
        name = f"{i:06d}"
        cv2.imwrite(str(out / "color" / f"{name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        rec = {"index": i, "timestamp": round(t0 + i * dt, 4), "color": f"color/{name}.jpg",
               "color_intrinsics": [Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2]], "color_size": list(color_size),
               "depth_intrinsics": [Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2]], "depth_size": list(depth_size),
               "camera_to_world": pose.reshape(-1).tolist()}
        if with_depth:
            cv2.imwrite(str(out / "depth" / f"{name}.png"), np.clip(depth * 1000, 0, 65535).astype(np.uint16))
            cv2.imwrite(str(out / "confidence" / f"{name}.png"), np.full(depth.shape, 2, np.uint8))
            rec["depth"], rec["confidence"] = f"depth/{name}.png", f"confidence/{name}.png"
        if people_masks:
            cv2.imwrite(str(out / "mask" / f"{name}.png"), ((lab == -1) * 255).astype(np.uint8))
            rec["mask"] = f"mask/{name}.png"
        if instances:
            cv2.imwrite(str(out / "instances" / f"{name}.png"), np.clip(lab, 0, 65535).astype(np.uint16))
            rec["instances"] = f"instances/{name}.png"
        if object_mask_label is not None:
            cv2.imwrite(str(out / "object_mask" / f"{name}.png"), ((lab == object_mask_label) * 255).astype(np.uint8))
            rec["object_mask"] = f"object_mask/{name}.png"
        for lbl in np.unique(lab):
            if lbl > 0:
                st = stats.setdefault(int(lbl), {"frames": [], "pixels": 0})
                st["frames"].append(i)
                st["pixels"] += int((lab == lbl).sum())
        lines.append(json.dumps(rec))
    (out / "frames.jsonl").write_text("\n".join(lines) + "\n")
    (out / "session.json").write_text(json.dumps({
        "format": "cleanscan-capture", "version": 1, "platform": "synthetic", "device": "synthetic kitchen",
        "depth_unit": 0.001, "confidence_levels": 3, "world_up": "y", "camera_convention": "opengl",
        "depth_source": "synthetic" if with_depth else "none",
        "extensions": ["objects"]}, indent=2))
    return stats


# --------------------------------------------------------------------------- session


def make_kitchen_session(out: str | Path, room_frames: int = 140, capture_frames: int = 40,
                         color_size=(640, 480), depth_size=(256, 192), seed: int = 0,
                         with_depth: bool = True) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    objs = kitchen_objects()
    sx, sy, sz = ROOM_K
    static_y = transformed(box((-sx / 2, -sy / 2, 0), (sx / 2, sy / 2, sz)), K_TO_Y)
    counter_y = transformed(box(*COUNTER), K_TO_Y)
    room_static = _static_placed(static_y, counter_y)

    def link_mesh_y(o: KObject, name: str, joint_value: float = 0.0, pose_k=None):
        m = o.links[name]
        T = K_TO_Y @ (o.pose_k if pose_k is None else pose_k)
        for j in o.joints:
            if j["child"] == name:
                T = T @ joint_motion(j, joint_value)
        return transformed(m, T)

    def placed_objects(skip=(), moved=None, opened=None):
        res = []
        for o in objs:
            if o.id in skip:
                continue
            pose = moved[1] if moved and moved[0] == o.id else None
            for name in o.links:
                val = o.open_value if opened == o.id else 0.0
                res.append(Placed(link_mesh_y(o, name, val, pose), o.track_id, o.color,
                                  transparent=o.mode == "transparent"))
        return res

    # ---- room pass ---------------------------------------------------------------------
    def room_scene(i):
        s = i / max(room_frames - 1, 1)
        return room_static + [Placed(transformed(person_k(s), K_TO_Y), -1, (0.9, 0.1, 0.1))] + placed_objects()

    poses = [room_pass_pose(i / room_frames) for i in range(room_frames)]
    stats = write_session(out, poses, room_scene, color_size, depth_size, rng, with_depth=with_depth,
                          instances=True)
    t_end = room_frames * 0.1

    # ---- what the phone's detector reports (cupboard: not a COCO class -> added by tap) ---
    (out / "thumbs").mkdir(exist_ok=True)
    tracks = []
    for o in objs:
        st = stats.get(o.track_id)
        if not st:
            continue
        center = K_TO_Y @ o.pose_k @ np.array([0, 0, 0.1, 1])
        if o.detected:
            fr = st["frames"][len(st["frames"]) // 2]
            img = cv2.imread(str(out / "color" / f"{fr:06d}.jpg"))
            lab = cv2.imread(str(out / "instances" / f"{fr:06d}.png"), cv2.IMREAD_UNCHANGED)
            ys, xs = np.nonzero(cv2.resize((lab == o.track_id).astype(np.uint8), (img.shape[1], img.shape[0]),
                                           interpolation=cv2.INTER_NEAREST))
            if len(xs):
                cv2.imwrite(str(out / "thumbs" / f"{o.track_id}.jpg"),
                            img[max(ys.min() - 8, 0):ys.max() + 8, max(xs.min() - 8, 0):xs.max() + 8])
            tracks.append({"id": o.track_id, "class": o.detected[0], "coco_id": o.detected[1], "score": 0.82,
                           "frames": len(st["frames"]), "first_frame": st["frames"][0],
                           "thumb": f"thumbs/{o.track_id}.jpg",
                           "approx_position_world": np.round(center[:3] + rng.normal(0, 0.03, 3), 3).tolist()})
    (out / "tracks.json").write_text(json.dumps(tracks, indent=1))
    # the cupboard was not detected: drop its labels from the instance images (the app only
    # labels tracks), the user taps it on a keyframe instead
    tap_obj = next(o for o in objs if not o.detected)
    for p in sorted((out / "instances").glob("*.png")):
        lab = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if (lab == tap_obj.track_id).any():
            lab[lab == tap_obj.track_id] = 0
            cv2.imwrite(str(p), lab)
    st = stats[tap_obj.track_id]
    seed_world = K_TO_Y @ tap_obj.pose_k @ np.array([0.0, -0.28, 0.4, 1.0])  # front of the cupboard

    # ---- guided captures --------------------------------------------------------------
    selection, truth = [], []
    t = t_end + 5.0
    for o in objs:
        caps = {}
        base_dir = out / "captures" / str(o.track_id)
        if o.mode == "pickable":
            spot_k = translate((0.0 if o.id == "bottle_a" else 0.35, -0.7, 0.0)) @ rot_z(0.9)
            center = spot_k[:3, 3] + [0, 0, 0.13]
            p = orbit_poses(center, 0.5, 0.42, capture_frames, turns=1.05)
            write_session(base_dir / "scan", p,
                          lambda i, o=o, spot_k=spot_k: room_static
                          + placed_objects(moved=(o.id, spot_k)),
                          color_size, depth_size, rng, t0=t, with_depth=with_depth, object_mask_label=o.track_id)
            caps["scan"] = f"captures/{o.track_id}/scan"
            t += capture_frames * 0.1 + 5
        elif o.mode == "articulated":
            front = o.pose_k[:3, :3] @ np.array([0, -1.0, 0])
            height = 0.9 if o.id == "fridge" else 0.45
            center = o.pose_k[:3, 3] + [0, 0, height]
            for stage, opened in (("closed", None), ("open", o.id)):
                # the user frames the whole object: a fridge needs ~2 m distance
                p = arc_poses(center, front, 2.0 if o.id == "fridge" else 1.5, capture_frames,
                              heights=(0.9, 1.7) if o.id == "fridge" else (0.9, 1.5))
                write_session(base_dir / stage, p,
                              lambda i, opened=opened: room_static + placed_objects(opened=opened),
                              color_size, depth_size, rng, t0=t, with_depth=with_depth)
                caps[stage] = f"captures/{o.track_id}/{stage}"
                t += capture_frames * 0.1 + 5
        else:  # transparent: side profile with the phone level, then one top-down shot
            c = o.pose_k[:3, 3]
            mid = c + [0, 0, 0.055]
            n = max(capture_frames // 3, 6)
            side = []
            for k in range(n):
                a = np.radians(-90 - 50 + 100 * k / max(n - 1, 1))  # in front of the glass (-y side)
                eye = mid + [0.33 * np.cos(a), 0.33 * np.sin(a), 0.0]
                side.append(look_at(eye, mid))
            top = [look_at(c + [0.01 * k, 0, 0.42], c + [0.01 * k, 0, 0]) for k in range(3)]
            for stage, p in (("profile", side), ("top", top)):
                write_session(base_dir / stage, p,
                              lambda i: room_static + placed_objects(),
                              color_size, depth_size, rng, t0=t, with_depth=with_depth,
                              object_mask_label=o.track_id)
                caps[stage] = f"captures/{o.track_id}/{stage}"
                t += len(p) * 0.1 + 5
        sel = {"id": o.track_id, "class": o.cls, "name": o.id, "mode": o.mode, "captures": caps,
               "source": "detected" if o.detected else "tap"}
        if not o.detected:
            fr = st["frames"][len(st["frames"]) // 2]
            sel["seed"] = {"frame": fr, "world": np.round(seed_world[:3], 4).tolist()}
        selection.append(sel)

        # ---- ground truth ----------------------------------------------------------------
        (out / "truth").mkdir(exist_ok=True)
        links = {}
        masses = {}
        table = physics.load_materials()
        for name, m in o.links.items():
            path = out / "truth" / f"{o.id}_{name}.obj"
            m2 = o3d.geometry.TriangleMesh(m)
            m2.remove_duplicated_vertices()
            V, F = np.asarray(m2.vertices), np.asarray(m2.triangles)
            from .scene import write_plain_obj

            write_plain_obj(path, V, F)
            links[name] = f"truth/{path.name}"
            spec = physics.resolve(o.link_classes.get(name, o.cls), table)
            masses[name] = physics.mass_properties(V, F, spec).mass
        truth.append({"id": o.id, "track_id": o.track_id, "class": o.cls, "mode": o.mode,
                      "pose_capture": (K_TO_Y @ o.pose_k).tolist(), "links": links,
                      "link_classes": o.link_classes, "joints": o.joints, "open_value": o.open_value,
                      "mass": masses})
    (out / "selection.json").write_text(json.dumps({"objects": selection}, indent=1))
    (out / "truth.json").write_text(json.dumps({
        "note": "ground truth for tests; poses map the object frame (Z up) into the capture frame (Y up)",
        "objects": truth}, indent=1))
    return out


# --------------------------------------------------------------------------- truth -> scene


def load_truth(session_dir: Path) -> dict:
    return json.loads((Path(session_dir) / "truth.json").read_text())


def objects_from_truth(session_dir: Path, T: np.ndarray, scene_dir: Path, table: dict, cfg) -> list:
    """Scene objects built from the ground truth (for tests of export and validation only):
    exact meshes and joints, placed with the capture -> scene transform ``T``."""
    from .rl import build_object
    from .scene import read_obj_vf

    out = []
    for t in load_truth(session_dir)["objects"]:
        links = []
        for name, rel in t["links"].items():
            V, F = read_obj_vf(Path(session_dir) / rel)
            links.append({"name": name, "V": V, "F": F, "class": t["link_classes"].get(name, t["class"])})
        type_ = "articulated" if t["mode"] == "articulated" else "pickable"
        out.append(build_object(scene_dir, t["id"], t["class"], type_, T @ np.asarray(t["pose_capture"]),
                                links, t["joints"], table, cfg, name=t["id"],
                                info={"source": "truth", "mode": t["mode"]}))
    return out
