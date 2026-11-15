"""The CleanScan RL scene bundle: ``scene.json`` + room + objects (see docs/RL_SCENES_PLAN.md).

Conventions everywhere: metres, kilograms, Z up, floor at z = 0, quaternions (w, x, y, z).

Every object has its own frame (``pose`` = object -> world). All link meshes, link
centres of mass, inertias and joint origins/axes are expressed in that object frame, so
link frames coincide with the object frame in the closed / rest configuration. That keeps
the three exporters (MJCF, USD, SDF) simple and identical in meaning.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from . import physics

log = logging.getLogger(__name__)

FORMAT = "cleanscan-scene"
VERSION = 1
TYPES = ("pickable", "articulated", "static")
JOINT_TYPES = ("revolute", "prismatic")


# --------------------------------------------------------------------------- data model


@dataclass
class Link:
    name: str
    visual: str | None  # path relative to the scene folder
    collision: list[str]  # convex pieces, relative paths
    mass: float
    com: list[float]
    inertia: list[list[float]]  # about the COM, object frame
    material: str = "generic"
    friction: float = 0.5
    restitution: float = 0.1
    mass_model: dict = field(default_factory=dict)


@dataclass
class Joint:
    name: str
    type: str  # revolute | prismatic
    parent: str
    child: str
    axis: list[float]  # unit vector, object frame
    origin: list[float]  # a point on the axis, object frame
    lower: float  # rad (revolute) or m (prismatic)
    upper: float
    damping: float = 0.5
    friction: float = 0.05
    effort: float = 50.0
    velocity: float = 2.0


@dataclass
class SceneObject:
    id: str
    cls: str
    type: str  # pickable | articulated | static
    pose: np.ndarray  # 4x4 object -> world
    links: list[Link]
    joints: list[Joint] = field(default_factory=list)
    name: str | None = None
    fixed_base: bool = False  # articulated: the root link is welded to the world
    overrides: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)  # provenance, quality metrics

    @property
    def root(self) -> Link:
        return self.links[0]

    def link(self, name: str) -> Link:
        return next(lk for lk in self.links if lk.name == name)


@dataclass
class Scene:
    name: str
    room: dict  # paths of the static room files (relative to the scene folder)
    objects: list[SceneObject] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def pose_to_dict(T: np.ndarray) -> dict:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x y z w
    return {"position": [round(float(v), 6) for v in T[:3, 3]],
            "quaternion": [round(float(v), 8) for v in (q[3], q[0], q[1], q[2])]}


def pose_from_dict(d: dict) -> np.ndarray:
    w, x, y, z = d.get("quaternion", (1, 0, 0, 0))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = d.get("position", (0, 0, 0))
    return T


def _round(x, nd=8):
    return np.round(np.asarray(x, np.float64), nd).tolist()


def scene_to_json(scene: Scene) -> dict:
    objs = []
    for o in scene.objects:
        d = {"id": o.id, "name": o.name or o.id, "class": o.cls, "type": o.type,
             "pose": pose_to_dict(o.pose), "fixed_base": o.fixed_base,
             "links": [{**asdict(lk), "com": _round(lk.com), "inertia": _round(lk.inertia, 10),
                        "mass": round(lk.mass, 6)} for lk in o.links],
             "joints": [asdict(j) for j in o.joints],
             "overrides": o.overrides, "info": o.info}
        if o.type == "pickable":  # flat copy of the single link for convenience
            lk = o.root
            d.update(mass=round(lk.mass, 6), com=_round(lk.com), inertia=_round(lk.inertia, 10),
                     friction=lk.friction, material=lk.material, visual=lk.visual,
                     collision=lk.collision)
        objs.append(d)
    return {"format": FORMAT, "version": VERSION, "units": "m", "up": "z",
            "name": scene.name, "room": scene.room, "objects": objs, "info": scene.info}


def save_scene(scene: Scene, scene_dir: Path) -> Path:
    path = Path(scene_dir) / "scene.json"
    path.write_text(json.dumps(scene_to_json(scene), indent=2, default=_json_default))
    return path


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    raise TypeError(type(o))


def load_scene(scene_dir: Path, apply_overrides: bool = True) -> Scene:
    """Read ``scene.json``. With ``apply_overrides`` the user's ``overrides`` are applied
    (the stored computed values stay untouched in the file)."""
    d = json.loads((Path(scene_dir) / "scene.json").read_text())
    if d.get("format") != FORMAT:
        raise ValueError(f"{scene_dir}: not a CleanScan scene")
    objs = []
    for od in d["objects"]:
        links = [Link(**lk) for lk in od["links"]]
        joints = [Joint(**j) for j in od.get("joints", [])]
        o = SceneObject(id=od["id"], cls=od["class"], type=od["type"], pose=pose_from_dict(od["pose"]),
                        links=links, joints=joints, name=od.get("name"),
                        fixed_base=od.get("fixed_base", False), overrides=od.get("overrides", {}),
                        info=od.get("info", {}))
        if apply_overrides and o.overrides:
            apply_object_overrides(o, o.overrides)
        objs.append(o)
    return Scene(d["name"], d["room"], objs, d.get("info", {}))


def _scale_link_mass(lk: Link, factor: float) -> None:
    lk.mass *= factor
    lk.inertia = (np.asarray(lk.inertia) * factor).tolist()


def apply_object_overrides(o: SceneObject, ov: dict, table: dict | None = None) -> None:
    """Supported keys: type, pose, fixed_base, mass, friction, restitution, material,
    density, links: {name: {same keys}}, joints: {name: {lower, upper, damping, friction,
    type, axis, origin, ...}}. A material/density change rescales mass and inertia
    (uniform density), ``mass`` sets it directly."""
    if "type" in ov:
        if ov["type"] not in TYPES:
            raise ValueError(f"{o.id}: type must be one of {TYPES}")
        o.type = ov["type"]
    if "pose" in ov:
        o.pose = pose_from_dict(ov["pose"])
    if "fixed_base" in ov:
        o.fixed_base = bool(ov["fixed_base"])
    link_ov = dict(ov.get("links", {}))
    top = {k: ov[k] for k in ("mass", "friction", "restitution", "material", "density") if k in ov}
    if top:  # object-level values apply to the root link (pickables have only one)
        link_ov[o.root.name] = {**top, **link_ov.get(o.root.name, {})}
    for name, lov in link_ov.items():
        lk = o.link(name)
        if "material" in lov or "density" in lov:
            table = table or physics.load_materials()
            mat = table["materials"].get(lov.get("material", lk.material), {})
            new_density = float(lov.get("density", mat.get("density", lk.mass_model.get("density", 0))))
            old_density = float(lk.mass_model.get("density", 0))
            if old_density > 0 and new_density > 0:
                _scale_link_mass(lk, new_density / old_density)
                lk.mass_model["density"] = new_density
            if "material" in lov:
                lk.material = lov["material"]
                lk.friction = float(mat.get("friction", lk.friction))
                lk.restitution = float(mat.get("restitution", lk.restitution))
        if "mass" in lov and lk.mass > 0:
            _scale_link_mass(lk, float(lov["mass"]) / lk.mass)
        for k in ("friction", "restitution"):
            if k in lov:
                setattr(lk, k, float(lov[k]))
    for name, jov in ov.get("joints", {}).items():
        j = next(j for j in o.joints if j.name == name)
        for k, v in jov.items():
            setattr(j, k, v)


# --------------------------------------------------------------------------- mesh files


def write_plain_obj(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    lines += [f"f {a} {b} {c}" for a, b, c in np.asarray(F) + 1]
    Path(path).write_text("\n".join(lines) + "\n")


def read_obj_vf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of an OBJ (positions only)."""
    m = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(m.vertices), np.asarray(m.triangles)


def write_visual(path: Path, V: np.ndarray, F: np.ndarray, colors: np.ndarray | None = None,
                 textured=None) -> None:
    """Visual OBJ: textured (``textured`` = texture.TexturedMesh in the object frame), else
    vertex-coloured mesh baked into a small texture-less MTL colour."""
    from .gazebo import write_obj

    if textured is not None:
        write_obj(path, textured)
        return
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F))
    write_obj(path, None, m)
    if colors is not None and len(colors):
        c = np.clip(np.asarray(colors).reshape(-1, 3).mean(0), 0, 1)
        mtl = Path(path).with_suffix(".mtl")
        mtl.write_text(mtl.read_text().replace("Kd 0.7 0.7 0.7", f"Kd {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}"))


def make_link(scene_dir: Path, rel_dir: str, name: str, V: np.ndarray, F: np.ndarray,
              spec: physics.PhysicsSpec, colors: np.ndarray | None = None, textured=None,
              decomposition: str = "auto", max_hulls: int = 16,
              quality: str = "fast") -> tuple[Link, dict]:
    """Write a link's visual + convex collision pieces under ``scene_dir/rel_dir`` and
    compute its mass properties. ``V``/``F``: watertight mesh in the object frame.
    Returns (link, info for the report)."""
    scene_dir = Path(scene_dir)
    d = scene_dir / rel_dir
    (d / "collision").mkdir(parents=True, exist_ok=True)
    write_visual(d / "visual.obj", V, F, colors, textured)
    pieces, method = physics.convex_decomposition(V, F, decomposition, max_hulls=max_hulls, quality=quality)
    coll = []
    for i, (pv, pf) in enumerate(pieces):
        p = d / "collision" / f"{i:03d}.obj"
        write_plain_obj(p, pv, pf)
        coll.append(str(p.relative_to(scene_dir)))
    mp = physics.mass_properties(V, F, spec)
    lk = Link(name=name, visual=str((d / "visual.obj").relative_to(scene_dir)), collision=coll,
              mass=mp.mass, com=mp.com.tolist(), inertia=mp.inertia.tolist(), material=spec.material,
              friction=spec.friction, restitution=spec.restitution,
              mass_model={**mp.model, "volume": mp.volume})
    return lk, {"convex_pieces": len(pieces), "decomposition": method,
                "watertight": physics.is_watertight(F), "volume_m3": mp.volume, "area_m2": mp.area}


# --------------------------------------------------------------------------- room collision


def support_heights(mesh_zup: o3d.geometry.TriangleMesh, bin_size: float = 0.01,
                    min_area: float = 0.05) -> np.ndarray:
    """Heights of horizontal, upward-facing surfaces (floor, counter tops, shelves)."""
    mesh = copy.deepcopy(mesh_zup)
    mesh.compute_triangle_normals()
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    n = np.asarray(mesh.triangle_normals)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    up = n[:, 2] > 0.95
    if not up.any():
        return np.zeros(0)
    z = (a + b + c)[up, 2] / 3
    w = area[up]
    lo, hi = z.min(), z.max()
    nb = max(int(np.ceil((hi - lo) / bin_size)) + 1, 1)
    hist = np.bincount(((z - lo) / bin_size).astype(int), weights=w, minlength=nb)
    peaks = []
    for i in np.argsort(-hist):
        if hist[i] < min_area:
            break
        if all(abs(i - j) > 2 for j in peaks):
            peaks.append(i)
    out = []
    for i in peaks:  # area-weighted mean height within the peak's bins
        sel = np.abs((z - lo) / bin_size - (i + 0.5)) <= 1.5
        out.append(float(np.average(z[sel], weights=w[sel])))
    return np.sort(np.array(out))


def room_collision_boxes(mesh_zup: o3d.geometry.TriangleMesh, voxel: float = 0.04,
                         skip_floor: bool = True) -> list[dict]:
    """Approximate the static room by axis-aligned boxes (MuJoCo collides meshes as convex
    hulls, so a room needs primitives). Surface voxels are merged greedily into boxes; box
    tops near a detected support surface are snapped to its exact height so objects rest
    exactly where they do on the visual mesh."""
    pts = np.asarray(mesh_zup.sample_points_uniformly(
        max(int(mesh_zup.get_surface_area() / (voxel / 3) ** 2), 1000)).points)
    lo = pts.min(0) - voxel
    idx = np.floor((pts - lo) / voxel).astype(np.int64)
    shape = idx.max(0) + 1
    occ = np.zeros(shape, bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if skip_floor:  # the floor is an infinite plane at z = 0 in every exporter
        floor_k = int(np.floor((0.0 - lo[2]) / voxel))
        occ[:, :, : floor_k + 1] = False
    tops = support_heights(mesh_zup)
    boxes = []
    todo = occ.copy()
    nx, ny, nz = shape
    for k in range(nz):  # greedy: grow in x, then y, then z
        for j in range(ny):
            i = 0
            while i < nx:
                if not todo[i, j, k]:
                    i += 1
                    continue
                i1 = i
                while i1 + 1 < nx and todo[i1 + 1, j, k]:
                    i1 += 1
                j1 = j
                while j1 + 1 < ny and todo[i:i1 + 1, j1 + 1, k].all():
                    j1 += 1
                k1 = k
                while k1 + 1 < nz and todo[i:i1 + 1, j:j1 + 1, k1 + 1].all():
                    k1 += 1
                todo[i:i1 + 1, j:j1 + 1, k:k1 + 1] = False
                bmin = lo + voxel * np.array([i, j, k])
                bmax = lo + voxel * np.array([i1 + 1, j1 + 1, k1 + 1])
                if len(tops):
                    near = tops[np.abs(tops - bmax[2]) <= voxel]
                    if len(near) and near[0] > bmin[2] + 0.005:
                        bmax[2] = near[np.argmin(np.abs(near - bmax[2]))]
                boxes.append({"center": _round((bmin + bmax) / 2, 5), "size": _round(bmax - bmin, 5)})
                i = i1 + 1
    log.info("room collision: %d boxes (voxel %.0f mm)", len(boxes), voxel * 1000)
    return boxes


def object_world_points(o: SceneObject, scene_dir: Path) -> np.ndarray:
    """Collision vertices of all links, in the world frame (rest configuration)."""
    pts = [read_obj_vf(Path(scene_dir) / c)[0] for lk in o.links for c in lk.collision]
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    return P @ o.pose[:3, :3].T + o.pose[:3, 3]


def support_patch(o: SceneObject, scene_dir: Path, heights: np.ndarray, below: float = 0.03,
                  above: float = 0.01, margin: float = 0.02, thickness: float = 0.02) -> tuple[dict, str | None]:
    """A thin static box under a pickable object, at the room's measured support height.

    The room pass rarely sees the surface under and right around an object, so the scanned
    counter often has a hole exactly there. Returns (box, warning or None)."""
    P = object_world_points(o, scene_dir)
    bottom = float(P[:, 2].min())
    cand = heights[(heights >= bottom - below) & (heights <= bottom + above)] if len(heights) else heights
    warn = None
    if len(cand):
        top = float(cand[np.argmin(np.abs(cand - bottom))])
    else:
        top = bottom
        warn = (f"{o.id}: no scanned support surface within {below * 100:.0f} cm below it; "
                f"support patch placed at its bottom (z = {bottom:.3f} m)")
    lo, hi = P[:, :2].min(0) - margin, P[:, :2].max(0) + margin
    box = {"object": o.id, "center": _round([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, top - thickness / 2], 5),
           "size": _round([hi[0] - lo[0], hi[1] - lo[1], thickness], 5), "top": round(top, 5),
           "object_bottom": round(bottom, 5)}
    return box, warn


def _inside_hulls(points: np.ndarray, hulls: list, margin: float) -> np.ndarray:
    from scipy.spatial import ConvexHull

    inside = np.zeros(len(points), bool)
    for H in hulls:
        if len(H) < 4:
            continue
        eq = ConvexHull(H).equations  # outward normal . x + offset <= 0 inside
        lo, hi = H.min(0) - margin, H.max(0) + margin
        cand = np.nonzero(((points >= lo) & (points <= hi)).all(1) & ~inside)[0]
        if len(cand):
            d = points[cand] @ eq[:, :3].T + eq[:, 3]
            inside[cand[(d <= margin).all(1)]] = True
    return inside


def object_hulls_world(o: SceneObject, scene_dir: Path) -> list[np.ndarray]:
    return [read_obj_vf(Path(scene_dir) / c)[0] @ o.pose[:3, :3].T + o.pose[:3, 3]
            for lk in o.links for c in lk.collision]


def carve_triangles(V: np.ndarray, F: np.ndarray, hulls: list, margin: float = 0.02) -> np.ndarray:
    """Mask of triangles to remove: centroid inside a (dilated) hull, or 2+ corners inside."""
    if not hulls or len(F) == 0:
        return np.zeros(len(F), bool)
    cen = V[F].mean(1)
    vin = _inside_hulls(V, hulls, margin)
    return _inside_hulls(cen, hulls, margin) | (vin[F].sum(1) >= 2)


def carve_room(aligned: o3d.geometry.TriangleMesh, textured, hulls: list, margin: float = 0.02):
    """Remove the selected objects' volumes from the static room (mesh + textured mesh).
    Returns (mesh, textured, triangles removed)."""
    from .texture import TexturedMesh

    V, F = np.asarray(aligned.vertices), np.asarray(aligned.triangles)
    drop = carve_triangles(V, F, hulls, margin)
    mesh = copy.deepcopy(aligned)
    mesh.remove_triangles_by_mask(drop)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    if textured is not None:
        tdrop = carve_triangles(textured.vertices, textured.triangles, hulls, margin)
        keep = ~tdrop
        textured = TexturedMesh(textured.vertices, textured.triangles[keep], textured.uvs[keep],
                                textured.normals, textured.texture)
    return mesh, textured, int(drop.sum())


def drop_boxes_in_hulls(boxes: list[dict], hulls: list, margin: float = 0.005) -> list[dict]:
    """Remove room collision boxes that reach into an object (leftovers of its scanned
    surface that carving missed): a box goes if its centre lies within half its smallest
    size (+ margin) of an object hull."""
    if not hulls or not boxes:
        return boxes
    keep = []
    for b in boxes:
        c = np.asarray(b["center"], float)[None]
        r = 0.5 * min(b["size"]) + margin
        if not _inside_hulls(c, hulls, r)[0]:
            keep.append(b)
    if len(keep) < len(boxes):
        log.info("dropped %d room boxes overlapping objects", len(boxes) - len(keep))
    return keep
