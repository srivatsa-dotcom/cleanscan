"""`cleanscan process-rl`: a capture session -> RL scene bundle (see docs/RL_SCENES_PLAN.md).

    <scene>/scene.json  room/  objects/<id>/  export/{scene.usda, scene.xml, world.sdf}  report.json
    <scene>.zip
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import open3d as o3d

from . import physics
from .pipeline import PipelineConfig, export_room, reconstruct_room
from .scene import (Joint, Scene, SceneObject, load_scene, make_link, pose_from_dict,
                    carve_room, clamp_link_above, drop_boxes_in_hulls, object_hulls_world, read_obj_vf, room_collision_boxes,
                    save_scene, support_heights, support_patch)

if TYPE_CHECKING:
    from .articulated import ArticulatedConfig
    from .pickable import PickableConfig
    from .transparent import TransparentConfig

log = logging.getLogger(__name__)


def _pickable_cfg():
    from .pickable import PickableConfig

    return PickableConfig()


def _transparent_cfg():
    from .transparent import TransparentConfig

    return TransparentConfig()


def _articulated_cfg():
    from .articulated import ArticulatedConfig

    return ArticulatedConfig()


@dataclass
class RLConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    materials: str | None = None  # extra materials.yaml merged over the packaged one
    decomposition: str = "auto"  # auto | coacd | vhacd | hull
    max_hulls: int = 16
    collision_quality: str = "fast"  # fast | best (CoACD presets)
    pickable: "PickableConfig" = field(default_factory=lambda: _pickable_cfg())
    articulated: "ArticulatedConfig" = field(default_factory=lambda: _articulated_cfg())
    transparent: "TransparentConfig" = field(default_factory=lambda: _transparent_cfg())
    room_box_voxel: float = 0.04  # MuJoCo room collision boxes
    declared_objects: str | None = None  # JSON file of objects given by hand (see below)
    objects: str = "auto"  # auto: from the app's selection | none | truth (synthetic tests only)
    carve_margin: float = 0.02  # m around each object removed from the static room
    zip: bool = True


# --------------------------------------------------------------------------- objects


def build_object(scene_dir: Path, obj_id: str, cls: str, type_: str, pose: np.ndarray,
                 links: list[dict], joints: list[dict] | None = None, table: dict | None = None,
                 cfg: RLConfig | None = None, name: str | None = None, fixed_base: bool | None = None,
                 info: dict | None = None) -> SceneObject:
    """Create an object from meshes in its own frame.

    ``links``: [{"name", "V", "F", optional "colors", "textured", "physics": {overrides}}],
    root first. ``joints``: [{"name", "type", "parent", "child", "axis", "origin", "lower",
    "upper", ...}] in the object frame."""
    cfg = cfg or RLConfig()
    table = table or physics.load_materials(cfg.materials)
    rel_root = f"objects/{obj_id}"
    if (scene_dir / rel_root).exists():
        shutil.rmtree(scene_dir / rel_root)
    built, link_info = [], {}
    for i, ld in enumerate(links):
        spec = physics.resolve(ld.get("class", cls), table, ld.get("physics"))
        rel = rel_root if len(links) == 1 else f"{rel_root}/links/{ld['name']}"
        lk, li = make_link(scene_dir, rel, ld["name"], np.asarray(ld["V"]), np.asarray(ld["F"]), spec,
                           colors=ld.get("colors"), textured=ld.get("textured"),
                           decomposition=cfg.decomposition, max_hulls=cfg.max_hulls,
                           quality=cfg.collision_quality, opacity=ld.get("opacity", 1.0))
        built.append(lk)
        link_info[ld["name"]] = li
    js = []
    for jd in joints or []:
        axis = np.asarray(jd["axis"], float)
        js.append(Joint(**{**jd, "axis": (axis / np.linalg.norm(axis)).tolist(),
                           "origin": list(map(float, jd["origin"]))}))
        if js[-1].type not in ("revolute", "prismatic"):
            raise ValueError(f"{obj_id}: joint type must be revolute or prismatic")
    if fixed_base is None:
        fixed_base = type_ == "articulated"
    return SceneObject(id=obj_id, cls=cls, type=type_, pose=np.asarray(pose, float), links=built,
                       joints=js, name=name, fixed_base=fixed_base,
                       info={**(info or {}), "links": link_info})


def load_declared_objects(path: Path, scene_dir: Path, table: dict, cfg: RLConfig) -> list[SceneObject]:
    """Objects described by hand (meshes in their own frame + pose), e.g.::

        {"objects": [
          {"id": "bottle_1", "class": "bottle", "type": "pickable", "mesh": "bottle.obj",
           "pose": {"position": [1.2, 0.5, 0.9], "quaternion": [1, 0, 0, 0]}},
          {"id": "cupboard_1", "class": "cupboard", "type": "articulated",
           "pose": {...}, "links": [{"name": "body", "mesh": "body.obj"},
                                     {"name": "door", "mesh": "door.obj"}],
           "joints": [{"name": "hinge", "type": "revolute", "parent": "body", "child": "door",
                       "axis": [0, 0, 1], "origin": [0.3, -0.2, 0], "lower": 0, "upper": 1.6}]}]}

    Mesh paths are relative to the JSON file. Meshes must be closed for solid objects."""
    path = Path(path)
    d = json.loads(path.read_text())
    out = []
    for od in d["objects"]:
        link_defs = od.get("links") or [{"name": "base", "mesh": od["mesh"]}]
        links = []
        for ld in link_defs:
            V, F = read_obj_vf(path.parent / ld["mesh"])
            links.append({"name": ld["name"], "V": V, "F": F, "physics": ld.get("physics")})
        out.append(build_object(scene_dir, od["id"], od["class"], od.get("type", "pickable"),
                                pose_from_dict(od.get("pose", {})), links, od.get("joints"), table, cfg,
                                name=od.get("name"), fixed_base=od.get("fixed_base"),
                                info={"source": "declared"}))
    return out


def objects_from_selection(session_dir: Path, room, scene_dir: Path, table: dict, cfg: RLConfig,
                           heights: np.ndarray, warnings: list[str]) -> list[SceneObject]:
    """The objects the user selected in the app, reconstructed from their guided captures."""
    from . import objects as objmod
    from . import pickable
    from .session import Session

    selected = objmod.load_selection(session_dir)
    if not selected:
        return []
    objmod.locate_objects(room.session, selected, np.asarray(room.aligned.vertices), room.T)
    out = []
    for s in selected:
        info = {"source": s.source, "mode": s.mode, "selection_id": s.id, **s.info}
        if s.mode == "articulated":
            obj = _articulated_object(s, Path(session_dir), room, scene_dir, table, cfg, info, warnings)
            if obj is not None:
                out.append(obj)
            continue
        if s.mode == "transparent":
            obj = _transparent_object(s, Path(session_dir), room, scene_dir, table, cfg, info, warnings, heights)
            if obj is not None:
                out.append(obj)
            continue
        if s.mode != "pickable":
            warnings.append(f"{s.slug}: unknown mode {s.mode!r}; left in the static room")
            continue
        cap = s.captures.get("scan")
        cap_dir = Path(session_dir) / cap if cap else None
        textured = None
        if cap_dir is not None and (cap_dir / "session.json").exists():
            try:
                res = pickable.reconstruct_pickable(Session.load(cap_dir), cfg.pickable)
                V, F, textured = res.V, res.F, res.textured
                info["reconstruction"] = res.info
            except Exception as exc:  # noqa: BLE001 - fall back to the room-pass shape
                warnings.append(f"{s.slug}: capture could not be reconstructed ({exc})")
                V = None
        else:
            V = None
            warnings.append(f"{s.slug}: no object capture; using a primitive from the room scan")
        if s.region is None or len(s.region) < 10:
            warnings.append(f"{s.slug}: not found in the room scan, cannot be placed; skipped")
            continue
        if V is None:
            V, F, pose = pickable.fallback_primitive(s.region, s.cls, heights)
            info["shape"] = "primitive"
        else:
            pose, reg = pickable.register_to_room(V, F, s.region, heights)
            info["registration"] = reg
            info["shape"] = "scanned"
        pose, dz = pickable.settle(pose, V, heights)
        info["settle_dz_m"] = round(dz, 4)
        if abs(dz) > 0.015:
            warnings.append(f"{s.slug}: moved {dz * 100:.1f} cm vertically onto its support; check the registration")
        out.append(build_object(scene_dir, s.slug, s.cls, "pickable", pose,
                                [{"name": "base", "V": V, "F": F, "textured": textured}], None, table, cfg,
                                name=s.name, info=info))
    return out


def _transparent_object(s, session_dir: Path, room, scene_dir: Path, table: dict, cfg: RLConfig,
                        info: dict, warnings: list[str], heights: np.ndarray) -> SceneObject | None:
    from . import pickable, transparent
    from .session import Session

    caps = {k: session_dir / v for k, v in s.captures.items()}
    prof = caps.get("profile")
    if prof is None or not (prof / "session.json").exists():
        warnings.append(f"{s.slug}: no side-profile capture; left in the static room")
        return None
    top = caps.get("top")
    top_ses = Session.load(top) if top is not None and (top / "session.json").exists() else None
    try:
        res = transparent.reconstruct_transparent(Session.load(prof), top_ses, cfg.transparent)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{s.slug}: silhouette reconstruction failed ({exc}); left in the static room")
        return None
    pose = room.T @ res.T_capture_object
    pose, dz = pickable.settle(pose, res.V, heights)
    info = {**info, "reconstruction": res.info, "shape": res.info["shape"], "settle_dz_m": round(dz, 4),
            "profile_r_z": np.round(res.profile, 4).tolist()}
    if res.info["shape"].startswith("cylinder"):
        warnings.append(f"{s.slug}: glass profile unreliable, using a cylinder")
    tv, sv = res.info.get("top_view_rim_radius_m"), res.info.get("rim_radius_side_m")
    if tv and sv and abs(tv - sv) > 0.2 * sv:
        warnings.append(f"{s.slug}: rim radius from the top view ({tv * 100:.1f} cm) and the side "
                        f"({sv * 100:.1f} cm) disagree")
    # (the silhouette never sees the glass's wall thickness; materials.yaml gives it)
    return build_object(scene_dir, s.slug, s.cls, "pickable", pose,
                        [{"name": "base", "V": res.V, "F": res.F, "colors": np.array([[0.85, 0.92, 0.95]]),
                          "opacity": 0.35}], None, table, cfg, name=s.name, info=info)


DRAWER_PARENTS = {"cupboard", "cabinet", "dresser", "desk", "kitchen cabinet", "chest of drawers"}


def _articulated_object(s, session_dir: Path, room, scene_dir: Path, table: dict, cfg: RLConfig,
                        info: dict, warnings: list[str]) -> SceneObject | None:
    from . import articulated
    from .session import Session
    from .texture import TextureConfig, bake_texture

    caps = {k: session_dir / v for k, v in s.captures.items()}
    if not all(k in caps and (caps[k] / "session.json").exists() for k in ("closed", "open")):
        warnings.append(f"{s.slug}: needs a closed and an open capture; left in the static room")
        return None
    Tinv = np.linalg.inv(room.T)
    region_c = None if s.region is None else s.region @ Tinv[:3, :3].T + Tinv[:3, 3]
    anchor = s.region.mean(0) if s.region is not None else (
        room.T[:3, :3] @ s.seed_world + room.T[:3, 3] if s.seed_world is not None else np.zeros(3))
    floor_y = float((Tinv @ np.array([anchor[0], anchor[1], 0.0, 1.0]))[1])
    closed, opened = Session.load(caps["closed"]), Session.load(caps["open"])
    try:
        res = articulated.reconstruct_articulated(closed, opened, room.session, floor_y, region_c, cfg.articulated)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{s.slug}: articulation could not be recovered ({exc}); left in the static room")
        return None
    j = dict(res.joint)
    part_name = "door" if j["type"] == "revolute" else "drawer"
    part_class = "drawer" if j["type"] == "prismatic" and s.cls.lower() in DRAWER_PARENTS else s.cls
    j["child"] = part_name
    textures = {}
    for name, (V, F) in (("body", res.body), (part_name, res.part)):
        try:
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V @ res.A[:3, :3].T + res.A[:3, 3]),
                                          o3d.utility.Vector3iVector(F))
            m.compute_vertex_normals()
            tm = bake_texture(closed, m, TextureConfig(size=cfg.articulated.texture_size, max_keyframes=12), None)
            from .pipeline import transform_textured

            textures[name] = transform_textured(tm, np.linalg.inv(res.A))
        except Exception as exc:  # noqa: BLE001 - texture is cosmetic
            log.warning("texture for %s/%s failed: %s", s.slug, name, exc)
    links = [{"name": "body", "V": res.body[0], "F": res.body[1], "class": s.cls, "textured": textures.get("body")},
             {"name": part_name, "V": res.part[0], "F": res.part[1], "class": part_class,
              "textured": textures.get(part_name)}]
    info = {**info, **res.info, "shape": "scanned"}
    if res.info.get("inferred_drawer_depth", {}).get("extended_m"):
        warnings.append(f"{s.slug}: hidden drawer depth inferred ({res.info['inferred_drawer_depth']['extended_m']} m)")
    obj = build_object(scene_dir, s.slug, s.cls, "articulated", room.T @ res.A, links, [j], table, cfg,
                       name=s.name, info=info)
    # convex pieces of a door can dip below the floor by a few mm (smoothing, decomposition)
    lifted = clamp_link_above(scene_dir, obj, part_name, cfg.articulated.part_floor_clearance / 2)
    if lifted:
        obj.info["collision_lifted_off_floor_m"] = round(lifted, 4)
    return obj


# --------------------------------------------------------------------------- export


def export_all(scene_dir: Path) -> dict[str, str]:
    """(Re-)write every simulator file from ``scene.json`` (with the user's overrides)."""
    from . import export_isaaclab, export_mjcf, export_usd, gazebo

    scene_dir = Path(scene_dir)
    scene = load_scene(scene_dir)
    files = {"mjcf": export_mjcf.export_mjcf(scene, scene_dir),
             "usd": export_usd.export_usd(scene, scene_dir)}
    files.update(gazebo.export_scene_sdf(scene, scene_dir))
    files["isaaclab"] = export_isaaclab.write_example(scene, scene_dir)
    return {k: str(Path(v).relative_to(scene_dir)) for k, v in files.items()}


def zip_scene(scene_dir: Path) -> Path:
    scene_dir = Path(scene_dir)
    return Path(shutil.make_archive(str(scene_dir), "zip", scene_dir.parent, scene_dir.name))


def process_rl(session_dir: str | Path, scene_dir: str | Path, cfg: RLConfig) -> dict:
    t0 = time.time()
    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    table = physics.load_materials(cfg.materials)
    pc = cfg.pipeline
    name = pc.name

    room = reconstruct_room(session_dir, pc)
    objects: list[SceneObject] = []
    heights = support_heights(room.aligned)
    warnings: list[str] = []
    if cfg.objects == "auto":
        objects += objects_from_selection(Path(session_dir), room, scene_dir, table, cfg, heights, warnings)
    elif cfg.objects == "truth":
        from .synthetic_kitchen import objects_from_truth

        objects += objects_from_truth(Path(session_dir), room.T, scene_dir, table, cfg)
    if cfg.declared_objects:
        objects += load_declared_objects(Path(cfg.declared_objects), scene_dir, table, cfg)

    # support heights come from the full room (above); then the objects are cut out of it
    hulls = [h for o in objects for h in object_hulls_world(o, scene_dir)]
    # fused surfaces are blurred by about a voxel, so the margin grows with the voxel size
    margin = max(cfg.carve_margin, 1.5 * pc.fusion.voxel_size)
    room.aligned, room.textured, n_carved = carve_room(room.aligned, room.textured, hulls, margin)
    log.info("removed %d room triangles occupied by %d objects", n_carved, len(objects))

    room_dir = scene_dir / "room"
    room_report = export_room(room, room_dir, pc)
    boxes = drop_boxes_in_hulls(room_collision_boxes(room.aligned, cfg.room_box_voxel), hulls)
    (room_dir / "collision_boxes.json").write_text(json.dumps(boxes))
    supports = []
    for o in objects:
        if o.type == "pickable":
            box, warn = support_patch(o, scene_dir, heights)
            supports.append(box)
            if warn:
                warnings.append(warn)
    rel_model = f"room/models/{name}"
    scene = Scene(
        name=name,
        room={"name": name, "models_dir": "room/models",
              "visual": f"{rel_model}/meshes/visual.obj", "collision": f"{rel_model}/meshes/collision.stl",
              "collision_boxes": "room/collision_boxes.json", "map": f"room/maps/{name}.yaml",
              "supports": supports, "support_heights": [round(float(h), 4) for h in heights],
              "mesh_ply": f"room/{name}_mesh.ply", "transform_capture_to_scene": room.T.tolist()},
        objects=objects,
        info={"session": str(session_dir), "generator": "cleanscan process-rl"})
    save_scene(scene, scene_dir)
    files = export_all(scene_dir)

    report = {
        "name": name, "units": "m", "up": "z",
        "room": {k: room_report[k] for k in ("frames", "depth_source", "triangles_visual",
                                             "triangles_collision", "ghost_triangles_removed", "extent_m")}
                | {"collision_boxes": len(boxes), "triangles_carved": n_carved},
        "objects": [{"id": o.id, "class": o.cls, "type": o.type,
                     "mass_kg": round(sum(lk.mass for lk in o.links), 4), **o.info} for o in objects],
        "supports": supports,
        "warnings": warnings,
        "outputs": files,
        "seconds": round(time.time() - t0, 1),
        "config": asdict(cfg),
    }
    from .scene import _json_default

    (scene_dir / "report.json").write_text(json.dumps(report, indent=2, default=lambda o: _json_default(o)
                                                      if isinstance(o, (np.ndarray, np.generic)) else str(o)))
    if cfg.zip:
        report["zip"] = str(zip_scene(scene_dir))
    log.info("RL scene ready in %.1fs -> %s", report["seconds"], scene_dir)
    return report
