"""MuJoCo MJCF export of a CleanScan scene (``export/scene.xml``).

* room: textured visual mesh (no collision) + an infinite floor plane at z = 0 + boxes
  approximating walls, counters and furniture (MuJoCo collides a mesh as its convex hull,
  so a whole-room mesh cannot be a collider)
* pickable objects: a body with a ``freejoint``, one mesh geom per convex piece
* articulated objects: the root body welded to the world (or with a freejoint if not
  ``fixed_base``), each moving link a child body with a ``hinge`` / ``slide`` joint
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np
from scipy.spatial.transform import Rotation

from .scene import Scene, SceneObject


def _f(v, nd=6) -> str:
    return " ".join(f"{float(x):.{nd}g}" for x in np.ravel(v))


def _quat_wxyz(R: np.ndarray) -> str:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return _f([w, x, y, z], 9)


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in name)


def _fullinertia(I) -> str:
    I = np.asarray(I)
    return _f([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]], 9)


class _Assets:
    def __init__(self, scene_dir: Path, export_dir: Path):
        self.scene_dir, self.export_dir = scene_dir, export_dir
        self.lines: list[str] = []
        self.names: set[str] = set()

    def rel(self, path: str) -> str:
        return os.path.relpath(self.scene_dir / path, self.export_dir).replace(os.sep, "/")

    def mesh(self, name: str, path: str, inertia: str = "convex") -> str:
        name = _safe(name)
        if name not in self.names:
            self.names.add(name)
            self.lines.append(f'    <mesh name="{name}" file={quoteattr(self.rel(path))} inertia="{inertia}"/>')
        return name

    def visual_material(self, name: str, obj_path: str) -> str | None:
        """Texture + material from the OBJ's MTL (map_Kd or Kd), or None."""
        mtl = (self.scene_dir / obj_path).with_suffix(".mtl")
        if not mtl.exists():
            return None
        tex, rgba = None, "0.7 0.7 0.7 1"
        for line in mtl.read_text().splitlines():
            parts = line.split()
            if parts[:1] == ["map_Kd"]:
                tex = str(Path(obj_path).parent / parts[1])
            elif parts[:1] == ["Kd"]:
                rgba = " ".join(parts[1:4]) + " 1"
        name = _safe(name)
        if tex:
            self.lines.append(f'    <texture name="{name}_tex" type="2d" file={quoteattr(self.rel(tex))}/>')
            self.lines.append(f'    <material name="{name}" texture="{name}_tex" rgba="1 1 1 1"/>')
        else:
            self.lines.append(f'    <material name="{name}" rgba="{rgba}"/>')
        return name


def _link_geoms(assets: _Assets, oid: str, lk, condim: int, indent: str) -> list[str]:
    out = []
    if lk.visual:
        vm = assets.mesh(f"{oid}_{lk.name}_visual", lk.visual, inertia="shell")
        mat = assets.visual_material(f"{oid}_{lk.name}_mat", lk.visual)
        out.append(f'{indent}<geom name="{_safe(oid)}_{_safe(lk.name)}_visual" type="mesh" mesh="{vm}" '
                   f'class="visual"' + (f' material="{mat}"' if mat else "") + "/>")
    for i, c in enumerate(lk.collision):
        cm = assets.mesh(f"{oid}_{lk.name}_c{i:03d}", c)
        out.append(f'{indent}<geom name="{_safe(oid)}_{_safe(lk.name)}_c{i:03d}" type="mesh" mesh="{cm}" '
                   f'class="collision" condim="{condim}" friction="{lk.friction:.3g} 0.005 0.0001"/>')
    return out


def _inertial(lk, indent: str) -> str:
    return (f'{indent}<inertial pos="{_f(lk.com)}" mass="{lk.mass:.6g}" '
            f'fullinertia="{_fullinertia(lk.inertia)}"/>')


def _object_xml(assets: _Assets, o: SceneObject) -> list[str]:
    oid = _safe(o.id)
    pos, quat = _f(o.pose[:3, 3]), _quat_wxyz(o.pose[:3, :3])
    out = [f'    <body name="{oid}" pos="{pos}" quat="{quat}">']
    if o.type == "static" or (o.type == "articulated" and o.fixed_base):
        pass  # welded to the world
    else:
        out.append(f'      <freejoint name="{oid}_free"/>')
    condim = 4 if o.type == "pickable" else 3
    root = o.root
    out.append(_inertial(root, "      "))
    out += _link_geoms(assets, oid, root, condim, "      ")
    # children of the root (one level, which covers doors, drawers, lids)
    for j in o.joints:
        child = o.link(j.child)
        out.append(f'      <body name="{oid}_{_safe(child.name)}">')
        jt = "hinge" if j.type == "revolute" else "slide"
        out.append(f'        <joint name="{oid}_{_safe(j.name)}" type="{jt}" pos="{_f(j.origin)}" '
                   f'axis="{_f(j.axis)}" range="{j.lower:.6g} {j.upper:.6g}" limited="true" '
                   f'damping="{j.damping:.4g}" frictionloss="{j.friction:.4g}" armature="0.01"/>')
        out.append(_inertial(child, "        "))
        out += _link_geoms(assets, oid, child, 3, "        ")
        out.append("      </body>")
    out.append("    </body>")
    return out


def export_mjcf(scene: Scene, scene_dir: Path, out_path: Path | None = None) -> Path:
    scene_dir = Path(scene_dir)
    out_path = Path(out_path or scene_dir / "export" / "scene.xml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assets = _Assets(scene_dir, out_path.parent)

    world = ['    <light name="sun" directional="true" pos="0 0 6" dir="-0.3 0.2 -1" diffuse="0.8 0.8 0.8"/>',
             '    <geom name="floor" type="plane" size="0 0 0.05" class="collision" condim="3" '
             'rgba="0.5 0.5 0.5 0" friction="0.8 0.005 0.0001"/>']
    room = scene.room
    if room.get("visual"):
        vm = assets.mesh("room_visual", room["visual"], inertia="shell")
        mat = assets.visual_material("room_mat", room["visual"])
        world.append(f'    <geom name="room_visual" type="mesh" mesh="{vm}" class="visual"'
                     + (f' material="{mat}"' if mat else "") + "/>")
    if room.get("collision_boxes"):
        boxes = json.loads((scene_dir / room["collision_boxes"]).read_text())
        for i, b in enumerate(boxes):
            world.append(f'    <geom name="room_box{i:04d}" type="box" pos="{_f(b["center"])}" '
                         f'size="{_f(np.asarray(b["size"]) / 2)}" class="collision" condim="3" '
                         'friction="0.8 0.005 0.0001"/>')
    for i, b in enumerate(room.get("supports", [])):  # patches under pickables (see scene.support_patch)
        world.append(f'    <geom name="support_{_safe(b["object"])}" type="box" pos="{_f(b["center"])}" '
                     f'size="{_f(np.asarray(b["size"]) / 2)}" class="collision" condim="3" '
                     'friction="0.8 0.005 0.0001"/>')
    excludes = []
    for o in scene.objects:
        world += _object_xml(assets, o)
        # links of one object never collide with each other (MuJoCo only filters a parent and
        # child when the parent is not welded to the world); joint limits keep them apart
        names = [_safe(o.id)] + [f"{_safe(o.id)}_{_safe(o.link(j.child).name)}" for j in o.joints]
        excludes += [f'    <exclude body1="{a}" body2="{b}"/>' for i, a in enumerate(names) for b in names[i + 1:]]

    xml = f"""<?xml version="1.0"?>
<!-- CleanScan RL scene "{scene.name}". Units: m, kg. Z up, floor at z = 0.
     Generated by `cleanscan export`; edit scene.json and re-export instead of this file. -->
<mujoco model={quoteattr(scene.name)}>
  <compiler angle="radian" autolimits="true" inertiafromgeom="false"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <size memory="256M"/>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.5 0.5 0.5"/><global offwidth="1280" offheight="960"/></visual>
  <default>
    <default class="visual"><geom contype="0" conaffinity="0" group="1" mass="0"/></default>
    <default class="collision"><geom group="3" rgba="0.2 0.6 0.9 0.4" solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  </default>
  <asset>
{chr(10).join(assets.lines)}
  </asset>
  <worldbody>
{chr(10).join(world)}
  </worldbody>
  <contact>
{chr(10).join(excludes)}
  </contact>
</mujoco>
"""
    out_path.write_text(xml)
    return out_path
