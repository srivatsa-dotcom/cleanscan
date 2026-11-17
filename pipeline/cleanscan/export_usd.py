"""USD export for Isaac Sim / Isaac Lab (``export/scene.usda``), UsdPhysics schemas only.

Written as plain ``.usda`` text, so writing needs no ``pxr``. ``cleanscan validate``
opens the stage with ``pxr`` when it is installed (``pip install usd-core``).

Layout::

    /World                       metersPerUnit = 1, upAxis = Z, kilogramsPerUnit = 1
      PhysicsScene
      Looks/                     preview materials + physics materials
      Room/visual                textured mesh
      Room/collision             triangle-mesh collider (approximation "none", static)
      Room/floor                 thin box collider just below z = 0
      <pickable>                 RigidBodyAPI + MassAPI, visual + convex collision pieces
      <articulated>              ArticulationRootAPI
        <link>                   RigidBodyAPI + MassAPI
        fix_to_world             PhysicsFixedJoint (fixed-base objects)
        <joint>                  PhysicsRevoluteJoint / PhysicsPrismaticJoint, limits,
                                 damping as a zero-stiffness PhysicsDriveAPI

Meshes are separate layers in ``export/usd/`` that ``scene.usda`` references.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from .scene import Scene, SceneObject, read_obj_vf

HEADER = """#usda 1.0
(
    defaultPrim = "{default}"
    metersPerUnit = 1
    kilogramsPerUnit = 1
    upAxis = "Z"
)
"""


def _safe(name: str) -> str:
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    return s if s and not s[0].isdigit() else "_" + s


def _v(x) -> str:
    return "(" + ", ".join(f"{float(a):.9g}" for a in np.ravel(x)) + ")"


def _arr(rows) -> str:
    return "[" + ", ".join(_v(r) for r in rows) + "]"


def _quat(R: np.ndarray) -> str:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return _v([w, x, y, z])


def _rot_x_to(axis) -> np.ndarray:
    """Rotation taking +X to ``axis`` (USD joints rotate/slide along a principal axis)."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    rot, _ = Rotation.align_vectors([a], [[1.0, 0.0, 0.0]])
    return rot.as_matrix()


def _mtl_texture(obj_path: Path) -> tuple[Path | None, list[float], float]:
    mtl = obj_path.with_suffix(".mtl")
    tex, kd, d = None, [0.7, 0.7, 0.7], 1.0
    if mtl.exists():
        for line in mtl.read_text().splitlines():
            p = line.split()
            if p[:1] == ["map_Kd"]:
                tex = obj_path.parent / p[1]
            elif p[:1] == ["Kd"]:
                kd = [float(v) for v in p[1:4]]
            elif p[:1] == ["d"]:
                d = float(p[1])
    return tex, kd, d


def write_mesh_layer(path: Path, V: np.ndarray, F: np.ndarray, uvs: np.ndarray | None = None) -> None:
    """A Mesh prim named "mesh" as the default prim of its own layer."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    lines = [HEADER.format(default="mesh"), 'def Mesh "mesh"', "{"]
    lines.append(f"    float3[] extent = [{_v(V.min(0))}, {_v(V.max(0))}]")
    lines.append(f"    int[] faceVertexCounts = [{', '.join(['3'] * len(F))}]")
    lines.append(f"    int[] faceVertexIndices = [{', '.join(map(str, F.ravel()))}]")
    lines.append(f"    point3f[] points = {_arr(V)}")
    if uvs is not None and len(uvs) == 3 * len(F):
        lines.append(f"    texCoord2f[] primvars:st = {_arr(uvs)} (")
        lines.append('        interpolation = "faceVarying"')
        lines.append("    )")
    lines.append('    uniform token orientation = "rightHanded"')
    lines.append('    uniform token subdivisionScheme = "none"')
    lines.append("}")
    path.write_text("\n".join(lines) + "\n")


class _Writer:
    def __init__(self, scene: Scene, scene_dir: Path, out_path: Path):
        self.scene, self.scene_dir, self.out = scene, scene_dir, out_path
        self.layer_dir = out_path.parent / "usd"
        self.layer_dir.mkdir(parents=True, exist_ok=True)
        self.looks: list[str] = []
        self.look_names: set[str] = set()

    def rel(self, p: Path) -> str:
        return os.path.relpath(p, self.out.parent).replace(os.sep, "/")

    # ---------------------------------------------------------------- materials
    def preview_material(self, name: str, obj_path: Path) -> str:
        name = _safe(name)
        if name in self.look_names:
            return f"/World/Looks/{name}"
        self.look_names.add(name)
        tex, kd, opacity = _mtl_texture(obj_path)
        m = [f'        def Material "{name}"', "        {",
             f"            token outputs:surface.connect = </World/Looks/{name}/surface.outputs:surface>",
             '            def Shader "surface"', "            {",
             '                uniform token info:id = "UsdPreviewSurface"',
             "                float inputs:roughness = 0.8" if opacity >= 1 else "                float inputs:roughness = 0.05"]
        if opacity < 1:
            m += [f"                float inputs:opacity = {opacity:.3g}", "                float inputs:ior = 1.5"]
        if tex is not None and tex.exists():
            m += [f"                color3f inputs:diffuseColor.connect = </World/Looks/{name}/tex.outputs:rgb>",
                  "                token outputs:surface", "            }",
                  '            def Shader "tex"', "            {",
                  '                uniform token info:id = "UsdUVTexture"',
                  f"                asset inputs:file = @{self.rel(tex)}@",
                  f"                float2 inputs:st.connect = </World/Looks/{name}/st.outputs:result>",
                  '                token inputs:wrapS = "clamp"', '                token inputs:wrapT = "clamp"',
                  "                float3 outputs:rgb", "            }",
                  '            def Shader "st"', "            {",
                  '                uniform token info:id = "UsdPrimvarReader_float2"',
                  '                string inputs:varname = "st"',
                  "                float2 outputs:result", "            }"]
        else:
            m += [f"                color3f inputs:diffuseColor = {_v(kd)}",
                  "                token outputs:surface", "            }"]
        m.append("        }")
        self.looks += m
        return f"/World/Looks/{name}"

    def physics_material(self, friction: float, restitution: float) -> str:
        name = _safe(f"phys_f{friction:.3f}_r{restitution:.3f}".replace(".", "p"))
        if name not in self.look_names:
            self.look_names.add(name)
            self.looks += [f'        def Material "{name}" (', '            prepend apiSchemas = ["PhysicsMaterialAPI"]',
                           "        )", "        {",
                           f"            float physics:staticFriction = {friction:.4g}",
                           f"            float physics:dynamicFriction = {friction:.4g}",
                           f"            float physics:restitution = {restitution:.4g}", "        }"]
        return f"/World/Looks/{name}"

    # ---------------------------------------------------------------- meshes
    def mesh_layer(self, key: str, obj_rel: str, with_uv: bool) -> Path:
        src = self.scene_dir / obj_rel
        dst = self.layer_dir / f"{_safe(key)}.usda"
        if with_uv:
            m = o3d.io.read_triangle_mesh(str(src))
            uv = np.asarray(m.triangle_uvs) if m.has_triangle_uvs() else None
            write_mesh_layer(dst, np.asarray(m.vertices), np.asarray(m.triangles), uv)
        else:
            V, F = read_obj_vf(src)
            write_mesh_layer(dst, V, F)
        return dst

    def visual_prim(self, key: str, obj_rel: str, indent: str) -> list[str]:
        layer = self.mesh_layer(key + "_visual", obj_rel, with_uv=True)
        mat = self.preview_material(key + "_mat", self.scene_dir / obj_rel)
        return [f'{indent}def Mesh "visual" (', f"{indent}    prepend references = @{self.rel(layer)}@",
                f'{indent}    prepend apiSchemas = ["MaterialBindingAPI"]', f"{indent})", f"{indent}{{",
                f"{indent}    rel material:binding = <{mat}>", f"{indent}}}"]

    def collider_prim(self, prim: str, key: str, obj_rel: str, approximation: str, phys_mat: str,
                      indent: str) -> list[str]:
        layer = self.mesh_layer(key, obj_rel, with_uv=False)
        return [f'{indent}def Mesh "{prim}" (', f"{indent}    prepend references = @{self.rel(layer)}@",
                f'{indent}    prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", '
                f'"MaterialBindingAPI"]', f"{indent})", f"{indent}{{",
                f'{indent}    uniform token physics:approximation = "{approximation}"',
                f'{indent}    uniform token purpose = "guide"',
                f"{indent}    rel material:binding:physics = <{phys_mat}>", f"{indent}}}"]

    # ---------------------------------------------------------------- bodies
    def link_body(self, o: SceneObject, lk, rigid: bool, indent: str, name: str | None = None) -> list[str]:
        key = f"{o.id}_{lk.name}"
        schemas = '["PhysicsRigidBodyAPI", "PhysicsMassAPI"]' if rigid else "[]"
        out = [f'{indent}def Xform "{_safe(name or lk.name)}" (', f"{indent}    prepend apiSchemas = {schemas}",
               f"{indent})", f"{indent}{{"]
        if rigid:
            w, R = np.linalg.eigh(np.asarray(lk.inertia))
            if np.linalg.det(R) < 0:
                R[:, 0] = -R[:, 0]
            out += [f"{indent}    float physics:mass = {lk.mass:.6g}",
                    f"{indent}    point3f physics:centerOfMass = {_v(lk.com)}",
                    f"{indent}    float3 physics:diagonalInertia = {_v(w)}",
                    f"{indent}    quatf physics:principalAxes = {_quat(R)}"]
        if lk.visual:
            out += self.visual_prim(key, lk.visual, indent + "    ")
        pm = self.physics_material(lk.friction, lk.restitution)
        if lk.collision:
            out += [f'{indent}    def Xform "collision"', f"{indent}    {{"]
            for i, c in enumerate(lk.collision):
                out += self.collider_prim(f"c{i:03d}", f"{key}_c{i:03d}", c, "convexHull", pm, indent + "        ")
            out.append(f"{indent}    }}")
        out.append(f"{indent}}}")
        return out

    def xform_ops(self, T: np.ndarray, indent: str) -> list[str]:
        return [f"{indent}double3 xformOp:translate = {_v(T[:3, 3])}",
                f"{indent}quatd xformOp:orient = {_quat(T[:3, :3])}",
                f'{indent}uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]']

    def object_prims(self, o: SceneObject) -> list[str]:
        name = _safe(o.id)
        if o.type in ("pickable", "static"):
            # the object prim itself is the rigid body (static: collision only)
            lines = self.link_body(o, o.root, rigid=o.type == "pickable", indent="    ", name=o.id)
            return lines[:4] + self.xform_ops(o.pose, "        ") + lines[4:]
        out = [f'    def Xform "{name}" (', '        prepend apiSchemas = ["PhysicsArticulationRootAPI"]',
               "    )", "    {"] + self.xform_ops(o.pose, "        ")
        for lk in o.links:
            out += self.link_body(o, lk, rigid=True, indent="        ")
        base = f"/World/{name}"
        if o.fixed_base:
            out += ['        def PhysicsFixedJoint "fix_to_world"', "        {",
                    f"            rel physics:body1 = <{base}/{_safe(o.root.name)}>",
                    f"            point3f physics:localPos0 = {_v(o.pose[:3, 3])}",
                    f"            quatf physics:localRot0 = {_quat(o.pose[:3, :3])}",
                    "            point3f physics:localPos1 = (0, 0, 0)",
                    "            quatf physics:localRot1 = (1, 0, 0, 0)", "        }"]
        for j in o.joints:
            R = _rot_x_to(j.axis)
            if j.type == "revolute":
                kind, drive = "PhysicsRevoluteJoint", "angular"
                lo, hi = np.degrees(j.lower), np.degrees(j.upper)
                damping = j.damping * np.pi / 180.0  # USD angular drive gains are per degree
            else:
                kind, drive = "PhysicsPrismaticJoint", "linear"
                lo, hi, damping = j.lower, j.upper, j.damping
            out += [f'        def {kind} "{_safe(j.name)}" (', f'            prepend apiSchemas = ["PhysicsDriveAPI:{drive}"]',
                    "        )", "        {",
                    f"            rel physics:body0 = <{base}/{_safe(j.parent)}>",
                    f"            rel physics:body1 = <{base}/{_safe(j.child)}>",
                    '            uniform token physics:axis = "X"',
                    f"            point3f physics:localPos0 = {_v(j.origin)}",
                    f"            quatf physics:localRot0 = {_quat(R)}",
                    f"            point3f physics:localPos1 = {_v(j.origin)}",
                    f"            quatf physics:localRot1 = {_quat(R)}",
                    f"            float physics:lowerLimit = {lo:.6g}",
                    f"            float physics:upperLimit = {hi:.6g}",
                    f"            float drive:{drive}:physics:stiffness = 0",
                    f"            float drive:{drive}:physics:damping = {damping:.6g}",
                    f'            uniform token drive:{drive}:physics:type = "force"', "        }"]
        out.append("    }")
        return out

    def room_prims(self) -> list[str]:
        room = self.scene.room
        out = ['    def Xform "Room"', "    {"]
        if room.get("visual"):
            out += self.visual_prim("room", room["visual"], "        ")
        pm = self.physics_material(0.8, 0.05)
        if room.get("collision"):
            col = self.scene_dir / room["collision"]
            obj = self.layer_dir / "room_collision.obj"
            m = o3d.io.read_triangle_mesh(str(col))
            from .scene import write_plain_obj

            write_plain_obj(obj, np.asarray(m.vertices), np.asarray(m.triangles))
            out += self.collider_prim("collision", "room_collision", os.path.relpath(obj, self.scene_dir),
                                      "none", pm, "        ")
        out += ['        def Cube "floor" (', '            prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]',
                "        )", "        {", "            double size = 1",
                "            double3 xformOp:translate = (0, 0, -0.05)",
                "            float3 xformOp:scale = (200, 200, 0.1)",
                '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                '            uniform token purpose = "guide"',
                f"            rel material:binding:physics = <{pm}>", "        }"]
        for b in room.get("supports", []):  # patches under pickables (see scene.support_patch)
            out += [f'        def Cube "support_{_safe(b["object"])}" (',
                    '            prepend apiSchemas = ["PhysicsCollisionAPI", "MaterialBindingAPI"]',
                    "        )", "        {", "            double size = 1",
                    f"            double3 xformOp:translate = {_v(b['center'])}",
                    f"            float3 xformOp:scale = {_v(b['size'])}",
                    '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                    '            uniform token purpose = "guide"',
                    f"            rel material:binding:physics = <{pm}>", "        }"]
        out.append("    }")
        return out

    def write(self) -> Path:
        body = ['    def PhysicsScene "PhysicsScene"', "    {",
                "        vector3f physics:gravityDirection = (0, 0, -1)",
                "        float physics:gravityMagnitude = 9.81", "    }"]
        body += self.room_prims()
        for o in self.scene.objects:
            body += self.object_prims(o)
        text = [HEADER.format(default="World"), f"# CleanScan RL scene \"{self.scene.name}\" (m, kg, Z up). "
                "Re-generate with `cleanscan export`.", 'def Xform "World"', "{",
                '    def Scope "Looks"', "    {"] + self.looks + ["    }"] + body + ["}"]
        self.out.write_text("\n".join(text) + "\n")
        return self.out


def export_usd(scene: Scene, scene_dir: Path, out_path: Path | None = None) -> Path:
    scene_dir = Path(scene_dir)
    out_path = Path(out_path or scene_dir / "export" / "scene.usda")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return _Writer(scene, scene_dir, out_path).write()
