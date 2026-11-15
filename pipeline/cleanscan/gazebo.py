"""Export the clean scan as a Gazebo model + world (Gazebo Classic 11 and Gazebo Sim)."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from xml.sax.saxutils import escape

import cv2
import numpy as np
import open3d as o3d

from .texture import TexturedMesh


def write_obj(path: Path, tm: TexturedMesh | None, mesh: o3d.geometry.TriangleMesh | None = None) -> None:
    """Textured OBJ+MTL (preferred) or plain grey OBJ when no texture is baked."""
    path = Path(path)
    mtl = path.with_suffix(".mtl")
    tex_name = path.stem + "_texture.png"
    if tm is not None:
        V, F, N = tm.vertices, tm.triangles, tm.normals
    else:
        mesh.compute_vertex_normals()
        V, F, N = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_normals)

    lines = [f"mtllib {mtl.name}", "usemtl scan"]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in V]
    lines += [f"vn {x:.4f} {y:.4f} {z:.4f}" for x, y, z in N]
    if tm is not None:
        lines += [f"vt {u:.6f} {v:.6f}" for u, v in tm.uvs.reshape(-1, 2)]
        for t, (a, b, c) in enumerate(F + 1):
            k = 3 * t + 1
            lines.append(f"f {a}/{k}/{a} {b}/{k + 1}/{b} {c}/{k + 2}/{c}")
    else:
        lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in F + 1]
    path.write_text("\n".join(lines) + "\n")

    m = ["newmtl scan", "Ka 0.8 0.8 0.8", "Kd 1.0 1.0 1.0", "Ks 0.0 0.0 0.0", "d 1.0", "illum 1"]
    if tm is not None:
        cv2.imwrite(str(path.parent / tex_name), cv2.cvtColor(tm.texture, cv2.COLOR_RGB2BGR))
        m.append(f"map_Kd {tex_name}")
    else:
        m[2] = "Kd 0.7 0.7 0.7"
    mtl.write_text("\n".join(m) + "\n")


def _model_sdf(name: str) -> str:
    uri = f"model://{name}/meshes"
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{escape(name)}">
    <static>true</static>
    <link name="scan">
      <visual name="visual">
        <cast_shadows>false</cast_shadows>
        <geometry><mesh><uri>{uri}/visual.obj</uri></mesh></geometry>
      </visual>
      <collision name="collision">
        <geometry><mesh><uri>{uri}/collision.stl</uri></mesh></geometry>
        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>
      </collision>
    </link>
  </model>
</sdf>
"""


def _model_config(name: str) -> str:
    return f"""<?xml version="1.0"?>
<model>
  <name>{escape(name)}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author><name>CleanScan</name></author>
  <description>Static environment scanned with CleanScan (people and animals removed).</description>
</model>
"""


_LIGHT = """    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>
"""

# Invisible collision plane exactly at the scanned floor: robots never fall
# through small holes in the scanned floor, and there is no z-fighting.
_GROUND = """    <model name="floor_collision">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>200 200</size></plane></geometry>
        </collision>
      </link>
    </model>
"""


def _world(name: str, flavour: str, extra: str = "") -> str:
    if flavour == "sim":
        version = "1.9"
        plugins = """    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
"""
    else:
        version, plugins = "1.6", ""
    return f"""<?xml version="1.0"?>
<sdf version="{version}">
  <world name="{escape(name)}_world">
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
{plugins}    <scene>
      <ambient>0.6 0.6 0.6 1</ambient>
      <background>0.7 0.8 0.9 1</background>
      <shadows>false</shadows>
    </scene>
{_LIGHT}{_GROUND}    <include>
      <uri>model://{escape(name)}</uri>
      <pose>0 0 0 0 0 0</pose>
    </include>
{extra}  </world>
</sdf>
"""


def export_gazebo(out_dir: Path, name: str, visual: TexturedMesh | None,
                  visual_mesh: o3d.geometry.TriangleMesh,
                  collision_mesh: o3d.geometry.TriangleMesh) -> dict[str, Path]:
    out_dir = Path(out_dir)
    model_dir = out_dir / "models" / name
    (model_dir / "meshes").mkdir(parents=True, exist_ok=True)
    (out_dir / "worlds").mkdir(parents=True, exist_ok=True)

    write_obj(model_dir / "meshes" / "visual.obj", visual, visual_mesh)
    collision_mesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh(str(model_dir / "meshes" / "collision.stl"), collision_mesh)
    (model_dir / "model.sdf").write_text(_model_sdf(name))
    (model_dir / "model.config").write_text(_model_config(name))

    classic = out_dir / "worlds" / f"{name}.world"
    sim = out_dir / "worlds" / f"{name}.sdf"
    classic.write_text(_world(name, "classic"))
    sim.write_text(_world(name, "sim"))

    script = out_dir / "run_gazebo.sh"
    script.write_text(f"""#!/usr/bin/env bash
# Usage: ./run_gazebo.sh [sim|classic]   (default: sim = Gazebo Harmonic/Ionic/Jetty)
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
export GZ_SIM_RESOURCE_PATH="$HERE/models${{GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}}"
export GAZEBO_MODEL_PATH="$HERE/models${{GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}}"
if [ "${{1:-sim}}" = "classic" ]; then
  exec gazebo --verbose "$HERE/worlds/{name}.world"
else
  exec gz sim -v 3 "$HERE/worlds/{name}.sdf"
fi
""")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return {"model": model_dir, "world_classic": classic, "world_sim": sim, "script": script}


# --------------------------------------------------------------------------- RL scenes


def _vec(v, nd=6) -> str:
    return " ".join(f"{float(x):.{nd}g}" for x in np.ravel(v))


def _sdf_pose(T: np.ndarray) -> str:
    from scipy.spatial.transform import Rotation

    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")  # fixed-axis roll, pitch, yaw
    return _vec(np.concatenate([T[:3, 3], rpy]))


def _link_sdf(o, lk, uri_of) -> str:
    I = np.asarray(lk.inertia)
    out = [f'    <link name="{escape(lk.name)}">',
           "      <inertial>",
           f"        <pose>{_vec(lk.com)} 0 0 0</pose>",
           f"        <mass>{lk.mass:.6g}</mass>",
           "        <inertia>"
           f"<ixx>{I[0, 0]:.6g}</ixx><ixy>{I[0, 1]:.6g}</ixy><ixz>{I[0, 2]:.6g}</ixz>"
           f"<iyy>{I[1, 1]:.6g}</iyy><iyz>{I[1, 2]:.6g}</iyz><izz>{I[2, 2]:.6g}</izz></inertia>",
           "      </inertial>"]
    if lk.visual:
        out.append(f'      <visual name="visual"><geometry><mesh><uri>{uri_of(lk.visual)}</uri></mesh>'
                   "</geometry></visual>")
    for i, c in enumerate(lk.collision):
        out.append(f'      <collision name="c{i:03d}"><geometry><mesh><uri>{uri_of(c)}</uri></mesh></geometry>'
                   f"<surface><friction><ode><mu>{lk.friction:.3g}</mu><mu2>{lk.friction:.3g}</mu2></ode>"
                   f"</friction><bounce><restitution_coefficient>{lk.restitution:.3g}"
                   "</restitution_coefficient></bounce></surface></collision>")
    out.append("    </link>")
    return "\n".join(out)


def _object_model_sdf(o, uri_of) -> str:
    parts = [f'<?xml version="1.0"?>\n<sdf version="1.6">\n  <model name="{escape(o.id)}">']
    if o.type == "static":
        parts.append("    <static>true</static>")
    parts += [_link_sdf(o, lk, uri_of) for lk in o.links]
    if o.type == "articulated" and o.fixed_base:
        parts.append(f'    <joint name="fix_to_world" type="fixed"><parent>world</parent>'
                     f"<child>{escape(o.root.name)}</child></joint>")
    for j in o.joints:
        # joint pose is relative to the child link frame (= the object frame here)
        parts.append(f'    <joint name="{escape(j.name)}" type="{j.type}">'
                     f"<pose>{_vec(j.origin)} 0 0 0</pose>"
                     f"<parent>{escape(j.parent)}</parent><child>{escape(j.child)}</child>"
                     f"<axis><xyz>{_vec(j.axis)}</xyz><limit><lower>{j.lower:.6g}</lower>"
                     f"<upper>{j.upper:.6g}</upper><effort>{j.effort:.4g}</effort>"
                     f"<velocity>{j.velocity:.4g}</velocity></limit><dynamics>"
                     f"<damping>{j.damping:.4g}</damping><friction>{j.friction:.4g}</friction>"
                     "</dynamics></axis></joint>")
    parts.append("  </model>\n</sdf>\n")
    return "\n".join(parts)


def export_scene_sdf(scene, scene_dir: Path) -> dict[str, Path]:
    """Gazebo export of an RL scene: ``export/world.sdf`` (Gazebo Sim), ``export/world.world``
    (Classic), one model per object in ``export/models/<id>`` and ``export/run_gazebo.sh``.
    The room model is the one written by ``cleanscan process`` inside ``room/``."""
    import shutil

    scene_dir = Path(scene_dir)
    exp = scene_dir / "export"
    models = exp / "models"
    models.mkdir(parents=True, exist_ok=True)
    includes = []
    for o in scene.objects:
        mdir = models / o.id
        if mdir.exists():
            shutil.rmtree(mdir)
        src = scene_dir / "objects" / o.id
        shutil.copytree(src, mdir / "meshes")

        def uri_of(rel, _oid=o.id):
            inner = Path(rel).relative_to(Path("objects") / _oid)
            return f"model://{_oid}/meshes/{inner.as_posix()}"

        (mdir / "model.sdf").write_text(_object_model_sdf(o, uri_of))
        (mdir / "model.config").write_text(_model_config(o.id).replace(
            "Static environment scanned with CleanScan (people and animals removed).",
            f"{o.type} object '{o.name or o.cls}' scanned with CleanScan."))
        includes.append(f"    <include>\n      <uri>model://{escape(o.id)}</uri>\n      <name>{escape(o.id)}</name>\n"
                        f"      <pose>{_sdf_pose(o.pose)}</pose>\n    </include>\n")
    sup = scene.room.get("supports", [])
    if sup:  # patches under pickables (see scene.support_patch), invisible static boxes
        cols = "".join(f'        <collision name="{escape(b["object"])}"><pose>{_vec(b["center"])} 0 0 0</pose>'
                       f'<geometry><box><size>{_vec(b["size"])}</size></box></geometry></collision>\n'
                       for b in sup)
        includes.append(f'    <model name="supports">\n      <static>true</static>\n      <link name="link">\n'
                        f"{cols}      </link>\n    </model>\n")
    room_name = scene.room.get("name", scene.name)
    sim = exp / "world.sdf"
    classic = exp / "world.world"
    sim.write_text(_world(room_name, "sim", "".join(includes)))
    classic.write_text(_world(room_name, "classic", "".join(includes)))
    room_models = os.path.relpath(scene_dir / scene.room.get("models_dir", "room/models"), exp)
    script = exp / "run_gazebo.sh"
    script.write_text(f"""#!/usr/bin/env bash
# Usage: ./run_gazebo.sh [sim|classic]   (default: sim = Gazebo Harmonic/Ionic/Jetty)
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
P="$HERE/models:$HERE/{room_models}"
export GZ_SIM_RESOURCE_PATH="$P${{GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}}"
export GAZEBO_MODEL_PATH="$P${{GAZEBO_MODEL_PATH:+:$GAZEBO_MODEL_PATH}}"
if [ "${{1:-sim}}" = "classic" ]; then
  exec gazebo --verbose "$HERE/world.world"
else
  exec gz sim -v 3 "$HERE/world.sdf"
fi
""")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return {"world_sim": sim, "world_classic": classic, "models": models, "script": script}
