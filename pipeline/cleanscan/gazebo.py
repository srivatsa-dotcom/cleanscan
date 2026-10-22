"""Export the clean scan as a Gazebo model + world (Gazebo Classic 11 and Gazebo Sim)."""

from __future__ import annotations

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


def _world(name: str, flavour: str) -> str:
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
  </world>
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
