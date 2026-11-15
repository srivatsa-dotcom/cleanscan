"""RL scene bundle: physics properties, scene.json, MJCF / USD / SDF export (milestone 1)."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
import pytest

from cleanscan import physics
from cleanscan.fusion import FusionConfig
from cleanscan.pipeline import PipelineConfig
from cleanscan.rl import RLConfig, export_all, process_rl
from cleanscan.scene import load_scene, room_collision_boxes, write_plain_obj
from cleanscan.synthetic import make_session
from cleanscan.texture import TextureConfig

mujoco = pytest.importorskip("mujoco")


def _vf(m):
    m.remove_duplicated_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles)


# --------------------------------------------------------------------------- physics


def test_solid_box_mass_and_inertia():
    a, b, c = 0.3, 0.2, 0.1
    V, F = _vf(o3d.geometry.TriangleMesh.create_box(a, b, c))
    spec = physics.PhysicsSpec("wood", 600.0, 0.6, 0.1, hollow=False)
    mp = physics.mass_properties(V, F, spec)
    m = 600.0 * a * b * c
    assert mp.mass == pytest.approx(m, rel=1e-6)
    assert np.allclose(mp.com, [a / 2, b / 2, c / 2])
    expect = m / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    assert np.allclose(np.diag(mp.inertia), expect, rtol=1e-6)
    assert np.allclose(mp.inertia - np.diag(np.diag(mp.inertia)), 0, atol=1e-12)


def test_thin_shell_sphere_inertia():
    r, t = 0.1, 0.002
    V, F = _vf(o3d.geometry.TriangleMesh.create_sphere(r, resolution=60))
    spec = physics.PhysicsSpec("glass", 2500.0, 0.4, 0.1, hollow=True, shell_thickness=t)
    mp = physics.mass_properties(V, F, spec)
    m = 2500.0 * 4 * np.pi * r * r * t
    assert mp.mass == pytest.approx(m, rel=0.01)
    assert np.allclose(np.diag(mp.inertia), 2 / 3 * m * r * r, rtol=0.02)


def test_fill_adds_water_and_lowers_nothing_for_symmetric_shapes():
    V, F = _vf(o3d.geometry.TriangleMesh.create_cylinder(0.04, 0.2, 40, 2))
    empty = physics.mass_properties(V, F, physics.resolve("bottle", physics.load_materials()))
    full = physics.mass_properties(V, F, physics.resolve("bottle", physics.load_materials(), {"fill": 1.0}))
    water = 1000 * np.pi * 0.04 ** 2 * 0.2
    assert full.mass - empty.mass == pytest.approx(water, rel=0.02)


def test_materials_table_and_override_file(tmp_path):
    table = physics.load_materials()
    s = physics.resolve("wine glass", table)
    assert (s.material, s.density, s.hollow) == ("glass", 2500, True)
    assert physics.resolve("unknown-thing", table).material == "generic"
    extra = tmp_path / "m.yaml"
    extra.write_text("classes:\n  wine glass: {material: steel}\n")
    assert physics.resolve("wine glass", physics.load_materials(extra)).density == 7850


def test_convex_hull_is_outward_and_closed():
    pts = np.random.default_rng(0).normal(size=(200, 3))
    V, F = physics.convex_hull(pts)
    assert physics.is_watertight(F)
    vol, _, _ = physics._solid_integrals(V, F)
    assert vol > 0


def test_room_boxes_snap_to_support_heights():
    room = o3d.geometry.TriangleMesh.create_box(4, 3, 2.5)
    table = o3d.geometry.TriangleMesh.create_box(1.0, 0.6, 0.737).translate((1.5, 1.0, 0.0))
    mesh = room + table
    boxes = room_collision_boxes(mesh, voxel=0.05)
    tops = [b["center"][2] + b["size"][2] / 2 for b in boxes
            if 1.6 < b["center"][0] < 2.4 and 1.1 < b["center"][1] < 1.5 and b["center"][2] < 1.0]
    assert tops and abs(max(tops) - 0.737) < 1e-3  # exact counter height, not the voxel grid


# --------------------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def scene_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("rl")
    ses = make_session(root / "session", n_frames=60, color_size=(320, 240), depth_size=(160, 120))
    # objects declared by hand: a bottle on the synthetic table, a cupboard with a hinged door
    c = o3d.geometry.TriangleMesh.create_cylinder(0.035, 0.24, 32, 1).translate((0, 0, 0.12))
    write_plain_obj(root / "bottle.obj", *_vf(c))
    body = o3d.geometry.TriangleMesh.create_box(0.5, 0.4, 0.8).translate((-0.25, -0.2, 0))
    write_plain_obj(root / "body.obj", *_vf(body))
    door = o3d.geometry.TriangleMesh.create_box(0.5, 0.02, 0.78).translate((-0.25, -0.225, 0.01))
    write_plain_obj(root / "door.obj", *_vf(door))
    (root / "objects.json").write_text(json.dumps({"objects": [
        {"id": "bottle_1", "class": "bottle", "type": "pickable", "mesh": "bottle.obj",
         "pose": {"position": [1.6, -1.2, 0.751], "quaternion": [1, 0, 0, 0]}},
        {"id": "cupboard_1", "class": "cupboard", "type": "articulated",
         "pose": {"position": [0.0, 2.28, 0.0], "quaternion": [1, 0, 0, 0]},
         "links": [{"name": "body", "mesh": "body.obj"}, {"name": "door", "mesh": "door.obj"}],
         "joints": [{"name": "hinge", "type": "revolute", "parent": "body", "child": "door",
                     "axis": [0, 0, 1], "origin": [0.25, -0.235, 0], "lower": 0, "upper": 1.6}]}]}))
    cfg = RLConfig(pipeline=PipelineConfig(name="kitchen", center_xy=False,
                                           texture=TextureConfig(size=512, max_keyframes=20),
                                           fusion=FusionConfig(voxel_size=0.04)),
                   declared_objects=str(root / "objects.json"))
    out = root / "scene"
    rep = process_rl(ses, out, cfg)
    return out, rep


def test_bundle_layout(scene_dir):
    out, rep = scene_dir
    for f in ["scene.json", "report.json", "export/scene.xml", "export/scene.usda", "export/world.sdf",
              "export/world.world", "export/run_gazebo.sh", "export/isaaclab_scene.py",
              "room/collision_boxes.json", "room/models/kitchen/meshes/visual.obj", "room/maps/kitchen.yaml",
              "objects/bottle_1/visual.obj", "objects/bottle_1/collision/000.obj",
              "objects/cupboard_1/links/door/visual.obj"]:
        assert (out / f).exists(), f
    assert (out.parent / "scene.zip").exists()
    d = json.loads((out / "scene.json").read_text())
    assert (d["units"], d["up"]) == ("m", "z")
    bottle = next(o for o in d["objects"] if o["id"] == "bottle_1")
    # PET shell 0.8 mm over the surface of a 7 x 24 cm cylinder
    area = 2 * np.pi * 0.035 * 0.24 + 2 * np.pi * 0.035 ** 2
    assert bottle["mass"] == pytest.approx(1380 * 0.0008 * area, rel=0.03)
    compile((out / "export/isaaclab_scene.py").read_text(), "isaaclab_scene.py", "exec")


def test_mujoco_scene_is_stable_and_door_opens(scene_dir):
    out, _ = scene_dir
    m = mujoco.MjModel.from_xml_path(str(out / "export/scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert all(c.dist > -0.003 for c in d.contact[: d.ncon])  # no interpenetration at t = 0
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bottle_1")
    p0 = d.xpos[b].copy()
    for _ in range(1000):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all()
    assert np.linalg.norm(d.xpos[b] - p0) < 0.01  # rests on the counter-height box
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cupboard_1_hinge")
    dof = m.jnt_dofadr[j]
    for _ in range(1500):
        d.qfrc_applied[dof] = 20.0
        mujoco.mj_step(m, d)
    assert d.qpos[m.jnt_qposadr[j]] == pytest.approx(1.6, abs=0.05)  # swings open to its limit


def test_usd_stage(scene_dir):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    out, _ = scene_dir
    stage = Usd.Stage.Open(str(out / "export/scene.usda"))
    assert UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1.0
    bodies = {str(p.GetPath()): UsdPhysics.MassAPI(p).GetMassAttr().Get()
              for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert set(bodies) == {"/World/bottle_1", "/World/cupboard_1/body", "/World/cupboard_1/door"}
    assert all(m > 0 for m in bodies.values())
    hinge = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/World/cupboard_1/hinge"))
    assert hinge.GetUpperLimitAttr().Get() == pytest.approx(np.degrees(1.6), abs=0.01)
    assert stage.GetPrimAtPath("/World/cupboard_1").HasAPI(UsdPhysics.ArticulationRootAPI)
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) >= 5
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/bottle_1/visual"))
    assert len(mesh.GetPointsAttr().Get()) > 50  # referenced mesh layer resolves


def test_sdf_files(scene_dir):
    out, _ = scene_dir
    world = ET.parse(out / "export/world.sdf").getroot()
    uris = [u.text for u in world.iter("uri")]
    assert {"model://kitchen", "model://bottle_1", "model://cupboard_1"} <= set(uris)
    model = ET.parse(out / "export/models/cupboard_1/model.sdf").getroot()
    joints = {j.get("name"): j.get("type") for j in model.iter("joint")}
    assert joints == {"fix_to_world": "fixed", "hinge": "revolute"}
    for uri in model.iter("uri"):
        rel = uri.text.replace("model://cupboard_1/", "")
        assert (out / "export/models/cupboard_1" / rel).exists()


def test_export_applies_overrides(scene_dir):
    out, _ = scene_dir
    d = json.loads((out / "scene.json").read_text())
    for o in d["objects"]:
        if o["id"] == "bottle_1":
            o["overrides"] = {"mass": 0.5, "friction": 0.9}
        if o["id"] == "cupboard_1":
            o["overrides"] = {"joints": {"hinge": {"upper": 1.2}}}
    (out / "scene.json").write_text(json.dumps(d))
    export_all(out)
    m = mujoco.MjModel.from_xml_path(str(out / "export/scene.xml"))
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "bottle_1")
    assert m.body_mass[b] == pytest.approx(0.5)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cupboard_1_hinge")
    assert m.jnt_range[j][1] == pytest.approx(1.2)
    sc = load_scene(out)
    assert sc.objects[0].root.friction == pytest.approx(0.9)
    # the stored computed values are untouched; overrides are applied on load
    assert load_scene(out, apply_overrides=False).objects[0].root.mass != pytest.approx(0.5)
