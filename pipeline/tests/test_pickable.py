"""Pickable objects (milestone 3): selection -> capture -> watertight mesh -> pose in the room."""

import json

import numpy as np
import open3d as o3d
import pytest

from cleanscan import physics
from cleanscan.pickable import PickableConfig, fallback_primitive, reconstruct_pickable, register_to_room
from cleanscan.scene import read_obj_vf
from cleanscan.session import Session
from cleanscan.synthetic_kitchen import load_truth
from cleanscan.validate import validate

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def scene(kitchen_scene):
    return kitchen_scene


def _truth(ses):
    return {t["id"]: t for t in load_truth(ses)["objects"]}


def test_bottles_pose_and_mass(scene):
    ses, out, rep = scene
    sc = json.loads((out / "scene.json").read_text())
    T = np.array(sc["room"]["transform_capture_to_scene"])
    truth = _truth(ses)
    bottles = [o for o in sc["objects"] if o["class"] == "bottle"]
    assert {o["name"] for o in bottles} == {"bottle_a", "bottle_b"}
    for o in bottles:
        t = truth[o["name"]]
        gt = T @ np.array(t["pose_capture"])
        err = np.linalg.norm(np.array(o["pose"]["position"]) - gt[:3, 3])
        assert err < 0.02, (o["id"], err)  # pose error < 2 cm
        gt_mass = sum(t["mass"].values())
        assert abs(o["mass"] / gt_mass - 1) < 0.30, (o["id"], o["mass"], gt_mass)  # mass within 30 %
        info = next(x for x in rep["objects"] if x["id"] == o["id"])
        assert info["shape"] == "scanned" and info["reconstruction"]["watertight"]
        assert (out / o["visual"]).exists() and (out / "objects" / o["id"] / "visual_texture.png").exists()


def test_room_is_carved_and_scene_validates(scene):
    _, out, rep = scene
    assert rep["room"]["triangles_carved"] > 0
    # every pickable gets a support patch (the glass is a pickable too)
    assert {s["object"] for s in rep["supports"]} == {"bottle_a_2", "bottle_b_3", "glass_4"}
    assert not [w for w in rep["warnings"] if "bottle" in w]
    res = validate(out)
    assert res["ok"], json.dumps(res, indent=1)[:2000]


def test_reconstruction_without_app_masks(kitchen_session):
    """Phones whose detector missed the object in the capture: depth-only carving."""
    t = _truth(kitchen_session)["bottle_b"]
    res = reconstruct_pickable(Session.load(kitchen_session / "captures/3/scan"),
                               PickableConfig(use_app_masks=False), texture=False)
    assert res.info["watertight"] and not res.info["used_app_masks"]
    spec = physics.resolve("bottle", physics.load_materials())
    V, F = read_obj_vf(kitchen_session / t["links"]["base"])
    gt = physics.mass_properties(V, F, spec)
    mp = physics.mass_properties(res.V, res.F, spec)
    assert abs(mp.mass / gt.mass - 1) < 0.10
    assert abs(res.V[:, 2].max() - V[:, 2].max()) < 0.01  # height within 1 cm


def test_registration_recovers_pose_from_one_sided_points():
    rng = np.random.default_rng(1)
    box = o3d.geometry.TriangleMesh.create_box(0.20, 0.08, 0.12).translate((-0.1, -0.04, 0))
    box.remove_duplicated_vertices()
    V, F = np.asarray(box.vertices), np.asarray(box.triangles)
    yaw, t = 0.7, np.array([1.3, -0.4, 0.9])
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, yaw))
    pts = np.asarray(box.sample_points_uniformly(3000).points) @ R.T + t
    region = pts[(pts[:, 0] < t[0] + 0.02) & (pts[:, 2] > t[2] + 0.01)]  # one side + top, as a room pass sees
    region = region + rng.normal(0, 0.003, region.shape)
    T, info = register_to_room(V, F, region, np.array([0.0, 0.9]))
    assert info["on_support"] and not info["symmetric"]
    assert np.linalg.norm(T[:3, 3] - t) < 0.01
    est = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    d = (est - np.degrees(yaw) + 90) % 180 - 90  # a box looks the same after 180 deg
    assert abs(d) < 3


def test_fallback_primitive_from_room_points():
    pts = np.random.default_rng(0).uniform([-0.04, -0.04, 0.9], [0.04, 0.04, 1.15], (400, 3)) + [0.5, 0.2, 0]
    V, F, T = fallback_primitive(pts, "bottle", np.array([0.0, 0.9]))
    assert physics.is_watertight(F)
    assert abs(T[2, 3] - 0.9) < 1e-6 and abs(T[0, 3] - 0.5) < 0.01
    assert 0.2 < V[:, 2].max() < 0.3


def test_selection_written_by_the_app_loads(kitchen_session):
    """tests/data/app_fixture is written by the app's own Kotlin code (tools/app-logic-test)."""
    from pathlib import Path

    from cleanscan import objects

    fixture = Path(__file__).parent / "data" / "app_fixture"
    sel = {s.id: s for s in objects.load_selection(fixture)}
    assert sel[1].name == 'water "bottle"' and sel[1].mode == "pickable" and sel[1].captures == {"scan": "captures/1/scan"}
    assert sel[101].source == "tap" and sel[101].seed_pixel == (57, 0.41, 0.63) and sel[101].seed_world is None
    tracks = json.loads((fixture / "tracks.json").read_text())
    assert [t["class"] for t in tracks] == ["bottle", "bottle"] and tracks[0]["thumb"] == "thumbs/1.jpg"
    # a tap on a room-pass frame becomes a 3D point from that frame's depth
    ses = Session.load(kitchen_session)
    fr = ses.frames[40]
    p = objects.seed_from_pixel(ses, fr.index, 0.5, 0.5)
    d = ses.load_depth(fr)
    cam = fr.pose_cv[:3, 3]
    assert p is not None and abs(np.linalg.norm(p - cam) - d[d.shape[0] // 2, d.shape[1] // 2]) < 0.1
