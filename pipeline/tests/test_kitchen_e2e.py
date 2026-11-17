"""End to end on the target setup: an iPhone 11 (no depth sensor) with the clip-on A010 depth
module, scanning the synthetic kitchen.

    synth-kitchen (camera-only + simulated A010)  ->  process-rl  ->  validate

Asserts (docs/RL_SCENES_PLAN.md §9): joint types right, axis error < 5 deg, hinge origin error
< 3 cm, slide travel within 3 cm, pickable pose error < 2 cm, masses within 30 % (pickables
and moving links), MuJoCo validation passes, module mount recovered (< 1 deg, < 1 cm).
"""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cleanscan.cli import main
from cleanscan.synthetic_kitchen import load_truth

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    ses, scene = root / "kitchen_session", root / "kitchen_scene"
    assert main(["synth-kitchen", str(ses), "--frames", "100", "--capture-frames", "30",
                 "--small", "--depth-module", "a010"]) == 0
    assert main(["process-rl", str(ses), "-o", str(scene), "--voxel", "0.03", "--texture", "none",
                 "--no-zip"]) == 0
    ok = main(["validate", str(scene)]) == 0
    return ses, scene, ok


def _pose(d):
    w, x, y, z = d["quaternion"]
    M = np.eye(4)
    M[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    M[:3, 3] = d["position"]
    return M


def _line_dist(p1, a1, p2, a2):
    n = np.cross(a1, a2)
    if np.linalg.norm(n) < 1e-6:
        return float(np.linalg.norm(np.cross(p2 - p1, a1)))
    return float(abs((p2 - p1) @ n) / np.linalg.norm(n))


def test_validation_passes(e2e):
    _, scene, ok = e2e
    rep = json.loads((scene / "report.json").read_text())
    assert ok, json.dumps(rep["validation"], indent=1)[:3000]
    assert rep["room"]["depth_source"].startswith("depth module (a010)")


def test_module_mount_recovered(e2e):
    ses, scene, _ = e2e
    truth = np.array(load_truth(ses)["T_camera_module"])
    est = np.array(json.loads((scene / "report.json").read_text())["room"]["depth_module"]["T_camera_module"])
    assert np.degrees(Rotation.from_matrix(truth[:3, :3].T @ est[:3, :3]).magnitude()) < 1.0
    assert np.linalg.norm(truth[:3, 3] - est[:3, 3]) < 0.01


def test_objects_against_ground_truth(e2e):
    ses, scene, _ = e2e
    sc = json.loads((scene / "scene.json").read_text())
    T = np.array(sc["room"]["transform_capture_to_scene"])
    truth = {t["id"]: t for t in load_truth(ses)["objects"]}
    objs = {o["name"]: o for o in sc["objects"]}
    assert set(objs) == set(truth)
    for name, o in objs.items():
        t = truth[name]
        P, G = _pose(o["pose"]), T @ np.array(t["pose_capture"])
        masses = {lk["name"]: lk["mass"] for lk in o["links"]}
        if o["type"] == "pickable":
            assert np.linalg.norm(P[:3, 3] - G[:3, 3]) < 0.02, name  # pose error < 2 cm
            assert abs(sum(masses.values()) / sum(t["mass"].values()) - 1) < 0.30, (name, masses)
            continue
        j, g = o["joints"][0], t["joints"][0]
        assert j["type"] == g["type"], name
        ea, eo = P[:3, :3] @ j["axis"], P[:3, :3] @ j["origin"] + P[:3, 3]
        ga, go = G[:3, :3] @ g["axis"], G[:3, :3] @ g["origin"] + G[:3, 3]
        assert np.degrees(np.arccos(np.clip(ea @ ga, -1, 1))) < 5.0, name
        if j["type"] == "revolute":
            assert _line_dist(eo, ea, go, ga) < 0.03, name
            assert abs(j["upper"] - g["upper"]) < np.radians(5), name
        else:
            assert abs(j["upper"] - g["upper"]) < 0.03, name
        # the body is welded to the world; the moving link's mass is what matters in simulation
        assert abs(masses[j["child"]] / t["mass"][g["child"]] - 1) < 0.30, (name, masses)
