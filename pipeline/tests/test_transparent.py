"""Transparent objects (milestone 5): glass from side silhouettes, without depth on the glass."""

import json

import numpy as np
import pytest

from cleanscan import physics
from cleanscan.scene import read_obj_vf
from cleanscan.session import Session
from cleanscan.synthetic_kitchen import load_truth
from cleanscan.transparent import TransparentConfig, reconstruct_transparent

pytest.importorskip("mujoco")


def _glass_truth(ses):
    return next(t for t in load_truth(ses)["objects"] if t["id"] == "glass")


def test_glass_in_scene_pose_mass_and_look(kitchen_scene):
    ses, out, rep = kitchen_scene
    sc = json.loads((out / "scene.json").read_text())
    T = np.array(sc["room"]["transform_capture_to_scene"])
    o = next(o for o in sc["objects"] if o["name"] == "glass")
    t = _glass_truth(ses)
    gt = T @ np.array(t["pose_capture"])
    assert o["type"] == "pickable" and o["material"] == "glass"
    assert np.linalg.norm(np.array(o["pose"]["position"]) - gt[:3, 3]) < 0.02
    assert abs(o["mass"] / sum(t["mass"].values()) - 1) < 0.30
    info = next(x for x in rep["objects"] if x["id"] == o["id"])
    assert info["shape"] == "surface of revolution"
    rec = info["reconstruction"]
    assert abs(rec["height_m"] - 0.11) < 0.01 and abs(rec["max_radius_m"] - 0.038) < 0.004
    assert abs(rec["top_view_rim_radius_m"] - 0.038) < 0.004  # independent check from above
    assert "d 0.35" in (out / "objects" / o["id"] / "visual.mtl").read_text()
    assert "inputs:opacity" in (out / "export" / "scene.usda").read_text()


def test_profile_matches_the_true_shape(kitchen_session):
    t = _glass_truth(kitchen_session)
    res = reconstruct_transparent(Session.load(kitchen_session / "captures/4/profile"),
                                  Session.load(kitchen_session / "captures/4/top"))
    assert physics.is_watertight(res.F)
    V, _ = read_obj_vf(kitchen_session / t["links"]["base"])
    # true tumbler: radius 3.0 cm at the base growing to 3.8 cm at 11 cm
    for z, r_true in ((0.02, 0.0315), (0.055, 0.034), (0.09, 0.0365)):
        r = np.interp(z, res.profile[:, 1], res.profile[:, 0])
        assert abs(r - r_true) < 0.003, (z, r, r_true)
    spec = physics.resolve("glass", physics.load_materials())
    gt = physics.mass_properties(*read_obj_vf(kitchen_session / t["links"]["base"]), spec)
    assert abs(physics.mass_properties(res.V, res.F, spec).mass / gt.mass - 1) < 0.1
    assert abs(V[:, 2].max() - res.V[:, 2].max()) < 0.01


def test_cylinder_fallback_with_too_few_views(kitchen_session):
    res = reconstruct_transparent(Session.load(kitchen_session / "captures/4/profile"),
                                  cfg=TransparentConfig(min_views=99))
    assert res.info["shape"].startswith("cylinder")
    assert abs(res.V[:, 2].max() - 0.11) < 0.01 and 0.03 < np.abs(res.V[:, 0]).max() < 0.045


def test_needs_object_masks(kitchen_session, tmp_path):
    import shutil

    cap = tmp_path / "profile"
    shutil.copytree(kitchen_session / "captures/4/profile", cap)
    shutil.rmtree(cap / "object_mask")
    with pytest.raises(RuntimeError, match="object mask"):
        reconstruct_transparent(Session.load(cap))
