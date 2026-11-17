"""Synthetic kitchen + `cleanscan validate` (milestone 2).

A scene built from the kitchen's ground-truth objects must pass every check; deliberately
broken copies must fail the check that is meant to catch them.
"""

import json
import shutil

import cv2
import numpy as np
import pytest

from cleanscan.cli import main
from cleanscan.fusion import FusionConfig
from cleanscan.pipeline import PipelineConfig
from cleanscan.rl import RLConfig, export_all, process_rl
from cleanscan.session import Session
from cleanscan.validate import validate

pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def kitchen(tmp_path_factory, kitchen_session):
    root = tmp_path_factory.mktemp("kitchen")
    ses = kitchen_session
    cfg = RLConfig(pipeline=PipelineConfig(name="kitchen", texture_mode="none",
                                           fusion=FusionConfig(voxel_size=0.03)),
                   objects="truth", zip=False)
    scene = root / "scene"
    process_rl(ses, scene, cfg)
    return ses, scene


def _variant(kitchen, tmp_path, edit):
    """Copy the good scene, apply ``edit(scene_json_dict)`` as overrides, re-export."""
    _, scene = kitchen
    dst = tmp_path / "scene"
    shutil.copytree(scene, dst)
    d = json.loads((dst / "scene.json").read_text())
    edit({o["id"]: o for o in d["objects"]})
    (dst / "scene.json").write_text(json.dumps(d))
    export_all(dst)
    return dst


def _checks(res):
    return {c["check"]: c["ok"] for c in res["checks"]}


def test_kitchen_session_contents(kitchen):
    ses, _ = kitchen
    room = Session.load(ses)
    assert len(room.frames) == 100 and room.has_sensor_depth
    tracks = {t["id"]: t for t in json.loads((ses / "tracks.json").read_text())}
    assert {t["class"] for t in tracks.values()} == {"refrigerator", "bottle", "cup"}
    assert all((ses / t["thumb"]).exists() for t in tracks.values())
    sel = {o["name"]: o for o in json.loads((ses / "selection.json").read_text())["objects"]}
    assert sel["cupboard"]["source"] == "tap" and "seed" in sel["cupboard"]
    assert set(sel["fridge"]["captures"]) == {"closed", "open"}
    assert set(sel["glass"]["captures"]) == {"profile", "top"} and sel["glass"]["mode"] == "transparent"
    for o in sel.values():
        for rel in o["captures"].values():
            cap = Session.load(ses / rel)  # every capture is a normal CleanScan session
            assert cap.frames[0].timestamp > room.frames[-1].timestamp
    # people are masked, tracked objects are labelled, the tapped cupboard is not
    lab = np.stack([cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in sorted((ses / "instances").glob("*.png"))])
    assert set(np.unique(lab)) <= {0, 1, 2, 3, 4} and (lab == 1).any() and not (lab == 101).any()
    assert any(cv2.imread(str(p), 0).any() for p in sorted((ses / "mask").glob("*.png")))
    truth = json.loads((ses / "truth.json").read_text())
    masses = {t["id"]: sum(t["mass"].values()) for t in truth["objects"]}
    assert 0.02 < masses["bottle_a"] < 0.2 and masses["fridge"] > 20


def test_truth_scene_passes_validation(kitchen):
    _, scene = kitchen
    res = validate(scene)
    assert res["ok"], json.dumps(res, indent=1)[:3000]
    joints = next(c for c in res["checks"] if c["check"] == "joint_limits")["details"]
    assert set(joints) == {"fridge/door_hinge", "cupboard/drawer_slide"}
    report = json.loads((scene / "report.json").read_text())
    assert report["validation"]["ok"]
    assert report["room"]["triangles_carved"] > 0
    assert main(["validate", str(scene)]) == 0


def test_detects_interpenetration(kitchen, tmp_path):
    def sink(objs):
        p = objs["bottle_a"]["pose"]
        objs["bottle_a"]["overrides"] = {"pose": {**p, "position": [*p["position"][:2], p["position"][2] - 0.03]}}

    res = validate(_variant(kitchen, tmp_path, sink))
    assert not _checks(res)["penetration"]


def test_detects_objects_not_at_rest(kitchen, tmp_path):
    def lift(objs):
        p = objs["bottle_b"]["pose"]
        objs["bottle_b"]["overrides"] = {"pose": {**p, "position": [*p["position"][:2], p["position"][2] + 0.10]}}

    c = _checks(validate(_variant(kitchen, tmp_path, lift)))
    assert not c["at_rest"] and c["fall_through"] and c["penetration"]


def test_detects_fall_through(kitchen, tmp_path):
    def move_off_counter(objs):
        p = objs["glass"]["pose"]
        objs["glass"]["overrides"] = {"pose": {**p, "position": [0.0, 0.0, p["position"][2]]}}

    c = _checks(validate(_variant(kitchen, tmp_path, move_off_counter)))
    assert not c["fall_through"] and not c["at_rest"]


def test_detects_unreachable_joint_limit(kitchen, tmp_path):
    def stiff(objs):
        objs["cupboard"]["overrides"] = {"joints": {"drawer_slide": {"damping": 1e6}}}

    res = validate(_variant(kitchen, tmp_path, stiff))
    assert not _checks(res)["joint_limits"]
    j = next(c for c in res["checks"] if c["check"] == "joint_limits")["details"]
    assert not j["cupboard/drawer_slide"]["reached_upper"] and j["fridge/door_hinge"]["reached_upper"]


def test_detects_nan(kitchen, tmp_path):
    dst = _variant(kitchen, tmp_path, lambda objs: None)
    xml = dst / "export" / "scene.xml"
    xml.write_text(xml.read_text().replace('gravity="0 0 -9.81"', 'gravity="0 0 nan"'))
    assert not _checks(validate(dst))["finite"]


def test_detects_bad_usd(kitchen, tmp_path):
    pytest.importorskip("pxr")
    dst = _variant(kitchen, tmp_path, lambda objs: None)
    usd = dst / "export" / "scene.usda"
    text = usd.read_text()
    i = text.index("float physics:mass = ")
    j = text.index("\n", i)
    usd.write_text(text[:i] + "float physics:mass = 0" + text[j:])
    res = validate(dst)
    assert not _checks(res)["usd"] and not res["ok"]
    assert main(["validate", str(dst)]) == 1
