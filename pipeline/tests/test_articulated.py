"""Articulated objects (milestone 4): closed + open captures -> revolute / prismatic joints."""

import json

import numpy as np
import open3d as o3d
import pytest
from scipy.spatial.transform import Rotation

from cleanscan import articulated as ar
from cleanscan.synthetic_kitchen import load_truth

pytest.importorskip("mujoco")


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


def _joints(kitchen_scene):
    ses, out, rep = kitchen_scene
    sc = json.loads((out / "scene.json").read_text())
    T = np.array(sc["room"]["transform_capture_to_scene"])
    truth = {t["id"]: t for t in load_truth(ses)["objects"]}
    res = {}
    for o in sc["objects"]:
        if o["type"] != "articulated":
            continue
        t = truth[o["name"]]
        P, G = _pose(o["pose"]), T @ np.array(t["pose_capture"])
        j, g = o["joints"][0], t["joints"][0]
        ea, eo = P[:3, :3] @ j["axis"], P[:3, :3] @ j["origin"] + P[:3, 3]
        ga, go = G[:3, :3] @ g["axis"], G[:3, :3] @ g["origin"] + G[:3, 3]
        masses = {lk["name"]: lk["mass"] for lk in o["links"]}
        res[o["name"]] = {"type": j["type"], "gt_type": g["type"],
                          "axis_err_deg": float(np.degrees(np.arccos(np.clip(ea @ ga, -1, 1)))),
                          "origin_err_m": _line_dist(eo, ea, go, ga), "upper": j["upper"], "gt_upper": g["upper"],
                          "moving_mass": masses[j["child"]], "gt_moving_mass": t["mass"][g["child"]]}
    return res


def test_joint_types_axes_and_origins(kitchen_scene):
    j = _joints(kitchen_scene)
    assert set(j) == {"fridge", "cupboard"}
    for name, r in j.items():
        assert r["type"] == r["gt_type"], name
        assert r["axis_err_deg"] < 5.0, (name, r)  # the axis points the right way (opening = +)
    assert j["fridge"]["type"] == "revolute" and j["fridge"]["origin_err_m"] < 0.03, j["fridge"]
    assert abs(j["fridge"]["upper"] - j["fridge"]["gt_upper"]) < np.radians(5)
    # a slide has no origin; its travel is the comparable quantity
    assert j["cupboard"]["type"] == "prismatic" and abs(j["cupboard"]["upper"] - j["cupboard"]["gt_upper"]) < 0.03


def test_moving_link_masses(kitchen_scene):
    # the body of a fixed-base object is welded to the world, so its mass has no effect in
    # simulation; the moving links' masses do
    for name, r in _joints(kitchen_scene).items():
        assert abs(r["moving_mass"] / r["gt_moving_mass"] - 1) < 0.30, (name, r)


def test_articulated_files_and_report(kitchen_scene):
    _, out, rep = kitchen_scene
    sc = json.loads((out / "scene.json").read_text())
    fridge = next(o for o in sc["objects"] if o["name"] == "fridge")
    assert [lk["name"] for lk in fridge["links"]] == ["body", "door"] and fridge["fixed_base"]
    cup = next(o for o in sc["objects"] if o["name"] == "cupboard")
    assert cup["joints"][0]["child"] == "drawer" and cup["links"][1]["material"] == "laminate"
    for o in (fridge, cup):
        for lk in o["links"]:
            assert (out / lk["visual"]).exists() and lk["collision"]
    assert any("drawer depth inferred" in w for w in rep["warnings"])
    info = next(o for o in rep["objects"] if o["id"] == fridge["id"])
    assert info["joint_fit"]["axis_snapped_deg"] is not None


def test_moving_parts_clear_the_floor(kitchen_scene):
    import mujoco

    _, out, _ = kitchen_scene
    m = mujoco.MjModel.from_xml_path(str(out / "export/scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for name in ("fridge_1_door", "cupboard_101_drawer"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        z = min((m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]]
                             + m.mesh_vertnum[m.geom_dataid[g]]] @ d.geom_xmat[g].reshape(3, 3).T
                 + d.geom_xpos[g])[:, 2].min()
                for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g])
        assert z > 0.002, (name, z)


def _panel(w=0.6, h=1.0, t=0.03, n=6000):
    """Surface points of a door panel with outward normals (= towards whoever sees that face)."""
    box = o3d.geometry.TriangleMesh.create_box(w, t, h).translate((-w / 2, -t, 0))
    box.compute_triangle_normals()
    pc = box.sample_points_uniformly(n, use_triangle_normal=True)
    return np.asarray(pc.points), np.asarray(pc.normals)


def test_fit_joint_recovers_hinge_and_slide():
    rng = np.random.default_rng(0)
    P, N = _panel()
    f = N[:, 1] < -0.9  # the face the user sees while the panel is closed
    front, Nf = P[f], N[f]
    # hinge on the right front edge, vertical, opened 1.4 rad; the open capture sees both
    # big faces (users walk around the open door) but not the top and bottom
    axis, origin = np.array([0, 0, 1.0]), np.array([0.3, -0.03, 0.0])
    M = ar._motion_revolute(axis, origin, 1.4)
    side = np.abs(N[:, 2]) < 0.5
    moved, Nm = P[side] @ M[:3, :3].T + M[:3, 3], N[side] @ M[:3, :3].T
    noisy = front + rng.normal(0, 0.002, front.shape)
    fit = ar.fit_joint(noisy, moved, ar.ArticulatedConfig(), Nc=Nf, No=Nm)
    fit = ar.refine_with_axis(fit, noisy, moved, axis, Nf, Nm)
    assert fit.type == "revolute" and abs(fit.value - 1.4) < np.radians(3)
    assert _line_dist(fit.origin, fit.axis, origin, axis) < 0.02 and fit.axis @ axis > 0.99
    # a drawer front pulled out 0.25 m
    M = ar._motion_prismatic(np.array([0, -1.0, 0]), 0.25)
    fit = ar.fit_joint(front, front @ M[:3, :3].T + M[:3, 3], ar.ArticulatedConfig())
    assert fit.type == "prismatic" and abs(fit.value - 0.25) < 0.01 and fit.axis[1] < -0.99


def test_drawer_depth_prior_extends_to_the_back():
    occ = np.zeros((20, 10, 8), bool)
    occ[:, 0, :] = True  # front panel
    occ[:, :, 0] = True  # tray bottom
    occ[[0, -1], :, :6] = True  # tray sides
    occ2, lo, info = ar._extend_drawer(occ, np.zeros(3), 0.02, np.array([0, -1.0, 0]), body_back_y=0.5)
    assert occ2.shape[1] > occ.shape[1] and info["extended_m"] > 0.2
    assert occ2[:, -1, :6].all()  # closed by a back panel as high as the sides
