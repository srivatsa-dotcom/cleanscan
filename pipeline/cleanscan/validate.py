"""`cleanscan validate <scene>`: does the scene behave in a physics engine?

MuJoCo (``export/scene.xml``), 2 ms steps, 5 s with no external forces:

* finite          no NaN / Inf in the state
* fall_through    no pickable ends below its support surface (or the floor)
* at_rest         no pickable moves more than 2 cm
* penetration     no contact deeper than 3 mm at t = 0 between different bodies
* joint_limits    every joint, driven alone by a bounded force, reaches both limits

USD (``export/scene.usda``, only if ``pxr`` / usd-core is installed): Z up, metres, every
rigid body has positive mass and a collider, joint targets exist, one articulation root
per articulated object.

Results are written to ``report.json["validation"]``; the command exits non-zero on failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scene import load_scene

log = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    duration: float = 5.0
    rest_tolerance: float = 0.02  # m
    fall_tolerance: float = 0.02  # m below the support surface counts as falling through
    penetration_tolerance: float = 0.003  # m
    joint_drive_time: float = 2.0  # s per direction
    joint_tolerance_frac: float = 0.05  # of the joint range ...
    joint_tolerance_rad: float = np.radians(2.0)  # ... or at least this
    joint_tolerance_m: float = 0.01


def _check(name: str, ok: bool, details) -> dict:
    return {"check": name, "ok": bool(ok), "details": details}


def validate_mujoco(scene_dir: Path, cfg: ValidationConfig) -> list[dict]:
    import mujoco

    scene_dir = Path(scene_dir)
    scene = load_scene(scene_dir)
    m = mujoco.MjModel.from_xml_path(str(scene_dir / "export" / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    results = []

    def name(obj, i):
        return mujoco.mj_id2name(m, obj, i) or str(i)

    # ---- penetration at t = 0 ------------------------------------------------------------
    deep = []
    for c in d.contact[: d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        # links joined by a joint are allowed to touch; the room (world body) is not
        if b1 == b2 or (min(b1, b2) > 0 and (m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1)):
            continue
        if c.dist < -cfg.penetration_tolerance:
            deep.append({"geoms": [name(mujoco.mjtObj.mjOBJ_GEOM, c.geom1), name(mujoco.mjtObj.mjOBJ_GEOM, c.geom2)],
                         "depth_m": round(-float(c.dist), 4)})
    results.append(_check("penetration", not deep, {"deepest": sorted(deep, key=lambda x: -x["depth_m"])[:10]}))

    # ---- 5 s at rest -----------------------------------------------------------------------
    supports = {b["object"]: b["top"] for b in scene.room.get("supports", [])}
    pick = {}
    for o in scene.objects:
        if o.type == "pickable":
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o.id)
            if bid >= 0:
                pick[o.id] = bid
    start = {k: d.xpos[b].copy() for k, b in pick.items()}
    bottom0 = {k: _body_bottom(m, d, b) for k, b in pick.items()}
    finite = True
    steps = int(round(cfg.duration / m.opt.timestep))
    for i in range(steps):
        mujoco.mj_step(m, d)
        if i % 50 == 0 and not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            finite = False
            break
    finite = finite and bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
    results.append(_check("finite", finite, {"time_s": round(float(d.time), 3)}))

    moved, fell = {}, {}
    for k, b in pick.items():
        disp = float(np.linalg.norm(d.xpos[b] - start[k])) if finite else float("nan")
        moved[k] = round(disp, 4)
        floor = supports.get(k, bottom0[k])
        if finite and _body_bottom(m, d, b) < min(floor, bottom0[k]) - cfg.fall_tolerance:
            fell[k] = {"support_z": round(floor, 4), "bottom_z": round(_body_bottom(m, d, b), 4)}
    bad = {k: v for k, v in moved.items() if not v <= cfg.rest_tolerance}
    results.append(_check("fall_through", finite and not fell, fell))
    results.append(_check("at_rest", finite and not bad, {"displacement_m": moved, "too_far": bad}))

    # ---- joints ----------------------------------------------------------------------------
    jres, ok = {}, True
    for o in scene.objects:
        for j in o.joints:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{o.id}_{j.name}")
            if jid < 0:
                jres[f"{o.id}/{j.name}"] = "missing"
                ok = False
                continue
            r = _drive_joint(m, jid, j, cfg)
            jres[f"{o.id}/{j.name}"] = r
            ok &= r["reached_upper"] and r["reached_lower"]
    results.append(_check("joint_limits", ok, jres))
    return results


def _body_bottom(m, d, bid: int) -> float:
    """Lowest point of a body's collision geoms (mesh hull vertices) in the world."""
    import mujoco

    z = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_contype[g] == 0:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = m.mesh_vert[a:a + n] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            z.append(v[:, 2].min())
        else:
            z.append(d.geom_xpos[g][2] - m.geom_rbound[g])
    return float(min(z)) if z else float(d.xpos[bid][2])


def _drive_joint(m, jid: int, j, cfg: ValidationConfig) -> dict:
    """From rest, push the joint with a bounded PD force towards each limit."""
    import mujoco

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    dof, qadr = m.jnt_dofadr[jid], m.jnt_qposadr[jid]
    child = m.jnt_bodyid[jid]
    mass = float(m.body_subtreemass[child])
    revolute = m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
    span = float(j.upper - j.lower)
    tol = max(cfg.joint_tolerance_frac * span, cfg.joint_tolerance_rad if revolute else cfg.joint_tolerance_m)
    # enough to lift / swing the part against gravity, damping and friction, not enough to
    # tunnel through obstacles
    fmax = (12.0 + 6.0 * mass) if revolute else (20.0 + 15.0 * mass)
    kp = fmax / max(span, 1e-3) * 4
    out = {"type": j.type, "lower": j.lower, "upper": j.upper, "tolerance": round(tol, 4)}
    for target, key in ((j.upper, "upper"), (j.lower, "lower")):
        for _ in range(int(cfg.joint_drive_time / m.opt.timestep)):
            q, qd = d.qpos[qadr], d.qvel[dof]
            d.qfrc_applied[dof] = float(np.clip(kp * (target - q) - 0.2 * kp * qd * 0.1, -fmax, fmax))
            mujoco.mj_step(m, d)
        q = float(d.qpos[qadr])
        out[f"reached_{key}"] = bool(np.isfinite(q) and abs(q - target) <= tol)
        out[f"value_at_{key}"] = round(q, 4)
    return out


def validate_usd(scene_dir: Path) -> list[dict]:
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        return [_check("usd", True, "skipped: pxr not installed (pip install usd-core)")]
    scene = load_scene(Path(scene_dir))
    stage = Usd.Stage.Open(str(Path(scene_dir) / "export" / "scene.usda"))
    problems = []
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        problems.append("upAxis is not Z")
    if UsdGeom.GetStageMetersPerUnit(stage) != 1.0:
        problems.append("metersPerUnit is not 1")
    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    for p in rigid:
        mass = UsdPhysics.MassAPI(p).GetMassAttr().Get()
        if not mass or mass <= 0:
            problems.append(f"{p.GetPath()}: no positive mass")
        if not any(c.HasAPI(UsdPhysics.CollisionAPI) for c in Usd.PrimRange(p)):
            problems.append(f"{p.GetPath()}: no collider")
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Joint):
            jt = UsdPhysics.Joint(p)
            for rel in (jt.GetBody0Rel(), jt.GetBody1Rel()):
                for t in rel.GetTargets():
                    if not stage.GetPrimAtPath(t).IsValid():
                        problems.append(f"{p.GetPath()}: target {t} missing")
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    n_art = sum(o.type == "articulated" for o in scene.objects)
    if len(roots) != n_art:
        problems.append(f"{len(roots)} articulation roots for {n_art} articulated objects")
    n_rigid = sum(len(o.links) for o in scene.objects if o.type != "static")
    if len(rigid) != n_rigid:
        problems.append(f"{len(rigid)} rigid bodies, expected {n_rigid}")
    return [_check("usd", not problems, {"problems": problems, "rigid_bodies": len(rigid),
                                         "articulation_roots": len(roots)})]


def validate(scene_dir: str | Path, cfg: ValidationConfig | None = None) -> dict:
    cfg = cfg or ValidationConfig()
    scene_dir = Path(scene_dir)
    checks = validate_mujoco(scene_dir, cfg) + validate_usd(scene_dir)
    result = {"ok": all(c["ok"] for c in checks), "engine": "mujoco", "checks": checks}
    try:
        import mujoco

        result["mujoco_version"] = mujoco.__version__
    except ImportError:  # pragma: no cover
        pass
    rp = scene_dir / "report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {}
    report["validation"] = result
    rp.write_text(json.dumps(report, indent=2, default=str))
    for c in checks:
        log.log(logging.INFO if c["ok"] else logging.ERROR, "%-13s %s", c["check"], "ok" if c["ok"] else "FAILED")
    return result
