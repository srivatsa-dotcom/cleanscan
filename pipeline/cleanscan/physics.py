"""Physics properties of scanned objects: materials, mass, inertia, collision shapes.

* ``materials.yaml`` maps an object class to a material (density, friction) and says
  whether the object is a hollow shell (bottle, glass, fridge) or solid.
* Mass and inertia come from the mesh: exact tetrahedral integration for solids,
  thin-shell surface integration for hollow objects.
* Collision shapes: convex decomposition with CoACD or V-HACD when installed (both
  optional), else a single convex hull.

All quantities are SI (m, kg, kg m^2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

WATER_DENSITY = 1000.0


# --------------------------------------------------------------------------- materials


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_materials(extra: str | Path | dict | None = None) -> dict:
    """The packaged table, with ``extra`` (a YAML path or a dict) merged on top."""
    import yaml

    table = yaml.safe_load(resources.files("cleanscan").joinpath("data/materials.yaml").read_text())
    if extra is not None:
        add = extra if isinstance(extra, dict) else yaml.safe_load(Path(extra).read_text())
        table = _deep_merge(table, add or {})
    return table


@dataclass
class PhysicsSpec:
    material: str
    density: float
    friction: float
    restitution: float
    hollow: bool
    shell_thickness: float = 0.0
    fill: float = 0.0


def resolve(cls: str, table: dict, overrides: dict | None = None) -> PhysicsSpec:
    """Physics spec for an object class, e.g. ``resolve("bottle", load_materials())``."""
    entry = dict(table.get("default", {}))
    entry.update(table.get("classes", {}).get((cls or "").lower(), {}))
    entry.update({k: v for k, v in (overrides or {}).items()
                  if k in ("material", "hollow", "shell_thickness", "fill", "density",
                           "friction", "restitution")})
    mat_name = entry.get("material", "generic")
    mat = dict(table.get("materials", {}).get(mat_name, {}))
    if not mat and "density" not in entry:
        log.warning("unknown material %r, using generic", mat_name)
        mat = dict(table["materials"]["generic"])
    return PhysicsSpec(
        material=mat_name,
        density=float(entry.get("density", mat.get("density", 800.0))),
        friction=float(entry.get("friction", mat.get("friction", 0.5))),
        restitution=float(entry.get("restitution", mat.get("restitution", 0.1))),
        hollow=bool(entry.get("hollow", False)),
        shell_thickness=float(entry.get("shell_thickness", 0.0)),
        fill=float(entry.get("fill", 0.0)),
    )


# --------------------------------------------------------------------------- mass properties


@dataclass
class MassProperties:
    mass: float
    com: np.ndarray  # (3,)
    inertia: np.ndarray  # (3, 3) about the COM, in the mesh frame
    volume: float  # enclosed volume (0 if the mesh is open)
    area: float
    model: dict = field(default_factory=dict)  # how it was computed (for overrides/report)

    def principal(self) -> tuple[np.ndarray, np.ndarray]:
        """(diagonal inertia, rotation whose columns are the principal axes, det +1)."""
        w, R = np.linalg.eigh(self.inertia)
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        return w, R


def _solid_integrals(V: np.ndarray, F: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, first moment and second-moment matrix ∫ x xᵀ dV of a closed mesh
    (sum of signed tetrahedra from the origin)."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    s = a + b + c  # (4th vertex is the origin)
    first = (vol[:, None] * s / 4.0).sum(0)
    outer = (np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b)
             + np.einsum("ni,nj->nij", c, c) + np.einsum("ni,nj->nij", s, s))
    second = (vol[:, None, None] / 20.0 * outer).sum(0)
    return float(vol.sum()), first, second


def _shell_integrals(V: np.ndarray, F: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Area, first moment and ∫ x xᵀ dA of the surface."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    s = a + b + c
    first = (area[:, None] * s / 3.0).sum(0)
    outer = (np.einsum("ni,nj->nij", a, a) + np.einsum("ni,nj->nij", b, b)
             + np.einsum("ni,nj->nij", c, c) + np.einsum("ni,nj->nij", s, s))
    second = (area[:, None, None] / 12.0 * outer).sum(0)
    return float(area.sum()), first, second


def _inertia_from(mass: float, first: np.ndarray, second: np.ndarray, measure: float):
    """Convert ∫x and ∫xxᵀ (over a measure) with uniform density to (com, inertia about com)."""
    com = first / measure
    C = second * (mass / measure) - mass * np.outer(com, com)  # covariance about the COM
    inertia = np.trace(C) * np.eye(3) - C
    return com, 0.5 * (inertia + inertia.T)


def is_watertight(F: np.ndarray) -> bool:
    """Every edge is shared by exactly two triangles with opposite orientation."""
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fwd = {tuple(x) for x in e}
    if len(fwd) != len(e):
        return False
    return all((b, a) in fwd for a, b in fwd)


def mass_properties(V, F, spec: PhysicsSpec) -> MassProperties:
    """Mass, COM and inertia of a mesh with uniform density.

    Solid objects need a closed mesh. Hollow objects are a shell of
    ``spec.shell_thickness`` over the surface, plus ``spec.fill`` (0..1) of the inner
    volume filled with water (modelled as a uniform solid of that mass)."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    vol, first_v, second_v = _solid_integrals(V, F)
    if vol < 0:  # inward-facing triangles: flip
        vol, first_v, second_v = -vol, -first_v, -second_v
    area, first_a, second_a = _shell_integrals(V, F)

    if spec.hollow and spec.shell_thickness > 0:
        mass = spec.density * area * spec.shell_thickness
        com, inertia = _inertia_from(mass, first_a, second_a, area)
        model = {"kind": "shell", "area": area, "shell_thickness": spec.shell_thickness}
        if spec.fill > 0 and vol > 1e-9:
            m_l = WATER_DENSITY * spec.fill * vol
            com_l, I_l = _inertia_from(m_l, first_v, second_v, vol)
            total = mass + m_l
            com_t = (mass * com + m_l * com_l) / total

            def shift(I, m, c):
                d = c - com_t
                return I + m * (d @ d * np.eye(3) - np.outer(d, d))

            inertia = shift(inertia, mass, com) + shift(I_l, m_l, com_l)
            mass, com = total, com_t
            model["fill"] = spec.fill
    else:
        if vol < 1e-9:
            raise ValueError("solid mass properties need a closed mesh with volume")
        mass = spec.density * vol
        com, inertia = _inertia_from(mass, first_v, second_v, vol)
        model = {"kind": "solid"}
    # keep the tensor physically valid (flat or tiny parts): eigenvalues >= small floor and
    # the triangle inequality, which simulators check
    w, R = np.linalg.eigh(inertia)
    floor = max(1e-9, 1e-4 * w.max())
    w = np.maximum(w, floor)
    for i in range(3):
        others = w.sum() - w[i]
        w[i] = min(w[i], others)
    inertia = R @ np.diag(w) @ R.T
    model.update(density=spec.density, material=spec.material)
    return MassProperties(float(mass), com, inertia, vol, area, model)


# --------------------------------------------------------------------------- convex shapes


def convex_hull(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull with outward-facing, consistently ordered triangles."""
    from scipy.spatial import ConvexHull

    V = np.asarray(V, np.float64)
    hull = ConvexHull(V)
    idx = np.unique(hull.simplices)
    remap = -np.ones(len(V), np.int64)
    remap[idx] = np.arange(len(idx))
    P = V[idx]
    F = remap[hull.simplices]
    c = P.mean(0)
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    flip = np.einsum("ij,ij->i", n, P[F[:, 0]] - c) < 0
    F[flip] = F[flip][:, [0, 2, 1]]
    return P, F


def convex_decomposition(V, F, method: str = "auto", max_hulls: int = 16,
                         threshold: float = 0.05) -> tuple[list[tuple[np.ndarray, np.ndarray]], str]:
    """Split a mesh into convex pieces. ``method``: auto | coacd | vhacd | hull.
    Returns (pieces, method actually used)."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    order = ["coacd", "vhacd", "hull"] if method == "auto" else [method, "hull"]
    for m in order:
        try:
            if m == "coacd":
                import coacd

                coacd.set_log_level("error")
                parts = coacd.run_coacd(coacd.Mesh(V, F), threshold=threshold,
                                        max_convex_hull=max_hulls, seed=0)
                pieces = [convex_hull(np.asarray(p[0])) for p in parts if len(p[0]) >= 4]
            elif m == "vhacd":
                import vhacdx

                parts = vhacdx.compute_vhacd(V, F.astype(np.uint32), maxConvexHulls=max_hulls)
                pieces = [convex_hull(np.asarray(p[0])) for p in parts if len(p[0]) >= 4]
            else:
                pieces = [convex_hull(V)]
            if pieces:
                return pieces, m
        except ImportError:
            continue
        except Exception as exc:  # noqa: BLE001 - a failing optional backend falls through
            log.warning("convex decomposition with %s failed (%s), trying the next method", m, exc)
    raise RuntimeError("convex decomposition failed")
