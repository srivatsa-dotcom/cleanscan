"""What the user selected in the app, and where each object is in the reconstructed room.

Reads ``tracks.json`` / ``selection.json`` / ``instances/`` (docs/RL_SCENES_PLAN.md §2) and
lifts the phone's 2D instance masks into 3D: every vertex of the fused room is projected into
each room-pass frame where it is visible (its depth agrees with the frame's depth) and votes
for the instance label it lands on. A vertex belongs to object k when most of the frames that
saw it say k.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .session import Session, load_label_image

log = logging.getLogger(__name__)

# default interaction mode per class (the app uses the same table); user choice wins
MODE_BY_CLASS = {
    **dict.fromkeys(["bottle", "cup", "mug", "bowl", "box", "book", "fork", "knife", "spoon", "vase",
                     "remote", "cell phone", "banana", "apple", "orange", "can", "jar", "pressure cooker"],
                    "pickable"),
    **dict.fromkeys(["refrigerator", "fridge", "cupboard", "cabinet", "drawer", "oven", "microwave", "door",
                     "dishwasher", "toilet"], "articulated"),
    **dict.fromkeys(["wine glass", "glass", "tumbler", "steel tumbler"], "transparent"),
}


def default_mode(cls: str) -> str:
    return MODE_BY_CLASS.get((cls or "").lower(), "pickable")


@dataclass
class Selected:
    id: int
    cls: str
    name: str
    mode: str
    source: str  # detected | tap
    captures: dict  # stage -> folder (relative to the session)
    seed_world: np.ndarray | None = None  # capture frame (Y up), tap-added objects
    region: np.ndarray | None = None  # room points of this object, scene frame (Z up)
    info: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        base = "".join(c if c.isalnum() else "_" for c in (self.name or self.cls).lower()).strip("_")
        return f"{base or 'object'}_{self.id}"


def load_selection(session_dir: Path) -> list[Selected]:
    session_dir = Path(session_dir)
    p = session_dir / "selection.json"
    if not p.exists():
        return []
    out = []
    for o in json.loads(p.read_text()).get("objects", []):
        seed = o.get("seed", {}).get("world")
        out.append(Selected(id=int(o["id"]), cls=o.get("class", "object"), name=o.get("name") or o.get("class"),
                            mode=o.get("mode") or default_mode(o.get("class")), source=o.get("source", "detected"),
                            captures=o.get("captures", {}),
                            seed_world=None if seed is None else np.asarray(seed, float)))
    return out


def lift_instances(session: Session, vertices_capture: np.ndarray, labels: set[int],
                   depth_tol: float = 0.05, min_votes: int = 2, min_frac: float = 0.5) -> dict[int, np.ndarray]:
    """For each label, the indices of room vertices (capture frame) that belong to it."""
    if not labels:
        return {}
    V = np.asarray(vertices_capture, float)
    votes = {k: np.zeros(len(V), np.int32) for k in labels}
    seen = np.zeros(len(V), np.int32)
    for fr in session.frames:
        lab = load_label_image(fr.extra_path("instances"), fr.depth_size)
        if lab is None or fr.depth_path is None:
            continue
        depth = session.load_depth(fr)
        E = fr.extrinsic_cv
        pc = V @ E[:3, :3].T + E[:3, 3]
        z = pc[:, 2]
        K = fr.K_depth
        w, h = fr.depth_size
        ok = z > 0.1
        u = np.full(len(V), -1, np.int64)
        v = np.full(len(V), -1, np.int64)
        u[ok] = np.floor(K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]).astype(np.int64)
        v[ok] = np.floor(K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]).astype(np.int64)
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.nonzero(ok)[0]
        d = depth[v[idx], u[idx]]
        vis = (d > 0) & (np.abs(d - z[idx]) < depth_tol + 0.02 * z[idx])
        idx = idx[vis]
        seen[idx] += 1
        lv = lab[v[idx], u[idx]]
        for k in labels:
            votes[k][idx[lv == k]] += 1
    out = {}
    for k in labels:
        sel = (votes[k] >= min_votes) & (votes[k] >= min_frac * np.maximum(seen, 1))
        out[k] = np.nonzero(sel)[0]
    return out


def locate_objects(session: Session, selected: list[Selected], room_vertices_scene: np.ndarray,
                   T: np.ndarray) -> None:
    """Fill ``Selected.region`` (scene frame) for detected objects from the room pass."""
    labels = {s.id for s in selected if s.source == "detected"}
    if not labels or not any(f.extra.get("instances") for f in session.frames):
        return
    Tinv = np.linalg.inv(T)
    Vc = room_vertices_scene @ Tinv[:3, :3].T + Tinv[:3, 3]
    regions = lift_instances(session, Vc, labels)
    for s in selected:
        if s.id in regions and len(regions[s.id]):
            s.region = room_vertices_scene[regions[s.id]]
            s.info["room_points"] = int(len(s.region))
            log.info("object %s (%s): %d room vertices", s.slug, s.cls, len(s.region))
        elif s.source == "detected":
            log.warning("object %s (%s) was not found in the room reconstruction", s.slug, s.cls)
