"""End-to-end tests on a synthetic room with a walking person and a dog."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import open3d as o3d
import pytest

from cleanscan.fusion import FusionConfig
from cleanscan.pipeline import PipelineConfig, run
from cleanscan.synthetic import PERSON_A, PERSON_B, make_session
from cleanscan.texture import TextureConfig


def _living_points(out_dir, name):
    """Count surface points left where the person walked / the dog ran."""
    rep = json.loads((out_dir / "report.json").read_text())
    T = np.array(rep["transform_capture_to_gazebo"])
    mesh = o3d.io.read_triangle_mesh(str(out_dir / f"{name}_mesh.ply"))
    p = np.asarray(mesh.sample_points_uniformly(200_000).points)
    pw = (p - T[:3, 3]) @ T[:3, :3]  # back into the capture frame (Y up)
    a, b = np.array(PERSON_A), np.array(PERSON_B)
    xz = pw[:, [0, 2]]
    t = np.clip(((xz - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    d = np.linalg.norm(xz - (a + t[:, None] * (b - a)), axis=1)
    person = (d < 0.2) & (pw[:, 1] > 0.15) & (pw[:, 1] < 1.7)
    r = np.hypot(pw[:, 0] / 1.8, pw[:, 2] / 1.5)
    dog = (np.abs(r - 1) < 0.08) & (pw[:, 1] > 0.08) & (pw[:, 1] < 0.45)
    table = (np.abs(pw[:, 0] - 1.6) < 0.7) & (np.abs(pw[:, 2] - 1.2) < 0.45) & (pw[:, 1] < 0.8)
    return int((person & ~table).sum()), int((dog & ~table).sum()), p


def _cfg(name, **kw):
    return PipelineConfig(name=name, texture=TextureConfig(size=512, max_keyframes=30),
                          max_visual_triangles=80_000, fusion=FusionConfig(voxel_size=0.03), **kw)


@pytest.fixture(scope="module")
def masked_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("masked")
    ses = make_session(root / "session", n_frames=100, color_size=(320, 240), depth_size=(160, 120))
    out = root / "out"
    rep = run(ses, out, _cfg("room"))
    return out, rep


def test_people_and_animals_removed_with_masks(masked_run):
    out, _ = masked_run
    person, dog, _ = _living_points(out, "room")
    assert person == 0
    assert dog == 0


def test_floor_is_z0_and_room_size(masked_run):
    out, rep = masked_run
    _, _, p = _living_points(out, "room")
    floor = p[p[:, 2] < 0.1, 2]
    assert abs(np.median(floor)) < 0.01
    ex = rep["extent_m"]
    assert 5.8 < ex[0] < 6.2 and 4.8 < ex[1] < 5.2  # room is 6 x 5 m


def test_gazebo_files(masked_run):
    out, _ = masked_run
    model = out / "models" / "room"
    for f in ["model.sdf", "model.config", "meshes/visual.obj", "meshes/visual.mtl",
              "meshes/visual_texture.png", "meshes/collision.stl"]:
        assert (model / f).exists(), f
    for w in ["worlds/room.world", "worlds/room.sdf", "models/room/model.sdf"]:
        root = ET.parse(out / w).getroot()
        assert root.tag == "sdf"
    assert "model://room/meshes/visual.obj" in (model / "model.sdf").read_text()
    assert "map_Kd visual_texture.png" in (model / "meshes" / "visual.mtl").read_text()


def test_occupancy_map(masked_run):
    out, _ = masked_run
    yaml = (out / "maps" / "room.yaml").read_text()
    assert "resolution: 0.05" in yaml
    data = (out / "maps" / "room.pgm").read_bytes()
    assert data.startswith(b"P5")
    body = np.frombuffer(data[-(len(data) // 2):], np.uint8)
    assert (body == 0).sum() > 100  # walls are occupied
    assert (body == 254).sum() > 1000  # floor is free


def test_ghost_removal_without_masks(tmp_path):
    ses = make_session(tmp_path / "session", n_frames=100, with_masks=False,
                       color_size=(320, 240), depth_size=(160, 120))
    with_ghost = run(ses, tmp_path / "a", _cfg("a", texture_mode="none"))
    without = run(ses, tmp_path / "b", _cfg("b", texture_mode="none", ghost_removal=False))
    pa, da, _ = _living_points(tmp_path / "a", "a")
    pb, db, _ = _living_points(tmp_path / "b", "b")
    assert with_ghost["ghost_triangles_removed"] > 0
    assert pa + da < 0.5 * (pb + db), (pa, da, pb, db)
