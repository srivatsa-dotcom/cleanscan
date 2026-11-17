"""Shared fixtures: the synthetic kitchen session and its RL scene are built once per test run."""

import pytest

from cleanscan.fusion import FusionConfig
from cleanscan.pipeline import PipelineConfig
from cleanscan.rl import RLConfig, process_rl
from cleanscan.synthetic_kitchen import make_kitchen_session


@pytest.fixture(scope="session")
def kitchen_session(tmp_path_factory):
    root = tmp_path_factory.mktemp("kitchen_session")
    return make_kitchen_session(root / "session", room_frames=100, capture_frames=30,
                                color_size=(320, 240), depth_size=(160, 120))


@pytest.fixture(scope="session")
def kitchen_scene(tmp_path_factory, kitchen_session):
    """`process-rl` on the kitchen with objects from the app's selection (not from truth)."""
    out = tmp_path_factory.mktemp("kitchen_scene") / "scene"
    cfg = RLConfig(pipeline=PipelineConfig(name="kitchen", texture_mode="none",
                                           fusion=FusionConfig(voxel_size=0.03)), zip=False)
    rep = process_rl(kitchen_session, out, cfg)
    return kitchen_session, out, rep
