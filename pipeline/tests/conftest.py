"""Shared fixtures: the synthetic kitchen session is generated once per test run."""

import pytest

from cleanscan.synthetic_kitchen import make_kitchen_session


@pytest.fixture(scope="session")
def kitchen_session(tmp_path_factory):
    root = tmp_path_factory.mktemp("kitchen_session")
    return make_kitchen_session(root / "session", room_frames=100, capture_frames=30,
                                color_size=(320, 240), depth_size=(160, 120))
