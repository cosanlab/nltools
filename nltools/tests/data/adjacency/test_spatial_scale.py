"""Adjacency has no implicit spatial mapping contract."""

import importlib.util
import inspect

import numpy as np
import pytest

from nltools.data import Adjacency


def test_adjacency_has_no_spatial_state_or_projection_api():
    adj = Adjacency(np.ones(3))
    assert not hasattr(adj, "spatial_scale")
    assert not hasattr(adj, "to_brain")
    assert "spatial_scale" not in inspect.signature(Adjacency).parameters
    assert "project" not in inspect.signature(adj.similarity).parameters
    assert importlib.util.find_spec("nltools.data.adjacency.spatial") is None
    with pytest.raises(TypeError):
        Adjacency(np.ones(3), spatial_scale=None)
    with pytest.raises(TypeError):
        adj.similarity(adj, project=True)
