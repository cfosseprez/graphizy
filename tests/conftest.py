"""
Pytest fixtures for the Graphizy test suite.

This file provides reusable components like sample data and configured
Graphing instances to all other test files, promoting clean and
non-repetitive test code.
"""
import pytest
import numpy as np
from graphizy import Graphing, GraphizyConfig

@pytest.fixture(scope="module")
def sample_array_data():
    """Provides sample data in the standard numpy array format [id, x, y]."""
    return np.array([
        [1, 10.0, 20.0],
        [2, 110.0, 20.0],
        [3, 60.0, 100.0],
        [4, 160.0, 100.0]
    ], dtype=float)

@pytest.fixture(scope="module")
def sample_dict_data():
    """Provides sample data in the dictionary format."""
    return {
        "id": [1, 2, 3, 4],
        "x": [10.0, 110.0, 60.0, 160.0],
        "y": [20.0, 20.0, 100.0, 100.0]
    }

@pytest.fixture
def default_config():
    """Provides a default GraphizyConfig instance."""
    return GraphizyConfig(dimension=(200, 200))

@pytest.fixture
def grapher(default_config):
    """Provides a default Graphing instance configured for array aspect."""
    return Graphing(config=default_config)

@pytest.fixture
def grapher_dict(default_config):
    """Provides a Graphing instance configured for dict aspect."""
    config = default_config.copy()
    config.graph.aspect = "dict"
    return Graphing(config=config)

@pytest.fixture
def blank_image():
    """Provides a blank 100x100 image for drawing tests."""
    return np.zeros((100, 100, 3), dtype=np.uint8)