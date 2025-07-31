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
    return Graphing(config=default_config, data_shape=[("id", int), ("x", float), ("y", float)])

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


@pytest.fixture(scope="module")
def weight_test_data():
    """
    Creates a simple numpy array of points with predictable distances
    specifically for testing weight calculations.
    """
    # (id, x, y)
    # distance(0,1) = 5 (3-4-5 triangle)
    # distance(0,2) = 10
    # distance(1,2) = 5
    # distance(0,3) = 10
    # distance(1,3) = ~8.06
    # distance(2,3) = ~8.94
    return np.array(
        [
            [0, 0, 0],
            [1, 3, 4],
            [2, 6, 8],
            [3, 10, 0],
        ],
        dtype=np.float32,
    )

@pytest.fixture
def weight_test_graph(weight_test_data):
    """Creates a simple igraph.Graph with all edges for weight testing."""
    import igraph as ig
    graph = ig.Graph(n=4, directed=False)
    graph.vs["id"] = weight_test_data[:, 0]
    graph.vs["x"] = weight_test_data[:, 1]
    graph.vs["y"] = weight_test_data[:, 2]
    # Add all possible edges for thorough testing
    graph.add_edges([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)])
    return graph

@pytest.fixture
def graph_with_age(weight_test_graph):
    """Creates a graph with a manual 'age' attribute on edges."""
    graph = weight_test_graph.copy()
    # Ages for edges (0,1), (0,2), (1,2), (0,3), (1,3), (2,3) respectively
    graph.es["age"] = [1, 5, 2, 10, 3, 4]
    return graph