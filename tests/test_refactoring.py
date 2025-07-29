"""
Tests for the refactored data handling in main.py and algorithms.py,
ensuring core logic is robust and increasing test coverage.
"""
import pytest
import numpy as np
from graphizy import Graphing, GraphizyConfig
from graphizy.algorithms import create_knn_graph, create_gabriel_graph, create_minimum_spanning_tree
from graphizy.exceptions import GraphCreationError, InvalidDimensionError

@pytest.fixture
def grapher():
    """Provides a default Graphing instance."""
    return Graphing(config=GraphizyConfig())

@pytest.fixture
def sample_array_data():
    """Provides sample data in the standard numpy array format."""
    return np.array([
        [1, 10.0, 20.0],
        [2, 30.0, 40.0],
        [3, 50.0, 60.0]
    ], dtype=float)

@pytest.fixture
def sample_dict_data():
    """Provides sample data in the dictionary format."""
    return {
        "id": [1, 2, 3],
        "x": [10.0, 30.0, 50.0],
        "y": [20.0, 40.0, 60.0]
    }

class TestGetDataAsArray:
    """Tests the internal _get_data_as_array helper method in Graphing."""

    def test_valid_array(self, grapher, sample_array_data):
        """Test that a valid numpy array passes through correctly."""
        grapher.aspect = "array"
        result = grapher._get_data_as_array(sample_array_data)
        np.testing.assert_array_equal(result, sample_array_data)

    def test_valid_dict_conversion(self, grapher, sample_dict_data, sample_array_data):
        """Test that a valid dictionary is correctly converted to an array."""
        grapher.aspect = "dict"
        result = grapher._get_data_as_array(sample_dict_data)
        np.testing.assert_array_equal(result, sample_array_data)

    def test_array_with_string_ids_fails(self, grapher):
        """Test that an array with string/object IDs raises an error."""
        grapher.aspect = "array"
        bad_data = np.array([['a', 10, 20]], dtype=object)
        with pytest.raises(GraphCreationError, match="requires numeric IDs"):
            grapher._get_data_as_array(bad_data)

    def test_dict_with_missing_keys_fails(self, grapher):
        """Test that a dict with missing keys raises an error."""
        grapher.aspect = "dict"
        bad_data = {"id": [1], "x": [10]}  # Missing 'y'
        with pytest.raises(GraphCreationError, match="must contain keys"):
            grapher._get_data_as_array(bad_data)

    def test_dict_with_mismatched_lengths_fails(self, grapher):
        """Test that a dict with mismatched list lengths raises an error."""
        grapher.aspect = "dict"
        bad_data = {"id": [1, 2], "x": [10], "y": [20, 30]}
        with pytest.raises(GraphCreationError, match="must have the same length"):
            grapher._get_data_as_array(bad_data)

    def test_wrong_data_type_for_aspect(self, grapher, sample_dict_data):
        """Test that providing the wrong data type for the configured aspect fails."""
        grapher.aspect = "array"
        with pytest.raises(GraphCreationError, match="Expected numpy array"):
            grapher._get_data_as_array(sample_dict_data)

class TestSimplifiedAlgorithms:
    """Tests the simplified algorithm functions that now only accept numpy arrays."""

    def test_create_knn_graph(self, sample_array_data):
        """Test the simplified k-NN graph creation."""
        graph = create_knn_graph(sample_array_data, k=2)
        assert graph.vcount() == 3
        assert graph.ecount() > 0

    def test_create_gabriel_graph(self, sample_array_data):
        """Test the simplified Gabriel graph creation."""
        graph = create_gabriel_graph(sample_array_data)
        assert graph.vcount() == 3
        assert graph.ecount() > 0

    def test_create_minimum_spanning_tree(self, sample_array_data):
        """Test the simplified MST graph creation."""
        graph = create_minimum_spanning_tree(sample_array_data, metric='euclidean')
        assert graph.vcount() == 3
        assert graph.ecount() == 2  # An MST with 3 vertices must have 2 edges