"""
Tests for core, low-level functions in graphizy.algorithms.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from graphizy.algorithms import (
    create_graph_array, make_subdiv, get_distance, graph_delaunay,
    normalize_id
)
from graphizy.exceptions import SubdivisionError, GraphCreationError

def test_normalize_id():
    """Test the normalize_id function for consistent ID formatting."""
    assert normalize_id(1.0) == "1"
    assert normalize_id(2) == "2"
    assert normalize_id("3.14") == "3.14"
    assert normalize_id(np.float64(4.0)) == "4"
    assert normalize_id(np.int32(5)) == "5"

def test_create_graph_array(sample_array_data):
    """Test graph creation from a standard numpy array."""
    graph = create_graph_array(sample_array_data)
    assert graph.vcount() == 4
    assert graph.ecount() == 0
    assert list(graph.vs["id"]) == [1.0, 2.0, 3.0, 4.0]
    assert list(graph.vs["x"]) == [10.0, 110.0, 60.0, 160.0]

def test_create_graph_array_validation():
    """Test that create_graph_array raises errors for invalid input."""
    with pytest.raises(GraphCreationError):
        create_graph_array(np.array([]))  # Empty
    with pytest.raises(GraphCreationError):
        create_graph_array(np.array([1, 2]))  # 1D array
    with pytest.raises(GraphCreationError):
        create_graph_array(np.array([[1, 10], [2, 20]]))  # Not enough columns

@patch('graphizy.algorithms.cv2')
def test_make_subdiv(mock_cv2):
    """Test the OpenCV subdivision wrapper."""
    mock_subdiv_instance = Mock()
    mock_cv2.Subdiv2D.return_value = mock_subdiv_instance
    points = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    dimensions = (100, 100)

    result = make_subdiv(points, dimensions)
    mock_cv2.Subdiv2D.assert_called_once_with((0, 0, 100, 100))
    assert mock_subdiv_instance.insert.call_count == 2
    assert result == mock_subdiv_instance

def test_make_subdiv_bounds_error():
    """Test that make_subdiv correctly checks for out-of-bounds points."""
    points = np.array([[150.0, 50.0]], dtype=np.float32)  # x > width
    dimensions = (100, 100)
    with pytest.raises(SubdivisionError, match="points with X >= 100"):
        make_subdiv(points, dimensions)

def test_get_distance():
    """Test the get_distance utility."""
    positions = np.array([[0, 0], [3, 4], [1, 1]])  # Distances: 5, sqrt(2)
    result = get_distance(positions, proximity_thresh=2.0)
    # Point 0 is close to point 2
    assert 2 in result[0]
    # Point 0 is not close to point 1
    assert 1 not in result[0]