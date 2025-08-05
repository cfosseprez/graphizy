
import pytest
import numpy as np
from graphizy.algorithms import make_subdiv
from graphizy.exceptions import SubdivisionError

def test_make_subdiv_out_of_bounds():
    """Verify that make_subdiv raises an error for out-of-bounds points."""
    # Create points where one is outside the dimensions
    points = np.array([[10, 10], [150, 150]])
    dimensions = (100, 100) # width, height

    # Use pytest.raises to assert that the correct exception is thrown
    with pytest.raises(SubdivisionError, match="points with X >= 100"):
        make_subdiv(points, dimensions)

def test_make_subdiv_empty_input():
    """Verify that make_subdiv handles empty input correctly."""
    with pytest.raises(SubdivisionError, match="Point array cannot be None or empty"):
        make_subdiv(np.array([]), (100, 100))