"""
Tests for the drawing.py module, focusing on error handling and edge cases.
"""
import pytest
import numpy as np
from graphizy.drawing import draw_point, draw_line, show_graph, save_graph
from graphizy.exceptions import DrawingError

@pytest.fixture
def blank_image():
    """Provides a blank 100x100 image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)

class TestDrawingFunctions:
    """Tests the core drawing utility functions."""

    def test_draw_point_errors(self, blank_image):
        """Test error conditions for draw_point."""
        with pytest.raises(DrawingError, match="Image cannot be None"):
            draw_point(None, (10, 10), (255, 0, 0))
        with pytest.raises(DrawingError, match="Point must have exactly 2 coordinates"):
            draw_point(blank_image, (10,), (255, 0, 0))
        with pytest.raises(DrawingError, match="Radius must be >= 1"):
            draw_point(blank_image, (10, 10), (255, 0, 0), radius=0)

    def test_draw_line_errors(self, blank_image):
        """Test error conditions for draw_line."""
        with pytest.raises(DrawingError, match="Image cannot be None"):
            draw_line(None, 0, 0, 10, 10, (255, 0, 0))
        with pytest.raises(DrawingError, match="Thickness must be >= 1"):
            draw_line(blank_image, 0, 0, 10, 10, (255, 0, 0), thickness=0)

    def test_show_graph_errors(self):
        """Test error conditions for show_graph."""
        with pytest.raises(DrawingError, match="image must be a valid numpy array"):
            show_graph(None)
        with pytest.raises(DrawingError, match="image is empty"):
            show_graph(np.array([]))

    def test_save_graph_errors(self, blank_image):
        """Test error conditions for save_graph."""
        with pytest.raises(DrawingError, match="Image cannot be None"):
            save_graph(None, "test.png")
        with pytest.raises(DrawingError, match="Filename cannot be empty"):
            save_graph(blank_image, "")