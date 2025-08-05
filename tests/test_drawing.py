"""
Tests for the drawing and visualization components.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from graphizy import Graphing
from graphizy.drawing import Visualizer
from graphizy.exceptions import DrawingError
from graphizy.positions import generate_and_format_positions

def test_visualizer_initialization(default_config):
    """Test that the Visualizer is correctly initialized within Graphing."""
    grapher = Graphing(config=default_config)
    assert isinstance(grapher.visualizer, Visualizer)
    assert grapher.visualizer.dimension == (200, 200)

def test_draw_graph_delegation(grapher, sample_array_data):
    """Test that grapher.draw_graph correctly delegates to the visualizer."""
    graph = grapher.make_graph("delaunay", sample_array_data)

    with patch.object(grapher.visualizer, 'draw_graph', wraps=grapher.visualizer.draw_graph) as mock_draw:
        image = grapher.draw_graph(graph)
        mock_draw.assert_called_once()
        # Check that the first argument is the graph
        assert mock_draw.call_args[0][0] == graph
        assert isinstance(image, np.ndarray)
        assert image.shape == (200, 200, 3)

def test_draw_memory_graph_delegation(grapher, sample_array_data):
    """Test that grapher.draw_memory_graph delegates correctly."""
    grapher.init_memory_manager()
    grapher.update_memory_with_graph(grapher.make_graph("delaunay", sample_array_data))
    mem_graph = grapher.make_memory_graph(sample_array_data)

    with patch.object(grapher.visualizer, 'draw_memory_graph', wraps=grapher.visualizer.draw_memory_graph) as mock_draw:
        image = grapher.draw_memory_graph(mem_graph, use_age_colors=True)
        mock_draw.assert_called_once()
        assert isinstance(image, np.ndarray)

@patch('graphizy.drawing.cv2.imwrite')
def test_save_graph(mock_imwrite, grapher, sample_array_data):
    """Test saving a graph image."""
    mock_imwrite.return_value = True
    graph = grapher.make_graph("delaunay", sample_array_data)
    image = grapher.draw_graph(graph)
    grapher.save_graph(image, "test_output.png")
    mock_imwrite.assert_called_once()

@patch('graphizy.drawing.cv2.imshow')
@patch('graphizy.drawing.cv2.waitKey')
@patch('graphizy.drawing.cv2.destroyWindow')
def test_show_graph(mock_destroy, mock_wait, mock_show, grapher, sample_array_data):
    """Test showing a graph image."""
    graph = grapher.make_graph("delaunay", sample_array_data)
    image = grapher.draw_graph(graph)
    grapher.show_graph(image, title="Test Show", block=True)
    mock_show.assert_called_once()
    mock_wait.assert_called_once_with(0)  # block=True means waitKey(0)
    mock_destroy.assert_called_once_with("Test Show")

def test_drawing_errors(grapher, blank_image):
    """Test error conditions for drawing functions."""
    with pytest.raises(DrawingError):
        grapher.draw_graph(None)  # Graph cannot be None

    with pytest.raises(DrawingError):
        grapher.save_graph(blank_image, "")  # Filename cannot be empty

def test_draw_graph_produces_valid_image():
    """Test that draw_graph returns a non-empty image of the correct shape."""
    grapher = Graphing(dimension=(100, 80))
    data = generate_and_format_positions(100, 80, 10)
    graph = grapher.make_graph("delaunay", data)

    # Call the drawing function
    image = grapher.draw_graph(graph)

    # Assert the output is a valid image array
    assert isinstance(image, np.ndarray)
    assert image.shape == (80, 100, 3) # height, width, channels
    assert image.dtype == np.uint8
    assert np.sum(image) > 0 # Check that the image is not just black