"""
Tests for the configuration classes in graphizy.config.
"""
import pytest
from graphizy import GraphizyConfig, DrawingConfig, GraphConfig, MemoryConfig
from graphizy.exceptions import InvalidDimensionError, InvalidAspectError

def test_drawing_config_defaults():
    """Test DrawingConfig default values."""
    config = DrawingConfig()
    assert config.line_color == (0, 255, 0)
    assert config.point_radius == 8

def test_drawing_config_validation():
    """Test DrawingConfig validation raises errors on invalid input."""
    with pytest.raises(ValueError):
        DrawingConfig(line_color=(255, 0))  # Wrong length
    with pytest.raises(ValueError):
        DrawingConfig(line_thickness=0)

def test_graph_config_defaults():
    """Test GraphConfig default values."""
    config = GraphConfig()
    assert config.dimension == (1200, 1200)
    assert config.aspect == "array"

def test_graph_config_validation():
    """Test GraphConfig validation raises errors on invalid input."""
    with pytest.raises(InvalidDimensionError):
        GraphConfig(dimension=(1200,))
    with pytest.raises(InvalidAspectError):
        GraphConfig(aspect="invalid_aspect")

def test_memory_config_validation():
    """Test MemoryConfig validation."""
    with pytest.raises(ValueError):
        MemoryConfig(max_memory_size=0)
    with pytest.raises(ValueError):
        MemoryConfig(max_iterations=0)

def test_graphizy_config_initialization():
    """Test master GraphizyConfig initialization and nested structure."""
    config = GraphizyConfig()
    assert isinstance(config.drawing, DrawingConfig)
    assert isinstance(config.graph, GraphConfig)
    assert isinstance(config.memory, MemoryConfig)
    assert config.graph.dimension == (1200, 1200)

def test_graphizy_config_update():
    """Test the update functionality of GraphizyConfig."""
    config = GraphizyConfig()
    original_line_color = config.drawing.line_color

    # Update a nested value
    config.update(drawing={"point_radius": 15})
    assert config.drawing.point_radius == 15
    # Ensure other values are not changed
    assert config.drawing.line_color == original_line_color

    # Update a top-level config object
    config.update(graph={"dimension": (800, 600)})
    assert config.graph.dimension == (800, 600)

    # Test updating with an invalid key
    with pytest.raises(ValueError):
        config.update(invalid_section={"key": "value"})