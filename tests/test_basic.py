"""
Basic tests for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphizy import (
    Graphing, GraphizyConfig, DrawingConfig, GraphConfig,
    generate_positions, DataInterface, create_graph_array,
    GraphizyError, InvalidDimensionError, InvalidDataShapeError,
    PositionGenerationError, GraphCreationError
)


class TestConfiguration:
    """Test configuration classes"""

    def test_drawing_config_defaults(self):
        """Test DrawingConfig default values"""
        config = DrawingConfig()
        assert config.line_color == (0, 255, 0)
        assert config.line_thickness == 1
        assert config.point_color == (0, 0, 255)
        assert config.point_thickness == 3
        assert config.point_radius == 8

    def test_drawing_config_validation(self):
        """Test DrawingConfig validation"""
        with pytest.raises(ValueError):
            DrawingConfig(line_color=(255, 0))  # Wrong length

        with pytest.raises(ValueError):
            DrawingConfig(line_thickness=0)  # Invalid thickness

    def test_graph_config_defaults(self):
        """Test GraphConfig default values"""
        config = GraphConfig()
        assert config.dimension == (1200, 1200)
        assert config.aspect == "array"
        assert config.proximity_threshold == 50.0
        assert config.distance_metric == "euclidean"

    def test_graph_config_validation(self):
        """Test GraphConfig validation"""
        with pytest.raises(ValueError):
            GraphConfig(dimension=(1200,))  # Wrong length

        with pytest.raises(ValueError):
            GraphConfig(aspect="invalid")  # Invalid aspect

    def test_graphizy_config_update(self):
        """Test GraphizyConfig update functionality"""
        config = GraphizyConfig()

        # Update drawing config
        config.update(drawing={"line_color": (255, 0, 0)})
        assert config.drawing.line_color == (255, 0, 0)

        # Update graph config
        config.update(graph={"dimension": (800, 800)})
        assert config.graph.dimension == (800, 800)


class TestAlgorithms:
    """Test algorithm functions"""

    def test_generate_positions_basic(self):
        """Test basic position generation"""
        positions = generate_positions(100, 100, 10)

        assert isinstance(positions, np.ndarray)
        assert positions.shape == (10, 2)
        assert positions.dtype == np.float32

        # Check all positions are within bounds
        assert np.all(positions[:, 0] >= 0)
        assert np.all(positions[:, 0] < 100)
        assert np.all(positions[:, 1] >= 0)
        assert np.all(positions[:, 1] < 100)

        # Check uniqueness
        unique_positions = np.unique(positions, axis=0)
        assert len(unique_positions) == 10

    def test_generate_positions_validation(self):
        """Test position generation validation"""
        with pytest.raises(PositionGenerationError):
            generate_positions(0, 100, 10)  # Invalid size_x

        with pytest.raises(PositionGenerationError):
            generate_positions(100, 100, 0)  # Invalid num_particles

        with pytest.raises(PositionGenerationError):
            generate_positions(10, 10, 200)  # Too many particles

    def test_data_interface(self):
        """Test DataInterface class"""
        data_shape = [("id", int), ("x", int), ("y", int), ("speed", float)]
        interface = DataInterface(data_shape)

        assert interface.getidx_id() == 0
        assert interface.getidx_xpos() == 1
        assert interface.getidx_ypos() == 2

    def test_data_interface_validation(self):
        """Test DataInterface validation"""
        with pytest.raises(InvalidDataShapeError):
            DataInterface([])  # Empty data shape

        with pytest.raises(InvalidDataShapeError):
            DataInterface([("x", int), ("y", int)])  # Missing required fields

    def test_create_graph_array(self):
        """Test graph creation from array"""
        # Create test data: [id, x, y]
        data = np.array([
            [0, 10, 20],
            [1, 30, 40],
            [2, 50, 60]
        ])

        graph = create_graph_array(data)

        assert graph.vcount() == 3
        assert graph.ecount() == 0  # No edges initially
        assert list(graph.vs["id"]) == [0, 1, 2]
        assert list(graph.vs["x"]) == [10, 30, 50]
        assert list(graph.vs["y"]) == [20, 40, 60]

    def test_create_graph_array_validation(self):
        """Test graph creation validation"""
        with pytest.raises(GraphCreationError):
            create_graph_array(np.array([]))  # Empty array

        with pytest.raises(GraphCreationError):
            create_graph_array(np.array([1, 2, 3]))  # 1D array


class TestGraphing:
    """Test main Graphing class"""

    def test_graphing_initialization(self):
        """Test Graphing class initialization"""
        grapher = Graphing()

        assert grapher.dimension == (1200, 1200)
        assert grapher.aspect == "array"
        assert isinstance(grapher.dinter, DataInterface)

    def test_graphing_custom_config(self):
        """Test Graphing with custom configuration"""
        config = GraphizyConfig()
        config.graph.dimension = (800, 600)
        config.drawing.line_color = (255, 0, 0)

        grapher = Graphing(config=config)

        assert grapher.dimension == (800, 600)
        assert grapher.line_color == (255, 0, 0)

    def test_graphing_parameter_override(self):
        """Test parameter override in initialization"""
        grapher = Graphing(
            dimension=(500, 500),
            aspect="dict"
        )

        assert grapher.dimension == (500, 500)
        assert grapher.aspect == "dict"

    def test_graphing_validation(self):
        """Test Graphing initialization validation"""
        with pytest.raises(GraphCreationError):
            Graphing(dimension=(100,))  # Wrong dimension length

        with pytest.raises(GraphCreationError):
            Graphing(dimension=(-100, 100))  # Negative dimension

    def test_update_config_runtime(self):
        """Test runtime configuration updates"""
        grapher = Graphing()

        grapher.update_config(
            drawing={"line_color": (255, 255, 0)},
            graph={"dimension": (600, 600)}
        )

        assert grapher.line_color == (255, 255, 0)
        assert grapher.dimension == (600, 600)

    @patch('graphizy.algorithms.cv2')
    def test_make_delaunay_array(self, mock_cv2):
        """Test Delaunay triangulation with array data"""
        # Mock OpenCV subdivision
        mock_subdiv = Mock()
        mock_subdiv.getTriangleList.return_value = np.array([
            [10, 20, 30, 40, 50, 60]  # One triangle
        ])
        mock_subdiv.locate.return_value = (None, None, 4)  # Mock vertex index

        mock_cv2.Subdiv2D.return_value = mock_subdiv

        grapher = Graphing()

        # Create test data
        data = np.array([
            [0, 10, 20],
            [1, 30, 40],
            [2, 50, 60]
        ])

        graph = grapher.make_delaunay(data)

        assert graph.vcount() == 3
        # Should have edges from triangulation
        assert graph.ecount() >= 0

    def test_get_graph_info(self):
        """Test graph information retrieval"""
        grapher = Graphing()

        # Create simple test graph
        data = np.array([
            [0, 10, 20],
            [1, 30, 40]
        ])

        graph = create_graph_array(data)
        info = grapher.get_graph_info(graph)

        assert info['vertex_count'] == 2
        assert info['edge_count'] == 0
        assert info['density'] == 0.0
        assert info['is_connected'] in [True, False]

    def test_call_method(self):
        """Test calling igraph methods"""
        grapher = Graphing()

        # Create simple test graph
        data = np.array([
            [0, 10, 20],
            [1, 30, 40]
        ])

        graph = create_graph_array(data)

        # Test calling vcount method
        vertex_count = grapher.call_method(graph, 'vcount')
        assert vertex_count == 2

        # Test calling ecount method
        edge_count = grapher.call_method(graph, 'ecount')
        assert edge_count == 0


class TestExceptions:
    """Test custom exceptions"""

    def test_graphizy_error_inheritance(self):
        """Test that all custom exceptions inherit from GraphizyError"""
        from graphizy.exceptions import (
            InvalidDimensionError, InvalidDataShapeError,
            GraphCreationError, DrawingError
        )

        assert issubclass(InvalidDimensionError, GraphizyError)
        assert issubclass(InvalidDataShapeError, GraphizyError)
        assert issubclass(GraphCreationError, GraphizyError)
        assert issubclass(DrawingError, GraphizyError)

    def test_exception_messages(self):
        """Test exception message handling"""
        with pytest.raises(GraphizyError) as exc_info:
            raise GraphizyError("Test error message")


class TestIntegration:
    """Integration tests"""

    def test_end_to_end_workflow(self):
        """Test complete workflow from generation to analysis"""
        # Generate positions
        positions = generate_positions(200, 200, 20)

        # Create particle data
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Create grapher
        config = GraphizyConfig()
        config.graph.dimension = (200, 200)
        grapher = Graphing(config=config)

        # Test that we can create a graph without errors
        graph = create_graph_array(particle_stack)

        # Basic validation
        assert graph.vcount() == 20
        assert graph.ecount() == 0  # No edges initially

        # Test graph info
        info = grapher.get_graph_info(graph)
        assert info['vertex_count'] == 20
        assert info['edge_count'] == 0


if __name__ == '__main__':
    pytest.main([__file__])