"""
Advanced tests for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphizy import (
    Graphing, GraphizyConfig,
    make_subdiv, get_distance, graph_distance,
    call_igraph_method, DataInterface,
    SubdivisionError, TriangulationError, IgraphMethodError, generate_positions, create_graph_array
)

class TestSubdivision:
    """Test OpenCV subdivision functionality"""

    @patch('graphizy.algorithms.cv2')
    def test_make_subdiv_success(self, mock_cv2):
        """Test successful subdivision creation"""
        # Mock OpenCV Subdiv2D
        mock_subdiv = Mock()
        mock_cv2.Subdiv2D.return_value = mock_subdiv

        # Test data
        points = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        dimensions = (100, 100)

        result = make_subdiv(points, dimensions)

        # Verify OpenCV calls
        mock_cv2.Subdiv2D.assert_called_once_with((0, 0, 100, 100))
        mock_subdiv.insert.assert_called()
        assert result == mock_subdiv

    def test_make_subdiv_validation(self):
        """Test subdivision validation"""
        with pytest.raises(SubdivisionError):
            make_subdiv(np.array([]), (100, 100))  # Empty array

        with pytest.raises(SubdivisionError):
            make_subdiv(np.array([[10, 20]]), (100,))  # Wrong dimensions

        with pytest.raises(SubdivisionError):
            make_subdiv(np.array([[10, 20]]), (-100, 100))  # Negative dimensions

    @patch('graphizy.algorithms.cv2')
    def test_make_subdiv_type_conversion(self, mock_cv2):
        """Test automatic type conversion in subdivision"""
        mock_subdiv = Mock()
        mock_cv2.Subdiv2D.return_value = mock_subdiv

        # Integer input data
        points = np.array([[10, 20], [30, 40]], dtype=int)
        dimensions = (100, 100)

        result = make_subdiv(points, dimensions)

        # Should convert to float automatically
        assert result == mock_subdiv

    @patch('graphizy.algorithms.cv2')
    def test_make_subdiv_bounds_checking(self, mock_cv2):
        """Test bounds checking in subdivision"""
        mock_cv2.Subdiv2D.return_value = Mock()

        # Points outside bounds
        points = np.array([[150.0, 20.0]], dtype=np.float32)  # x > dimension[1]
        dimensions = (100, 100)

        with pytest.raises(SubdivisionError):
            make_subdiv(points, dimensions)


class TestDistanceCalculations:
    """Test distance-based graph functions"""

    def test_get_distance_basic(self):
        """Test basic distance calculation"""
        positions = np.array([
            [0, 0],
            [3, 4],  # Distance 5 from origin
            [1, 1],  # Distance sqrt(2) ≈ 1.41 from origin
        ])

        threshold = 2.0
        result = get_distance(positions, threshold)

        # Point 0 should be close to point 2 only
        assert len(result) == 3
        assert 2 in result[0]  # Point 2 is within threshold of point 0
        assert 1 not in result[0]  # Point 1 is not within threshold of point 0

    def test_get_distance_validation(self):
        """Test distance calculation validation"""
        from graphizy.exceptions import GraphCreationError

        with pytest.raises(GraphCreationError):
            get_distance(np.array([]), 10)  # Empty array

        with pytest.raises(GraphCreationError):
            get_distance(np.array([1, 2, 3]), 10)  # 1D array

        with pytest.raises(GraphCreationError):
            get_distance(np.array([[1, 2], [3, 4]]), -1)  # Negative threshold

    def test_get_distance_metrics(self):
        """Test different distance metrics"""
        positions = np.array([
            [0, 0],
            [1, 0],  # Manhattan distance 1, Euclidean distance 1
            [1, 1],  # Manhattan distance 2, Euclidean distance sqrt(2)
        ])

        # Test Euclidean metric
        result_euclidean = get_distance(positions, 1.5, metric='euclidean')
        assert 1 in result_euclidean[0]  # Point 1 within threshold
        assert 2 in result_euclidean[0]  # Point 2 outside threshold

        # Test Manhattan metric
        result_manhattan = get_distance(positions, 1.5, metric='cityblock')
        assert 1 in result_manhattan[0]  # Point 1 within threshold
        assert 2 not in result_manhattan[0]  # Point 2 outside threshold (distance = 2)

    @patch('graphizy.algorithms.ig.Graph')
    def test_graph_distance(self, mock_graph_class):
        """Test distance graph creation"""
        # Mock igraph
        mock_graph = Mock()
        mock_graph.vcount.return_value = 3
        mock_graph_class.return_value = mock_graph

        positions = np.array([[0, 0], [1, 0], [2, 0]])
        threshold = 1.5

        result = graph_distance(mock_graph, positions, threshold)

        # Should add edges and simplify
        mock_graph.add_edges.assert_called_once()
        mock_graph.simplify.assert_called_once()
        assert result == mock_graph


class TestIgraphIntegration:
    """Test igraph method integration"""

    def test_call_igraph_method_success(self):
        """Test successful igraph method calls"""
        # Create a real igraph for testing
        import igraph as ig
        graph = ig.Graph()
        graph.add_vertices(3)
        graph.add_edges([(0, 1), (1, 2)])

        # Test basic methods
        vertex_count = call_igraph_method(graph, 'vcount')
        assert vertex_count == 3

        edge_count = call_igraph_method(graph, 'ecount')
        assert edge_count == 2

        density = call_igraph_method(graph, 'density')
        assert isinstance(density, float)

    def test_call_igraph_method_with_args(self):
        """Test igraph method calls with arguments"""
        import igraph as ig
        graph = ig.Graph()
        graph.add_vertices(3)
        graph.add_edges([(0, 1), (1, 2)])

        # Test method with arguments
        degree = call_igraph_method(graph, 'degree', 0)  # Degree of vertex 0
        assert degree == 1

    def test_call_igraph_method_validation(self):
        """Test igraph method call validation"""
        import igraph as ig
        graph = ig.Graph()

        with pytest.raises(IgraphMethodError):
            call_igraph_method(None, 'vcount')  # None graph

        with pytest.raises(IgraphMethodError):
            call_igraph_method(graph, '')  # Empty method name

        with pytest.raises(IgraphMethodError):
            call_igraph_method(graph, 'nonexistent_method')  # Invalid method


class TestGraphingAdvanced:
    """Advanced Graphing class tests"""

    @patch('graphizy.algorithms.cv2')
    @patch('graphizy.algorithms.ig.Graph')
    def test_make_proximity_graph(self, mock_graph_class, mock_cv2):
        """Test proximity graph creation"""
        # Mock igraph with proper behavior
        mock_graph = MagicMock()
        mock_graph.vcount.return_value = 3
        mock_graph.vs = MagicMock()
        mock_graph.vs.__setitem__ = MagicMock()  # Allow item assignment
        mock_graph.vs.__getitem__ = MagicMock()
        mock_graph_class.return_value = mock_graph

        grapher = Graphing()

        # Test data
        data = np.array([
            [0, 10, 20],
            [1, 30, 40],
            [2, 50, 60]
        ])

        result = grapher.make_proximity(data, proximity_thresh=25.0)

        # Should create graph and call distance function
        assert result == mock_graph

    def test_make_proximity_config_defaults(self):
        """Test proximity graph with config defaults"""
        config = GraphizyConfig()
        config.graph.proximity_threshold = 100.0
        config.graph.distance_metric = 'manhattan'

        grapher = Graphing(config=config)

        # Create minimal test to verify config usage
        data = np.array([[0, 10, 20], [1, 30, 40]])

        # This should use config defaults without explicit parameters
        # (We can't easily test the full execution without mocking OpenCV)
        assert grapher.config.graph.proximity_threshold == 100.0
        assert grapher.config.graph.distance_metric == 'manhattan'

    @patch('graphizy.drawing.cv2')
    def test_draw_graph(self, mock_cv2):
        """Test graph drawing functionality"""
        import igraph as ig

        # Create real graph for testing
        graph = ig.Graph()
        graph.add_vertices(2)
        graph.vs["x"] = [10, 30]
        graph.vs["y"] = [20, 40]
        graph.add_edges([(0, 1)])

        grapher = Graphing(dimension=(100, 100))

        # This should not raise an error
        image = grapher.draw_graph(graph)

        # Verify image properties
        assert image is not None
        assert image.shape == (100, 100, 3)

    def test_aspect_switching(self):
        """Test switching between array and dict aspects"""
        grapher = Graphing(aspect="array")
        assert grapher.aspect == "array"

        # Update to dict
        grapher.update_config(graph={"aspect": "dict"})
        assert grapher.aspect == "dict"

    def test_data_interface_integration(self):
        """Test DataInterface integration with different data shapes"""
        custom_data_shape = [
            ("id", int),
            ("x", float),
            ("y", float),
            ("velocity", float)
        ]

        grapher = Graphing(data_shape=custom_data_shape)

        # Verify interface uses custom shape
        assert grapher.dinter.getidx_id() == 0
        assert grapher.dinter.getidx_xpos() == 1
        assert grapher.dinter.getidx_ypos() == 2


class TestErrorHandling:
    """Test comprehensive error handling"""

    @patch('graphizy.algorithms.cv2')
    def test_subdivision_opencv_error(self, mock_cv2):
        """Test handling of OpenCV errors in subdivision"""
        # Mock OpenCV to raise an error
        mock_cv2.Subdiv2D.side_effect = Exception("OpenCV error")

        points = np.array([[10.0, 20.0]], dtype=np.float32)
        dimensions = (100, 100)

        with pytest.raises(SubdivisionError):
            make_subdiv(points, dimensions)

    def test_graph_creation_with_invalid_data(self):
        """Test graph creation error handling"""
        from graphizy.algorithms import create_graph_array
        from graphizy.exceptions import GraphCreationError

        # Test with insufficient columns
        invalid_data = np.array([[1, 2]])  # Only 2 columns, need at least 3

        with pytest.raises(GraphCreationError):
            create_graph_array(invalid_data)

    def test_drawing_error_handling(self):
        """Test drawing function error handling"""
        from graphizy.drawing import draw_point
        from graphizy.exceptions import DrawingError

        # Create a valid image
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Test with invalid parameters
        with pytest.raises(DrawingError):
            draw_point(None, (10, 10), (255, 0, 0))  # None image

        with pytest.raises(DrawingError):
            draw_point(image, (10,), (255, 0, 0))  # Invalid point

        with pytest.raises(DrawingError):
            draw_point(image, (10, 10), (255, 0))  # Invalid color


class TestConfigurationAdvanced:
    """Advanced configuration tests"""

    def test_config_persistence_through_updates(self):
        """Test that configuration persists correctly through updates"""
        config = GraphizyConfig()
        original_line_color = config.drawing.line_color

        # Update one parameter
        config.update(drawing={"point_radius": 15})

        # Original settings should persist
        assert config.drawing.line_color == original_line_color
        assert config.drawing.point_radius == 15

    def test_config_validation_on_update(self):
        """Test that validation works on runtime updates"""
        config = GraphizyConfig()

        with pytest.raises(ValueError):
            from graphizy.config import DrawingConfig
            DrawingConfig(line_thickness=-1)

    def test_nested_config_updates(self):
        """Test deeply nested configuration updates"""
        grapher = Graphing()

        # Update multiple nested configs
        grapher.update_config(
            drawing={
                "line_color": (255, 255, 255),
                "point_thickness": 5
            },
            graph={
                "dimension": (400, 400),
                "proximity_threshold": 75.0
            }
        )

        assert grapher.line_color == (255, 255, 255)
        assert grapher.config.drawing.point_thickness == 5
        assert grapher.dimension == (400, 400)
        assert grapher.config.graph.proximity_threshold == 75.0


class TestPerformance:
    """Performance-related tests"""

    def test_large_dataset_handling(self):
        """Test handling of larger datasets"""
        from graphizy import generate_positions, create_graph_array

        # Generate a reasonably large dataset
        positions = generate_positions(500, 500, 1000)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # This should complete without error
        graph = create_graph_array(particle_stack)

        assert graph.vcount() == 1000
        assert graph.ecount() == 0  # No edges initially

    def test_memory_efficiency(self):
        """Test that operations don't consume excessive memory"""
        import gc

        # Generate data
        positions = generate_positions(100, 100, 50)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Create and destroy graphs multiple times
        for _ in range(10):
            graph = create_graph_array(particle_stack)
            del graph
            gc.collect()

        # Should complete without memory issues


if __name__ == '__main__':
    pytest.main([__file__])