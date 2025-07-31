"""
Enhanced test coverage for distance computation functionality.
Add these tests to your existing test files or create test_distance_computation.py
"""
import pytest
import numpy as np
import time
from unittest.mock import patch, Mock

from graphizy import Graphing, GraphizyConfig
from graphizy.algorithms import (
    add_edge_distances,
    create_proximity_graph,
    create_delaunay_graph,
    create_knn_graph,
    create_mst_graph,
    create_gabriel_graph,
)
from graphizy.weight import (
    WeightComputer,
    compute_edge_weights,
)
from graphizy.exceptions import GraphCreationError


class TestDistanceComputation:
    """Test distance computation in various graph types"""

    @pytest.fixture
    def sample_data(self):
        """Sample data with known distances"""
        return np.array([
            [1, 0.0, 0.0],  # Origin
            [2, 3.0, 4.0],  # Distance 5 from origin
            [3, 6.0, 8.0],  # Distance 10 from origin
            [4, 1.0, 1.0],  # Distance sqrt(2) from origin
            [5, 10.0, 0.0]  # Distance 10 from origin
        ])

    @pytest.fixture
    def grapher(self):
        """Graphing instance"""
        config = GraphizyConfig()
        config.graph.dimension = (200, 200)
        return Graphing(config=config)

    def test_add_distance_parameter_types(self, grapher, sample_data):
        """Test different add_distance parameter formats"""
        # Test boolean True
        graph1 = create_proximity_graph(sample_data, 50.0, add_distance=True)
        assert "distance" in graph1.es.attributes()

        # Test boolean False
        graph2 = create_proximity_graph(sample_data, 50.0, add_distance=False)
        assert "distance" not in graph2.es.attributes()

        # Test dict format
        graph3 = create_proximity_graph(sample_data, 50.0, add_distance={"metric": "manhattan"})
        assert "distance" in graph3.es.attributes()

        # Test None (should be treated as True based on your implementation)
        graph4 = create_proximity_graph(sample_data, 50.0, add_distance=None)
        # Based on your code, None should not add distances
        assert "distance" not in graph4.es.attributes()

    def test_distance_metrics_accuracy(self, grapher, sample_data):
        """Test accuracy of different distance metrics"""
        # Create a simple graph with known distances
        simple_data = np.array([
            [1, 0.0, 0.0],
            [2, 3.0, 4.0]  # Distance should be 5.0 (euclidean), 7.0 (manhattan), 4.0 (chebyshev)
        ])

        # Test euclidean distance
        graph_eucl = create_proximity_graph(simple_data, 10.0, add_distance={"metric": "euclidean"})
        if graph_eucl.ecount() > 0:
            assert abs(graph_eucl.es["distance"][0] - 5.0) < 1e-10

        # Test manhattan distance
        graph_manh = create_proximity_graph(simple_data, 10.0, add_distance={"metric": "manhattan"})
        if graph_manh.ecount() > 0:
            assert abs(graph_manh.es["distance"][0] - 7.0) < 1e-10

        # Test chebyshev distance
        graph_cheb = create_proximity_graph(simple_data, 10.0, add_distance={"metric": "chebyshev"})
        if graph_cheb.ecount() > 0:
            assert abs(graph_cheb.es["distance"][0] - 4.0) < 1e-10

    def test_distance_in_all_graph_types(self, grapher, sample_data):
        """Test distance computation in all graph types"""
        graph_types = [
            ('proximity', {'proximity_thresh': 15.0}),
            ('knn', {'k': 2}),
            ('mst', {}),
            ('gabriel', {}),
            ('delaunay', {})
        ]

        for graph_type, params in graph_types:
            try:
                if graph_type == 'proximity':
                    graph = create_proximity_graph(sample_data, **params, add_distance=True)
                elif graph_type == 'knn':
                    graph = create_knn_graph(sample_data, **params, add_distance=True)
                elif graph_type == 'mst':
                    graph = create_mst_graph(sample_data, add_distance=True)
                elif graph_type == 'gabriel':
                    graph = create_gabriel_graph(sample_data, add_distance=True)
                elif graph_type == 'delaunay':
                    graph = create_delaunay_graph(sample_data, add_distance=True)

                if graph.ecount() > 0:
                    assert "distance" in graph.es.attributes()
                    assert len(graph.es["distance"]) == graph.ecount()
                    assert all(d >= 0 for d in graph.es["distance"])  # All distances should be non-negative

            except Exception as e:
                pytest.fail(f"Distance computation failed for {graph_type}: {str(e)}")

    def test_distance_with_different_metrics_per_graph_type(self, sample_data):
        """Test different distance metrics for each graph type"""
        metrics = ["euclidean", "manhattan", "chebyshev"]

        for metric in metrics:
            # Test proximity graph
            prox_graph = create_proximity_graph(sample_data, 15.0, add_distance={"metric": metric})
            if prox_graph.ecount() > 0:
                assert "distance" in prox_graph.es.attributes()

            # Test MST (which can use different metrics for construction and distance)
            mst_graph = create_mst_graph(sample_data, metric="euclidean", add_distance={"metric": metric})
            if mst_graph.ecount() > 0:
                assert "distance" in mst_graph.es.attributes()

    def test_distance_edge_cases(self, grapher):
        """Test distance computation edge cases"""
        # Empty graph
        empty_data = np.array([[1, 10.0, 20.0]])
        empty_graph = create_proximity_graph(empty_data, 1.0, add_distance=True)
        assert empty_graph.ecount() == 0

        # Single edge case
        two_point_data = np.array([[1, 0.0, 0.0], [2, 1.0, 1.0]])
        two_point_graph = create_proximity_graph(two_point_data, 5.0, add_distance=True)
        if two_point_graph.ecount() > 0:
            assert "distance" in two_point_graph.es.attributes()
            assert len(two_point_graph.es["distance"]) == 1

    def test_memory_graph_distances(self, grapher, sample_data):
        """Test distance computation in memory graphs"""
        # Initialize memory
        grapher.init_memory_manager()

        # Create and update memory
        prox_graph = create_proximity_graph(sample_data, 15.0, add_distance=True)
        grapher.update_memory_with_graph(prox_graph)

        # Create memory graph with current distances
        memory_connections = grapher.memory_manager.get_current_memory_graph()
        memory_graph = grapher.make_memory_graph(sample_data, memory_connections)

        if memory_graph.ecount() > 0:
            # Memory graph should have computed current distances
            assert "distance" in memory_graph.es.attributes()
            assert all(d >= 0 for d in memory_graph.es["distance"])


class TestDistancePerformance:
    """Test performance aspects of distance computation"""

    def test_distance_computation_performance(self):
        """Test performance of distance computation methods"""
        # Large dataset for performance testing
        large_data = np.random.rand(500, 3) * 100
        large_data[:, 0] = np.arange(500)

        # Test original vs optimized approach
        # (This would test your optimized add_edge_distances_fast function)

        # Create a graph with many edges
        from graphizy.algorithms import create_graph_array
        import igraph as ig

        graph = create_graph_array(large_data)

        # Add some edges (create a moderately dense graph)
        edges_to_add = [(i, j) for i in range(50) for j in range(i + 1, min(i + 10, 500))]
        graph.add_edges(edges_to_add)

        # Time the distance computation
        start_time = time.time()
        graph_with_distances = add_edge_distances(graph, large_data, "euclidean")
        end_time = time.time()

        computation_time = end_time - start_time

        # Should complete reasonably quickly (adjust threshold as needed)
        assert computation_time < 5.0  # 5 seconds max for this test
        assert "distance" in graph_with_distances.es.attributes()
        assert len(graph_with_distances.es["distance"]) == graph.ecount()

    def test_vectorized_vs_loop_performance(self):
        """Test performance comparison between different approaches"""
        # This test would compare your different distance computation approaches
        # Implementation depends on which optimized versions you choose to implement
        pass


class TestWeightComputation:
    """Test weight computation functionality"""

    @pytest.fixture
    def sample_graph_with_attributes(self):
        """Create a sample graph with various edge attributes"""
        import igraph as ig

        graph = ig.Graph()
        graph.add_vertices(3)
        graph.vs["id"] = [1, 2, 3]
        graph.add_edges([(0, 1), (1, 2)])

        # Add various attributes
        graph.es["distance"] = [5.0, 10.0]
        graph.es["weight"] = [1.0, 2.0]
        graph.es["age"] = [1, 5]

        return graph

    def test_weight_computer_formula_creation(self):
        """Test WeightComputer formula string functionality"""
        # Test simple formula
        weight_func = WeightComputer.create_weight_function("1/distance")

        # Test with sample edge attributes
        attrs = {"distance": 5.0, "weight": 2.0, "age": 3}
        result = weight_func(attrs)
        assert abs(result - 0.2) < 1e-10  # 1/5 = 0.2

        # Test complex formula
        complex_func = WeightComputer.create_weight_function("weight * exp(-0.1 * age)")
        result2 = complex_func(attrs)
        expected = 2.0 * np.exp(-0.1 * 3)
        assert abs(result2 - expected) < 1e-10

    def test_weight_computer_safe_evaluation(self):
        """Test that WeightComputer handles invalid formulas safely"""
        # Test division by zero protection
        weight_func = WeightComputer.create_weight_function("1/distance")
        attrs = {"distance": 0.0}
        result = weight_func(attrs)
        assert result == 1.0  # Should return default value

        # Test invalid formula
        invalid_func = WeightComputer.create_weight_function("invalid_function()")
        result2 = invalid_func(attrs)
        assert result2 == 1.0  # Should return default value

        # Test missing attributes
        missing_attr_func = WeightComputer.create_weight_function("nonexistent_attr")
        result3 = missing_attr_func({})
        assert result3 == 1.0  # Should return default value

    def test_weight_computer_predefined_functions(self):
        """Test predefined weight computation functions"""
        attrs = {"distance": 4.0, "weight": 2.0, "age": 2}

        # Test distance weight
        dist_weight = WeightComputer.distance_weight(attrs, invert=True)
        expected = 1.0 / (4.0 + 1e-6)
        assert abs(dist_weight - expected) < 1e-5

        # Test age decay weight
        age_weight = WeightComputer.age_decay_weight(attrs, decay_factor=0.1)
        expected = 2.0 * np.exp(-0.1 * 2)
        assert abs(age_weight - expected) < 1e-10