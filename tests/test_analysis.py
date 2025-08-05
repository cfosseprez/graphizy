# tests/test_analysis.py

import pytest
import numpy as np
from unittest.mock import patch

from graphizy import Graphing, GraphizyConfig
from graphizy.analysis import GraphAnalysisResult
from graphizy.exceptions import GraphCreationError


@pytest.fixture
def setup_grapher_and_graph():
    """A fixture to create a Graphing instance and a sample graph for testing."""
    config = GraphizyConfig(dimension=(200, 200))
    grapher = Graphing(config=config)

    # Create a disconnected graph to test component-based logic
    # Component 1: 3 nodes (triangle)
    # Component 2: 2 nodes (line)
    # Component 3: 1 isolated node
    data = np.array([
        [0, 10, 10], [1, 20, 10], [2, 10, 20],
        [3, 100, 100], [4, 110, 100],
        [5, 150, 150]
    ])
    graph = grapher.make_graph("proximity", data, proximity_thresh=15.0)
    return grapher, graph


class TestGraphAnalysisResult:
    """Tests for the GraphAnalysisResult class."""

    def test_initialization(self, setup_grapher_and_graph):
        """Test that the result object is created correctly and lazily."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        assert isinstance(results, GraphAnalysisResult)
        assert results._graph == graph
        assert results._grapher == grapher
        assert not results._metric_cache  # Cache should be empty initially

    def test_lazy_loading_and_caching(self, setup_grapher_and_graph):
        """Test that metrics are computed on first access and then cached."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        with patch.object(grapher, 'call_method_safe', wraps=grapher.call_method_safe) as mock_call:
            # First access: should call the method
            diam = results.diameter
            assert diam is not None
            mock_call.assert_called_once_with(graph, 'diameter', component_mode="largest", default_value=None)

            # Second access: should NOT call the method again
            mock_call.reset_mock()
            diam_cached = results.diameter
            assert diam_cached == diam
            mock_call.assert_not_called()

            # Verify the cache has the value with the correct key format
            kwargs = {'component_mode': 'largest', 'default_value': None}
            expected_key = f"diameter_{sorted(kwargs.items())}"
            assert expected_key in results._metric_cache

    def test_basic_properties(self, setup_grapher_and_graph):
        """Test the correctness of the main properties via object access."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        assert results.vertex_count == 6
        assert results.edge_count == 4
        assert not results.is_connected
        assert results.num_components == 3
        assert results.diameter == 1.0  # Diameter of the largest component (the triangle)

    def test_get_metric_on_demand(self, setup_grapher_and_graph):
        """Test computing a metric that is not a default property."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        pagerank = results.get_metric('pagerank', return_format='list')
        assert isinstance(pagerank, list)
        assert len(pagerank) == results.vertex_count

        # Check that it's now cached with the correct key format
        kwargs = {'return_format': 'list'}
        expected_key = f"pagerank_{sorted(kwargs.items())}"
        assert expected_key in results._metric_cache

    def test_get_top_n_by(self, setup_grapher_and_graph):
        """Test the helper for getting top N nodes by a metric."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        top_3_degree = results.get_top_n_by('degree', n=3)
        assert len(top_3_degree) == 3
        degrees = [d for _, d in top_3_degree]
        assert degrees == [2, 2, 2]

        with pytest.raises(TypeError, match="did not return a dictionary"):
            results.get_top_n_by('is_connected')

    def test_get_metric_stats(self, setup_grapher_and_graph):
        """Test the helper for getting descriptive statistics."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        degree_stats = results.get_metric_stats('degree')
        assert isinstance(degree_stats, dict)
        assert 'mean' in degree_stats
        assert degree_stats['mean'] == pytest.approx(8 / 6)
        assert degree_stats['min'] == 0
        assert degree_stats['max'] == 2
        assert degree_stats['median'] == 1.5

    def test_summary_and_str(self, setup_grapher_and_graph):
        """Test the string representation and summary."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        summary = results.summary()
        assert "Graph Analysis Summary" in summary
        assert "Vertices: 6" in summary
        assert "Connected: False" in summary
        assert "Components: 3" in summary
        assert str(results) == summary

    def test_error_on_none_graph(self):
        """Test that get_graph_info raises an error for a None graph."""
        grapher = Graphing()
        with pytest.raises(GraphCreationError, match="Cannot get info for a None graph"):
            grapher.get_graph_info(None)

    # --- New Tests for Hybrid Dictionary/Object Behavior ---

    def test_dictionary_access_for_properties(self, setup_grapher_and_graph):
        """Test dictionary-style access for standard properties."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        # Test that dictionary access works and gives the same result as property access
        assert results['vertex_count'] == results.vertex_count
        assert results['edge_count'] == results.edge_count
        assert results['density'] == results.density
        assert results['is_connected'] == results.is_connected
        assert results['diameter'] == results.diameter

    def test_dictionary_access_for_on_demand_metrics(self, setup_grapher_and_graph):
        """Test dictionary-style access triggers on-demand computation."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        # Accessing a non-property metric via __getitem__ should trigger get_metric
        with patch.object(results, 'get_metric', wraps=results.get_metric) as mock_get_metric:
            # 'pagerank' is not a default property
            pagerank = results['pagerank']
            assert pagerank is not None
            mock_get_metric.assert_called_once_with('pagerank')

            # Accessing again should hit the cache and not call get_metric again
            mock_get_metric.reset_mock()
            pagerank_cached = results['pagerank']
            assert pagerank_cached is not None
            mock_get_metric.assert_not_called()

    def test_dictionary_access_key_error(self, setup_grapher_and_graph):
        """Test that accessing a non-existent key raises a KeyError."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        with pytest.raises(KeyError, match="Metric or property 'non_existent_metric' not found"):
            _ = results['non_existent_metric']

    def test_contains_operator(self, setup_grapher_and_graph):
        """Test the 'in' operator for checking metric availability."""
        grapher, graph = setup_grapher_and_graph
        results = grapher.get_graph_info(graph)

        # Should be True for standard properties
        assert 'density' in results
        assert 'vertex_count' in results

        # Should be False for non-existent properties
        assert 'non_existent_metric' not in results