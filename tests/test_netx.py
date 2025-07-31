"""
Tests for NetworkX bridge functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import graphizy components
from graphizy import Graphing, GraphizyConfig
from graphizy.exceptions import DependencyError, GraphCreationError

# Test if NetworkX is available
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


@pytest.fixture
def sample_data():
    """Sample data for testing"""
    return np.array([
        [1, 10.0, 20.0],
        [2, 30.0, 40.0],
        [3, 50.0, 60.0],
        [4, 70.0, 80.0],
        [5, 90.0, 100.0]
    ])


@pytest.fixture
def grapher():
    """Graphing instance for testing"""
    config = GraphizyConfig()
    config.graph.dimension = (200, 200)
    return Graphing(config=config)


@pytest.fixture
def sample_igraph(grapher, sample_data):
    """Sample igraph with edges and attributes"""
    # Create a proximity graph with known structure
    graph = grapher.make_graph(graph_type="proximity", data_points=sample_data, proximity_thresh=50.0)

    # Add some test attributes
    if graph.ecount() > 0:
        graph.es["weight"] = [1.0] * graph.ecount()
        graph.es["test_attr"] = ["test"] * graph.ecount()

    return graph


class TestNetworkXAvailability:
    """Test NetworkX availability and dependency handling"""

    def test_networkx_available(self):
        """Test if NetworkX is properly detected"""
        if NETWORKX_AVAILABLE:
            assert nx is not None
            assert hasattr(nx, 'Graph')
        else:
            pytest.skip("NetworkX not available for testing")

    @patch('graphizy.networkx_bridge.NETWORKX_AVAILABLE', False)
    def test_networkx_unavailable_error(self):
        """Test error when NetworkX is not available"""
        # This would test the dependency error handling
        # In real implementation, this would import the bridge module
        pass


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
class TestNetworkXConversion:
    """Test conversion between igraph and NetworkX"""

    def test_to_networkx_basic(self, sample_igraph):
        """Test basic igraph to NetworkX conversion"""
        from graphizy.networkx_bridge import to_networkx

        nx_graph = to_networkx(sample_igraph)

        # Check basic structure
        assert isinstance(nx_graph, (nx.Graph, nx.DiGraph))
        assert nx_graph.number_of_nodes() == sample_igraph.vcount()
        assert nx_graph.number_of_edges() == sample_igraph.ecount()

        # Check node attributes
        for node in nx_graph.nodes():
            assert node in [v["id"] for v in sample_igraph.vs]

    def test_to_networkx_with_attributes(self, sample_igraph):
        """Test conversion preserving attributes"""
        from graphizy.networkx_bridge import to_networkx

        nx_graph = to_networkx(sample_igraph, copy_vertex_attrs=True, copy_edge_attrs=True)

        # Check vertex attributes
        for node, data in nx_graph.nodes(data=True):
            assert "x" in data
            assert "y" in data
            assert "id" in data

        # Check edge attributes if edges exist
        if nx_graph.number_of_edges() > 0:
            for u, v, data in nx_graph.edges(data=True):
                if "weight" in sample_igraph.es.attributes():
                    assert "weight" in data
                if "test_attr" in sample_igraph.es.attributes():
                    assert "test_attr" in data

    def test_to_networkx_without_attributes(self, sample_igraph):
        """Test conversion without copying attributes"""
        from graphizy.networkx_bridge import to_networkx

        nx_graph = to_networkx(sample_igraph, copy_vertex_attrs=False, copy_edge_attrs=False)

        # Should have basic structure but minimal attributes
        assert nx_graph.number_of_nodes() == sample_igraph.vcount()
        assert nx_graph.number_of_edges() == sample_igraph.ecount()

        # Check that only essential attributes are present
        for node, data in nx_graph.nodes(data=True):
            # Should have minimal attributes
            assert len(data) <= 1  # May have default attributes

    def test_to_networkx_directed(self, grapher, sample_data):
        """Test directed graph conversion"""
        from graphizy.networkx_bridge import to_networkx

        # Create a directed graph (knn creates directed edges)
        igraph_directed = grapher.make_graph(graph_type="knn", data_points=sample_data, k=2)

        # Convert preserving direction
        nx_directed = to_networkx(igraph_directed, directed=True)
        assert isinstance(nx_directed, nx.DiGraph)

        # Convert forcing undirected
        nx_undirected = to_networkx(igraph_directed, directed=False)
        assert isinstance(nx_undirected, nx.Graph)

    def test_to_networkx_empty_graph(self, grapher):
        """Test conversion of empty graph"""
        from graphizy.networkx_bridge import to_networkx

        # Create empty graph
        empty_data = np.array([[1, 10.0, 20.0]])  # Single point, no edges
        empty_graph = grapher.make_graph(graph_type="proximity", data_points=empty_data, proximity_thresh=1.0)  # Very small threshold

        nx_graph = to_networkx(empty_graph)
        assert nx_graph.number_of_nodes() == 1
        assert nx_graph.number_of_edges() == 0

    def test_from_networkx_basic(self):
        """Test NetworkX to igraph conversion"""
        from graphizy.networkx_bridge import from_networkx

        # Create simple NetworkX graph
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from([1, 2, 3])
        nx_graph.add_edges_from([(1, 2), (2, 3)])

        igraph_result = from_networkx(nx_graph)

        # Check basic structure
        assert igraph_result.vcount() == 3
        assert igraph_result.ecount() == 2
        assert not igraph_result.is_directed()

    def test_from_networkx_with_attributes(self):
        """Test NetworkX to igraph with attributes"""
        from graphizy.networkx_bridge import from_networkx

        # Create NetworkX graph with attributes
        nx_graph = nx.Graph()
        nx_graph.add_node(1, x=10, y=20, name="A")
        nx_graph.add_node(2, x=30, y=40, name="B")
        nx_graph.add_edge(1, 2, weight=5.0, color="red")

        igraph_result = from_networkx(nx_graph, copy_vertex_attrs=True, copy_edge_attrs=True)

        # Check vertex attributes
        assert "id" in igraph_result.vs.attributes()
        assert "x" in igraph_result.vs.attributes()
        assert "y" in igraph_result.vs.attributes()
        assert "name" in igraph_result.vs.attributes()

        # Check edge attributes
        assert "weight" in igraph_result.es.attributes()
        assert "color" in igraph_result.es.attributes()

    def test_round_trip_conversion(self, sample_igraph):
        """Test igraph -> NetworkX -> igraph round trip"""
        from graphizy.networkx_bridge import to_networkx, from_networkx

        # Convert to NetworkX and back
        nx_graph = to_networkx(sample_igraph)
        igraph_result = from_networkx(nx_graph)

        # Check structure is preserved
        assert igraph_result.vcount() == sample_igraph.vcount()
        assert igraph_result.ecount() == sample_igraph.ecount()
        assert igraph_result.is_directed() == sample_igraph.is_directed()


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
class TestNetworkXAnalyzer:
    """Test NetworkX analyzer functionality"""

    def test_analyzer_initialization(self, grapher):
        """Test NetworkX analyzer initialization"""
        from graphizy.networkx_bridge import NetworkXAnalyzer

        analyzer = NetworkXAnalyzer(grapher)
        assert analyzer.grapher is grapher
        assert hasattr(analyzer, '_nx_cache')

    def test_get_networkx_from_graph_type(self, grapher, sample_data):
        """Test getting NetworkX graph from graph type"""
        from graphizy.networkx_bridge import NetworkXAnalyzer

        # Create graphs
        grapher.set_graph_type(['proximity', 'delaunay'])
        graphs = grapher.update_graphs(sample_data, proximity_thresh=50.0)

        analyzer = NetworkXAnalyzer(grapher)

        # Get NetworkX version
        nx_prox = analyzer.get_networkx('proximity')
        assert isinstance(nx_prox, (nx.Graph, nx.DiGraph))

        # Test caching
        nx_prox2 = analyzer.get_networkx('proximity', use_cache=True)
        assert nx_prox is nx_prox2  # Same object due to caching

    def test_get_networkx_manual_graph(self, grapher, sample_data):
        """Test getting NetworkX graph from manual igraph"""
        from graphizy.networkx_bridge import NetworkXAnalyzer

        manual_graph = grapher.make_graph(graph_type="proximity", data_points=sample_data, proximity_thresh=50.0)
        analyzer = NetworkXAnalyzer(grapher)

        nx_graph = analyzer.get_networkx(igraph_graph=manual_graph)
        assert isinstance(nx_graph, (nx.Graph, nx.DiGraph))
        assert nx_graph.number_of_nodes() == manual_graph.vcount()

    def test_analyzer_comprehensive_analysis(self, grapher, sample_data):
        """Test comprehensive NetworkX analysis"""
        from graphizy.networkx_bridge import NetworkXAnalyzer

        # Create a well-connected graph for analysis
        graph = grapher.make_graph(graph_type="proximity", data_points=sample_data, proximity_thresh=100.0)  # Higher threshold for connections
        analyzer = NetworkXAnalyzer(grapher)

        analysis = analyzer.analyze(igraph_graph=graph)

        # Check basic metrics
        assert 'nodes' in analysis
        assert 'edges' in analysis
        assert 'density' in analysis
        assert analysis['nodes'] == len(sample_data)

        # Check connectivity metrics
        if graph.ecount() > 0:
            assert 'connected' in analysis or 'strongly_connected' in analysis

    def test_analyzer_with_small_graph(self, grapher):
        """Test analyzer with small graphs (edge cases)"""
        from graphizy.networkx_bridge import NetworkXAnalyzer

        # Very small graph
        small_data = np.array([[1, 10.0, 20.0], [2, 30.0, 40.0]])
        small_graph = grapher.make_graph(graph_type="proximity", data_points=small_data, proximity_thresh=50.0)

        analyzer = NetworkXAnalyzer(grapher)
        analysis = analyzer.analyze(igraph_graph=small_graph)

        assert analysis['nodes'] == 2
        # Should handle small graphs gracefully

    def test_analyzer_cache_management(self, grapher, sample_data):
        """Test cache management in analyzer"""
        from graphizy.networkx_bridge import NetworkXAnalyzer

        grapher.set_graph_type('proximity')
        grapher.update_graphs(sample_data, proximity_thresh=50.0)

        analyzer = NetworkXAnalyzer(grapher)

        # Initial cache should be empty
        assert len(analyzer._nx_cache) == 0

        # Get graph (should cache)
        nx_graph1 = analyzer.get_networkx('proximity')
        assert len(analyzer._nx_cache) == 1

        # Clear cache
        analyzer.clear_cache()
        assert len(analyzer._nx_cache) == 0

        # Get again (should re-cache)
        nx_graph2 = analyzer.get_networkx('proximity')
        assert len(analyzer._nx_cache) == 1
        assert nx_graph1 is not nx_graph2  # Different objects after cache clear


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
class TestMainClassIntegration:
    """Test integration with main Graphing class"""

    def test_get_networkx_analyzer_method(self, grapher):
        """Test get_networkx_analyzer method in main class"""
        analyzer = grapher.get_networkx_analyzer()
        from graphizy.networkx_bridge import NetworkXAnalyzer
        assert isinstance(analyzer, NetworkXAnalyzer)
        assert analyzer.grapher is grapher

    def test_to_networkx_method(self, grapher, sample_data):
        """Test to_networkx method in main class"""
        # Create a graph
        prox_graph = grapher.make_graph(graph_type="proximity", data_points=sample_data, proximity_thresh=50.0)

        # Convert using main class method
        nx_graph = grapher.to_networkx(igraph_graph=prox_graph)
        assert isinstance(nx_graph, (nx.Graph, nx.DiGraph))
        assert nx_graph.number_of_nodes() == prox_graph.vcount()

    def test_to_networkx_with_graph_type(self, grapher, sample_data):
        """Test to_networkx with graph type"""
        # Set up current graphs
        grapher.set_graph_type('proximity')
        grapher.update_graphs(sample_data, proximity_thresh=50.0)

        # Convert using graph type
        nx_graph = grapher.to_networkx(graph_type='proximity')
        assert isinstance(nx_graph, (nx.Graph, nx.DiGraph))


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_invalid_graph_conversion(self):
        """Test conversion with invalid inputs"""
        if not NETWORKX_AVAILABLE:
            pytest.skip("NetworkX not available")

        from graphizy.networkx_bridge import to_networkx

        with pytest.raises(GraphCreationError):
            to_networkx(None)

    def test_analyzer_with_missing_graph_type(self, grapher):
        """Test analyzer with non-existent graph type"""
        if not NETWORKX_AVAILABLE:
            pytest.skip("NetworkX not available")

        from graphizy.networkx_bridge import NetworkXAnalyzer

        analyzer = NetworkXAnalyzer(grapher)

        with pytest.raises(ValueError):
            analyzer.get_networkx('nonexistent_type')

    def test_analyzer_with_none_graph(self, grapher, sample_data):
        """Test analyzer when graph is None"""
        if not NETWORKX_AVAILABLE:
            pytest.skip("NetworkX not available")

        from graphizy.networkx_bridge import NetworkXAnalyzer

        # Set up a graph type but make it None
        grapher.set_graph_type('proximity')
        grapher.current_graphs = {'proximity': None}

        analyzer = NetworkXAnalyzer(grapher)

        with pytest.raises(ValueError):
            analyzer.get_networkx('proximity')


@pytest.mark.skipif(not NETWORKX_AVAILABLE, reason="NetworkX not available")
class TestPerformanceAspects:
    """Test performance-related aspects"""

    def test_large_graph_conversion(self, grapher):
        """Test conversion with larger graphs"""
        # Create larger dataset
        large_data = np.random.rand(100, 3) * 100
        large_data[:, 0] = np.arange(100)  # Set IDs

        large_graph = grapher.make_graph(graph_type="proximity", data_points=large_data, proximity_thresh=20.0)

        from graphizy.networkx_bridge import to_networkx

        # Should handle larger graphs efficiently
        nx_graph = to_networkx(large_graph)
        assert nx_graph.number_of_nodes() == large_graph.vcount()
        assert nx_graph.number_of_edges() == large_graph.ecount()

    def test_multiple_conversions_performance(self, grapher, sample_data):
        """Test multiple conversions for performance"""
        from graphizy.networkx_bridge import NetworkXAnalyzer

        grapher.set_graph_type(['proximity', 'delaunay', 'mst'])
        graphs = grapher.update_graphs(sample_data, proximity_thresh=50.0)

        analyzer = NetworkXAnalyzer(grapher)

        # Convert multiple graph types
        for graph_type in ['proximity', 'mst']:  # Skip delaunay if it has issues
            if graphs.get(graph_type) is not None:
                nx_graph = analyzer.get_networkx(graph_type)
                assert isinstance(nx_graph, (nx.Graph, nx.DiGraph))
