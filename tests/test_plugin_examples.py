"""
Tests for the plugin_examples.py file to ensure the examples are correct,
functional, and serve as a good template for users.
"""
import pytest
import numpy as np
from graphizy import Graphing, format_positions
# Import the example module to ensure plugins are registered
import graphizy.plugins_examples as plugin_examples

@pytest.fixture
def grapher():
    """Provides a default Graphing instance."""
    return Graphing(dimension=(400, 400))

@pytest.fixture
def sample_data():
    """Provides sample data for testing plugins."""
    positions = np.array([
        [100, 100], [120, 100], [200, 200], [220, 200],
        [150, 50], [150, 250]
    ])
    return format_positions(positions)

class TestExamplePlugins:
    """Tests the plugins defined in plugins_examples.py."""

    def test_plugins_are_registered(self, grapher):
        """Verify that the example plugins are available."""
        available_plugins = grapher.list_graph_types()
        assert "radial" in available_plugins
        assert "connect_to_center" in available_plugins

        info = Graphing.get_plugin_info("radial")
        assert info['info']['category'] == "custom_example"


    def test_radial_plugin_ring_connections(self, grapher, sample_data):
        """Test the radial plugin with ring_connections=True."""
        graph = grapher.make_graph(
            "radial",
            sample_data,
            center_x=150,
            center_y=150,
            ring_connections=True,
            ring_tolerance=25.0 # Increased tolerance slightly for robustness
        )
        assert graph.vcount() == 6
        # Expecting edges between (100,100)-(120,100) and (200,200)-(220,200)
        assert graph.ecount() >= 2

    def test_radial_plugin_spoke_connections(self, grapher, sample_data):
        """Test the radial plugin with ring_connections=False (spokes)."""
        graph = grapher.make_graph(
            "radial",
            sample_data,
            center_x=150,
            center_y=150,
            ring_connections=False
        )
        assert graph.vcount() == 6
        # Expecting an edge between (150,50) and (150,250) as they are on a vertical spoke
        assert graph.ecount() >= 1

    def test_connect_to_center_plugin(self, grapher, sample_data):
        """Test the decorator-based connect_to_center plugin."""
        graph = grapher.make_graph("connect_to_center", sample_data)
        # Should have original 6 points + 1 center point
        assert graph.vcount() == 7
        # Should have 6 edges, one from each original point to the center
        assert graph.ecount() == 6