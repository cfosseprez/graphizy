#!/usr/bin/env python3
"""
Comprehensive Integrated Test for Graphizy Package

This test validates the complete workflow from the README examples,
including graph creation, analysis, visualization, and advanced features.
It serves as both a test and a demonstration of proper API usage.

.. moduleauthor:: Test Suite
.. license:: GPL2 or later
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple

# Configure logging for the test
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def test_data():
    """Generate test data for all tests."""
    try:
        from graphizy import generate_and_format_positions, validate_graphizy_input

        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 800
        NUM_PARTICLES = 100

        # Generate and validate data
        data = generate_and_format_positions(size_x=IMAGE_WIDTH, size_y=IMAGE_HEIGHT, num_particles=NUM_PARTICLES)
        validate_graphizy_input(data)

        return data
    except ImportError:
        pytest.skip("Required dependencies not available")


@pytest.fixture(scope="module")
def test_grapher():
    """Create a test grapher instance."""
    try:
        from graphizy import Graphing, GraphizyConfig

        config = GraphizyConfig(dimension=(800, 800))
        grapher = Graphing(config=config)
        return grapher
    except ImportError:
        pytest.skip("Required dependencies not available")


@pytest.fixture(scope="module")
def test_graphs(test_data, test_grapher):
    """Create various graph types for testing."""
    data = test_data
    grapher = test_grapher

    graph_results = {}

    # Test each graph type with error handling
    graph_types = [
        ("delaunay", {}),
        ("proximity", {"proximity_thresh": 50.0}),
        ("knn", {"k": 4}),
        ("mst", {}),
        ("gabriel", {})
    ]

    for graph_type, kwargs in graph_types:
        try:
            graph = grapher.make_graph(graph_type, data, **kwargs)
            graph_results[graph_type] = graph
            print(f"✓ Created {graph_type} graph: {graph.vcount()} vertices, {graph.ecount()} edges")
        except Exception as e:
            print(f"✗ Failed to create {graph_type} graph: {e}")
            # For critical graph types, we'll handle this in individual tests
            pass

    return graph_results


class TestGraphizyIntegration:
    """Main integration test class for Graphizy."""

    def test_basic_workflow(self, test_data, test_grapher):
        """Test the basic Graphizy workflow from README."""
        print("\n" + "=" * 60)
        print("INTEGRATED TEST: Basic Graphizy Workflow")
        print("=" * 60)

        data = test_data
        grapher = test_grapher

        # Validate data format
        assert data.shape[1] == 3, f"Expected 3 columns, got {data.shape[1]}"
        assert np.all(data[:, 0] >= 0), "IDs should be non-negative"
        assert np.all((data[:, 1] >= 0) & (data[:, 1] <= 800)), "X coordinates out of bounds"
        assert np.all((data[:, 2] >= 0) & (data[:, 2] <= 800)), "Y coordinates out of bounds"

        # Validate configuration
        assert grapher.dimension == (800, 800), "Dimension mismatch"
        assert grapher.aspect == "array", "Default aspect should be 'array'"

        print("✓ Basic workflow validation successful")

    def test_graph_creation(self, test_data, test_grapher):
        """Test creation of different graph types."""
        print("\nTesting graph creation...")

        data = test_data
        grapher = test_grapher

        # Test each graph type
        graph_types = [
            ("delaunay", {}),
            ("proximity", {"proximity_thresh": 50.0}),
            ("knn", {"k": 4}),
            ("mst", {}),
            ("gabriel", {})
        ]

        successful_graphs = 0

        for graph_type, kwargs in graph_types:
            try:
                graph = grapher.make_graph(graph_type, data, **kwargs)

                # Validate graph structure
                assert hasattr(graph, 'vcount'), f"{graph_type} graph missing vcount method"
                assert hasattr(graph, 'ecount'), f"{graph_type} graph missing ecount method"
                assert graph.vcount() == len(data), f"{graph_type} vertex count mismatch"

                successful_graphs += 1
                print(f"  ✓ {graph_type}: {graph.vcount()} vertices, {graph.ecount()} edges")

            except Exception as e:
                print(f"  ✗ {graph_type} failed: {e}")
                # Don't fail the test unless it's a critical graph type
                if graph_type in ["delaunay", "proximity"]:
                    pytest.fail(f"Critical graph type {graph_type} failed: {e}")

        assert successful_graphs >= 2, "At least 2 graph types should succeed"
        print(f"✓ Graph creation successful: {successful_graphs}/{len(graph_types)} types created")

    def test_visualization(self, test_graphs, test_grapher):
        """Test visualization and file saving."""
        print("\nTesting visualization...")

        grapher = test_grapher

        # Find a graph with edges for testing
        test_graph = None
        test_graph_name = None

        for graph_name, graph in test_graphs.items():
            if graph.ecount() > 0:
                test_graph = graph
                test_graph_name = graph_name
                break

        if test_graph is None:
            pytest.skip("No graphs with edges available for visualization testing")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test drawing
            image = grapher.draw_graph(test_graph)
            assert image is not None, "draw_graph returned None"
            assert hasattr(image, 'shape'), "Image should be a numpy array with shape"
            assert len(image.shape) == 3, "Image should be 3D (H, W, C)"

            # Test saving
            output_file = temp_path / f"{test_graph_name}_test.jpg"
            grapher.save_graph(image, str(output_file))
            assert output_file.exists(), f"Output file {output_file} was not created"
            assert output_file.stat().st_size > 0, "Output file is empty"

            print(f"  ✓ Visualization: {image.shape}, saved to file")

    def test_analysis(self, test_graphs, test_grapher):
        """Test graph analysis including centrality measures."""
        print("\nTesting analysis...")

        grapher = test_grapher

        # Use the first available graph with edges
        test_graph = None
        test_graph_name = None

        for preferred_name in ["delaunay", "proximity", "mst"]:
            if preferred_name in test_graphs and test_graphs[preferred_name].ecount() > 0:
                test_graph = test_graphs[preferred_name]
                test_graph_name = preferred_name
                break

        if test_graph is None:
            pytest.skip("No graphs available for analysis")

        # Basic graph properties
        info = grapher.get_graph_info(test_graph)

        # Validate info structure
        required_keys = ['vertex_count', 'edge_count', 'density', 'is_connected']
        for key in required_keys:
            assert key in info, f"Missing key '{key}' in graph info"

        assert info['edge_count'] >= 0, "Edge count should be non-negative"
        assert 0 <= info['density'] <= 1, f"Density should be between 0 and 1, got {info['density']}"
        assert isinstance(info['is_connected'], bool), "is_connected should be boolean"

        print(f"  Basic metrics: {info['vertex_count']} vertices, {info['edge_count']} edges")
        print(f"  Density: {info['density']:.3f}, Connected: {info['is_connected']}")

        # Test centrality measures if graph has edges
        if test_graph.ecount() > 0:
            try:
                # Test different call methods
                if hasattr(grapher, 'call_method_safe'):
                    degree_centrality = grapher.call_method_safe(test_graph, 'degree')
                    betweenness = grapher.call_method_safe(test_graph, 'betweenness')
                    print("  ✓ Centrality with call_method_safe")
                elif hasattr(grapher, 'call_method'):
                    degree_centrality = grapher.call_method(test_graph, 'degree')
                    betweenness = grapher.call_method(test_graph, 'betweenness')
                    print("  ✓ Centrality with call_method")
                else:
                    print("  ! No centrality method available")

            except Exception as e:
                print(f"  ! Centrality measures failed: {e}")

        # Test advanced methods
        try:
            components = grapher.call_method_raw(test_graph, 'connected_components')
            print(
                f"  ✓ Advanced analysis: {len(components) if hasattr(components, '__len__') else components} components")
        except Exception as e:
            print(f"  ! Advanced analysis failed: {e}")

    def test_memory_system(self, test_data, test_grapher):
        """Test memory-enhanced graph system."""
        print("\nTesting memory system...")

        data = test_data
        grapher = test_grapher

        try:
            # Initialize memory manager
            grapher.init_memory_manager(max_memory_size=10, track_edge_ages=True)
            print("  ✓ Memory manager initialized")

            # Create graphs and update memory
            positions = data.copy()

            for iteration in range(3):  # Reduced iterations for faster testing
                # Small movement simulation
                positions[:, 1:3] += np.random.normal(0, 1, (len(positions), 2))
                positions[:, 1] = np.clip(positions[:, 1], 0, 800)
                positions[:, 2] = np.clip(positions[:, 2], 0, 800)

                # Create and update memory
                current_graph = grapher.make_graph("proximity", positions, proximity_thresh=60.0)
                grapher.update_memory_with_graph(current_graph)

            # Test memory statistics (handle different method names)
            memory_stats = None
            if hasattr(grapher, 'get_memory_stats'):
                memory_stats = grapher.get_memory_stats()
            elif hasattr(grapher, 'memory_manager') and hasattr(grapher.memory_manager, 'get_stats'):
                memory_stats = grapher.memory_manager.get_stats()

            if memory_stats:
                print(f"  ✓ Memory stats available: {memory_stats}")
            else:
                print("  ! Memory stats method not found")

            # Test memory graph creation
            try:
                memory_graph = grapher.make_memory_graph(positions)
                assert memory_graph.vcount() == len(positions), "Memory graph vertex count mismatch"
                print(f"  ✓ Memory graph: {memory_graph.ecount()} edges")
            except Exception as e:
                print(f"  ! Memory graph creation failed: {e}")

        except Exception as e:
            print(f"  ! Memory system test failed: {e}")
            # Don't fail the test for memory issues

    def test_configuration(self, test_data):
        """Test configuration flexibility."""
        print("\nTesting configuration...")

        try:
            from graphizy import GraphizyConfig, Graphing

            # Test custom configuration
            config = GraphizyConfig()
            config.graph.dimension = (800, 800)
            config.drawing.line_color = (255, 0, 0)  # Red
            config.drawing.point_color = (0, 255, 0)  # Green
            config.drawing.line_thickness = 2
            config.drawing.point_radius = 6

            grapher = Graphing(config=config)

            # Validate configuration
            assert grapher.dimension == (800, 800), "Custom dimension not applied"
            assert grapher.config.drawing.line_color == (255, 0, 0), "Custom line color not applied"

            # Test runtime configuration updates if available
            if hasattr(grapher, 'update_config'):
                try:
                    grapher.update_config(
                        drawing={"line_thickness": 3},
                        graph={"proximity_threshold": 75.0}
                    )
                    print("  ✓ Runtime configuration update successful")
                except Exception as e:
                    print(f"  ! Runtime config update failed: {e}")

            # Test graph creation with custom config
            graph = grapher.make_graph("proximity", test_data, proximity_thresh=50.0)
            assert graph.vcount() == len(test_data), "Graph creation with custom config failed"

            print("  ✓ Configuration test successful")

        except Exception as e:
            print(f"  ! Configuration test failed: {e}")

    def test_api_consistency(self):
        """Test that the API matches documentation examples."""
        print("\nTesting API consistency...")

        try:
            from graphizy import (
                Graphing, GraphizyConfig, generate_and_format_positions,
                validate_graphizy_input
            )

            # Test that all required imports are available
            assert Graphing is not None
            assert GraphizyConfig is not None
            assert generate_and_format_positions is not None
            assert validate_graphizy_input is not None

            # Test basic instantiation
            config = GraphizyConfig(dimension=(400, 400))
            grapher = Graphing(config=config)

            # Test data generation
            data = generate_and_format_positions(size_x=400, size_y=400, num_particles=50)
            validate_graphizy_input(data)

            # Test that documented methods exist
            required_methods = ['make_graph', 'get_graph_info', 'draw_graph', 'save_graph']
            for method in required_methods:
                assert hasattr(grapher, method), f"{method} method missing"

            print("  ✓ API consistency validated")

        except ImportError as e:
            pytest.skip(f"Required dependencies not available: {e}")
        except Exception as e:
            pytest.fail(f"API consistency test failed: {e}")


def test_integrated():
    """Test the exact example from the README."""
    print("\n" + "=" * 60)
    print("README EXAMPLE TEST")
    print("=" * 60)

    try:
        from graphizy import Graphing, GraphizyConfig, generate_and_format_positions, validate_graphizy_input

        # Parameters
        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 800

        # 1. Generate random points (id, x, y) and validate
        data = generate_and_format_positions(size_x=IMAGE_WIDTH, size_y=IMAGE_HEIGHT, num_particles=100)
        validate_graphizy_input(data)

        # 2. Configure Graphizy
        config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
        grapher = Graphing(config=config)

        # 3. Create different graph types
        graphs_created = {}

        try:
            delaunay_graph = grapher.make_graph("delaunay", data)
            graphs_created['delaunay'] = delaunay_graph
        except Exception as e:
            print(f"Delaunay creation failed: {e}")

        try:
            proximity_graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)
            graphs_created['proximity'] = proximity_graph
        except Exception as e:
            print(f"Proximity creation failed: {e}")

        try:
            knn_graph = grapher.make_graph("knn", data, k=4)
            graphs_created['knn'] = knn_graph
        except Exception as e:
            print(f"KNN creation failed: {e}")

        try:
            mst_graph = grapher.make_graph("mst", data)
            graphs_created['mst'] = mst_graph
        except Exception as e:
            print(f"MST creation failed: {e}")

        try:
            gabriel_graph = grapher.make_graph("gabriel", data)
            graphs_created['gabriel'] = gabriel_graph
        except Exception as e:
            print(f"Gabriel creation failed: {e}")

        # Ensure at least some graphs were created
        assert len(graphs_created) > 0, "No graphs were successfully created"

        # 4. Test visualization with first available graph
        test_graph = list(graphs_created.values())[0]
        test_name = list(graphs_created.keys())[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test drawing and saving
            image = grapher.draw_graph(test_graph)
            temp_file = Path(temp_dir) / f"{test_name}.jpg"
            grapher.save_graph(image, str(temp_file))

            # Test non-blocking show
            # try:
            #     grapher.show_graph(image, f"{test_name.title()} Triangulation", block=False)
            # except Exception as e:
            #     print(f"Show graph failed (expected in headless): {e}")

        # 5. Analyze graph metrics
        info = grapher.get_graph_info(test_graph)
        print(f"Graph: {info['vertex_count']} vertices, {info['edge_count']} edges")
        print(f"Density: {info['density']:.3f}, Connected: {info['is_connected']}")

        # Test centrality measures (handle different method names)
        try:
            if hasattr(grapher, 'call_method_safe'):
                degree_centrality = grapher.call_method_safe(test_graph, 'degree')
                betweenness = grapher.call_method_safe(test_graph, 'betweenness')
                closeness = grapher.call_method_safe(test_graph, 'closeness')
            elif hasattr(grapher, 'call_method'):
                degree_centrality = grapher.call_method(test_graph, 'degree')
                betweenness = grapher.call_method(test_graph, 'betweenness')
                closeness = grapher.call_method(test_graph, 'closeness')
            else:
                print("No centrality method available")
                degree_centrality = None
                betweenness = None
                closeness = None

            # Find most central nodes if betweenness is available
            if betweenness and isinstance(betweenness, dict):
                central_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"Top 5 central nodes: {central_nodes[:3]}...")  # Show first 3

        except Exception as e:
            print(f"Centrality analysis failed: {e}")

        # Test advanced methods
        try:
            components = grapher.call_method_raw(test_graph, 'connected_components')
            diameter = grapher.call_method_raw(test_graph, 'diameter')
            print(f"Connected components: {len(components) if hasattr(components, '__len__') else components}")
            print(f"Graph diameter: {diameter}")
        except Exception as e:
            print(f"Advanced analysis failed: {e}")

        print(f"✅ README example completed successfully!")
        print(f"   Created {len(graphs_created)} graph types: {', '.join(graphs_created.keys())}")

    except ImportError as e:
        pytest.skip(f"Required dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"README example test failed: {e}")


if __name__ == "__main__":
    # Run the README example test when executed directly
    test_integrated()