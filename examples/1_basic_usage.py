#!/usr/bin/env python3
"""
Basic Usage Examples for the Graphizy Package

This script serves as a user-friendly guide to the core features of Graphizy.
It demonstrates how to:
1. Generate different types of graphs from point data (Delaunay, Proximity, K-Nearest, MST, Gabriel).
2. Customize the visual style of the graphs (colors, sizes, etc.).
3. Analyze graph properties and statistics using the latest API.
4. Compare the characteristics of different graph types.
5. Use the modern make_graph() unified interface.

The code is designed to be clear and follows the latest API patterns.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

# --- Core Libraries ---
import numpy as np
import logging
import sys
from pathlib import Path

# --- Graphizy Imports ---
from graphizy import (
    Graphing,  # The main class that does all the work.
    GraphizyConfig,  # A class for configuring graph settings and styles.
    generate_and_format_positions,  # A helper function to create random 2D points.
    validate_graphizy_input,  # A helper function to validate coordinates
    GraphizyError  # Custom error class for handling package-specific issues.
)
# This utility helps create an 'output' directory if it doesn't exist.
from graphizy.utils import setup_output_directory

# --- Initial Setup ---
# Configure logging to show messages at the INFO level or higher.
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# =============================================================================
# EXAMPLE 1: DELAUNAY TRIANGULATION
# =============================================================================

def example_delaunay_triangulation():
    """
    Demonstrates creating a Delaunay triangulation using the modern API.

    A Delaunay triangulation connects a set of points to form a mesh of triangles.
    A key property is that no point is inside the circumcircle of any triangle,
    which tends to create well-proportioned, non-skinny triangles.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 1: DELAUNAY TRIANGULATION")
    print("=" * 50)

    try:
        # Define the dimensions of our canvas and the number of points (nodes).
        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600
        NUM_PARTICLES = 60

        # --- Data Generation ---
        print(f"Generating {NUM_PARTICLES} random points on an {IMAGE_WIDTH}x{IMAGE_HEIGHT} canvas...")
        # Use the updated helper function that returns properly formatted data
        particle_stack = generate_and_format_positions(
            size_x=IMAGE_WIDTH,
            size_y=IMAGE_HEIGHT,
            num_particles=NUM_PARTICLES
        )
        validate_graphizy_input(particle_stack)

        # --- Styling and Graph Creation ---
        config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
        # Use the updated drawing configuration method
        config.drawing.line_color = (255, 0, 0)  # Red lines (B, G, R in OpenCV)
        config.drawing.point_color = (0, 0, 255)  # Blue points
        config.drawing.line_thickness = 1
        config.drawing.point_radius = 5

        # Initialize the main Graphing object with our custom configuration.
        grapher = Graphing(config=config)

        # UPDATED: Use the modern unified make_graph interface
        print("Creating the Delaunay triangulation...")
        delaunay_graph = grapher.make_graph("delaunay", particle_stack)

        # --- Analysis ---
        # Get and display key statistics about the generated graph.
        info = grapher.get_graph_info(delaunay_graph)
        print("Delaunay Graph Statistics:")
        print(f"  - Vertices (Nodes): {info['vertex_count']}")
        print(f"  - Edges (Connections): {info['edge_count']}")
        print(f"  - Density: {info['density']:.4f}")
        print(f"  - Is Connected: {info['is_connected']}")

        if info.get('average_path_length') is not None:
            print(f"  - Average Path Length: {info['average_path_length']:.4f}")

        # Additional analysis using the updated API
        connectivity_info = grapher.get_connectivity_info(delaunay_graph)
        print(f"  - Number of Components: {connectivity_info['num_components']}")

        # --- Output ---
        # Draw the graph onto an image canvas and save it to a file.
        image = grapher.draw_graph(delaunay_graph)
        output_path = setup_output_directory() / "delaunay_triangulation.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Graph saved to: {output_path}")

        return delaunay_graph, particle_stack

    except GraphizyError as e:
        print(f"Error during Delaunay triangulation: {e}")
        return None, None


# =============================================================================
# EXAMPLE 2: PROXIMITY GRAPH
# =============================================================================

def example_proximity_graph(particle_stack):
    """
    Demonstrates creating a proximity graph using the updated API.

    This connects any two points if the distance between them is less than
    a specified threshold. Uses the modern make_graph interface.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 2: PROXIMITY GRAPH")
    print("=" * 50)

    if particle_stack is None:
        print("Skipping proximity graph example because particle data is missing.")
        return None

    try:
        THRESHOLD = 80.0
        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600

        # Create configuration using the updated API
        config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
        config.drawing.line_color = (255, 0, 0)  # Red lines
        config.drawing.point_color = (255, 255, 0)  # Yellow points
        config.drawing.line_thickness = 2
        config.drawing.point_radius = 8

        grapher = Graphing(config=config)

        # --- Graph Creation using modern unified API ---
        print(f"Creating proximity graph with threshold {THRESHOLD} pixels...")
        proximity_graph = grapher.make_graph(
            "proximity",
            particle_stack,
            proximity_thresh=THRESHOLD,
            metric='euclidean'
        )

        # --- Analysis using updated methods ---
        info = grapher.get_graph_info(proximity_graph)
        print("Proximity Graph Statistics:")
        print(f"  - Vertices: {info['vertex_count']}")
        print(f"  - Edges: {info['edge_count']}")
        print(f"  - Density: {info['density']:.4f}")
        print(f"  - Is Connected: {info['is_connected']}")

        if info.get('average_path_length') is not None:
            print(f"  - Average Path Length: {info['average_path_length']:.4f}")

        # Use the updated connection analysis method
        connections_per_node = grapher.get_connections_per_object(proximity_graph)
        avg_connections = np.mean(list(connections_per_node.values()))
        max_connections = max(connections_per_node.values()) if connections_per_node else 0

        print(f"  - Average Connections per Node: {avg_connections:.2f}")
        print(f"  - Maximum Connections for a Single Node: {max_connections}")

        # --- Output ---
        image = grapher.draw_graph(proximity_graph)
        output_path = setup_output_directory() / "proximity_graph.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Graph saved to: {output_path}")

        return proximity_graph

    except GraphizyError as e:
        print(f"Error during proximity graph creation: {e}")
        return None


# =============================================================================
# EXAMPLE 3: K-NEAREST NEIGHBORS (KNN) GRAPH
# =============================================================================

def example_k_nearest_neighbors(particle_stack):
    """
    Demonstrates creating a K-Nearest Neighbors graph using the modern API.

    In a KNN graph, every point is connected to its 'K' closest neighbors.
    Uses the unified make_graph interface.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 3: K-NEAREST NEIGHBORS (KNN) GRAPH")
    print("=" * 50)

    if particle_stack is None:
        print("Skipping k-nearest neighbors example because particle data is missing.")
        return None

    try:
        K = 4
        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600

        config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
        config.drawing.line_color = (0, 255, 255)  # Cyan lines
        config.drawing.point_color = (255, 0, 255)  # Magenta points
        config.drawing.line_thickness = 2
        config.drawing.point_radius = 10

        grapher = Graphing(config=config)

        # --- Graph Creation using modern unified API ---
        print(f"Creating k-nearest neighbors graph with K={K}...")
        knn_graph = grapher.make_graph("knn", particle_stack, k=K)

        # --- Analysis ---
        info = grapher.get_graph_info(knn_graph)
        print("K-Nearest Neighbors Graph Statistics:")
        print(f"  - Vertices: {info['vertex_count']}")
        print(f"  - Edges: {info['edge_count']}")
        print(f"  - Density: {info['density']:.4f}")
        print(f"  - Is Connected: {info['is_connected']}")

        if info.get('average_path_length') is not None:
            print(f"  - Average Path Length: {info['average_path_length']:.4f}")

        # Verify the connectivity
        connections_per_node = grapher.get_connections_per_object(knn_graph)
        avg_connections = np.mean(list(connections_per_node.values()))

        print(f"  - Expected Average Connections (approx.): {K * 2}")
        print(f"  - Actual Average Connections: {avg_connections:.2f}")

        # --- Output ---
        image = grapher.draw_graph(knn_graph)
        output_path = setup_output_directory() / "k_nearest_neighbors.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Graph saved to: {output_path}")

        return knn_graph

    except (GraphizyError, ImportError) as e:
        print(f"Error during k-nearest neighbors graph creation: {e}")
        return None


# =============================================================================
# EXAMPLE 4: MINIMUM SPANNING TREE (MST) GRAPH
# =============================================================================

def example_minimum_spanning_tree(particle_stack):
    """
    Demonstrates creating a Minimum Spanning Tree using the modern API.

    MST creates the minimal connected graph by selecting the shortest edges
    that connect all vertices without creating cycles.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 4: MINIMUM SPANNING TREE (MST)")
    print("=" * 50)

    if particle_stack is None:
        print("Skipping MST example because particle data is missing.")
        return None

    try:
        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600

        config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
        config.drawing.line_color = (0, 128, 255)  # Orange lines
        config.drawing.point_color = (128, 0, 128)  # Purple points
        config.drawing.line_thickness = 3
        config.drawing.point_radius = 6

        grapher = Graphing(config=config)

        # --- Graph Creation using modern unified API ---
        print("Creating minimum spanning tree...")
        mst_graph = grapher.make_graph("mst", particle_stack, metric="euclidean")

        # --- Analysis ---
        info = grapher.get_graph_info(mst_graph)
        print("Minimum Spanning Tree Statistics:")
        print(f"  - Vertices: {info['vertex_count']}")
        print(f"  - Edges: {info['edge_count']}")
        print(f"  - Density: {info['density']:.4f}")
        print(f"  - Is Connected: {info['is_connected']}")  # Always True for MST

        # Verify MST property: edges = vertices - 1
        n_vertices = info['vertex_count']
        n_edges = info['edge_count']
        expected_edges = n_vertices - 1
        print(f"  - MST Property (edges = vertices - 1): {n_edges == expected_edges}")
        print(f"    Expected edges: {expected_edges}, Actual edges: {n_edges}")

        if info.get('average_path_length') is not None:
            print(f"  - Average Path Length: {info['average_path_length']:.4f}")

        # --- Output ---
        image = grapher.draw_graph(mst_graph)
        output_path = setup_output_directory() / "minimum_spanning_tree.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Graph saved to: {output_path}")

        return mst_graph

    except (GraphizyError, ImportError) as e:
        print(f"Error during MST creation: {e}")
        return None


# =============================================================================
# EXAMPLE 5: GABRIEL GRAPH
# =============================================================================

def example_gabriel_graph(particle_stack):
    """
    Demonstrates creating a Gabriel graph using the modern API.

    Gabriel graph connects two points if no other point lies within the circle
    having the two points as diameter endpoints.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 5: GABRIEL GRAPH")
    print("=" * 50)

    if particle_stack is None:
        print("Skipping Gabriel graph example because particle data is missing.")
        return None

    try:
        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600

        config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
        config.drawing.line_color = (128, 255, 0)  # Green-yellow lines
        config.drawing.point_color = (255, 128, 0)  # Orange points
        config.drawing.line_thickness = 2
        config.drawing.point_radius = 7

        grapher = Graphing(config=config)

        # --- Graph Creation using modern unified API ---
        print("Creating Gabriel graph...")
        gabriel_graph = grapher.make_graph("gabriel", particle_stack)

        # --- Analysis ---
        info = grapher.get_graph_info(gabriel_graph)
        print("Gabriel Graph Statistics:")
        print(f"  - Vertices: {info['vertex_count']}")
        print(f"  - Edges: {info['edge_count']}")
        print(f"  - Density: {info['density']:.4f}")
        print(f"  - Is Connected: {info['is_connected']}")  # May be False

        if info.get('average_path_length') is not None:
            print(f"  - Average Path Length: {info['average_path_length']:.4f}")

        # Gabriel graph is always a subset of Delaunay triangulation
        connectivity_info = grapher.get_connectivity_info(gabriel_graph)
        if connectivity_info['num_components'] > 1:
            print(f"  - Number of Components: {connectivity_info['num_components']}")
            print(f"  - Largest Component Size: {connectivity_info['largest_component_size']}")

        # --- Output ---
        image = grapher.draw_graph(gabriel_graph)
        output_path = setup_output_directory() / "gabriel_graph.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Graph saved to: {output_path}")

        return gabriel_graph

    except (GraphizyError, ImportError) as e:
        print(f"Error during Gabriel graph creation: {e}")
        return None


# =============================================================================
# EXAMPLE 6: COMPREHENSIVE GRAPH COMPARISON
# =============================================================================

def example_comprehensive_comparison(graphs):
    """
    Compares the statistics of all generated graphs side-by-side.
    Updated to include all modern graph types and use latest API.
    """
    print("\n" + "=" * 60)
    print(" EXAMPLE 6: COMPREHENSIVE GRAPH COMPARISON")
    print("=" * 60)

    # Filter out None graphs
    valid_graphs = {name: graph for name, graph in graphs.items() if graph is not None}

    if not valid_graphs:
        print("No valid graphs to compare.")
        return

    try:
        # Create a single grapher for analysis
        grapher = Graphing()

        # Print table header
        print(f"{'Property':<25} " + "".join([f"{name:<15}" for name in valid_graphs.keys()]))
        print("-" * (25 + 15 * len(valid_graphs)))

        # Basic properties
        for prop in ['vertex_count', 'edge_count', 'density', 'is_connected']:
            row = f"{prop.replace('_', ' ').title():<25}"
            for name, graph in valid_graphs.items():
                info = grapher.get_graph_info(graph)
                value = info.get(prop, 'N/A')
                if isinstance(value, float):
                    row += f"{value:<15.3f}"
                else:
                    row += f"{str(value):<15}"
            print(row)

        # Connectivity analysis
        print("\n" + "=" * 60)
        print("DETAILED CONNECTIVITY ANALYSIS")
        print("=" * 60)

        for name, graph in valid_graphs.items():
            print(f"\n--- {name.upper()} GRAPH ---")

            # Use safe method calls for disconnected graphs
            connectivity_info = grapher.get_connectivity_info(graph)
            print(f"  Connected: {connectivity_info['is_connected']}")
            print(f"  Components: {connectivity_info['num_components']}")

            if connectivity_info['num_components'] > 1:
                print(f"  Largest component: {connectivity_info['largest_component_size']} vertices")
                print(f"  Connectivity ratio: {connectivity_info['connectivity_ratio']:.1%}")

            # Safely compute path length
            avg_path = grapher.call_method_safe(graph, "average_path_length",
                                                component_mode="largest", default_value=None)
            if avg_path is not None:
                print(f"  Average path length: {avg_path:.3f}")

            # Compute degree statistics
            connections = grapher.get_connections_per_object(graph)
            if connections:
                degree_values = list(connections.values())
                print(f"  Degree - Avg: {np.mean(degree_values):.2f}, "
                      f"Range: {min(degree_values)}-{max(degree_values)}")

        # Theoretical properties comparison
        print("\n" + "=" * 60)
        print("THEORETICAL PROPERTIES")
        print("=" * 60)

        properties = {
            "Delaunay": "Always connected, ~3n edges, planar",
            "Proximity": "Variable connectivity, depends on threshold",
            "KNN": "Variable connectivity, k*n directed edges",
            "MST": "Always connected, exactly n-1 edges",
            "Gabriel": "Subset of Delaunay, may be disconnected"
        }

        for name, graph in valid_graphs.items():
            if name.title() in properties:
                print(f"  {name.title()}: {properties[name.title()]}")

    except Exception as e:
        print(f"Comparison failed: {e}")


# =============================================================================
# EXAMPLE 7: CONFIGURATION SHOWCASE
# =============================================================================

def example_configuration_showcase():
    """
    Shows how to create multiple graphs with different visual styles using
    the updated configuration system.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 7: CONFIGURATION SHOWCASE")
    print("=" * 50)

    try:
        # Use smaller dataset for quick showcase
        IMAGE_WIDTH, IMAGE_HEIGHT = 400, 400
        NUM_PARTICLES = 30

        positions = generate_and_format_positions(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_PARTICLES)

        # Define style configurations using the updated config system
        styles = [
            {
                "name": "classic_dark",
                "drawing": {
                    "line_color": (0, 255, 0),  # Green
                    "point_color": (0, 0, 255),  # Blue
                    "line_thickness": 1,
                    "point_radius": 6
                }
            },
            {
                "name": "bold_neon",
                "drawing": {
                    "line_color": (255, 0, 255),  # Magenta
                    "point_color": (0, 255, 255),  # Cyan
                    "line_thickness": 3,
                    "point_radius": 12
                }
            },
            {
                "name": "minimal_grayscale",
                "drawing": {
                    "line_color": (128, 128, 128),  # Gray
                    "point_color": (64, 64, 64),  # Dark gray
                    "line_thickness": 1,
                    "point_radius": 4
                }
            }
        ]

        output_dir = setup_output_directory()

        # Create graphs with different styles
        for style in styles:
            print(f"Creating '{style['name']}' style graph...")

            # Use the modern configuration approach
            config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
            config.update(drawing=style["drawing"])

            grapher = Graphing(config=config)

            # Use the modern make_graph interface
            graph = grapher.make_graph("delaunay", positions)

            # Draw and save
            image = grapher.draw_graph(graph)
            filename = f"style_{style['name']}.jpg"
            grapher.save_graph(image, str(output_dir / filename))
            print(f"  Saved {filename}")

        print("\nStyle showcase complete!")

    except Exception as e:
        print(f"Configuration showcase failed: {e}")


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    Runs all the basic usage examples using the updated API.
    """
    print("=" * 60)
    print("Graphizy Basic Usage Examples - Updated for v0.1.17+")
    print("=" * 60)

    try:
        # Run examples sequentially
        delaunay_graph, particle_stack = example_delaunay_triangulation()
        proximity_graph = example_proximity_graph(particle_stack)
        knn_graph = example_k_nearest_neighbors(particle_stack)
        mst_graph = example_minimum_spanning_tree(particle_stack)
        gabriel_graph = example_gabriel_graph(particle_stack)

        # Collect all graphs for comparison
        all_graphs = {
            "delaunay": delaunay_graph,
            "proximity": proximity_graph,
            "knn": knn_graph,
            "mst": mst_graph,
            "gabriel": gabriel_graph
        }

        # Run comparisons and showcase
        example_comprehensive_comparison(all_graphs)
        example_configuration_showcase()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("Key Updates in this Version:")
        print("  • Uses modern make_graph() unified interface")
        print("  • Includes MST and Gabriel graph types")
        print("  • Updated configuration system with GraphizyConfig")
        print("  • Enhanced error handling and analysis methods")
        print("  • Comprehensive graph comparison including all types")
        print(f"Check the 'examples/output/' folder for generated images.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())