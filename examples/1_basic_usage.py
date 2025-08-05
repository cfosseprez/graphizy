# examples/1_basic_usage.py

#!/usr/bin/env python3
"""
Basic Usage Examples for the Graphizy Package

This script serves as a user-friendly guide to the core features of Graphizy.
It demonstrates how to:
1. Generate different types of graphs from point data (Delaunay, Proximity, K-Nearest, MST, Gabriel).
2. Customize the visual style of the graphs (colors, sizes, etc.).
3. Analyze graph properties and statistics using the modern, lazy-loading API.
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
import random

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
        particle_stack = generate_and_format_positions(
            size_x=IMAGE_WIDTH,
            size_y=IMAGE_HEIGHT,
            num_particles=NUM_PARTICLES
        )
        validate_graphizy_input(particle_stack)

        # --- Styling and Graph Creation ---
        config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
        config.drawing.line_color = (255, 0, 0)  # Red lines (B, G, R in OpenCV)
        config.drawing.point_color = (0, 0, 255)  # Blue points
        config.drawing.line_thickness = 1
        config.drawing.point_radius = 5

        # Define the simple data shape we are using to prevent warnings.
        simple_data_shape = [('id', int), ('x', float), ('y', float)]

        # Initialize the main Graphing object with our custom configuration and data_shape.
        grapher = Graphing(config=config, data_shape=simple_data_shape)

        # Use the modern unified make_graph interface
        print("Creating the Delaunay triangulation...")
        delaunay_graph = grapher.make_graph("delaunay", particle_stack)

        # --- Analysis ---
        # Get the lazy-loading analysis object. This call is instantaneous.
        results = grapher.get_graph_info(delaunay_graph)

        # Access metrics as properties. Computation happens on first access.
        print("Delaunay Graph Statistics:")
        print(f"  - Vertices (Nodes): {results.vertex_count}")
        print(f"  - Edges (Connections): {results.edge_count}")
        print(f"  - Density: {results.density:.4f}")
        print(f"  - Is Connected: {results.is_connected}")

        if results.average_path_length is not None:
            print(f"  - Average Path Length: {results.average_path_length:.4f}")

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
    a specified threshold.
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

        config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
        config.drawing.line_color = (255, 0, 0)
        config.drawing.point_color = (255, 255, 0)
        config.drawing.line_thickness = 2
        config.drawing.point_radius = 8

        simple_data_shape = [('id', int), ('x', float), ('y', float)]
        grapher = Graphing(config=config, data_shape=simple_data_shape)

        # --- Graph Creation ---
        print(f"Creating proximity graph with threshold {THRESHOLD} pixels...")
        proximity_graph = grapher.make_graph(
            "proximity",
            particle_stack,
            proximity_thresh=THRESHOLD,
            metric='euclidean'
        )

        # --- Analysis ---
        results = grapher.get_graph_info(proximity_graph)
        print("Proximity Graph Statistics:")
        print(f"  - Vertices: {results.vertex_count}")
        print(f"  - Edges: {results.edge_count}")
        print(f"  - Density: {results.density:.4f}")
        print(f"  - Is Connected: {results.is_connected}")
        print(f"  - Number of Components: {results.num_components}")

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
    Demonstrates creating a K-Nearest Neighbors graph.

    In a KNN graph, every point is connected to its 'K' closest neighbors.
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
        config.drawing.line_color = (0, 255, 255)
        config.drawing.point_color = (255, 0, 255)
        config.drawing.line_thickness = 2
        config.drawing.point_radius = 10

        simple_data_shape = [('id', int), ('x', float), ('y', float)]
        grapher = Graphing(config=config, data_shape=simple_data_shape)

        # --- Graph Creation ---
        print(f"Creating k-nearest neighbors graph with K={K}...")
        knn_graph = grapher.make_graph("knn", particle_stack, k=K)

        # --- Analysis ---
        results = grapher.get_graph_info(knn_graph)
        print("K-Nearest Neighbors Graph Statistics:")
        print(f"  - Vertices: {results.vertex_count}")
        print(f"  - Edges: {results.edge_count}")
        print(f"  - Density: {results.density:.4f}")

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
    Demonstrates creating a Minimum Spanning Tree.

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
        config.drawing.line_color = (0, 128, 255)
        config.drawing.point_color = (128, 0, 128)
        config.drawing.line_thickness = 3
        config.drawing.point_radius = 6

        simple_data_shape = [('id', int), ('x', float), ('y', float)]
        grapher = Graphing(config=config, data_shape=simple_data_shape)

        # --- Graph Creation ---
        print("Creating minimum spanning tree...")
        mst_graph = grapher.make_graph("mst", particle_stack, metric="euclidean")

        # --- Analysis ---
        results = grapher.get_graph_info(mst_graph)
        print("Minimum Spanning Tree Statistics:")
        print(f"  - Vertices: {results.vertex_count}")
        print(f"  - Edges: {results.edge_count}")
        print(f"  - Is Connected: {results.is_connected}")  # Always True for MST

        # Verify MST property: edges = vertices - 1
        expected_edges = results.vertex_count - 1
        print(f"  - MST Property (edges = vertices - 1): {results.edge_count == expected_edges}")
        print(f"    Expected edges: {expected_edges}, Actual edges: {results.edge_count}")

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
    Demonstrates creating a Gabriel graph.

    A Gabriel graph connects two points if no other point lies within the circle
    having the two points as its diameter.
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
        config.drawing.line_color = (128, 255, 0)
        config.drawing.point_color = (255, 128, 0)
        config.drawing.line_thickness = 2
        config.drawing.point_radius = 7

        simple_data_shape = [('id', int), ('x', float), ('y', float)]
        grapher = Graphing(config=config, data_shape=simple_data_shape)

        # --- Graph Creation ---
        print("Creating Gabriel graph...")
        gabriel_graph = grapher.make_graph("gabriel", particle_stack)

        # --- Analysis ---
        results = grapher.get_graph_info(gabriel_graph)
        print("Gabriel Graph Statistics:")
        print(f"  - Vertices: {results.vertex_count}")
        print(f"  - Edges: {results.edge_count}")
        print(f"  - Is Connected: {results.is_connected}")

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
    """
    print("\n" + "=" * 60)
    print(" EXAMPLE 6: COMPREHENSIVE GRAPH COMPARISON")
    print("=" * 60)

    valid_graphs = {name: graph for name, graph in graphs.items() if graph is not None}
    if not valid_graphs:
        print("No valid graphs to compare.")
        return

    try:
        simple_data_shape = [('id', int), ('x', float), ('y', float)]
        grapher = Graphing(data_shape=simple_data_shape)

        # Print table header
        print(f"{'Property':<25} " + "".join([f"{name.title():<15}" for name in valid_graphs.keys()]))
        print("-" * (25 + 15 * len(valid_graphs)))

        # Basic properties
        for prop in ['vertex_count', 'edge_count', 'density', 'is_connected']:
            row = f"{prop.replace('_', ' ').title():<25}"
            for name, graph in valid_graphs.items():
                results = grapher.get_graph_info(graph)
                value = results[prop]  # Use dictionary-style access for dynamic properties
                if isinstance(value, float):
                    row += f"{value:<15.3f}"
                else:
                    row += f"{str(value):<15}"
            print(row)

    except Exception as e:
        print(f"Comparison failed: {e}")


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    Runs all the basic usage examples using the updated API.
    """
    print("=" * 60)
    print("Graphizy Basic Usage Examples - Updated for v0.1.18+")
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

        # Run comparisons
        example_comprehensive_comparison(all_graphs)

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print(f"Check the 'examples/output/' folder for generated images.")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())