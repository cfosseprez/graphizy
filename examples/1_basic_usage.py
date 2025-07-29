#!/usr/bin/env python3
"""
Basic Usage Examples for the Graphizy Package

This script serves as a user-friendly guide to the core features of Graphizy.
It demonstrates how to:
1. Generate different types of graphs from point data (Delaunay, Proximity, K-Nearest).
2. Customize the visual style of the graphs (colors, sizes, etc.).
3. Analyze graph properties and statistics.
4. Compare the characteristics of different graph types.

The code is designed to be clear and easy to follow.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

# --- Core Libraries ---
import numpy as np
import logging
import sys

# --- Graphizy Imports ---
from graphizy import (
    Graphing,  # The main class that does all the work.
    GraphizyConfig,  # A class for configuring graph settings and styles.
    generate_positions,  # A helper function to create random 2D points.
    GraphizyError  # Custom error class for handling package-specific issues.
)

# This utility helps create an 'output' directory if it doesn't exist.
from graphizy.utils import setup_output_directory

# --- Initial Setup ---
# Configure logging to show messages at the INFO level or higher.
# This helps in tracking the script's progress and diagnosing issues.
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')


# =============================================================================
# EXAMPLE 1: DELAUNAY TRIANGULATION
# =============================================================================

def example_delaunay_triangulation():
    """
    Demonstrates creating a Delaunay triangulation.

    A Delaunay triangulation connects a set of points to form a mesh of triangles.
    A key property is that no point is inside the circumcircle of any triangle,
    which tends to create well-proportioned, non-skinny triangles.
    It's widely used in computational geometry and spatial analysis.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 1: DELAUNAY TRIANGULATION")
    print("=" * 50)

    try:
        # Define the dimensions of our canvas and the number of points (nodes).
        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600
        NUM_PARTICLES = 60

        # --- Data Generation ---
        # Create random (x, y) coordinates for our particles within the canvas.
        print(f"Generating {NUM_PARTICLES} random points on an {IMAGE_WIDTH}x{IMAGE_HEIGHT} canvas...")
        positions = generate_positions(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_PARTICLES)

        # The grapher needs data in a specific format: an array where each row is
        # [ID, x_position, y_position]. We create unique IDs for each particle.
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # --- Styling and Graph Creation ---
        # Create a configuration object to customize the graph's appearance.
        config = GraphizyConfig(dimension=(IMAGE_WIDTH, IMAGE_HEIGHT))
        config.set_drawing(line_color=(255, 0, 0), point_color = (0, 0, 255), line_thickness = 1, point_radius=5)

        # Initialize the main Graphing object with our custom configuration.
        grapher = Graphing(config=config)

        # This is the core step: generate the Delaunay graph from our particle data.
        print("Creating the Delaunay triangulation...")
        delaunay_graph = grapher.make_delaunay(particle_stack)

        # --- Analysis ---
        # Get and display key statistics about the generated graph.
        info = grapher.get_graph_info(delaunay_graph)
        print("Delaunay Graph Statistics:")
        print(f"  - Vertices (Nodes): {info['vertex_count']}")
        print(f"  - Edges (Connections): {info['edge_count']}")
        # Density: How many of the possible connections exist? (1.0 is a fully connected graph).
        print(f"  - Density: {info['density']:.4f}")
        # Is Connected: Can you travel from any node to any other node?
        print(f"  - Is Connected: {info['is_connected']}")
        if info['average_path_length']:
            print(f"  - Average Path Length: {info['average_path_length']:.4f}")

        # --- Output ---
        # Draw the graph onto an image canvas and save it to a file.
        image = grapher.draw_graph(delaunay_graph)
        output_path = setup_output_directory() / "delaunay_triangulation.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Graph saved to: {output_path}")

        # Return the graph and data for use in the next examples.
        return delaunay_graph, particle_stack

    except GraphizyError as e:
        # Catch any errors specific to the Graphizy package.
        print(f"Error during Delaunay triangulation: {e}")
        return None, None


# =============================================================================
# EXAMPLE 2: PROXIMITY GRAPH
# =============================================================================

def example_proximity_graph(particle_stack):
    """
    Demonstrates creating a proximity graph.

    This is a simple and intuitive graph type. An edge (connection) is created
    between any two points if the distance between them is less than a specified
    'threshold'. It's like defining a "personal space" around each point; any
    other point that enters this space becomes a neighbor.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 2: PROXIMITY GRAPH")
    print("=" * 50)

    # Skip this example if the previous one failed to produce data.
    if particle_stack is None:
        print("Skipping proximity graph example because particle data is missing.")
        return None

    try:
        # --- Configuration ---
        # The most important parameter here is the THRESHOLD.
        # A larger threshold will result in more connections and a denser graph.
        THRESHOLD = 80.0
        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600

        # We'll create a new configuration to show how styles can be different.
        config = GraphizyConfig()
        config.graph.dimension = (IMAGE_WIDTH, IMAGE_HEIGHT)
        config.graph.proximity_threshold = THRESHOLD  # Set the distance threshold.
        config.drawing.line_color = (255, 0, 0)  # Red lines.
        config.drawing.point_color = (255, 255, 0)  # Yellow points.
        config.drawing.line_thickness = 2
        config.drawing.point_radius = 8

        grapher = Graphing(config=config)

        # --- Graph Creation ---
        # Create the graph. Any two points closer than THRESHOLD pixels will be connected.
        print(f"Creating proximity graph with a distance threshold of {THRESHOLD} pixels...")
        proximity_graph = grapher.make_proximity(
            particle_stack,
            proximity_thresh=THRESHOLD,
            metric='euclidean'  # Use standard straight-line distance.
        )

        # --- Analysis ---
        info = grapher.get_graph_info(proximity_graph)
        print("Proximity Graph Statistics:")
        print(f"  - Vertices: {info['vertex_count']}")
        print(f"  - Edges: {info['edge_count']}")
        print(f"  - Density: {info['density']:.4f}")
        print(f"  - Is Connected: {info['is_connected']}")
        if info['average_path_length']:
            print(f"  - Average Path Length: {info['average_path_length']:.4f}")

        # We can also perform more specific analysis, like checking connections per node.
        connections_per_node = grapher.get_connections_per_object(proximity_graph)
        avg_connections = np.mean(list(connections_per_node.values()))
        max_connections = max(connections_per_node.values())

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
    Demonstrates creating a K-Nearest Neighbors (KNN) graph.

    In a KNN graph, every point is connected to its 'K' closest neighbors.
    This is different from a proximity graph, where a point could have 0 or
    many neighbors. In a basic KNN graph, every point will have exactly K
    outgoing connections. It's very useful for finding local clusters in data.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 3: K-NEAREST NEIGHBORS (KNN) GRAPH")
    print("=" * 50)

    if particle_stack is None:
        print("Skipping k-nearest neighbors example because particle data is missing.")
        return None

    try:
        # This example shows using an algorithm directly from the package.
        from graphizy.algorithms import create_k_nearest_graph

        # --- Configuration ---
        # 'K' is the number of nearest neighbors to connect to for each point.
        K = 4
        IMAGE_WIDTH, IMAGE_HEIGHT = 800, 600

        # Another style configuration.
        config = GraphizyConfig()
        config.graph.dimension = (IMAGE_WIDTH, IMAGE_HEIGHT)
        config.drawing.line_color = (0, 255, 255)  # Cyan lines.
        config.drawing.point_color = (255, 0, 255)  # Magenta points.
        config.drawing.line_thickness = 2
        config.drawing.point_radius = 10

        grapher = Graphing(config=config)

        # --- Graph Creation ---
        # The algorithm creates the graph structure, which we can then analyze and draw.
        print(f"Creating k-nearest neighbors graph with K={K}...")
        # Note: A->B being a nearest neighbor doesn't mean B->A is. The graph
        # is initially "directed". For analysis, we treat connections as two-way.
        knn_graph = create_k_nearest_graph(particle_stack, k=K, aspect="array")

        # --- Analysis ---
        info = grapher.get_graph_info(knn_graph)
        print("K-Nearest Neighbors Graph Statistics:")
        print(f"  - Vertices: {info['vertex_count']}")
        print(f"  - Edges: {info['edge_count']}")
        print(f"  - Density: {info['density']:.4f}")
        print(f"  - Is Connected: {info['is_connected']}")
        if info['average_path_length']:
            print(f"  - Average Path Length: {info['average_path_length']:.4f}")

        # Verify the connectivity. The average number of connections should be around 2*K
        # because if A is connected to B, the analysis tools treat B as connected to A.
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
# EXAMPLE 4: GRAPH COMPARISON
# =============================================================================

def example_graph_comparison(delaunay_graph, proximity_graph, knn_graph):
    """
    Compares the statistics of the three generated graphs side-by-side.

    This helps to highlight the different structural properties that result
    from each graph creation algorithm.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 4: GRAPH COMPARISON")
    print("=" * 50)

    # Skip if any of the previous examples failed.
    if not all([delaunay_graph, proximity_graph, knn_graph]):
        print("Skipping comparison because one or more graphs are missing.")
        return

    try:
        # We only need the Graphing object for its analysis methods.
        grapher = Graphing()

        graphs = {
            "Delaunay": delaunay_graph,
            "Proximity": proximity_graph,
            "K-Nearest": knn_graph
        }

        # Print a formatted table header.
        print(f"{'Property':<20} {'Delaunay':<12} {'Proximity':<12} {'K-Nearest':<12}")
        print("-" * 60)

        # Loop through key statistics and print a row for each.
        for prop in ['vertex_count', 'edge_count', 'density', 'is_connected']:
            # Format the property name (e.g., 'vertex_count' -> 'Vertex Count').
            row = f"{prop.replace('_', ' ').title():<20}"
            for name, graph in graphs.items():
                info = grapher.get_graph_info(graph)
                value = info[prop]
                # Format floats and strings differently for alignment.
                if isinstance(value, float):
                    row += f"{value:<12.3f}"
                else:
                    row += f"{str(value):<12}"
            print(row)

        # Compare connectivity distributions.
        print("\nConnectivity Analysis (Number of connections per node):")
        for name, graph in graphs.items():
            connections = grapher.get_connections_per_object(graph)
            avg_conn = np.mean(list(connections.values()))
            max_conn = max(connections.values())
            min_conn = min(connections.values())

            print(f"  - {name}:")
            print(f"    - Average: {avg_conn:.2f}")
            print(f"    - Range (Min-Max): {min_conn} - {max_conn}")

    except Exception as e:
        print(f"Comparison failed: {e}")


# =============================================================================
# EXAMPLE 5: CONFIGURATION SHOWCASE
# =============================================================================

def example_configuration_showcase():
    """
    Shows how to easily create multiple graphs with different visual styles.

    This demonstrates the flexibility of the GraphizyConfig object for
    customizing the final rendered image.
    """
    print("\n" + "=" * 50)
    print(" EXAMPLE 5: CONFIGURATION SHOWCASE")
    print("=" * 50)

    try:
        # Use a smaller dataset for this quick visual showcase.
        IMAGE_WIDTH, IMAGE_HEIGHT = 400, 400
        NUM_PARTICLES = 30

        positions = generate_positions(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_PARTICLES)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Define a list of different style settings.
        # This is a clean way to manage multiple configurations.
        styles = [
            {
                "name": "classic_dark",
                "line_color": (0, 255, 0),  # Green
                "point_color": (0, 0, 255),  # Blue
                "line_thickness": 1,
                "point_radius": 6
            },
            {
                "name": "bold_neon",
                "line_color": (255, 0, 255),  # Magenta
                "point_color": (0, 255, 255),  # Cyan
                "line_thickness": 3,
                "point_radius": 12
            },
            {
                "name": "minimal_grayscale",
                "line_color": (128, 128, 128),  # Gray
                "point_color": (64, 64, 64),  # Dark gray
                "line_thickness": 1,
                "point_radius": 4
            }
        ]

        output_dir = setup_output_directory()

        # Loop through each defined style.
        for style in styles:
            print(f"Creating and saving '{style['name']}' style graph...")

            # Create a new config and apply the style settings from our list.
            config = GraphizyConfig()
            config.graph.dimension = (IMAGE_WIDTH, IMAGE_HEIGHT)
            config.drawing.line_color = style["line_color"]
            config.drawing.point_color = style["point_color"]
            config.drawing.line_thickness = style["line_thickness"]
            config.drawing.point_radius = style["point_radius"]

            # Create a grapher with this specific config.
            grapher = Graphing(config=config)
            # We'll use a Delaunay graph for all style examples.
            graph = grapher.make_delaunay(particle_stack)

            # Draw the graph and save it with a descriptive filename.
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
    Runs all the basic usage examples in a sequence.

    This function acts as the main controller for the script.
    """
    print("=" * 60)
    print("Starting Graphizy Basic Usage Examples")
    print("=" * 60)

    try:
        # --- Run Examples Sequentially ---
        # The output from the first example (particle_stack) is used as input
        # for the subsequent examples.
        delaunay_graph, particle_stack = example_delaunay_triangulation()
        proximity_graph = example_proximity_graph(particle_stack)
        knn_graph = example_k_nearest_neighbors(particle_stack)

        # --- Compare and Finalize ---
        # After creating all graphs, run the comparison.
        example_graph_comparison(delaunay_graph, proximity_graph, knn_graph)

        # Finally, run the styling showcase.
        example_configuration_showcase()

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print(f"Check the 'examples/output/' folder for the generated images.")

    except Exception as e:
        # A general catch-all for any unexpected errors.
        print(f"\nAn unexpected error occurred during the execution: {e}")
        # Print the full error traceback for debugging.
        import traceback
        traceback.print_exc()
        return 1  # Return a non-zero exit code to indicate failure.

    return 0  # Return 0 to indicate success.


# This is a standard Python construct.
# It ensures that the `main()` function is called only when this script is
# executed directly (e.g., `python your_script_name.py`), not when it's
# imported as a module into another script.
if __name__ == "__main__":
    sys.exit(main())