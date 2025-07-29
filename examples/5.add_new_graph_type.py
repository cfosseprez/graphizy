#!/usr/bin/env python3
"""
Example 5: Adding Custom Graph Types to Graphizy

This example demonstrates how to extend Graphizy with custom graph types
using the plugin system. You can add new graph algorithms without modifying
the core library code.

Key Features Demonstrated:
- Plugin registration system using decorators
- Custom graph creation functions
- Parameter validation and documentation
- Integration with existing API
- Automatic discovery and listing
"""

import numpy as np
from graphizy import Graphing, graph_type_plugin
import random
import igraph as ig


@graph_type_plugin(
    name="random_edges",
    description="Randomly connects points with a specified probability",
    category="community",
    author="Random Graph Developer",
    version="1.0.0",
    parameters={
        "edge_probability": {
            "type": "float",
            "default": 0.3,
            "description": "Probability of connection between any two points (0.0-1.0)"
        },
        "seed": {
            "type": "int",
            "default": None,
            "description": "Random seed for reproducibility"
        }
    }
)
def create_random_graph(data_points: np.ndarray, dimension: tuple, **kwargs) -> ig.Graph:
    """
    Create a graph with random edges between points.

    This is a simple example plugin that demonstrates the plugin system.
    Each pair of points has a probability of being connected.

    Args:
        data_points: Point data as array with columns [id, x, y]
        dimension: The (width, height) of the graph canvas (part of the plugin interface).
        **kwargs: Additional parameters including:
                 - edge_probability: Probability of connection between any two points (default: 0.3)
                 - seed: Random seed for reproducibility (default: None)

    Returns:
        igraph Graph object with random connections
    """
    from graphizy.algorithms import create_graph_array

    # Get parameters with defaults
    edge_probability = kwargs.get('edge_probability', 0.3)
    seed = kwargs.get('seed', None)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create base graph structure
    graph = create_graph_array(data_points)
    n_vertices = len(graph.vs)

    # Add random edges
    edges_to_add = []
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if random.random() < edge_probability:
                edges_to_add.append((i, j))

    if edges_to_add:
        graph.add_edges(edges_to_add)

    return graph


@graph_type_plugin(
    name="star",
    description="Creates a star topology with one central hub",
    category="topology",
    author="Star Graph Expert",
    version="1.0.0",
    parameters={
        "center_id": {
            "type": "int",
            "default": None,
            "description": "ID of existing point to use as center"
        },
        "center_x": {
            "type": "float",
            "default": None,
            "description": "X coordinate for center point"
        },
        "center_y": {
            "type": "float",
            "default": None,
            "description": "Y coordinate for center point"
        },
        "add_center_node": {
            "type": "bool",
            "default": True,
            "description": "Whether to add a new center node"
        }
    }
)
def create_star_graph(data_points: np.ndarray, dimension: tuple, **kwargs) -> ig.Graph:
    """
    Create a star topology graph with one central hub.

    Connects all points to a single central point. The center can be:
    - An existing point (specified by center_id)
    - The geometric center of all points (default)
    - A specified coordinate (center_x, center_y)

    Args:
        data_points: Point data as array with columns [id, x, y]
        dimension: The (width, height) of the graph canvas (part of the plugin interface).
        **kwargs: Additional parameters including:
                 - center_id: ID of existing point to use as center (default: None)
                 - center_x, center_y: Coordinates for new center point (default: geometric center)
                 - add_center_node: Whether to add a new center node (default: True)

    Returns:
        igraph Graph object with star topology
    """
    from graphizy.algorithms import create_graph_array

    # Get parameters
    center_id = kwargs.get('center_id', None)
    center_x = kwargs.get('center_x', None)
    center_y = kwargs.get('center_y', None)
    add_center_node = kwargs.get('add_center_node', True)

    # Create base graph structure
    graph = create_graph_array(data_points)
    n_vertices = len(graph.vs)

    if center_id is not None:
        # Use existing point as center
        center_vertex_idx = graph.vs.find(id=center_id).index
        if center_vertex_idx is None:
            raise ValueError(f"Center ID {center_id} not found in data")

        # Connect all other vertices to the center
        edges_to_add = [(center_vertex_idx, i) for i in range(n_vertices) if i != center_vertex_idx]
        if edges_to_add:
            graph.add_edges(edges_to_add)

    elif add_center_node:
        # Add a new center node
        if center_x is None:
            center_x = np.mean(data_points[:, 1])
        if center_y is None:
            center_y = np.mean(data_points[:, 2])

        # Add center vertex
        new_id = np.max(data_points[:, 0]) + 1
        graph.add_vertex(id=new_id, x=center_x, y=center_y, name=str(new_id))
        center_vertex_idx = n_vertices

        # Connect all original vertices to the center
        edges_to_add = [(center_vertex_idx, i) for i in range(n_vertices)]
        graph.add_edges(edges_to_add)

    else:
        # Connect all vertices to the first vertex (no new center)
        center_vertex_idx = 0
        edges_to_add = [(center_vertex_idx, i) for i in range(1, n_vertices)]
        graph.add_edges(edges_to_add)

    return graph


@graph_type_plugin(
    name="grid",
    description="Creates grid-like connections between nearby points",
    category="geometric",
    author="Grid Graph Creator",
    version="1.0.0",
    parameters={
        "grid_spacing": {
            "type": "float",
            "default": None,
            "description": "Distance threshold for grid connections (auto-calculated if None)"
        },
        "include_diagonals": {
            "type": "bool",
            "default": False,
            "description": "Whether to connect diagonal neighbors"
        },
        "tolerance": {
            "type": "float",
            "default": 0.1,
            "description": "Tolerance factor for grid alignment (fraction of grid_spacing)"
        }
    }
)
def create_grid_graph(data_points: np.ndarray, dimension: tuple, **kwargs) -> ig.Graph:
    """
    Create a grid-like graph by connecting points to nearby grid neighbors.

    Arranges points in a grid pattern and connects each point to its
    immediate neighbors (up, down, left, right, and optionally diagonals).

    Args:
        data_points: Point data as array with columns [id, x, y]
        dimension: The (width, height) of the graph canvas (part of the plugin interface).
        **kwargs: Additional parameters including:
                 - grid_spacing: Distance threshold for grid connections (default: auto-calculated)
                 - include_diagonals: Whether to connect diagonal neighbors (default: False)
                 - tolerance: Tolerance for grid alignment (default: 10% of grid_spacing)

    Returns:
        igraph Graph object with grid-like connections
    """
    from graphizy.algorithms import create_graph_array
    from scipy.spatial.distance import pdist

    # Get parameters
    grid_spacing = kwargs.get('grid_spacing', None)
    include_diagonals = kwargs.get('include_diagonals', False)
    tolerance_factor = kwargs.get('tolerance', 0.1)

    # Create base graph structure
    graph = create_graph_array(data_points)
    positions = data_points[:, 1:3]  # x, y coordinates

    # Auto-calculate grid spacing if not provided
    if grid_spacing is None and len(positions) > 1:
        # Use the minimum non-zero distance as grid spacing
        distances = pdist(positions)
        non_zero_distances = distances[distances > 1e-9]
        if len(non_zero_distances) > 0:
            grid_spacing = np.min(non_zero_distances)
        else: # All points are at the same location
            grid_spacing = 1.0
    elif grid_spacing is None:
        grid_spacing = 1.0 # Default for single point case

    tolerance = grid_spacing * tolerance_factor

    # Find grid connections
    edges_to_add = []
    n_points = len(positions)

    for i in range(n_points):
        for j in range(i + 1, n_points):
            pos_i = positions[i]
            pos_j = positions[j]

            dx = abs(pos_i[0] - pos_j[0])
            dy = abs(pos_i[1] - pos_j[1])

            # Check for horizontal/vertical neighbors
            is_horizontal = (dy <= tolerance) and (abs(dx - grid_spacing) <= tolerance)
            is_vertical = (dx <= tolerance) and (abs(dy - grid_spacing) <= tolerance)

            # Check for diagonal neighbors (if enabled)
            is_diagonal = False
            if include_diagonals:
                is_diagonal = (abs(dx - grid_spacing) <= tolerance) and \
                              (abs(dy - grid_spacing) <= tolerance)

            if is_horizontal or is_vertical or is_diagonal:
                edges_to_add.append((i, j))

    if edges_to_add:
        graph.add_edges(edges_to_add)

    return graph


def main():
    """Demonstrate adding custom graph types to Graphizy."""

    print("Demonstrating easy graph type extension in Graphizy")
    print("=" * 60)

    # NOTE: Registration is now handled automatically by the @graph_type_plugin decorators
    # when this file is imported. No need for manual registration calls here.

    # Create a Graphing instance
    grapher = Graphing(dimension=(400, 400))

    # List all available graph types (including our new ones)
    print("Available graph types:")
    all_types = grapher.list_graph_types()
    for name, info in all_types.items():
        print(f"  • {name}: {info['description']}")
        if 'category' in info and info['category'] != 'built-in':
            print(f"    └─ Category: {info['category']}, Author: {info.get('author', 'Unknown')}")

    print("=" * 60)
    print("Creating graphs with new types:")

    # Generate sample data
    np.random.seed(42)
    n_points = 20
    data = np.column_stack([
        np.arange(n_points),  # IDs
        np.random.rand(n_points) * 300 + 50,  # X coordinates
        np.random.rand(n_points) * 300 + 50  # Y coordinates
    ])

    try:
        # Test the random edges graph
        print("\nCreating random edges graph...")
        random_graph = grapher.make_graph("random_edges", data, edge_probability=0.4, seed=42)
        print(f"   Random graph: {random_graph.vcount()} vertices, {random_graph.ecount()} edges")

        # Test the star graph
        print("\nCreating star graph...")
        star_graph = grapher.make_graph("star", data, center_x=200, center_y=200)
        print(f"   Star graph: {star_graph.vcount()} vertices, {star_graph.ecount()} edges")

        # Test the grid graph
        print("\nCreating grid graph...")
        # Create a more grid-like dataset for better demonstration
        grid_data = np.array([
            [i, (i % 5) * 50 + 50, (i // 5) * 50 + 50]
            for i in range(20)
        ])
        grid_graph = grapher.make_graph("grid", grid_data, grid_spacing=50, include_diagonals=True)
        print(f"   Grid graph: {grid_graph.vcount()} vertices, {grid_graph.ecount()} edges")

        # Test with built-in type for comparison
        print("\nCreating Delaunay graph for comparison...")
        delaunay_graph = grapher.make_graph("delaunay", data)
        print(f"   Delaunay graph: {delaunay_graph.vcount()} vertices, {delaunay_graph.ecount()} edges")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Get detailed info about our new graph types:")

    try:
        # Get info about our custom graph types using the correct method
        star_info = Graphing.get_plugin_info("star")
        print(f"\nStar Graph Info:")
        print(f"   Description: {star_info['info']['description']}")
        print(f"   Category: {star_info['info']['category']}")
        print(f"   Parameters: {list(star_info['parameters'].keys())}")

        random_info = Graphing.get_plugin_info("random_edges")
        print(f"\nRandom Edges Graph Info:")
        print(f"   Description: {random_info['info']['description']}")
        print(f"   Parameters: {list(random_info['parameters'].keys())}")

        grid_info = Graphing.get_plugin_info("grid")
        print(f"\nGrid Graph Info:")
        print(f"   Description: {grid_info['info']['description']}")
        print(f"   Parameters: {list(grid_info['parameters'].keys())}")

    except Exception as e:
        print(f"Could not get plugin info: {e}")

    print("\n" + "=" * 60)
    print("Success! New graph types added with minimal code!")
    print("Key Benefits:")
    print(" - No core files modified")
    print(" - Automatic parameter validation")
    print(" - Discoverable through list_graph_types()")
    print(" - Integrated documentation")
    print(" - Same API as built-in types")
    print(" - Easy to distribute as separate packages")

    # Optional: Save visualizations
    try:
        print("\nSaving visualizations...")

        # Draw and save some graphs
        random_image = grapher.draw_graph(random_graph)
        grapher.save_graph(random_image, "random_edges_example.png")
        print("   Saved random_edges_example.png")

        star_image = grapher.draw_graph(star_graph)
        grapher.save_graph(star_image, "star_graph_example.png")
        print("   Saved star_graph_example.png")

        grid_image = grapher.draw_graph(grid_graph)
        grapher.save_graph(grid_image, "grid_graph_example.png")
        print("   Saved grid_graph_example.png")

    except Exception as e:
        print(f"Could not save visualizations: {e}")


if __name__ == "__main__":
    main()