#!/usr/bin/env python3
"""
Example 5: Adding Custom Graph Types to Graphizy - Updated for v0.1.17+

This example demonstrates the enhanced plugin system for adding custom graph types.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
import random
import sys
import igraph as ig

from graphizy import (
    Graphing, GraphizyConfig, graph_type_plugin,
    generate_and_format_positions, validate_graphizy_input
)
from graphizy.utils import setup_output_directory

random.seed(42)
np.random.seed(42)


@graph_type_plugin(
    name="enhanced_random",
    description="Advanced random graph with configurable clustering and distance decay",
    category="probabilistic",
    author="Graphizy Example Developer",
    version="2.0.0",
    parameters={
        "edge_probability": {
            "type": "float",
            "default": 0.3,
            "description": "Base probability of connection (0.0-1.0)",
            "min": 0.0,
            "max": 1.0
        },
        "clustering_factor": {
            "type": "float", 
            "default": 1.0,
            "description": "Clustering enhancement factor (>1.0 increases clustering)",
            "min": 0.1,
            "max": 5.0
        },
        "distance_decay": {
            "type": "float",
            "default": 0.0,
            "description": "Distance-based probability decay (0.0 = no decay)",
            "min": 0.0,
            "max": 1.0
        },
        "seed": {
            "type": "int",
            "default": None,
            "description": "Random seed for reproducibility"
        }
    }
)
def create_enhanced_random_graph(data_points: np.ndarray, dimension: tuple, **kwargs) -> ig.Graph:
    """Create an enhanced random graph with clustering and distance-based probability."""
    from graphizy.algorithms import create_graph_array

    # Extract parameters with validation
    edge_probability = max(0.0, min(1.0, kwargs.get('edge_probability', 0.3)))
    clustering_factor = max(0.1, min(5.0, kwargs.get('clustering_factor', 1.0)))
    distance_decay = max(0.0, min(1.0, kwargs.get('distance_decay', 0.0)))
    seed = kwargs.get('seed', None)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Create base graph structure
    graph = create_graph_array(data_points)
    n_vertices = len(graph.vs)
    
    if n_vertices < 2:
        return graph

    # Extract positions for distance calculations
    positions = data_points[:, 1:3]
    max_distance = np.sqrt(dimension[0]**2 + dimension[1]**2)

    # Basic random connections with distance decay
    edges_to_add = []
    
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            # Calculate connection probability
            base_prob = edge_probability
            
            # Apply distance decay if enabled
            if distance_decay > 0:
                distance = np.linalg.norm(positions[i] - positions[j])
                normalized_distance = distance / max_distance
                distance_modifier = np.exp(-distance_decay * normalized_distance * 5)
                connection_prob = base_prob * distance_modifier
            else:
                connection_prob = base_prob
            
            # Random connection decision
            if random.random() < connection_prob:
                edges_to_add.append((i, j))

    # Clustering enhancement
    if clustering_factor > 1.0 and len(edges_to_add) > 0:
        temp_graph = graph.copy()
        if edges_to_add:
            temp_graph.add_edges(edges_to_add)
        
        # Find triangulation opportunities
        clustering_edges = []
        for i in range(n_vertices):
            neighbors_i = set(temp_graph.neighbors(i))
            
            for j in neighbors_i:
                neighbors_j = set(temp_graph.neighbors(j))
                common_neighbors = neighbors_i & neighbors_j
                
                for k in common_neighbors:
                    if not temp_graph.are_connected(i, k):
                        triangle_prob = (clustering_factor - 1.0) * 0.1
                        if random.random() < triangle_prob:
                            clustering_edges.append((min(i, k), max(i, k)))
        
        clustering_edges = list(set(clustering_edges))
        edges_to_add.extend(clustering_edges)

    # Remove duplicates and add all edges
    edges_to_add = list(set(edges_to_add))
    if edges_to_add:
        graph.add_edges(edges_to_add)

    return graph


@graph_type_plugin(
    name="geometric_star",
    description="Advanced star topology with geometric optimization",
    category="topology",
    author="Graphizy Topology Expert",
    version="2.0.0",
    parameters={
        "hub_selection": {
            "type": "str",
            "default": "centroid",
            "description": "Hub selection method: 'centroid', 'geometric_median', 'random'",
            "choices": ["centroid", "geometric_median", "random"]
        },
        "create_hub": {
            "type": "bool",
            "default": True,
            "description": "Whether to create a new hub node or use existing point"
        },
        "spokes_only": {
            "type": "bool",
            "default": True,
            "description": "If True, only hub-to-node connections"
        },
        "peripheral_probability": {
            "type": "float",
            "default": 0.1,
            "description": "Probability of peripheral connections (if spokes_only=False)",
            "min": 0.0,
            "max": 1.0
        }
    }
)
def create_geometric_star_graph(data_points: np.ndarray, dimension: tuple, **kwargs) -> ig.Graph:
    """Create an advanced geometric star graph with intelligent hub placement."""
    from graphizy.algorithms import create_graph_array

    # Parameter extraction and validation
    hub_selection = kwargs.get('hub_selection', 'centroid')
    create_hub = kwargs.get('create_hub', True)
    spokes_only = kwargs.get('spokes_only', True)
    peripheral_prob = max(0.0, min(1.0, kwargs.get('peripheral_probability', 0.1)))

    # Create base graph
    graph = create_graph_array(data_points)
    n_vertices = len(graph.vs)
    
    if n_vertices == 0:
        return graph

    positions = data_points[:, 1:3]
    
    # Determine hub location and index
    hub_vertex_idx = None
    hub_position = None

    if hub_selection == 'centroid':
        # Geometric centroid
        hub_position = np.mean(positions, axis=0)
        
    elif hub_selection == 'geometric_median':
        # Geometric median (more robust to outliers)
        hub_position = calculate_geometric_median(positions)
        
    elif hub_selection == 'random':
        # Random existing point
        hub_vertex_idx = random.randint(0, n_vertices - 1)
        hub_position = positions[hub_vertex_idx]
        create_hub = False
    
    # Create hub vertex if needed
    if create_hub and hub_position is not None:
        new_id = np.max(data_points[:, 0]) + 1
        graph.add_vertex(id=new_id, x=hub_position[0], y=hub_position[1], name=str(new_id))
        hub_vertex_idx = n_vertices  # Index of newly added vertex
        n_vertices += 1

    # Create spoke connections
    spoke_edges = []
    
    for i in range(n_vertices):
        if i == hub_vertex_idx:
            continue
        spoke_edges.append((hub_vertex_idx, i))

    # Add spoke edges
    if spoke_edges:
        graph.add_edges(spoke_edges)

    # Add peripheral connections if requested
    if not spokes_only and peripheral_prob > 0:
        peripheral_edges = []
        
        for i in range(n_vertices):
            if i == hub_vertex_idx:
                continue
                
            for j in range(i + 1, n_vertices):
                if j == hub_vertex_idx:
                    continue
                    
                if random.random() < peripheral_prob:
                    peripheral_edges.append((i, j))
        
        if peripheral_edges:
            graph.add_edges(peripheral_edges)

    return graph


def calculate_geometric_median(points: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """Calculate the geometric median of a set of points using Weiszfeld's algorithm."""
    if len(points) == 0:
        return np.array([0.0, 0.0])
    if len(points) == 1:
        return points[0].copy()
    
    # Initialize with centroid
    median = np.mean(points, axis=0)
    tolerance = 1e-6
    
    for _ in range(max_iterations):
        distances = np.linalg.norm(points - median, axis=1)
        
        # Avoid division by zero
        non_zero_mask = distances > tolerance
        if not np.any(non_zero_mask):
            break
        
        weights = np.zeros_like(distances)
        weights[non_zero_mask] = 1.0 / distances[non_zero_mask]
        
        if np.sum(weights) == 0:
            break
            
        new_median = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
        
        # Check convergence
        if np.linalg.norm(new_median - median) < tolerance:
            break
            
        median = new_median
    
    return median


def main():
    """Demonstrate the enhanced custom graph types and plugin system."""
    output_dir = setup_output_directory()

    print("Demonstrating Enhanced Custom Graph Types in Graphizy v0.1.17+")
    print("=" * 70)

    # Create a Graphing instance
    config = GraphizyConfig(dimension=(500, 500))
    grapher = Graphing(config=config)

    # List all available graph types (including new ones)
    print("\nAvailable graph types after plugin registration:")
    all_types = grapher.list_graph_types()
    
    builtin_types = []
    custom_types = []
    
    for name, info in all_types.items():
        if hasattr(info, 'category') and info.category != 'built-in':
            custom_types.append((name, info))
        else:
            builtin_types.append((name, info))

    print(f"\nBuilt-in graph types ({len(builtin_types)}):")
    for name, info in builtin_types:
        print(f"  • {name}: {info.description}")

    print(f"\nCustom graph types ({len(custom_types)}):")
    for name, info in custom_types:
        author = getattr(info, 'author', 'Unknown')
        version = getattr(info, 'version', '1.0.0')
        print(f"  • {name}: {info.description}")
        print(f"    └─ Author: {author}, Version: {version}, Category: {info.category}")

    # Generate sample data sets for testing
    print("\n" + "=" * 70)
    print("TESTING CUSTOM GRAPH TYPES")
    print("=" * 70)

    # Test data
    test_data = generate_and_format_positions(500, 500, 30)

    # Test each custom graph type
    custom_graph_tests = [
        ("enhanced_random", {"edge_probability": 0.4, "clustering_factor": 2.0, "distance_decay": 0.3}),
        ("geometric_star", {"hub_selection": "geometric_median", "spokes_only": False, "peripheral_probability": 0.2})
    ]

    print(f"\nTesting custom graph types:")

    for graph_type, params in custom_graph_tests:
        try:
            print(f"\n• Creating {graph_type} graph...")
            
            # Create graph using the unified interface
            graph = grapher.make_graph(graph_type, test_data, **params)
            
            if graph:
                # Analyze results
                info = grapher.get_graph_info(graph)
                
                print(f"    Success: {info['vertex_count']} vertices, {info['edge_count']} edges")
                print(f"    Density: {info['density']:.3f}, Connected: {info['is_connected']}")
                
                # Save visualization
                try:
                    image = grapher.draw_graph(graph)
                    filename = f"{graph_type}_example.jpg"
                    grapher.save_graph(image, str(output_dir / filename))
                    print(f"    Saved: {filename}")
                except Exception as viz_error:
                    print(f"    Visualization failed: {viz_error}")
            
            else:
                print(f"    Failed: No graph created")
                
        except Exception as e:
            print(f"    Error: {e}")

    # Plugin information demonstration
    print(f"\n" + "=" * 70)
    print("DETAILED PLUGIN INFORMATION")
    print("=" * 70)

    for graph_type, _ in custom_graph_tests:
        try:
            plugin_info = Graphing.get_plugin_info(graph_type)
            
            print(f"\n• {graph_type.replace('_', ' ').title()} Plugin:")
            print(f"  Description: {plugin_info['info']['description']}")
            print(f"  Category: {plugin_info['info']['category']}")
            print(f"  Author: {plugin_info['info']['author']}")
            print(f"  Version: {plugin_info['info']['version']}")
            
            print(f"  Parameters:")
            for param_name, param_info in plugin_info['parameters'].items():
                default_val = param_info.get('default', 'None')
                param_type = param_info.get('type', 'unknown')
                description = param_info.get('description', 'No description')
                
                print(f"    - {param_name} ({param_type}): {description}")
                print(f"      Default: {default_val}")

        except Exception as e:
            print(f"  Error getting plugin info for {graph_type}: {e}")

    print("\n" + "=" * 70)
    print("SUCCESS! Enhanced custom graph types demonstrated!")
    print("=" * 70)
    
    print("\nKey Enhancements Demonstrated:")
    print("  • Advanced parameter validation and documentation")
    print("  • Sophisticated geometric and probabilistic algorithms")
    print("  • Compatibility with memory and weight systems")
    print("  • Comprehensive plugin information and metadata")
    print("  • Same API as built-in graph types")


if __name__ == "__main__":
    main()
