#!/usr/bin/env python3
"""
Graph Metrics and Analysis Examples for Graphizy

This script demonstrates how to compute and interpret various graph metrics
using the Graphizy library. It covers:

1.  Fundamental graph metrics (size, density, connectivity).
2.  Node-level centrality analysis (degree, betweenness, closeness).
3.  Advanced structural properties (clustering, assortativity).
4.  Direct access to the underlying igraph library for custom analysis.
5.  Comparative analysis of metrics across different graph generation methods.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict, Any

from graphizy import (
    Graphing, GraphizyConfig, generate_positions,
)
# This utility helps create an 'output' directory if it doesn't exist.
from graphizy.utils import setup_output_directory

# Setup logging for informative output.
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')


def create_sample_graphs() -> (Dict[str, Any], Graphing, np.ndarray):
    """
    Generates a set of distinct graph types from the same point cloud.
    This provides a basis for comparing how generation methods affect graph properties.
    """
    print("\n" + "=" * 60)
    print("CREATING SAMPLE GRAPHS FOR ANALYSIS")
    print("=" * 60)

    # Generate a common set of 2D points (nodes).
    WIDTH, HEIGHT = 600, 600
    NUM_PARTICLES = 40
    positions = generate_positions(WIDTH, HEIGHT, NUM_PARTICLES)
    particle_ids = np.arange(len(positions))
    particle_stack = np.column_stack((particle_ids, positions))

    # Initialize a Graphing instance to create graphs.
    config = GraphizyConfig()
    config.graph.dimension = (WIDTH, HEIGHT)
    grapher = Graphing(config=config)

    # Create a dictionary to hold different graph types.
    graphs = {}
    print("Creating various graph types from the same set of nodes...")

    # Delaunay graphs are planar and tend to be sparse but well-connected.
    graphs['delaunay'] = grapher.make_graph(graph_type="delaunay", data_points=particle_stack)

    # Proximity graphs connect nodes based on a distance threshold.
    # A larger threshold creates a denser graph.
    graphs['proximity_dense'] = grapher.make_graph(graph_type="proximity", data_points=particle_stack, proximity_thresh=100.0)
    # A smaller threshold creates a sparser graph, which may be disconnected.
    graphs['proximity_sparse'] = grapher.make_graph(graph_type="proximity", data_points=particle_stack, proximity_thresh=60.0)

    # K-Nearest Neighbor graphs connect each node to its 'k' closest neighbors.
    try:
        from graphizy.algorithms import create_knn_graph
        graphs['knn'] = create_knn_graph(particle_stack, k=4, aspect="array")
    except ImportError:
        print("K-nearest graph requires additional dependencies (e.g., scikit-learn). Skipping.")
        graphs['knn'] = None

    return graphs, grapher, particle_stack


def example_basic_metrics(graphs: Dict[str, Any], grapher: Graphing):
    """Demonstrates computing the most fundamental graph metrics."""
    print("\n" + "=" * 60)
    print("BASIC GRAPH METRICS")
    print("=" * 60)
    print("These metrics provide a high-level overview of a graph's size and structure.")

    # Table header for the metrics.
    print(f"\n{'Graph Type':<20} {'Vertices':<10} {'Edges':<8} {'Density':<10} {'Connected':<10}")
    print("-" * 65)

    for name, graph in graphs.items():
        if graph is None: continue

        # The `get_graph_info` method is a high-level API call that conveniently
        # computes and returns a dictionary of common graph metrics.
        info = grapher.get_graph_info(graph)

        # Density: Ratio of actual edges to the total number of possible edges.
        # Indicates how "complete" or "crowded" the graph is.
        density = info.get('density', 0.0)

        print(
            f"{name:<20} {info.get('vertex_count', 0):<10} {info.get('edge_count', 0):<8} {density:<10.3f} {str(info.get('is_connected', 'N/A')):<10}")

    print("\nInsight: Notice how the 'proximity_dense' graph has the highest edge count and density,")
    print("while the 'delaunay' and 'knn' graphs are much sparser.")


def example_connectivity_analysis(graphs: Dict[str, Any], grapher: Graphing):
    """
    Demonstrates metrics related to graph connectivity and traversal.
    """
    print("\n" + "=" * 60)
    print("CONNECTIVITY ANALYSIS")
    print("=" * 60)
    print("These metrics describe how well connected the graph is and the typical distances between nodes.")

    for name, graph in graphs.items():
        if graph is None: continue
        print(f"\n--- Analysis for: {name.upper()} ---")

        # Use the high-level API for quick stats.
        info = grapher.get_graph_info(graph)
        print(f"  Is Connected: {info.get('is_connected')}")
        # Diameter: The longest shortest path between any two nodes in the graph.
        if info.get('diameter') is not None:
            print(f"  Diameter: {info['diameter']}")
        # Average Path Length: The average distance between all pairs of nodes.
        if info.get('average_path_length') is not None:
            print(f"  Avg. Path Length: {info['average_path_length']:.3f}")

        # For more detailed analysis, use `call_method_raw` to access igraph functions directly.
        # This provides the raw output from the underlying library.
        components = grapher.call_method_raw(graph, 'connected_components')
        num_components = len(components)
        print(f"  Components: {num_components}")

        # If the graph is disconnected, analyze its largest component.
        if num_components > 1:
            giant_component_size = max(len(c) for c in components)
            print(
                f"  Giant Component Size: {giant_component_size} of {graph.vcount()} nodes ({giant_component_size / graph.vcount():.1%})")

        # Global Clustering Coefficient (Transitivity): Measures the tendency of nodes to cluster together.
        # A high value means friends of a node are also likely to be friends with each other.
        transitivity = grapher.get_graph_info(graph).get('transitivity', 'N/A')
        if isinstance(transitivity, float):
            print(f"  Global Clustering: {transitivity:.3f}")


def example_centrality_measures(graphs: Dict[str, Any], grapher: Graphing):
    """
    Demonstrates various centrality measures to identify important nodes.
    """
    print("\n" + "=" * 60)
    print("CENTRALITY MEASURES")
    print("=" * 60)
    print("Centrality metrics help identify the most influential or critical nodes in a network.")

    # We will perform a detailed analysis on one representative graph.
    name = 'delaunay'
    graph = graphs.get(name)
    if graph is None: return
    print(f"\n--- Detailed centrality analysis for '{name.upper()}' graph ---")

    # 1. Degree Centrality: The number of connections a node has.
    print("\n- Degree (Number of connections):")
    degrees = grapher.call_method_safe(graph, 'degree')
    degree_values = list(degrees.values())
    print(f"  Average: {np.mean(degree_values):.2f}, Range: {min(degree_values)}-{max(degree_values)}")

    # 2. Betweenness Centrality: Measures how often a node lies on the shortest path between other nodes.
    # High betweenness indicates a "bridge" node that connects disparate parts of the graph.
    print("\n- Betweenness (Bridge-like role):")
    # `call_method_safe()` is a helper that returns a user-friendly dictionary mapping node IDs to their values.
    betweenness = grapher.call_method_safe(graph, 'betweenness')
    sorted_bet = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top 3 nodes: {[(item[0], f'{item[1]:.2f}') for item in sorted_bet[:3]]}")

    # 3. Closeness Centrality: Measures the average distance from a node to all other nodes.
    # A high value means the node is, on average, "close" to all others and can spread information efficiently.
    print("\n- Closeness (Ease of reaching other nodes):")
    if grapher.call_method_safe(graph, 'is_connected'):
        closeness = grapher.call_method_safe(graph, 'closeness')
        sorted_close = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top 3 nodes: {[(item[0], f'{item[1]:.3f}') for item in sorted_close[:3]]}")
    else:
        print("  Skipped (graph is not fully connected).")


def example_direct_igraph_usage(graphs: Dict[str, Any], grapher: Graphing):
    """
    Demonstrates how to use the direct igraph interface for custom analysis.
    This is for advanced use cases where a specific metric is not available
    through the high-level `graphizy` API.
    """
    print("\n" + "=" * 60)
    print("DIRECT IGRAPH INTERFACE USAGE")
    print("=" * 60)
    print("Use `call_method_raw` to access any method from the underlying igraph object.")

    name = 'delaunay'
    graph = graphs.get(name)
    if graph is None: return
    print(f"\n--- Direct analysis examples using '{name.upper()}' graph ---")

    # Example: Perform a degree distribution analysis.
    print("\n- Analyzing degree distribution:")
    # `call_method_raw` returns the direct, unprocessed output from the igraph function.
    degree_sequence = grapher.call_method_raw(graph, 'degree')

    from collections import Counter
    degree_counts = Counter(degree_sequence)
    print(f"  Degree counts: {dict(sorted(degree_counts.items()))}")

    # Example: Custom analysis to find nodes that are local degree maxima.
    print("\n- Custom Analysis: Find local degree maxima")
    local_maxima_nodes = []
    for node_id in range(graph.vcount()):
        node_degree = degree_sequence[node_id]
        neighbor_ids = grapher.call_method_raw(graph, 'neighbors', node_id)
        if not neighbor_ids: continue

        neighbor_degrees = [degree_sequence[nid] for nid in neighbor_ids]
        if all(node_degree >= n_degree for n_degree in neighbor_degrees):
            local_maxima_nodes.append(node_id)
    print(f"  Nodes with degree >= all neighbors: {local_maxima_nodes}")


def example_comparing_graph_types(graphs: Dict[str, Any], grapher: Graphing):
    """Generates a summary table to compare metrics across all graph types."""
    print("\n" + "=" * 60)
    print("COMPARING GRAPH TYPES")
    print("=" * 60)

    comparison_data = {}
    metrics_to_compare = [
        ('Vertices', 'vertex_count'), ('Edges', 'edge_count'), ('Density', 'density'),
        ('Connected', 'is_connected'), ('Avg Path Len', 'average_path_length'), ('Clustering', 'transitivity')
    ]

    for name, graph in graphs.items():
        if graph is None: continue
        comparison_data[name] = grapher.get_graph_info(graph)

    # Display comparison table
    header = f"{'Metric':<15}" + "".join([f"{name:<18}" for name in comparison_data.keys()])
    print(header)
    print("-" * len(header))

    for display_name, metric_key in metrics_to_compare:
        row = f"{display_name:<15}"
        for name in comparison_data.keys():
            value = comparison_data.get(name, {}).get(metric_key)
            if value is None:
                row += f"{'N/A':<18}"
            elif isinstance(value, bool):
                row += f"{str(value):<18}"
            elif isinstance(value, (int, np.integer)):
                row += f"{value:<18}"
            else:
                row += f"{value:<18.3f}"
        print(row)


def main():
    """Run all graph metrics examples."""
    print("Graphizy: Graph Metrics and Analysis Examples")
    print("=" * 70)

    try:
        graphs, grapher, particle_stack = create_sample_graphs()

        example_basic_metrics(graphs, grapher)
        example_connectivity_analysis(graphs, grapher)
        example_centrality_measures(graphs, grapher)
        example_direct_igraph_usage(graphs, grapher)
        example_comparing_graph_types(graphs, grapher)

        # Create visualizations for context.
        print("\nCreating visualizations of the sample graphs...")
        output_dir = setup_output_directory()
        for name, graph in graphs.items():
            if graph:
                image = grapher.draw_graph(graph)
                grapher.save_graph(image, str(output_dir / f"metrics_graph_{name}.jpg"))
        print(f"Visualizations saved to '{output_dir}'.")

        print("\n" + "=" * 70)
        print("All examples completed successfully.")
        print("\nKey Takeaways:")
        print("  - `get_graph_info()` is for a quick, high-level summary.")
        print("  - `call_method()` is for node-wise metrics (e.g., centrality).")
        print("  - `call_method_raw()` gives you direct access to the full power of `igraph`.")

    except Exception as e:
        print(f"\nExamples failed with an error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())