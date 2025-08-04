#!/usr/bin/env python3
"""
Graph Metrics and Analysis Examples for Graphizy - Updated for v0.1.17+

This script demonstrates comprehensive graph analysis using the latest Graphizy API.
It covers:

1. Fundamental graph metrics using modern get_graph_info() method
2. Advanced node-level centrality analysis with resilient methods
3. Component analysis and connectivity handling
4. Direct igraph integration with updated call_method_safe() and call_method_brutal()
5. Comparative analysis across all supported graph types (including MST and Gabriel)
6. Memory and weight system integration examples
7. Creating graphs with custom node attributes.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
import logging
from pathlib import Path
import sys
from typing import Dict, Any
from collections import Counter
import random

from graphizy import (
    Graphing, GraphizyConfig, generate_and_format_positions,
    validate_graphizy_input
)
from graphizy.utils import setup_output_directory

# Setup logging for informative output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def create_comprehensive_sample_graphs() -> (Dict[str, Any], Graphing, np.ndarray):
    """
    Generates a comprehensive set of graph types using the modern API.
    Updated to include all available graph types in Graphizy v0.1.17+.
    """
    print("\n" + "=" * 60)
    print("CREATING COMPREHENSIVE SAMPLE GRAPHS FOR ANALYSIS")
    print("=" * 60)

    # Generate common set of 2D points
    WIDTH, HEIGHT = 600, 600
    NUM_PARTICLES = 50
    particle_stack = generate_and_format_positions(WIDTH, HEIGHT, NUM_PARTICLES)
    validate_graphizy_input(particle_stack)

    # Initialize grapher with modern configuration
    config = GraphizyConfig(dimension=(WIDTH, HEIGHT))
    grapher = Graphing(config=config)

    # Create comprehensive set of graph types using modern unified API
    graphs = {}
    print("Creating various graph types using make_graph() interface...")

    try:
        # Geometric graphs
        print("  • Creating Delaunay triangulation...")
        graphs['delaunay'] = grapher.make_graph("delaunay", particle_stack)

        print("  • Creating Gabriel graph...")
        graphs['gabriel'] = grapher.make_graph("gabriel", particle_stack)

        # Proximity-based graphs
        print("  • Creating dense proximity graph...")
        graphs['proximity_dense'] = grapher.make_graph("proximity", particle_stack,
                                                      proximity_thresh=100.0)

        print("  • Creating sparse proximity graph...")
        graphs['proximity_sparse'] = grapher.make_graph("proximity", particle_stack,
                                                       proximity_thresh=60.0)

        # Fixed-degree graphs
        print("  • Creating K-nearest neighbors graph...")
        graphs['knn'] = grapher.make_graph("knn", particle_stack, k=4)

        # Minimal connectivity
        print("  • Creating minimum spanning tree...")
        graphs['mst'] = grapher.make_graph("mst", particle_stack, metric="euclidean")

        # Count successful creations
        successful = sum(1 for g in graphs.values() if g is not None)
        print(f"Successfully created {successful}/{len(graphs)} graph types.")

        return graphs, grapher, particle_stack

    except Exception as e:
        print(f"Error creating sample graphs: {e}")
        # Filter out failed graphs
        graphs = {k: v for k, v in graphs.items() if v is not None}
        return graphs, grapher, particle_stack


def example_modern_basic_metrics(graphs: Dict[str, Any], grapher: Graphing):
    """
    Demonstrates the modern get_graph_info() API for comprehensive graph analysis.
    """
    print("\n" + "=" * 60)
    print("MODERN BASIC GRAPH METRICS")
    print("=" * 60)
    print("Using the enhanced get_graph_info() method for unified analysis.")

    # Enhanced table with more metrics
    header_line = f"{'Graph Type':<18} {'Vertices':<9} {'Edges':<7} {'Density':<8} {'Connected':<10} {'Components':<11} {'Avg Path':<9}"
    print(f"\n{header_line}")
    print("-" * len(header_line))

    for name, graph in graphs.items():
        if graph is None:
            continue

        # Modern comprehensive analysis
        info = grapher.get_graph_info(graph)
        connectivity_info = grapher.get_connectivity_info(graph)

        # Format values for display
        density = info.get('density', 0.0)
        avg_path = info.get('average_path_length')
        avg_path_str = f"{avg_path:.2f}" if avg_path is not None else "N/A"

        print(f"{name:<18} {info.get('vertex_count', 0):<9} "
              f"{info.get('edge_count', 0):<7} {density:<8.3f} "
              f"{str(info.get('is_connected', False)):<10} "
              f"{connectivity_info['num_components']:<11} {avg_path_str:<9}")

    print("\nInsights:")
    print("  • MST has exactly n-1 edges (minimal connectivity)")
    print("  • Gabriel is typically sparser than Delaunay")
    print("  • Proximity graphs vary greatly with threshold")
    print("  • KNN provides controlled vertex degree")


def example_advanced_connectivity_analysis(graphs: Dict[str, Any], grapher: Graphing):
    """
    Demonstrates advanced connectivity analysis using modern resilient methods.
    """
    print("\n" + "=" * 60)
    print("ADVANCED CONNECTIVITY ANALYSIS")
    print("=" * 60)
    print("Using call_method_safe() for robust analysis of disconnected graphs.")

    for name, graph in graphs.items():
        if graph is None:
            continue

        print(f"\n--- {name.upper()} GRAPH ANALYSIS ---")

        # Comprehensive connectivity analysis
        connectivity_info = grapher.get_connectivity_info(graph)

        print(f"  Basic connectivity:")
        print(f"    • Connected: {connectivity_info['is_connected']}")
        print(f"    • Components: {connectivity_info['num_components']}")

        if connectivity_info['num_components'] > 1:
            print(f"    • Largest component: {connectivity_info['largest_component_size']} vertices")
            print(f"    • Connectivity ratio: {connectivity_info['connectivity_ratio']:.1%}")
            print(f"    • Isolated vertices: {connectivity_info['isolation_ratio']:.1%}")

        # Resilient path analysis - handles disconnected graphs gracefully
        diameter = grapher.call_method_safe(graph, 'diameter',
                                          component_mode="largest",
                                          default_value=None)
        if diameter is not None:
            print(f"    • Diameter (largest component): {diameter}")

        avg_path = grapher.call_method_safe(graph, 'average_path_length',
                                          component_mode="largest",
                                          default_value=None)
        if avg_path is not None:
            print(f"    • Avg path length (largest component): {avg_path:.3f}")

        # Global clustering analysis
        transitivity = grapher.call_method_safe(graph, 'transitivity_undirected',
                                              default_value=0.0)
        print(f"    • Global clustering coefficient: {transitivity:.3f}")


def example_advanced_centrality_measures(graphs: Dict[str, Any], grapher: Graphing):
    """
    Demonstrates centrality analysis using the robust call_method_safe() API.
    """
    print("\n" + "=" * 60)
    print("ADVANCED CENTRALITY MEASURES")
    print("=" * 60)
    print("Using call_method_safe() for resilient centrality computation.")

    # Analyze Delaunay graph in detail (typically well-connected)
    delaunay_graph = graphs.get('delaunay')
    if delaunay_graph is None:
        print("No Delaunay graph available for detailed analysis.")
        return

    print(f"\n--- DETAILED CENTRALITY ANALYSIS: DELAUNAY GRAPH ---")

    # 1. Degree Centrality (always works)
    print("\n• Degree Centrality:")
    degrees = grapher.call_method_safe(delaunay_graph, 'degree', "dict")
    degree_values = list(degrees.values()) if isinstance(degrees, dict) else []

    if degree_values:
        print(f"    Average: {np.mean(degree_values):.2f}")
        print(f"    Range: {min(degree_values)} - {max(degree_values)}")

        # Degree distribution
        degree_dist = Counter(degree_values)
        print(f"    Distribution: {dict(sorted(degree_dist.items()))}")

    # 2. Betweenness Centrality (handles disconnected graphs)
    print("\n• Betweenness Centrality:")
    betweenness = grapher.call_method_safe(delaunay_graph, 'betweenness',
                                         "dict", component_mode="all",
                                         default_value=0.0)

    if isinstance(betweenness, dict) and betweenness:
        sorted_bet = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        print(f"    Top 5 bridges: {[(item[0], f'{item[1]:.3f}') for item in sorted_bet[:5]]}")
        print(f"    Average: {np.mean(list(betweenness.values())):.3f}")

    # 3. Closeness Centrality (robust handling of disconnected components)
    print("\n• Closeness Centrality:")
    closeness = grapher.call_method_safe(delaunay_graph, 'closeness',
                                       "dict", component_mode="connected_only",
                                       default_value=0.0)

    if isinstance(closeness, dict) and closeness:
        sorted_close = sorted(closeness.items(), key=lambda x: x[1], reverse=True)
        print(f"    Top 5 central: {[(item[0], f'{item[1]:.3f}') for item in sorted_close[:5]]}")

    # 4. Comparative centrality analysis across graph types
    print(f"\n--- CENTRALITY COMPARISON ACROSS GRAPH TYPES ---")

    centrality_comparison = {}
    for name, graph in graphs.items():
        if graph is None:
            continue

        # Compute average betweenness for each graph type
        bet_centrality = grapher.call_method_safe(graph, 'betweenness',
                                                component_mode="all",
                                                default_value=0.0)

        if isinstance(bet_centrality, dict):
            avg_betweenness = np.mean(list(bet_centrality.values()))
            centrality_comparison[name] = avg_betweenness

    print("Average Betweenness Centrality by Graph Type:")
    for name, avg_bet in sorted(centrality_comparison.items(),
                               key=lambda x: x[1], reverse=True):
        print(f"    {name:<18}: {avg_bet:.4f}")


def example_direct_igraph_integration(graphs: Dict[str, Any], grapher: Graphing):
    """
    Demonstrates the enhanced igraph integration with call_method_brutal().
    """
    print("\n" + "=" * 60)
    print("DIRECT IGRAPH INTEGRATION")
    print("=" * 60)
    print("Using call_method_brutal() for advanced igraph method access.")

    # Use MST for predictable structure
    mst_graph = graphs.get('mst')
    if mst_graph is None:
        print("No MST graph available for analysis.")
        return

    print(f"\n--- ADVANCED IGRAPH METHODS: MST GRAPH ---")

    try:
        # 1. Degree sequence analysis with different return formats
        print("\n• Degree Sequence Analysis:")

        # Get as dict (auto-formatted)
        degree_dict = grapher.call_method_brutal(mst_graph, 'degree', "dict")
        print(f"    Degree as dict (first 5): {dict(list(degree_dict.items())[:5])}")

        # Get as raw list
        degree_list = grapher.call_method_brutal(mst_graph, 'degree', "list")
        print(f"    Degree as list: {degree_list[:10]}...")

        # Degree distribution analysis
        degree_counts = Counter(degree_list)
        print(f"    Degree distribution: {dict(sorted(degree_counts.items()))}")

        # 2. Advanced graph properties
        print("\n• Advanced Graph Properties:")

        # Edge connectivity (minimum cut)
        try:
            edge_connectivity = grapher.call_method_brutal(mst_graph, 'edge_connectivity', "raw")
            print(f"    Edge connectivity: {edge_connectivity}")
        except:
            print(f"    Edge connectivity: Not computed (requires connected graph)")

        # Radius and diameter
        try:
            radius = grapher.call_method_brutal(mst_graph, 'radius', "raw")
            diameter = grapher.call_method_brutal(mst_graph, 'diameter', "raw")
            print(f"    Radius: {radius}, Diameter: {diameter}")
        except:
            print(f"    Radius/Diameter: Not computed (disconnected graph)")

        # 3. Custom analysis using raw igraph access
        print("\n• Custom Analysis Examples:")

        # Find articulation points (cut vertices)
        try:
            articulation_points = grapher.call_method_brutal(mst_graph, 'articulation_points', "raw")
            if articulation_points:
                # Map back to original IDs
                original_ids = [mst_graph.vs[idx]["id"] for idx in articulation_points]
                print(f"    Articulation points (original IDs): {original_ids}")
            else:
                print(f"    No articulation points found")
        except Exception as e:
            print(f"    Articulation points analysis failed: {e}")

        # Edge betweenness for community detection
        try:
            edge_betweenness = grapher.call_method_brutal(mst_graph, 'edge_betweenness', "list")
            if edge_betweenness:
                max_eb_idx = np.argmax(edge_betweenness)
                max_eb_value = edge_betweenness[max_eb_idx]
                print(f"    Highest edge betweenness: {max_eb_value:.3f} (edge {max_eb_idx})")
        except Exception as e:
            print(f"    Edge betweenness analysis failed: {e}")

    except Exception as e:
        print(f"Advanced igraph integration failed: {e}")


def example_graph_type_performance_comparison(graphs: Dict[str, Any], grapher: Graphing):
    """
    Compares computational and structural properties across all graph types.
    """
    print("\n" + "=" * 60)
    print("COMPREHENSIVE GRAPH TYPE COMPARISON")
    print("=" * 60)

    comparison_data = {}

    # Collect comprehensive metrics for each graph type
    for name, graph in graphs.items():
        if graph is None:
            continue

        print(f"Analyzing {name}...")

        # Basic metrics
        info = grapher.get_graph_info(graph)
        connectivity_info = grapher.get_connectivity_info(graph)

        # Advanced metrics with error handling
        transitivity = grapher.call_method_safe(graph, 'transitivity_undirected',
                                              default_value=0.0)

        betweenness_dict = grapher.call_method_safe(graph, 'betweenness',
                                                  component_mode="all",
                                                  default_value=0.0)
        avg_betweenness = (np.mean(list(betweenness_dict.values()))
                          if isinstance(betweenness_dict, dict) else 0.0)

        comparison_data[name] = {
            'vertices': info.get('vertex_count', 0),
            'edges': info.get('edge_count', 0),
            'density': info.get('density', 0.0),
            'connected': info.get('is_connected', False),
            'components': connectivity_info['num_components'],
            'clustering': transitivity,
            'avg_betweenness': avg_betweenness,
            'connectivity_ratio': connectivity_info['connectivity_ratio']
        }

    # Display comprehensive comparison table
    if comparison_data:
        print("\nCOMPREHENSIVE COMPARISON TABLE:")

        metrics = [
            ('Vertices', 'vertices', '{}'),
            ('Edges', 'edges', '{}'),
            ('Density', 'density', '{:.3f}'),
            ('Connected', 'connected', '{}'),
            ('Components', 'components', '{}'),
            ('Clustering', 'clustering', '{:.3f}'),
            ('Avg Betweenness', 'avg_betweenness', '{:.4f}'),
            ('Connectivity %', 'connectivity_ratio', '{:.1%}')
        ]

        # Table header
        header = f"{'Metric':<15}" + "".join([f"{name:<12}" for name in comparison_data.keys()])
        print(header)
        print("-" * len(header))

        # Table rows
        for display_name, key, fmt in metrics:
            row = f"{display_name:<15}"
            for name in comparison_data.keys():
                value = comparison_data[name][key]
                try:
                    formatted_value = fmt.format(value)
                except:
                    formatted_value = str(value)
                row += f"{formatted_value:<12}"
            print(row)

    # Graph type recommendations
    print(f"\nGRAPH TYPE SELECTION GUIDE:")
    print(f"  • Delaunay: Optimal for mesh generation, always connected")
    print(f"  • Gabriel: Subset of Delaunay, geometric proximity constraints")
    print(f"  • Proximity: Local neighborhoods, adjustable with threshold")
    print(f"  • K-NN: Fixed degree networks, good for ML applications")
    print(f"  • MST: Minimal connectivity, tree structure (n-1 edges)")


def example_custom_node_attributes():
    """
    Demonstrates creating graphs with custom node attributes using the flexible
    `add_more` parameter in `generate_and_format_positions`.
    """
    print("\n" + "=" * 60)
    print("CREATING GRAPHS WITH CUSTOM NODE ATTRIBUTES")
    print("=" * 60)

    # 1. Define the extra attributes you want to generate using lambda functions
    extra_attributes = {
        'velocity': lambda: random.uniform(0, 5.0),
        'rotation': lambda: random.uniform(-np.pi, np.pi),
        'mass': lambda: random.choice([1.0, 2.0, 5.0])
    }
    print(f"Defined extra attributes: {list(extra_attributes.keys())}")

    # 2. Generate the data using the new `add_more` dictionary
    WIDTH, HEIGHT = 600, 600
    NUM_PARTICLES = 20
    particle_stack_custom = generate_and_format_positions(
        WIDTH, HEIGHT, NUM_PARTICLES,
        add_more=extra_attributes
    )
    # The resulting array will have columns: [id, x, y, velocity, rotation, mass]
    print(f"Generated data shape: {particle_stack_custom.shape}")


    # 3. IMPORTANT: Define the data_shape to match the generated data.
    # This tells Graphizy what each column means so it can create the attributes.
    data_shape = [
        ('id', int),
        ('x', float),
        ('y', float),
        ('velocity', float),
        ('rotation', float),
        ('mass', float)
    ]

    # 4. Initialize a new Graphing instance with the custom data_shape
    config = GraphizyConfig(dimension=(WIDTH, HEIGHT))
    # The key step is passing the data_shape to the Graphing constructor
    grapher_custom = Graphing(config=config, data_shape=data_shape)
    print("Initialized Graphing instance with custom data_shape.")

    # 5. Create a graph from the data with custom attributes
    # The data_interface will now correctly map all columns to vertex attributes.
    graph = grapher_custom.make_graph("delaunay", particle_stack_custom)
    print(f"Created a '{graph.summary()}' with custom attributes.")

    # 6. Access and verify the custom attributes on the graph vertices
    print("\nVerifying custom attributes on the graph:")
    if graph.vcount() > 0:
        first_vertex = graph.vs[0]
        print(f"  Vertex 0 attributes: {first_vertex.attributes()}")

        # Check if all attributes are present
        available_attrs = graph.vs.attributes()
        print(f"  All available vertex attributes: {available_attrs}")

        # Example: Analyze one of the custom attributes
        if 'mass' in available_attrs:
            masses = graph.vs['mass']
            print(f"  Average mass of all particles: {np.mean(masses):.2f}")

        if 'velocity' in available_attrs:
            velocities = graph.vs['velocity']
            print(f"  Max velocity: {np.max(velocities):.2f}")
    else:
        print("Graph has no vertices to inspect.")


def main():
    """Run all modern graph metrics examples."""
    print("Graphizy: Graph Metrics and Analysis Examples - v0.1.17+ Edition")
    print("=" * 70)

    try:
        # Create comprehensive sample graphs
        graphs, grapher, particle_stack = create_comprehensive_sample_graphs()

        # Run all analysis examples
        example_modern_basic_metrics(graphs, grapher)
        example_advanced_connectivity_analysis(graphs, grapher)
        example_advanced_centrality_measures(graphs, grapher)
        example_direct_igraph_integration(graphs, grapher)
        example_graph_type_performance_comparison(graphs, grapher)
        example_custom_node_attributes()

        # Create visualizations
        print("\nCreating visualizations...")
        output_dir = setup_output_directory()
        for name, graph in graphs.items():
            if graph:
                try:
                    image = grapher.draw_graph(graph)
                    grapher.save_graph(image, str(output_dir / f"metrics_graph_{name}.jpg"))
                except Exception as e:
                    logging.error(f"Failed to save {name}: {e}")

        print(f"Visualizations saved to '{output_dir}'.")

        print("\n" + "=" * 70)
        print("All examples completed successfully.")
        print("\nKey Features Demonstrated:")
        print("  • get_graph_info() for comprehensive metrics")
        print("  • call_method_safe() for resilient analysis")
        print("  • call_method_brutal() for advanced igraph access")
        print("  • Modern connectivity analysis methods")
        print("  • Support for all graph types (including MST and Gabriel)")
        print("  • Creation of graphs with custom node attributes")

    except Exception as e:
        print(f"\nExamples failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())