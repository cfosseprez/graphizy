# examples/2_graph_metrics.py

#!/usr/bin/env python3
"""
Graph Metrics and Analysis Examples for Graphizy - Modern API

This script demonstrates comprehensive graph analysis using the lazy-loading
`GraphAnalysisResult` object, which provides a clean, efficient, and powerful
interface for exploring graph properties.

This example covers:
1.  **Lazy-Loading Metrics**: How properties like `.density` or `.diameter` are
    computed on-demand, making analysis fast and responsive.
2.  **On-the-Fly Computation**: Using `.get_metric()` to compute any `igraph`
    metric that wasn't pre-defined.
3.  **Statistical Helpers**: Using `.get_top_n_by()` and `.get_metric_stats()`
    to quickly derive insights from graph data.
4.  **Resilient Analysis**: How the system gracefully handles disconnected graphs.
5.  **Custom Attributes**: Creating and analyzing graphs with custom node data.

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
import random

from graphizy import (
    Graphing, GraphizyConfig, generate_and_format_positions
)
from graphizy.utils import setup_output_directory

# Setup logging for informative output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def create_sample_graphs() -> (Dict[str, Any], Graphing):
    """Generates a set of sample graphs for analysis."""
    print("\n" + "=" * 60)
    print("STEP 1: CREATING SAMPLE GRAPHS FOR ANALYSIS")
    print("=" * 60)

    WIDTH, HEIGHT = 600, 600
    # A disconnected graph is more interesting for analysis
    particle_stack = np.array([
        # Component 1
        [0, 50, 50], [1, 100, 100], [2, 50, 150], [3, 150, 50],
        # Component 2
        [4, 400, 400], [5, 450, 450], [6, 400, 500],
        # Isolated node
        [7, 300, 300]
    ])

    config = GraphizyConfig(dimension=(WIDTH, HEIGHT))
    grapher = Graphing(config=config)

    graphs = {
        'delaunay': grapher.make_graph("delaunay", particle_stack),
        'proximity': grapher.make_graph("proximity", particle_stack, proximity_thresh=75.0)
    }
    print("Successfully created Delaunay and Proximity graphs.")
    return graphs, grapher


def example_lazy_loading_and_summary(graphs: Dict[str, Any], grapher: Graphing):
    """Demonstrates the lazy-loading `GraphAnalysisResult` object."""
    print("\n" + "=" * 60)
    print("STEP 2: LAZY-LOADING METRICS & AUTOMATIC SUMMARIES")
    print("=" * 60)

    delaunay_graph = graphs.get('delaunay')
    if not delaunay_graph:
        return

    # This call is INSTANTANEOUS. No metrics are computed yet.
    results = grapher.get_graph_info(delaunay_graph)
    print("Created GraphAnalysisResult object. (No computation performed yet)")

    # --- Accessing properties triggers computation ONCE ---
    print("\nAccessing properties computes them on-demand:")
    print(f"  - Accessing .is_connected...")
    is_connected = results.is_connected  # Computation happens here
    print(f"    > Result: {is_connected}")

    print(f"  - Accessing .is_connected again...")
    is_connected_cached = results.is_connected  # This is instant (from cache)
    print(f"    > Result: {is_connected_cached} (from cache)")

    # --- The summary() method provides a clean overview ---
    print("\nThe .summary() method gives a clean overview of key metrics:")
    print(results.summary())


def example_on_the_fly_analysis(graphs: Dict[str, Any], grapher: Graphing):
    """Demonstrates computing new metrics and using helper methods."""
    print("\n" + "=" * 60)
    print("STEP 3: ON-THE-FLY ANALYSIS & STATISTICAL HELPERS")
    print("=" * 60)

    proximity_graph = graphs.get('proximity')
    if not proximity_graph:
        return

    results = grapher.get_graph_info(proximity_graph)
    print("Analyzing Proximity Graph:")
    print(results.summary())

    # 1. Use get_top_n_by() for quick insights
    print("\n• Top 3 Hubs (by Degree):")
    top_degree = results.get_top_n_by('degree', n=3)
    print(f"    {top_degree}")

    print("\n• Top 3 Bridges (by Betweenness):")
    top_betweenness = results.get_top_n_by('betweenness', n=3)
    print(f"    {[(node, f'{val:.2f}') for node, val in top_betweenness]}")

    # 2. Use get_metric_stats() for statistical summaries
    print("\n• Statistical analysis of Degree:")
    degree_stats = results.get_metric_stats('degree')
    print(f"    Mean: {degree_stats['mean']:.2f}, Std: {degree_stats['std']:.2f}, "
          f"Range: [{degree_stats['min']}, {degree_stats['max']}]")

    # 3. Compute any other igraph metric using get_metric()
    print("\n• Pagerank (computed on-demand):")
    pagerank_values = results.get_metric('pagerank', return_format='dict')
    top_pagerank = sorted(pagerank_values.items(), key=lambda item: item[1], reverse=True)
    print(f"    Top 3 by Pagerank: {[(node, f'{val:.3f}') for node, val in top_pagerank[:3]]}")


def example_custom_node_attributes(grapher: Graphing):
    """Demonstrates creating graphs with custom node attributes."""
    print("\n" + "=" * 60)
    print("STEP 4: CREATING GRAPHS WITH CUSTOM NODE ATTRIBUTES")
    print("=" * 60)

    # 1. Define the extra attributes to generate
    extra_attributes = {
        'velocity': lambda: random.uniform(0, 5.0),
        'mass': lambda: random.choice([1.0, 2.0, 5.0])
    }
    print(f"Defined extra attributes: {list(extra_attributes.keys())}")

    # 2. Generate data with these attributes
    particle_stack_custom = generate_and_format_positions(
        600, 600, 20, add_more=extra_attributes
    )

    # 3. IMPORTANT: Define the data_shape to match the generated data
    data_shape = [
        ('id', int), ('x', float), ('y', float),
        ('velocity', float), ('mass', float)
    ]

    # 4. Initialize a new Graphing instance with the custom data_shape
    grapher_custom = Graphing(config=grapher.config, data_shape=data_shape)
    print("Initialized Graphing instance with custom data_shape.")

    # 5. Create a graph. The custom attributes are added automatically.
    graph = grapher_custom.make_graph("delaunay", particle_stack_custom)
    print(f"Created a '{graph.summary()}' with custom attributes.")

    # 6. Access and verify the custom attributes
    print("\nVerifying custom attributes on the graph:")
    if graph.vcount() > 0:
        first_vertex = graph.vs[0]
        print(f"  Vertex 0 attributes: {first_vertex.attributes()}")
        print(f"  All available vertex attributes: {graph.vs.attributes()}")

        # 7. Analyze the custom attributes using the result object
        results = grapher_custom.get_graph_info(graph)
        mass_stats = results.get_metric_stats('mass')
        print(f"\n  Mass statistics: Mean={mass_stats['mean']:.2f}, Max={mass_stats['max']:.2f}")


def main():
    """Run all modern graph metrics examples."""
    print("Graphizy: Modern Analysis with GraphAnalysisResult")
    print("=" * 70)

    try:
        graphs, grapher = create_sample_graphs()
        example_lazy_loading_and_summary(graphs, grapher)
        example_on_the_fly_analysis(graphs, grapher)
        example_custom_node_attributes(grapher)

        print("\n" + "=" * 70)
        print("All examples completed successfully.")

    except Exception as e:
        print(f"\nExamples failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())