#!/usr/bin/env python3
"""
Advanced Memory Graph Examples for Graphizy

This script demonstrates advanced use cases for Graphizy's MemoryManager,
focusing on how to build and analyze temporal graphs where connections
evolve over time.

Key features demonstrated:
1.  Updating memory from heterogeneous sources (proximity, Delaunay, custom logic).
2.  Visualizing edge age using color and alpha gradients.
3.  Implementing and integrating custom connection-defining functions.
4.  Performing comparative analysis of different memory-building strategies.
"""

from graphizy import Graphing
import numpy as np
import time
import sys

# It's good practice to wrap optional dependencies in a try-except block.
try:
    from scipy.spatial.distance import cdist
except ImportError:
    print("Warning: SciPy not found. Some custom functions may not work.")
    cdist = None

def example_delaunay_memory():
    """Demonstrates memory updates using Delaunay triangulation over time."""
    print("=== Delaunay Memory Example ===")
    
    # Initialize the main Graphing object and its associated MemoryManager.
    # `track_edge_ages=True` is essential for temporal visualizations.
    # `max_iterations=10` means the memory has a sliding window of 10 steps.
    grapher = Graphing(dimension=(800, 600))
    grapher.init_memory_manager(max_memory_size=100, max_iterations=10, track_edge_ages=True)
    
    # Define initial particle positions.
    positions = np.array([
        [1, 100.0, 100.0], [2, 300.0, 100.0], [3, 500.0, 100.0],
        [4, 200.0, 250.0], [5, 400.0, 250.0], [6, 100.0, 400.0],
        [7, 300.0, 400.0], [8, 500.0, 400.0],
    ], dtype=float)
    
    print(f"Initial positions: {len(positions)} objects")
    
    # This loop simulates a dynamic system over 8 time steps.
    for iteration in range(8):
        # Apply small random movements to simulate particle drift.
        if iteration > 0:
            movement = np.random.normal(0, 15, (len(positions), 2))
            positions[:, 1:3] += movement
            positions[:, 1] = np.clip(positions[:, 1], 50, 750)
            positions[:, 2] = np.clip(positions[:, 2], 50, 550)
        
        # This is the core memory update step. It computes a Delaunay triangulation
        # for the current positions and adds the resulting edges to the memory.
        current_connections = grapher.update_memory_with_delaunay(positions)
        
        # Calculate and report the number of unique connections found in this step.
        total_connections = sum(len(conns) for conns in current_connections.values()) // 2
        print(f"Iteration {iteration + 1}: Found {total_connections} Delaunay connections.")
    
    # `make_memory_graph` compiles all connections stored in memory into a single graph object.
    memory_graph = grapher.make_memory_graph(positions)
    
    # `get_memory_analysis` provides aggregate statistics about the memory's state.
    analysis = grapher.get_memory_analysis()
    print(f"Final Analysis: {analysis}")
    
    # `draw_memory_graph` is a specialized drawing function that can use edge ages
    # to modify visual properties, creating a temporal heatmap of connections.
    image = grapher.draw_memory_graph(memory_graph, use_age_colors=True)
    grapher.save_graph(image, "delaunay_memory_aged.jpg")
    grapher.show_graph(image, "Delaunay Memory (Age-based colors)")
    
    return grapher, memory_graph


def example_mixed_memory_sources():
    """Demonstrates accumulating memory from different logical sources."""
    print("\n=== Mixed Memory Sources Example ===")
    
    grapher = Graphing(dimension=(600, 600))
    grapher.init_memory_manager(max_memory_size=50, max_iterations=15, track_edge_ages=True)
    
    positions = np.array([
        [1, 150.0, 150.0], [2, 450.0, 150.0], [3, 300.0, 300.0], 
        [4, 150.0, 450.0], [5, 450.0, 450.0], [6, 300.0, 100.0],
    ], dtype=float)
    
    # Phase 1: Use proximity-based connections for the first 3 iterations.
    print("\n--- Phase 1: Proximity-based Memory ---")
    for i in range(3):
        positions[:, 1:3] += np.random.normal(0, 20, (len(positions), 2))
        grapher.update_memory_with_proximity(positions, proximity_thresh=150)
    
    # Phase 2: Switch to Delaunay-based connections for the next 3 iterations.
    print("\n--- Phase 2: Delaunay-based Memory ---")
    for i in range(3):
        positions[:, 1:3] += np.random.normal(0, 15, (len(positions), 2))
        grapher.update_memory_with_delaunay(positions)
    
    # Phase 3: Define a custom connection function for k-nearest neighbors.
    # This function must accept a positions array and return a graph-like object.
    def k_nearest_connector(positions, k=2):
        """Custom connector that finds the k-nearest neighbors for each object."""
        from graphizy.algorithms import create_graph_array
        if not cdist:
             raise ImportError("SciPy is required for this custom function.")

        # `create_graph_array` initializes an empty graph structure from positions.
        graph = create_graph_array(positions)
        pos_2d = positions[:, 1:3]
        
        # Use SciPy's cdist for efficient distance calculation.
        distances = cdist(pos_2d, pos_2d)
        edges = []
        
        for i, row in enumerate(distances):
            # Find indices of the k nearest points, excluding the point itself (index 0).
            nearest_indices = np.argsort(row)[1:k+1]
            for j in nearest_indices:
                # Use a sorted tuple to represent an undirected edge.
                edge = tuple(sorted((int(positions[i, 0]), int(positions[j, 0]))))
                edges.append(edge)
        
        if edges:
            graph.add_edges(list(set(edges)))
        return graph.get_connections_dict()

    print("\n--- Phase 3: Custom K-Nearest Neighbors Memory ---")
    for i in range(3):
        positions[:, 1:3] += np.random.normal(0, 10, (len(positions), 2))
        # `update_memory_with_custom` executes the provided function and updates memory.
        grapher.update_memory_with_custom(positions, k_nearest_connector, k=2)

    # Final analysis and visualization.
    memory_graph = grapher.make_memory_graph(positions)
    analysis = grapher.get_memory_analysis()
    
    print("\nFinal analysis of mixed sources:")
    print(f"  Total memory connections: {analysis['total_connections']}")
    print(f"  Objects with memory: {analysis['objects_with_memory']}")
    if 'edge_age_stats' in analysis:
        stats = analysis['edge_age_stats']
        print(f"  Edge Age Stats -> Min: {stats['min_age']}, Max: {stats['max_age']}, Avg: {stats['avg_age']:.2f}")
    
    # Draw both a standard and an age-tinted version to compare.
    normal_image = grapher.draw_graph(memory_graph)
    aged_image = grapher.draw_memory_graph(memory_graph, use_age_colors=True, alpha_range=(0.2, 1.0))
    
    grapher.save_graph(normal_image, "mixed_memory_normal.jpg")
    grapher.save_graph(aged_image, "mixed_memory_aged.jpg")
    
    grapher.show_graph(aged_image, "Mixed Memory (Age-based)")
    
    return grapher, memory_graph


def compare_memory_strategies():
    """Performs a side-by-side comparison of different memory strategies."""
    print("\n=== Memory Strategy Comparison ===")
    
    base_positions = np.array([
        [1, 100.0, 100.0], [2, 200.0, 120.0], [3, 300.0, 110.0],
        [4, 150.0, 200.0], [5, 250.0, 210.0], [6, 180.0, 300.0],
    ], dtype=float)
    
    # Define the strategies to be tested.
    strategies = {
        "Proximity": lambda g, p: g.update_memory_with_proximity(p, proximity_thresh=100),
        "Delaunay": lambda g, p: g.update_memory_with_delaunay(p),
    }
    
    results = {}
    
    for name, update_func in strategies.items():
        print(f"\n--- Testing Strategy: {name} ---")
        
        grapher = Graphing(dimension=(400, 400))
        grapher.init_memory_manager(max_memory_size=40, max_iterations=6, track_edge_ages=True)
        positions = base_positions.copy()
        
        # Run a short simulation for this strategy.
        for iteration in range(5):
            # Using a seed ensures the random movements are identical for each strategy,
            # allowing for a fair comparison.
            np.random.seed(42 + iteration)
            movement = np.random.normal(0, 15, (len(positions), 2))
            positions[:, 1:3] += movement
            
            update_func(grapher, positions)
        
        # Store the final state for analysis and visualization.
        results[name] = {
            "graph": grapher.make_memory_graph(positions),
            "analysis": grapher.get_memory_analysis(),
            "grapher": grapher
        }
        print(f"  Final State -> Connections: {results[name]['analysis']['total_connections']}, Avg Age: {results[name]['analysis']['edge_age_stats']['avg_age']:.2f}")

    # Visualize the final memory state of each strategy.
    print("\n--- Visualizing Comparison Results ---")
    for name, result in results.items():
        image = result["grapher"].draw_memory_graph(result["graph"], use_age_colors=True)
        filename = f"comparison_{name.lower()}_memory.jpg"
        result["grapher"].save_graph(image, filename)
        print(f"Saved {name} visualization to {filename}")
        result["grapher"].show_graph(image, f"{name} Memory Strategy")
        time.sleep(0.5)
    
    return results


def main():
    """Main function to orchestrate and run all examples."""
    print("Running Advanced Memory Graph Examples")
    print("=" * 50)
    
    try:
        example_delaunay_memory()
        example_mixed_memory_sources()
        # The custom function example from the original prompt can be added here if desired.
        compare_memory_strategies()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully.")
        print("Check the generated .jpg files for visualizations.")
        
    except Exception as e:
        print(f"\nAn error occurred during the examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()