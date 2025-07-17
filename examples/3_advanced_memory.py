#!/usr/bin/env python3
"""
Advanced Memory Examples for the Graphizy Package

This script demonstrates the "memory" capabilities of Graphizy, which allow for
tracking graph connections over time. It covers:

1.  How to configure and use the MemoryManager to track connections.
2.  Different strategies for updating memory (e.g., from proximity or Delaunay graphs).
3.  How to visualize memory, including "aging" effects where old connections fade.
4.  Analyzing memory statistics to understand network evolution.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

# --- Core Libraries ---
import numpy as np
import logging
import sys

# --- Graphizy Imports ---
# Import the main components from the graphizy package.
from graphizy import (
    Graphing,  # The main class for graphing and analysis.
    GraphizyConfig,  # The configuration object for settings and styles.
    MemoryManager,  # The core class for tracking connections over time.
    generate_positions,  # A helper function to create random 2D points.
    create_memory_graph,  # A function to build a renderable graph from memory data.
    GraphizyError  # Custom error class for package-specific issues.
)

# This utility helps create an 'output' directory if it doesn't exist.
from graphizy.utils import setup_output_directory

# --- Initial Setup ---
# Configure logging to show informative messages as the script runs.
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')


# =============================================================================
# EXAMPLE 1: BASIC MEMORY MANAGER
# =============================================================================

def example_basic_memory_manager():
    """
    Introduces the basic functionality of the MemoryManager.

    Think of the MemoryManager like human memory. It can be configured to have
    a long-term memory (remembering everything) or a short-term memory
    (only remembering the most recent events). This example compares both types.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: BASIC MEMORY MANAGER")
    print("=" * 60)

    try:
        # --- Configuration ---
        # We will create two different memory managers to see how their settings affect them.
        print("Creating memory managers with different configurations...")

        # This manager is like a long-term memory. It has a large capacity (max_memory_size)
        # and no limit on how many past iterations it remembers (max_iterations=None).
        # `track_edge_ages=True` tells it to count how "old" each connection is.
        basic_mgr = MemoryManager(
            max_memory_size=6,
            max_iterations=None,  # `None` means unlimited iterations.
            track_edge_ages=True
        )

        # This manager represents a short-term memory. It has a smaller capacity
        # and will only remember the connections from the last 3 updates.
        limited_mgr = MemoryManager(
            max_memory_size=3,
            max_iterations=3,  # Only keeps data from the last 3 calls to `add_connections`.
            track_edge_ages=True
        )

        # --- Simulation ---
        # We'll simulate a network changing over 5 time steps. Each dictionary
        # represents the active connections at that moment.
        scenarios = [
            {"A": ["B", "C"], "B": ["A"], "C": ["A", "D"], "D": ["C"]},
            {"A": ["B"], "B": ["A", "C"], "C": ["B", "D"], "D": ["C"]},
            {"A": ["D"], "B": ["C"], "C": ["B"], "D": ["A", "C"]},
            {"A": ["B", "C"], "B": ["A"], "C": ["A"], "D": []},
            {"A": ["B"], "B": ["A", "C", "D"], "C": ["B"], "D": ["B"]}
        ]

        print(f"\nSimulating {len(scenarios)} connection scenarios over time...")

        for i, connections in enumerate(scenarios):
            print(f"\nProcessing Iteration {i + 1}: Current connections are {connections}")

            # The `add_connections` method is the core function to update the memory.
            # It adds new connections and updates the age of existing ones.
            basic_mgr.add_connections(connections)
            limited_mgr.add_connections(connections)

            # Let's check the state of each manager after the update.
            # Notice how the limited manager's connection count might be capped.
            basic_stats = basic_mgr.get_memory_stats()
            limited_stats = limited_mgr.get_memory_stats()

            print(f"   - Basic manager (long-term) total connections: {basic_stats['total_connections']}")
            print(f"   - Limited manager (short-term) total connections: {limited_stats['total_connections']}")

        # --- Final Analysis ---
        # At the end, we can see the final state of both memories.
        print("\nFinal Memory Analysis:")
        print("Basic Manager (unlimited iterations):")
        final_basic = basic_mgr.get_memory_stats()
        for key, value in final_basic.items():
            print(f"   - {key}: {value}")

        print("\nLimited Manager (remembers last 3 iterations):")
        final_limited = limited_mgr.get_memory_stats()
        for key, value in final_limited.items():
            print(f"   - {key}: {value}")

        return basic_mgr, limited_mgr

    except Exception as e:
        print(f"Basic memory manager example failed: {e}")
        return None, None


# =============================================================================
# EXAMPLE 2: MEMORY UPDATE STRATEGIES
# =============================================================================

def example_memory_update_strategies():
    """
    Shows how to automatically update memory from spatial data.

    Instead of manually defining connections, we can use Graphizy's built-in
    graph generation methods (like proximity or Delaunay) to find connections
    and feed them directly into the memory manager.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MEMORY UPDATE STRATEGIES")
    print("=" * 60)

    try:
        # --- Data Generation ---
        WIDTH, HEIGHT = 400, 400
        NUM_PARTICLES = 25

        positions = generate_positions(WIDTH, HEIGHT, NUM_PARTICLES)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Create separate memory managers to test each strategy independently.
        strategies = {
            "proximity": MemoryManager(max_memory_size=20, track_edge_ages=True),
            "delaunay": MemoryManager(max_memory_size=20, track_edge_ages=True)
        }

        # Initialize the main Graphing object.
        config = GraphizyConfig()
        config.graph.dimension = (WIDTH, HEIGHT)
        grapher = Graphing(config=config)

        print("Testing different strategies for updating memory from spatial data...")

        results = {}

        # --- Strategy 1: Proximity-based Memory ---
        # Connections are based on how close points are to each other.
        print("\nTesting Proximity-based memory updates:")
        grapher.memory_manager = strategies["proximity"]  # Assign the manager to the grapher.
        # This helper function finds all points within 80.0 pixels of each other
        # and automatically updates the memory manager.
        grapher.update_memory_with_proximity(particle_stack, proximity_thresh=80.0)
        results["proximity"] = grapher.get_memory_stats()
        print(f"   Memory updated. Total connections found: {results['proximity']['total_connections']}")

        # --- Strategy 2: Delaunay-based Memory ---
        # Connections are based on the Delaunay triangulation of the points.
        print("\nTesting Delaunay-based memory updates:")
        grapher.memory_manager = strategies["delaunay"]
        # This function computes the Delaunay triangulation and updates memory with the edges.
        grapher.update_memory_with_delaunay(particle_stack, dimension=(WIDTH, HEIGHT))
        results["delaunay"] = grapher.get_memory_stats()
        print(f"   Memory updated. Total connections found: {results['delaunay']['total_connections']}")

        # --- Comparison ---
        print("\nStrategy Comparison:")
        print(f"{'Strategy':<15} {'Connections':<12} {'Objects':<8} {'Avg/Object':<12}")
        print("-" * 50)

        for strategy, stats in results.items():
            avg_per_object = stats['total_connections'] / stats['total_objects'] if stats['total_objects'] > 0 else 0
            print(
                f"{strategy:<15} {stats['total_connections']:<12} {stats['total_objects']:<8} {avg_per_object:<12.2f}")

        # --- Visualization ---
        # Now, let's visualize the memory state created by each strategy.
        output_dir = setup_output_directory()
        colors = {"proximity": (255, 0, 0), "delaunay": (0, 255, 0)}  # Red for prox, Green for delaunay

        for strategy_name, memory_mgr in strategies.items():
            try:
                # `create_memory_graph` converts the raw memory data into a graph
                # object that Graphizy knows how to draw.
                memory_graph = create_memory_graph(
                    particle_stack,
                    memory_mgr.get_current_memory_graph(),
                    aspect="array"
                )

                # Configure the drawing style for this visualization.
                config.drawing.line_color = colors[strategy_name]
                config.drawing.point_color = (255, 255, 255)  # White points on a black background.
                config.drawing.line_thickness = 2
                config.drawing.point_radius = 6

                grapher_viz = Graphing(config=config)

                # Draw the graph and save it to a file.
                image = grapher_viz.draw_graph(memory_graph)
                filename = f"memory_strategy_{strategy_name}.jpg"
                grapher_viz.save_graph(image, str(output_dir / filename))
                print(f"   {strategy_name.capitalize()} visualization saved to {filename}")

            except Exception as viz_error:
                print(f"   Could not create {strategy_name} visualization: {viz_error}")

        return results

    except Exception as e:
        print(f"Memory update strategies example failed: {e}")
        return None


# =============================================================================
# EXAMPLE 3: MEMORY WITH AGING EFFECTS
# =============================================================================

def example_memory_with_aging():
    """
    Demonstrates visualizing memory with "aging" effects.

    This example simulates particles moving over several time steps. As we
    update the memory, old connections that are no longer present will "age".
    We can then create a visualization where older connections appear more
    faded, giving a sense of the history of the network.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: MEMORY AGING VISUALIZATION")
    print("=" * 60)

    try:
        # --- Setup ---
        # This manager will track connections over 8 time steps.
        memory_mgr = MemoryManager(max_memory_size=15, max_iterations=8, track_edge_ages=True)

        # Create our initial set of particles.
        WIDTH, HEIGHT = 600, 600
        NUM_PARTICLES = 20
        positions = generate_positions(WIDTH, HEIGHT, NUM_PARTICLES)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Setup the Graphing object and assign the memory manager to it.
        config = GraphizyConfig()
        config.graph.dimension = (WIDTH, HEIGHT)
        grapher = Graphing(config=config)
        grapher.memory_manager = memory_mgr

        output_dir = setup_output_directory()

        # --- Simulation Loop ---
        print("Simulating memory evolution over 6 time steps...")
        for iteration in range(6):
            print(f"\nProcessing Time Step {iteration + 1}:")

            # In each step after the first, we move the particles slightly.
            if iteration > 0:
                movement = np.random.normal(0, 15, (len(particle_stack), 2))
                particle_stack[:, 1:3] += movement
                # Ensure particles stay within the canvas boundaries.
                particle_stack[:, 1] = np.clip(particle_stack[:, 1], 10, WIDTH - 10)
                particle_stack[:, 2] = np.clip(particle_stack[:, 2], 10, HEIGHT - 10)

            # Update the memory based on the new particle positions.
            current_connections = grapher.update_memory_with_proximity(particle_stack, proximity_thresh=120.0)

            total_current = sum(len(conns) for conns in current_connections.values()) // 2
            print(f"   - Connections detected in this step: {total_current}")

            stats = memory_mgr.get_memory_stats()
            print(f"   - Total connections in memory: {stats['total_connections']}")

            # Every two iterations, we'll save a picture of the memory.
            if iteration % 2 == 1:
                # First, create a graph object from the current memory state.
                memory_graph_current = grapher.make_memory_graph(particle_stack)

                # Now, draw the graph using the special memory drawing function.
                # `use_age_colors=True` enables the aging effect.
                # `alpha_range` controls the transparency: newer edges will be fully opaque (1.0),
                # while the oldest edges will be nearly transparent (0.3).
                try:
                    aging_image = grapher.draw_memory_graph(
                        memory_graph_current,
                        use_age_colors=True,
                        alpha_range=(0.3, 1.0)  # From faded to fully visible.
                    )

                    output_path = output_dir / f"memory_aging_step_{iteration + 1}.jpg"
                    grapher.save_graph(aging_image, str(output_path))
                    print(f"   Aging visualization saved to {output_path}")

                except Exception as draw_error:
                    print(f"   Could not create aging visualization: {draw_error}")

    except Exception as e:
        print(f"Memory aging example failed: {e}")


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Runs all the advanced memory examples in sequence."""
    print("Graphizy Advanced Memory Examples")
    print("=" * 70)

    try:
        # Run the examples one by one.
        example_basic_memory_manager()
        example_memory_update_strategies()
        example_memory_with_aging()

        print("\n" + "=" * 70)
        print("All advanced memory examples completed successfully!")
        print("Check the 'examples/output/' folder for the generated images.")

        # Summarize the key lessons from these examples.
        print("\nKey Insights:")
        print("  - Memory managers are powerful tools for tracking how networks change over time.")
        print(
            "  - You can limit memory by size or by time (iterations) to model realistic scenarios like short-term memory.")
        print(
            "  - Memory can be updated automatically from spatial data using different strategies (Proximity, Delaunay, etc.).")
        print(
            "  - Visualizing edge 'age' helps to understand the history and stability of connections in a dynamic network.")

    except Exception as e:
        print(f"\nAdvanced memory examples failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


# This standard Python construct ensures that the `main()` function is called
# only when this script is executed directly.
if __name__ == "__main__":
    sys.exit(main())