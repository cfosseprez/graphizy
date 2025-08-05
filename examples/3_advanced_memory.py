#!/usr/bin/env python3
"""
Advanced Memory Examples for Graphizy - Updated for v0.1.17+

This script demonstrates the enhanced memory capabilities of Graphizy v0.1.17+.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
import logging
import sys

from graphizy import (
    Graphing, GraphizyConfig, generate_and_format_positions,
    validate_graphizy_input, GraphizyError
)
from graphizy.utils import setup_output_directory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def example_modern_memory_integration():
    """Demonstrates modern memory integration with smart defaults."""
    print("\n" + "=" * 60)
    print("EXAMPLE: MODERN MEMORY INTEGRATION WITH SMART DEFAULTS")
    print("=" * 60)

    try:
        # Initialize grapher with memory system
        # Define the shape of the data we are using to prevent warnings
        simple_data_shape = [('id', int), ('x', float), ('y', float)]
        config = GraphizyConfig(dimension=(600, 600))
        # Pass the correct data_shape during initialization
        grapher = Graphing(config=config, data_shape=simple_data_shape)
        
        # Initialize memory manager with enhanced features
        grapher.init_memory_manager(
            max_memory_size=100,
            max_iterations=20,
            track_edge_ages=True
        )
        
        # Initialize weight system for comprehensive analysis
        grapher.init_weight_computer(method="distance", target_attribute="weight")

        # Generate sample data
        data = generate_and_format_positions(600, 600, 40)
        validate_graphizy_input(data)

        print("Demonstrating smart memory defaults...")

        # Simulate temporal evolution
        for iteration in range(10):
            print(f"\nIteration {iteration + 1}:")
            
            # Evolve positions
            if iteration > 0:
                data[:, 1:3] += np.random.normal(0, 8, (len(data), 2))
                data[:, 1] = np.clip(data[:, 1], 50, 550)
                data[:, 2] = np.clip(data[:, 2], 50, 550)

            # Create memory-enhanced graph with smart defaults
            memory_graph = grapher.make_graph("proximity", data, proximity_thresh=80.0)
            # Automatically: use_memory=True, update_memory=True, compute_weights=True
            
            if memory_graph:
                info = grapher.get_graph_info(memory_graph)
                print(f"  Graph: {info['edge_count']} edges")
                
                # Check edge attributes
                edge_attrs = memory_graph.es.attributes()
                if 'memory_based' in edge_attrs:
                    memory_edges = sum(1 for e in memory_graph.es if e['memory_based'])
                    current_edges = memory_graph.ecount() - memory_edges
                    print(f"  Memory edges: {memory_edges}, Current edges: {current_edges}")
                
                if 'weight' in edge_attrs:
                    weights = memory_graph.es['weight']
                    print(f"  Weights: avg={np.mean(weights):.3f}")

            # Save visualizations every few iterations
            if iteration % 3 == 0:
                output_dir = setup_output_directory()
                try:
                    memory_image = grapher.draw_memory_graph(
                        memory_graph,
                        use_age_colors=True,
                        alpha_range=(0.3, 1.0)
                    )
                    filename = f"memory_evolution_{iteration:02d}.jpg"
                    grapher.save_graph(memory_image, str(output_dir / filename))
                    print(f"  Saved: {filename}")
                except Exception as e:
                    print(f"  Visualization failed: {e}")

        # Final memory analysis
        memory_stats = grapher.get_memory_analysis()
        print(f"\nFinal Memory Statistics:")
        print(f"  Total connections: {memory_stats.get('total_connections', 0)}")
        print(f"  Total objects: {memory_stats.get('total_objects', 0)}")
        
        if 'edge_age_stats' in memory_stats:
            age_stats = memory_stats['edge_age_stats']
            print(f"  Average edge age: {age_stats.get('avg_age', 0):.1f}")

        print("Modern memory integration completed successfully!")

    except Exception as e:
        print(f"Memory integration failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all advanced memory examples."""
    print("Graphizy Advanced Memory Examples - v0.1.17+ Edition")
    print("=" * 70)

    try:
        example_modern_memory_integration()

        print("\n" + "=" * 70)
        print("Advanced memory examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Modern memory manager with smart defaults")
        print("  • Automatic memory integration with make_graph()")
        print("  • Age-based visualization with temporal effects")
        print("  • Integration with weight computation systems")

    except Exception as e:
        print(f"Examples failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
