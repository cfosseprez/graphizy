# examples/3_configuration_and_weights.py

#!/usr/bin/env python3
"""
Configuration and Weight System Examples for Graphizy

This script provides a detailed guide on how to configure the Graphing object
and use the powerful edge weight computation system.

This example covers:
1.  **Easy Configuration**: Using simple keyword arguments to set up Graphing.
2.  **Advanced Configuration**: Using the `GraphizyConfig` object for full control.
3.  **Weight System**: Computing edge weights based on distance.
4.  **Custom Formulas**: Defining your own formulas to compute custom edge attributes.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
import logging
from pathlib import Path
import sys

from graphizy import (
    Graphing,
    GraphizyConfig,
    generate_and_format_positions,
    GraphizyError
)
from graphizy.utils import setup_output_directory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# =============================================================================
# EXAMPLE 1: EASY CONFIGURATION WITH KEYWORD ARGUMENTS
# =============================================================================

def example_easy_configuration():
    """
    Demonstrates the simplest way to configure Graphizy using keyword arguments.
    This is the recommended approach for most use cases.
    """
    print("\n" + "=" * 60)
    print(" EXAMPLE 1: EASY CONFIGURATION WITH KEYWORD ARGUMENTS")
    print("=" * 60)

    try:
        # --- Data Generation ---
        data = generate_and_format_positions(size_x=500, size_y=500, num_particles=40)

        # --- Easy Initialization ---
        # Pass configuration settings directly as keyword arguments.
        # Graphizy will automatically route them to the correct config section.
        print("Initializing Graphing with direct keyword arguments...")
        grapher = Graphing(
            dimension=(500, 500),
            line_color=(0, 128, 255),  # Routed to DrawingConfig
            point_radius=10,           # Routed to DrawingConfig
            proximity_thresh=100.0     # Routed to GraphConfig
        )

        # Create a graph using these settings
        graph = grapher.make_graph("proximity", data)

        # --- Output ---
        image = grapher.draw_graph(graph)
        output_path = setup_output_directory() / "config_easy_style.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Graph with easy configuration saved to: {output_path}")

    except GraphizyError as e:
        print(f"Easy configuration example failed: {e}")


# =============================================================================
# EXAMPLE 2: ADVANCED CONFIGURATION WITH THE CONFIG OBJECT
# =============================================================================

def example_advanced_configuration():
    """
    Demonstrates the power-user approach using the GraphizyConfig object.
    This gives you full control for complex or reusable configurations.
    """
    print("\n" + "=" * 60)
    print(" EXAMPLE 2: ADVANCED CONFIGURATION WITH THE CONFIG OBJECT")
    print("=" * 60)

    try:
        # --- Data Generation ---
        data = generate_and_format_positions(size_x=500, size_y=500, num_particles=40)

        # --- Advanced Initialization ---
        print("Creating a detailed GraphizyConfig object...")
        # 1. Create a config object
        my_config = GraphizyConfig()

        # 2. Set nested properties for fine-grained control
        my_config.graph.dimension = (500, 500)
        my_config.drawing.line_color = (255, 0, 255)  # Magenta
        my_config.drawing.point_color = (255, 255, 0) # Cyan
        my_config.drawing.line_thickness = 3

        # 3. Initialize Graphing with the pre-made config object
        grapher = Graphing(config=my_config)

        # Create a graph
        graph = grapher.make_graph("delaunay", data)

        # --- Output ---
        image = grapher.draw_graph(graph)
        output_path = setup_output_directory() / "config_advanced_style.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Graph with advanced configuration saved to: {output_path}")

    except GraphizyError as e:
        print(f"Advanced configuration example failed: {e}")


# =============================================================================
# EXAMPLE 3: CONFIGURING THE WEIGHT SYSTEM
# =============================================================================

def example_weight_configuration():
    """
    Demonstrates how to configure and use the edge weight computation system.
    """
    print("\n" + "=" * 60)
    print(" EXAMPLE 3: CONFIGURING THE WEIGHT SYSTEM")
    print("=" * 60)

    try:
        # --- Data and Grapher Setup ---
        data = generate_and_format_positions(size_x=500, size_y=500, num_particles=20)
        grapher = Graphing(dimension=(500, 500))

        # --- Part 1: Default Weight Computation (Distance) ---
        print("\n--- Part 1: Computing default 'distance' attribute ---")
        # By default, auto_compute_weights is True, using the 'distance' method.
        # The result is stored in the 'weight' edge attribute.
        graph_with_weights = grapher.make_graph("proximity", data, proximity_thresh=150.0)

        # Analyze the results
        if 'weight' in graph_with_weights.es.attributes():
            weights = graph_with_weights.es['weight']
            print(f"  Successfully computed {len(weights)} edge weights.")
            print(f"  Weight stats (distance): Mean={np.mean(weights):.2f}, Max={np.max(weights):.2f}")
        else:
            print("  'weight' attribute not found.")

        # --- Part 2: Computing a Custom Weight with a Formula ---
        print("\n--- Part 2: Computing a custom 'strength' attribute from a formula ---")
        # First, ensure the 'distance' attribute exists on the graph
        grapher.compute_edge_attribute(graph_with_weights, "distance", method="distance")

        # Now, compute a new attribute 'strength' based on the 'distance'
        grapher.compute_edge_attribute(
            graph_with_weights,
            attribute_name="strength",
            method="formula",
            formula="1 / (distance + 1)"  # A common inverse distance formula
        )

        # Analyze the new attribute
        if 'strength' in graph_with_weights.es.attributes():
            strengths = graph_with_weights.es['strength']
            print(f"  Successfully computed {len(strengths)} 'strength' attributes.")
            print(f"  Strength stats (1/dist): Mean={np.mean(strengths):.3f}, Max={np.max(strengths):.3f}")
        else:
            print("  'strength' attribute not found.")

    except GraphizyError as e:
        print(f"Weight configuration example failed: {e}")


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Runs all the configuration examples."""
    print("=" * 60)
    print("Graphizy Configuration and Weight System Examples")
    print("=" * 60)

    try:
        example_easy_configuration()
        example_advanced_configuration()
        example_weight_configuration()

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