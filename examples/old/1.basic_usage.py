#!/usr/bin/env python3
"""
Basic usage examples for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

import numpy as np
import logging
from pathlib import Path

# Add src to path for development
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphizy import (
    Graphing, GraphizyConfig, generate_positions,
    GraphizyError
)

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)


def example_basic_delaunay():
    """Basic Delaunay triangulation example"""
    print("=== Basic Delaunay Triangulation ===")

    try:
        # Generate random positions
        SIZE = 600
        NUM_PARTICLES = 50

        positions = generate_positions(SIZE, SIZE, NUM_PARTICLES)
        print(f"Generated {len(positions)} positions")

        # Create particle data with IDs
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Create grapher
        grapher = Graphing(dimension=(SIZE, SIZE))

        # Create Delaunay triangulation
        print("Creating Delaunay triangulation...")
        delaunay_graph = grapher.make_delaunay(particle_stack)

        # Get graph statistics
        info = grapher.get_graph_info(delaunay_graph)
        print(f"Graph info: {info}")

        # Draw and save
        image = grapher.draw_graph(delaunay_graph)
        output_path = Path("") / "basic_delaunay.jpg"
        output_path.parent.mkdir(exist_ok=True)
        grapher.save_graph(image, str(output_path))
        print(f"Saved to {output_path}")

    except GraphizyError as e:
        print(f"Error: {e}")


def example_proximity_graph():
    """Proximity graph example"""
    print("\n=== Proximity Graph ===")

    try:
        # Generate positions
        SIZE = 800
        NUM_PARTICLES = 75
        THRESHOLD = 60.0

        positions = generate_positions(SIZE, SIZE, NUM_PARTICLES)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Create grapher
        grapher = Graphing(dimension=(SIZE, SIZE))

        # Create proximity graph
        print(f"Creating proximity graph with threshold {THRESHOLD}...")
        proximity_graph = grapher.make_proximity(
            particle_stack,
            proximity_thresh=THRESHOLD,
            metric='euclidean'
        )

        # Get statistics
        info = grapher.get_graph_info(proximity_graph)
        print(f"Proximity graph - Vertices: {info['vertex_count']}, Edges: {info['edge_count']}")
        print(f"Density: {info['density']:.4f}, Connected: {info['is_connected']}")

        # Draw and save
        image = grapher.draw_graph(proximity_graph)
        output_path = Path("") / "proximity_graph.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Saved to {output_path}")

    except GraphizyError as e:
        print(f"Error: {e}")


def example_custom_configuration():
    """Custom configuration example"""
    print("\n=== Custom Configuration ===")

    try:
        # Create custom configuration
        config = GraphizyConfig()

        # Customize drawing parameters
        config.drawing.line_color = (0, 0, 255)  # Red lines (BGR format)
        config.drawing.line_thickness = 2
        config.drawing.point_color = (255, 255, 0)  # Cyan points
        config.drawing.point_radius = 12
        config.drawing.point_thickness = 3

        # Customize graph parameters - Keep rectangle to debug x,y swap
        config.graph.dimension = (600, 900)  # width=900, height=600
        config.graph.proximity_threshold = 80.0

        # Generate data with DEBUG
        print(f"🔍 Generating positions for: width={900}, height={600}")
        positions = generate_positions(600, 900, 100)

        # Debug: Check the actual position ranges
        print(f"📊 Generated positions:")
        print(f"   X range: {positions[:, 0].min():.1f} to {positions[:, 0].max():.1f} (should be 0 to 899)")
        print(f"   Y range: {positions[:, 1].min():.1f} to {positions[:, 1].max():.1f} (should be 0 to 599)")
        print(f"🎯 Config dimensions: {config.graph.dimension} (width, height)")

        # Check if any points are out of expected bounds
        x_out = (positions[:, 0] >= 900).sum()
        y_out = (positions[:, 1] >= 600).sum()
        print(f"⚠️  Points out of bounds: X={x_out}, Y={y_out}")

        # Show a few sample points
        print(f"📍 First 5 points: {positions[:5]}")

        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Create grapher with custom config
        grapher = Graphing(config=config)

        # Debug: Check what make_subdiv sees
        print(f"🔧 About to call make_subdiv with:")
        print(f"   Points shape: {positions.shape}")
        print(f"   Dimensions: {config.graph.dimension}")

        # Create both graphs
        delaunay_graph = grapher.make_delaunay(particle_stack)
        proximity_graph = grapher.make_proximity(particle_stack)

        # Draw with custom styling
        delaunay_image = grapher.draw_graph(delaunay_graph)
        proximity_image = grapher.draw_graph(proximity_graph)

        # Save both
        output_dir = Path("")
        output_dir.mkdir(exist_ok=True)

        grapher.save_graph(delaunay_image, str(output_dir / "custom_delaunay.jpg"))
        grapher.save_graph(proximity_image, str(output_dir / "custom_proximity.jpg"))

        print("✅ Saved custom styled graphs")

    except GraphizyError as e:
        print(f"❌ Error: {e}")
        # Let's also print more details about where it failed
        import traceback
        print("📋 Full traceback:")
        traceback.print_exc()

def example_runtime_config_updates():
    """Runtime configuration updates example"""
    print("\n=== Runtime Configuration Updates ===")

    try:
        # Start with default config
        grapher = Graphing()

        # Generate data
        positions = generate_positions(500, 500, 60)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Create initial graph with default settings
        graph = grapher.make_delaunay(particle_stack)
        default_image = grapher.draw_graph(graph)

        # Update configuration at runtime
        grapher.update_config(
            drawing={
                "line_color": (0, 255, 255),  # Yellow lines
                "line_thickness": 3,
                "point_color": (255, 0, 255),  # Magenta points
                "point_radius": 15
            }
        )

        # Draw with updated configuration
        updated_image = grapher.draw_graph(graph)

        # Save both versions
        output_dir = Path("")
        output_dir.mkdir(exist_ok=True)

        grapher.save_graph(default_image, str(output_dir / "default_style.jpg"))
        grapher.save_graph(updated_image, str(output_dir / "updated_style.jpg"))

        print("Saved graphs with default and updated styling")

    except GraphizyError as e:
        print(f"Error: {e}")


def example_igraph_methods():
    """Example of calling various igraph methods"""
    print("\n=== igraph Methods ===")

    try:
        # Generate connected graph data
        positions = generate_positions(400, 400, 30)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        grapher = Graphing(dimension=(400, 400))

        # Create both types of graphs
        delaunay_graph = grapher.make_delaunay(particle_stack)
        proximity_graph = grapher.make_proximity(particle_stack, proximity_thresh=70.0)

        print("Delaunay Graph Analysis:")
        print(f"  Vertex count: {grapher.call_method(delaunay_graph, 'vcount')}")
        print(f"  Edge count: {grapher.call_method(delaunay_graph, 'ecount')}")
        print(f"  Is connected: {grapher.call_method(delaunay_graph, 'is_connected')}")
        print(f"  Density: {grapher.call_method(delaunay_graph, 'density'):.4f}")

        # Try to get diameter (might fail if disconnected)
        try:
            diameter = grapher.call_method(delaunay_graph, 'diameter')
            print(f"  Diameter: {diameter}")
        except Exception as e:
            print(f"  Diameter: Cannot calculate (possibly disconnected)")

        print("\nProximity Graph Analysis:")
        print(f"  Vertex count: {grapher.call_method(proximity_graph, 'vcount')}")
        print(f"  Edge count: {grapher.call_method(proximity_graph, 'ecount')}")
        print(f"  Is connected: {grapher.call_method(proximity_graph, 'is_connected')}")
        print(f"  Density: {grapher.call_method(proximity_graph, 'density'):.4f}")

        # Get degree sequence
        degree_seq = grapher.call_method(proximity_graph, 'degree')
        print(f"  Average degree: {np.mean(degree_seq):.2f}")
        print(f"  Max degree: {max(degree_seq)}")

    except GraphizyError as e:
        print(f"Error: {e}")


def example_custom_data_structure():
    """Example with custom data structure"""
    print("\n=== Custom Data Structure ===")

    try:
        # Define custom data structure
        data_shape = [
            ("id", int),  # Must be 'id'
            ("x", float),  # Must be 'x'
            ("y", float),  # Must be 'y'
            ("velocity", float),
            ("mass", float)
        ]

        # Generate positions
        positions = generate_positions(600, 600, 40, convert=True)

        # Create extended data with velocity and mass
        n_particles = len(positions)
        particle_data = np.column_stack([
            np.arange(n_particles),  # particle_id
            positions[:, 0],  # position_x
            positions[:, 1],  # position_y
            np.random.uniform(0, 10, n_particles),  # velocity
            np.random.uniform(1, 5, n_particles)  # mass
        ])

        # Create grapher with custom data structure
        grapher = Graphing(
            dimension=(600, 600),
            data_shape=data_shape
        )

        # Create graph (uses only position_x and position_y for triangulation)
        graph = grapher.make_delaunay(particle_data)

        print(f"Created graph with {graph.vcount()} vertices from custom data structure")

        # Access additional data
        velocities = particle_data[:, 3]
        masses = particle_data[:, 4]
        print(f"Velocity range: {velocities.min():.2f} - {velocities.max():.2f}")
        print(f"Mass range: {masses.min():.2f} - {masses.max():.2f}")

        # Draw and save
        image = grapher.draw_graph(graph)
        output_path = Path("") / "custom_data_structure.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Saved to {output_path}")

    except GraphizyError as e:
        print(f"Error: {e}")


def example_comparison_analysis():
    """Compare Delaunay vs Proximity graphs"""
    print("\n=== Graph Comparison Analysis ===")

    try:
        # Generate same dataset for both graphs
        SIZE = 700
        NUM_PARTICLES = 80
        THRESHOLD = 45.0

        positions = generate_positions(SIZE, SIZE, NUM_PARTICLES)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        grapher = Graphing(dimension=(SIZE, SIZE))

        # Create both graphs
        print("Creating graphs...")
        delaunay_graph = grapher.make_delaunay(particle_stack)
        proximity_graph = grapher.make_proximity(particle_stack, proximity_thresh=THRESHOLD)

        # Analyze both
        del_info = grapher.get_graph_info(delaunay_graph)
        prox_info = grapher.get_graph_info(proximity_graph)

        print(f"\nComparison for {NUM_PARTICLES} particles:")
        print("=" * 50)
        print(f"{'Property':<20} {'Delaunay':<15} {'Proximity':<15}")
        print("=" * 50)
        print(f"{'Vertices':<20} {del_info['vertex_count']:<15} {prox_info['vertex_count']:<15}")
        print(f"{'Edges':<20} {del_info['edge_count']:<15} {prox_info['edge_count']:<15}")
        print(f"{'Density':<20} {del_info['density']:<15.4f} {prox_info['density']:<15.4f}")
        print(f"{'Connected':<20} {del_info['is_connected']:<15} {prox_info['is_connected']:<15}")

        if del_info['average_path_length'] and prox_info['average_path_length']:
            print(
                f"{'Avg Path Length':<20} {del_info['average_path_length']:<15.4f} {prox_info['average_path_length']:<15.4f}")

        if del_info['diameter'] and prox_info['diameter']:
            print(f"{'Diameter':<20} {del_info['diameter']:<15} {prox_info['diameter']:<15}")

        # Create side-by-side visualization
        del_image = grapher.draw_graph(delaunay_graph)

        # Change colors for proximity graph
        grapher.update_config(
            drawing={
                "line_color": (255, 0, 0),  # Blue lines
                "point_color": (0, 255, 255)  # Yellow points
            }
        )
        prox_image = grapher.draw_graph(proximity_graph)

        # Save both
        output_dir = Path("")
        output_dir.mkdir(exist_ok=True)

        grapher.save_graph(del_image, str(output_dir / "comparison_delaunay.jpg"))
        grapher.save_graph(prox_image, str(output_dir / "comparison_proximity.jpg"))

        print(f"\nGraphs saved to {output_dir}/")

    except GraphizyError as e:
        print(f"Error: {e}")


def example_error_handling():
    """Demonstrate error handling"""
    print("\n=== Error Handling ===")

    # Example 1: Invalid dimensions
    try:
        grapher = Graphing(dimension=(-100, 100))
    except GraphizyError as e:
        print(f"Caught expected error: {type(e).__name__}: {e}")

    # Example 2: Invalid data shape
    try:
        from graphizy.algorithms import DataInterface
        invalid_shape = [("x", int)]  # Missing required 'id' and 'y'
        interface = DataInterface(invalid_shape)
    except GraphizyError as e:
        print(f"Caught expected error: {type(e).__name__}: {e}")

    # Example 3: Invalid position generation
    try:
        positions = generate_positions(10, 10, 200)  # Too many particles
    except GraphizyError as e:
        print(f"Caught expected error: {type(e).__name__}: {e}")

    # Example 4: Invalid method call
    try:
        from graphizy.algorithms import create_graph_array
        graph = create_graph_array(np.array([[0, 10, 20]]))
        from graphizy import call_igraph_method
        result = call_igraph_method(graph, 'nonexistent_method')
    except GraphizyError as e:
        print(f"Caught expected error: {type(e).__name__}: {e}")

    print("Error handling examples completed successfully!")


def main():
    """Run all examples"""
    print("Graphizy Examples")
    print("================")

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create output directory
    output_dir = Path("")
    output_dir.mkdir(exist_ok=True)

    try:
        # Run all examples
        example_basic_delaunay()
        example_proximity_graph()
        example_custom_configuration()
        example_runtime_config_updates()
        example_igraph_methods()
        example_custom_data_structure()
        example_comparison_analysis()
        example_error_handling()

        print(f"\n✅ All examples completed successfully!")
        print(f"📁 Output files saved to: {output_dir.absolute()}")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()