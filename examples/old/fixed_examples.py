#!/usr/bin/env python3
"""
Fixed graphizy examples with proper coordinate system handling

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
        output_path = Path("old") / "basic_delaunay.jpg"
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
        output_path = Path("old") / "proximity_graph.jpg"
        grapher.save_graph(image, str(output_path))
        print(f"Saved to {output_path}")

    except GraphizyError as e:
        print(f"Error: {e}")


def example_custom_configuration():
    """Custom configuration example - FIXED coordinate system"""
    print("\n=== Custom Configuration (FIXED) ===")

    try:
        # Create custom configuration
        config = GraphizyConfig()

        # Customize drawing parameters
        config.drawing.line_color = (0, 0, 255)  # Red lines (BGR format)
        config.drawing.line_thickness = 2
        config.drawing.point_color = (255, 255, 0)  # Cyan points
        config.drawing.point_radius = 12
        config.drawing.point_thickness = 3

        # FIXED: Properly align coordinate systems
        WIDTH = 900
        HEIGHT = 600
        config.graph.dimension = (WIDTH, HEIGHT)  # (width, height) for OpenCV
        config.graph.proximity_threshold = 80.0

        # FIXED: Generate positions with correct parameters
        # generate_positions(size_x, size_y) where size_x = width, size_y = height
        print(f"🔧 Generating positions for canvas: {WIDTH}x{HEIGHT}")
        positions = generate_positions(WIDTH, HEIGHT, 100)

        # Debug: Check the actual position ranges
        print(f"📊 Generated positions:")
        print(f"   X range: {positions[:, 0].min():.1f} to {positions[:, 0].max():.1f} (should be 0 to {WIDTH-1})")
        print(f"   Y range: {positions[:, 1].min():.1f} to {positions[:, 1].max():.1f} (should be 0 to {HEIGHT-1})")
        print(f"🎯 Config dimensions: {config.graph.dimension} (width, height)")

        # Check if any points are out of expected bounds
        x_out = (positions[:, 0] >= WIDTH).sum()
        y_out = (positions[:, 1] >= HEIGHT).sum()
        print(f"⚠️  Points out of bounds: X={x_out}, Y={y_out}")

        # Show a few sample points
        print(f"📍 First 5 points: {positions[:5]}")

        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))

        # Create grapher with custom config
        grapher = Graphing(config=config)

        print(f"🔧 About to call triangulation with:")
        print(f"   Points shape: {positions.shape}")
        print(f"   Canvas dimensions: {config.graph.dimension}")

        # Create both graphs
        delaunay_graph = grapher.make_delaunay(particle_stack)
        proximity_graph = grapher.make_proximity(particle_stack)

        # Draw with custom styling
        delaunay_image = grapher.draw_graph(delaunay_graph)
        proximity_image = grapher.draw_graph(proximity_graph)

        # Save both
        output_dir = Path("old")
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


def example_coordinate_system_test():
    """Test to verify coordinate system is working correctly"""
    print("\n=== Coordinate System Test ===")
    
    try:
        # Test different canvas sizes
        test_cases = [
            (400, 300, "4:3 landscape"),
            (300, 400, "3:4 portrait"), 
            (500, 500, "1:1 square"),
            (800, 600, "4:3 landscape large")
        ]
        
        for width, height, description in test_cases:
            print(f"\n🧪 Testing {description}: {width}x{height}")
            
            # Generate positions
            positions = generate_positions(width, height, 20)
            particle_ids = np.arange(len(positions))
            particle_stack = np.column_stack((particle_ids, positions))
            
            # Check bounds
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
            
            x_valid = (0 <= x_min) and (x_max < width)
            y_valid = (0 <= y_min) and (y_max < height)
            
            print(f"   X: {x_min:.0f} to {x_max:.0f} (valid: {x_valid})")
            print(f"   Y: {y_min:.0f} to {y_max:.0f} (valid: {y_valid})")
            
            # Test triangulation
            try:
                grapher = Graphing(dimension=(width, height))
                graph = grapher.make_delaunay(particle_stack)
                print(f"   ✅ Triangulation successful: {graph.vcount()} vertices, {graph.ecount()} edges")
            except Exception as e:
                print(f"   ❌ Triangulation failed: {e}")
                
    except Exception as e:
        print(f"❌ Coordinate system test failed: {e}")
        import traceback
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
        output_dir = Path("old")
        output_dir.mkdir(exist_ok=True)

        grapher.save_graph(default_image, str(output_dir / "default_style.jpg"))
        grapher.save_graph(updated_image, str(output_dir / "updated_style.jpg"))

        print("Saved graphs with default and updated styling")

    except GraphizyError as e:
        print(f"Error: {e}")


def example_memory_functionality():
    """Test the new memory functionality"""
    print("\n=== Memory Functionality Test ===")
    
    try:
        from graphizy import MemoryManager, create_memory_graph
        
        # Create memory manager
        memory_mgr = MemoryManager(max_memory_size=20, max_iterations=10, track_edge_ages=True)
        print("✅ Memory manager created")
        
        # Create test positions
        positions = generate_positions(400, 400, 30)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))
        
        # Create grapher with memory
        grapher = Graphing(dimension=(400, 400))
        grapher.init_memory_manager(max_memory_size=20, max_iterations=5)
        
        # Simulate multiple iterations
        for i in range(3):
            print(f"\n🔄 Iteration {i + 1}:")
            
            # Update memory with proximity connections
            connections = grapher.update_memory_with_proximity(particle_stack, 80.0)
            
            total_connections = sum(len(conns) for conns in connections.values()) // 2
            print(f"   Current proximity connections: {total_connections}")
            
            # Add some random movement for next iteration
            if i < 2:  # Don't move on last iteration
                particle_stack[:, 1:3] += np.random.normal(0, 10, (len(particle_stack), 2))
                # Keep within bounds
                particle_stack[:, 1] = np.clip(particle_stack[:, 1], 0, 399)
                particle_stack[:, 2] = np.clip(particle_stack[:, 2], 0, 399)
        
        # Create final memory graph
        memory_graph = grapher.make_memory_graph(particle_stack)
        stats = grapher.get_memory_stats()
        
        print(f"\n📊 Final memory stats:")
        print(f"   Total objects: {stats['total_objects']}")
        print(f"   Memory connections: {stats['total_connections']}")
        print(f"   Memory graph edges: {memory_graph.ecount()}")
        
        # Try to draw memory graph
        try:
            memory_image = grapher.draw_memory_graph(memory_graph, use_age_colors=True)
            output_path = Path("old") / "memory_graph.jpg"
            grapher.save_graph(memory_image, str(output_path))
            print(f"✅ Memory graph saved to {output_path}")
        except Exception as e:
            print(f"⚠️  Could not save memory graph image: {e}")
        
        print("✅ Memory functionality test completed")
        
    except Exception as e:
        print(f"❌ Memory functionality test failed: {e}")
        import traceback
        traceback.print_exc()


def example_error_handling():
    """Demonstrate error handling"""
    print("\n=== Error Handling ===")

    # Example 1: Invalid dimensions
    try:
        grapher = Graphing(dimension=(-100, 100))
    except GraphizyError as e:
        print(f"✅ Caught expected error: {type(e).__name__}")

    # Example 2: Invalid position generation
    try:
        positions = generate_positions(10, 10, 200)  # Too many particles
    except GraphizyError as e:
        print(f"✅ Caught expected error: {type(e).__name__}")

    print("✅ Error handling examples completed successfully!")


def main():
    """Run all examples"""
    print("🧠 Graphizy Examples with Memory Functionality")
    print("=" * 50)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create output directory
    output_dir = Path("old")
    output_dir.mkdir(exist_ok=True)

    try:
        # Run all examples
        example_basic_delaunay()
        example_proximity_graph()
        example_coordinate_system_test()  # NEW: Test coordinate system
        example_custom_configuration()     # FIXED: Proper coordinates
        example_runtime_config_updates()
        example_memory_functionality()     # NEW: Test memory features
        example_error_handling()

        print(f"\n🎉 All examples completed successfully!")
        print(f"📁 Output files saved to: {output_dir.absolute()}")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
