#!/usr/bin/env python3
"""
Trace exactly where the error is coming from
"""

import sys
import traceback
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
)

import numpy as np
from graphizy import Graphing, GraphizyConfig, generate_positions

def trace_error_source():
    """Trace exactly where the error occurs"""

    print("=== Tracing Error Source ===")

    # Setup that works
    config = GraphizyConfig()
    config.graph.dimension = (900, 600)  # width=900, height=600
    config.graph.proximity_threshold = 80.0

    print(f"🔍 Generating positions for: width={900}, height={600}")
    positions = generate_positions(600, 900, 100)  # Swapped as you found works

    print(f"📊 Generated positions:")
    print(f"   X range: {positions[:, 0].min():.1f} to {positions[:, 0].max():.1f}")
    print(f"   Y range: {positions[:, 1].min():.1f} to {positions[:, 1].max():.1f}")
    print(f"🎯 Config dimensions: {config.graph.dimension}")

    particle_ids = np.arange(len(positions))
    particle_stack = np.column_stack((particle_ids, positions))

    # Create grapher
    print("\n1. Creating Graphing object...")
    try:
        grapher = Graphing(config=config)
        print("✅ Graphing object created successfully")
    except Exception as e:
        print(f"❌ Failed to create Graphing object: {e}")
        traceback.print_exc()
        return

    # Test make_delaunay step by step
    print("\n2. Testing make_delaunay...")
    try:
        print("  2a. Calling make_delaunay...")
        delaunay_graph = grapher.make_delaunay(particle_stack)
        print("✅ make_delaunay completed successfully")
        print(f"   Graph: {delaunay_graph.vcount()} vertices, {delaunay_graph.ecount()} edges")
    except Exception as e:
        print(f"❌ make_delaunay failed: {e}")
        traceback.print_exc()
        return

    # Test make_proximity
    print("\n3. Testing make_proximity...")
    try:
        print("  3a. Calling make_proximity...")
        proximity_graph = grapher.make_proximity(particle_stack)
        print("✅ make_proximity completed successfully")
        print(f"   Graph: {proximity_graph.vcount()} vertices, {proximity_graph.ecount()} edges")
    except Exception as e:
        print(f"❌ make_proximity failed: {e}")
        traceback.print_exc()
        return

    # Test drawing
    print("\n4. Testing drawing...")
    try:
        print("  4a. Drawing delaunay graph...")
        delaunay_image = grapher.draw_graph(delaunay_graph)
        print("✅ Delaunay drawing completed")

        print("  4b. Drawing proximity graph...")
        proximity_image = grapher.draw_graph(proximity_graph)
        print("✅ Proximity drawing completed")
    except Exception as e:
        print(f"❌ Drawing failed: {e}")
        traceback.print_exc()
        return

    # Test saving
    print("\n5. Testing saving...")
    try:
        from pathlib import Path
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        print("  5a. Saving delaunay...")
        grapher.save_graph(delaunay_image, str(output_dir / "trace_delaunay.jpg"))
        print("✅ Delaunay saved")

        print("  5b. Saving proximity...")
        grapher.save_graph(proximity_image, str(output_dir / "trace_proximity.jpg"))
        print("✅ Proximity saved")
    except Exception as e:
        print(f"❌ Saving failed: {e}")
        traceback.print_exc()
        return

    print("\n✅ All operations completed successfully!")

def trace_make_subdiv_specifically():
    """Test make_subdiv specifically with detailed tracing"""
    print("\n=== Tracing make_subdiv Specifically ===")

    from graphizy.algorithms import make_subdiv

    # Generate the exact same positions
    positions = generate_positions(600, 900, 100)
    dimensions = (900, 600)  # width, height

    print(f"Testing make_subdiv with:")
    print(f"  Positions shape: {positions.shape}")
    print(f"  Position ranges: X[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], Y[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")
    print(f"  Dimensions: {dimensions}")

    try:
        print("Calling make_subdiv...")
        subdiv = make_subdiv(positions, dimensions, do_print=True)
        print("✅ make_subdiv completed successfully")

        # Test triangle list generation
        triangle_list = subdiv.getTriangleList()
        print(f"✅ Generated {len(triangle_list)} triangles")

    except Exception as e:
        print(f"❌ make_subdiv failed: {e}")
        traceback.print_exc()

        # Additional debugging
        print(f"\n🔍 Debugging make_subdiv failure:")
        width, height = dimensions
        print(f"  Expected X range: [0, {width})")
        print(f"  Actual X range: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}]")
        print(f"  X valid: {(positions[:, 0] >= 0).all() and (positions[:, 0] < width).all()}")

        print(f"  Expected Y range: [0, {height})")
        print(f"  Actual Y range: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")
        print(f"  Y valid: {(positions[:, 1] >= 0).all() and (positions[:, 1] < height).all()}")

def inspect_drawing_function():
    """Check if the error is in the drawing functions"""
    print("\n=== Inspecting Drawing Functions ===")

    from graphizy.drawing import draw_point, draw_line

    # Create a test image
    test_image = np.zeros((600, 900, 3), dtype=np.uint8)  # height, width, channels

    print(f"Test image shape: {test_image.shape}")

    # Test some drawing operations
    test_points = [
        (100, 100),  # Normal point
        (800, 500),  # Near edge
        (0, 0),      # Corner
        (899, 599),  # Max valid coordinates for 900x600
    ]

    for i, point in enumerate(test_points):
        try:
            print(f"Testing point {i}: {point}")
            draw_point(test_image, point, (0, 255, 0), radius=5)
            print(f"✅ Point {point} drawn successfully")
        except Exception as e:
            print(f"❌ Point {point} failed: {e}")

    # Test a line
    try:
        print("Testing line drawing...")
        draw_line(test_image, 0, 0, 100, 100, (255, 0, 0))
        print("✅ Line drawn successfully")
    except Exception as e:
        print(f"❌ Line drawing failed: {e}")

if __name__ == "__main__":
    # Run all traces
    trace_error_source()
    trace_make_subdiv_specifically()
    inspect_drawing_function()

    print("\n=== Trace Complete ===")