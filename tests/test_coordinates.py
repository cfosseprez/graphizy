#!/usr/bin/env python3
"""
Test coordinate system compatibility
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphizy import generate_positions, make_subdiv, Graphing


def test_coordinate_system():
    """Test that coordinates are handled correctly"""
    print("=== Coordinate System Test ===")

    # Test case 1: Square dimensions
    print("\nTest 1: Square (800x800)")
    positions = generate_positions(800, 800, 50)
    print(f"X range: {positions[:, 0].min():.0f} to {positions[:, 0].max():.0f}")
    print(f"Y range: {positions[:, 1].min():.0f} to {positions[:, 1].max():.0f}")

    try:
        subdiv = make_subdiv(positions, (800, 800))
        print("✓ Square subdivision created successfully")
    except Exception as e:
        print(f"✗ Square subdivision failed: {e}")

    # Test case 2: Rectangular dimensions (width > height)
    print("\nTest 2: Rectangle (900x600) - width > height")
    positions = generate_positions(900, 600, 50)
    print(f"X range: {positions[:, 0].min():.0f} to {positions[:, 0].max():.0f} (should be 0-899)")
    print(f"Y range: {positions[:, 1].min():.0f} to {positions[:, 1].max():.0f} (should be 0-599)")

    try:
        subdiv = make_subdiv(positions, (900, 600))
        print("✓ Rectangle subdivision created successfully")
    except Exception as e:
        print(f"✗ Rectangle subdivision failed: {e}")

    # Test case 3: Rectangular dimensions (height > width)
    print("\nTest 3: Rectangle (600x900) - height > width")
    positions = generate_positions(600, 900, 50)
    print(f"X range: {positions[:, 0].min():.0f} to {positions[:, 0].max():.0f} (should be 0-599)")
    print(f"Y range: {positions[:, 1].min():.0f} to {positions[:, 1].max():.0f} (should be 0-899)")

    try:
        subdiv = make_subdiv(positions, (600, 900))
        print("✓ Rectangle subdivision created successfully")
    except Exception as e:
        print(f"✗ Rectangle subdivision failed: {e}")


def test_edge_cases():
    """Test edge cases for coordinates"""
    print("\n=== Edge Cases Test ===")

    # Test case 1: Points at exact boundaries
    print("\nTest 1: Boundary points")
    positions = np.array([
        [0.0, 0.0],  # Bottom-left corner
        [899.0, 0.0],  # Bottom-right corner
        [0.0, 599.0],  # Top-left corner
        [899.0, 599.0],  # Top-right corner
        [450.0, 300.0]  # Center
    ], dtype=np.float32)

    try:
        subdiv = make_subdiv(positions, (900, 600))
        print("✓ Boundary points handled correctly")
    except Exception as e:
        print(f"✗ Boundary points failed: {e}")

    # Test case 2: Points just outside boundaries (should fail)
    print("\nTest 2: Out-of-bounds points (should fail)")
    bad_positions = np.array([
        [900.0, 300.0],  # X too large
        [450.0, 600.0],  # Y too large
    ], dtype=np.float32)

    try:
        subdiv = make_subdiv(bad_positions, (900, 600))
        print("✗ Out-of-bounds points should have failed but didn't")
    except Exception as e:
        print(f"✓ Out-of-bounds points correctly rejected: {type(e).__name__}")


def test_graphing_integration():
    """Test full graphing workflow"""
    print("\n=== Graphing Integration Test ===")

    # Test the exact case that was failing
    positions = generate_positions(900, 600, 20)
    particle_ids = np.arange(len(positions))
    particle_stack = np.column_stack((particle_ids, positions))

    try:
        grapher = Graphing(dimension=(900, 600))
        graph = grapher.make_delaunay(particle_stack)
        print(f"✓ Delaunay graph created: {graph.vcount()} vertices, {graph.ecount()} edges")

        # Test proximity graph too
        prox_graph = grapher.make_proximity(particle_stack, proximity_thresh=50.0)
        print(f"✓ Proximity graph created: {prox_graph.vcount()} vertices, {prox_graph.ecount()} edges")

    except Exception as e:
        print(f"✗ Graphing integration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_coordinate_system()
    test_edge_cases()
    test_graphing_integration()
    print("\n=== Testing Complete ===")