#!/usr/bin/env python3
"""Debug script to find where coordinates get swapped"""

import numpy as np
from graphizy import generate_positions, Graphing, GraphizyConfig


def test_coordinate_generation():
    """Test if generate_positions is working correctly"""
    print("=== Testing generate_positions ===")

    width, height = 900, 600
    positions = generate_positions(width, height, 10)

    print(f"Generated positions for width={width}, height={height}:")
    print(f"Shape: {positions.shape}")
    print(f"X range: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}] (expected [0, {width - 1}])")
    print(f"Y range: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}] (expected [0, {height - 1}])")
    print(f"First 5 positions:\n{positions[:5]}")

    # Check bounds
    x_ok = np.all(positions[:, 0] >= 0) and np.all(positions[:, 0] < width)
    y_ok = np.all(positions[:, 1] >= 0) and np.all(positions[:, 1] < height)

    print(f"\nX bounds check: {'✓' if x_ok else '✗'}")
    print(f"Y bounds check: {'✓' if y_ok else '✗'}")

    return positions


def test_graphing_dimensions():
    """Test how Graphing class handles dimensions"""
    print("\n=== Testing Graphing dimensions ===")

    width, height = 900, 600
    config = GraphizyConfig()
    config.graph.dimension = (width, height)

    grapher = Graphing(config=config)

    print(f"Config dimension: {config.graph.dimension}")
    print(f"Grapher dimension: {grapher.dimension}")
    print(f"Are they equal? {config.graph.dimension == grapher.dimension}")

    return grapher


def test_opencv_rect():
    """Test OpenCV rectangle creation"""
    print("\n=== Testing OpenCV rect creation ===")

    width, height = 900, 600

    # This is what happens in make_subdiv
    rect = (0, 0, width, height)
    print(f"OpenCV rect for ({width}, {height}): {rect}")
    print(f"This means: x in [0, {width}), y in [0, {height})")


def test_full_pipeline():
    """Test the full pipeline to see where it breaks"""
    print("\n=== Testing full pipeline ===")

    width, height = 900, 600
    num_particles = 10

    # 1. Generate positions
    positions = generate_positions(width, height, num_particles)
    print(f"1. Generated {len(positions)} positions")
    print(f"   X range: [{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}]")
    print(f"   Y range: [{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")

    # 2. Create particle stack
    particle_ids = np.arange(len(positions))
    particle_stack = np.column_stack((particle_ids, positions))
    print(f"\n2. Created particle stack with shape: {particle_stack.shape}")

    # 3. Create grapher
    config = GraphizyConfig()
    config.graph.dimension = (width, height)
    grapher = Graphing(config=config)
    print(f"\n3. Created grapher with dimension: {grapher.dimension}")

    # 4. Extract position array (this is what make_delaunay does)
    from graphizy.algorithms import DataInterface
    dinter = DataInterface([("id", int), ("x", int), ("y", int), ("speed", float), ("feedback", bool)])

    pos_array = np.stack((
        particle_stack[:, dinter.getidx_xpos()],
        particle_stack[:, dinter.getidx_ypos()]
    ), axis=1)

    print(f"\n4. Extracted position array:")
    print(f"   Shape: {pos_array.shape}")
    print(f"   X range: [{pos_array[:, 0].min():.1f}, {pos_array[:, 0].max():.1f}]")
    print(f"   Y range: [{pos_array[:, 1].min():.1f}, {pos_array[:, 1].max():.1f}]")

    # 5. Check bounds (what make_subdiv does)
    print(f"\n5. Checking bounds against dimension {grapher.dimension}:")
    x_violations = np.sum(pos_array[:, 0] >= width)
    y_violations = np.sum(pos_array[:, 1] >= height)
    print(f"   X violations (>= {width}): {x_violations}")
    print(f"   Y violations (>= {height}): {y_violations}")

    if y_violations > 0:
        print(f"\n   ⚠️  Problem detected! Y values exceed height limit")
        print(f"   Max Y value: {pos_array[:, 1].max():.1f} (limit: {height})")


if __name__ == "__main__":
    test_coordinate_generation()
    test_graphing_dimensions()
    test_opencv_rect()
    test_full_pipeline()