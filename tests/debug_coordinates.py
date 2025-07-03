#!/usr/bin/env python3
"""
Debug coordinate system to understand what's happening
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphizy import generate_positions


def debug_coordinate_system():
    """Debug the exact coordinate issue"""
    print("=== Debugging Coordinate System ===")

    # Test case: Rectangle (900x600) that's failing
    print("\nGenerating positions for (900, 600):")
    positions = generate_positions(900, 600, 20)

    print(f"Positions shape: {positions.shape}")
    print(f"X range: {positions[:, 0].min():.0f} to {positions[:, 0].max():.0f}")
    print(f"Y range: {positions[:, 1].min():.0f} to {positions[:, 1].max():.0f}")
    print(f"Positions dtype: {positions.dtype}")

    # Check what the bounds checking sees
    dimensions = (900, 600)
    print(f"\nDimensions: {dimensions}")
    print(f"dimensions[0] (width): {dimensions[0]}")
    print(f"dimensions[1] (height): {dimensions[1]}")

    # Current bounds checking logic (what's in your code now)
    print(f"\nCurrent bounds checking:")
    x_min_ok = np.all(positions[:, 0] >= 0)
    x_max_ok_current = np.all(positions[:, 0] < dimensions[1])  # Check against height
    y_min_ok = np.all(positions[:, 1] >= 0)
    y_max_ok_current = np.all(positions[:, 1] < dimensions[0])  # Check against width

    print(f"X >= 0: {x_min_ok}")
    print(f"X < {dimensions[1]} (height): {x_max_ok_current}")
    print(f"Y >= 0: {y_min_ok}")
    print(f"Y < {dimensions[0]} (width): {y_max_ok_current}")

    # What the bounds checking SHOULD be
    print(f"\nCorrect bounds checking:")
    x_max_ok_correct = np.all(positions[:, 0] < dimensions[0])  # Check against width
    y_max_ok_correct = np.all(positions[:, 1] < dimensions[1])  # Check against height

    print(f"X < {dimensions[0]} (width): {x_max_ok_correct}")
    print(f"Y < {dimensions[1]} (height): {y_max_ok_correct}")

    # Show which points are causing issues
    print(f"\nPoints that violate current bounds:")
    bad_x = positions[positions[:, 0] >= dimensions[1]]
    bad_y = positions[positions[:, 1] >= dimensions[0]]

    if len(bad_x) > 0:
        print(f"X coordinates >= {dimensions[1]}: {bad_x[:, 0]}")
    if len(bad_y) > 0:
        print(f"Y coordinates >= {dimensions[0]}: {bad_y[:, 1]}")

    # Show exact values that are problematic
    print(f"\nDetailed analysis:")
    print(f"Max X coordinate: {positions[:, 0].max()}")
    print(f"Checking X < {dimensions[1]}: {positions[:, 0].max() < dimensions[1]}")
    print(f"Should check X < {dimensions[0]}: {positions[:, 0].max() < dimensions[0]}")

    print(f"Max Y coordinate: {positions[:, 1].max()}")
    print(f"Checking Y < {dimensions[0]}: {positions[:, 1].max() < dimensions[0]}")
    print(f"Should check Y < {dimensions[1]}: {positions[:, 1].max() < dimensions[1]}")


def test_old_vs_new_logic():
    """Compare old vs new coordinate logic"""
    print("\n=== Old vs New Logic Comparison ===")

    positions = generate_positions(900, 600, 10)
    dimensions = (900, 600)  # (width, height)

    print(f"Generated positions for dimensions {dimensions}")
    print(f"Sample positions:\n{positions[:5]}")

    # OLD LOGIC (from your working version)
    print(f"\nOLD LOGIC (working):")
    print(f"Rectangle: (0, 0, {dimensions[1]}, {dimensions[0]}) = (0, 0, {dimensions[1]}, {dimensions[0]})")
    print(f"This means OpenCV sees width={dimensions[1]}, height={dimensions[0]}")

    # In old logic, bounds should be:
    old_x_ok = np.all(positions[:, 0] < dimensions[1])  # X < height
    old_y_ok = np.all(positions[:, 1] < dimensions[0])  # Y < width
    print(f"Old bounds: X < {dimensions[1]} = {old_x_ok}, Y < {dimensions[0]} = {old_y_ok}")

    # NEW LOGIC (what makes sense)
    print(f"\nNEW LOGIC (intuitive):")
    print(f"Rectangle: (0, 0, {dimensions[0]}, {dimensions[1]}) = (0, 0, {dimensions[0]}, {dimensions[1]})")
    print(f"This means OpenCV sees width={dimensions[0]}, height={dimensions[1]}")

    # In new logic, bounds should be:
    new_x_ok = np.all(positions[:, 0] < dimensions[0])  # X < width
    new_y_ok = np.all(positions[:, 1] < dimensions[1])  # Y < height
    print(f"New bounds: X < {dimensions[0]} = {new_x_ok}, Y < {dimensions[1]} = {new_y_ok}")


if __name__ == "__main__":
    debug_coordinate_system()
    test_old_vs_new_logic()