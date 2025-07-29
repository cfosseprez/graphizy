"""
Position generation and formatting utilities for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import random
from typing import List, Tuple, Union, Optional
import numpy as np

from graphizy.exceptions import PositionGenerationError

def format_positions(
    positions: Union[np.ndarray, List[Tuple[float, float]]],
    ids: Optional[Union[np.ndarray, List]] = None,
    start_id: int = 0
) -> np.ndarray:
    """
    Formats 2D positions into the standard graphizy data array by adding IDs.

    This is a convenience function to create the standard `(n, 3)` data array
    (id, x, y) required by many graphizy functions. It can either use a
    provided list of IDs or generate sequential ones.

    Args:
        positions: A NumPy array of shape (n, 2) or a list of (x, y) tuples.
        ids: An optional list or NumPy array of IDs. If provided, its length
             must match the number of positions.
        start_id: The starting integer for sequential IDs, used only if `ids`
                  is not provided (default: 0).

    Returns:
        A NumPy array of shape (n, 3) with columns [id, x, y].

    Raises:
        ValueError: If the input positions are not in a valid 2D format, or
                    if the length of provided IDs does not match the number
                    of positions.
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions, dtype=np.float32)

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"Positions must be a 2D array or list of tuples with shape (n, 2), got {positions.shape}")

    num_particles = len(positions)

    if ids is not None:
        if len(ids) != num_particles:
            raise ValueError(f"The number of provided IDs ({len(ids)}) must match the number of positions ({num_particles}).")
        particle_ids = np.array(ids)
    else:
        # Generate sequential IDs if none are provided
        particle_ids = np.arange(start_id, start_id + num_particles)

    return np.column_stack((particle_ids, positions))

def generate_positions(size_x: int, size_y: int, num_particles: int,
                       to_array: bool = True, convert: bool = True) -> Union[List, np.ndarray]:
    """Generate a number of non-repetitive positions.

    Args:
        size_x: Size of the target array in x
        size_y: Size of the target array in y
        num_particles: Number of particles to place in the array
        to_array: If the output should be converted to numpy array
        convert: If the output should be converted to float

    Returns:
        List or numpy array of positions

    Raises:
        PositionGenerationError: If position generation fails
    """
    try:
        if size_x <= 0 or size_y <= 0:
            raise PositionGenerationError("Size dimensions must be positive")
        if num_particles <= 0:
            raise PositionGenerationError("Number of particles must be positive")
        if num_particles > size_x * size_y:
            raise PositionGenerationError("Number of particles cannot exceed grid size")

        rand_points = []
        excluded = set()
        i = 0

        max_attempts = num_particles * 10  # Prevent infinite loops
        attempts = 0

        while i < num_particles and attempts < max_attempts:
            x = random.randrange(0, size_x)
            y = random.randrange(0, size_y)
            attempts += 1

            if (x, y) in excluded:
                continue

            rand_points.append((x, y))
            i += 1
            excluded.add((x, y))

        if i < num_particles:
            raise PositionGenerationError(f"Could only generate {i} unique positions out of {num_particles} requested")

        if to_array:
            if convert:
                rand_points = np.array(rand_points).astype("float32")
            else:
                rand_points = np.array(rand_points)

        return rand_points

    except Exception as e:
        raise PositionGenerationError(f"Failed to generate positions: {str(e)}")

def generate_and_format_positions(
    size_x: int, size_y: int, num_particles: int,
    start_id: int = 0,
    to_array: bool = True,
    convert: bool = True
) -> np.ndarray:
    """
    Convenience function: generate unique positions and format with IDs.

    Returns:
        np.ndarray of shape (num_particles, 3) with columns (id, x, y).
    """
    positions = generate_positions(size_x, size_y, num_particles, to_array=True, convert=convert)
    return format_positions(positions, start_id=start_id)


