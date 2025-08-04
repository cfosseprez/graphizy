"""
Position generation and formatting utilities for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import random
from typing import List, Tuple, Union, Optional, Dict, Callable, Any
import numpy as np

from graphizy.exceptions import PositionGenerationError


def format_positions(
    positions: Union[np.ndarray, List[Tuple[float, ...]]],
    ids: Optional[Union[np.ndarray, List]] = None,
    start_id: int = 0
) -> np.ndarray:
    """
    Formats positions into the standard graphizy data array by adding IDs.
    This function is now more flexible and can handle positions with extra columns.

    Args:
        positions: A NumPy array of shape (n, m) or a list of (x, y, ...) tuples,
                   where m >= 2.
        ids: An optional list or NumPy array of IDs. If provided, its length
             must match the number of positions.
        start_id: The starting integer for sequential IDs, used only if `ids`
                  is not provided (default: 0).

    Returns:
        A NumPy array of shape (n, m+1) with columns [id, x, y, ...].

    Raises:
        ValueError: If the input positions are not in a valid 2D format, or
                    if the length of provided IDs does not match the number
                    of positions.
    """
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions, dtype=np.float32)

    if positions.ndim != 2:
        raise ValueError(f"Positions must be a 2D array, got {positions.ndim}D shape")

    if positions.shape[1] < 2:
        raise ValueError(f"Positions must have at least 2 columns for x and y, got {positions.shape[1]}")

    num_particles = len(positions)

    if ids is not None:
        if len(ids) != num_particles:
            raise ValueError(f"The number of provided IDs ({len(ids)}) must match the number of positions ({num_particles}).")
        particle_ids = np.array(ids)
    else:
        # Generate sequential IDs if none are provided
        particle_ids = np.arange(start_id, start_id + num_particles)

    # np.column_stack is perfect for prepending the ID column to the existing position data
    return np.column_stack((particle_ids, positions))


def generate_positions(
    size_x: int,
    size_y: int,
    num_particles: int,
    to_array: bool = True,
    add_more: Optional[Dict[str, Callable[[], Any]]] = None,
    convert: bool = True
) -> Union[List, np.ndarray]:
    """
    Generate a number of non-repetitive positions with optional extra attributes.

    Args:
        size_x: Size of the target array in x.
        size_y: Size of the target array in y.
        num_particles: Number of particles to place in the array.
        to_array: If the output should be converted to a numpy array.
        add_more: A dictionary defining extra attributes to generate.
                  Keys are attribute names (str), and values are functions
                  that return a random value for that attribute.
                  Example: `{'velocity': lambda: random.uniform(0, 5)}`
        convert: If the output should be converted to float (when using to_array).

    Returns:
        List or numpy array of positions. Each position is a tuple:
        (x, y, extra_val_1, extra_val_2, ...).

    Raises:
        PositionGenerationError: If position generation fails.
    """
    try:
        if size_x <= 0 or size_y <= 0:
            raise PositionGenerationError("Size dimensions must be positive")
        if num_particles <= 0:
            raise PositionGenerationError("Number of particles must be positive")
        if num_particles > size_x * size_y:
            raise PositionGenerationError("Number of particles cannot exceed grid size")

        rand_points = []

        # For dense grids, it's faster to generate all points and sample.
        if num_particles > (size_x * size_y) / 2:
            all_points = [(x, y) for x in range(size_x) for y in range(size_y)]
            sampled_points = random.sample(all_points, num_particles)

            for x, y in sampled_points:
                point_data = [x, y]
                if add_more:
                    for generator_func in add_more.values():
                        point_data.append(generator_func())
                rand_points.append(tuple(point_data))
        else:
            # For sparse grids, the original method is fine and uses less memory.
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

                point_data = [x, y]
                if add_more:
                    for generator_func in add_more.values():
                        point_data.append(generator_func())

                rand_points.append(tuple(point_data))
                i += 1
                excluded.add((x, y))

            if i < num_particles:
                raise PositionGenerationError(f"Could only generate {i} unique positions out of {num_particles} requested")

        if to_array:
            if convert:
                rand_points = np.array(rand_points, dtype=np.float32)
            else:
                rand_points = np.array(rand_points)

        return rand_points

    except Exception as e:
        raise PositionGenerationError(f"Failed to generate positions: {str(e)}")


def generate_and_format_positions(
    size_x: int, size_y: int, num_particles: int,
    start_id: int = 0,
    add_more: Optional[Dict[str, Callable[[], Any]]] = None,
    to_array=True,
    convert: bool = True
) -> np.ndarray:
    """
    Convenience function: generate unique positions and format with IDs.

    Args:
        size_x: Size of the target array in x.
        size_y: Size of the target array in y.
        num_particles: Number of particles to place in the array.
        start_id: The starting integer for sequential IDs.
        add_more: A dictionary defining extra attributes to generate.
                  Example: `{'velocity': lambda: random.uniform(0, 5)}`
        to_array: If the output should be converted to a numpy array.
        convert: If the output should be converted to float.

    Returns:
        np.ndarray of shape (num_particles, n) with columns (id, x, y, ...).
    """
    positions = generate_positions(
        size_x, size_y, num_particles,
        to_array=to_array, add_more=add_more, convert=convert
    )
    # The new format_positions can handle the extra columns
    return format_positions(positions, start_id=start_id)