"""
Data interface for Graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import logging
import time
import random
import timeit
from typing import List, Tuple, Dict, Any, Union, Optional
from scipy.spatial.distance import cdist, pdist, squareform
import numpy as np
from collections import defaultdict, deque

from graphizy.exceptions import (
    InvalidPointArrayError, SubdivisionError, TriangulationError,
    GraphCreationError, PositionGenerationError, DependencyError,
    IgraphMethodError, handle_subdivision_bounds_error, InvalidDataShapeError
)



class DataInterface:
    """Interface for handling different data formats"""

    def __init__(self, data_shape: Optional[List[Tuple[str, type]]] = None):
        """Initialize data interface

        Args:
            data_shape: List of tuples defining data structure

        Raises:
            InvalidDataShapeError: If data shape is invalid
        """


        try:
            # Set default data shape if none provided
            if data_shape is None:
                data_shape = [('id', int), ('x', float), ('y', float)]

            # Validate data_shape
            if not isinstance(data_shape, list):
                raise InvalidDataShapeError("Data shape input should be a list")
            if not data_shape:
                raise InvalidDataShapeError("Data shape cannot be empty")
            if not all(isinstance(item, tuple) and len(item) == 2 for item in data_shape):
                raise InvalidDataShapeError("Data shape elements should be tuples of (name, type)")

            # Keep data_shape
            self.data_shape = data_shape

            # Find data indexes
            data_idx = {}
            for i, variable in enumerate(data_shape):
                if not isinstance(variable[0], str):
                    raise InvalidDataShapeError("Variable names must be strings")
                data_idx[variable[0]] = i

            self.data_idx = data_idx

            # Validate required fields
            required_fields = ['id', 'x', 'y']
            missing_fields = [field for field in required_fields if field not in self.data_idx]
            if missing_fields:
                raise InvalidDataShapeError(f"Required fields missing: {missing_fields}")

        except Exception as e:
            raise InvalidDataShapeError(f"Failed to initialize data interface: {str(e)}")

    def getidx_id(self) -> int:
        """Get index of id column"""
        return self.data_idx["id"]

    def getidx_xpos(self) -> int:
        """Get index of x position column"""
        return self.data_idx["x"]

    def getidx_ypos(self) -> int:
        """Get index of y position column"""
        return self.data_idx["y"]

    def validate_array(self, point_array: np.ndarray) -> None:
        """
        Validate that array has the correct structure for this data interface.

        Args:
            point_array: Array to validate

        Raises:
            InvalidPointArrayError: If array structure is invalid
        """
        try:
            if point_array is None or point_array.size == 0:
                raise InvalidPointArrayError("Point array cannot be None or empty")
            if point_array.ndim != 2:
                raise InvalidPointArrayError("Point array must be 2D")

            required_cols = max(self.getidx_id(), self.getidx_xpos(), self.getidx_ypos()) + 1
            if point_array.shape[1] < required_cols:
                raise InvalidPointArrayError(
                    f"Point array doesn't have enough columns. "
                    f"Need at least {required_cols}, got {point_array.shape[1]}"
                )

            id_col_idx = self.getidx_id()
            if point_array.shape[1] > id_col_idx:
                id_column_dtype = point_array[:, id_col_idx].dtype
                if not np.issubdtype(id_column_dtype, np.number):
                     raise InvalidPointArrayError("Object IDs must be numeric, not string or object type")

        except Exception as e:
            if isinstance(e, InvalidPointArrayError):
                raise
            raise InvalidPointArrayError(f"Array validation failed: {str(e)}")

    def to_array(self, data_points: Union[np.ndarray, Dict[str, Any]], validate_data: bool = False) -> np.ndarray:
        """
        Convert data points to standardized array format.

        Automatically detects input format and converts accordingly.

        Args:
            data_points: Input data in array or dict format
            validate_data: Whether to validate the data (careful at each call this will degrade the performances)

        Returns:
            np.ndarray: Data in standardized array format

        Raises:
            InvalidPointArrayError: If conversion fails or data is invalid
        """
        try:
            if isinstance(data_points, np.ndarray):
                if validate_data:
                    self.validate_array(data_points)
                return data_points

            elif isinstance(data_points, dict):
                # Convert dictionary to array
                required_keys = ["id", "x", "y"]
                if not all(k in data_points for k in required_keys):
                    raise InvalidPointArrayError(f"Dict data must contain keys: {required_keys}")

                if not data_points["id"]:
                    raise InvalidPointArrayError("Input dictionary cannot be empty")

                # Check all values are lists/arrays of same length
                lengths = [len(v) for v in data_points.values()]
                if len(set(lengths)) > 1:
                    raise InvalidPointArrayError("All dict values must have same length")

                # Build the array using only the columns present in the input dictionary,
                # but ordered according to the instance's data_shape.
                ordered_columns = []
                for attr_name, _ in self.data_shape:
                    if attr_name in data_points:
                        ordered_columns.append(data_points[attr_name])

                if not ordered_columns:
                    raise InvalidPointArrayError(
                        "Could not construct array from dictionary; no matching keys found in data_shape.")

                return np.column_stack(ordered_columns)

            else:
                raise InvalidPointArrayError(
                    f"Invalid data type: {type(data_points)}. "
                    f"Expected numpy array or dictionary."
                )

        except Exception as e:
            if isinstance(e, InvalidPointArrayError):
                raise
            raise InvalidPointArrayError(f"Failed to convert data to array format: {str(e)}")

    def to_dict(self, point_array: np.ndarray) -> Dict[str, Any]:
        """Convert point array to dictionary format

        Args:
            point_array: Array to convert

        Returns:
            Dictionary with id, x, y keys

        Raises:
            InvalidPointArrayError: If conversion fails
        """
        try:
            if point_array is None or point_array.size == 0:
                raise InvalidPointArrayError("Point array cannot be None or empty")
            if point_array.ndim != 2:
                raise InvalidPointArrayError("Point array must be 2D")
            if point_array.shape[1] < max(self.getidx_id(), self.getidx_xpos(), self.getidx_ypos()) + 1:
                raise InvalidPointArrayError("Point array doesn't have enough columns for the specified data shape")

            max_required_index = max(self.getidx_id(), self.getidx_xpos(), self.getidx_ypos())
            if point_array.shape[1] <= max_required_index:
                raise InvalidPointArrayError(
                    f"Point array has {point_array.shape[1]} columns, but the data shape "
                    f"requires an index of at least {max_required_index}."
                )

            point_dict = {
                "id": point_array[:, self.getidx_id()],
                "x": point_array[:, self.getidx_xpos()],
                "y": point_array[:, self.getidx_ypos()]
            }

            return point_dict

        except Exception as e:
            raise InvalidPointArrayError(f"Failed to convert point array: {str(e)}")