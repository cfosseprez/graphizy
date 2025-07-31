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

    def __init__(self, data_shape: List[Tuple[str, type]]):
        """Initialize data interface

        Args:
            data_shape: List of tuples defining data structure

        Raises:
            InvalidDataShapeError: If data shape is invalid
        """


        try:
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

    def to_array(self, data_points: Union[np.ndarray, Dict[str, Any]], aspect: str) -> np.ndarray:
        """
        Convert data points to standardized array format.

        Args:
            data_points: Input data in array or dict format
            aspect: Expected input format ("array" or "dict")

        Returns:
            np.ndarray: Data in standardized array format
        """
        if aspect == "array":
            if not isinstance(data_points, np.ndarray):
                raise InvalidPointArrayError("Expected numpy array for 'array' aspect")

            # Validate array structure
            self.convert(data_points)  # This validates the array
            return data_points

        elif aspect == "dict":
            if isinstance(data_points, dict):
                required_keys = ["id", "x", "y"]
                if not all(k in data_points for k in required_keys):
                    raise InvalidPointArrayError(f"Dict data must contain keys: {required_keys}")

                if len(set(len(v) for v in data_points.values())) > 1:
                    raise InvalidPointArrayError("All dict values must have same length")

                # Convert to array format
                array_data = np.zeros((len(data_points["id"]), len(self.data_shape)))
                array_data[:, self.getidx_id()] = data_points["id"]
                array_data[:, self.getidx_xpos()] = data_points["x"]
                array_data[:, self.getidx_ypos()] = data_points["y"]

                return array_data

            elif isinstance(data_points, np.ndarray):
                # Validate and return
                self.convert(data_points)  # Validates structure
                return data_points
            else:
                raise InvalidPointArrayError("Invalid data type for dict aspect")
        else:
            raise InvalidPointArrayError(f"Unknown aspect: {aspect}")


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

            point_dict = {
                "id": point_array[:, self.getidx_id()],
                "x": point_array[:, self.getidx_xpos()],
                "y": point_array[:, self.getidx_ypos()]
            }

            return point_dict

        except Exception as e:
            raise InvalidPointArrayError(f"Failed to convert point array: {str(e)}")