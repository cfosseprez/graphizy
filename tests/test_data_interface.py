"""
Tests for the Data interface
"""
import pytest
import numpy as np

from graphizy.data_interface import DataInterface
from graphizy.exceptions import (
    GraphCreationError, InvalidDimensionError, InvalidAspectError,
    IgraphMethodError, DrawingError, InvalidPointArrayError
)

class TestDataInterface:
    """Test DataInterface class separately."""

    def test_data_interface_array_validation(self):
        """Test DataInterface array validation."""
        dinter = DataInterface()

        # Test with string IDs (should fail)
        string_id_data = np.array([["a", 10, 20], ["b", 30, 40]], dtype=object)
        with pytest.raises(InvalidPointArrayError, match="must be numeric"):
            dinter.to_array(string_id_data, validate_data=True)

        # Test with insufficient columns
        small_array = np.array([[1, 10]])  # Missing y column
        with pytest.raises(InvalidPointArrayError, match="enough columns"):
            dinter.to_array(small_array, validate_data=True)

        # Test valid array
        valid_array = np.array([[1, 10, 20], [2, 30, 40]], dtype=float)
        result = dinter.to_array(valid_array)
        np.testing.assert_array_equal(result, valid_array)

    def test_data_interface_dict_conversion(self):
        """Test DataInterface dictionary conversion."""
        dinter = DataInterface()

        # Test with proper dict
        dict_data = {"id": [1, 2, 3], "x": [10, 20, 30], "y": [15, 25, 35]}
        result = dinter.to_array(dict_data)
        expected = np.array([[1, 10, 15], [2, 20, 25], [3, 30, 35]])
        np.testing.assert_array_equal(result, expected)

        # Test with missing keys
        incomplete_dict = {"id": [1, 2], "x": [10, 20]}  # Missing 'y'
        with pytest.raises(InvalidPointArrayError, match="Dict data must contain"):
            dinter.to_array(incomplete_dict)

        # Test with mismatched lengths
        mismatched_dict = {"id": [1, 2], "x": [10, 20, 30], "y": [15, 25]}
        with pytest.raises(InvalidPointArrayError, match="same length"):
            dinter.to_array(mismatched_dict)

        # Test with empty data
        empty_dict = {"id": [], "x": [], "y": []}
        with pytest.raises(InvalidPointArrayError, match="cannot be empty"):
            dinter.to_array(empty_dict)

    def test_data_interface_to_dict(self):
        """Test DataInterface array to dict conversion."""
        dinter = DataInterface()

        array_data = np.array([[1, 10, 20], [2, 30, 40]])
        result = dinter.to_dict(array_data)

        expected = {
            "id": np.array([1, 2]),
            "x": np.array([10, 30]),
            "y": np.array([20, 40])
        }

        np.testing.assert_array_equal(result["id"], expected["id"])
        np.testing.assert_array_equal(result["x"], expected["x"])
        np.testing.assert_array_equal(result["y"], expected["y"])

    def test_data_interface_edge_cases(self):
        """Test DataInterface edge cases."""
        dinter = DataInterface()

        # Test with invalid input types
        with pytest.raises(InvalidPointArrayError, match="Invalid data type"):
            dinter.to_array("invalid_string")

        with pytest.raises(InvalidPointArrayError, match="Invalid data type"):
            dinter.to_array(123)

        # Test with None
        with pytest.raises(InvalidPointArrayError):
            dinter.to_array(None)


class TestDataInterfaceCustomDataShape:
    """Test DataInterface with custom data shapes."""

    def test_custom_data_shape(self):
        """Test DataInterface with custom column ordering."""
        # Custom data shape: [id, y, x] instead of [id, x, y]
        custom_shape = [('id', int), ('y', float), ('x', float)]
        dinter = DataInterface(custom_shape)

        # Test array with custom ordering
        array_data = np.array([[1, 20, 10], [2, 40, 30]])  # [id, y, x]
        result = dinter.to_dict(array_data)

        expected = {
            "id": np.array([1, 2]),
            "x": np.array([10, 30]),  # Note: x comes from column 2
            "y": np.array([20, 40])   # Note: y comes from column 1
        }

        np.testing.assert_array_equal(result["id"], expected["id"])
        np.testing.assert_array_equal(result["x"], expected["x"])
        np.testing.assert_array_equal(result["y"], expected["y"])

    def test_custom_data_shape_dict_conversion(self):
        """Test dict to array conversion with custom data shape."""
        custom_shape = [('id', int), ('y', float), ('x', float)]
        dinter = DataInterface(custom_shape)

        dict_data = {"id": [1, 2], "x": [10, 30], "y": [20, 40]}
        result = dinter.to_array(dict_data)

        # Should produce [id, y, x] columns
        expected = np.array([[1, 20, 10], [2, 40, 30]])
        np.testing.assert_array_equal(result, expected)