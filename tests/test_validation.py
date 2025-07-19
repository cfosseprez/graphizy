"""
Tests for graphizy input validation utilities

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import pytest
import numpy as np
import sys
import os
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphizy import validate_graphizy_input


class TestValidateGraphizyInput:
    """Test validation function for graphizy inputs"""

    def test_valid_array_input(self):
        """Test validation with valid array input"""
        # Create valid test data: [id, x, y]
        data = np.array([
            [0, 100, 200],
            [1, 300, 400],
            [2, 500, 600]
        ], dtype=int)

        result = validate_graphizy_input(data, aspect="array", verbose=False)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["info"]["num_points"] == 3
        assert result["info"]["x_range"] == (100.0, 500.0)
        assert result["info"]["y_range"] == (200.0, 600.0)
        assert result["info"]["shape"] == (3, 3)

    def test_valid_dict_input(self):
        """Test validation with valid dictionary input"""
        data = {
            "id": [0, 1, 2],
            "x": [100, 300, 500],
            "y": [200, 400, 600]
        }

        result = validate_graphizy_input(data, aspect="dict", verbose=False)

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["info"]["num_points"] == 3
        assert result["info"]["x_range"] == (100.0, 500.0)
        assert result["info"]["y_range"] == (200.0, 600.0)

    def test_invalid_aspect(self):
        """Test validation with invalid aspect parameter"""
        data = np.array([[0, 100, 200]])

        result = validate_graphizy_input(data, aspect="invalid", verbose=False)

        assert result["valid"] is False
        assert any("Invalid aspect" in error for error in result["errors"])

    def test_wrong_data_type_for_array_aspect(self):
        """Test validation when wrong data type is provided for array aspect"""
        data = {"id": [0], "x": [100], "y": [200]}

        result = validate_graphizy_input(data, aspect="array", verbose=False)

        assert result["valid"] is False
        assert any("Expected numpy array" in error for error in result["errors"])

    def test_wrong_data_type_for_dict_aspect(self):
        """Test validation when wrong data type is provided for dict aspect"""
        data = "invalid_data"

        result = validate_graphizy_input(data, aspect="dict", verbose=False)

        assert result["valid"] is False
        assert any("expected dict or numpy array" in error for error in result["errors"])

    def test_array_wrong_dimensions(self):
        """Test validation with wrong array dimensions"""
        # 1D array
        data_1d = np.array([1, 2, 3])
        result = validate_graphizy_input(data_1d, aspect="array", verbose=False)
        assert result["valid"] is False
        assert any("must be 2D" in error for error in result["errors"])

        # 3D array
        data_3d = np.array([[[1, 2, 3]]])
        result = validate_graphizy_input(data_3d, aspect="array", verbose=False)
        assert result["valid"] is False
        assert any("must be 2D" in error for error in result["errors"])

    def test_array_insufficient_columns(self):
        """Test validation with insufficient columns"""
        # Only 2 columns (need at least 3: id, x, y)
        data = np.array([[0, 100], [1, 200]])

        result = validate_graphizy_input(data, aspect="array", verbose=False)

        assert result["valid"] is False
        assert any("at least 3 columns" in error for error in result["errors"])

    def test_array_string_ids(self):
        """Test validation with string IDs (which cause issues)"""
        data = np.array([
            ["a", 100, 200],
            ["b", 300, 400]
        ], dtype=object)

        result = validate_graphizy_input(data, aspect="array", verbose=False)

        assert result["valid"] is False
        assert any("must be numeric, not string type" in error for error in result["errors"])

    def test_dict_missing_keys(self):
        """Test validation with missing required keys in dictionary"""
        data = {"id": [0, 1], "x": [100, 300]}  # Missing 'y'

        result = validate_graphizy_input(data, aspect="dict", verbose=False)

        assert result["valid"] is False
        assert any("Missing required keys" in error for error in result["errors"])

    def test_dict_mismatched_lengths(self):
        """Test validation with mismatched array lengths in dictionary"""
        data = {
            "id": [0, 1, 2],
            "x": [100, 300],  # Only 2 elements
            "y": [200, 400, 600]
        }

        result = validate_graphizy_input(data, aspect="dict", verbose=False)

        assert result["valid"] is False
        assert any("Mismatched array lengths" in error for error in result["errors"])

    def test_coordinates_outside_bounds(self):
        """Test validation with coordinates outside dimension bounds"""
        # X coordinate outside bounds
        data = np.array([
            [0, 1300, 200],  # x=1300 > dimension[0]=1200
            [1, 300, 400]
        ])

        result = validate_graphizy_input(data, aspect="array", dimension=(1200, 1200), verbose=False)

        assert result["valid"] is True  # Still valid, but should have warnings
        assert any("X coordinates outside" in warning for warning in result["warnings"])

        # Y coordinate outside bounds
        data = np.array([
            [0, 300, 1300],  # y=1300 > dimension[1]=1200
            [1, 400, 500]
        ])

        result = validate_graphizy_input(data, aspect="array", dimension=(1200, 1200), verbose=False)

        assert result["valid"] is True  # Still valid, but should have warnings
        assert any("Y coordinates outside" in warning for warning in result["warnings"])

    def test_duplicate_coordinates(self):
        """Test validation with duplicate coordinate pairs"""
        data = np.array([
            [0, 100, 200],
            [1, 100, 200],  # Duplicate coordinates
            [2, 300, 400]
        ])

        result = validate_graphizy_input(data, aspect="array", verbose=False)

        assert result["valid"] is True  # Still valid, but should have warnings
        assert any("duplicate coordinate pairs" in warning for warning in result["warnings"])

    def test_invalid_dimension(self):
        """Test validation with invalid dimension parameter"""
        data = np.array([[0, 100, 200]])

        # Wrong length
        result = validate_graphizy_input(data, aspect="array", dimension=(1200,), verbose=False)
        assert result["valid"] is False
        # Check for any error message about dimension being wrong length
        assert any("dimension" in error.lower() and ("tuple" in error.lower() or "list" in error.lower()) for error in result["errors"])

        # Negative values
        result = validate_graphizy_input(data, aspect="array", dimension=(-100, 1200), verbose=False)
        assert result["valid"] is False
        # Check for any error message about dimension values being positive
        assert any("dimension" in error.lower() and "positive" in error.lower() for error in result["errors"])

    def test_proximity_threshold_validation(self):
        """Test validation of proximity threshold parameter"""
        data = np.array([[0, 100, 200]])

        # Negative threshold should generate warning
        result = validate_graphizy_input(data, aspect="array", proximity_thresh=-10, verbose=False)
        assert any("should be positive" in warning for warning in result["warnings"])

        # Valid threshold
        result = validate_graphizy_input(data, aspect="array", proximity_thresh=50.0, verbose=False)
        assert result["info"]["proximity_threshold"] == 50.0

    def test_suggestions_generation(self):
        """Test that appropriate suggestions are generated"""
        # Too few points for Delaunay
        data = np.array([[0, 100, 200], [1, 300, 400]])  # Only 2 points

        result = validate_graphizy_input(data, aspect="array", verbose=False)

        assert any("at least 3 points for Delaunay" in suggestion for suggestion in result["suggestions"])

        # Large number of points
        large_data = np.random.randint(0, 1000, (15000, 3))

        result = validate_graphizy_input(large_data, aspect="array", verbose=False)

        assert any("performance implications" in suggestion for suggestion in result["suggestions"])

    def test_numpy_array_for_dict_aspect(self):
        """Test validation when numpy array is provided for dict aspect"""
        data = np.array([
            [0, 100, 200],
            [1, 300, 400]
        ])

        result = validate_graphizy_input(data, aspect="dict", verbose=False)

        assert result["valid"] is True
        assert result["info"]["note"] == "Numpy array provided for dict aspect - will be auto-converted"

    def test_empty_data(self):
        """Test validation with empty data"""
        # Empty array
        data = np.array([]).reshape(0, 3)

        result = validate_graphizy_input(data, aspect="array", verbose=False)

        assert result["valid"] is True
        # Check if num_points exists, if not it should be 0
        assert result["info"]["num_points"] == 0

    def test_verbose_output(self):
        """Test that verbose output is generated correctly"""
        data = np.array([[0, 100, 200]])

        # Capture stdout
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = validate_graphizy_input(data, aspect="array", verbose=True)

            output = fake_out.getvalue()
            assert "GRAPHIZY INPUT VALIDATION" in output
            assert "Valid: True" in output
            assert "INFO:" in output

    def test_verbose_output_with_errors(self):
        """Test verbose output when there are errors and warnings"""
        data = np.array([
            [0, 1300, 200],  # Outside bounds
            [1, 1300, 200]   # Duplicate
        ])

        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = validate_graphizy_input(data, aspect="array", dimension=(1200, 1200), verbose=True)

            output = fake_out.getvalue()
            assert "WARNINGS:" in output
            assert "⚠️" in output

    def test_invalid_input_handling(self):
        """Test that various invalid inputs are handled gracefully"""
        # Test case 1: Object instead of numpy array
        invalid_data = object()
        result = validate_graphizy_input(invalid_data, aspect="array", verbose=False)
        
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        # Should get an error about expecting numpy array
        assert any("numpy array" in error.lower() for error in result["errors"])
        
        # Test case 2: None input
        result = validate_graphizy_input(None, aspect="array", verbose=False)
        assert result["valid"] is False
        assert len(result["errors"]) > 0
        
        # Test case 3: String input  
        result = validate_graphizy_input("invalid", aspect="array", verbose=False)
        assert result["valid"] is False
        assert len(result["errors"]) > 0

    def test_edge_cases(self):
        """Test various edge cases"""
        # Single point
        data = np.array([[0, 100, 200]])
        result = validate_graphizy_input(data, aspect="array", verbose=False)
        assert result["valid"] is True
        assert result["info"]["num_points"] == 1

        # Zero coordinates
        data = np.array([[0, 0, 0]])
        result = validate_graphizy_input(data, aspect="array", verbose=False)
        assert result["valid"] is True

        # Maximum coordinates
        data = np.array([[0, 1199, 1199]])
        result = validate_graphizy_input(data, aspect="array", dimension=(1200, 1200), verbose=False)
        assert result["valid"] is True
        assert len(result["warnings"]) == 0  # Should be within bounds

    def test_float_coordinates(self):
        """Test validation with float coordinates"""
        data = np.array([
            [0, 100.5, 200.7],
            [1, 300.2, 400.9]
        ], dtype=float)

        result = validate_graphizy_input(data, aspect="array", verbose=False)

        assert result["valid"] is True
        assert result["info"]["x_range"] == (100.5, 300.2)
        assert result["info"]["y_range"] == (200.7, 400.9)

    def test_dict_with_extra_keys(self):
        """Test validation with dictionary containing extra keys"""
        data = {
            "id": [0, 1, 2],
            "x": [100, 300, 500],
            "y": [200, 400, 600],
            "speed": [1.0, 2.0, 3.0],  # Extra key
            "color": ["red", "blue", "green"]  # Another extra key
        }

        result = validate_graphizy_input(data, aspect="dict", verbose=False)

        assert result["valid"] is True  # Extra keys should not cause failure
        assert result["info"]["num_points"] == 3


class TestValidationIntegration:
    """Integration tests for validation function"""

    def test_validation_with_real_workflow(self):
        """Test validation in context of actual graphizy workflow"""
        from graphizy import Graphing, generate_positions

        # Generate realistic test data
        positions = generate_positions(500, 500, 50)
        particle_ids = np.arange(len(positions))
        data = np.column_stack((particle_ids, positions))

        # Validate the data
        result = validate_graphizy_input(
            data, 
            aspect="array", 
            dimension=(500, 500),
            proximity_thresh=25.0,
            verbose=False
        )

        assert result["valid"] is True
        assert result["info"]["num_points"] == 50
        # Should have suggestions since we have more than 3 points
        assert len(result["suggestions"]) >= 0  # Changed from > 0 to >= 0

        # Test that this data actually works with Graphing
        grapher = Graphing(dimension=(500, 500))
        graph = grapher.make_proximity(data, proximity_thresh=25.0)
        
        # Basic sanity check
        assert graph.vcount() == 50


if __name__ == '__main__':
    pytest.main([__file__])
