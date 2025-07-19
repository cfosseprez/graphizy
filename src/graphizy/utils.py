"""
Just some utilities functions
"""
from pathlib import Path
import numpy as np


def setup_output_directory():
    """Create output directory for saving images"""
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_graphizy_input(data_points, aspect="array", data_shape=None, dimension=(1200, 1200),
                            proximity_thresh=None, verbose=True):
    """
    Helper function to validate input data for graphizy operations

    Args:
        data_points: Input data (numpy array or dict)
        aspect: "array" or "dict"
        data_shape: Expected data shape structure
        dimension: Image dimensions (width, height)
        proximity_thresh: Proximity threshold if applicable
        verbose: Print detailed validation info

    Returns:
        dict: Validation results with errors, warnings, and info
    """


    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
        "suggestions": []
    }

    try:
        # Validate dimension FIRST (independent of data)
        if not isinstance(dimension, (tuple, list)) or len(dimension) != 2:
            results["errors"].append("Dimension must be tuple/list of 2 integers")
            results["valid"] = False
        elif dimension[0] <= 0 or dimension[1] <= 0:
            results["errors"].append("Dimension values must be positive")
            results["valid"] = False
        else:
            results["info"]["dimension"] = dimension
        
        # Validate proximity threshold if provided
        if proximity_thresh is not None:
            if proximity_thresh <= 0:
                results["warnings"].append("Proximity threshold should be positive")
            results["info"]["proximity_threshold"] = proximity_thresh

        # Check aspect parameter
        if aspect not in ["array", "dict"]:
            results["errors"].append(f"Invalid aspect '{aspect}'. Must be 'array' or 'dict'")
            results["valid"] = False

        # Validate data_points based on aspect
        if aspect == "array":
            if not isinstance(data_points, np.ndarray):
                results["errors"].append(f"Expected numpy array for aspect 'array', got {type(data_points)}")
                results["valid"] = False
            else:
                # Check array properties
                results["info"]["shape"] = data_points.shape
                results["info"]["dtype"] = str(data_points.dtype)

                # Check for minimum required columns (id, x, y)
                if data_points.ndim != 2:
                    results["errors"].append(f"Data array must be 2D, got {data_points.ndim}D")
                    results["valid"] = False
                elif data_points.shape[1] < 3:
                    results["errors"].append(
                        f"Data array needs at least 3 columns (id, x, y), got {data_points.shape[1]}")
                    results["valid"] = False
                else:
                    # Check for string/object IDs (which cause issues)
                    if data_points.dtype.kind in ['U', 'S', 'O']:
                        results["errors"].append("Object IDs must be numeric, not string type")
                        results["valid"] = False

                    # Check coordinate ranges
                    if data_points.shape[0] > 0:
                        x_coords = data_points[:, 1]  # assuming x is column 1
                        y_coords = data_points[:, 2]  # assuming y is column 2

                        results["info"]["x_range"] = (float(np.min(x_coords)), float(np.max(x_coords)))
                        results["info"]["y_range"] = (float(np.min(y_coords)), float(np.max(y_coords)))
                        results["info"]["num_points"] = data_points.shape[0]

                        # Check if coordinates are within dimension bounds
                        if np.any(x_coords < 0) or np.any(x_coords >= dimension[0]):
                            results["warnings"].append(f"X coordinates outside dimension bounds [0, {dimension[0]})")
                        if np.any(y_coords < 0) or np.any(y_coords >= dimension[1]):
                            results["warnings"].append(f"Y coordinates outside dimension bounds [0, {dimension[1]})")

                        # Check for duplicate coordinates
                        unique_coords = np.unique(np.column_stack((x_coords, y_coords)), axis=0)
                        if len(unique_coords) < len(x_coords):
                            results["warnings"].append(
                                f"Found {len(x_coords) - len(unique_coords)} duplicate coordinate pairs")
                    else:
                        # Empty array case
                        results["info"]["num_points"] = 0

        elif aspect == "dict":
            if isinstance(data_points, dict):
                # Check required keys
                required_keys = ["id", "x", "y"]
                missing_keys = [key for key in required_keys if key not in data_points]
                if missing_keys:
                    results["errors"].append(f"Missing required keys: {missing_keys}")
                    results["valid"] = False
                else:
                    # Check array lengths match
                    lengths = {key: len(data_points[key]) for key in required_keys}
                    if len(set(lengths.values())) > 1:
                        results["errors"].append(f"Mismatched array lengths: {lengths}")
                        results["valid"] = False
                    else:
                        results["info"]["num_points"] = lengths["x"]
                        if lengths["x"] > 0:
                            results["info"]["x_range"] = (float(min(data_points["x"])), float(max(data_points["x"])))
                            results["info"]["y_range"] = (float(min(data_points["y"])), float(max(data_points["y"])))

                            # Check coordinate bounds
                            if any(x < 0 or x >= dimension[0] for x in data_points["x"]):
                                results["warnings"].append(f"X coordinates outside dimension bounds [0, {dimension[0]})")
                            if any(y < 0 or y >= dimension[1] for y in data_points["y"]):
                                results["warnings"].append(f"Y coordinates outside dimension bounds [0, {dimension[1]})")

            elif isinstance(data_points, np.ndarray):
                results["info"]["note"] = "Numpy array provided for dict aspect - will be auto-converted"
                # Validate as array first
                temp_result = validate_graphizy_input(data_points, "array", data_shape, dimension, proximity_thresh,
                                                      False)
                if not temp_result["valid"]:
                    results["errors"].extend(temp_result["errors"])
                    results["valid"] = False
            else:
                results["errors"].append(f"For dict aspect, expected dict or numpy array, got {type(data_points)}")
                results["valid"] = False

        # Add suggestions based on findings
        if results["valid"]:
            if results["info"].get("num_points", 0) < 3:
                results["suggestions"].append("Need at least 3 points for Delaunay triangulation")
            if results["info"].get("num_points", 0) > 10000:
                results["suggestions"].append("Large number of points - consider performance implications")

        if verbose:
            print("=== GRAPHIZY INPUT VALIDATION ===")
            print(f"Valid: {results['valid']}")

            if results["errors"]:
                print("\nERRORS:")
                for error in results["errors"]:
                    print(f"  ❌ {error}")

            if results["warnings"]:
                print("\nWARNINGS:")
                for warning in results["warnings"]:
                    print(f"  ⚠️  {warning}")

            if results["info"]:
                print("\nINFO:")
                for key, value in results["info"].items():
                    print(f"  ℹ️  {key}: {value}")

            if results["suggestions"]:
                print("\nSUGGESTIONS:")
                for suggestion in results["suggestions"]:
                    print(f"  💡 {suggestion}")

    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Validation failed with exception: {str(e)}")

    return results