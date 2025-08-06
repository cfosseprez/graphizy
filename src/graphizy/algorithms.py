"""
Graph algorithms for graphizy

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
    IgraphMethodError
)
from graphizy.data_interface import DataInterface
from .exceptions import handle_subdivision_bounds_error, InvalidDataShapeError

try:
    import cv2
except ImportError:
    raise DependencyError("OpenCV is required but not installed. Install with: pip install opencv-python")

try:
    import igraph as ig
except ImportError:
    raise DependencyError("python-igraph is required but not installed. Install with: pip install python-igraph")

try:
    from scipy.spatial.distance import pdist, squareform
except ImportError:
    raise DependencyError("scipy is required but not installed. Install with: pip install scipy")


def normalize_distance_metric(metric: str) -> str:
    """
    Normalize distance metric names to scipy-compatible format.
    
    Args:
        metric: Distance metric name
        
    Returns:
        Scipy-compatible metric name
    """
    metric_mapping = {
        'manhattan': 'cityblock',
        'l1': 'cityblock',
        'euclidean': 'euclidean',
        'l2': 'euclidean',
        'chebyshev': 'chebyshev',
        'linf': 'chebyshev'
    }
    return metric_mapping.get(metric.lower(), metric.lower())


def normalize_id(obj_id: Any) -> str:
    """
    Normalize object ID to consistent string format for real-time applications.
    
    Optimized for performance:
    - Handles int, float, str inputs
    - Converts float IDs like 1.0, 2.0 to "1", "2"  
    - Preserves non-integer floats as-is
    
    Args:
        obj_id: Object ID of any type
        
    Returns:
        Normalized string ID
    """
    if isinstance(obj_id, str):
        return obj_id
    elif isinstance(obj_id, (int, np.integer)):
        return str(int(obj_id))
    elif isinstance(obj_id, (float, np.floating)):
        # Check if it's an integer float (e.g., 1.0, 2.0)
        if obj_id.is_integer():
            return str(int(obj_id))
        else:
            return str(obj_id)
    else:
        return str(obj_id)





def make_subdiv(point_array: np.ndarray, dimensions: Union[List, Tuple],
                do_print: bool = False) -> Any:
    """Make the opencv subdivision with enhanced error handling

    Args:
        point_array: A numpy array of the points to add
        dimensions: The dimension of the image (width, height)
        do_print: Whether to print debug information

    Returns:
        An opencv subdivision object

    Raises:
        SubdivisionError: If subdivision creation fails
    """
    logger = logging.getLogger('graphizy.algorithms.make_subdiv')

    try:
        # Input validation with enhanced error reporting
        if point_array is None or point_array.size == 0:
            raise SubdivisionError("Point array cannot be None or empty", point_array, dimensions)

        if len(dimensions) != 2:
            raise SubdivisionError("Dimensions must be a tuple/list of 2 values", point_array, dimensions)

        if dimensions[0] <= 0 or dimensions[1] <= 0:
            raise SubdivisionError("Dimensions must be positive", point_array, dimensions)

        width, height = dimensions
        logger.debug(f"make_subdiv: {len(point_array)} points, dimensions {dimensions}")
        logger.debug(
            f"Point ranges: X[{point_array[:, 0].min():.1f}, {point_array[:, 0].max():.1f}], Y[{point_array[:, 1].min():.1f}, {point_array[:, 1].max():.1f}]")

        # Check type and convert if needed
        if not isinstance(point_array.flat[0], (np.floating, float)):
            logger.warning(f"Converting points from {type(point_array.flat[0])} to float32")
            if isinstance(point_array, np.ndarray):
                point_array = point_array.astype("float32")
            else:
                particle_stack = [[float(x), float(y)] for x, y in point_array]
                point_array = np.array(particle_stack)

        # Enhanced bounds checking with detailed error reporting
        # Validate X coordinates
        if np.any(point_array[:, 0] < 0):
            bad_points = point_array[point_array[:, 0] < 0]
            raise SubdivisionError(f"Found {len(bad_points)} points with X < 0", point_array, dimensions)

        if np.any(point_array[:, 0] >= width):

            handle_subdivision_bounds_error(point_array, dimensions, 'x')

        # Validate Y coordinates
        if np.any(point_array[:, 1] < 0):
            bad_points = point_array[point_array[:, 1] < 0]
            raise SubdivisionError(f"Found {len(bad_points)} points with Y < 0", point_array, dimensions)

        if np.any(point_array[:, 1] >= height):

            handle_subdivision_bounds_error(point_array, dimensions, 'y')

        # Timer
        timer = time.time()

        # Create rectangle (normal coordinate system: width, height)
        rect = (0, 0, width, height)
        logger.debug(f"Creating OpenCV rectangle: {rect}")

        if do_print:
            unique_points = len(np.unique(point_array, axis=0))
            print(f"Processing {len(point_array)} positions ({unique_points} unique)")
            print(f"Rectangle: {rect}")
            outside_count = (point_array[:, 0] >= width).sum() + (point_array[:, 1] >= height).sum()
            print(f"Points outside bounds: {outside_count}")

        # Create subdivision
        subdiv = cv2.Subdiv2D(rect)

        # Insert points into subdiv with error tracking
        logger.debug(f"Inserting {len(point_array)} points into subdivision")
        points_list = [tuple(point) for point in point_array]

        failed_insertions = 0
        for i, point in enumerate(points_list):
            try:
                subdiv.insert(point)
            except cv2.error as e:
                failed_insertions += 1
                logger.warning(f"Failed to insert point {i} {point}: {e}")
                continue

        if failed_insertions > 0:
            logger.warning(f"Failed to insert {failed_insertions}/{len(points_list)} points")
            if failed_insertions == len(points_list):
                raise SubdivisionError("Failed to insert all points", point_array, dimensions)

        elapsed_time = round((time.time() - timer) * 1000, 3)
        logger.debug(f"Subdivision creation took {elapsed_time}ms")

        return subdiv

    except SubdivisionError:
        # Re-raise SubdivisionError as-is (they already have context)
        raise
    except Exception as e:
        # Convert other exceptions to SubdivisionError with context
        error = SubdivisionError(f"Failed to create subdivision: {str(e)}", point_array, dimensions,
                                 original_exception=e)
        error.log_error()
        raise error

def make_delaunay(subdiv: Any) -> np.ndarray:
    """Return a Delaunay triangulation

    Args:
        subdiv: An opencv subdivision

    Returns:
        A triangle list

    Raises:
        TriangulationError: If triangulation fails
    """
    try:
        if subdiv is None:
            raise TriangulationError("Subdivision cannot be None")

        triangle_list = subdiv.getTriangleList()

        if len(triangle_list) == 0:
            logging.warning("No triangles found in subdivision")

        return triangle_list

    except Exception as e:
        raise TriangulationError(f"Failed to create Delaunay triangulation: {str(e)}")


def get_delaunay(point_array: np.ndarray, dim: Union[List, Tuple]) -> np.ndarray:
    """Make the delaunay triangulation of set of points

    Args:
        point_array: Array of points
        dim: Dimensions

    Returns:
        Triangle list

    Raises:
        TriangulationError: If triangulation fails
    """
    try:
        subdiv = make_subdiv(point_array, dim)
        return make_delaunay(subdiv)
    except Exception as e:
        raise TriangulationError(f"Failed to get Delaunay triangulation: {str(e)}")


def find_vertex(trilist: List, subdiv: Any, g: Any) -> Any:
    """Find vertices in triangulation and add edges to graph

    Args:
        trilist: List of triangles
        subdiv: OpenCV subdivision
        g: igraph Graph object

    Returns:
        Modified graph

    Raises:
        GraphCreationError: If vertex finding fails
    """
    try:
        if trilist is None or len(trilist) == 0:
            raise GraphCreationError("Triangle list cannot be empty")
        if subdiv is None:
            raise GraphCreationError("Subdivision cannot be None")
        if g is None:
            raise GraphCreationError("Graph cannot be None")

        for tri in trilist:
            if len(tri) != 6:
                logging.warning(f"Invalid triangle format: expected 6 values, got {len(tri)}")
                continue

            try:
                vertex1, _ = subdiv.findNearest((tri[0], tri[1]))
                vertex2, _ = subdiv.findNearest((tri[2], tri[3]))
                vertex3, _ = subdiv.findNearest((tri[4], tri[5]))

                # -4 because https://stackoverflow.com/a/52377891/18493005
                edges = [
                    (vertex1 - 4, vertex2 - 4),
                    (vertex2 - 4, vertex3 - 4),
                    (vertex3 - 4, vertex1 - 4),
                ]

                # Validate vertex indices
                max_vertex = g.vcount()
                valid_edges = []
                for edge in edges:
                    if 0 <= edge[0] < max_vertex and 0 <= edge[1] < max_vertex:
                        valid_edges.append(edge)
                    else:
                        logging.warning(f"Invalid edge {edge}, graph has {max_vertex} vertices")

                if valid_edges:
                    g.add_edges(valid_edges)

            except Exception as e:
                logging.warning(f"Failed to process triangle {tri}: {e}")
                continue

        return g

    except Exception as e:
        raise GraphCreationError(f"Failed to find vertices: {str(e)}")


def _are_points_collinear(points, tolerance=1e-10):
    """
    Check if points are approximately collinear

    Args:
        points: numpy array of shape (n, 2) with x, y coordinates
        tolerance: tolerance for collinearity check

    Returns:
        bool: True if points are collinear
    """
    if len(points) < 3:
        return True

    # Use cross product to check collinearity
    # For points A, B, C: they're collinear if (B-A) × (C-A) ≈ 0
    A, B, C = points[0], points[1], points[2]

    # Cross product in 2D: (B-A) × (C-A) = (B_x-A_x)(C_y-A_y) - (B_y-A_y)(C_x-A_x)
    cross_product = ((B[0] - A[0]) * (C[1] - A[1]) -
                     (B[1] - A[1]) * (C[0] - A[0]))

    return abs(cross_product) < tolerance

def graph_delaunay(graph: Any, subdiv: Any, trilist: List) -> Any:
    """From CV to original ID and igraph

    Args:
        graph: igraph object
        subdiv: OpenCV subdivision
        trilist: List of triangles

    Returns:
        Modified graph

    Raises:
        GraphCreationError: If graph creation fails
    """
    try:
        if graph is None:
            raise GraphCreationError("Graph cannot be None")
        if subdiv is None:
            raise GraphCreationError("Subdivision cannot be None")
        if trilist is None or len(trilist) == 0:
            num_vertices = len(graph.vs)

            if num_vertices < 3:
                raise GraphCreationError(
                    f"Delaunay triangulation requires at least 3 points, got {num_vertices}. "
                    f"Provide more points for meaningful triangulation."
                )
            elif num_vertices == 3:
                # Special case: exactly 3 points should form 1 triangle
                # Check if points are collinear
                positions = np.array([(v["x"], v["y"]) for v in graph.vs])

                if _are_points_collinear(positions):
                    raise GraphCreationError(
                        "Cannot create Delaunay triangulation: the 3 points are collinear. "
                        "Provide points that form a proper triangle."
                    )
                else:
                    # Create the single triangle manually
                    logging.warning("Creating manual triangle for 3-point case")
                    graph.add_edge(0, 1)
                    graph.add_edge(1, 2)
                    graph.add_edge(2, 0)
                    return graph
            elif num_vertices <= 10:
                # Small dataset: provide more helpful error message
                raise GraphCreationError(
                    f"No valid triangles found for {num_vertices} points. "
                    f"This can happen with collinear points or points outside the valid range. "
                    f"Try using more well-distributed points (recommended: ≥10 points)."
                )
            else:
                # Larger dataset with no triangles: likely a serious issue
                raise GraphCreationError(
                    f"No triangles found in Delaunay triangulation for {num_vertices} points. "
                    f"Check that points are within valid coordinate ranges and not all collinear."
                )

        edges_set = set()

        # Iterate over the triangle list
        for tri in trilist:
            if len(tri) != 6:
                logging.warning(f"Invalid triangle format: expected 6 values, got {len(tri)}")
                continue

            try:
                vertex1 = subdiv.locate((tri[0], tri[1]))[2] - 4
                vertex2 = subdiv.locate((tri[2], tri[3]))[2] - 4
                vertex3 = subdiv.locate((tri[4], tri[5]))[2] - 4

                # Validate vertex indices
                max_vertex = graph.vcount()
                if not (0 <= vertex1 < max_vertex and 0 <= vertex2 < max_vertex and 0 <= vertex3 < max_vertex):
                    logging.warning(
                        f"Invalid vertices: {vertex1}, {vertex2}, {vertex3} for graph with {max_vertex} vertices")
                    continue

                edges_set.add((vertex1, vertex2))
                edges_set.add((vertex2, vertex3))
                edges_set.add((vertex3, vertex1))

            except Exception as e:
                logging.warning(f"Failed to process triangle {tri}: {e}")
                continue

        # Convert to list and remove duplicates
        edges_set = list({*map(tuple, map(sorted, edges_set))})

        if edges_set:
            graph.add_edges(edges_set)

        # Remove redundant edges
        graph.simplify()

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create Delaunay graph: {str(e)}")


def get_distance(position_array: np.ndarray, proximity_thresh: float,
                 metric: str = "euclidean") -> List[List[int]]:
    """Filter points by proximity, return the points within specified distance of each point

    Args:
        position_array: Array of position of shape (n, 2)
        proximity_thresh: Only keep points within this distance
        metric: Type of distance calculated

    Returns:
        List of lists containing indices of nearby points

    Raises:
        GraphCreationError: If distance calculation fails
    """
    try:
        if position_array is None or position_array.size == 0:
            raise GraphCreationError("Position array cannot be None or empty")
        if position_array.ndim != 2 or position_array.shape[1] != 2:
            raise GraphCreationError("Position array must be 2D with shape (n, 2)")
        if proximity_thresh <= 0:
            raise GraphCreationError("Proximity threshold must be positive")

        # Normalize the metric name to scipy-compatible format
        normalized_metric = normalize_distance_metric(metric)
        square_dist = squareform(pdist(position_array, metric=normalized_metric))
        proxi_list = []

        for i, row in enumerate(square_dist):
            nearby_indices = np.where((row < proximity_thresh) & (row > 0))[0].tolist()
            proxi_list.append(nearby_indices)

        return proxi_list

    except Exception as e:
        raise GraphCreationError(f"Failed to calculate distances: {str(e)}")


def graph_distance(graph: Any, position2d: np.ndarray, proximity_thresh: float,
                   metric: str = "euclidean") -> Any:
    """Construct a distance graph

    Args:
        graph: igraph Graph object
        position2d: 2D position array
        proximity_thresh: Distance threshold
        metric: Distance metric

    Returns:
        Modified graph

    Raises:
        GraphCreationError: If distance graph creation fails
    """
    try:
        if graph is None:
            raise GraphCreationError("Graph cannot be None")

        # Get the list of points within distance of each other
        proxi_list = get_distance(position2d, proximity_thresh, metric)

        # Make the edges
        edges_set = set()
        for i, point_list in enumerate(proxi_list):
            if i >= graph.vcount():
                logging.warning(f"Point index {i} exceeds graph vertex count {graph.vcount()}")
                continue

            valid_points = [x for x in point_list if x < graph.vcount()]
            if len(valid_points) != len(point_list):
                logging.warning(f"Some points in proximity list exceed graph vertex count")

            tlist = [(i, x) for x in valid_points]
            edges_set.update(tlist)

        edges_set = list({*map(tuple, map(sorted, edges_set))})

        # Add the edges
        if edges_set:
            graph.add_edges(edges_set)

        # Simplify the graph
        graph.simplify()

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create distance graph: {str(e)}")


def create_graph_array(point_array: np.ndarray, data_shape: Optional[List[Tuple[str, Any]]] = None) -> Any:
    """Create a graph from a point array, dynamically adding attributes.

    Args:
        point_array: Array of points with columns corresponding to the data_shape.
        data_shape: List of tuples defining the data structure, e.g.,
                    [('id', int), ('x', float), ('velocity', float)].
                    If None, defaults to the legacy [id, x, y] behavior.

    Returns:
        igraph Graph object.

    Raises:
        GraphCreationError: If graph creation fails.
    """
    try:
        if point_array is None or point_array.size == 0:
            raise GraphCreationError("Point array cannot be None or empty")
        if point_array.ndim != 2:
            raise GraphCreationError(f"Point array must be 2D, got {point_array.ndim}D")

        timer = time.time()
        n_vertices = len(point_array)
        graph = ig.Graph(n=n_vertices)

        # If no data_shape is provided, fall back to the old hardcoded behavior for compatibility.
        if data_shape is None:
            logging.warning("No data_shape provided to create_graph_array, defaulting to [id, x, y].")
            if point_array.shape[1] < 3:
                raise GraphCreationError("Point array must have at least 3 columns [id, x, y] for default mode.")

            graph.vs["id"] = point_array[:, 0]
            graph.vs["x"] = point_array[:, 1]
            graph.vs["y"] = point_array[:, 2]
            # Use the ID as the name for a nice summary string.
            graph.vs["name"] = [normalize_id(val) for val in point_array[:, 0]]

            logging.debug(f"Graph creation (default) took {round((time.time() - timer) * 1000, 3)}ms")
            return graph

        # --- DYNAMIC ATTRIBUTE CREATION ---
        id_col_index = -1
        for i, (attr_name, attr_type) in enumerate(data_shape):
            if i < point_array.shape[1]:
                # igraph's 'name' attribute must be a string.
                if attr_name == "name":
                    graph.vs[attr_name] = [str(val) for val in point_array[:, i]]
                else:
                    graph.vs[attr_name] = point_array[:, i]

                if attr_name == "id":
                    id_col_index = i
            else:
                logging.debug(
                    f"[!] data_shape specifies attribute '{attr_name}' at column {i}, "
                    f"but data only has {point_array.shape[1]} columns. Skipping."
                )

        # Ensure 'name' attribute exists for compatibility, even if not in data_shape.
        # This makes graph.summary() look good.
        if "name" not in [ds[0] for ds in data_shape]:
            if id_col_index != -1:
                # Use the 'id' column to create the 'name' attribute.
                id_values = point_array[:, id_col_index]
                graph.vs["name"] = [normalize_id(val) for val in id_values]
            else:
                # Fallback if no 'id' is present.
                graph.vs["name"] = [str(i) for i in range(n_vertices)]

        logging.debug(f"Graph creation (dynamic) took {round((time.time() - timer) * 1000, 3)}ms")
        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create graph from array: {str(e)}") from e


def create_graph_dict(point_dict: Dict[str, Any]) -> Any:
    """Create a graph from a point dictionary.

    This function dynamically adds all keys from the dictionary as vertex attributes.

    Args:
        point_dict: Dictionary with keys 'id', 'x', 'y', and other optional attributes.
                    All values should be lists or arrays of the same length.

    Returns:
        igraph Graph object

    Raises:
        GraphCreationError: If graph creation fails
    """
    try:
        if not point_dict:
            raise GraphCreationError("Point dictionary cannot be empty")

        # Core keys are required for basic functionality
        required_keys = ['id', 'x', 'y']
        missing_keys = [key for key in required_keys if key not in point_dict]
        if missing_keys:
            raise GraphCreationError(f"Missing required keys: {missing_keys}")

        # Check that all arrays have the same length
        lengths = {key: len(val) for key, val in point_dict.items()}
        if len(set(lengths.values())) > 1:
            raise GraphCreationError(f"All arrays in dictionary must have the same length. Got: {lengths}")

        timer = time.time()
        n_vertices = len(point_dict["id"])
        graph = ig.Graph(n=n_vertices)

        # Dynamically add all provided attributes
        for attr_name, values in point_dict.items():
            if attr_name == "name":
                graph.vs[attr_name] = [str(val) for val in values]
            else:
                graph.vs[attr_name] = values

        # Ensure 'name' attribute exists for compatibility if not provided
        if "name" not in point_dict:
            id_values = point_dict["id"]
            graph.vs["name"] = [normalize_id(val) for val in id_values]

        logging.debug(f"Graph creation from dict took {round((time.time() - timer) * 1000, 3)}ms")
        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create graph from dictionary: {str(e)}") from e


def call_igraph_method(graph: Any, method_name: str, *args, **kwargs) -> Any:
    """Call any igraph method on the graph safely

    Args:
        graph: igraph Graph object
        method_name: Name of the method to call
        *args: Positional arguments for the method
        **kwargs: Keyword arguments for the method

    Returns:
        Result of the method call

    Raises:
        IgraphMethodError: If method call fails
    """
    try:
        if graph is None:
            raise IgraphMethodError("Graph cannot be None")
        if not method_name:
            raise IgraphMethodError("Method name cannot be empty")
        if not hasattr(graph, method_name):
            raise IgraphMethodError(f"Graph does not have method '{method_name}'")

        method = getattr(graph, method_name)
        if not callable(method):
            raise IgraphMethodError(f"'{method_name}' is not a callable method")

        result = method(*args, **kwargs)
        logging.debug(f"Successfully called {method_name} on graph")
        return result

    except Exception as e:
        raise IgraphMethodError(f"Failed to call method '{method_name}': {str(e)}")


def create_delaunay_graph(data_points: Union[np.ndarray, Dict[str, Any]],
                          aspect: str = "array", dimension: Tuple[int, int] = (1200, 1200),
                          data_shape: Optional[List[Tuple[str, Any]]] = None) -> Any:
    """Create a Delaunay triangulation graph from point data

    Args:
        data_points: Point data as array or dictionary
        aspect: Data format ("array" or "dict")
        dimension: Image dimensions (width, height)
        data_shape: shape of the data to pass (if extra column of information)

    Returns:
        igraph Graph object with Delaunay triangulation

    Raises:
        GraphCreationError: If Delaunay graph creation fails
    """
    try:
        timer0 = time.time()

        # Create and populate the graph with points
        if aspect == "array":
            if not isinstance(data_points, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")

            # Simple type check - reject string/object IDs
            if data_points.dtype.kind in ['U', 'S', 'O']:
                raise GraphCreationError("Object IDs must be numeric, not string type")

            graph = create_graph_array(data_points, data_shape=data_shape)

            # Make triangulation with appropriate columns (assuming standard format [id, x, y])
            pos_array = np.stack((
                data_points[:, 1],  # x position (column 1)
                data_points[:, 2]  # y position (column 2)
            ), axis=1)
            subdiv = make_subdiv(pos_array, dimension)
            tri_list = subdiv.getTriangleList()

        elif aspect == "dict":
            if isinstance(data_points, dict):
                graph = create_graph_dict(data_points)
                pos_array = np.stack((data_points["x"], data_points["y"]), axis=1)
            elif isinstance(data_points, np.ndarray):
                # Convert array to dict format first
                data_interface = DataInterface()  # Use default data shape
                data_points = data_interface.to_array(data_points)
                graph = create_graph_dict(data_points)
                pos_array = np.stack((data_points["x"], data_points["y"]), axis=1)
            else:
                raise GraphCreationError("Invalid data format for 'dict' aspect")

            subdiv = make_subdiv(pos_array, dimension)
            tri_list = subdiv.getTriangleList()
        else:
            raise GraphCreationError("Graph data interface could not be understood")

        logging.debug(f"Creation and Triangulation took {round((time.time() - timer0) * 1000, 3)}ms")

        timer1 = time.time()
        # Populate edges
        graph = graph_delaunay(graph, subdiv, tri_list)
        logging.debug(f"Conversion took {round((time.time() - timer1) * 1000, 3)}ms")

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create Delaunay graph: {str(e)}")


def create_proximity_graph(data_points: Union[np.ndarray, Dict[str, Any]],
                           proximity_thresh: float, aspect: str = "array",
                           metric: str = "euclidean",
                           data_shape: Optional[List[Tuple[str, Any]]] = None
                           ) -> Any:
    """Create a proximity graph from point data

    Args:
        data_points: Point data as array or dictionary
        proximity_thresh: Distance threshold for connections
        aspect: Data format ("array" or "dict")
        metric: Distance metric to use for the graph construction
        data_shape: shape of the data to pass (if extra column of information)

    Returns:
        igraph Graph object with proximity connections and distance attributes

    Raises:
        GraphCreationError: If proximity graph creation fails
    """
    try:
        timer_prox = timeit.default_timer()

        if aspect == "array":
            if not isinstance(data_points, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")

            graph = create_graph_array(data_points, data_shape=data_shape)
            pos_array = np.stack((
                data_points[:, 1],  # x position (column 1)
                data_points[:, 2]  # y position (column 2)
            ), axis=1)

        elif aspect == "dict":
            if isinstance(data_points, dict):
                graph = create_graph_dict(data_points)
                pos_array = np.stack((data_points["x"], data_points["y"]), axis=1)
            elif isinstance(data_points, np.ndarray):
                data_interface = DataInterface()
                data_points = data_interface.to_array(data_points)
                graph = create_graph_dict(data_points)
                pos_array = np.stack((data_points["x"], data_points["y"]), axis=1)
            else:
                raise GraphCreationError("Invalid data format for 'dict' aspect")
        else:
            raise GraphCreationError("Graph data interface could not be understood")

        # Create proximity connections with optimized vectorized approach
        graph = graph_distance_optimized(graph, pos_array, proximity_thresh, metric=metric)

        end_prox = timeit.default_timer()
        logging.debug(f"Distance calculation took {round((end_prox - timer_prox) * 1000, 3)}ms")

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create proximity graph: {str(e)}")


def graph_distance_optimized(graph: Any, position2d: np.ndarray, proximity_thresh: float,
                             metric: str = "euclidean",
                             ) -> Any:
    """Construct a distance graph using optimized vectorized operations

    Args:
        graph: igraph Graph object
        position2d: 2D position array
        proximity_thresh: Distance threshold
        metric: Distance metric

    Returns:
        Modified graph with distance attributes on edges

    Raises:
        GraphCreationError: If distance graph creation fails
    """
    try:
        if graph is None:
            raise GraphCreationError("Graph cannot be None")

        if position2d is None or position2d.size == 0:
            raise GraphCreationError("Position array cannot be None or empty")

        if position2d.ndim != 2 or position2d.shape[1] != 2:
            raise GraphCreationError("Position array must be 2D with shape (n, 2)")

        if proximity_thresh <= 0:
            raise GraphCreationError("Proximity threshold must be positive")

        # Normalize the metric name to scipy-compatible format
        normalized_metric = normalize_distance_metric(metric)

        # Calculate full distance matrix using scipy's optimized pdist + squareform
        square_dist = squareform(pdist(position2d, metric=normalized_metric))

        # Use upper triangle indices to avoid duplicate edges (k=1 excludes diagonal)
        i_idx, j_idx = np.triu_indices_from(square_dist, k=1)

        # Apply threshold filter: distance < threshold AND distance > 0
        distances = square_dist[i_idx, j_idx]
        mask = (distances < proximity_thresh) & (distances > 0)

        # Filter valid edges and their weights
        valid_i = i_idx[mask]
        valid_j = j_idx[mask]
        valid_distances = distances[mask]

        # Validate vertex indices against graph
        max_vertex = graph.vcount()
        vertex_mask = (valid_i < max_vertex) & (valid_j < max_vertex)

        if not np.all(vertex_mask):
            logging.warning(f"Some vertices exceed graph vertex count {max_vertex}")
            valid_i = valid_i[vertex_mask]
            valid_j = valid_j[vertex_mask]
            valid_distances = valid_distances[vertex_mask]

        # Create edge list and add all edges at once
        if len(valid_i) > 0:
            edges_to_add = list(zip(valid_i, valid_j))
            graph.add_edges(edges_to_add)

            # Add distance attributes
            graph.es['distance'] = valid_distances.tolist()
            graph.es['weight'] = valid_distances.tolist()  # Many algorithms expect 'weight'

        # Simplify graph to remove any potential duplicates
        graph.simplify()

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create optimized distance graph: {str(e)}")


def create_knn_graph(positions: np.ndarray, k: int = 3, aspect: str = "array",
                     data_shape: Optional[List[Tuple[str, Any]]] = None) -> Any:
    """Create graph connecting each point to its k nearest neighbors

    Args:
        positions: Point data array
        k: Number of nearest neighbors
        aspect: Data format
        data_shape: shape of the data to pass (if extra column of information)
    """
    try:
        # Validate k parameter
        if k <= 0:
            raise GraphCreationError("k must be positive")

        if k >= len(positions):
            raise GraphCreationError(f"k ({k}) must be less than number of points ({len(positions)})")

        if aspect == "array":
            graph = create_graph_array(positions, data_shape)
            pos_2d = positions[:, 1:3]
        else:
            raise NotImplementedError("Dict aspect not implemented for k-nearest")

        # Calculate distances
        distances = cdist(pos_2d, pos_2d)

        # Find k nearest neighbors for each point
        edges_to_add = []
        for i, row in enumerate(distances):
            nearest_indices = np.argsort(row)[:k + 1]
            nearest_indices = nearest_indices[nearest_indices != i][:k]

            for j in nearest_indices:
                edge = tuple(sorted([i, j]))
                edges_to_add.append(edge)

        # Remove duplicates and add edges
        unique_edges = list(set(edges_to_add))
        if unique_edges:
            graph.add_edges(unique_edges)

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create k-nearest graph: {str(e)}")


def create_mst_graph(positions: np.ndarray, aspect: str = "array",
                     metric: str = "euclidean",
                     data_shape: Optional[List[Tuple[str, Any]]] = None
                     ) -> Any:
    """Create minimum spanning tree graph from a standardized array.

    Args:
        positions: Point data array
        aspect: Data format
        metric: Distance metric for MST construction
        data_shape: shape of the data to pass (if extra column of information)
    """
    try:
        if aspect == "array":
            graph = create_graph_array(positions, data_shape=data_shape)
            pos_2d = positions[:, 1:3]
        else:
            raise NotImplementedError("Dict aspect not implemented for MST")

        assert isinstance(metric, str), f"Expected 'metric' to be a string, got {type(metric).__name__}"
        normalized_metric = normalize_distance_metric(metric)
        distances = squareform(pdist(pos_2d, metric=normalized_metric))

        # Create complete graph for MST algorithm
        complete_graph = ig.Graph(n=len(positions), directed=False)
        edges_to_add = []
        weights = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                edges_to_add.append((i, j))
                weights.append(distances[i, j])

        complete_graph.add_edges(edges_to_add)
        complete_graph.es['weight'] = weights

        # Get MST
        mst_graph = complete_graph.spanning_tree(weights="weight")

        # Transfer edges to original graph
        graph.add_edges(mst_graph.get_edgelist())
        graph.es['weight'] = mst_graph.es['weight']


        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create MST: {str(e)}")

def create_gabriel_graph(positions: np.ndarray, aspect: str = "array",
                         data_shape: Optional[List[Tuple[str, Any]]] = None) -> Any:
    """
    Create a Gabriel graph from point positions using an optimized, vectorized approach.

    A Gabriel graph is a subgraph of the Delaunay triangulation where for any
    edge (p, q), the disk with diameter pq contains no other point r.

    Args:
        positions: Point data array with shape (n, >=3) containing [id, x, y, ...].
        aspect: Data format (currently only "array" is supported).
        data_shape: shape of the data to pass (if extra column of information)

    Returns:
        igraph.Graph: An igraph Graph object representing the Gabriel graph.

    Raises:
        GraphCreationError: If graph creation fails.
    """
    try:
        if aspect != "array":
            raise NotImplementedError("Dict aspect is not yet implemented for Gabriel graph")

        # 1. Initial setup
        graph = create_graph_array(positions, data_shape=data_shape)
        # Use float64 for better precision in geometric calculations
        pos_2d = positions[:, 1:3].astype(np.float64)
        n_points = len(pos_2d)

        if n_points < 2:
            return graph

        # 2. Start with the Delaunay triangulation. This is a crucial optimization,
        # as a Gabriel graph is always a subgraph of the Delaunay graph. This
        # reduces the number of candidate edges from O(N^2) to O(N).
        max_x, max_y = np.max(pos_2d, axis=0)
        temp_graph = create_delaunay_graph(
            positions,
            aspect="array",
            dimension=(int(max_x) + 1, int(max_y) + 1),
            data_shape=data_shape
        )

        if temp_graph.ecount() == 0:
            return graph  # No edges to check

        # 3. Get Delaunay edges and corresponding point coordinates
        delaunay_edges = np.array(temp_graph.get_edgelist())
        source_indices, target_indices = delaunay_edges[:, 0], delaunay_edges[:, 1]

        p1s = pos_2d[source_indices]  # Coordinates of all source points
        p2s = pos_2d[target_indices]  # Coordinates of all target points

        # 4. Vectorized calculation of disk centers and radii for ALL edges at once
        # Center of the disk for each edge (p1, p2) is (p1+p2)/2
        centers = (p1s + p2s) / 2.0
        # Squared radius of the disk is ||(p1-p2)/2||^2
        radii_sq = np.sum(((p1s - p2s) / 2.0)**2, axis=1)

        # 5. Vectorized check for the Gabriel condition.
        # We check if any other point `pk` falls inside the disk of an edge `(pi, pj)`.
        # Condition: ||pk - center_ij||^2 < radius_ij^2

        # Expand dimensions for broadcasting to compute the difference between every
        # disk center and every point in the dataset.
        # `dist_sq_matrix[e, k]` will be the squared distance from the center of edge `e` to point `k`.
        dist_sq_matrix = cdist(centers, pos_2d, 'sqeuclidean')

        # Compare every distance to the corresponding edge's radius.
        # `is_inside[e, k]` is True if point `k` is inside the disk of edge `e`.
        is_inside = dist_sq_matrix < radii_sq[:, np.newaxis] - 1e-10  # Tolerance for float precision

        # 6. Mask out the endpoints for each edge.
        # A point cannot invalidate an edge it belongs to. We create a boolean
        # mask to efficiently set `is_inside` to False for these cases.
        n_edges = len(delaunay_edges)
        row_indices = np.arange(n_edges)
        is_inside[row_indices, source_indices] = False
        is_inside[row_indices, target_indices] = False

        # 7. Identify Gabriel edges.
        # An edge is a Gabriel edge if NO other point is inside its disk.
        # We check if `any` value is True along the points axis (axis=1).
        has_intruder = np.any(is_inside, axis=1)
        gabriel_edges = delaunay_edges[~has_intruder]

        # 8. Add the final Gabriel edges to the graph
        if gabriel_edges.size > 0:
            graph.add_edges(gabriel_edges.tolist())

        return graph

    except Exception as e:
        # Wrap the exception for consistent error handling
        raise GraphCreationError(f"Failed to create Gabriel graph: {str(e)}") from e

def create_voronoi_cell_graph(positions: np.ndarray, dimension: Tuple[int, int],
                              aspect: str = "array",
                              ) -> Any:
    """
    Create graph from Voronoi diagram structure:
    - Nodes are Voronoi vertices (intersections of cell boundaries)
    - Edges connect adjacent Voronoi vertices
    """
    from scipy.spatial import Voronoi

    try:
        if aspect == "array":
            pos_2d = positions[:, 1:3]
        else:
            raise NotImplementedError("Dict aspect not implemented for Voronoi cell graph")

        # Compute Voronoi diagram
        vor = Voronoi(pos_2d)

        # Create graph with Voronoi vertices as nodes
        n_vertices = len(vor.vertices)
        graph = ig.Graph(n=n_vertices)

        # Set vertex attributes (Voronoi vertex coordinates)
        graph.vs["id"] = list(range(n_vertices))
        graph.vs["x"] = vor.vertices[:, 0]
        graph.vs["y"] = vor.vertices[:, 1]
        graph.vs["name"] = list(range(n_vertices))

        # Add edges between adjacent Voronoi vertices
        edges_to_add = []
        for ridge_vertices in vor.ridge_vertices:
            if -1 not in ridge_vertices:  # Skip infinite ridges
                edges_to_add.append(tuple(ridge_vertices))

        if edges_to_add:
            graph.add_edges(edges_to_add)

        # Create position array for distance calculation
        voronoi_positions = np.column_stack([
            graph.vs["id"], graph.vs["x"], graph.vs["y"]
        ])


        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create Voronoi cell graph: {str(e)}")

def create_visibility_graph(positions: np.ndarray, obstacles: Optional[List] = None,
                            aspect: str = "array",
                            ) -> Any:
    """
    Create visibility graph where points are connected if they have line-of-sight.

    Args:
        positions: Point data array
        obstacles: List of obstacle polygons (optional)
        aspect: Data format
        add_distance: Whether to add distance attributes
    """
    try:
        if aspect == "array":
            graph = create_graph_array(positions)
            pos_2d = positions[:, 1:3]
        else:
            raise NotImplementedError("Dict aspect not implemented for visibility graph")

        n_points = len(pos_2d)
        edges_to_add = []

        # Check visibility between all pairs
        for i in range(n_points):
            for j in range(i + 1, n_points):
                if _has_line_of_sight(pos_2d[i], pos_2d[j], obstacles):
                    edges_to_add.append((i, j))

        if edges_to_add:
            graph.add_edges(edges_to_add)

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create visibility graph: {str(e)}")


def _has_line_of_sight(p1: np.ndarray, p2: np.ndarray, obstacles: Optional[List] = None) -> bool:
    """Check if two points have unobstructed line of sight"""
    if obstacles is None:
        return True

    # Check if line segment p1-p2 intersects any obstacle
    for obstacle in obstacles:
        if _line_intersects_polygon(p1, p2, obstacle):
            return False
    return True


def _line_intersects_polygon(p1: np.ndarray, p2: np.ndarray, polygon: np.ndarray) -> bool:
    """Check if line segment intersects with polygon using ray casting"""
    # Implementation of line-polygon intersection
    # This is a standard computational geometry algorithm
    pass

def create_graph(data_points: Union[np.ndarray, Dict[str, Any]],
                 graph_type: str, aspect: str = "array",
                 dimension: Tuple[int, int] = (1200, 1200), **kwargs) -> Any:
    """Create any type of graph from point data

    Args:
        data_points: Point data as array or dictionary
        graph_type: Type of graph ("delaunay", "proximity", "knn", "mst", "gabriel")
        aspect: Data format ("array" or "dict")
        dimension: Image dimensions (width, height)
        **kwargs: Graph-specific parameters

    Returns:
        igraph Graph object

    Raises:
        GraphCreationError: If graph creation fails
        ValueError: If unknown graph type
    """
    try:
        graph_type = graph_type.lower()

        if graph_type == "delaunay":
            return create_delaunay_graph(data_points, aspect, dimension)

        elif graph_type == "proximity":
            proximity_thresh = kwargs.get('proximity_thresh', 100.0)
            metric = kwargs.get('metric', 'euclidean')
            return create_proximity_graph(data_points, proximity_thresh, aspect, metric)

        elif graph_type == "knn" or graph_type == "k_nearest":
            k = kwargs.get('k', 4)
            return create_knn_graph(data_points, k, aspect)

        elif graph_type == "mst" or graph_type == "minimum_spanning_tree":
            metric = kwargs.get('metric', 'euclidean')
            return create_mst_graph(data_points, aspect, metric)

        elif graph_type == "gabriel":
            return create_gabriel_graph(data_points, aspect)

        else:
            raise ValueError(f"Unknown graph type: {graph_type}. "
                             f"Supported types: delaunay, proximity, knn, mst, gabriel")

    except Exception as e:
        raise GraphCreationError(f"Failed to create {graph_type} graph: {str(e)}")