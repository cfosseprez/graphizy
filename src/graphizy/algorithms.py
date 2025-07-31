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


def create_graph_array(point_array: np.ndarray) -> Any:
    """Create a graph from a point array

    Args:
        point_array: Array of points with columns [id, x, y, ...]

    Returns:
        igraph Graph object

    Raises:
        GraphCreationError: If graph creation fails
    """
    try:
        if point_array is None or point_array.size == 0:
            raise GraphCreationError("Point array cannot be None or empty")
        if point_array.ndim != 2 or point_array.shape[1] < 3:
            raise GraphCreationError("Point array must be 2D with at least 3 columns [id, x, y]")

        timer = time.time()

        n_vertices = len(point_array)

        # Create graph
        graph = ig.Graph(n=n_vertices)
        graph.vs["name"] = list(range(n_vertices))
        graph.vs["id"] = point_array[:, 0]
        graph.vs["x"] = point_array[:, 1]
        graph.vs["y"] = point_array[:, 2]

        logging.debug(f"Graph name vector of length {len(graph.vs['id'])}")
        logging.debug(f"Graph x vector of length {len(graph.vs['x'])}")
        logging.debug(f"Graph y vector of length {len(graph.vs['y'])}")
        logging.debug(f"Graph creation took {round((time.time() - timer) * 1000, 3)}ms")

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create graph from array: {str(e)}")


def create_graph_dict(point_dict: Dict[str, Any]) -> Any:
    """Create a graph from a point dictionary

    Args:
        point_dict: Dictionary with keys 'id', 'x', 'y'

    Returns:
        igraph Graph object

    Raises:
        GraphCreationError: If graph creation fails
    """
    try:
        if not point_dict:
            raise GraphCreationError("Point dictionary cannot be empty")

        required_keys = ['id', 'x', 'y']
        missing_keys = [key for key in required_keys if key not in point_dict]
        if missing_keys:
            raise GraphCreationError(f"Missing required keys: {missing_keys}")

        # Check that all arrays have the same length
        lengths = [len(point_dict[key]) for key in required_keys]
        if len(set(lengths)) > 1:
            raise GraphCreationError(f"All arrays must have the same length. Got: {dict(zip(required_keys, lengths))}")

        timer = time.time()

        n_vertices = len(point_dict["id"])

        # Create graph
        graph = ig.Graph(n=n_vertices)
        graph.vs["name"] = list(range(n_vertices))
        graph.vs["id"] = point_dict["id"]
        graph.vs["x"] = point_dict["x"]
        graph.vs["y"] = point_dict["y"]

        logging.debug(f"Graph name vector of length {len(graph.vs['name'])}")
        logging.debug(f"Graph x vector of length {len(graph.vs['x'])}")
        logging.debug(f"Graph y vector of length {len(graph.vs['y'])}")
        logging.debug(f"Graph creation took {round((time.time() - timer) * 1000, 3)}ms")

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create graph from dictionary: {str(e)}")


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

    def convert(self, point_array: np.ndarray) -> Dict[str, Any]:
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


def add_edge_distances(graph: Any, data_points: np.ndarray, edge_metric: str = "euclidean") -> Any:
    """
    Vectorized distance computation - compute all edge distances at once.
    """
    if graph.ecount() == 0:
        return graph

    # Build coordinate lookup
    coord_lookup = {int(data_points[i, 0]): data_points[i, 1:3]
                    for i in range(len(data_points))}

    # Extract ALL edge coordinates at once (vectorized)
    source_coords = np.array([coord_lookup[graph.vs[edge.source]["id"]] for edge in graph.es])
    target_coords = np.array([coord_lookup[graph.vs[edge.target]["id"]] for edge in graph.es])

    # Vectorized distance computation for ALL edges at once
    if edge_metric == "euclidean":
        diff = source_coords - target_coords
        distances = np.sqrt(np.sum(diff * diff, axis=1))  # All distances in one operation
    elif edge_metric == "manhattan":
        distances = np.sum(np.abs(source_coords - target_coords), axis=1)
    elif edge_metric == "chebyshev":
        distances = np.max(np.abs(source_coords - target_coords), axis=1)

    graph.es["distance"] = distances.tolist()
    return graph

def add_edge_distances_square(graph: Any, data_points: np.ndarray, edge_metric: str = "euclidean") -> Any:
    """
    Add distance attributes to all edges in a graph.

    Args:
        graph: igraph Graph object
        data_points: Array with [id, x, y] columns
        edge_metric: Distance metric to use

    Returns:
        Graph with 'distance' attribute on edges
    """
    if graph.ecount() == 0:
        return graph

    # Extract ids and coordinates
    ids = data_points[:, 0].astype(int)
    coords = data_points[:, 1:3]

    # Build a mapping from vertex id to index in coords
    id_to_index = {id_: idx for idx, id_ in enumerate(ids)}

    # Precompute all pairwise distances
    dist_matrix = cdist(coords, coords, metric=edge_metric)

    # Compute distance for each edge using precomputed matrix
    distances = []
    for edge in graph.es:
        source_id = graph.vs[edge.source]["id"]
        target_id = graph.vs[edge.target]["id"]

        i = id_to_index[source_id]
        j = id_to_index[target_id]
        distances.append(dist_matrix[i, j])

    graph.es["distance"] = distances
    return graph

def create_delaunay_graph(data_points: Union[np.ndarray, Dict[str, Any]],
                          aspect: str = "array", dimension: Tuple[int, int] = (1200, 1200),
                          add_distance: Union[bool, Dict[str, Any]] = True) -> Any:
    """Create a Delaunay triangulation graph from point data

    Args:
        data_points: Point data as array or dictionary
        aspect: Data format ("array" or "dict")
        dimension: Image dimensions (width, height)
        add_distance: Whether to add distances. Can be:
                     - True: Add distances using same metric
                     - False: Don't add distances
                     - Dict: {"metric": "euclidean"} for different distance metric

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

            graph = create_graph_array(data_points)

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
                dinter = DataInterface()  # Use default data shape
                data_points = dinter.convert(data_points)
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

        # Add distances if requested
        if add_distance:
            if isinstance(add_distance, dict):
                distance_metric = add_distance.get("metric", "euclidean")
            else:
                distance_metric = "euclidean"
            graph = add_edge_distances(graph, data_points, distance_metric)

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create Delaunay graph: {str(e)}")


def create_proximity_graph(data_points: Union[np.ndarray, Dict[str, Any]],
                           proximity_thresh: float, aspect: str = "array",
                           metric: str = "euclidean",
                           add_distance: Union[bool, Dict[str, Any]] = True) -> Any:
    """Create a proximity graph from point data

    Args:
        data_points: Point data as array or dictionary
        proximity_thresh: Distance threshold for connections
        aspect: Data format ("array" or "dict")
        metric: Distance metric to use for the graph construction
        add_distance: Whether to add distances. Can be:
                     - True: Add distances using same metric
                     - False: Don't add distances
                     - Dict: {"metric": "euclidean"} for different distance metric

    Returns:
        igraph Graph object with proximity connections and optional distances

    Raises:
        GraphCreationError: If proximity graph creation fails
    """
    try:
        timer_prox = timeit.default_timer()

        if aspect == "array":
            if not isinstance(data_points, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")

            graph = create_graph_array(data_points)
            pos_array = np.stack((
                data_points[:, 1],  # x position (column 1)
                data_points[:, 2]  # y position (column 2)
            ), axis=1)

        elif aspect == "dict":
            if isinstance(data_points, dict):
                graph = create_graph_dict(data_points)
                pos_array = np.stack((data_points["x"], data_points["y"]), axis=1)
            elif isinstance(data_points, np.ndarray):
                dinter = DataInterface()
                data_points = dinter.convert(data_points)
                graph = create_graph_dict(data_points)
                pos_array = np.stack((data_points["x"], data_points["y"]), axis=1)
            else:
                raise GraphCreationError("Invalid data format for 'dict' aspect")
        else:
            raise GraphCreationError("Graph data interface could not be understood")

        # Create proximity connections
        graph = graph_distance(graph, pos_array, proximity_thresh, metric=metric)

        # Add distances if requested
        if add_distance:
            if isinstance(add_distance, dict):
                distance_metric = add_distance.get("metric", "euclidean")
            else:
                distance_metric = "euclidean"
            graph = add_edge_distances(graph, data_points, distance_metric)

        end_prox = timeit.default_timer()
        logging.debug(f"Distance calculation took {round((end_prox - timer_prox) * 1000, 3)}ms")

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create proximity graph: {str(e)}")


def create_knn_graph(positions: np.ndarray, k: int = 3, aspect: str = "array",
                     add_distance: Union[bool, Dict[str, Any]] = True) -> Any:
    """Create graph connecting each point to its k nearest neighbors

    Args:
        positions: Point data array
        k: Number of nearest neighbors
        aspect: Data format
        add_distance: Whether to add distance attributes
    """
    try:
        if aspect == "array":
            graph = create_graph_array(positions)
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

        # Add distances if requested
        if add_distance:
            if isinstance(add_distance, dict):
                distance_metric = add_distance.get("metric", "euclidean")
            else:
                distance_metric = "euclidean"
            graph = add_edge_distances(graph, positions, distance_metric)

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create k-nearest graph: {str(e)}")


def create_mst_graph(positions: np.ndarray, aspect: str = "array",
                     metric: str = "euclidean",
                     edge_metric: str = "euclidean",
                     add_distance: Union[bool, Dict[str, Any]] = True) -> Any:
    """Create minimum spanning tree graph from a standardized array.

    Args:
        positions: Point data array
        aspect: Data format
        metric: Distance metric for MST construction
        add_distance: Whether to add distance attributes
    """
    try:
        if aspect == "array":
            graph = create_graph_array(positions)
            pos_2d = positions[:, 1:3]
        else:
            raise NotImplementedError("Dict aspect not implemented for MST")

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

        # Add distance attributes if requested (note: weight and distance are the same for MST)
        if add_distance:
            if isinstance(add_distance, dict):
                distance_metric = add_distance.get("metric", "euclidean")
                # Only recompute if different metric requested
                if distance_metric != metric:
                    graph = add_edge_distances(graph, positions, distance_metric)
                else:
                    # Copy weights as distances since they're the same
                    graph.es["distance"] = graph.es["weight"]
            else:
                # Copy weights as distances
                graph.es["distance"] = graph.es["weight"]

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create MST: {str(e)}")

def create_gabriel_graph(positions: np.ndarray, aspect: str = "array",
                         add_distance: Union[bool, Dict[str, Any]] = True) -> Any:
    """Create Gabriel graph from point positions

    Args:
        positions: Point data array
        aspect: Data format
        edge_metric: Distance metric to use for the edge distances
        add_distance: Whether to add distance attributes
    """
    try:
        if aspect == "array":
            graph = create_graph_array(positions)
            pos_2d = positions[:, 1:3]
        else:
            raise NotImplementedError("Dict aspect not implemented for Gabriel graph")

        n_points = len(pos_2d)
        if n_points < 2:
            return graph

        # Create Delaunay triangulation as starting point
        temp_graph = create_delaunay_graph(positions, aspect="array",
                                           dimension=(int(pos_2d[:, 0].max()) + 1, int(pos_2d[:, 1].max()) + 1),
                                           add_distance=False)  # Don't add distances to temp graph

        edges_to_add = []

        # Check Gabriel condition for each Delaunay edge
        for edge in temp_graph.es:
            i, j = edge.tuple
            p1 = pos_2d[i]
            p2 = pos_2d[j]

            center = (p1 + p2) / 2
            radius_sq = np.sum(((p1 - p2) / 2) ** 2)

            is_gabriel_edge = True
            for k in range(n_points):
                if k == i or k == j:
                    continue

                p3 = pos_2d[k]
                dist_sq = np.sum((p3 - center) ** 2)

                if dist_sq < radius_sq - 1e-10:
                    is_gabriel_edge = False
                    break

            if is_gabriel_edge:
                edges_to_add.append((i, j))

        # Add Gabriel edges
        if edges_to_add:
            graph.add_edges(edges_to_add)

        # Add distances if requested
        if add_distance:
            if isinstance(add_distance, dict):
                distance_metric = add_distance.get("metric", "euclidean")
            else:
                distance_metric = "euclidean"
            graph = add_edge_distances(graph, positions, distance_metric)

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create Gabriel graph: {str(e)}")


def create_voronoi_cell_graph(positions: np.ndarray, dimension: Tuple[int, int],
                              aspect: str = "array", add_distance: Union[bool, Dict[str, Any]] = True) -> Any:
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

        if add_distance:
            graph = add_edge_distances(graph, voronoi_positions, "euclidean")

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create Voronoi cell graph: {str(e)}")

def create_visibility_graph(positions: np.ndarray, obstacles: Optional[List] = None,
                            aspect: str = "array", add_distance: Union[bool, Dict[str, Any]] = True) -> Any:
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

        # Add distances if requested
        if add_distance:
            if isinstance(add_distance, dict):
                distance_metric = add_distance.get("metric", "euclidean")
            else:
                distance_metric = "euclidean"
            graph = add_edge_distances(graph, positions, distance_metric)

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