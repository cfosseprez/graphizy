"""
Graph algorithms for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

import logging
import time
import random
import timeit
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np

from .exceptions import (
    InvalidPointArrayError, SubdivisionError, TriangulationError,
    GraphCreationError, PositionGenerationError, DependencyError,
    IgraphMethodError
)

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
            from .exceptions import handle_subdivision_bounds_error
            handle_subdivision_bounds_error(point_array, dimensions, 'x')

        # Validate Y coordinates
        if np.any(point_array[:, 1] < 0):
            bad_points = point_array[point_array[:, 1] < 0]
            raise SubdivisionError(f"Found {len(bad_points)} points with Y < 0", point_array, dimensions)

        if np.any(point_array[:, 1] >= height):
            from .exceptions import handle_subdivision_bounds_error
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
            raise GraphCreationError("Triangle list cannot be empty")

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

        square_dist = squareform(pdist(position_array, metric=metric))
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
        from .exceptions import InvalidDataShapeError

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