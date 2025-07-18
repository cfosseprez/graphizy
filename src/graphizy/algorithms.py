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
import numpy as np
from collections import defaultdict, deque

from graphizy.exceptions import (
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


class MemoryManager:
    """Manages memory connections between objects with optional edge aging support"""

    def __init__(self, max_memory_size: int = 100, max_iterations: int = None, track_edge_ages: bool = True):
        """Initialize memory manager

        Args:
            max_memory_size: Maximum number of connections to keep per object
            max_iterations: Maximum number of iterations to keep connections (None = unlimited)
            track_edge_ages: Whether to track edge ages for visualization
        """
        self.max_memory_size = max_memory_size
        self.max_iterations = max_iterations
        self.track_edge_ages = track_edge_ages
        self.current_iteration = 0

        # Memory structure: {object_id: deque([(connected_id, iteration), ...])}
        self.memory = defaultdict(lambda: deque(maxlen=max_memory_size))

        # Track all unique object IDs that have been seen
        self.all_objects = set()
        
        # Track edge ages (only if enabled)
        if self.track_edge_ages:
            self.edge_ages = {}  # {(obj1, obj2): {"first_seen": iter, "last_seen": iter}}

    def add_connections(self, connections: Dict[str, List[str]]) -> None:
        """Add new connections to memory

        Args:
            connections: Dictionary like {"A": ["C", "D"], "B": [], ...}
        """
        self.current_iteration += 1

        # Track current edges for age updates (if enabled)
        if self.track_edge_ages:
            current_edges = set()

        for obj_id, connected_ids in connections.items():
            self.all_objects.add(obj_id)

            # Add each connection with current iteration timestamp
            for connected_id in connected_ids:
                self.all_objects.add(connected_id)

                # Track edge age (if enabled)
                if self.track_edge_ages:
                    edge_key = tuple(sorted([obj_id, connected_id]))
                    current_edges.add(edge_key)
                    
                    if edge_key not in self.edge_ages:
                        self.edge_ages[edge_key] = {
                            "first_seen": self.current_iteration,
                            "last_seen": self.current_iteration
                        }
                    else:
                        self.edge_ages[edge_key]["last_seen"] = self.current_iteration

                # Add bidirectional connections
                self.memory[obj_id].append((connected_id, self.current_iteration))
                self.memory[connected_id].append((obj_id, self.current_iteration))

        # Clean old iterations if max_iterations is set
        if self.max_iterations:
            self._clean_old_iterations()

    def _clean_old_iterations(self) -> None:
        """Remove connections older than max_iterations"""
        cutoff_iteration = self.current_iteration - self.max_iterations

        for obj_id in self.memory:
            # Filter connections to keep only recent ones
            self.memory[obj_id] = deque(
                [(connected_id, iteration) for connected_id, iteration in self.memory[obj_id]
                 if iteration > cutoff_iteration],
                maxlen=self.max_memory_size
            )
        
        # Clean old edge ages (if tracking enabled)
        if self.track_edge_ages and hasattr(self, 'edge_ages'):
            self.edge_ages = {
                edge_key: age_info 
                for edge_key, age_info in self.edge_ages.items()
                if age_info["last_seen"] > cutoff_iteration
            }

    def get_current_memory_graph(self) -> Dict[str, List[str]]:
        """Get current memory as a graph dictionary

        Returns:
            Dictionary with current memory connections
        """
        result = {}

        # Include all objects, even those with no connections
        for obj_id in self.all_objects:
            connections = []
            if obj_id in self.memory:
                # Get unique connections (remove duplicates and self-connections)
                unique_connections = set()
                for connected_id, _ in self.memory[obj_id]:
                    if connected_id != obj_id:
                        unique_connections.add(connected_id)
                connections = list(unique_connections)

            result[obj_id] = connections

        return result

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory state"""
        total_connections = sum(len(connections) for connections in self.memory.values())

        stats = {
            "total_objects": len(self.all_objects),
            "total_connections": total_connections // 2,  # Divide by 2 because connections are bidirectional
            "current_iteration": self.current_iteration,
            "objects_with_memory": len([obj for obj in self.all_objects if obj in self.memory and self.memory[obj]]),
            "max_memory_size": self.max_memory_size,
            "max_iterations": self.max_iterations,
            "edge_aging_enabled": self.track_edge_ages
        }
        
        # Add edge age statistics if tracking is enabled
        if self.track_edge_ages and hasattr(self, 'edge_ages') and self.edge_ages:
            current_iter = self.current_iteration
            ages = [current_iter - info["first_seen"] for info in self.edge_ages.values()]
            
            stats["edge_age_stats"] = {
                "min_age": min(ages),
                "max_age": max(ages), 
                "avg_age": sum(ages) / len(ages),
                "total_aged_edges": len(ages)
            }

        return stats
    
    def get_edge_ages(self) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Get age information for all edges (if tracking enabled)"""
        if not self.track_edge_ages or not hasattr(self, 'edge_ages'):
            return {}
        return self.edge_ages.copy()
    
    def get_edge_age_normalized(self, max_age: int = None) -> Dict[Tuple[str, str], float]:
        """Get normalized edge ages (0.0 = newest, 1.0 = oldest)
        
        Args:
            max_age: Maximum age to consider (uses current max if None)
            
        Returns:
            Dictionary mapping edge to normalized age (0.0-1.0)
        """
        if not self.track_edge_ages or not hasattr(self, 'edge_ages') or not self.edge_ages:
            return {}
        
        if max_age is None:
            ages = [self.current_iteration - info["first_seen"] for info in self.edge_ages.values()]
            max_age = max(ages) if ages else 1
        
        if max_age == 0:
            return {edge: 0.0 for edge in self.edge_ages.keys()}
        
        normalized_ages = {}
        for edge_key, age_info in self.edge_ages.items():
            age = self.current_iteration - age_info["first_seen"]
            normalized_age = min(age / max_age, 1.0)
            normalized_ages[edge_key] = normalized_age
        
        return normalized_ages



def create_memory_graph(current_positions: Union[np.ndarray, Dict[str, Any]],
                        memory_connections: Dict[str, List[str]],
                        aspect: str = "array") -> Any:
    """Create a graph with current positions and memory-based edges

    Args:
        current_positions: Current positions as array [id, x, y, ...] or dict
        memory_connections: Memory connections {"obj_id": ["connected_id1", "connected_id2"]}
        aspect: Data format ("array" or "dict")

    Returns:
        igraph Graph object with memory-based edges

    Raises:
        GraphCreationError: If graph creation fails
    """
    try:
        # Create basic graph with positions
        if aspect == "array":
            if not isinstance(current_positions, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")
            graph = create_graph_array(current_positions)

        elif aspect == "dict":
            if isinstance(current_positions, np.ndarray):
                # Convert array to dict format if needed
                data_interface = DataInterface([("id", int), ("x", int), ("y", int)])
                current_positions = data_interface.convert(current_positions)

            if not isinstance(current_positions, dict):
                raise GraphCreationError("Expected dictionary for 'dict' aspect")
            graph = create_graph_dict(current_positions)
        else:
            raise GraphCreationError("Aspect must be 'array' or 'dict'")

        # Create mapping from normalized object ID to vertex index
        id_to_vertex = {}
        for i, obj_id in enumerate(graph.vs["id"]):
            normalized_id = normalize_id(obj_id)
            id_to_vertex[normalized_id] = i

        # Add memory-based edges  
        edges_to_add = []
        for obj_id, connected_ids in memory_connections.items():
            # Normalize the source ID
            obj_id_norm = normalize_id(obj_id)
            if obj_id_norm not in id_to_vertex:
                logging.warning(f"Object {obj_id} (normalized: {obj_id_norm}) in memory but not in current positions")
                continue

            vertex_from = id_to_vertex[obj_id_norm]

            for connected_id in connected_ids:
                # Normalize the target ID
                connected_id_norm = normalize_id(connected_id)
                if connected_id_norm not in id_to_vertex:
                    logging.warning(f"Connected object {connected_id} (normalized: {connected_id_norm}) in memory but not in current positions")
                    continue

                vertex_to = id_to_vertex[connected_id_norm]

                # Avoid self-loops and ensure consistent edge ordering
                if vertex_from != vertex_to:
                    edge = tuple(sorted([vertex_from, vertex_to]))
                    edges_to_add.append(edge)

        # Remove duplicates and add edges
        unique_edges = list(set(edges_to_add))
        if unique_edges:
            graph.add_edges(unique_edges)

            # Add memory attribute to edges
            graph.es["memory_based"] = [True] * len(unique_edges)

        logging.debug(f"Created memory graph with {graph.vcount()} vertices and {graph.ecount()} memory-based edges")

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create memory graph: {str(e)}")


def update_memory_from_proximity(current_positions: Union[np.ndarray, Dict[str, Any]],
                                 proximity_thresh: float,
                                 memory_manager: MemoryManager,
                                 metric: str = "euclidean",
                                 aspect: str = "array") -> Dict[str, List[str]]:
    """Update memory manager with current proximity connections

    Args:
        current_positions: Current positions
        proximity_thresh: Distance threshold for proximity
        memory_manager: MemoryManager instance to update
        metric: Distance metric
        aspect: Data format

    Returns:
        Current proximity connections dictionary
    """
    try:


        # Extract position data and create ID mapping
        if aspect == "array":
            if not isinstance(current_positions, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")

            # Normalize IDs consistently 
            object_ids = [normalize_id(obj_id) for obj_id in current_positions[:, 0]]
            positions_2d = current_positions[:, 1:3].astype(float)

        elif aspect == "dict":
            if isinstance(current_positions, np.ndarray):
                data_interface = DataInterface([("id", int), ("x", int), ("y", int)])
                current_positions = data_interface.convert(current_positions)

            # Normalize IDs consistently
            object_ids = [normalize_id(obj_id) for obj_id in current_positions["id"]]
            positions_2d = np.column_stack([current_positions["x"], current_positions["y"]])

        else:
            raise GraphCreationError("Aspect must be 'array' or 'dict'")

        # Get proximity connections
        proximity_indices = get_distance(positions_2d, proximity_thresh, metric)

        # Convert indices to object IDs
        current_connections = {}
        for i, nearby_indices in enumerate(proximity_indices):
            obj_id = object_ids[i]
            connected_ids = [object_ids[j] for j in nearby_indices]
            current_connections[obj_id] = connected_ids

        # Ensure all objects are represented (even those with no connections)
        for obj_id in object_ids:
            if obj_id not in current_connections:
                current_connections[obj_id] = []

        # Update memory manager
        memory_manager.add_connections(current_connections)

        return current_connections

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from proximity: {str(e)}")


def update_memory_from_graph(graph: Any, memory_manager: MemoryManager) -> Dict[str, List[str]]:
    """Update memory manager from any igraph Graph object

    Args:
        graph: Any igraph Graph object (Delaunay, proximity, custom, etc.)
        memory_manager: MemoryManager instance to update

    Returns:
        Current connections dictionary
    """
    try:
        if graph is None:
            raise GraphCreationError("Graph cannot be None")
        if memory_manager is None:
            raise GraphCreationError("Memory manager cannot be None")

        # Extract connections from the graph
        current_connections = {}

        # Initialize all vertices with empty connections
        for vertex in graph.vs:
            obj_id = normalize_id(vertex["id"])  # FIXED: Added normalization
            current_connections[obj_id] = []

        # Add edges as bidirectional connections
        for edge in graph.es:
            vertex1_id = normalize_id(graph.vs[edge.tuple[0]]["id"])  # FIXED: Added normalization
            vertex2_id = normalize_id(graph.vs[edge.tuple[1]]["id"])  # FIXED: Added normalization

            current_connections[vertex1_id].append(vertex2_id)
            current_connections[vertex2_id].append(vertex1_id)

        # Update memory manager
        memory_manager.add_connections(current_connections)

        return current_connections

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from graph: {str(e)}")


def update_memory_from_delaunay(current_positions: Union[np.ndarray, Dict[str, Any]],
                                memory_manager: MemoryManager,
                                aspect: str = "array",
                                dimension: Tuple[int, int] = (1200, 1200)) -> Dict[str, List[str]]:
    """Update memory manager with Delaunay triangulation connections

    Args:
        current_positions: Current positions
        memory_manager: MemoryManager instance to update
        aspect: Data format
        dimension: Canvas dimensions for triangulation

    Returns:
        Current Delaunay connections dictionary
    """
    try:
        # Create temporary graph for Delaunay triangulation
        if aspect == "array":
            temp_graph = create_graph_array(current_positions)
            pos_array = np.stack((
                current_positions[:, 1],  # x positions
                current_positions[:, 2]  # y positions
            ), axis=1)
        elif aspect == "dict":
            if isinstance(current_positions, np.ndarray):
                data_interface = DataInterface([("id", int), ("x", int), ("y", int)])
                current_positions = data_interface.convert(current_positions)
            temp_graph = create_graph_dict(current_positions)
            pos_array = np.column_stack([current_positions["x"], current_positions["y"]])
        else:
            raise GraphCreationError("Aspect must be 'array' or 'dict'")

        # Create Delaunay triangulation
        subdiv = make_subdiv(pos_array, dimension)
        tri_list = subdiv.getTriangleList()
        delaunay_graph = graph_delaunay(temp_graph, subdiv, tri_list)

        # Update memory from the Delaunay graph
        return update_memory_from_graph(delaunay_graph, memory_manager)

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from Delaunay: {str(e)}")


def update_memory_from_custom_function(current_positions: Union[np.ndarray, Dict[str, Any]],
                                       memory_manager: MemoryManager,
                                       connection_function: callable,
                                       aspect: str = "array",
                                       **kwargs) -> Dict[str, List[str]]:
    """Update memory using a custom connection function

    Args:
        current_positions: Current positions
        memory_manager: MemoryManager instance to update
        connection_function: Function that takes positions and returns graph
        aspect: Data format
        **kwargs: Additional arguments for the connection function

    Returns:
        Current connections dictionary
    """
    try:
        # Call the custom function to create a graph
        custom_graph = connection_function(current_positions, aspect=aspect, **kwargs)

        # Update memory from the custom graph
        return update_memory_from_graph(custom_graph, memory_manager)

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from custom function: {str(e)}")

# Example usage function
def example_memory_graph_usage():
    """Example of how to use the memory graph functionality"""

    # Example data - current positions
    current_positions = np.array([
        [1, 100, 100],  # Object A at (100, 100)
        [2, 200, 150],  # Object B at (200, 150)
        [3, 120, 300],  # Object C at (120, 300)
        [4, 400, 100],  # Object D at (400, 100)
    ])

    # Example memory connections (historical proximities)
    memory_connections = {
        "1": ["3", "4"],  # A was connected to C and D
        "2": [],  # B has no memory connections
        "3": ["1"],  # C was connected to A
        "4": ["1"],  # D was connected to A
    }

    # Create memory graph
    graph = create_memory_graph(current_positions, memory_connections, aspect="array")

    print(f"Memory graph: {graph.vcount()} vertices, {graph.ecount()} edges")

    # Using with MemoryManager
    memory_mgr = MemoryManager(max_memory_size=50, max_iterations=10)

    # Simulate multiple iterations
    for iteration in range(5):
        # Simulate changing proximity connections each iteration
        proximity_connections = {
            "1": ["2"] if iteration % 2 == 0 else ["3"],
            "2": ["1"] if iteration % 2 == 0 else [],
            "3": ["1"] if iteration % 2 == 1 else ["4"],
            "4": ["3"] if iteration % 2 == 1 else [],
        }

        memory_mgr.add_connections(proximity_connections)

    # Get final memory state
    final_memory = memory_mgr.get_current_memory_graph()
    final_graph = create_memory_graph(current_positions, final_memory, aspect="array")

    stats = memory_mgr.get_memory_stats()
    print(f"Final memory stats: {stats}")

    return final_graph


# Example custom connection functions
def create_k_nearest_graph(positions: np.ndarray, k: int = 3, aspect: str = "array") -> Any:
    """Create graph connecting each point to its k nearest neighbors"""
    try:
        from scipy.spatial.distance import cdist

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
            # Get indices of k+1 nearest (including self), then exclude self
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


def create_minimum_spanning_tree(positions: np.ndarray, aspect: str = "array") -> Any:
    """Create minimum spanning tree graph"""
    try:
        if aspect == "array":
            graph = create_graph_array(positions)
            pos_2d = positions[:, 1:3]
        else:
            raise NotImplementedError("Dict aspect not implemented for MST")

        # Create complete graph first
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(pos_2d))

        # Add all edges with weights
        edges_to_add = []
        weights = []

        n = len(positions)
        for i in range(n):
            for j in range(i + 1, n):
                edges_to_add.append((i, j))
                weights.append(distances[i, j])

        graph.add_edges(edges_to_add)
        graph.es["weight"] = weights

        # Get minimum spanning tree
        mst = graph.spanning_tree(weights="weight")

        return mst

    except Exception as e:
        raise GraphCreationError(f"Failed to create MST: {str(e)}")

def create_gabriel_graph(positions: np.ndarray, aspect: str = "array") -> Any:
    """Create Gabriel graph from point positions
    
    Gabriel graph connects two points if no other point lies within the circle
    having the two points as diameter endpoints.
    
    Args:
        positions: Point positions as array [id, x, y, ...] or 2D positions
        aspect: Data format ("array" or "dict")
        
    Returns:
        igraph Graph object with Gabriel graph connections
        
    Raises:
        GraphCreationError: If Gabriel graph creation fails
    """
    try:
        if aspect == "array":
            graph = create_graph_array(positions)
            pos_2d = positions[:, 1:3]
        else:
            raise NotImplementedError("Dict aspect not implemented for Gabriel graph")
        
        n_points = len(pos_2d)
        edges_to_add = []
        
        # For each pair of points
        for i in range(n_points):
            for j in range(i + 1, n_points):
                p1 = pos_2d[i]
                p2 = pos_2d[j]
                
                # Calculate circle center (midpoint) and radius
                center = (p1 + p2) / 2
                radius = np.linalg.norm(p1 - p2) / 2
                
                # Check if any other point lies within the circle
                is_gabriel_edge = True
                for k in range(n_points):
                    if k == i or k == j:
                        continue
                        
                    p3 = pos_2d[k]
                    distance_to_center = np.linalg.norm(p3 - center)
                    
                    # If point is strictly inside the circle, this is not a Gabriel edge
                    if distance_to_center < radius - 1e-10:  # Small epsilon for numerical stability
                        is_gabriel_edge = False
                        break
                
                if is_gabriel_edge:
                    edges_to_add.append((i, j))
        
        # Add edges to graph
        if edges_to_add:
            graph.add_edges(edges_to_add)
        
        return graph
        
    except Exception as e:
        raise GraphCreationError(f"Failed to create Gabriel graph: {str(e)}")