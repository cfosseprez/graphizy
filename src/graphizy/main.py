"""
Main graphing class for graphizy

This module provides the primary interface for creating, manipulating, and visualizing
various types of graphs including Delaunay triangulations, proximity graphs, k-nearest
neighbor graphs, Gabriel graphs, minimum spanning trees, and memory-based graphs.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez

Examples:
    Basic usage::

        from graphizy import Graphing
        import numpy as np

        # Create sample data
        data = np.random.rand(100, 3)  # 100 points with [id, x, y]

        # Initialize graphing object
        grapher = Graphing(dimension=(800, 600), aspect="array")

        # Create different types of graphs
        delaunay_graph = grapher.make_delaunay(data)
        proximity_graph = grapher.make_proximity(data, proximity_thresh=50.0)

        # Visualize
        image = grapher.draw_graph(delaunay_graph)
        grapher.show_graph(image, title="Delaunay Triangulation")
"""

import logging
import time
import timeit
from typing import Union, Dict, Any, List, Tuple, Optional
import numpy as np

from graphizy.config import GraphizyConfig, DrawingConfig, GraphConfig
from graphizy.exceptions import (
    InvalidAspectError, InvalidDimensionError, GraphCreationError,
    IgraphMethodError, DrawingError
)
from graphizy.algorithms import (
    create_graph_array, create_graph_dict, DataInterface, call_igraph_method,
    create_delaunay_graph, create_proximity_graph,
    create_minimum_spanning_tree, create_knn_graph, create_gabriel_graph,
    create_memory_graph, MemoryManager, update_memory_from_custom_function,
    update_memory_from_graph, update_memory_from_delaunay, update_memory_from_proximity
)
from graphizy.drawing import Visualizer


class Graphing:
    """
    Main graphing class for creating and visualizing various types of graphs.

    This class provides a unified interface for creating different types of graphs
    from point data, including geometric graphs (Delaunay, Gabriel), proximity-based
    graphs (k-NN, proximity), and spanning trees. It also supports memory-based
    graphs for temporal analysis and comprehensive graph visualization.

    The class supports two data formats:
    - "array": NumPy arrays with columns [id, x, y]
    - "dict": Dictionaries with keys "id", "x", "y"

    Attributes:
        config (GraphizyConfig): Configuration object containing graph and drawing settings
        dimension (Tuple[int, int]): Canvas dimensions (width, height)
        aspect (str): Data format ("array" or "dict")
        dinter (DataInterface): Data interface for handling different data formats
        memory_manager (MemoryManager): Optional memory manager for temporal graphs

        # Drawing configuration shortcuts
        line_thickness (int): Thickness of graph edges
        line_color (Tuple[int, int, int]): RGB color for edges
        point_thickness (int): Thickness of point borders
        point_radius (int): Radius of graph vertices
        point_color (Tuple[int, int, int]): RGB color for vertices

    Examples:
        >>> # Basic initialization
        >>> grapher = Graphing(dimension=(800, 600), aspect="array")

        >>> # With custom configuration
        >>> config = GraphizyConfig()
        >>> config.drawing.line_color = (255, 0, 0)  # Red edges
        >>> grapher = Graphing(config=config)

        >>> # Create and visualize a graph
        >>> data = np.random.rand(50, 3)
        >>> graph = grapher.make_delaunay(data)
        >>> image = grapher.draw_graph(graph)
        >>> grapher.show_graph(image)
    """

    def __init__(self,
                 dimension: Union[Tuple[int, int], List[int]] = None,
                 data_shape: List[Tuple[str, type]] = None,
                 aspect: str = "array",
                 config: Optional[GraphizyConfig] = None,
                 **kwargs):
        """
        Initialize Graphing object with specified parameters.

        Args:
            dimension: Canvas dimensions as (width, height). Defaults to config default.
            data_shape: Data structure definition for custom data formats.
                       Defaults to [(id, int), (x, float), (y, float)].
            aspect: Data format specification. Either "array" for NumPy arrays
                   or "dict" for dictionary format.
            config: Pre-configured GraphizyConfig object. If None, creates default config.
            **kwargs: Additional configuration parameters that override config settings.
                     Can include nested parameters like drawing={'line_color': (255,0,0)}.

        Raises:
            InvalidDimensionError: If dimension is not a 2-element tuple/list with positive integers.
            InvalidAspectError: If aspect is not "array" or "dict".
            GraphCreationError: If initialization fails due to configuration issues.

        Examples:
            >>> # Basic initialization with default config
            >>> grapher = Graphing(dimension=(800, 600))

            >>> # Custom aspect and drawing parameters
            >>> grapher = Graphing(
            ...     dimension=(1024, 768),
            ...     aspect="dict",
            ...     line_color=(255, 0, 0),
            ...     point_radius=5
            ... )

            >>> # Using pre-configured config object
            >>> config = GraphizyConfig()
            >>> config.drawing.line_thickness = 3
            >>> grapher = Graphing(config=config, dimension=(640, 480))
        """
        try:
            # Initialize configuration with sensible defaults
            if config is None:
                config = GraphizyConfig()

            # Update configuration with provided parameters
            if dimension is not None:
                config.graph.dimension = tuple(dimension)
            if data_shape is not None:
                config.graph.data_shape = data_shape
            if aspect != "array":
                config.graph.aspect = aspect

            # Apply any additional configuration overrides
            if kwargs:
                config.update(**kwargs)

            self.config = config

            # Validate and set dimensions
            if not isinstance(self.config.graph.dimension, (tuple, list)) or len(self.config.graph.dimension) != 2:
                raise InvalidDimensionError("Dimension must be a tuple/list of 2 integers")
            if self.config.graph.dimension[0] <= 0 or self.config.graph.dimension[1] <= 0:
                raise InvalidDimensionError("Dimension values must be positive")

            self.dimension = self.config.graph.dimension

            # Validate aspect parameter
            valid_aspects = ["dict", "array"]
            if self.config.graph.aspect not in valid_aspects:
                raise InvalidAspectError(
                    f"Invalid aspect '{self.config.graph.aspect}'. Must be one of {valid_aspects}")
            self.aspect = self.config.graph.aspect

            # Initialize data interface for handling different data formats
            self.dinter = DataInterface(self.config.graph.data_shape)


            # Initialize memory manager as None (created on-demand)
            self.memory_manager = None

            # Initialize the visualizer
            self.visualizer = Visualizer(self.config.drawing, self.config.graph.dimension)


            logging.info(f"Graphing object initialized: {self.dimension} canvas, '{self.aspect}' aspect")

        except (InvalidDimensionError, InvalidAspectError):
            # Re-raise specific exceptions as-is
            raise
        except Exception as e:
            raise GraphCreationError(f"Failed to initialize Graphing object: {str(e)}")


    @property
    def drawing_config(self) -> DrawingConfig:
        """
        Get current drawing configuration.

        Returns:
            DrawingConfig: Current drawing configuration object containing
                          line and point styling parameters.
        """
        return self.config.drawing

    @property
    def graph_config(self) -> GraphConfig:
        """
        Get current graph configuration.

        Returns:
            GraphConfig: Current graph configuration object containing
                        dimension, aspect, and algorithm parameters.
        """
        return self.config.graph

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters at runtime.

        This method allows dynamic reconfiguration of the Graphing object
        without requiring re-initialization. Changes are applied immediately
        and cached values are updated.

        Args:
            **kwargs: Configuration parameters to update. Can include nested
                     parameters using dictionary syntax:
                     - drawing={'line_color': (255,0,0), 'point_radius': 8}
                     - graph={'proximity_threshold': 100.0}
                     - Direct parameters: line_thickness=3, aspect='dict'

        Raises:
            GraphCreationError: If configuration update fails due to invalid parameters.

        Examples:
            >>> # Update drawing parameters
            >>> grapher.update_config(
            ...     drawing={'line_color': (0, 255, 0), 'line_thickness': 2}
            ... )

            >>> # Update graph parameters
            >>> grapher.update_config(
            ...     graph={'proximity_threshold': 75.0, 'distance_metric': 'manhattan'}
            ... )

            >>> # Mixed updates
            >>> grapher.update_config(
            ...     line_color=(255, 255, 0),
            ...     graph={'dimension': (1200, 800)}
            ... )
        """
        try:
            self.config.update(**kwargs)

            # Update instance variables if graph config changed
            if 'graph' in kwargs or 'dimension' in kwargs:
                self.dimension = self.config.graph.dimension

            if 'graph' in kwargs or 'aspect' in kwargs:
                self.aspect = self.config.graph.aspect

            if 'graph' in kwargs and 'data_shape' in kwargs.get('graph', {}):
                self.dinter = DataInterface(self.config.graph.data_shape)

            logging.info("Configuration updated successfully")

        except Exception as e:
            raise GraphCreationError(f"Failed to update configuration: {str(e)}")

    @staticmethod
    def identify_graph(graph: Any) -> Any:
        """
        Replace graph vertex names with proper particle IDs for consistency.

        This method ensures that graph vertices have consistent naming by setting
        the "name" attribute to match the "id" attribute. This is useful for
        maintaining data consistency across different graph operations.

        Args:
            graph: igraph Graph object to modify.

        Returns:
            Any: The modified graph object with updated vertex names.

        Raises:
            GraphCreationError: If graph is None or modification fails.

        Note:
            This method modifies the graph in-place and also returns it for
            method chaining convenience.

        Examples:
            >>> graph = grapher.make_delaunay(data)
            >>> identified_graph = Graphing.identify_graph(graph)
            >>> # Now graph.vs["name"] == graph.vs["id"] for all vertices
        """
        try:
            if graph is None:
                raise GraphCreationError("Graph cannot be None")
            graph.vs["name"] = graph.vs["id"]
            return graph
        except Exception as e:
            raise GraphCreationError(f"Failed to identify graph: {str(e)}")

    def _get_data_as_array(self, data_points: Union[np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Internal helper to ensure data is in the standard NumPy array format.
        Handles conversion from 'dict' aspect if necessary.
        """
        if self.aspect == "array":
            if not isinstance(data_points, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")
            # Validate data types - reject string/object IDs for consistency
            if data_points.dtype.kind in ['U', 'S', 'O']:
                raise GraphCreationError("Array format requires numeric IDs, not string/object types")
            return data_points

        elif self.aspect == "dict":
            if isinstance(data_points, dict):
                # Ensure all required keys are present and have same length
                required_keys = ["id", "x", "y"]
                if not all(k in data_points for k in required_keys):
                    raise GraphCreationError(f"Dict data must contain required keys: {required_keys}")
                if len(set(len(v) for v in data_points.values())) > 1:
                    raise GraphCreationError("All lists in data dictionary must have the same length.")

                return np.column_stack((
                    data_points["id"],
                    data_points["x"],
                    data_points["y"]
                ))
            elif isinstance(data_points, np.ndarray):
                # Allow passing an array even if aspect is 'dict'
                return data_points
            else:
                raise GraphCreationError("Dict aspect requires a dictionary or NumPy array as input")

        else:
            raise GraphCreationError(f"Unknown aspect '{self.aspect}'. Use 'array' or 'dict'")


    def make_delaunay(self, data_points: Union[np.ndarray, Dict[str, Any]]) -> Any:
        """
        Create a Delaunay triangulation graph from point data.

        Delaunay triangulation connects points such that no point lies inside the
        circumcircle of any triangle in the triangulation. This creates a graph
        where nearby points are connected while avoiding overly long connections.

        The resulting graph has useful properties:
        - Maximizes the minimum angle of triangles
        - Forms the dual of the Voronoi diagram
        - Connects each point to its natural neighbors

        Args:
            data_points: Point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays

        Returns:
            Any: igraph Graph object representing the Delaunay triangulation.
                 Contains vertices with attributes: id, x, y, name
                 Contains edges connecting Delaunay neighbors.

        Raises:
            GraphCreationError: If input data is invalid, aspect mismatch occurs,
                               or triangulation computation fails.

        Examples:
            >>> # Array format
            >>> data = np.array([[1, 10.0, 20.0], [2, 30.0, 40.0], [3, 50.0, 60.0]])
            >>> delaunay_graph = grapher.make_delaunay(data)

            >>> # Dictionary format
            >>> data = {"id": [1, 2, 3], "x": [10.0, 30.0, 50.0], "y": [20.0, 40.0, 60.0]}
            >>> grapher.aspect = "dict"
            >>> delaunay_graph = grapher.make_delaunay(data)

        Note:
            - Requires at least 3 non-collinear points for meaningful triangulation
            - Points with string/object IDs are not supported in array format
            - Performance: O(n log n) for n points
        """
        try:
            timer0 = time.time()
            # Use the new helper method
            data_array = self._get_data_as_array(data_points)
            graph = create_delaunay_graph(data_array, aspect="array", dimension=self.dimension)
            elapsed_ms = round((time.time() - timer0) * 1000, 3)
            logging.debug(f"Delaunay triangulation completed in {elapsed_ms}ms")
            return graph
        except Exception as e:
            raise GraphCreationError(f"Failed to create Delaunay graph: {str(e)}")

    def make_proximity(self,
                      data_points: Union[np.ndarray, Dict[str, Any]],
                      proximity_thresh: float = None,
                      metric: str = None) -> Any:
        """
        Create a proximity graph connecting points within a distance threshold.

        Proximity graphs connect all pairs of points that are closer than a specified
        threshold distance. This creates dense local connections while maintaining
        sparsity for distant points.

        Args:
            data_points: Point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays
            proximity_thresh: Maximum distance for connecting points. If None, uses
                            config.graph.proximity_threshold. Smaller values create
                            sparser graphs, larger values create denser graphs.
            metric: Distance metric to use. If None, uses config.graph.distance_metric.
                   Common options: 'euclidean', 'manhattan', 'chebyshev'.

        Returns:
            Any: igraph Graph object representing the proximity graph.
                 Contains vertices with attributes: id, x, y, name
                 Contains edges between all point pairs within threshold distance.

        Raises:
            GraphCreationError: If input data is invalid, aspect mismatch occurs,
                               or proximity computation fails.

        Examples:
            >>> # Basic proximity graph with default threshold
            >>> data = np.random.rand(100, 3) * 100  # Scale for meaningful distances
            >>> prox_graph = grapher.make_proximity(data)

            >>> # Custom threshold and metric
            >>> prox_graph = grapher.make_proximity(
            ...     data,
            ...     proximity_thresh=25.0,
            ...     metric='manhattan'
            ... )

            >>> # Adaptive threshold based on data
            >>> # Use ~10% of data range as threshold
            >>> data_range = np.ptp(data[:, 1:3])  # Range of x,y coordinates
            >>> adaptive_thresh = data_range * 0.1
            >>> prox_graph = grapher.make_proximity(data, proximity_thresh=adaptive_thresh)

        Note:
            - Performance: O(n²) for distance computation, can be expensive for large datasets
            - Threshold selection significantly affects graph connectivity
            - Very small thresholds may result in disconnected graphs
            - Very large thresholds may result in nearly complete graphs
        """
        try:
            if proximity_thresh is None:
                proximity_thresh = self.config.graph.proximity_threshold
            if metric is None:
                metric = self.config.graph.distance_metric

            timer_prox = time.time()
            # Use the new helper method
            data_array = self._get_data_as_array(data_points)
            graph = create_proximity_graph(data_array, proximity_thresh, aspect="array", metric=metric)
            elapsed_ms = round((time.time() - timer_prox) * 1000, 3)
            logging.debug(f"Proximity graph (thresh={proximity_thresh}, metric={metric}) "
                          f"completed in {elapsed_ms}ms")
            return graph
        except Exception as e:
            raise GraphCreationError(f"Failed to create proximity graph: {str(e)}")

    def make_knn(self, data_points: Union[np.ndarray, Dict[str, Any]], k: int = 4) -> Any:
        """
        Create a k-nearest neighbors graph.

        k-NN graphs connect each point to its k closest neighbors, creating a graph
        where each vertex has exactly k outgoing edges (or fewer if there are fewer
        than k other points). This creates locally dense connections while maintaining
        a controlled edge count.

        Args:
            data_points: Point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays
            k: Number of nearest neighbors to connect to each point. Must be positive
               and less than the total number of points. Typical values: 3-8 for
               most applications.

        Returns:
            Any: igraph Graph object representing the k-NN graph.
                 Contains vertices with attributes: id, x, y, name
                 Contains directed edges from each point to its k nearest neighbors.

        Raises:
            GraphCreationError: If input data is invalid, k is invalid,
                               aspect mismatch occurs, or k-NN computation fails.

        Examples:
            >>> # Basic k-NN with default k=4
            >>> data = np.random.rand(50, 3) * 100
            >>> knn_graph = grapher.make_knn(data)

            >>> # Higher connectivity with k=8
            >>> knn_graph = grapher.make_knn(data, k=8)

            >>> # Low connectivity for sparse graph
            >>> knn_graph = grapher.make_knn(data, k=2)

            >>> # Dictionary format
            >>> data_dict = {
            ...     "id": list(range(100)),
            ...     "x": np.random.rand(100) * 100,
            ...     "y": np.random.rand(100) * 100
            ... }
            >>> grapher.aspect = "dict"
            >>> knn_graph = grapher.make_knn(data_dict, k=6)

        Note:
            - Performance: O(n² log k) with naive implementation, O(n log n) with efficient structures
            - Resulting graph may be directed (A→B doesn't imply B→A)
            - For undirected k-NN, consider post-processing to make edges mutual
            - k should be much smaller than total number of points for meaningful results
            - Larger k values increase connectivity but may include distant neighbors
        """
        try:
            timer_knn = timeit.default_timer()

            if k <= 0:
                raise GraphCreationError("k must be positive")

            # Centralize data conversion
            data_array = self._get_data_as_array(data_points)

            if k >= len(data_array):
                raise GraphCreationError(f"k ({k}) must be less than number of points ({len(data_array)})")

            # Call simplified algorithm function (aspect no longer needed)
            graph = create_knn_graph(data_array, k=k)

            elapsed_ms = round((timeit.default_timer() - timer_knn) * 1000, 3)
            logging.debug(f"k-NN graph (k={k}) completed in {elapsed_ms}ms")

            return graph

        except Exception as e:
            raise GraphCreationError(f"Failed to create k-NN graph: {str(e)}")

    def make_gabriel(self, data_points: Union[np.ndarray, Dict[str, Any]]) -> Any:
        """
        Create a Gabriel graph from point data.

        A Gabriel graph connects two points if and only if the circle having these
        two points as diameter endpoints contains no other points. This creates a
        subset of the Delaunay triangulation with interesting geometric properties
        and tends to connect points that have a "clear line of sight" to each other.

        Gabriel graphs are useful for:
        - Network topology design where interference matters
        - Computational geometry applications
        - Sparse geometric graph construction
        - Path planning with obstacle avoidance concepts

        Args:
            data_points: Point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays

        Returns:
            Any: igraph Graph object representing the Gabriel graph.
                 Contains vertices with attributes: id, x, y, name
                 Contains edges between Gabriel-connected points.

        Raises:
            GraphCreationError: If input data is invalid, aspect mismatch occurs,
                               or Gabriel graph computation fails.

        Examples:
            >>> # Basic Gabriel graph
            >>> data = np.array([[1, 10, 10], [2, 20, 10], [3, 15, 20], [4, 30, 30]])
            >>> gabriel_graph = grapher.make_gabriel(data)

            >>> # Gabriel graph from random points
            >>> np.random.seed(42)
            >>> data = np.column_stack([
            ...     range(50),  # IDs
            ...     np.random.rand(50) * 100,  # X coordinates
            ...     np.random.rand(50) * 100   # Y coordinates
            ... ])
            >>> gabriel_graph = grapher.make_gabriel(data)

            >>> # Dictionary format
            >>> data_dict = {
            ...     "id": [1, 2, 3, 4],
            ...     "x": [10, 20, 15, 30],
            ...     "y": [10, 10, 20, 30]
            ... }
            >>> grapher.aspect = "dict"
            >>> gabriel_graph = grapher.make_gabriel(data_dict)

        Note:
            - Gabriel graphs are always subgraphs of Delaunay triangulations
            - Generally sparser than Delaunay triangulations
            - Maintains good connectivity while avoiding "long" edges
            - Performance: O(n³) naive implementation, O(n² log n) with optimizations
            - Results in undirected graphs
            - May produce disconnected components for certain point distributions
        """
        try:
            timer_gabriel = timeit.default_timer()

            # Centralize data conversion
            data_array = self._get_data_as_array(data_points)

            # Call simplified algorithm function (aspect no longer needed)
            graph = create_gabriel_graph(data_array)

            elapsed_ms = round((timeit.default_timer() - timer_gabriel) * 1000, 3)
            logging.debug(f"Gabriel graph completed in {elapsed_ms}ms")

            return graph

        except Exception as e:
            raise GraphCreationError(f"Failed to create Gabriel graph: {str(e)}")

    def make_mst(self,
                data_points: Union[np.ndarray, Dict[str, Any]],
                metric: str = None) -> Any:
        """
        Create a Minimum Spanning Tree (MST) graph from point data.

        An MST connects all points with the minimum total edge weight (distance)
        while maintaining connectivity. This creates a tree structure (no cycles)
        that spans all vertices with the smallest possible total edge cost.

        MSTs are useful for:
        - Network design with minimum cost connectivity
        - Hierarchical clustering visualization
        - Finding natural data structure
        - Creating sparse connected graphs

        Args:
            data_points: Point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays
            metric: Distance metric for edge weights. If None, uses
                   config.graph.distance_metric. Options include:
                   - 'euclidean': Standard Euclidean distance (default)
                   - 'manhattan': Sum of absolute differences
                   - 'chebyshev': Maximum coordinate difference

        Returns:
            Any: igraph Graph object representing the MST.
                 Contains vertices with attributes: id, x, y, name
                 Contains exactly (n-1) edges forming a tree structure.
                 Edge weights represent distances between connected points.

        Raises:
            GraphCreationError: If input data is invalid, aspect mismatch occurs,
                               or MST computation fails.

        Examples:
            >>> # Basic MST with default Euclidean distance
            >>> data = np.random.rand(20, 3) * 100
            >>> mst_graph = grapher.make_mst(data)

            >>> # MST with Manhattan distance
            >>> mst_graph = grapher.make_mst(data, metric='manhattan')

            >>> # MST from clustered data to see natural groupings
            >>> # Create two clusters
            >>> cluster1 = np.column_stack([range(25), np.random.rand(25)*20, np.random.rand(25)*20])
            >>> cluster2 = np.column_stack([range(25,50), np.random.rand(25)*20+50, np.random.rand(25)*20+50])
            >>> data = np.vstack([cluster1, cluster2])
            >>> mst_graph = grapher.make_mst(data)

            >>> # Dictionary format
            >>> data_dict = {
            ...     "id": list(range(30)),
            ...     "x": np.random.rand(30) * 100,
            ...     "y": np.random.rand(30) * 100
            ... }
            >>> grapher.aspect = "dict"
            >>> mst_graph = grapher.make_mst(data_dict)

        Note:
            - Always produces exactly (n-1) edges for n vertices
            - Guaranteed to be connected (single component)
            - No cycles by definition
            - Unique if all edge weights are distinct
            - Performance: O(E log V) with efficient algorithms (Kruskal's/Prim's)
            - Sensitive to distance metric choice
        """
        try:
            if metric is None:
                metric = self.config.graph.distance_metric

            timer_mst = timeit.default_timer()

            # Centralize data conversion
            data_array = self._get_data_as_array(data_points)

            # Call simplified algorithm function (aspect no longer needed)
            graph = create_minimum_spanning_tree(data_array, metric=metric)

            elapsed_ms = round((timeit.default_timer() - timer_mst) * 1000, 3)
            logging.debug(f"MST computation completed in {elapsed_ms}ms")

            return graph

        except Exception as e:
            raise GraphCreationError(f"Failed to create MST graph: {str(e)}")

    # ============================================================================
    # PLUGIN SYSTEM METHODS
    # ============================================================================

    def make_graph(self, graph_type: str, data_points: Union[np.ndarray, Dict[str, Any]], **kwargs) -> Any:
        """
        Create a graph using the extensible plugin system.

        This method provides access to both built-in and community-contributed
        graph types through a unified interface. It automatically handles data
        format conversion and passes the appropriate parameters to the graph
        creation algorithm.

        Args:
            graph_type: Name of the graph type to create. Built-in types include:
                       'delaunay', 'proximity', 'knn', 'gabriel', 'mst', 'memory'.
                       Additional types may be available through plugins.
            data_points: Point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays
            **kwargs: Graph-type specific parameters. These vary by graph type:
                     - proximity: proximity_thresh, metric
                     - knn: k (number of neighbors)
                     - mst: metric
                     - memory: memory_connections
                     - Custom plugins may have their own parameters

        Returns:
            Any: igraph Graph object of the specified type.

        Raises:
            ValueError: If graph_type is not found in the registry.
            GraphCreationError: If graph creation fails due to invalid parameters
                               or computation errors.

        Examples:
            >>> # Create built-in graph types
            >>> delaunay_graph = grapher.make_graph('delaunay', data)
            >>> proximity_graph = grapher.make_graph('proximity', data, proximity_thresh=60.0)
            >>> knn_graph = grapher.make_graph('knn', data, k=5)
            >>> mst_graph = grapher.make_graph('mst', data, metric='manhattan')

            >>> # Example with custom plugin (hypothetical)
            >>> voronoi_graph = grapher.make_graph('voronoi', data, boundary_mode='reflect')

            >>> # Check available graph types
            >>> available_types = grapher.list_graph_types()
            >>> print(available_types.keys())

        Note:
            - Plugin system allows for extensibility without modifying core code
            - Built-in graph types are always available
            - Community plugins may have additional dependencies
            - Use list_graph_types() to see all available graph types
            - Use get_graph_info() to see parameters for specific graph types
        """
        try:
            from .plugins_logic import get_graph_registry

            # Centralize data conversion BEFORE calling the plugin system
            data_array = self._get_data_as_array(data_points)

            registry = get_graph_registry()
            # Pass the standardized array to the registry.
            # The 'aspect' parameter is no longer needed by the plugin's create_graph method.
            return registry.create_graph(
                graph_type=graph_type,
                data_points=data_array,
                dimension=self.dimension,
                **kwargs
            )
        except Exception as e:
            raise GraphCreationError(f"Failed to create {graph_type} graph: {str(e)}")

    @staticmethod
    def list_graph_types(category: Optional[str] = None) -> Dict[str, Any]:
        """
        List all available graph types in the plugin registry.

        Args:
            category: Optional category filter to show only specific types:
                     - 'built-in': Core graph types included with graphizy
                     - 'community': Community-contributed plugins
                     - 'experimental': Experimental or unstable plugins
                     - None: Show all available types

        Returns:
            Dict[str, Any]: Dictionary mapping graph type names to their information.
                           Each entry contains metadata about the graph type including
                           description, category, version, and available parameters.

        Examples:
            >>> # List all graph types
            >>> all_types = Graphing.list_graph_types()
            >>> for name, info in all_types.items():
            ...     print(f"{name}: {info['description']}")

            >>> # List only built-in types
            >>> builtin_types = Graphing.list_graph_types(category='built-in')

            >>> # Check if specific type is available
            >>> available_types = Graphing.list_graph_types()
            >>> if 'delaunay' in available_types:
            ...     print("Delaunay triangulation is available")
        """
        from .plugins_logic import get_graph_registry

        registry = get_graph_registry()
        return registry.list_plugins(category)

    @staticmethod
    def get_plugin_info(graph_type: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific graph type.

        Args:
            graph_type: Name of the graph type to query.

        Returns:
            Dict[str, Any]: Detailed information including:
                           - info: General information (description, category, version)
                           - parameters: List of available parameters with descriptions
                           - examples: Usage examples if available
                           - requirements: Special requirements or dependencies

        Raises:
            ValueError: If graph_type is not found in the registry.

        Examples:
            >>> # Get info about proximity graphs
            >>> prox_info = Graphing.get_plugin_info('proximity')  # ✅ Fixed method name
            >>> print(prox_info['info']['description'])
            >>> print("Parameters:", prox_info['parameters'])

            >>> # Check parameter details before calling
            >>> knn_info = Graphing.get_plugin_info('knn')  # ✅ Fixed method name
            >>> k_param = knn_info['parameters']['k']
            >>> print(f"k parameter: {k_param['description']}")
        """
        from .plugins_logic import get_graph_registry

        registry = get_graph_registry()
        plugin = registry.get_plugin(graph_type)
        return {
            "info": plugin.info.__dict__,
            "parameters": plugin.info.parameters
        }

    # ============================================================================
    # VISUALIZATION METHODS (Delegated to Visualizer)
    # ============================================================================

    def draw_graph(self, graph: Any, **kwargs) -> np.ndarray:
        """
        Draw a graph to an image array.

        This method provides a convenient top-level API by delegating the drawing
        task to the internal Visualizer instance.

        Args:
            graph: igraph Graph object to draw.
            **kwargs: Additional arguments for the visualizer, e.g., 'radius', 'thickness'.

        Returns:
            np.ndarray: An RGB image array of the drawn graph.
        """
        try:
            return self.visualizer.draw_graph(graph, **kwargs)
        except Exception as e:
            raise DrawingError(f"Failed to draw graph: {e}") from e

    def draw_memory_graph(self, graph: Any, **kwargs) -> np.ndarray:
        """
        Draw a memory graph with optional age-based coloring.

        Delegates to the Visualizer's draw_memory_graph method.

        Args:
            graph: igraph Graph object to draw.
            **kwargs: Additional arguments like 'use_age_colors', 'alpha_range'.

        Returns:
            np.ndarray: An RGB image array of the drawn memory graph.
        """
        try:
            return self.visualizer.draw_memory_graph(graph, **kwargs)
        except Exception as e:
            raise DrawingError(f"Failed to draw memory graph: {e}") from e

    def overlay_graph(self, image_graph: np.ndarray, graph: Any) -> np.ndarray:
        """
        Overlay an additional graph onto an existing image.

        Delegates to the Visualizer's overlay_graph method.

        Args:
            image_graph: The base image to draw on.
            graph: The igraph Graph object to overlay.

        Returns:
            np.ndarray: The modified image array.
        """
        try:
            return self.visualizer.overlay_graph(image_graph, graph)
        except Exception as e:
            raise DrawingError(f"Failed to overlay graph: {e}") from e

    def overlay_collision(self, image_graph: np.ndarray, graph: Any) -> np.ndarray:
        """
        Overlay an additional graph onto an existing image.

        Delegates to the Visualizer's overlay_graph method.

        Args:
            image_graph: The base image to draw on.
            graph: The igraph Graph object to overlay.

        Returns:
            np.ndarray: The modified image array.
        """
        try:
            return self.visualizer.overlay_collision(image_graph, graph)
        except Exception as e:
            raise DrawingError(f"Failed to overlay graph: {e}") from e

    def show_graph(self, image_graph: np.ndarray, title: str = "Graphizy", **kwargs) -> None:
        """
        Display a graph image in a window.

        Delegates to the Visualizer's show_graph method.

        Args:
            image_graph: The image array to display.
            title: The title of the window.
            **kwargs: Additional arguments like 'block'.
        """
        try:
            self.visualizer.show_graph(image_graph, title, **kwargs)
        except Exception as e:
            raise DrawingError(f"Failed to show graph: {e}") from e

    def save_graph(self, image_graph: np.ndarray, filename: str) -> None:
        """
        Save a graph image to a file.

        Delegates to the Visualizer's save_graph method.

        Args:
            image_graph: The image array to save.
            filename: The path to save the file to.
        """
        try:
            self.visualizer.save_graph(image_graph, filename)
        except Exception as e:
            raise DrawingError(f"Failed to save graph: {e}") from e

    # ============================================================================
    # MEMORY MANAGEMENT METHODS
    # ============================================================================

    def init_memory_manager(self,
                           max_memory_size: int = 100,
                           max_iterations: int = None,
                           track_edge_ages: bool = True) -> 'MemoryManager':
        """
        Initialize memory manager for temporal graph analysis.

        The memory manager enables tracking of graph connections over time,
        allowing for analysis of persistent vs. transient relationships and
        temporal patterns in graph structure.

        Args:
            max_memory_size: Maximum number of connections to remember. Older
                           connections are forgotten when this limit is reached.
                           Larger values provide longer memory but use more resources.
            max_iterations: Maximum number of time steps to track. If None,
                          tracks indefinitely until max_memory_size is reached.
            track_edge_ages: Whether to track the age/persistence of each edge.
                           Enables advanced temporal analysis but uses more memory.

        Returns:
            MemoryManager: The initialized memory manager instance.

        Raises:
            GraphCreationError: If memory manager initialization fails.

        Examples:
            >>> # Basic memory manager
            >>> memory_mgr = grapher.init_memory_manager()

            >>> # Large memory for long-term analysis
            >>> memory_mgr = grapher.init_memory_manager(
            ...     max_memory_size=1000,
            ...     max_iterations=100,
            ...     track_edge_ages=True
            ... )

            >>> # Lightweight memory for real-time applications
            >>> memory_mgr = grapher.init_memory_manager(
            ...     max_memory_size=50,
            ...     track_edge_ages=False
            ... )

        Note:
            - Must be called before using memory-based graph methods
            - Only one memory manager per Graphing instance
            - Memory manager persists until explicitly reset or object destroyed
        """
        try:
            self.memory_manager = MemoryManager(max_memory_size, max_iterations, track_edge_ages)
            logging.info(f"Memory manager initialized: max_size={max_memory_size}, "
                        f"max_iterations={max_iterations}, track_ages={track_edge_ages}")
            self.visualizer.memory_manager=self.memory_manager
            return self.memory_manager
        except Exception as e:
            raise GraphCreationError(f"Failed to initialize memory manager: {str(e)}")

    def make_memory_graph(self,
                         data_points: Union[np.ndarray, Dict[str, Any]],
                         memory_connections: Optional[Dict] = None) -> Any:
        """
        Create a graph based on accumulated memory connections.

        Memory graphs use historical connection data to create graphs that
        represent persistent relationships over time. This is useful for
        analyzing temporal stability and identifying core vs. peripheral
        connections in dynamic systems.

        Args:
            data_points: Current point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays
            memory_connections: Optional explicit memory connections. If None,
                              uses the current memory manager's accumulated connections.
                              Format: {(id1, id2): connection_strength, ...}

        Returns:
            Any: igraph Graph object representing memory-based connections.
                 Edge weights may represent connection persistence/frequency.

        Raises:
            GraphCreationError: If memory manager is not initialized and no
                               memory_connections provided, or if graph creation fails.

        Examples:
            >>> # Initialize memory and accumulate connections over time
            >>> grapher.init_memory_manager(max_memory_size=200)
            >>>
            >>> # Update memory with multiple proximity snapshots
            >>> for t in range(10):
            ...     dynamic_data = get_data_at_time(t)  # Your data source
            ...     grapher.update_memory_with_proximity(dynamic_data)
            >>>
            >>> # Create memory graph from accumulated connections
            >>> current_data = get_current_data()
            >>> memory_graph = grapher.make_memory_graph(current_data)

            >>> # Use explicit memory connections
            >>> custom_memory = {(1, 2): 0.8, (2, 3): 0.6, (1, 3): 0.3}
            >>> memory_graph = grapher.make_memory_graph(data, custom_memory)

        Note:
            - Requires either initialized memory manager or explicit connections
            - Memory graphs can be much sparser than instantaneous graphs
            - Edge weights typically represent temporal persistence
            - Useful for identifying stable vs. transient relationships
        """
        try:
            if memory_connections is None:
                if self.memory_manager is None:
                    raise GraphCreationError("No memory manager initialized and no connections provided")
                memory_connections = self.memory_manager.get_current_memory_graph()

            return create_memory_graph(data_points, memory_connections, self.aspect)
        except Exception as e:
            raise GraphCreationError(f"Failed to create memory graph: {str(e)}")

    def update_memory_with_proximity(self,
                                   data_points: Union[np.ndarray, Dict[str, Any]],
                                   proximity_thresh: float = None) -> None:
        """
        Update memory manager with current proximity-based connections.

        This method computes proximity connections for the current data and
        adds them to the memory manager's historical record. Over time, this
        builds up a picture of which points are frequently close to each other.

        Args:
            data_points: Current point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays
            proximity_thresh: Distance threshold for connections. If None, uses
                            config.graph.proximity_threshold.

        Raises:
            GraphCreationError: If memory manager is not initialized or update fails.

        Examples:
            >>> # Initialize memory tracking
            >>> grapher.init_memory_manager(max_memory_size=500)
            >>>
            >>> # Simulate temporal data and track proximity over time
            >>> for timestep in range(50):
            ...     # Get data for current timestep (your implementation)
            ...     current_data = simulate_moving_points(timestep)
            ...
            ...     # Update memory with current proximity connections
            ...     grapher.update_memory_with_proximity(current_data, proximity_thresh=25.0)
            >>>
            >>> # Analyze accumulated memory
            >>> stats = grapher.get_memory_stats()
            >>> print(f"Total connections remembered: {stats['total_connections']}")

            >>> # Create memory graph showing persistent connections
            >>> memory_graph = grapher.make_memory_graph(current_data)

        Note:
            - Must call init_memory_manager() first
            - Each call adds current proximity connections to memory
            - Memory manager automatically handles size limits and aging
            - Frequent calls with similar data will strengthen connection memories
        """
        try:
            if self.memory_manager is None:
                raise GraphCreationError("Memory manager not initialized. Call init_memory_manager() first")

            if proximity_thresh is None:
                proximity_thresh = self.config.graph.proximity_threshold

            return update_memory_from_proximity(
                data_points,
                proximity_thresh,
                self.memory_manager,
                self.config.graph.distance_metric,
                self.aspect
            )
        except Exception as e:
            raise GraphCreationError(f"Failed to update memory with proximity: {str(e)}")

    def update_memory_with_delaunay(self, data_points: Union[np.ndarray, Dict[str, Any]]) -> None:
        """
        Update memory manager with Delaunay triangulation connections.

        This method computes Delaunay triangulation for the current data and
        adds the resulting connections to memory. This is useful for tracking
        natural neighbor relationships over time.

        Args:
            data_points: Point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays

        Raises:
            GraphCreationError: If memory manager is not initialized or update fails.

        Examples:
            >>> # Track Delaunay connections over time for moving points
            >>> grapher.init_memory_manager(max_memory_size=300)
            >>>
            >>> # Simulate points moving and track neighbor changes
            >>> for t in range(20):
            ...     # Move points according to some dynamics
            ...     data = update_point_positions(data, t)
            ...
            ...     # Track which points remain Delaunay neighbors
            ...     grapher.update_memory_with_delaunay(data)
            >>>
            >>> # Create graph of persistent neighbor relationships
            >>> persistent_neighbors = grapher.make_memory_graph(data)

        Note:
            - Delaunay connections change as points move
            - Memory tracks which pairs were ever Delaunay neighbors
            - Useful for understanding local neighborhood stability
            - Combines well with other memory update methods
        """
        try:
            if self.memory_manager is None:
                raise GraphCreationError("Memory manager not initialized")

            return update_memory_from_delaunay(
                data_points,
                self.memory_manager,
                self.aspect,
                self.dimension
            )
        except Exception as e:
            raise GraphCreationError(f"Failed to update memory with Delaunay: {str(e)}")

    def update_memory_with_graph(self, graph: Any) -> None:
        """
        Update memory manager from any existing graph object.

        This method extracts connections from an existing igraph Graph object
        and adds them to the memory manager. This is useful for incorporating
        connections computed by external algorithms or for combining multiple
        graph types in memory.

        Args:
            graph: igraph Graph object with vertices having "id" attributes
                   and edges representing connections to remember.

        Raises:
            GraphCreationError: If memory manager is not initialized or update fails.

        Examples:
            >>> # Initialize memory
            >>> grapher.init_memory_manager()
            >>>
            >>> # Create various graph types and add to memory
            >>> delaunay_graph = grapher.make_delaunay(data)
            >>> grapher.update_memory_with_graph(delaunay_graph)
            >>>
            >>> knn_graph = grapher.make_knn(data, k=5)
            >>> grapher.update_memory_with_graph(knn_graph)
            >>>
            >>> # Memory now contains union of both graph types
            >>> combined_memory_graph = grapher.make_memory_graph(data)

            >>> # Update with external graph
            >>> external_graph = some_external_algorithm(data)
            >>> grapher.update_memory_with_graph(external_graph)

        Note:
            - Graph must have vertex "id" attributes matching your data
            - All edges in the graph will be added to memory
            - Useful for combining multiple graph construction methods
            - Can be used with any igraph-compatible graph object
        """
        try:
            if self.memory_manager is None:
                raise GraphCreationError("Memory manager not initialized")

            return update_memory_from_graph(graph, self.memory_manager)
        except Exception as e:
            raise GraphCreationError(f"Failed to update memory with graph: {str(e)}")

    def update_memory_with_custom(self,
                                 data_points: Union[np.ndarray, Dict[str, Any]],
                                 connection_function: callable,
                                 **kwargs) -> None:
        """
        Update memory using a custom connection function.

        This method allows integration of custom graph algorithms or connection
        rules with the memory system. The connection function should return
        pairs of point IDs that should be connected.

        Args:
            data_points: Point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays
            connection_function: Callable that takes data_points and returns connections.
                                Should return iterable of (id1, id2) tuples or similar.
            **kwargs: Additional arguments passed to the connection function.

        Raises:
            GraphCreationError: If memory manager is not initialized or update fails.

        Examples:
            >>> # Define custom connection rule
            >>> def angular_connections(data_points, angle_thresh=45):
            ...     \"\"\"Connect points with similar angles from origin\"\"\"
            ...     connections = []
            ...     # Your custom logic here
            ...     angles = np.arctan2(data_points[:, 2], data_points[:, 1])  # y, x
            ...     for i, angle_i in enumerate(angles):
            ...         for j, angle_j in enumerate(angles[i+1:], i+1):
            ...             if abs(angle_i - angle_j) < np.radians(angle_thresh):
            ...                 connections.append((data_points[i, 0], data_points[j, 0]))
            ...     return connections
            >>>
            >>> # Initialize memory and use custom function
            >>> grapher.init_memory_manager()
            >>> grapher.update_memory_with_custom(
            ...     data,
            ...     angular_connections,
            ...     angle_thresh=30
            ... )

            >>> # Example with lambda function for simple rules
            >>> grapher.update_memory_with_custom(
            ...     data,
            ...     lambda pts: [(pts[i,0], pts[j,0]) for i in range(len(pts))
            ...                  for j in range(i+1, len(pts))
            ...                  if abs(pts[i,1] - pts[j,1]) < 10]  # Connect similar x-coords
            ... )

        Note:
            - Connection function should be efficient for large datasets
            - Function should return iterable of (id1, id2) pairs
            - Memory manager handles deduplication and aging automatically
            - Useful for domain-specific connection rules
        """
        try:
            if self.memory_manager is None:
                raise GraphCreationError("Memory manager not initialized")

            return update_memory_from_custom_function(
                data_points,
                self.memory_manager,
                connection_function,
                self.aspect,
                **kwargs
            )
        except Exception as e:
            raise GraphCreationError(f"Failed to update memory with custom function: {str(e)}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the current memory state.

        Returns:
            Dict[str, Any]: Memory statistics including:
                           - total_connections: Number of unique connections remembered
                           - memory_usage: Current memory utilization
                           - oldest_connection: Age of oldest remembered connection
                           - newest_connection: Age of newest connection
                           - connection_frequency: Distribution of connection frequencies
                           - And other temporal analysis metrics

        Examples:
            >>> stats = grapher.get_memory_stats()
            >>> print(f"Remembering {stats['total_connections']} connections")
            >>> print(f"Memory {stats['memory_usage']:.1%} full")

            >>> # Check if memory is getting full
            >>> if stats['memory_usage'] > 0.9:
            ...     print("Memory nearly full, oldest connections being forgotten")
        """
        if self.memory_manager is None:
            return {"error": "Memory manager not initialized"}
        return self.memory_manager.get_memory_stats()

    def get_memory_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive memory analysis including age statistics.

        This method provides detailed analysis of the temporal patterns in
        the memory manager's data, including connection persistence, age
        distributions, and temporal stability metrics.

        Returns:
            Dict[str, Any]: Comprehensive analysis including:
                           - Basic statistics (count, usage, etc.)
                           - Age distributions and temporal patterns
                           - Connection persistence metrics
                           - Stability analysis
                           - Temporal trends

        Examples:
            >>> analysis = grapher.get_memory_analysis()
            >>> print("Connection age distribution:")
            >>> for age, count in analysis['age_distribution'].items():
            ...     print(f"  Age {age}: {count} connections")
            >>>
            >>> print(f"Average connection persistence: {analysis['avg_persistence']:.2f}")
        """
        try:
            if self.memory_manager is None:
                return {"error": "Memory manager not initialized"}

            return self.memory_manager.get_memory_stats()

        except Exception as e:
            return {"error": f"Failed to get memory analysis: {str(e)}"}




    # ============================================================================
    # GRAPH ANALYSIS AND METRICS METHODS
    # ============================================================================

    @staticmethod
    def get_connections_per_object(graph: Any) -> Dict[Any, int]:
        """
        Calculate the degree (number of connections) for each vertex in the graph.

        This method provides a user-friendly mapping from original object IDs
        to their connectivity counts, which is essential for analyzing graph
        structure and identifying hubs or isolated nodes.

        Args:
            graph: igraph Graph object with vertices having "id" attributes.

        Returns:
            Dict[Any, int]: Dictionary mapping each object's original ID to its degree.
                           Empty dict if graph is None or has no vertices.

        Raises:
            IgraphMethodError: If degree calculation fails.

        Examples:
            >>> connections = Graphing.get_connections_per_object(graph)
            >>> print(f"Object 101 has {connections[101]} connections")
            >>>
            >>> # Find most connected objects
            >>> sorted_objects = sorted(connections.items(), key=lambda x: x[1], reverse=True)
            >>> print(f"Most connected: {sorted_objects[:5]}")
            >>>
            >>> # Find isolated objects
            >>> isolated = [obj_id for obj_id, degree in connections.items() if degree == 0]
            >>> print(f"Isolated objects: {isolated}")

            >>> # Degree distribution analysis
            >>> from collections import Counter
            >>> degree_dist = Counter(connections.values())
            >>> print(f"Degree distribution: {dict(degree_dist)}")

        Note:
            - Returns degree in graph-theoretic sense (number of incident edges)
            - For undirected graphs, each edge contributes 1 to each endpoint's degree
            - For directed graphs, returns total degree (in-degree + out-degree)
            - Empty graphs return empty dictionary
            - Object IDs must be stored in vertex "id" attribute
        """
        try:
            if graph is None or graph.vcount() == 0:
                return {}

            # Get degrees and map to original IDs
            degrees = graph.degree()
            object_ids = graph.vs["id"]

            return {obj_id: degree for obj_id, degree in zip(object_ids, degrees)}

        except Exception as e:
            raise IgraphMethodError(f"Failed to get connections per object: {str(e)}")

    @staticmethod
    def average_path_length(graph: Any) -> float:
        """
        Calculate the average shortest path length between all pairs of vertices.

        This metric indicates how "close" vertices are to each other on average.
        Lower values suggest better connectivity and shorter communication paths.

        Args:
            graph: igraph Graph object, must be connected for meaningful results.

        Returns:
            float: Average path length across all vertex pairs.

        Raises:
            IgraphMethodError: If calculation fails (e.g., disconnected graph).

        Examples:
            >>> avg_path = Graphing.average_path_length(graph)
            >>> print(f"Average path length: {avg_path:.2f}")

            >>> # Compare different graph types
            >>> delaunay_avg = Graphing.average_path_length(delaunay_graph)
            >>> mst_avg = Graphing.average_path_length(mst_graph)
            >>> print(f"Delaunay: {delaunay_avg:.2f}, MST: {mst_avg:.2f}")

        Note:
            - Requires connected graph (use call_method_safe for disconnected graphs)
            - Computed over all pairs of vertices
            - Values typically range from 1 (complete graph) to n-1 (path graph)
            - Higher values indicate less efficient connectivity
        """
        try:
            return call_igraph_method(graph, "average_path_length")
        except Exception as e:
            raise IgraphMethodError(f"Failed to calculate average path length: {str(e)}")

    @staticmethod
    def density(graph: Any) -> float:
        """
        Calculate the density of the graph.

        Density is the ratio of actual edges to possible edges, indicating
        how close the graph is to being complete. Values range from 0 (no edges)
        to 1 (complete graph).

        Args:
            graph: igraph Graph object.

        Returns:
            float: Graph density between 0.0 and 1.0.

        Examples:
            >>> density = Graphing.density(graph)
            >>> print(f"Graph density: {density:.3f} ({density*100:.1f}% of possible edges)")

            >>> # Compare sparsity of different graph types
            >>> print(f"Delaunay density: {Graphing.density(delaunay_graph):.3f}")
            >>> print(f"MST density: {Graphing.density(mst_graph):.3f}")
            >>> print(f"k-NN density: {Graphing.density(knn_graph):.3f}")

        Note:
            - MSTs have density 2(n-1)/(n(n-1)) = 2/(n) for n vertices
            - Complete graphs have density 1.0
            - Empty graphs have density 0.0
            - Useful for comparing graph sparsity
        """
        try:
            return call_igraph_method(graph, "density")
        except Exception as e:
            raise IgraphMethodError(f"Failed to calculate density: {str(e)}")

    def call_method_brutal(self, graph: Any, method_name: str, return_format: str = "auto", *args, **kwargs) -> Any:
        """
        Call any igraph method with intelligent return type formatting.

        This method provides flexible access to igraph's extensive method library
        with automatic formatting of results into user-friendly formats. It handles
        the conversion between igraph's internal representations and more intuitive
        Python data structures.

        Args:
            graph: igraph Graph object to operate on.
            method_name: Name of the igraph method to call (e.g., "betweenness", "closeness").
            return_format: Output format specification:
                          - "auto": Intelligent format detection (recommended)
                          - "dict": Force dict format {object_id: value} for per-vertex results
                          - "list": Force list format [value1, value2, ...] for array results
                          - "raw": Return exactly what igraph provides (no processing)
            *args: Positional arguments passed to the igraph method.
            **kwargs: Keyword arguments passed to the igraph method.

        Returns:
            Any: Method result formatted according to return_format:
                 - Per-vertex results: dict mapping object_id -> value (auto/dict)
                 - Per-edge results: list of values (auto/list)
                 - Scalar results: single value (all formats)
                 - Complex results: depends on method and format

        Raises:
            IgraphMethodError: If method call fails or method doesn't exist.
            ValueError: If return_format is invalid.

        Examples:
            >>> # Get degree centrality as dict
            >>> degrees = grapher.call_method_brutal(graph, "degree", "dict")
            >>> print(f"Object 5 degree: {degrees[5]}")

            >>> # Get betweenness centrality with auto-formatting
            >>> betweenness = grapher.call_method_brutal(graph, "betweenness")
            >>> # Returns dict {object_id: betweenness_value}

            >>> # Get raw igraph output
            >>> raw_closeness = grapher.call_method_brutal(graph, "closeness", "raw")

            >>> # Call method with parameters
            >>> shortest_paths = grapher.call_method_brutal(
            ...     graph, "shortest_paths", "raw",
            ...     source=0, target=5
            ... )

            >>> # Edge-related method (returns list)
            >>> edge_betweenness = grapher.call_method_brutal(graph, "edge_betweenness")

        Note:
            - "auto" format is usually the most convenient
            - Per-vertex methods automatically map to object IDs when possible
            - Some methods may not support all return formats
            - Use "raw" format when you need igraph's exact output
            - Method availability depends on igraph version and graph type
        """
        try:
            # Validate return_format parameter
            valid_formats = ["auto", "dict", "list", "raw"]
            if return_format not in valid_formats:
                raise ValueError(f"return_format must be one of {valid_formats}, got: {return_format}")

            # Call the underlying igraph method
            result = call_igraph_method(graph, method_name, *args, **kwargs)

            # Handle return formatting based on parameter
            if return_format == "raw":
                return result

            elif return_format == "list":
                # Force list format for list-like results
                if isinstance(result, list):
                    return result
                elif hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
                    return list(result)
                else:
                    return result

            elif return_format == "dict":
                # Force dict format for per-vertex results
                if isinstance(result, list):
                    if len(result) == graph.vcount():
                        return {obj_id: value for obj_id, value in zip(graph.vs["id"], result)}
                    else:
                        # List doesn't match vertex count, return as-is with warning
                        logging.warning(f"Method {method_name} returned list of length {len(result)} "
                                      f"but graph has {graph.vcount()} vertices. Returning raw list.")
                        return result
                elif hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
                    # Convert other iterables to list and try dict conversion
                    result_list = list(result)
                    if len(result_list) == graph.vcount():
                        return {obj_id: value for obj_id, value in zip(graph.vs["id"], result_list)}
                    else:
                        return result_list
                else:
                    return result

            elif return_format == "auto":
                # Intelligent automatic formatting (enhanced logic)
                if isinstance(result, list):
                    # Check if it's a per-vertex result
                    if len(result) == graph.vcount():
                        # Per-vertex result - return as dict mapping object_id -> value
                        return {obj_id: value for obj_id, value in zip(graph.vs["id"], result)}
                    elif len(result) == graph.ecount():
                        # Per-edge result - return as list (could enhance later for edge mapping)
                        return result
                    else:
                        # Other list result (like connected components) - return as-is
                        return result

                elif isinstance(result, (int, float, bool, str, type(None))):
                    # Scalar values or None - return as-is
                    return result

                elif hasattr(result, '__iter__') and not isinstance(result, (str, dict)):
                    # Other iterable types (like igraph specific objects)
                    try:
                        result_list = list(result)
                        if len(result_list) == graph.vcount():
                            # Looks like per-vertex data
                            return {obj_id: value for obj_id, value in zip(graph.vs["id"], result_list)}
                        else:
                            return result_list
                    except:
                        # If conversion fails, return as-is
                        return result

                else:
                    # Complex objects, custom types, etc. - return as-is
                    return result

        except Exception as e:
            raise IgraphMethodError(f"Failed to call method '{method_name}': {str(e)}")

    def get_connectivity_info(self, graph: Any) -> Dict[str, Any]:
        """
        Get comprehensive connectivity information about the graph.

        This method analyzes the graph's connectivity structure, identifying
        connected components and providing statistics about graph cohesion.
        Essential for understanding graph topology and planning analyses.

        Args:
            graph: igraph Graph object to analyze.

        Returns:
            Dict[str, Any]: Comprehensive connectivity information:
                           - is_connected: Boolean indicating if graph is fully connected
                           - num_components: Number of disconnected components
                           - components: List of vertex lists for each component
                           - component_sizes: List of component sizes
                           - largest_component_size: Size of largest component
                           - largest_component_index: Index of largest component
                           - connectivity_ratio: Fraction of vertices in largest component
                           - isolation_ratio: Fraction of isolated vertices (size-1 components)

        Examples:
            >>> conn_info = grapher.get_connectivity_info(graph)
            >>> if conn_info['is_connected']:
            ...     print("Graph is fully connected")
            ... else:
            ...     print(f"Graph has {conn_info['num_components']} components")
            ...     print(f"Largest component: {conn_info['largest_component_size']} vertices")

            >>> # Analyze fragmentation
            >>> if conn_info['isolation_ratio'] > 0.1:
            ...     print(f"Warning: {conn_info['isolation_ratio']:.1%} vertices are isolated")

            >>> # Focus analysis on largest component
            >>> if not conn_info['is_connected']:
            ...     largest_comp = conn_info['components'][conn_info['largest_component_index']]
            ...     subgraph = graph.subgraph(largest_comp)
            ...     # Analyze subgraph...

        Note:
            - Connected components are maximal sets of mutually reachable vertices
            - Component indices refer to the components list
            - Isolated vertices form size-1 components
            - Useful for determining appropriate analysis methods
        """
        try:
            components_result = self.call_method_brutal(graph, 'connected_components', "raw")
            
            # Convert to list of lists if it's an igraph-specific type
            if hasattr(components_result, '__iter__') and not isinstance(components_result, list):
                components = [list(comp) for comp in components_result]
            else:
                components = components_result
                
            is_connected = len(components) == 1

            component_sizes = [len(comp) for comp in components]

            connectivity_info = {
                'is_connected': is_connected,
                'num_components': len(components),
                'components': components,
                'component_sizes': component_sizes,
                'largest_component_size': max(component_sizes) if component_sizes else 0,
                'largest_component_index': np.argmax(component_sizes) if component_sizes else None,
                'connectivity_ratio': max(component_sizes) / graph.vcount() if graph.vcount() > 0 and component_sizes else 0,
                'isolation_ratio': sum(1 for size in component_sizes if size == 1) / graph.vcount() if graph.vcount() > 0 else 0
            }

            return connectivity_info

        except Exception as e:
            raise IgraphMethodError(f"Failed to get connectivity info: {str(e)}")

    def is_connected(self, graph: Any) -> bool:
        """
        Check if the graph is connected (single component).

        Args:
            graph: igraph Graph object to test.

        Returns:
            bool: True if graph is connected, False otherwise.
        """
        return self.call_method_safe(graph, 'is_connected')

    def call_method_safe(self, graph: Any, method_name: str, return_format: str = "auto",
                         component_mode: str = "connected_only", handle_disconnected: bool = True,
                         default_value: Any = None, *args, **kwargs) -> Any:
        """
        Resilient version of call_method that handles disconnected graphs intelligently.

        Many graph algorithms fail on disconnected graphs. This method provides
        robust computation by applying different strategies for handling disconnected
        components, with graceful fallback to default values when computation fails.

        Args:
            graph: igraph Graph object to analyze.
            method_name: Name of the igraph method to call.
            return_format: Output format ("auto", "dict", "list", "raw").
            component_mode: Strategy for disconnected graphs:
                           - "all": Compute on all components separately
                           - "largest": Compute only on largest component
                           - "connected_only": Compute only on components with >1 vertex
            handle_disconnected: Whether to apply special disconnected graph handling.
            default_value: Value to return/use when computation fails (default: None).
            *args: Positional arguments for the igraph method.
            **kwargs: Keyword arguments for the igraph method.

        Returns:
            Any: Method result with appropriate disconnected graph handling and formatting.

        Examples:
            >>> # Safe diameter computation (fails on disconnected graphs normally)
            >>> diameter = grapher.call_method_safe(graph, "diameter", default_value=float('inf'))

            >>> # Betweenness centrality for all components
            >>> betweenness = grapher.call_method_safe(
            ...     graph, "betweenness", "dict",
            ...     component_mode="all", default_value=0.0
            ... )

            >>> # Average path length only for largest component
            >>> avg_path = grapher.call_method_safe(
            ...     graph, "average_path_length",
            ...     component_mode="largest", default_value=None
            ... )

            >>> # Robust clustering coefficient
            >>> clustering = grapher.call_method_safe(
            ...     graph, "transitivity_local_undirected", "dict",
            ...     component_mode="connected_only", default_value=0.0
            ... )

        Note:
            - Automatically detects connectivity-sensitive methods
            - Provides meaningful results even for highly fragmented graphs
            - Maps component-level results back to full graph vertex space
            - Graceful degradation with informative logging
            - Essential for robust analysis pipelines
        """
        try:
            # Methods that always work regardless of connectivity
            CONNECTIVITY_SAFE_METHODS = {
                'degree', 'density', 'vcount', 'ecount', 'connected_components',
                'transitivity_undirected', 'transitivity_local_undirected', 'is_connected'
            }

            # Methods that fail on disconnected graphs
            CONNECTIVITY_SENSITIVE_METHODS = {
                'diameter', 'average_path_length', 'betweenness', 'closeness',
                'shortest_paths', 'get_shortest_paths'
            }

            # If method is connectivity-safe or we're not handling disconnected graphs, use normal call
            if (method_name in CONNECTIVITY_SAFE_METHODS or not handle_disconnected):
                try:
                    result = self.call_method_brutal(graph, method_name, return_format, *args, **kwargs)
                    # Handle NaN values in the result
                    return self._clean_nan_values(result, default_value)
                except Exception as e:
                    if default_value is not None:
                        return default_value
                    raise

            # For connectivity-sensitive methods, check connectivity first
            connectivity_info = self.get_connectivity_info(graph)

            if connectivity_info['is_connected']:
                # Graph is connected - safe to compute normally
                result = self.call_method_brutal(graph, method_name, return_format, *args, **kwargs)
                return self._clean_nan_values(result, default_value)

            # Graph is disconnected - handle based on component_mode
            if component_mode == "largest":
                return self._compute_on_largest_component(graph, connectivity_info, method_name,
                                                          return_format, default_value, *args, **kwargs)
            elif component_mode == "all":
                return self._compute_on_all_components(graph, connectivity_info, method_name,
                                                       return_format, default_value, *args, **kwargs)
            elif component_mode == "connected_only":
                return self._compute_on_connected_components(graph, connectivity_info, method_name,
                                                             return_format, default_value, *args, **kwargs)
            else:
                raise ValueError(
                    f"Invalid component_mode: {component_mode}. Must be 'largest', 'all', or 'connected_only'")

        except Exception as e:
            if default_value is not None:
                logging.warning(f"Method '{method_name}' failed: {e}. Returning default value: {default_value}")
                return default_value
            raise IgraphMethodError(f"Failed to call resilient method '{method_name}': {str(e)}")

    def _clean_nan_values(self, result, default_value=0.0):
        """Clean NaN and inf values from results, replacing with default_value."""
        if isinstance(result, (list, np.ndarray)):
            return [default_value if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x for x in result]
        elif isinstance(result, dict):
            return {k: (default_value if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else v)
                    for k, v in result.items()}
        elif isinstance(result, float) and (np.isnan(result) or np.isinf(result)):
            return default_value
        return result

    def _compute_on_largest_component(self, graph, connectivity_info, method_name, return_format,
                                      default_value, *args, **kwargs):
        """Compute metric on largest component only."""
        components = connectivity_info['components']
        largest_component = max(components, key=len) if components else []

        if len(largest_component) < 2:
            # Component too small for meaningful computation
            if return_format in ["list", "dict"]:
                return [default_value] * graph.vcount() if default_value is not None else []
            return default_value

        try:
            # Create subgraph of largest component
            subgraph = graph.subgraph(largest_component)
            result = self.call_method_brutal(subgraph, method_name, "raw", *args, **kwargs)

            # Map result back to full graph if needed
            if return_format in ["list", "dict"] and isinstance(result, list):
                # Create full result array with default values
                full_result = [default_value] * graph.vcount()
                for i, vertex_idx in enumerate(largest_component):
                    if i < len(result):
                        full_result[vertex_idx] = result[i]

                if return_format == "dict":
                    return {graph.vs[i]["id"]: full_result[i] for i in range(len(full_result))}
                return full_result

            return self._format_result(result, return_format, graph)

        except Exception as e:
            logging.warning(f"Computation on largest component failed: {e}")
            if return_format in ["list", "dict"]:
                return [default_value] * graph.vcount() if default_value is not None else []
            return default_value

    def _compute_on_all_components(self, graph, connectivity_info, method_name, return_format,
                                   default_value, *args, **kwargs):
        """Compute metric on all components separately."""
        components = connectivity_info['components']

        if return_format in ["list", "dict"]:
            full_result = [default_value] * graph.vcount()
        else:
            component_results = []

        for component in components:
            if len(component) < 2:
                # Component too small - use default values
                if return_format in ["list", "dict"]:
                    for vertex_idx in component:
                        full_result[vertex_idx] = default_value
                else:
                    component_results.append(default_value)
                continue

            try:
                # Create subgraph and compute metric
                subgraph = graph.subgraph(component)
                result = self.call_method_brutal(subgraph, method_name, "raw", *args, **kwargs)

                if return_format in ["list", "dict"]:
                    # Map results back to full graph
                    if isinstance(result, list):
                        for i, vertex_idx in enumerate(component):
                            if i < len(result):
                                full_result[vertex_idx] = result[i]
                    else:
                        # Scalar result - apply to all nodes in component
                        for vertex_idx in component:
                            full_result[vertex_idx] = result
                else:
                    component_results.append(result)

            except Exception as e:
                logging.warning(f"Computation on component failed: {e}")
                if return_format in ["list", "dict"]:
                    for vertex_idx in component:
                        full_result[vertex_idx] = default_value
                else:
                    component_results.append(default_value)

        if return_format in ["list", "dict"]:
            if return_format == "dict":
                return {graph.vs[i]["id"]: full_result[i] for i in range(len(full_result))}
            return full_result
        else:
            return component_results

    def _compute_on_connected_components(self, graph, connectivity_info, method_name, return_format,
                                         default_value, *args, **kwargs):
        """Compute metric only on components with size > 1."""
        components = connectivity_info['components']
        connected_components = [comp for comp in components if len(comp) > 1]

        if not connected_components:
            # No connected components
            if return_format in ["list", "dict"]:
                result = [default_value] * graph.vcount()
                if return_format == "dict":
                    return {graph.vs[i]["id"]: result[i] for i in range(len(result))}
                return result
            return default_value

        # Use the all components approach but only for connected ones
        modified_connectivity = dict(connectivity_info)
        modified_connectivity['components'] = connected_components

        return self._compute_on_all_components(graph, modified_connectivity, method_name,
                                               return_format, default_value, *args, **kwargs)

    def _format_result(self, result, return_format, graph):
        """Format result according to return_format."""
        if return_format == "raw":
            return result
        elif return_format == "list":
            return list(result) if hasattr(result, '__iter__') and not isinstance(result, str) else [result]
        elif return_format == "dict":
            if isinstance(result, list) and len(result) == graph.vcount():
                return {graph.vs[i]["id"]: result[i] for i in range(len(result))}
            else:
                return {"global": result}
        else:  # auto
            return result

    def compute_component_metrics(self, graph: Any, metrics_list: List[str],
                                  component_mode: str = "largest") -> Dict[str, Any]:
        """
        Compute multiple graph metrics with consistent component handling.

        This method efficiently computes multiple metrics on the same graph
        with unified handling of disconnected components. Ideal for comprehensive
        graph analysis with consistent treatment of connectivity issues.

        Args:
            graph: igraph Graph object to analyze.
            metrics_list: List of metric names to compute. Examples:
                         ['degree', 'betweenness', 'closeness', 'diameter',
                          'transitivity_undirected', 'average_path_length']
            component_mode: Strategy for disconnected graphs ("all", "largest", "connected_only").

        Returns:
            Dict[str, Any]: Dictionary with computed metrics:
                           - connectivity_info: Detailed connectivity analysis
                           - [metric_name]: Result for each requested metric
                           - Failed metrics are set to None with warning logged

        Examples:
            >>> # Comprehensive analysis of a graph
            >>> metrics = grapher.compute_component_metrics(
            ...     graph,
            ...     ['degree', 'betweenness', 'closeness', 'diameter', 'transitivity_undirected'],
            ...     component_mode="all"
            ... )
            >>>
            >>> print(f"Graph diameter: {metrics['diameter']}")
            >>> print(f"Average degree: {np.mean(list(metrics['degree'].values()))}")
            >>>
            >>> # Check connectivity
            >>> if not metrics['connectivity_info']['is_connected']:
            ...     print(f"Warning: Graph has {metrics['connectivity_info']['num_components']} components")

            >>> # Focus on largest component only
            >>> largest_metrics = grapher.compute_component_metrics(
            ...     graph,
            ...     ['average_path_length', 'diameter', 'betweenness'],
            ...     component_mode="largest"
            ... )

        Note:
            - Provides comprehensive analysis in a single call
            - Handles disconnected graphs gracefully
            - Includes connectivity analysis automatically
            - Failed metrics are logged but don't stop other computations
            - Efficient for multiple related metrics on same graph
        """
        try:
            results = {}
            connectivity_info = self.get_connectivity_info(graph)

            # Add connectivity information
            results['connectivity_info'] = connectivity_info

            # Compute each metric
            for metric_name in metrics_list:
                try:
                    result = self.call_method_safe(
                        graph, metric_name, "auto",
                        component_mode=component_mode,
                        handle_disconnected=True,
                        default_value=0.0
                    )
                    results[metric_name] = result

                except Exception as e:
                    logging.warning(f"Failed to compute {metric_name}: {e}")
                    results[metric_name] = None

            return results

        except Exception as e:
            raise IgraphMethodError(f"Failed to compute component metrics: {str(e)}")

    @staticmethod
    def call_method_raw(graph: Any, method_name: str, *args, **kwargs) -> Any:
        """
        Call any igraph method on the graph, returning unformatted output.

        This method provides direct access to igraph's methods without any
        processing or formatting of the results. Useful when you need the
        exact output format that igraph provides.

        Args:
            graph: igraph Graph object to operate on.
            method_name: Name of the igraph method to call.
            *args: Positional arguments for the method.
            **kwargs: Keyword arguments for the method.

        Returns:
            Any: Exact result from the igraph method call, no processing applied.

        Raises:
            IgraphMethodError: If method call fails or method doesn't exist.

        Examples:
            >>> # Get raw degree sequence
            >>> raw_degrees = Graphing.call_method_raw(graph, "degree")
            >>> print(type(raw_degrees))  # <class 'list'>

            >>> # Get raw connected components
            >>> raw_components = Graphing.call_method_raw(graph, "connected_components")
            >>> print(type(raw_components))  # igraph-specific type

            >>> # Call with parameters
            >>> raw_paths = Graphing.call_method_raw(
            ...     graph, "shortest_paths",
            ...     source=0, target=[1, 2, 3]
            ... )

        Note:
            - No processing, formatting, or error handling beyond basic method call
            - Returns exactly what igraph provides (may be igraph-specific types)
            - Use when you need maximum control over the output format
            - Static method - can be called without Graphing instance
        """
        return call_igraph_method(graph, method_name, *args, **kwargs)

    def get_graph_info(self, graph: Any) -> Dict[str, Any]:
        """
        Get comprehensive information about the graph structure and properties.

        This method provides a detailed summary of the graph's basic properties
        and advanced metrics, with robust handling of edge cases like empty
        graphs or disconnected components.

        Args:
            graph: igraph Graph object to analyze.

        Returns:
            Dict[str, Any]: Comprehensive graph information:
                           - vertex_count: Number of vertices
                           - edge_count: Number of edges
                           - density: Graph density (0.0 to 1.0)
                           - is_connected: Whether graph is connected
                           - average_path_length: Average distance between vertices (if connected)
                           - diameter: Maximum shortest path length (if connected)
                           - transitivity: Global clustering coefficient (if has edges)

        Examples:
            >>> info = grapher.get_graph_info(graph)
            >>> print(f"Graph: {info['vertex_count']} vertices, {info['edge_count']} edges")
            >>> print(f"Density: {info['density']:.3f}, Connected: {info['is_connected']}")
            >>>
            >>> if info['diameter'] is not None:
            ...     print(f"Diameter: {info['diameter']}, Avg path: {info['average_path_length']:.2f}")
            >>>
            >>> if info['transitivity'] is not None:
            ...     print(f"Clustering: {info['transitivity']:.3f}")

            >>> # Check for degenerate cases
            >>> if info['vertex_count'] == 0:
            ...     print("Empty graph")
            >>> elif info['edge_count'] == 0:
            ...     print("Graph with no edges (isolated vertices)")

        Note:
            - Handles empty graphs and graphs without edges gracefully
            - Advanced metrics set to None if not computable
            - Safe for disconnected graphs (uses call_method_safe internally)
            - Provides foundation for more detailed analysis
        """
        try:
            info = {}

            # Basic properties (always computable)
            info['vertex_count'] = self.call_method_safe(graph, 'vcount')
            info['edge_count'] = self.call_method_safe(graph, 'ecount')
            info['density'] = self.density(graph)
            info['is_connected'] = self.call_method_safe(graph, 'is_connected')

            # Advanced properties (conditional on graph structure)
            if info['vertex_count'] > 0:
                if info['edge_count'] > 0:
                    # Graph has edges - compute path-based metrics
                    try:
                        info['average_path_length'] = self.call_method_safe(
                            graph, 'average_path_length',
                            component_mode="largest", default_value=None
                        )
                    except:
                        info['average_path_length'] = None

                    try:
                        info['diameter'] = self.call_method_safe(
                            graph, 'diameter',
                            component_mode="largest", default_value=None
                        )
                    except:
                        info['diameter'] = None

                    try:
                        info['transitivity'] = self.call_method_safe(
                            graph, 'transitivity_undirected',
                            default_value=None
                        )
                    except:
                        info['transitivity'] = None
                else:
                    # No edges - path metrics undefined
                    info['average_path_length'] = None
                    info['diameter'] = None
                    info['transitivity'] = None
            else:
                # Empty graph - all metrics undefined
                info['average_path_length'] = None
                info['diameter'] = None
                info['transitivity'] = None

            return info

        except Exception as e:
            raise IgraphMethodError(f"Failed to get graph info: {str(e)}")

