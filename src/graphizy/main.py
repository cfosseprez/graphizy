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
from typing import Union, Dict, Any, List, Tuple, Optional, TYPE_CHECKING
import numpy as np
from networkx.algorithms.clique import make_max_clique_graph

if TYPE_CHECKING:
    from .networkx_bridge import NetworkXAnalyzer
    from .streaming import StreamManager, AsyncStreamManager

from graphizy.config import (
    GraphizyConfig, DrawingConfig, GraphConfig, MemoryConfig, WeightConfig,
    GenerationConfig, LoggingConfig,
)
from graphizy.exceptions import (
    InvalidAspectError, InvalidDimensionError, GraphCreationError,
    IgraphMethodError, DrawingError,
)
from graphizy.algorithms import (
    create_graph_array, create_graph_dict, call_igraph_method,
    create_delaunay_graph, create_proximity_graph,
    create_mst_graph, create_knn_graph, create_gabriel_graph,
)
from graphizy.analysis import GraphAnalysisResult
from graphizy.data_interface import DataInterface
from graphizy.memory import (
    MemoryManager, update_memory_from_custom_function
)
from graphizy.weight import (WeightComputer, setup_realtime_weight_computer)
from graphizy.drawing import Visualizer
from graphizy.plugins_logic import get_graph_registry


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
                 config: Optional[GraphizyConfig] = None,
                 **kwargs):
        """
        Initialize Graphing object with a flexible configuration system.

        You can provide a pre-made GraphizyConfig object for detailed control,
        or override specific settings directly with keyword arguments for ease of use.

        Args:
            config: A pre-configured GraphizyConfig object. If None, a default
                    config is created.
            **kwargs: Keyword arguments to override default settings. These are
                      applied on top of the provided or default config.
                      Examples:
                        - dimension=(800, 600)
                        - line_color=(255, 0, 0)
                        - proximity_thresh=75.0
                        - data_shape=[('id', int), ('x', int), ('y', int)]

        Raises:
            GraphCreationError: If initialization fails due to configuration issues.

        Examples:
            >>> # Easiest way: Use keyword arguments
            >>> grapher = Graphing(dimension=(800, 600), line_color=(255, 0, 0))

            >>> # Power-user way: Create and pass a config object
            >>> my_config = GraphizyConfig()
            >>> my_config.drawing.point_radius = 10
            >>> grapher = Graphing(config=my_config)

            >>> # Hybrid: Use a base config and override with a keyword
            >>> grapher = Graphing(config=my_config, point_radius=5) # 5 wins
        """
        try:
            # If a config object is passed, use it and update it.
            # If not, create a new config directly from the keyword arguments,
            # which ensures the __post_init__ validation in the config classes runs.
            if config:
                self.config = config.copy()
                self.config.update(**kwargs)
            else:
                self.config = GraphizyConfig(**kwargs)

            # Set main attributes from the final configuration
            self.dimension = self.config.graph.dimension
            self.aspect = self.config.graph.aspect

            # Initialize data interface for handling different data formats
            self.data_interface = DataInterface(self.config.graph.data_shape)

            # Get the graph registry
            self.registry = get_graph_registry()

            # Initialize optional managers
            self.memory_manager = None
            self.weight_computer = None
            self.fast_computer = None
            if self.config.weight.auto_compute_weights:
                logging.info("Auto-computation of weights is enabled.")
                self.init_weight_computer()

            # Initialize the visualizer
            self.visualizer = Visualizer(self.config.drawing, self.config.graph.dimension)

            logging.info(f"Graphing object initialized: {self.dimension} canvas, '{self.aspect}' aspect")


        except (InvalidDimensionError, InvalidAspectError, ValueError) as e:
            # Re-raise specific, expected configuration errors as-is.
            raise

        except Exception as e:
            # Wrap any other unexpected errors in a generic GraphCreationError.
            raise GraphCreationError(f"Failed to initialize Graphing object: {str(e)}") from e

    # ============================================================================
    # CONFIGURATIONS FUNCTIONS
    # ============================================================================

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
                self.data_interface = DataInterface(self.config.graph.data_shape)

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

    def set_graph_type(self, graph_type: Union[str, List[str], Tuple[str]], **default_kwargs):
        """
        Set the type(s) of graph to generate automatically during updates.

        This method configures the Graphing object to automatically create specific
        graph types when update_graphs() is called with new data. Supports single
        or multiple graph types with default parameters.

        Args:
            graph_type: Graph type(s) to generate automatically. Can be:
                       - str: Single graph type (e.g., 'delaunay')
                       - List[str]: Multiple graph types (e.g., ['delaunay', 'proximity'])
                       - Tuple[str]: Multiple graph types as tuple
            **default_kwargs: Default parameters for graph creation, applied to all types.
                             Type-specific parameters can be set using update_graph_params().

        Raises:
            ValueError: If any graph_type is not recognized.
            GraphCreationError: If configuration fails.

        Examples:
            >>> # Set single graph type
            >>> grapher.set_graph_type('delaunay')

            >>> # Set multiple graph types
            >>> grapher.set_graph_type(['delaunay', 'proximity', 'knn'])

            >>> # Set with default parameters
            >>> grapher.set_graph_type('proximity', proximity_thresh=50.0, metric='euclidean')

            >>> # Set multiple types with defaults
            >>> grapher.set_graph_type(['knn', 'gabriel'], k=6)  # k applies only to knn
        """
        try:
            # Normalize input to list
            if isinstance(graph_type, str):
                self.graph_types = [graph_type]
            elif isinstance(graph_type, (list, tuple)):
                self.graph_types = list(graph_type)
            else:
                raise ValueError(f"graph_type must be str, list, or tuple, got {type(graph_type)}")

            # Validate all graph types are recognized
            available_types = set(self.list_graph_types().keys())
            for gtype in self.graph_types:
                if gtype not in available_types:
                    raise ValueError(f"Unknown graph type '{gtype}'. Available: {sorted(available_types)}")

            # Store default parameters for each graph type
            self.graph_type_params = {}
            for gtype in self.graph_types:
                self.graph_type_params[gtype] = default_kwargs.copy()

            # Store current graphs (will be populated by update_graphs)
            self.current_graphs = {}

            logging.info(f"Graph types set to: {self.graph_types}")
            if default_kwargs:
                logging.info(f"Default parameters: {default_kwargs}")

        except Exception as e:
            raise GraphCreationError(f"Failed to set graph type: {str(e)}")

    def clear_graph_types(self):
        """
        Clear all configured graph types and current graphs.
        """
        self.graph_types = []
        self.graph_type_params = {}
        self.current_graphs = {}
        logging.info("Cleared all graph types")

    def get_graph_type_info(self) -> Dict[str, Any]:
        """
        Get information about current graph type configuration.

        Returns:
            Dict[str, Any]: Configuration information including types, parameters, and status.
        """
        if not hasattr(self, 'graph_types'):
            return {'configured': False, 'message': 'No graph types configured'}

        return {
            'configured': True,
            'graph_types': self.graph_types.copy(),
            'parameters': self.graph_type_params.copy(),
            'current_graphs_available': {
                gtype: (graph is not None)
                for gtype, graph in getattr(self, 'current_graphs', {}).items()
            }
        }

    def update_graph_params(self, graph_type: str, **kwargs):
        """
        Update parameters for a specific graph type.

        Args:
            graph_type: The graph type to update parameters for.
            **kwargs: Parameters to set for this graph type.

        Examples:
            >>> grapher.set_graph_type(['proximity', 'knn'])
            >>> grapher.update_graph_params('proximity', proximity_thresh=75.0, metric='manhattan')
            >>> grapher.update_graph_params('knn', k=8)
        """
        if not hasattr(self, 'graph_types') or graph_type not in self.graph_types:
            raise ValueError(f"Graph type '{graph_type}' not in current types: {getattr(self, 'graph_types', [])}")

        self.graph_type_params[graph_type].update(kwargs)
        logging.info(f"Updated parameters for '{graph_type}': {kwargs}")

    def update(self, **kwargs):
        """
        Update configuration values at runtime from keyword arguments.
        This method can intelligently route flat keys (e.g., 'line_color')
        to the correct nested config object (e.g., self.drawing).
        """
        for key, value in kwargs.items():
            # Check for nested dictionary updates first (e.g., drawing={...})
            if hasattr(self, key) and isinstance(getattr(self, key),
                                                 (DrawingConfig, GraphConfig, MemoryConfig, WeightConfig,
                                                  GenerationConfig, LoggingConfig)):
                config_obj = getattr(self, key)
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        if hasattr(config_obj, nested_key):
                            setattr(config_obj, nested_key, nested_value)
                        else:
                            raise ValueError(f"Unknown config key in '{key}': {nested_key}")
                else:
                    # Allow replacing the whole object, e.g., config.update(drawing=my_drawing_config)
                    setattr(self, key, value)
            # Route flat keys to the correct sub-config
            elif hasattr(self.drawing, key):
                setattr(self.drawing, key, value)
            elif hasattr(self.graph, key):
                setattr(self.graph, key, value)
            elif hasattr(self.generation, key):
                setattr(self.generation, key, value)
            elif hasattr(self.memory, key):
                setattr(self.memory, key, value)
            elif hasattr(self.weight, key):
                setattr(self.weight, key, value)
            elif hasattr(self.logging, key):
                setattr(self.logging, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")

    # ============================================================================
    # CONVENIENT CONVERSION
    # ============================================================================
    def _get_data_as_array(self, data_points: Union[np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """
        Internal helper that delegates to DataInterface for all conversions.
        """
        try:
            return self.data_interface.to_array(data_points)
        except Exception as e:
            raise GraphCreationError(f"Failed to convert data to array format: {str(e)}")

    # ============================================================================
    # CORE UPDATES FUNCTIONS
    # ============================================================================

    def update_graphs(self, data_points: Union[np.ndarray, Dict[str, Any]],
                      update_memory: Optional[bool] = None,
                      use_memory: Optional[bool] = None,
                      compute_weights: Optional[bool] = None,
                      **override_kwargs) -> Dict[str, Any]:
        """
        Update all configured graph types with new data using smart memory and weight defaults.

        This method automatically creates graphs of all types specified by set_graph_type()
        using the provided data. Optionally updates memory manager, computes weights, and
        returns all generated graphs. Uses the same smart defaults as make_graph().

        Processing Flow: graph → memory → weights (for each graph type)

        Args:
            data_points: New point data in the format specified by self.aspect.
            update_memory: Whether to update memory manager with new graphs.
                          If None and memory manager exists, defaults based on use_memory.
                          Only works if memory_manager is initialized.
            use_memory: Whether to create memory-enhanced graphs from existing connections.
                       If None and memory manager exists, defaults to True.
                       Only works if memory_manager is initialized.
            compute_weights: Whether to compute edge weights for final graphs.
                            If None, uses self.auto_compute_weights default.
                            Only works if weight_computer is initialized.
            **override_kwargs: Parameters that override defaults for this update only.

        Returns:
            Dict[str, Any]: Dictionary mapping graph type names to generated graph objects.
                           Each graph follows the pipeline: base_graph → memory → weights

        Smart Defaults:
            Memory:
            - If memory_manager exists and use_memory=None → use_memory=True
            - If use_memory=True and update_memory=None → update_memory=True
            - If no memory_manager → both default to False

            Weights:
            - If compute_weights=None → use self.auto_compute_weights
            - If weight_computer not set → weights skipped regardless of setting

        Examples:
            >>> # Set up automatic graph generation with memory and weights
            >>> grapher.set_graph_type(['delaunay', 'proximity', 'knn'])
            >>> grapher.init_memory_manager(max_memory_size=200)
            >>> grapher.init_weight_computer(WeightComputer(method="distance"))
            >>> grapher.update_graph_params('proximity', proximity_thresh=60.0)
            >>> grapher.update_graph_params('knn', k=5)

            >>> # Basic update - uses all smart defaults
            >>> new_data = np.random.rand(100, 3) * 100
            >>> graphs = grapher.update_graphs(new_data)
            >>> # Each graph: structure → memory → weights

            >>> # Explicit control over all processing steps
            >>> graphs = grapher.update_graphs(
            ...     new_data,
            ...     use_memory=False,      # Skip memory
            ...     update_memory=True,    # But learn from current
            ...     compute_weights=True   # Force weights
            ... )

            >>> # Memory + parameter overrides
            >>> graphs = grapher.update_graphs(
            ...     new_data,
            ...     use_memory=True,
            ...     compute_weights=False,  # Skip weights this time
            ...     proximity_thresh=75.0   # Override proximity threshold
            ... )

            >>> # Different settings per call
            >>> learning_graphs = grapher.update_graphs(new_data, use_memory=False, update_memory=True)
            >>> memory_graphs = grapher.update_graphs(new_data, use_memory=True, update_memory=False)

        Note:
            - All graphs follow the same pipeline: graph → memory → weights
            - Memory processing can change which edges exist before weights are computed
            - Weight computation adds attributes to final edge set
            - Smart defaults minimize configuration while maintaining full control
            - Failed graphs are set to None but don't stop other graph generation
        """
        try:
            if not hasattr(self, 'graph_types'):
                raise GraphCreationError("No graph types set. Call set_graph_type() first.")

            # Apply smart defaults based on memory manager state
            if self.memory_manager is not None:
                # Memory manager exists - default to using memory
                if use_memory is None:
                    use_memory = True
                # If using memory, default to updating it too (continuous learning)
                if use_memory and update_memory is None:
                    update_memory = True
            else:
                # No memory manager - default to no memory operations
                if use_memory is None:
                    use_memory = False
                if update_memory is None:
                    update_memory = False

            # Apply smart defaults for weight computation
            if compute_weights is None:
                compute_weights = self.config.weight.auto_compute_weights

            timer_start = time.time()
            updated_graphs = {}

            # Generate each configured graph type with full pipeline
            for graph_type in self.graph_types:
                try:
                    # Get stored parameters for this graph type
                    graph_params = self.graph_type_params[graph_type].copy()

                    # Create the graph using make_graph with full pipeline: graph → memory → weights
                    graph = self.make_graph(
                        graph_type=graph_type,
                        data_points=data_points,
                        graph_params=graph_params,
                        update_memory=update_memory,  # Memory processing
                        use_memory=use_memory,  # Memory processing
                        compute_weights=compute_weights,  # Weight processing
                        **override_kwargs  # These override graph_params
                    )
                    updated_graphs[graph_type] = graph

                    logging.debug(f"Updated {graph_type} graph successfully")

                except Exception as e:
                    logging.error(f"Failed to update {graph_type} graph: {e}")
                    updated_graphs[graph_type] = None

            # Store current graphs
            self.current_graphs = updated_graphs

            elapsed_ms = round((time.time() - timer_start) * 1000, 3)
            successful_updates = sum(1 for g in updated_graphs.values() if g is not None)

            # Enhanced logging with memory and weight info
            processing_status = []
            if self.memory_manager is not None:
                processing_status.append(f"memory: use={use_memory}, update={update_memory}")
            if self.weight_computer is not None:
                processing_status.append(f"weights: compute={compute_weights}")

            status_str = f" ({', '.join(processing_status)})" if processing_status else ""

            logging.info(
                f"Updated {successful_updates}/{len(self.graph_types)} graphs in {elapsed_ms}ms{status_str}")

            return updated_graphs

        except Exception as e:
            raise GraphCreationError(f"Failed to update graphs: {str(e)}")

    def update_graphs_memory_only(self, data_points: Union[np.ndarray, Dict[str, Any]],
                                  compute_weights: Optional[bool] = None,
                                  **override_kwargs) -> Dict[str, Any]:
        """
        Convenience method to update graphs using only memory (no current data learning).

        This creates graphs purely from accumulated memory connections without updating
        the memory with current data. Useful for seeing what the "remembered" graph
        structure looks like. Can still compute weights on the memory-based edges.

        Args:
            data_points: Current point data (positions only, connections from memory).
            compute_weights: Whether to compute weights on memory-based edges.
                            If None, uses auto_compute_weights default.
            **override_kwargs: Parameter overrides for graph creation.

        Returns:
            Dict[str, Any]: Dictionary of memory-based graphs with optional weights.

        Examples:
            >>> # Build up memory over time
            >>> grapher.update_graphs(data1)  # Learn from data1
            >>> grapher.update_graphs(data2)  # Learn from data2

            >>> # See what the accumulated memory looks like (with weights)
            >>> memory_graphs = grapher.update_graphs_memory_only(current_data, compute_weights=True)

            >>> # Pure memory structure without weights
            >>> memory_structure = grapher.update_graphs_memory_only(current_data, compute_weights=False)
        """
        return self.update_graphs(
            data_points=data_points,
            use_memory=True,
            update_memory=False,  # Don't learn from current
            compute_weights=compute_weights,
            **override_kwargs
        )

    def update_graphs_learning_only(self, data_points: Union[np.ndarray, Dict[str, Any]],
                                    compute_weights: Optional[bool] = None,
                                    **override_kwargs) -> Dict[str, Any]:
        """
        Convenience method to create regular graphs and update memory (no memory usage).

        This creates graphs from current data and adds the connections to memory
        for future use, but doesn't use existing memory for the current graphs.
        Can still compute weights on the current edges.

        Args:
            data_points: Current point data.
            compute_weights: Whether to compute weights on current edges.
                            If None, uses auto_compute_weights default.
            **override_kwargs: Parameter overrides for graph creation.

        Returns:
            Dict[str, Any]: Dictionary of current graphs with optional weights
                           (memory updated as side effect).

        Examples:
            >>> # Build up memory without using it yet (with weights for analysis)
            >>> grapher.update_graphs_learning_only(data1, compute_weights=True)
            >>> grapher.update_graphs_learning_only(data2, compute_weights=True)

            >>> # Now use accumulated memory
            >>> memory_graphs = grapher.update_graphs_memory_only(current_data)
        """
        return self.update_graphs(
            data_points=data_points,
            use_memory=False,  # Don't use existing memory
            update_memory=True,  # But learn from current
            compute_weights=compute_weights,
            **override_kwargs
        )


    def get_current_graphs(self) -> Dict[str, Any]:
        """
        Get the most recently generated graphs.

        Returns:
            Dict[str, Any]: Dictionary of current graphs by type name.
        """
        return getattr(self, 'current_graphs', {})

    def get_current_graph(self, graph_type: str) -> Any:
        """
        Get the most recent graph of a specific type.

        Args:
            graph_type: The type of graph to retrieve.

        Returns:
            Any: The igraph Graph object, or None if not available.
        """
        current_graphs = self.get_current_graphs()
        return current_graphs.get(graph_type, None)

    # ============================================================================
    # PLUGIN SYSTEM METHODS
    # ============================================================================

    def make_graph(self, graph_type: str, data_points: Union[np.ndarray, Dict[str, Any]],
                   graph_params: Optional[Dict] = None,
                   update_memory: Optional[bool] = None,
                   use_memory: Optional[bool] = None,
                   compute_weights: Optional[bool] = None,
                   do_timing: bool = False,
                   validate_data: bool = False,
                   **kwargs) -> Any:
        """
        Create a graph using the extensible plugin system with intelligent memory defaults.

        This method provides access to both built-in and community-contributed
        graph types through a unified interface. It automatically handles data
        format conversion and passes the appropriate parameters to the graph
        creation algorithm. Optionally integrates with memory system using smart defaults.


        Args:
            graph_type: Name of the graph type to create. Built-in types include:
                       'delaunay', 'proximity', 'knn', 'gabriel', 'mst', 'memory'.
                       Additional types may be available through plugins.
            data_points: Point data in the format specified by self.aspect:
                        - For "array": NumPy array with shape (n, 3) containing [id, x, y]
                        - For "dict": Dictionary with keys "id", "x", "y" as lists/arrays
            graph_params: Dictionary of parameters specific to the graph type.
                         If None, uses empty dict. These are the algorithm-specific parameters:
                         - proximity: {'proximity_thresh': 50.0, 'metric': 'euclidean'}
                         - knn: {'k': 5}
                         - mst: {'metric': 'euclidean'}
                         - etc.
            update_memory: Whether to update memory manager with the created graph.
                          If None and memory manager exists, defaults based on use_memory.
                          Only works if memory_manager is initialized.
            use_memory: Whether to create a memory-enhanced graph from existing connections.
                       If None and memory manager exists, defaults to True.
                       Only works if memory_manager is initialized.
            compute_weights: Whether to compute edges weights. Only works if weight_computer is initialized.
            do_timing: Whether to print the performances
            validate_data: Whether to validate the data (careful at each call this will degrade the performances)
            **kwargs: Additional graph-type specific parameters that override graph_params.
                     These are merged with graph_params, with kwargs taking precedence.

        Returns:
            Any: igraph Graph object of the specified type, optionally memory-enhanced.

        Raises:
            ValueError: If graph_type is not found in the registry.
            GraphCreationError: If graph creation fails due to invalid parameters
                               or computation errors.

        Smart Defaults:
            - If memory_manager exists and use_memory=None → use_memory=True
            - If use_memory=True and update_memory=None → update_memory=True
            - If no memory_manager → both default to False

        Examples:
            >>> # Simple direct usage (most common)
            >>> graph = grapher.make_graph('delaunay', data)
            >>> connections = grapher.make_graph('proximity', data, proximity_thresh=80.0)
            >>> knn_graph = grapher.make_graph('knn', data, k=5)

            >>> # Using graph_params dictionary (for complex configs)
            >>> prox_params = {'proximity_thresh': 75.0, 'metric': 'manhattan'}
            >>> graph = grapher.make_graph('proximity', data, graph_params=prox_params)

            >>> # Mixed usage - kwargs override graph_params
            >>> graph = grapher.make_graph('proximity', data,
            ...                          graph_params={'proximity_thresh': 50.0},
            ...                          proximity_thresh=100.0)  # This wins

            >>> # Memory control with direct parameters
            >>> graph = grapher.make_graph('knn', data, k=8, use_memory=False, update_memory=True)

            >>> # Both styles work seamlessly
            >>> algorithm_params = {'proximity_thresh': 60.0, 'metric': 'euclidean'}
            >>> graph1 = grapher.make_graph('proximity', data, graph_params=algorithm_params)
            >>> graph2 = grapher.make_graph('proximity', data, proximity_thresh=60.0, metric='euclidean')
            >>> # graph1 and graph2 are equivalent

        Note:
            - Direct kwargs are the most convenient: make_graph('proximity', data, proximity_thresh=80.0)
            - graph_params provides clean organization for complex configurations
            - kwargs override graph_params for convenient parameter overrides
            - Both styles can be mixed: graph_params for base config, kwargs for overrides
            - Smart defaults make memory usage automatic when memory_manager exists
            - Explicit parameters always override defaults
            - use_memory=True: Uses EXISTING memory connections from previous calls
            - update_memory=True: Adds current graph connections to memory for future use
            - Memory creates historical connection patterns for temporal analysis
        """

        # Handle parameters
        if graph_params is None:
            graph_params = {}
        final_params = graph_params.copy()
        final_params.update(kwargs)

        # Smart defaults for memory
        if self.memory_manager is not None:
            if use_memory is None:
                use_memory = True
            if use_memory and update_memory is None:
                update_memory = True
        else:
            if use_memory is None:
                use_memory = False
            if update_memory is None:
                update_memory = False

        # Smart defaults for weights
        if compute_weights is None:
            compute_weights = self.config.weight.auto_compute_weights

        try:
            data_array = self.data_interface.to_array(data_points, validate_data=validate_data)

            # STEP 1: Create base graph
            if do_timing:
                start_time_graph = time.perf_counter()
            graph = self.registry.create_graph(
                graph_type=graph_type,
                data_points=data_array,
                dimension=self.dimension,
                data_shape=self.data_interface.data_shape,
                **final_params
            )
            if do_timing:
                end_time_graph = time.perf_counter() - start_time_graph
                start_time_memory = time.perf_counter()

            # STEP 2: Apply memory processing (modifies graph structure)
            graph = self._maybe_apply_memory(graph, use_memory, update_memory)

            if do_timing:
                end_time_memory = time.perf_counter() - start_time_memory
                start_time_weights = time.perf_counter()

            # STEP 3: Compute weights (adds attributes to existing edges)
            graph = self._maybe_compute_weights(graph, compute_weights)

            if do_timing:
                end_time_weights = time.perf_counter() - start_time_weights
                print(
                    f"Graph '{graph_type}' with data shape: {data_array.shape} > {end_time_graph*1000:.1f} ms /"
                    f" memory >{end_time_memory*1000}/ ms"
                    f" weights >{end_time_weights*1000} ms")

            return graph

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

    def draw_all_graphs(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Draw all current graphs to image arrays.

        Args:
            **kwargs: Drawing parameters passed to draw_graph().

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping graph types to image arrays.
        """
        images = {}
        current_graphs = self.get_current_graphs()

        for graph_type, graph in current_graphs.items():
            if graph is not None:
                try:
                    images[graph_type] = self.draw_graph(graph, **kwargs)
                except Exception as e:
                    logging.error(f"Failed to draw {graph_type} graph: {e}")
                    images[graph_type] = None

        return images

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

    def show_all_graphs(self, **kwargs):
        """
        Display all current graphs in separate windows.

        Args:
            **kwargs: Parameters passed to show_graph().
        """
        images = self.draw_all_graphs()

        for graph_type, image in images.items():
            if image is not None:
                title = kwargs.get('title', f"Graphizy - {graph_type.title()}")
                self.show_graph(image, title=title)

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

    def _ensure_memory_integration(self, operation_name: str):
        """Helper to check memory manager state before operations"""
        if self.memory_manager is None:
            logging.warning(f"{operation_name} called but no memory manager initialized")
            return False
        return True

    def _maybe_apply_memory(self, graph: Any, use_memory: bool, update_memory: bool) -> Any:
        """
        Apply memory processing to modify graph structure based on historical connections.

        This method provides flexible control over memory operations in graph processing workflows.
        It can learn from current graphs to build temporal connection patterns and/or create
        memory-enhanced graphs that combine current and historical connections.

        Memory processing enables analysis of temporal stability, identification of core vs.
        peripheral connections, and tracking of relationship persistence over time in dynamic systems.

        Args:
            graph (Any): Input igraph Graph object to process. Must have vertex "id" attributes
                        for proper memory integration. All vertex and edge attributes are preserved.
            use_memory (bool): If True, returns a memory-enhanced graph that includes both
                              current connections and historical connections from memory.
                              If False, returns the original graph (potentially after memory update).
            update_memory (bool): If True, learns connection patterns from the input graph
                                 and adds them to the memory system for future use.
                                 If False, uses existing memory without learning from current graph.

        Returns:
            Any: igraph Graph object with behavior determined by parameter combination:

            - use_memory=False, update_memory=True: Returns original graph unchanged,
              but memory system is updated with current connections for future use.

            - use_memory=True, update_memory=True: Returns memory-enhanced graph with
              additional edges from historical connections. Enhanced graph includes:
              * Original edges with all attributes preserved
              * Memory edges with attributes: "memory_based"=True, "age"=iterations, "weight"=strength

            - use_memory=True, update_memory=False: Returns memory-enhanced graph using
              existing memory without learning from current graph.

            - use_memory=False, update_memory=False: Returns original graph unchanged
              (no memory operations performed).

        Raises:
            GraphCreationError: If memory operations fail due to invalid graph structure,
                               missing vertex IDs, or internal memory system errors.

        Examples:
            >>> # Learn from current graph without enhancement (training mode)
            >>> result = grapher._maybe_apply_memory(
            ...     proximity_graph,
            ...     use_memory=False,
            ...     update_memory=True
            ... )
            >>> # result is identical to proximity_graph, but memory updated
            >>> assert result.ecount() == proximity_graph.ecount()

            >>> # Create memory-enhanced graph with historical connections (analysis mode)
            >>> enhanced = grapher._maybe_apply_memory(
            ...     current_graph,
            ...     use_memory=True,
            ...     update_memory=True
            ... )
            >>> print(f"Original: {current_graph.ecount()} edges")
            >>> print(f"Enhanced: {enhanced.ecount()} edges")
            >>> # Enhanced graph has current + historical edges

            >>> # Use existing memory without learning (inference mode)
            >>> memory_only = grapher._maybe_apply_memory(
            ...     sparse_graph,
            ...     use_memory=True,
            ...     update_memory=False
            ... )
            >>> # Uses existing memory patterns without updating from sparse_graph

        Note:
            - Requires memory_manager to be initialized via init_memory_manager()
            - Memory-enhanced graphs preserve all original vertex and edge attributes
            - Historical connections are only added if both vertices exist in current graph
            - Memory cleanup occurs automatically when configured size limits are exceeded
            - Performance scales with graph size; vectorized operations provide 5-10x speedup
            - Logging at DEBUG level provides detailed operation timing and edge counts

        See Also:
            init_memory_manager(): Initialize memory system with size and aging parameters
            get_memory_stats(): Retrieve statistics about current memory state and performance
            clear_memory(): Clear all historical connections from memory system
            MemoryManager.add_graph_vectorized(): Low-level vectorized memory operations
        """
        if not self.memory_manager:
            return graph

        if update_memory and not use_memory:
            # Learn from current graph but do not return the memory graph
            self.memory_manager.add_graph_vectorized(graph, return_memory_graph=False)
            logging.debug("Updated memory with current graph connections")
            return graph

        if use_memory:
            # Learn from current graph and create memory-enhanced graph
            memory_graph = self.memory_manager.add_graph_vectorized(graph, return_memory_graph=True)
            logging.debug("Created memory-enhanced graph from historical connections")
            return memory_graph

        return graph

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
            # if memory_connections is None:
            if self.memory_manager is None:
                raise GraphCreationError("No memory manager initialized and no connections provided")
                # memory_connections = self.memory_manager.get_current_memory_graph()

            return self.memory_manager.create_memory_graph(data_points)
        except Exception as e:
            raise GraphCreationError(f"Failed to create memory graph: {str(e)}")


    def update_memory_with_graph(self, graph: Any) -> Dict[str, List[str]]:
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

            return self.memory_manager.add_graph_vectorized(graph)
        except Exception as e:
            raise GraphCreationError(f"Failed to update memory with graph: {str(e)}")

    def update_memory_with_custom(self,
                                 data_points: Union[np.ndarray, Dict[str, Any]],
                                 connection_function: callable,
                                 **kwargs) -> Dict[str, List[str]]:
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
    # WEIGHT COMPUTATION METHODS
    # ============================================================================

    def init_weight_computer(self, **kwargs):
        """Initialize flexible weight computer."""
        # If parameters are not passed directly, use the ones from the config
        if 'method' not in kwargs:
            kwargs['method'] = self.config.weight.weight_method
        if 'target_attribute' not in kwargs:
            kwargs['target_attribute'] = self.config.weight.weight_attribute
        if 'formula' not in kwargs and kwargs.get('method') == "formula":
            kwargs['formula'] = self.config.weight.weight_formula

        self.weight_computer = WeightComputer(**kwargs)
        self.config.weight.auto_compute_weights = True

    def compute_edge_attribute(self, graph, attribute_name, method="formula", **kwargs):
        """Compute any edge attribute."""
        if not hasattr(self, 'weight_computer'):
            self.weight_computer = WeightComputer()

        return self.weight_computer.compute_attribute(
            graph, attribute_name, method=method, **kwargs
        )

    def setup_fast_attributes(self, **config):
        """Setup fast attribute computer for real-time use."""
        self.fast_computer = setup_realtime_weight_computer(**config)

    def compute_all_attributes_fast(self, graph):
        """Compute all pre-configured attributes quickly."""
        if hasattr(self, 'fast_computer'):
            return self.fast_computer.compute_multiple_attributes_fast(
                graph, self.fast_computer._default_config
            )
        return graph

    def _maybe_compute_weights(self, graph: Any, compute_weights: bool = None) -> Any:
        """Internal method to auto-compute weights if enabled"""

        # Use the passed parameter, falling back to config if None
        if compute_weights is None:
            compute_weights = self.config.weight.auto_compute_weights

        if self.fast_computer:
            self.compute_all_attributes_fast(graph)

        if compute_weights and self.weight_computer is not None:
            return self.weight_computer.compute_weights(graph)

        return graph

    def get_weight_analysis(self) -> Dict[str, Any]:
        """:Todo implement the weight statistics"""
        pass


    # ============================================================================
    # NETWORKX BRIDGE
    # ============================================================================

    def get_networkx_analyzer(self) -> 'NetworkXAnalyzer':
        """
        Get NetworkX analyzer for advanced graph analysis.

        Returns:
            NetworkXAnalyzer instance for this Graphing object

        Examples:
            >>> # Get analyzer
            >>> nx_analyzer = grapher.get_networkx_analyzer()
            >>>
            >>> # Analyze current graphs
            >>> analysis = nx_analyzer.analyze('delaunay')
            >>> print(f"Communities: {analysis['num_communities']}")
            >>>
            >>> # Direct NetworkX access
            >>> nx_graph = nx_analyzer.get_networkx('proximity')
            >>> custom_centrality = nx.eigenvector_centrality(nx_graph)
        """
        from .networkx_bridge import NetworkXAnalyzer
        return NetworkXAnalyzer(self)

    def to_networkx(self, graph_type: str = None, igraph_graph: Any = None) -> Any:
        """
        Convert graph to NetworkX format.

        Args:
            graph_type: Type from current graphs
            igraph_graph: Manual igraph to convert

        Returns:
            NetworkX Graph object
        """
        from .networkx_bridge import to_networkx

        if igraph_graph is not None:
            return to_networkx(igraph_graph)

        if graph_type is None:
            raise ValueError("Must provide either graph_type or igraph_graph")

        current_graphs = self.get_current_graphs()
        if graph_type not in current_graphs:
            raise ValueError(f"Graph type '{graph_type}' not found")

        return to_networkx(current_graphs[graph_type])

    # ============================================================================
    # ASYNC STEAM METHOD
    # ============================================================================

    def create_stream_manager(self, buffer_size: int = 1000,
                              update_interval: float = 0.1,
                              auto_memory: bool = True) -> 'StreamManager':
        """Create a stream manager for real-time data processing"""
        from .streaming import StreamManager
        return StreamManager(self, buffer_size, update_interval, auto_memory)

    def create_async_stream_manager(self, buffer_size: int = 1000) -> 'AsyncStreamManager':
        """Create async stream manager for high-performance streaming"""
        from .streaming import AsyncStreamManager
        return AsyncStreamManager(self, buffer_size)

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
            dens = call_igraph_method(graph, "density")
            if np.isnan(dens):
                dens = 0.0
            return dens
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

        except ValueError:
            raise
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
            if not hasattr(graph, method_name):
                raise IgraphMethodError(f"Graph does not have method '{method_name}'")

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

    def get_graph_info(self, graph: Any) -> GraphAnalysisResult:
        """
        Get a lazy-loading analysis object for the graph.

        This method is the entry point for all graph analysis. It returns a
        powerful result object where metrics are computed on-demand, ensuring
        maximum performance and a responsive user experience.

        Args:
            graph: igraph Graph object to analyze.

        Returns:
            GraphAnalysisResult: An object for lazily accessing graph metrics.

        Examples:
            >>> # This call is instantaneous
            >>> results = grapher.get_graph_info(graph)

            >>> # The first access to a property computes the metric
            >>> print(f"Density: {results.density}")

            >>> # Subsequent access is instant (from cache)
            >>> print(f"Graph density is {results.density:.4f}")

            >>> # Compute advanced metrics on the fly
            >>> top_hubs = results.get_top_n_by('degree', n=3)
            >>> betweenness_stats = results.get_metric_stats('betweenness')
        """
        if graph is None:
            raise GraphCreationError("Cannot get info for a None graph.")

        # The method is now just a factory. All computation is deferred.
        return GraphAnalysisResult(graph, self)
