"""
Graphizy - A graph maker for computational geometry and network visualization

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

from graphizy.main import Graphing
from graphizy.config import GraphizyConfig, DrawingConfig, GraphConfig, GenerationConfig, LoggingConfig, MemoryConfig
from graphizy.algorithms import (
    generate_positions, make_subdiv, make_delaunay, get_delaunay,
    get_distance, graph_distance, create_graph_array, create_graph_dict, DataInterface,
    call_igraph_method, MemoryManager, create_memory_graph, update_memory_from_proximity,
    update_memory_from_graph, update_memory_from_delaunay, update_memory_from_custom_function,
    create_minimum_spanning_tree, create_k_nearest_graph, create_gabriel_graph
)
from graphizy.drawing import (
    draw_point, draw_line, draw_delaunay, show_graph, save_graph,
    draw_memory_graph_with_aging, create_memory_graph_image
)
from graphizy.exceptions import (
    GraphizyError, InvalidDimensionError, InvalidDataShapeError,
    InvalidAspectError, InvalidPointArrayError, SubdivisionError,
    TriangulationError, GraphCreationError, DrawingError,
    PositionGenerationError, IgraphMethodError, ConfigurationError,
    DependencyError
)
from graphizy.utils import validate_graphizy_input
from graphizy.plugins import (
    GraphTypePlugin, GraphTypeInfo, register_graph_type, 
    get_graph_registry, graph_type_plugin
)

# Import built-in plugins to auto-register them
from graphizy import builtin_plugins

__author__ = "Charles Fosseprez"
__email__ = "charles.fosseprez.pro@gmail.com"
__license__ = "GPL2 or later"

__all__ = [
    # Main class
    "Graphing",

    # Configuration classes
    "GraphizyConfig",
    "DrawingConfig",
    "GraphConfig",
    "GenerationConfig",
    "LoggingConfig",
    "MemoryConfig",

    # Algorithm functions
    "generate_positions",
    "make_subdiv",
    "make_delaunay",
    "get_delaunay",
    "get_distance",
    "graph_distance",  # Added this missing function
    "create_graph_array",
    "create_graph_dict",
    "DataInterface",
    "call_igraph_method",
    "create_minimum_spanning_tree",
    "create_k_nearest_graph",
    "create_gabriel_graph",

    # Memory functions
    "MemoryManager",
    "create_memory_graph",
    "update_memory_from_proximity",
    "update_memory_from_graph",
    "update_memory_from_delaunay",
    "update_memory_from_custom_function",

    # Drawing functions
    "draw_point",
    "draw_line",
    "draw_delaunay",
    "show_graph",
    "save_graph",
    "draw_memory_graph_with_aging",
    "create_memory_graph_image",

    # Exceptions
    "GraphizyError",
    "InvalidDimensionError",
    "InvalidDataShapeError",
    "InvalidAspectError",
    "InvalidPointArrayError",
    "SubdivisionError",
    "TriangulationError",
    "GraphCreationError",
    "DrawingError",
    "PositionGenerationError",
    "IgraphMethodError",
    "ConfigurationError",
    "DependencyError",

    # Utility functions
    "validate_graphizy_input",
    
    # Plugin System
    "GraphTypePlugin",
    "GraphTypeInfo", 
    "register_graph_type",
    "get_graph_registry",
    "graph_type_plugin",
]