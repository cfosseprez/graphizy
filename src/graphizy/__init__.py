"""
Graphizy - A graph maker for computational geometry and network visualization

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

from graphizy.main import Graphing
from graphizy.config import GraphizyConfig, DrawingConfig, GraphConfig, GenerationConfig, LoggingConfig, MemoryConfig
from graphizy.algorithms import (
    generate_positions, make_subdiv, make_delaunay, get_delaunay,
    get_distance, graph_distance, create_graph_array, create_graph_dict, DataInterface,
    call_igraph_method, MemoryManager, create_memory_graph, update_memory_from_proximity
)
from graphizy.drawing import (
    draw_point, draw_line, draw_delaunay, show_graph, save_graph
)
from graphizy.exceptions import (
    GraphizyError, InvalidDimensionError, InvalidDataShapeError,
    InvalidAspectError, InvalidPointArrayError, SubdivisionError,
    TriangulationError, GraphCreationError, DrawingError,
    PositionGenerationError, IgraphMethodError, ConfigurationError,
    DependencyError
)

__version__ = "0.1.0"
__author__ = "Charles Fosseprez"
__email__ = "charles.fosseprez.me@gmail.com"
__license__ = "MIT"

__all__ = [
    # Main class
    "Graphing",

    # Configuration classes
    "GraphizyConfig",
    "DrawingConfig",
    "GraphConfig",
    "GenerationConfig",
    "LoggingConfig",

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

    # Memory function
    "MemoryManager",
    "create_memory_graph",
    "update_memory_from_proximity",
    "MemoryConfig",

    # Drawing functions
    "draw_point",
    "draw_line",
    "draw_delaunay",
    "show_graph",
    "save_graph",

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
]