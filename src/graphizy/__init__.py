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
    make_subdiv, make_delaunay, get_delaunay,
    get_distance, graph_distance, create_graph_array, create_graph_dict,
    call_igraph_method, create_mst_graph, create_knn_graph, create_gabriel_graph
)
from graphizy.data_interface import DataInterface
from graphizy.memory import (
    MemoryManager
)
from graphizy.positions import (
generate_positions, format_positions, generate_and_format_positions
)
from graphizy.weight import WeightComputer
from graphizy.drawing import (
    draw_point, draw_line, draw_delaunay, show_graph, save_graph,
    draw_memory_graph_with_aging, create_memory_graph_image
)
from graphizy.analysis import (
    GraphAnalysisResult, PercolationAnalyzer, SocialNetworkAnalyzer, AccessibilityAnalyzer,
    PercolationResult, SocialRole, AccessibilityResult
)
from graphizy.exceptions import (
    GraphizyError, InvalidDimensionError, InvalidDataShapeError,
    InvalidAspectError, InvalidPointArrayError, SubdivisionError,
    TriangulationError, GraphCreationError, DrawingError,
    PositionGenerationError, IgraphMethodError, ConfigurationError,
    DependencyError
)
from graphizy.utils import validate_graphizy_input
from graphizy.plugins_logic import (
    GraphTypePlugin, GraphTypeInfo, register_graph_type, 
    get_graph_registry, graph_type_plugin
)
from graphizy.simulator import BrownianSimulator

# Import built-in plugins to auto-register them
from graphizy.builtin_plugins import register_all_builtins
register_all_builtins()


__author__ = "Charles Fosseprez"
__email__ = "charles.fosseprez.pro@gmail.com"
__license__ = "GPL2 or later"

__version__ = "0.1.20"

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

    # Formatting function
    "format_positions",
    "generate_positions",
    "generate_and_format_positions",

    # Algorithm functions
    "make_subdiv",
    "make_delaunay",
    "get_delaunay",
    "get_distance",
    "graph_distance",  # Added this missing function
    "create_graph_array",
    "create_graph_dict",
    "DataInterface",
    "call_igraph_method",
    "create_mst_graph",
    "create_knn_graph",
    "create_gabriel_graph",

    # Memory functions
    "MemoryManager",
    "create_memory_graph",
    "update_memory_from_graph",

    # Weight functions
    "WeightComputer",

    # Drawing functions
    "draw_point",
    "draw_line",
    "draw_delaunay",
    "show_graph",
    "save_graph",
    "draw_memory_graph_with_aging",
    "create_memory_graph_image",

    # Analysis classes and results
    "GraphAnalysisResult",
    "PercolationAnalyzer", 
    "SocialNetworkAnalyzer",
    "AccessibilityAnalyzer",
    "PercolationResult",
    "SocialRole",
    "AccessibilityResult",

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

    # Simulation
    "BrownianSimulator",
]
