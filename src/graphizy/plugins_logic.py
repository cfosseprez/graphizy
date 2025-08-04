# In: src/graphizy/plugins_logic.py

"""
Graph Type Plugin System for Graphizy

This module provides a plugin-based architecture for easily adding new graph types
without modifying core graphizy files.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional, Callable, List, Tuple
import numpy as np
from dataclasses import dataclass
import logging

@dataclass
class GraphTypeInfo:
    """Information about a graph type for documentation and discovery"""
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]  # param_name -> {type, default, description}
    category: str = "custom"  # built-in, community, experimental, etc.
    author: str = ""
    version: str = "1.0.0"
    requires_external_deps: bool = False
    external_deps: list = None

    def __post_init__(self):
        if self.external_deps is None:
            self.external_deps = []


class GraphTypePlugin(ABC):
    """Abstract base class for graph type plugins"""

    @property
    @abstractmethod
    def info(self) -> GraphTypeInfo:
        """Return information about this graph type"""
        pass

    @abstractmethod
    def create_graph(self, data_points: np.ndarray,
                     dimension: tuple, data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
        """
        Create a graph from the given data points.

        Args:
            data_points: Standardized NumPy array of shape (n, m).
            dimension: Image dimensions (width, height).
            data_shape: List of tuples defining the data structure.
            **kwargs: Algorithm-specific parameters.

        Returns:
            igraph Graph object.
        """
        pass

    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """Validate and process parameters for this graph type."""
        processed = {}
        for param_name, param_info in self.info.parameters.items():
            if param_name in kwargs:
                processed[param_name] = kwargs[param_name]
            elif 'default' in param_info:
                processed[param_name] = param_info['default']
        return processed


class GraphTypeRegistry:
    """Registry for managing graph type plugins"""

    def __init__(self):
        self._plugins: Dict[str, GraphTypePlugin] = {}
        self._categories: Dict[str, list] = {}

    def register(self, plugin: GraphTypePlugin) -> None:
        """Register a new graph type plugin"""
        name = plugin.info.name
        if name in self._plugins:
            logging.warning(f"Graph type '{name}' is already registered. Overwriting.")
        self._plugins[name] = plugin
        category = plugin.info.category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
        logging.debug(f"✅ Registered graph type: {name} ({category})")

    def get_plugin(self, name: str) -> GraphTypePlugin:
        """Get a registered plugin by name"""
        if name not in self._plugins:
            available = list(self._plugins.keys())
            raise ValueError(f"Graph type '{name}' not found. Available: {available}")
        return self._plugins[name]

    def list_plugins(self, category: Optional[str] = None) -> Dict[str, GraphTypeInfo]:
        """List all registered plugins, optionally filtered by category"""
        if category:
            if category not in self._categories:
                return {}
            names = self._categories[category]
            return {name: self._plugins[name].info for name in names}
        return {name: plugin.info for name, plugin in self._plugins.items()}

    # --- FIX IS HERE: Added `data_shape` to the signature ---
    def create_graph(self, graph_type: str, data_points: np.ndarray,
                     dimension: tuple, data_shape: List[Tuple[str, Any]], **kwargs) -> Any:
        """Create a graph using a registered plugin"""
        plugin = self.get_plugin(graph_type)
        processed_kwargs = plugin.validate_parameters(**kwargs)
        # Pass data_shape to the plugin's create_graph method
        return plugin.create_graph(data_points, dimension, data_shape, **processed_kwargs)


# Global registry instance
_global_registry = GraphTypeRegistry()


def register_graph_type(plugin: GraphTypePlugin) -> None:
    """Register a graph type plugin globally"""
    _global_registry.register(plugin)


def get_graph_registry() -> GraphTypeRegistry:
    """Get the global graph type registry"""
    return _global_registry


def graph_type_plugin(name: str, description: str, parameters: Dict = None,
                     category: str = "custom", **info_kwargs):
    """Decorator to easily create graph type plugins from functions"""
    if parameters is None:
        parameters = {}

    def decorator(func: Callable):
        class FunctionPlugin(GraphTypePlugin):
            @property
            def info(self):
                return GraphTypeInfo(
                    name=name,
                    description=description,
                    parameters=parameters,
                    category=category,
                    **info_kwargs
                )

            # --- FIX IS HERE: Added `data_shape` to the signature ---
            def create_graph(self, data_points, dimension, data_shape, **kwargs):
                # Pass data_shape to the decorated function
                return func(data_points, dimension, data_shape=data_shape, **kwargs)

        register_graph_type(FunctionPlugin())
        return func
    return decorator