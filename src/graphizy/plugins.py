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
from typing import Any, Dict, Union, Optional, Callable
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
    def create_graph(self, data_points: Union[np.ndarray, Dict[str, Any]], 
                     aspect: str, dimension: tuple, **kwargs) -> Any:
        """
        Create a graph from the given data points
        
        Args:
            data_points: Input data (array or dict format)
            aspect: "array" or "dict" 
            dimension: Image dimensions (width, height)
            **kwargs: Algorithm-specific parameters
            
        Returns:
            igraph Graph object
            
        Raises:
            Exception: If graph creation fails
        """
        pass
    
    def validate_parameters(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and process parameters for this graph type
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Processed parameters dictionary
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Default implementation - just return the kwargs
        # Subclasses can override for custom validation
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
            raise ValueError(f"Graph type '{name}' is already registered")
        
        # Check external dependencies if required
        if plugin.info.requires_external_deps:
            missing_deps = self._check_dependencies(plugin.info.external_deps)
            if missing_deps:
                raise ImportError(
                    f"Graph type '{name}' requires missing dependencies: {missing_deps}"
                )
        
        self._plugins[name] = plugin
        
        # Add to category
        category = plugin.info.category
        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(name)
        
        logging.debug(f"✅ Registered graph type: {name} ({category})")
    
    def unregister(self, name: str) -> None:
        """Unregister a graph type plugin"""
        if name not in self._plugins:
            raise ValueError(f"Graph type '{name}' is not registered")
        
        plugin = self._plugins[name]
        category = plugin.info.category
        
        del self._plugins[name]
        self._categories[category].remove(name)
        
        if not self._categories[category]:
            del self._categories[category]
    
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
    
    def list_categories(self) -> Dict[str, int]:
        """List all categories and the number of plugins in each"""
        return {cat: len(plugins) for cat, plugins in self._categories.items()}
    
    def create_graph(self, graph_type: str, data_points: Union[np.ndarray, Dict[str, Any]], 
                     aspect: str, dimension: tuple, **kwargs) -> Any:
        """Create a graph using a registered plugin"""
        plugin = self.get_plugin(graph_type)
        
        # Validate parameters
        processed_kwargs = plugin.validate_parameters(**kwargs)
        
        # Create the graph
        return plugin.create_graph(data_points, aspect, dimension, **processed_kwargs)
    
    def get_parameter_info(self, graph_type: str) -> Dict[str, Dict[str, Any]]:
        """Get parameter information for a graph type"""
        plugin = self.get_plugin(graph_type)
        return plugin.info.parameters
    
    def _check_dependencies(self, deps: list) -> list:
        """Check if external dependencies are available"""
        missing = []
        for dep in deps:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        return missing


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
    """
    Decorator to easily create graph type plugins from functions
    
    Args:
        name: Graph type name
        description: Description of the graph type
        parameters: Parameter definitions
        category: Plugin category
        **info_kwargs: Additional info parameters
    
    Example:
        @graph_type_plugin(
            name="my_graph",
            description="My custom graph algorithm",
            parameters={
                "threshold": {"type": float, "default": 50.0, "description": "Distance threshold"}
            }
        )
        def my_graph_algorithm(data_points, aspect, dimension, threshold=50.0):
            # Implementation here
            return graph
    """
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
            
            def create_graph(self, data_points, aspect, dimension, **kwargs):
                return func(data_points, aspect, dimension, **kwargs)
        
        # Auto-register the plugin
        register_graph_type(FunctionPlugin())
        return func
    
    return decorator
