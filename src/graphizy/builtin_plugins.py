"""
Built-in graph type plugins for Graphizy

This module contains plugin implementations for all the built-in graph types.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
from typing import Union, Dict, Any

from .plugins import GraphTypePlugin, GraphTypeInfo, register_graph_type
from .algorithms import (
    create_graph_array, create_graph_dict, make_subdiv, graph_delaunay,
    graph_distance, create_minimum_spanning_tree, create_gabriel_graph
)


class DelaunayPlugin(GraphTypePlugin):
    """Delaunay triangulation graph plugin"""
    
    @property
    def info(self):
        return GraphTypeInfo(
            name="delaunay",
            description="Creates a Delaunay triangulation connecting nearby points optimally",
            parameters={},
            category="built-in",
            author="Graphizy Team",
            version="1.0.0"
        )
    
    def create_graph(self, data_points: Union[np.ndarray, Dict[str, Any]], 
                     aspect: str, dimension: tuple, **kwargs):
        """Create Delaunay triangulation graph"""
        from .main import Graphing  # Import here to avoid circular import
        
        # Use existing implementation
        grapher = Graphing(dimension=dimension, aspect=aspect)
        return grapher.make_delaunay(data_points)


class ProximityPlugin(GraphTypePlugin):
    """Proximity graph plugin"""
    
    @property
    def info(self):
        return GraphTypeInfo(
            name="proximity",
            description="Connects points within a specified distance threshold",
            parameters={
                "proximity_thresh": {
                    "type": float,
                    "default": 50.0,
                    "description": "Maximum distance for connecting points"
                },
                "metric": {
                    "type": str,
                    "default": "euclidean",
                    "description": "Distance metric to use"
                }
            },
            category="built-in",
            author="Graphizy Team",
            version="1.0.0"
        )
    
    def create_graph(self, data_points: Union[np.ndarray, Dict[str, Any]], 
                     aspect: str, dimension: tuple, **kwargs):
        """Create proximity graph"""
        from .main import Graphing  # Import here to avoid circular import
        
        proximity_thresh = kwargs.get("proximity_thresh", 50.0)
        metric = kwargs.get("metric", "euclidean")
        
        grapher = Graphing(dimension=dimension, aspect=aspect)
        return grapher.make_proximity(data_points, proximity_thresh, metric)


class MSTPlugin(GraphTypePlugin):
    """Minimum Spanning Tree graph plugin"""
    
    @property
    def info(self):
        return GraphTypeInfo(
            name="mst",
            description="Creates a minimum spanning tree connecting all points with minimum total edge weight",
            parameters={
                "metric": {
                    "type": str,
                    "default": "euclidean",
                    "description": "Distance metric for edge weights"
                }
            },
            category="built-in",
            author="Graphizy Team",
            version="1.0.0"
        )
    
    def create_graph(self, data_points: Union[np.ndarray, Dict[str, Any]], 
                     aspect: str, dimension: tuple, **kwargs):
        """Create minimum spanning tree graph"""
        from .main import Graphing  # Import here to avoid circular import
        
        metric = kwargs.get("metric", "euclidean")
        
        grapher = Graphing(dimension=dimension, aspect=aspect)
        return grapher.make_mst(data_points, metric)


class GabrielPlugin(GraphTypePlugin):
    """Gabriel graph plugin"""
    
    @property
    def info(self):
        return GraphTypeInfo(
            name="gabriel",
            description="Creates a Gabriel graph where two points are connected if no other point lies within their diameter circle",
            parameters={},
            category="built-in",
            author="Graphizy Team",
            version="1.0.0"
        )
    
    def create_graph(self, data_points: Union[np.ndarray, Dict[str, Any]], 
                     aspect: str, dimension: tuple, **kwargs):
        """Create Gabriel graph"""
        from .main import Graphing  # Import here to avoid circular import
        
        grapher = Graphing(dimension=dimension, aspect=aspect)
        return grapher.make_gabriel(data_points)


# Register all built-in plugins
def register_builtin_plugins():
    """Register all built-in graph type plugins"""
    register_graph_type(DelaunayPlugin())
    register_graph_type(ProximityPlugin())
    register_graph_type(MSTPlugin())
    register_graph_type(GabrielPlugin())


# Auto-register when module is imported
register_builtin_plugins()
