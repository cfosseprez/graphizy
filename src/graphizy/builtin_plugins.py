"""
Built-in graph type plugins for Graphizy

This module contains plugin implementations for all the built-in graph types.
It demonstrates the best practice for creating plugins by calling the low-level
algorithm functions directly, avoiding circular dependencies and unnecessary overhead.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
from typing import Any

from .plugins_logic import GraphTypePlugin, GraphTypeInfo, register_graph_type
# Import the core algorithm functions directly, NOT the Graphing class
from .algorithms import (
    create_delaunay_graph, create_proximity_graph,
    create_mst_graph, create_gabriel_graph, create_knn_graph,
    create_visibility_graph, create_voronoi_cell_graph
)


class DelaunayPlugin(GraphTypePlugin):
    """Delaunay triangulation graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="delaunay",
            description="Creates a Delaunay triangulation connecting nearby points optimally",
            parameters={}, category="built-in", author="Charles Fosseprez", version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, **kwargs) -> Any:
        """Create Delaunay triangulation graph by calling the algorithm directly."""
        return create_delaunay_graph(data_points, dimension=dimension)


class ProximityPlugin(GraphTypePlugin):
    """Proximity graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="proximity",
            description="Connects points within a specified distance threshold",
            parameters={
                "proximity_thresh": {"type": float, "default": 50.0, "description": "Maximum distance for connecting points"},
                "metric": {"type": str, "default": "euclidean", "description": "Distance metric to use"}
            },
            category="built-in", author="Charles Fosseprez", version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, **kwargs) -> Any:
        """Create proximity graph by calling the algorithm directly."""
        proximity_thresh = kwargs.get("proximity_thresh", 50.0)
        metric = kwargs.get("metric", "euclidean")
        return create_proximity_graph(data_points, proximity_thresh, metric=metric)


class MSTPlugin(GraphTypePlugin):
    """Minimum Spanning Tree graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="mst",
            description="Creates a minimum spanning tree connecting all points with minimum total edge weight",
            parameters={"metric": {"type": str, "default": "euclidean", "description": "Distance metric for edge weights"}},
            category="built-in", author="Charles Fosseprez", version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, **kwargs) -> Any:
        """Create minimum spanning tree graph by calling the algorithm directly."""
        metric = kwargs.get("metric", "euclidean")
        return create_mst_graph(data_points, metric=metric)


class GabrielPlugin(GraphTypePlugin):
    """Gabriel graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="gabriel",
            description="Creates a Gabriel graph where no other point lies within the diameter circle of two connected points",
            parameters={}, category="built-in", author="Charles Fosseprez", version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, **kwargs) -> Any:
        """Create Gabriel graph by calling the algorithm directly."""
        return create_gabriel_graph(data_points)


class KNNPlugin(GraphTypePlugin):
    """K-Nearest Neighbors graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="knn",
            description="Connects each point to its k nearest neighbors",
            parameters={"k": {"type": int, "default": 4, "description": "Number of neighbors"}},
            category="built-in", author="Charles Fosseprez", version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, **kwargs) -> Any:
        """Create k-NN graph by calling the algorithm directly."""
        k = kwargs.get("k", 4)
        return create_knn_graph(data_points, k=k)



class VisibilityGraphPlugin(GraphTypePlugin):
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="visibility",
            description="Graph connecting points with unobstructed line-of-sight",
            parameters={
                "obstacles": {"type": "list", "default": None, "description": "List of obstacle polygons"}
            },
            category="built-in"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, **kwargs) -> Any:
        return create_visibility_graph(data_points, **kwargs)


class VoronoiCellGraphPlugin(GraphTypePlugin):
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="voronoi_cells",
            description="Graph of Voronoi diagram structure (vertices and ridges)",
            parameters={},
            category="built-in"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, **kwargs) -> Any:
        return create_voronoi_cell_graph(data_points, dimension, **kwargs)


# Register all built-in plugins
def register_builtin_plugins():
    """Register all built-in graph type plugins"""
    register_graph_type(DelaunayPlugin())
    register_graph_type(ProximityPlugin())
    register_graph_type(MSTPlugin())
    register_graph_type(GabrielPlugin())
    register_graph_type(KNNPlugin())


# Auto-register when module is imported
register_builtin_plugins()