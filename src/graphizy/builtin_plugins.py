"""
Built-in Graph Type Plugins for Graphizy

This module defines and registers the core graph construction algorithms
as plugins, making them available through the unified Graphing API.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
from typing import Any, List, Tuple, Optional

from .plugins_logic import GraphTypePlugin, GraphTypeInfo, register_graph_type
# Import the core algorithm functions that the plugins will wrap
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
            description="Creates a Delaunay triangulation connecting nearby points optimally.",
            parameters={},
            category="built-in",
            author="Charles Fosseprez",
            version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple,data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
        """Create Delaunay triangulation graph by calling the algorithm directly."""
        # Pass data_shape down to the algorithm function
        return create_delaunay_graph(data_points, dimension=dimension, data_shape=data_shape)


class ProximityPlugin(GraphTypePlugin):
    """Proximity graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="proximity",
            description="Connects points within a specified distance threshold.",
            parameters={
                "proximity_thresh": {"type": float, "default": 50.0, "description": "Maximum distance for connecting points"},
                "metric": {"type": str, "default": "euclidean", "description": "Distance metric to use"}
            },
            category="built-in",
            author="Charles Fosseprez",
            version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
        """Create proximity graph by calling the algorithm directly."""
        # Pass data_shape and other params down to the algorithm function
        return create_proximity_graph(data_points, data_shape=data_shape, **kwargs)


class KNNPlugin(GraphTypePlugin):
    """K-Nearest Neighbors graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="knn",
            description="Connects each point to its 'k' nearest neighbors.",
            parameters={"k": {"type": int, "default": 4, "description": "Number of nearest neighbors to connect"}},
            category="built-in",
            author="Charles Fosseprez",
            version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
        """Create k-nearest neighbors graph by calling the algorithm directly."""
        # Pass data_shape and other params down to the algorithm function
        return create_knn_graph(data_points, data_shape=data_shape, **kwargs)


class MSTPlugin(GraphTypePlugin):
    """Minimum Spanning Tree graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="mst",
            description="Creates a minimum spanning tree connecting all points with minimum total edge weight.",
            parameters={"metric": {"type": str, "default": "euclidean", "description": "Distance metric for edge weights"}},
            category="built-in",
            author="Charles Fosseprez",
            version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
        """Create minimum spanning tree graph by calling the algorithm directly."""
        # Pass data_shape and other params down to the algorithm function
        return create_mst_graph(data_points, data_shape=data_shape, **kwargs)


class GabrielPlugin(GraphTypePlugin):
    """Gabriel graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="gabriel",
            description="A subgraph of Delaunay where the disk of every edge is empty.",
            parameters={},
            category="built-in",
            author="Charles Fosseprez",
            version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
        """Create Gabriel graph by calling the algorithm directly."""
        # Pass data_shape down to the algorithm function
        return create_gabriel_graph(data_points, data_shape=data_shape)


class VoronoiCellPlugin(GraphTypePlugin):
    """Voronoi cell graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="voronoi_cell",
            description="Creates a graph from Voronoi vertices and ridges.",
            parameters={},
            category="built-in",
            author="Charles Fosseprez",
            version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
        """Create Voronoi cell graph by calling the algorithm directly."""
        # Note: Voronoi creates its own vertices, so data_shape is not used by the algorithm,
        # but the signature must match the abstract base class.
        return create_voronoi_cell_graph(data_points, dimension=dimension)


class VisibilityPlugin(GraphTypePlugin):
    """Visibility graph plugin"""
    @property
    def info(self) -> GraphTypeInfo:
        return GraphTypeInfo(
            name="visibility",
            description="Connects points if they have an unobstructed line of sight.",
            parameters={"obstacles": {"type": list, "default": None, "description": "List of obstacle polygons"}},
            category="built-in",
            author="Charles Fosseprez",
            version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple, data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
        """Create visibility graph by calling the algorithm directly."""
        # Pass data_shape and other params down to the algorithm function
        return create_visibility_graph(data_points, data_shape=data_shape, **kwargs)


def register_all_builtins():
    """
    A convenience function to register all built-in plugins.
    This is typically called once when the graphizy package is initialized.
    """
    register_graph_type(DelaunayPlugin())
    register_graph_type(ProximityPlugin())
    register_graph_type(KNNPlugin())
    register_graph_type(MSTPlugin())
    register_graph_type(GabrielPlugin())
    register_graph_type(VoronoiCellPlugin())
    register_graph_type(VisibilityPlugin())
