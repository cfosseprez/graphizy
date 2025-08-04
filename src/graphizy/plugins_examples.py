"""
Plugin System Examples and Documentation

This file demonstrates how to easily add new graph types to Graphizy. It serves
as a best-practice template for creating your own plugins.

The process involves three simple steps:
1.  **Write the Core Logic**: Create a function that builds your graph. This
    function should accept a standardized NumPy array of shape (n, 3)
    containing [id, x, y] columns.
2.  **Create a Plugin Wrapper**: Wrap your logic in a plugin class or use a
    decorator. This is the "glue" that tells Graphizy about your new graph
    type, its name, and its parameters.
3.  **Register the Plugin**: A single function call makes your new graph
    type available everywhere in Graphizy.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
from typing import Any, Optional, List, Tuple

# Import the plugin system
from graphizy.plugins_logic import GraphTypePlugin, GraphTypeInfo, register_graph_type, graph_type_plugin
from .algorithms import create_graph_array

# ============================================================================
# STEP 1: DEFINE THE CORE GRAPH LOGIC
#
# This function contains the actual algorithm. It is self-contained and
# operates on a standardized NumPy array. For a real plugin, this function
# would typically live in `algorithms.py` or your own separate module.
# ============================================================================

def create_radial_graph(data_points: np.ndarray, data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
    """
    Creates a graph connecting points based on their radial distance from a center.

    Args:
        data_points: A NumPy array of shape (n, m) with columns defined by data_shape.
        data_shape: The shape of the data, used to create the base graph.
        **kwargs: Algorithm-specific parameters.
    """
    # --- Parameter Parsing ---
    center_x = kwargs.get("center_x")
    center_y = kwargs.get("center_y")
    radius_threshold = kwargs.get("radius_threshold", 150.0)
    ring_connections = kwargs.get("ring_connections", True)
    ring_tolerance = kwargs.get("ring_tolerance", 30.0)
    spoke_tolerance_rad = kwargs.get("spoke_tolerance_rad", 0.2)

    # --- Graph and Position Setup ---
    # FIX: Pass data_shape to create the base graph with all attributes.
    graph = create_graph_array(data_points, data_shape=data_shape)
    positions = data_points[:, 1:3]

    if center_x is None or center_y is None:
        center = positions.mean(axis=0)
    else:
        center = np.array([center_x, center_y])

    # --- Vectorized Calculations ---
    vectors = positions - center
    radial_distances = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # --- Edge Creation ---
    valid_indices = np.where(radial_distances <= radius_threshold)[0]
    edges_to_add = []
    for i_idx in range(len(valid_indices)):
        for j_idx in range(i_idx + 1, len(valid_indices)):
            i = valid_indices[i_idx]
            j = valid_indices[j_idx]

            if ring_connections:
                if abs(radial_distances[i] - radial_distances[j]) <= ring_tolerance:
                    edges_to_add.append((i, j))
            else:
                angle_diff = abs(angles[i] - angles[j])
                if angle_diff <= spoke_tolerance_rad or angle_diff >= (2 * np.pi - spoke_tolerance_rad):
                    edges_to_add.append((i, j))

    if edges_to_add:
        graph.add_edges(edges_to_add)

    return graph
# ============================================================================
# STEP 2: CREATE PLUGINS TO EXPOSE THE LOGIC TO GRAPHIZY
#
# Now, we wrap the core logic in a plugin. This is the "glue" that
# connects your algorithm to the Graphizy system.
# ============================================================================

# --- METHOD A: The Class-Based Plugin (Recommended & Most Flexible) ---

class RadialGraphPlugin(GraphTypePlugin):
    """A plugin to create graphs based on radial and ring connections."""

    @property
    def info(self) -> GraphTypeInfo:
        """Provides metadata about this plugin for discovery and documentation."""
        return GraphTypeInfo(
            name="radial",
            description="Connects points based on their radial distance from a center.",
            parameters={
                "center_x": {"type": float, "default": None, "description": "X-coordinate of the center."},
                "center_y": {"type": float, "default": None, "description": "Y-coordinate of the center."},
                "radius_threshold": {"type": float, "default": 150.0, "description": "Max radius for connections."},
                "ring_connections": {"type": bool, "default": True, "description": "True for rings, False for spokes."},
            },
            category="custom_example",
            author="Graphizy Team",
            version="1.0.0"
        )

    def create_graph(self, data_points: np.ndarray, dimension: tuple,
                     data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
        """
        This method is called by Graphizy. It acts as a bridge to the core logic.
        It receives the standardized data and passes it to the algorithm function.
        """
        # The `data_points` are already validated and converted to a NumPy array.
        # We just need to call our core logic function.
        return create_radial_graph(data_points, data_shape=data_shape, **kwargs)


# --- METHOD B: The Decorator-Based Plugin (Quick & Easy for Simple Cases) ---

@graph_type_plugin(
    name="connect_to_center",
    description="Connects all points to the geometric center of the canvas.",
    parameters={},
    category="custom_example"
)
def connect_to_center_graph(data_points: np.ndarray, dimension: tuple,
                            data_shape: Optional[List[Tuple[str, Any]]] = None, **kwargs) -> Any:
    """
    A simple graph algorithm implemented directly as a decorated function.
    The decorator automatically handles creating and registering the plugin.
    """
    graph = create_graph_array(data_points, data_shape=data_shape)
    num_points = len(data_points)

    if num_points < 2:
        return graph

    center_id = np.max(data_points[:, 0]) + 1
    center_x, center_y = dimension[0] / 2, dimension[1] / 2
    graph.add_vertex(name=str(center_id), id=center_id, x=center_x, y=center_y)

    center_vertex_index = graph.vcount() - 1
    edges_to_add = [(i, center_vertex_index) for i in range(num_points)]
    graph.add_edges(edges_to_add)

    return graph


# ============================================================================
# STEP 3: REGISTER YOUR PLUGINS
#
# This single call makes the `RadialGraphPlugin` available to `grapher.make_graph()`.
# The decorator-based plugin is registered automatically.
# ============================================================================

def register_example_plugins():
    """A function to explicitly register all class-based plugins in this file."""
    register_graph_type(RadialGraphPlugin())

# This line ensures that simply importing this file is enough to register the plugins.
register_example_plugins()
