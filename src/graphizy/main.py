"""
Main graphing class for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

import logging
import time
import timeit
from typing import Union, Dict, Any, List, Tuple, Optional
import numpy as np

from .config import GraphizyConfig, DrawingConfig, GraphConfig
from .exceptions import (
    InvalidAspectError, InvalidDimensionError, GraphCreationError,
    IgraphMethodError, DrawingError
)
from .algorithms import (
    create_graph_array, create_graph_dict, DataInterface, make_subdiv,
    graph_delaunay, graph_distance, call_igraph_method
)
from .drawing import draw_point, draw_line, show_graph, save_graph


class Graphing:
    """Main graphing class for creating and visualizing graphs"""

    def __init__(self, dimension: Union[Tuple[int, int], List[int]] = None,
                 data_shape: List[Tuple[str, type]] = None, aspect: str = "array",
                 config: Optional[GraphizyConfig] = None, **kwargs):
        """Initialize Graphing object

        Args:
            dimension: Image dimensions (width, height)
            data_shape: Data structure definition
            aspect: Data format ("array" or "dict")
            config: Configuration object
            **kwargs: Additional configuration overrides

        Raises:
            InvalidDimensionError: If dimensions are invalid
            InvalidAspectError: If aspect is invalid
        """
        try:
            # Initialize configuration
            if config is None:
                config = GraphizyConfig()

            # Update with provided parameters
            if dimension is not None:
                config.graph.dimension = tuple(dimension)
            if data_shape is not None:
                config.graph.data_shape = data_shape
            if aspect != "array":
                config.graph.aspect = aspect

            # Update with any additional kwargs
            if kwargs:
                config.update(**kwargs)

            self.config = config

            # Validate parameters
            if not isinstance(self.config.graph.dimension, (tuple, list)) or len(self.config.graph.dimension) != 2:
                raise InvalidDimensionError("Dimension must be a tuple/list of 2 integers")
            if self.config.graph.dimension[0] <= 0 or self.config.graph.dimension[1] <= 0:
                raise InvalidDimensionError("Dimension values must be positive")

            self.dimension = self.config.graph.dimension

            # Validate aspect
            list_aspect = ["dict", "array"]
            if self.config.graph.aspect not in list_aspect:
                raise InvalidAspectError(
                    f"Wrong aspect specified. Must be one of {list_aspect}, got: {self.config.graph.aspect}")
            self.aspect = self.config.graph.aspect

            # Initialize data interface
            self.dinter = DataInterface(self.config.graph.data_shape)

            # Store drawing parameters from config
            self.line_thickness = self.config.drawing.line_thickness
            self.line_color = self.config.drawing.line_color
            self.point_thickness = self.config.drawing.point_thickness
            self.point_radius = self.config.drawing.point_radius
            self.point_color = self.config.drawing.point_color

            logging.info("Graphing object initialized successfully")

        except Exception as e:
            raise GraphCreationError(f"Failed to initialize Graphing object: {str(e)}")

    @property
    def drawing_config(self) -> DrawingConfig:
        """Get current drawing configuration"""
        return self.config.drawing

    @property
    def graph_config(self) -> GraphConfig:
        """Get current graph configuration"""
        return self.config.graph

    def update_config(self, **kwargs) -> None:
        """Update configuration at runtime

        Args:
            **kwargs: Configuration parameters to update
        """
        try:
            self.config.update(**kwargs)

            # Update instance variables if they changed
            if 'drawing' in kwargs:
                self.line_thickness = self.config.drawing.line_thickness
                self.line_color = self.config.drawing.line_color
                self.point_thickness = self.config.drawing.point_thickness
                self.point_radius = self.config.drawing.point_radius
                self.point_color = self.config.drawing.point_color

            if 'graph' in kwargs:
                self.dimension = self.config.graph.dimension
                self.aspect = self.config.graph.aspect
                if 'data_shape' in kwargs.get('graph', {}):
                    self.dinter = DataInterface(self.config.graph.data_shape)

            logging.info("Configuration updated successfully")

        except Exception as e:
            raise GraphCreationError(f"Failed to update configuration: {str(e)}")

    @staticmethod
    def identify_graph(graph: Any) -> Any:
        """Replace graph name with proper particle ids

        Args:
            graph: igraph Graph object

        Returns:
            Modified graph
        """
        try:
            if graph is None:
                raise GraphCreationError("Graph cannot be None")
            graph.vs["name"] = graph.vs["id"]
            return graph
        except Exception as e:
            raise GraphCreationError(f"Failed to identify graph: {str(e)}")

    def make_delaunay(self, data_points: Union[np.ndarray, Dict[str, Any]]) -> Any:
        """Make a delaunay graph

        Args:
            data_points: Point data as array or dictionary

        Returns:
            igraph Graph object with Delaunay triangulation

        Raises:
            GraphCreationError: If Delaunay graph creation fails
        """
        try:
            timer0 = time.time()

            # Create and populate the graph with points
            if self.aspect == "array":
                if not isinstance(data_points, np.ndarray):
                    raise GraphCreationError("Expected numpy array for 'array' aspect")

                graph = create_graph_array(data_points)

                # Make triangulation with appropriate columns
                pos_array = np.stack((
                    data_points[:, self.dinter.getidx_xpos()],
                    data_points[:, self.dinter.getidx_ypos()]
                ), axis=1)
                subdiv = make_subdiv(pos_array, self.dimension)
                tri_list = subdiv.getTriangleList()

            elif self.aspect == "dict":
                if isinstance(data_points, dict):
                    graph = create_graph_dict(data_points)
                    pos_array = np.stack((data_points["x"], data_points["y"]), axis=1)
                elif isinstance(data_points, np.ndarray):
                    data_points = self.dinter.convert(data_points)
                    graph = create_graph_dict(data_points)
                    pos_array = np.stack((data_points["x"], data_points["y"]), axis=1)
                else:
                    raise GraphCreationError("Invalid data format for 'dict' aspect")

                subdiv = make_subdiv(pos_array, self.dimension)
                tri_list = subdiv.getTriangleList()
            else:
                raise GraphCreationError("Graph data interface could not be understood")

            logging.debug(f"Creation and Triangulation took {round((time.time() - timer0) * 1000, 3)}ms")

            timer1 = time.time()
            # Populate edges
            graph = graph_delaunay(graph, subdiv, tri_list)
            logging.debug(f"Conversion took {round((time.time() - timer1) * 1000, 3)}ms")

            return graph

        except Exception as e:
            raise GraphCreationError(f"Failed to create Delaunay graph: {str(e)}")

    def make_proximity(self, data_points: np.ndarray, proximity_thresh: float = None,
                       metric: str = None) -> Any:
        """Make a proximity graph

        Args:
            data_points: Point data as array
            proximity_thresh: Distance threshold for connections
            metric: Distance metric to use

        Returns:
            igraph Graph object with proximity connections

        Raises:
            GraphCreationError: If proximity graph creation fails
        """
        try:
            # Use config defaults if not provided
            if proximity_thresh is None:
                proximity_thresh = self.config.graph.proximity_threshold
            if metric is None:
                metric = self.config.graph.distance_metric

            timer_prox = timeit.default_timer()

            graph = create_graph_array(data_points)
            pos_array = np.stack((
                data_points[:, self.dinter.getidx_xpos()],
                data_points[:, self.dinter.getidx_ypos()]
            ), axis=1)

            graph = graph_distance(graph, pos_array, proximity_thresh, metric=metric)

            end_prox = timeit.default_timer()
            logging.debug(f"Distance calculation took {round((end_prox - timer_prox) * 1000, 3)}ms")

            return graph

        except Exception as e:
            raise GraphCreationError(f"Failed to create proximity graph: {str(e)}")

    def draw_graph(self, graph: Any, radius: int = None, thickness: int = None) -> np.ndarray:
        """Draw the graph from igraph

        Args:
            graph: igraph Graph object
            radius: Point radius (uses config default if None)
            thickness: Point thickness (uses config default if None)

        Returns:
            Image array

        Raises:
            DrawingError: If drawing fails
        """
        try:
            if graph is None:
                raise DrawingError("Graph cannot be None")

            # Use config defaults if not provided
            if radius is None:
                radius = self.point_radius
            if thickness is None:
                thickness = self.point_thickness

            # self.dimension is (width, height), so we need to swap for NumPy
            width, height = self.dimension
            image_graph = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw points
            for point in graph.vs:
                draw_point(image_graph, (point["x"], point["y"]), self.point_color,
                           thickness=thickness, radius=radius)

            # Draw edges
            for edge in graph.es:
                x0, y0 = int(graph.vs["x"][edge.tuple[0]]), int(graph.vs["y"][edge.tuple[0]])
                x1, y1 = int(graph.vs["x"][edge.tuple[1]]), int(graph.vs["y"][edge.tuple[1]])
                draw_line(image_graph, x0, y0, x1, y1, self.line_color, thickness=self.line_thickness)

            return image_graph

        except Exception as e:
            raise DrawingError(f"Failed to draw graph: {str(e)}")

    def overlay_graph(self, image_graph: np.ndarray, graph: Any) -> np.ndarray:
        """Overlay graph on existing image

        Args:
            image_graph: Existing image
            graph: igraph Graph object

        Returns:
            Modified image array

        Raises:
            DrawingError: If overlay fails
        """
        try:
            if image_graph is None:
                raise DrawingError("Image cannot be None")
            if graph is None:
                raise DrawingError("Graph cannot be None")

            # Draw points
            for point in graph.vs:
                draw_point(image_graph, (point["x"], point["y"]), self.point_color,
                           thickness=self.point_thickness, radius=self.point_radius)

            # Draw edges
            for edge in graph.es:
                x0, y0 = int(graph.vs["x"][edge.tuple[0]]), int(graph.vs["y"][edge.tuple[0]])
                x1, y1 = int(graph.vs["x"][edge.tuple[1]]), int(graph.vs["y"][edge.tuple[1]])
                draw_line(image_graph, x0, y0, x1, y1, self.line_color, self.line_thickness)

            return image_graph

        except Exception as e:
            raise DrawingError(f"Failed to overlay graph: {str(e)}")

    def overlay_collision(self, image_graph: np.ndarray, graph: Any) -> np.ndarray:
        """Overlay collision points on graph

        Args:
            image_graph: Existing image
            graph: igraph Graph object

        Returns:
            Modified image array

        Raises:
            DrawingError: If overlay fails
        """
        try:
            if image_graph is None:
                raise DrawingError("Image cannot be None")
            if graph is None:
                raise DrawingError("Graph cannot be None")

            for edge in graph.es:
                x0, y0 = int(graph.vs["x"][edge.tuple[0]]), int(graph.vs["y"][edge.tuple[0]])
                x1, y1 = int(graph.vs["x"][edge.tuple[1]]), int(graph.vs["y"][edge.tuple[1]])

                # Draw edge
                draw_line(image_graph, x0, y0, x1, y1, self.line_color, self.line_thickness)

                # Draw midpoint
                mid_x = int((x0 + x1) / 2)
                mid_y = int((y0 + y1) / 2)
                draw_point(image_graph, (mid_x, mid_y), self.point_color, radius=25, thickness=6)

            return image_graph

        except Exception as e:
            raise DrawingError(f"Failed to overlay collision points: {str(e)}")

    @staticmethod
    def show_graph(image_graph: np.ndarray, title: str = "My beautiful graph") -> None:
        """Display graph image

        Args:
            image_graph: Image to display
            title: Window title
        """
        show_graph(image_graph, title)

    @staticmethod
    def save_graph(image_graph: np.ndarray, filename: str) -> None:
        """Save graph image to file

        Args:
            image_graph: Image to save
            filename: Output filename
        """
        save_graph(image_graph, filename)

    # Graph analysis methods (keeping all original methods)
    @staticmethod
    def average_path_length(graph: Any) -> float:
        """Calculate average path length of graph

        Args:
            graph: igraph Graph object

        Returns:
            Average path length
        """
        try:
            return call_igraph_method(graph, "average_path_length")
        except Exception as e:
            raise IgraphMethodError(f"Failed to calculate average path length: {str(e)}")

    @staticmethod
    def density(graph: Any) -> float:
        """Calculate density of graph

        Args:
            graph: igraph Graph object

        Returns:
            Graph density
        """
        try:
            return call_igraph_method(graph, "density")
        except Exception as e:
            raise IgraphMethodError(f"Failed to calculate density: {str(e)}")

    @staticmethod
    def call_method(graph: Any, method_name: str, *args, **kwargs) -> Any:
        """Call any igraph method on the graph

        Args:
            graph: igraph Graph object
            method_name: Name of the method to call
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method

        Returns:
            Result of the method call

        Raises:
            IgraphMethodError: If method call fails
        """
        return call_igraph_method(graph, method_name, *args, **kwargs)

    def get_graph_info(self, graph: Any) -> Dict[str, Any]:
        """Get comprehensive information about the graph

        Args:
            graph: igraph Graph object

        Returns:
            Dictionary with graph statistics
        """
        try:
            info = {}

            # Basic properties
            info['vertex_count'] = self.call_method(graph, 'vcount')
            info['edge_count'] = self.call_method(graph, 'ecount')
            info['density'] = self.density(graph)
            info['is_connected'] = self.call_method(graph, 'is_connected')

            # Advanced properties (if graph is not empty)
            if info['vertex_count'] > 0:
                if info['edge_count'] > 0:
                    try:
                        info['average_path_length'] = self.average_path_length(graph)
                    except:
                        info['average_path_length'] = None

                    try:
                        info['diameter'] = self.call_method(graph, 'diameter')
                    except:
                        info['diameter'] = None

                    try:
                        info['transitivity'] = self.call_method(graph, 'transitivity_undirected')
                    except:
                        info['transitivity'] = None
                else:
                    info['average_path_length'] = None
                    info['diameter'] = None
                    info['transitivity'] = None

            return info

        except Exception as e:
            raise IgraphMethodError(f"Failed to get graph info: {str(e)}")