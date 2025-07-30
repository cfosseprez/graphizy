"""
Drawing utilities for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import logging
import numpy as np
from typing import Tuple, Union, Any

from .exceptions import DrawingError, DependencyError
from .config import DrawingConfig

try:
    import cv2
except ImportError:
    raise DependencyError("OpenCV is required but not installed. Install with: pip install opencv-python")



class Visualizer:
    """
    Handles all visualization tasks for Graphizy.

    This class is responsible for drawing graphs, overlaying information,
    and managing the display or saving of the resulting images. It separates
    the visualization logic from the graph creation and analysis logic.
    """

    def __init__(self, config: DrawingConfig, dimension: Tuple[int, int]):
        """
        Initialize the Visualizer.

        Args:
            config: A DrawingConfig object with styling parameters.
            dimension: A tuple (width, height) for the canvas.
        """
        self.config = config
        self.dimension = dimension
        self.memory_manager=None # Injected by parent method

    def update_config(self, config: DrawingConfig, dimension: Tuple[int, int]):
        """Update the visualizer's configuration at runtime."""
        self.config = config
        self.dimension = dimension

    def draw_graph(self, graph: Any, radius: int = None, thickness: int = None) -> np.ndarray:
        """
        Draw a graph to an image array with customizable appearance.

        This is the primary method for converting igraph Graph objects into
        visual representations. It handles both vertices (as circles) and
        edges (as lines) with configurable styling.

        Args:
            graph: igraph Graph object with vertices having "x", "y" coordinates
            radius: Point radius override. If None, uses self.point_radius from config.
            thickness: Point border thickness override. If None, uses self.point_thickness.
            direct_show: If True, immediately display the graph using show_graph().
                        Convenient for interactive use.
            kwargs_show: Additional parameters passed to show_graph() if direct_show=True.
                        E.g., {'title': 'My Graph', 'block': False}

        Returns:
            np.ndarray: RGB image array with shape (height, width, 3) and dtype uint8.
                       Background is black (0,0,0), drawn elements use configured colors.

        Raises:
            DrawingError: If graph is None, missing coordinates, or drawing operations fail.

        Examples:
            >>> # Basic drawing
            >>> graph = grapher.make_delaunay(data)
            >>> image = grapher.draw_graph(graph)
            >>> grapher.show_graph(image)

            >>> # Custom appearance
            >>> image = grapher.draw_graph(graph, radius=10, thickness=3)

            >>> # Draw and show immediately
            >>> image = grapher.draw_graph(
            ...     graph,
            ...     direct_show=True,
            ...     kwargs_show={'title': 'Delaunay Triangulation', 'block': True}
            ... )

            >>> # Multiple graphs on same image
            >>> image = grapher.draw_graph(delaunay_graph)
            >>> image = grapher.overlay_graph(image, mst_graph)  # Overlay MST
            >>> grapher.show_graph(image, title="Combined Graph")

        Note:
            - Graph must have vertex attributes "x" and "y" for coordinates
            - Coordinates are in image pixel space (0 to dimension)
            - Drawing order: edges first, then vertices (vertices on top)
            - Image dimensions set by self.dimension from config
            - Colors set by self.point_color and self.line_color from config
        """
        try:
            if graph is None:
                raise DrawingError("Graph cannot be None")

            # Use config defaults if parameters not provided
            radius = radius if radius is not None else self.config.point_radius
            thickness = thickness if thickness is not None else self.config.point_thickness

            # Create image array
            width, height = self.dimension
            image_graph = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw edges first
            for edge in graph.es:
                source_idx, target_idx = edge.tuple
                x0, y0 = int(graph.vs[source_idx]["x"]), int(graph.vs[source_idx]["y"])
                x1, y1 = int(graph.vs[target_idx]["x"]), int(graph.vs[target_idx]["y"])
                draw_line(image_graph, x0, y0, x1, y1, self.config.line_color, self.config.line_thickness)

            # Draw vertices on top
            for vertex in graph.vs:
                point_coords = (vertex["x"], vertex["y"])
                draw_point(image_graph, point_coords, self.config.point_color,
                           thickness=thickness, radius=radius)

            return image_graph
        except Exception as e:
            raise DrawingError(f"Failed to draw graph: {str(e)}")

    def draw_memory_graph(self, graph: Any, radius: int = None, thickness: int = None,
                    use_age_colors: bool = True, alpha_range: Tuple[float, float] = (0.3, 1.0)) -> np.ndarray:
        """
        Draw memory graph with optional age-based edge coloring and transparency.

        This specialized drawing method can visualize temporal information by
        varying edge appearance based on connection age/persistence in memory.
        Newer or more frequent connections can be highlighted while older
        connections are drawn more subtly.

        Args:
            graph: igraph Graph object to draw
            radius: Point radius override. If None, uses config default.
            thickness: Point thickness override. If None, uses config default.
            use_age_colors: Whether to apply age-based styling to edges.
                          If True, requires memory_manager with age tracking enabled.
            alpha_range: (min_alpha, max_alpha) tuple for transparency range.
                        Older connections use min_alpha, newer use max_alpha.

        Returns:
            np.ndarray: Image array representing the drawn graph.

        Raises:
            DrawingError: If drawing fails or memory manager required but not available.

        Examples:
            >>> # Basic memory graph drawing
            >>> memory_graph = grapher.make_memory_graph(data)
            >>> image = grapher.draw_memory_graph(memory_graph)
            >>> grapher.show_graph(image, title="Memory Graph")

            >>> # Custom age-based visualization
            >>> image = grapher.draw_memory_graph(
            ...     memory_graph,
            ...     use_age_colors=True,
            ...     alpha_range=(0.1, 1.0),  # Very faded old connections
            ...     radius=8,
            ...     thickness=2
            ... )

            >>> # Disable age coloring for standard appearance
            >>> image = grapher.draw_memory_graph(
            ...     memory_graph,
            ...     use_age_colors=False
            ... )

        Note:
            - Age-based coloring requires memory manager with track_edge_ages=True
            - Alpha blending may not be supported in all drawing backends
            - Performance may be slower with age-based coloring for large graphs
            - Falls back to standard drawing if age information unavailable
        """
        try:
            if graph is None:
                raise DrawingError("Graph cannot be None")
            if self.memory_manager is None:
                raise DrawingError("The memory was not initialized")

            width, height = self.dimension
            image = np.zeros((height, width, 3), dtype=np.uint8)

            draw_memory_graph_with_aging(
                image, graph, self.memory_manager,
                point_color=self.config.point_color,
                line_color=self.config.line_color,
                point_radius=radius if radius is not None else self.config.point_radius,
                point_thickness=thickness if thickness is not None else self.config.point_thickness,
                line_thickness=self.config.line_thickness,
                use_age_colors=use_age_colors,
                alpha_range=alpha_range
            )
            return image
        except Exception as e:
            raise DrawingError(f"Failed to draw memory graph: {str(e)}")

    def overlay_graph(self, image_graph: np.ndarray, graph: Any) -> np.ndarray:
        """
        Overlay additional graph elements onto an existing image.

        This method allows combining multiple graphs in a single visualization
        by drawing additional vertices and edges on top of an existing image.
        Useful for comparing different graph types or showing graph evolution.

        Args:
            image_graph: Existing image array to draw on. Modified in-place.
            graph: igraph Graph object to overlay with vertices having "x", "y" coordinates.

        Returns:
            np.ndarray: The modified image array (same object as input for chaining).

        Raises:
            DrawingError: If either image or graph is None, or if drawing operations fail.

        Examples:
            >>> # Compare Delaunay triangulation with MST
            >>> delaunay_graph = grapher.make_delaunay(data)
            >>> mst_graph = grapher.make_mst(data)
            >>>
            >>> # Draw Delaunay as base
            >>> image = grapher.draw_graph(delaunay_graph)
            >>>
            >>> # Overlay MST with different color
            >>> grapher.update_config(line_color=(255, 0, 0))  # Red for MST
            >>> image = grapher.overlay_graph(image, mst_graph)
            >>> grapher.show_graph(image, title="Delaunay + MST")

            >>> # Chain multiple overlays
            >>> image = grapher.draw_graph(delaunay_graph)
            >>> image = grapher.overlay_graph(image, mst_graph)
            >>> image = grapher.overlay_graph(image, knn_graph)

        Note:
            - Image is modified in-place and also returned for method chaining
            - Overlay uses current drawing configuration (colors, thickness, etc.)
            - Later overlays draw on top of earlier ones
            - Vertices are drawn on top of edges within each overlay
            - Consider using different colors for different overlays
        """
        try:
            if image_graph is None or graph is None:
                raise DrawingError("Image and graph cannot be None")

            # Draw edges
            for edge in graph.es:
                source_idx, target_idx = edge.tuple
                x0, y0 = int(graph.vs[source_idx]["x"]), int(graph.vs[source_idx]["y"])
                x1, y1 = int(graph.vs[target_idx]["x"]), int(graph.vs[target_idx]["y"])
                draw_line(image_graph, x0, y0, x1, y1, self.config.line_color, self.config.line_thickness)

            # Draw vertices
            for vertex in graph.vs:
                point_coords = (vertex["x"], vertex["y"])
                draw_point(image_graph, point_coords, self.config.point_color,
                           thickness=self.config.point_thickness, radius=self.config.point_radius)
            return image_graph
        except Exception as e:
            raise DrawingError(f"Failed to overlay graph: {str(e)}")



    def overlay_collision(self, image_graph: np.ndarray, graph: Any) -> np.ndarray:
        """
        Overlay collision/intersection points on graph edges.

        This debugging/analysis method draws midpoints of all edges with
        prominent markers. Useful for visualizing edge density, detecting
        potential intersections, or highlighting edge midpoints for analysis.

        Args:
            image_graph: Existing image array to draw on. Modified in-place.
            graph: igraph Graph object with edges to mark.

        Returns:
            np.ndarray: The modified image array with collision points added.

        Raises:
            DrawingError: If either image or graph is None, or if drawing operations fail.

        Examples:
            >>> # Visualize edge midpoints for analysis
            >>> graph = grapher.make_delaunay(data)
            >>> image = grapher.draw_graph(graph)
            >>> image = grapher.overlay_collision(image, graph)
            >>> grapher.show_graph(image, title="Graph with Edge Midpoints")

            >>> # Analyze edge density in different regions
            >>> dense_graph = grapher.make_proximity(data, proximity_thresh=50)
            >>> image = grapher.draw_graph(dense_graph)
            >>> image = grapher.overlay_collision(image, dense_graph)

        Note:
            - Collision points are drawn as large, prominent circles
            - Useful for debugging edge placement and density analysis
            - Midpoint calculation uses integer arithmetic (may have rounding)
            - Collision markers use current point_color configuration
        """
        try:
            if image_graph is None:
                raise DrawingError("Image cannot be None")
            if graph is None:
                raise DrawingError("Graph cannot be None")

            for edge in graph.es:
                source_idx, target_idx = edge.tuple
                x0, y0 = int(graph.vs[source_idx]["x"]), int(graph.vs[source_idx]["y"])
                x1, y1 = int(graph.vs[target_idx]["x"]), int(graph.vs[target_idx]["y"])

                # Draw the edge
                draw_line(image_graph, x0, y0, x1, y1, self.config.line_color, self.config.line_thickness)

                # Draw prominent midpoint marker
                mid_x = int((x0 + x1) / 2)
                mid_y = int((y0 + y1) / 2)
                draw_point(image_graph, (mid_x, mid_y), self.config.point_color, radius=25, thickness=6)

            return image_graph

        except Exception as e:
            raise DrawingError(f"Failed to overlay collision points: {str(e)}")

    def show_graph(self, image_graph: np.ndarray, title: str = "Graphizy", **kwargs) -> None:
        """Display a graph image in a window."""
        show_graph(image_graph, title, **kwargs)

    def save_graph(self, image_graph: np.ndarray, filename: str) -> None:
        """Save a graph image to a file."""
        save_graph(image_graph, filename)


def draw_point(img: np.ndarray, p: Tuple[float, float], color: Tuple[int, int, int],
               radius: int = 4, thickness: int = 1) -> None:
    """Draw a point on the image with enhanced error handling

    Args:
        img: Image array to draw on
        p: Point coordinates (x, y)
        color: Color tuple (B, G, R)
        radius: Point radius
        thickness: Line thickness

    Raises:
        DrawingError: If drawing operation fails
    """
    logger = logging.getLogger('graphizy.drawing.draw_point')

    try:
        # Input validation
        if img is None:
            raise DrawingError("Image cannot be None", img, p)
        if len(p) != 2:
            raise DrawingError("Point must have exactly 2 coordinates", img, p)
        if len(color) != 3:
            raise DrawingError("Color must be a tuple of 3 values", img, p)
        if radius < 1:
            raise DrawingError("Radius must be >= 1", img, p)
        if thickness < 1:
            raise DrawingError("Thickness must be >= 1", img, p)

        x, y = int(p[0]), int(p[1])

        # Enhanced bounds checking - log warning but don't crash
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            from .exceptions import log_warning_with_context
            log_warning_with_context(
                f"Point ({x}, {y}) is outside image bounds {img.shape}",
                point_coordinates=(x, y),
                image_shape=img.shape,
                image_bounds=f"[0, {img.shape[1]}) x [0, {img.shape[0]})"
            )
            # Return early instead of attempting to draw
            return

        # Draw the point
        cv2.circle(img, (x, y), radius, color, thickness)
        cv2.drawMarker(img, (x, y), color, markerType=cv2.MARKER_CROSS,
                       markerSize=radius, thickness=1, line_type=cv2.LINE_8)

        logger.debug(f"Successfully drew point at ({x}, {y})")

    except DrawingError:
        # Re-raise DrawingError as-is
        raise
    except Exception as e:
        # Convert other exceptions to DrawingError
        error = DrawingError(f"Failed to draw point: {str(e)}", img, p, original_exception=e)
        error.log_error()
        raise error


def draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int,
              color: Tuple[int, int, int], thickness: int = 1) -> None:
    """Draw a line on the image with enhanced error handling

    Args:
        img: Image array to draw on
        x0, y0: Start point coordinates
        x1, y1: End point coordinates
        color: Color tuple (B, G, R)
        thickness: Line thickness

    Raises:
        DrawingError: If drawing operation fails
    """
    logger = logging.getLogger('graphizy.drawing.draw_line')

    try:
        # Input validation
        if img is None:
            raise DrawingError("Image cannot be None", img, (x0, y0, x1, y1))
        if len(color) != 3:
            raise DrawingError("Color must be a tuple of 3 values", img, (x0, y0, x1, y1))
        if thickness < 1:
            raise DrawingError("Thickness must be >= 1", img, (x0, y0, x1, y1))

        # Convert to integers
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # Check if line endpoints are within reasonable bounds (allow some overflow for partial lines)
        max_coord = max(img.shape[0], img.shape[1]) * 2
        if any(abs(coord) > max_coord for coord in [x0, y0, x1, y1]):
            from .exceptions import log_warning_with_context
            log_warning_with_context(
                f"Line coordinates ({x0}, {y0}) to ({x1}, {y1}) are extremely large",
                line_coords=(x0, y0, x1, y1),
                image_shape=img.shape
            )

        # Draw the line
        cv2.line(img, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA, 0)

        logger.debug(f"Successfully drew line from ({x0}, {y0}) to ({x1}, {y1})")

    except DrawingError:
        # Re-raise DrawingError as-is
        raise
    except Exception as e:
        # Convert other exceptions to DrawingError
        error = DrawingError(f"Failed to draw line: {str(e)}", img, (x0, y0, x1, y1), original_exception=e)
        error.log_error()
        raise error


def draw_delaunay(img: np.ndarray, subdiv: Any, color_line: Tuple[int, int, int] = (0, 255, 0),
                  thickness_line: int = 1, color_point: Tuple[int, int, int] = (0, 0, 255),
                  thickness_point: int = 1) -> None:
    """Draw delaunay triangles from openCV Subdiv2D

    Args:
        img: Image to draw on
        subdiv: OpenCV Subdiv2D object
        color_line: Line color (B, G, R)
        thickness_line: Line thickness
        color_point: Point color (B, G, R)
        thickness_point: Point thickness

    Raises:
        DrawingError: If drawing operation fails
    """
    try:
        if img is None:
            raise DrawingError("Image cannot be None")
        if subdiv is None:
            raise DrawingError("Subdivision cannot be None")

        triangle_list = subdiv.getTriangleList()

        if len(triangle_list) == 0:
            logging.warning("No triangles found in subdivision")
            return

        for t in triangle_list:
            if len(t) != 6:
                logging.warning(f"Invalid triangle format: expected 6 values, got {len(t)}")
                continue

            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))

            # Draw points
            draw_point(img, pt1, color_point, thickness=thickness_point)
            draw_point(img, pt2, color_point, thickness=thickness_point)
            draw_point(img, pt3, color_point, thickness=thickness_point)

            # Draw lines
            draw_line(img, *pt1, *pt2, color_line, thickness_line)
            draw_line(img, *pt2, *pt3, color_line, thickness_line)
            draw_line(img, *pt1, *pt3, color_line, thickness_line)

    except Exception as e:
        raise DrawingError(f"Failed to draw Delaunay triangulation: {str(e)}")


def show_graph(
    image_graph: np.ndarray,
    title: str = "My beautiful graph",
    block: bool = False,
    delay_display: int = 100
) -> None:
    """
    Display a graph image using the configured display backend.

    This is a convenience wrapper around the global show_graph function
    that provides consistent image display with customizable options.

    Args:
        image_graph: Image array to display with shape (height, width, 3).
        title: Window title for the display.
        **kwargs: Additional arguments passed to the underlying show_graph function:
                 - block: Whether to block execution until window is closed
                 - delay_display: Delay before showing (for animations)
                 - save_path: Optionally save image while displaying
                 - Backend-specific options

    Examples:
        >>> image = grapher.draw_graph(graph)
        >>> Graphing.show_graph(image, title="Delaunay Triangulation")

        >>> # Non-blocking display for animations
        >>> Graphing.show_graph(image, title="Animation Frame", block=False)

        >>> # Display with save
        >>> Graphing.show_graph(
        ...     image,
        ...     title="Final Result",
        ...     save_path="output.png"
        ... )

    Note:
        - Display backend depends on system configuration (matplotlib, opencv, etc.)
        - Window behavior (blocking, resizing) depends on backend
        - Static method can be called without Graphing instance
    """
    try:
        if image_graph is None or not isinstance(image_graph, np.ndarray):
            raise DrawingError("Provided image must be a valid numpy array.")
        if image_graph.size == 0:
            raise DrawingError("Provided image is empty.")

        cv2.imshow(title, image_graph)
        cv2.waitKey(0 if block else delay_display)
        try:
            cv2.destroyWindow(title)
        except Exception:
            pass

    except Exception as e:
        raise DrawingError(f"Failed to display graph: {e}")

def save_graph(image_graph: np.ndarray, filename: str) -> None:
    """
    Save graph image to file.

    This is a convenience wrapper around the global save_graph function
    that handles various image formats and provides error handling.

    Args:
        image_graph: Image array to save with shape (height, width, 3).
        filename: Output filename with extension. Extension determines format
                 (e.g., .png, .jpg, .tiff, .bmp).

    Examples:
        >>> image = grapher.draw_graph(graph)
        >>> Graphing.save_graph(image, "delaunay_triangulation.png")

        >>> # Save with timestamp
        >>> import datetime
        >>> timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        >>> filename = f"graph_{timestamp}.png"
        >>> Graphing.save_graph(image, filename)

    Note:
        - File format determined by extension
        - Overwrites existing files without warning
        - Directory must exist (not created automatically)
        - Static method can be called without Graphing instance
    """
    try:
        if image_graph is None:
            raise DrawingError("Image cannot be None")
        if image_graph.size == 0:
            raise DrawingError("Image cannot be empty")
        if not filename:
            raise DrawingError("Filename cannot be empty")

        success = cv2.imwrite(filename, image_graph)
        if not success:
            raise DrawingError(f"Failed to save image to {filename}")

        logging.info(f"Graph saved to {filename}")

    except Exception as e:
        raise DrawingError(f"Failed to save graph: {str(e)}")


def draw_memory_graph_with_aging(img: np.ndarray, graph: Any, memory_manager: Any,
                                 point_color: Tuple[int, int, int], line_color: Tuple[int, int, int],
                                 point_radius: int = 8, point_thickness: int = 3,
                                 line_thickness: int = 1, use_age_colors: bool = True,
                                 alpha_range: Tuple[float, float] = (0.3, 1.0)) -> None:
    """Draw memory graph with optional edge aging visualization

    Args:
        img: Image array to draw on
        graph: igraph Graph object
        memory_manager: MemoryManager instance (for edge ages)
        point_color: Color for points (B, G, R)
        line_color: Base color for lines (B, G, R)
        point_radius: Point radius
        point_thickness: Point thickness
        line_thickness: Line thickness
        use_age_colors: Whether to use age-based edge coloring
        alpha_range: (min_alpha, max_alpha) for age-based transparency

    Raises:
        DrawingError: If drawing operation fails
    """
    try:
        if img is None:
            raise DrawingError("Image cannot be None")
        if graph is None:
            raise DrawingError("Graph cannot be None")

        # Draw points first
        for point in graph.vs:
            draw_point(img, (point["x"], point["y"]), point_color,
                       radius=point_radius, thickness=point_thickness)

        # Draw edges with optional age-based styling
        if (use_age_colors and memory_manager and
                hasattr(memory_manager, 'track_edge_ages') and memory_manager.track_edge_ages and
                hasattr(memory_manager, 'get_edge_age_normalized')):

            edge_ages = memory_manager.get_edge_age_normalized()

            for edge in graph.es:
                x0, y0 = int(graph.vs["x"][edge.tuple[0]]), int(graph.vs["y"][edge.tuple[0]])
                x1, y1 = int(graph.vs["x"][edge.tuple[1]]), int(graph.vs["y"][edge.tuple[1]])

                # Get edge age for color/alpha calculation
                vertex1_id = str(graph.vs[edge.tuple[0]]["id"])
                vertex2_id = str(graph.vs[edge.tuple[1]]["id"])
                edge_key = tuple(sorted([vertex1_id, vertex2_id]))

                if edge_key in edge_ages:
                    age_normalized = edge_ages[edge_key]
                    # Older edges are more transparent (higher age = lower alpha)
                    alpha = alpha_range[1] - (age_normalized * (alpha_range[1] - alpha_range[0]))

                    # Apply alpha to color (simple blend with background)
                    aged_color = tuple(int(c * alpha) for c in line_color)
                else:
                    aged_color = line_color

                draw_line(img, x0, y0, x1, y1, aged_color, thickness=line_thickness)
        else:
            # Standard edge drawing
            for edge in graph.es:
                x0, y0 = int(graph.vs["x"][edge.tuple[0]]), int(graph.vs["y"][edge.tuple[0]])
                x1, y1 = int(graph.vs["x"][edge.tuple[1]]), int(graph.vs["y"][edge.tuple[1]])
                draw_line(img, x0, y0, x1, y1, line_color, thickness=line_thickness)

        logging.debug(f"Successfully drew memory graph with aging (use_age_colors={use_age_colors})")

    except Exception as e:
        raise DrawingError(f"Failed to draw memory graph with aging: {str(e)}")


def create_memory_graph_image(graph: Any, memory_manager: Any, dimension: Tuple[int, int],
                              point_color: Tuple[int, int, int] = (0, 0, 255),
                              line_color: Tuple[int, int, int] = (0, 255, 0),
                              point_radius: int = 8, point_thickness: int = 3,
                              line_thickness: int = 1, use_age_colors: bool = True,
                              alpha_range: Tuple[float, float] = (0.3, 1.0)) -> np.ndarray:
    """Create a complete memory graph image

    Args:
        graph: igraph Graph object
        memory_manager: MemoryManager instance
        dimension: Image dimensions (width, height)
        point_color: Color for points (B, G, R)
        line_color: Base color for lines (B, G, R)
        point_radius: Point radius
        point_thickness: Point thickness
        line_thickness: Line thickness
        use_age_colors: Whether to use age-based edge coloring
        alpha_range: (min_alpha, max_alpha) for age-based transparency

    Returns:
        Image array

    Raises:
        DrawingError: If image creation fails
    """
    try:
        if graph is None:
            raise DrawingError("Graph cannot be None")
        if len(dimension) != 2:
            raise DrawingError("Dimension must be a tuple of (width, height)")

        width, height = dimension
        # Create image (height, width, 3) for OpenCV
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw the memory graph
        draw_memory_graph_with_aging(
            image, graph, memory_manager, point_color, line_color,
            point_radius, point_thickness, line_thickness,
            use_age_colors, alpha_range
        )

        return image

    except Exception as e:
        raise DrawingError(f"Failed to create memory graph image: {str(e)}")


