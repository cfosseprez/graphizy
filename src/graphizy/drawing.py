"""
Drawing utilities for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

import logging
import numpy as np
from typing import Tuple, Union, Any

from .exceptions import DrawingError, DependencyError

try:
    import cv2
except ImportError:
    raise DependencyError("OpenCV is required but not installed. Install with: pip install opencv-python")


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


def show_graph(image_graph: np.ndarray, title: str = "My beautiful graph") -> None:
    """Display graph image using OpenCV

    Args:
        image_graph: Image array to display
        title: Window title

    Raises:
        DrawingError: If display operation fails
    """
    try:
        if image_graph is None:
            raise DrawingError("Image cannot be None")
        if image_graph.size == 0:
            raise DrawingError("Image cannot be empty")

        cv2.imshow(title, image_graph)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        raise DrawingError(f"Failed to display graph: {str(e)}")


def save_graph(image_graph: np.ndarray, filename: str) -> None:
    """Save graph image to file

    Args:
        image_graph: Image array to save
        filename: Output filename

    Raises:
        DrawingError: If save operation fails
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