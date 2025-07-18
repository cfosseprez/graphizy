"""
Custom exceptions for graphizy package with enhanced error tracking

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import logging
import traceback
import sys
from typing import Any, Dict, Optional
from datetime import datetime


class GraphizyError(Exception):
    """Base exception for all graphizy errors with enhanced debugging"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None,
                 original_exception: Optional[Exception] = None):
        self.message = message
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()

        # Auto-capture frame info
        frame = sys._getframe(1)
        self.function_name = frame.f_code.co_name
        self.module_name = frame.f_globals.get('__name__', 'unknown')
        self.line_number = frame.f_lineno

        # Capture safe local variables
        self._capture_safe_locals(frame.f_locals)

        super().__init__(self._format_message())

    def _capture_safe_locals(self, locals_dict: Dict[str, Any]) -> None:
        """Safely capture local variables for debugging"""
        safe_locals = {}
        for name, value in locals_dict.items():
            if name.startswith('_'):
                continue
            try:
                if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                    safe_locals[name] = f"array(shape={value.shape}, dtype={value.dtype})"
                elif isinstance(value, (int, float, str, bool)) and len(str(value)) < 100:
                    safe_locals[name] = str(value)
                elif isinstance(value, (list, tuple)) and len(value) < 5:
                    safe_locals[name] = str(value)[:100]
                else:
                    safe_locals[name] = f"{type(value).__name__}(...)"
            except:
                safe_locals[name] = "<unprintable>"

        self.safe_locals = safe_locals

    def _format_message(self) -> str:
        """Format comprehensive error message"""
        msg_parts = [f"{self.message}"]

        if hasattr(self, 'function_name'):
            msg_parts.append(f"  Location: {self.module_name}.{self.function_name}:{self.line_number}")

        if self.context:
            msg_parts.append("  Context:")
            for key, value in self.context.items():
                msg_parts.append(f"    {key}: {value}")

        if hasattr(self, 'safe_locals') and self.safe_locals:
            msg_parts.append("  Local Variables:")
            for key, value in self.safe_locals.items():
                msg_parts.append(f"    {key}: {value}")

        if self.original_exception:
            msg_parts.append(f"  Caused by: {type(self.original_exception).__name__}: {self.original_exception}")

        return "\n".join(msg_parts)

    def log_error(self) -> None:
        """Log this error with full context"""
        logger = logging.getLogger(f'graphizy.{self.module_name}')
        logger.error(f"GraphizyError: {self.message}")
        logger.debug(f"Full context: {self._format_message()}")


class InvalidDimensionError(GraphizyError):
    """Raised when invalid dimensions are provided"""

    def __init__(self, message: str, dimensions: Any = None, **kwargs):
        context = {}
        if dimensions is not None:
            context['dimensions'] = str(dimensions)
        super().__init__(message, context, **kwargs)


class InvalidDataShapeError(GraphizyError):
    """Raised when invalid data shape is provided"""

    def __init__(self, message: str, data_shape: Any = None, **kwargs):
        context = {}
        if data_shape is not None:
            context['data_shape'] = str(data_shape)
        super().__init__(message, context, **kwargs)


class InvalidAspectError(GraphizyError):
    """Raised when invalid aspect is provided"""
    pass


class InvalidPointArrayError(GraphizyError):
    """Raised when invalid point array is provided"""

    def __init__(self, message: str, array_info: Dict[str, Any] = None, **kwargs):
        context = array_info or {}
        super().__init__(message, context, **kwargs)


class SubdivisionError(GraphizyError):
    """Raised when OpenCV subdivision fails"""

    def __init__(self, message: str, point_array=None, dimensions=None, **kwargs):
        context = {}

        # Add point array info
        if point_array is not None:
            try:
                context['point_array_shape'] = str(point_array.shape)
                context['point_array_dtype'] = str(point_array.dtype)
                context['x_range'] = f"[{point_array[:, 0].min():.1f}, {point_array[:, 0].max():.1f}]"
                context['y_range'] = f"[{point_array[:, 1].min():.1f}, {point_array[:, 1].max():.1f}]"
            except:
                context['point_array'] = "Could not extract array info"

        # Add dimensions info
        if dimensions is not None:
            context['dimensions'] = str(dimensions)

        super().__init__(message, context, **kwargs)


class TriangulationError(GraphizyError):
    """Raised when Delaunay triangulation fails"""
    pass


class GraphCreationError(GraphizyError):
    """Raised when graph creation fails"""
    pass


class DrawingError(GraphizyError):
    """Raised when drawing operations fail"""

    def __init__(self, message: str, image=None, point=None, **kwargs):
        context = {}

        if image is not None:
            try:
                context['image_shape'] = str(image.shape)
                context['image_dtype'] = str(image.dtype)
            except:
                context['image'] = "Could not extract image info"

        if point is not None:
            context['point'] = str(point)

        super().__init__(message, context, **kwargs)


class PositionGenerationError(GraphizyError):
    """Raised when position generation fails"""

    def __init__(self, message: str, size_x: int = None, size_y: int = None,
                 num_particles: int = None, **kwargs):
        context = {}
        if size_x is not None:
            context['size_x'] = size_x
        if size_y is not None:
            context['size_y'] = size_y
        if num_particles is not None:
            context['num_particles'] = num_particles
        super().__init__(message, context, **kwargs)


class IgraphMethodError(GraphizyError):
    """Raised when igraph method execution fails"""

    def __init__(self, message: str, method_name: str = None, graph_info: Dict[str, Any] = None, **kwargs):
        context = {}
        if method_name:
            context['method'] = method_name
        if graph_info:
            context.update(graph_info)
        super().__init__(message, context, **kwargs)


class ConfigurationError(GraphizyError):
    """Raised when configuration is invalid"""
    pass


class DependencyError(GraphizyError):
    """Raised when required dependencies are missing"""
    pass


# Utility functions for better error creation
def handle_subdivision_bounds_error(point_array, dimensions, coordinate_type='x'):
    """Create detailed subdivision bounds error"""
    width, height = dimensions

    if coordinate_type == 'x':
        bad_points = point_array[point_array[:, 0] >= width]
        message = f"Found {len(bad_points)} points with X >= {width} (width limit)"
        if len(bad_points) > 0:
            message += f". Max X: {point_array[:, 0].max():.1f}"
    else:
        bad_points = point_array[point_array[:, 1] >= height]
        message = f"Found {len(bad_points)} points with Y >= {height} (height limit)"
        if len(bad_points) > 0:
            message += f". Max Y: {point_array[:, 1].max():.1f}"

    raise SubdivisionError(message, point_array, dimensions)


def log_warning_with_context(message: str, **context):
    """Log warning with context without raising exception"""
    logger = logging.getLogger('graphizy.warnings')
    logger.warning(message)
    if context:
        for key, value in context.items():
            logger.debug(f"  {key}: {value}")