"""
Configuration module for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

from dataclasses import dataclass, field
from typing import Tuple, Union, Any, Optional, Dict
import logging

from graphizy.exceptions import InvalidDimensionError, InvalidAspectError, InvalidDataShapeError

@dataclass
class DrawingConfig:
    """Configuration for drawing parameters"""
    line_color: Tuple[int, int, int] = (0, 255, 0)
    line_thickness: int = 1
    point_color: Tuple[int, int, int] = (0, 0, 255)
    point_thickness: int = 3
    point_radius: int = 8

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not isinstance(self.line_color, (tuple, list)) or len(self.line_color) != 3:
            raise ValueError("line_color must be a tuple/list of 3 integers")
        if not isinstance(self.point_color, (tuple, list)) or len(self.point_color) != 3:
            raise ValueError("point_color must be a tuple/list of 3 integers")
        if self.line_thickness < 1:
            raise ValueError("line_thickness must be >= 1")
        if self.point_thickness < 1:
            raise ValueError("point_thickness must be >= 1")
        if self.point_radius < 1:
            raise ValueError("point_radius must be >= 1")


@dataclass
class GraphConfig:
    """Configuration for graph creation parameters"""
    dimension: Tuple[int, int] = (1200, 1200)
    data_shape: list = field(
        default_factory=lambda: [("id", int), ("x", int), ("y", int), ("speed", float), ("feedback", bool)])
    aspect: str = "array"
    proximity_threshold: float = 50.0
    distance_metric: str = "euclidean"

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not isinstance(self.dimension, (tuple, list)) or len(self.dimension) != 2:
            raise InvalidDimensionError("dimension must be a tuple/list of 2 integers", dimensions=self.dimension)
        if self.dimension[0] <= 0 or self.dimension[1] <= 0:
            raise InvalidDimensionError("dimension values must be positive", dimensions=self.dimension)
        if self.aspect not in ["array", "dict"]:
            raise InvalidAspectError(f"aspect must be 'array' or 'dict', got '{self.aspect}'")
        if self.proximity_threshold <= 0:
            raise ValueError("proximity_threshold must be positive")
        if not isinstance(self.data_shape, list):
            raise InvalidDataShapeError("data_shape must be a list of tuples")

@dataclass
class MemoryConfig:
    """Configuration for memory graph parameters"""
    max_memory_size: int = 3
    max_iterations: Optional[int] = None
    auto_update_from_proximity: bool = True
    memory_decay_factor: float = 1.0  # Future: for weighted memory decay

    def __post_init__(self):
        if self.max_memory_size <= 0:
            raise ValueError("max_memory_size must be positive")
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive or None")


@dataclass
class WeightConfig:
    """Configuration for weight computation"""
    auto_compute_weights: bool = True
    weight_method: str = "distance"  # "distance", "age", "formula", "function", "combined"
    normalize_weights: bool = True
    weight_range: Tuple[float, float] = (0.0, 1.0)
    distance_metric: str = "euclidean"
    weight_attribute: str = "weight"
    distance_attribute: str = "distance"

    # Formula-specific settings
    weight_formula: Optional[str] = None  # e.g., "1/(distance + 0.1)"

    # Advanced settings
    epsilon: float = 1e-10  # Prevent division by zero
    default_value: float = 1.0  # Fallback for failed calculations

    # Age-based settings (if using memory system)
    age_mode: str = "exponential"  # "direct", "inverse", "exponential"
    decay_rate: float = 0.1

    # Combined weight settings
    weight_factors: Dict[str, float] = field(default_factory=lambda: {"distance": 1.0})


@dataclass
class GenerationConfig:
    """Configuration for position generation"""
    size_x: int = 1200
    size_y: int = 1200
    num_particles: int = 200
    to_array: bool = True
    convert_to_float: bool = True

    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.size_x <= 0 or self.size_y <= 0:
            raise ValueError("size_x and size_y must be positive")
        if self.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if self.num_particles > self.size_x * self.size_y:
            raise ValueError("num_particles cannot exceed size_x * size_y")


@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    def __post_init__(self):
        """Validate and setup logging configuration"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")

        logging.basicConfig(
            level=getattr(logging, self.level.upper()),
            format=self.format
        )


class GraphizyConfig:
    """Master configuration class combining all config sections"""
    drawing: DrawingConfig = field(default_factory=DrawingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    weight: WeightConfig = field(default_factory=WeightConfig)

    def __init__(self, **kwargs):
        graph_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in GraphConfig.__dataclass_fields__}
        drawing_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in DrawingConfig.__dataclass_fields__}
        gen_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in GenerationConfig.__dataclass_fields__}
        logging_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in LoggingConfig.__dataclass_fields__}
        memory_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in MemoryConfig.__dataclass_fields__}
        weight_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in WeightConfig.__dataclass_fields__}

        self.graph = GraphConfig(**graph_kwargs)
        self.drawing = DrawingConfig(**drawing_kwargs)
        self.generation = GenerationConfig(**gen_kwargs)
        self.logging = LoggingConfig(**logging_kwargs)
        self.memory = MemoryConfig(**memory_kwargs)
        self.weight = WeightConfig(**weight_kwargs)

        if kwargs:
            raise ValueError(f"Unknown configuration keys: {list(kwargs.keys())}")

    def update(self, **kwargs):
        """Update configuration values at runtime"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                config_obj = getattr(self, key)
                if hasattr(config_obj, '__dataclass_fields__') and isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        if hasattr(config_obj, nested_key):
                            setattr(config_obj, nested_key, nested_value)
                        else:
                            raise ValueError(f"Unknown config key: {key}.{nested_key}")
                else:
                    setattr(self, key, value)
            # Handle flat parameters that should be routed to appropriate sub-configs
            elif hasattr(self.drawing, key):
                setattr(self.drawing, key, value)
            elif hasattr(self.graph, key):
                setattr(self.graph, key, value)
            elif hasattr(self.generation, key):
                setattr(self.generation, key, value)
            elif hasattr(self.logging, key):
                setattr(self.logging, key, value)
            elif hasattr(self.memory, key):
                setattr(self.memory, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")

    def set_drawing(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.drawing, key):
                setattr(self.drawing, key, value)
            else:
                raise ValueError(f"Invalid drawing config key: {key}")

    def set_graph(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.graph, key):
                setattr(self.graph, key, value)
            else:
                raise ValueError(f"Invalid graph config key: {key}")

    def set_generation(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.generation, key):
                setattr(self.generation, key, value)
            else:
                raise ValueError(f"Invalid generation config key: {key}")

    def set_logging(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.logging, key):
                setattr(self.logging, key, value)
            else:
                raise ValueError(f"Invalid logging config key: {key}")

    def set_memory(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.memory, key):
                setattr(self.memory, key, value)
            else:
                raise ValueError(f"Invalid memory config key: {key}")

    def copy(self) -> 'GraphizyConfig':
        """Create a deep copy of the configuration"""
        import copy as copy_module
        return copy_module.deepcopy(self)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "drawing": {
                "line_color": self.drawing.line_color,
                "line_thickness": self.drawing.line_thickness,
                "point_color": self.drawing.point_color,
                "point_thickness": self.drawing.point_thickness,
                "point_radius": self.drawing.point_radius,
            },
            "graph": {
                "dimension": self.graph.dimension,
                "data_shape": self.graph.data_shape,
                "aspect": self.graph.aspect,
                "proximity_threshold": self.graph.proximity_threshold,
                "distance_metric": self.graph.distance_metric,
            },
            "generation": {
                "size_x": self.generation.size_x,
                "size_y": self.generation.size_y,
                "num_particles": self.generation.num_particles,
                "to_array": self.generation.to_array,
                "convert_to_float": self.generation.convert_to_float,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
            },
            "memory": {
                "max_memory_size": self.memory.max_memory_size,
                "max_iterations": self.memory.max_iterations,
                "auto_update_from_proximity": self.memory.auto_update_from_proximity,
                "memory_decay_factor": self.memory.memory_decay_factor,
            }
        }
