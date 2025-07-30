import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional


class WeightComputer:
    """
    Flexible weight computation using edge attributes.
    """

    @staticmethod
    def create_weight_function(formula: str) -> callable:
        """
        Create weight function from formula string.

        Args:
            formula: Formula using edge attribute names
                    Examples: "1/distance", "age * 0.1", "weight / (distance + 1)"

        Returns:
            Callable that takes edge attributes dict and returns weight
        """
        # Safe evaluation context
        safe_dict = {
            "sqrt": np.sqrt,
            "log": np.log,
            "exp": np.exp,
            "abs": abs,
            "min": min,
            "max": max,
            "pow": pow,
        }

        def weight_func(edge_attrs: dict) -> float:
            # Add edge attributes to evaluation context
            eval_dict = {**safe_dict, **edge_attrs}
            try:
                return float(eval(formula, {"__builtins__": {}}, eval_dict))
            except:
                return 1.0  # Default weight

        return weight_func

    @staticmethod
    def distance_weight(edge_attrs: dict) -> float:
        """Weight inversely proportional to distance."""
        distance = edge_attrs.get("distance", 1.0)
        return 1.0 / (distance + 1e-6)  # Avoid division by zero

    @staticmethod
    def age_decay_weight(edge_attrs: dict, decay_factor: float = 0.1) -> float:
        """Weight that decays with age."""
        age = edge_attrs.get("age", 0)
        base_weight = edge_attrs.get("weight", 1.0)
        return base_weight * np.exp(-decay_factor * age)

    @staticmethod
    def combined_weight(edge_attrs: dict) -> float:
        """Combined distance and age weighting."""
        distance = edge_attrs.get("distance", 1.0)
        age = edge_attrs.get("age", 0)
        base_weight = edge_attrs.get("weight", 1.0)

        distance_factor = 1.0 / (distance + 1e-6)
        age_factor = np.exp(-0.1 * age)

        return base_weight * distance_factor * age_factor


def compute_edge_weights(graph: Any, weight_function: callable) -> Any:
    """
    Compute edge weights using a callable function.

    Args:
        graph: igraph Graph with edge attributes
        weight_function: Callable taking edge attributes dict, returning weight

    Returns:
        Graph with updated 'computed_weight' attribute
    """
    if graph.ecount() == 0:
        return graph

    computed_weights = []

    for edge in graph.es:
        edge_attrs = {k: v for k, v in edge.attributes().items()}
        weight = weight_function(edge_attrs)
        computed_weights.append(weight)

    graph.es["computed_weight"] = computed_weights
    return graph