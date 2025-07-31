import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional, Callable


class WeightComputer:
    """
    Flexible weight computation using edge attributes.
    """

    @staticmethod
    def create_weight_function(formula: str) -> Callable[[dict], float]:
        """
        Create weight function from formula string.

        Args:
            formula: Formula using edge attribute names
                    Examples: "1/distance", "age * 0.1", "weight / (distance + 1)"

        Returns:
            Callable that takes edge attributes dict and returns weight

        Raises:
            ValueError: If formula is invalid or unsafe
        """
        if not isinstance(formula, str) or not formula.strip():
            raise ValueError("Formula must be a non-empty string")

        # Enhanced safe evaluation context
        safe_dict = {
            # Math functions
            "sqrt": np.sqrt,
            "log": np.log,
            "log10": np.log10,
            "exp": np.exp,
            "abs": abs,
            "min": min,
            "max": max,
            "pow": pow,
            # Trigonometric functions
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            # Additional useful functions
            "ceil": np.ceil,
            "floor": np.floor,
            "round": round,
            # Constants
            "pi": np.pi,
            "e": np.e,
        }

        # Validate formula doesn't contain dangerous operations
        dangerous_keywords = [
            "import", "exec", "eval", "open", "file", "__",
            "globals", "locals", "vars", "dir", "getattr", "setattr"
        ]

        formula_lower = formula.lower()
        for keyword in dangerous_keywords:
            if keyword in formula_lower:
                raise ValueError(f"Unsafe keyword '{keyword}' found in formula")

        def weight_func(edge_attrs: dict) -> float:
            # Add edge attributes to evaluation context
            eval_dict = {**safe_dict, **edge_attrs}
            try:
                result = eval(formula, {"__builtins__": {}}, eval_dict)

                # Handle potential invalid results
                if np.isnan(result) or np.isinf(result) or result < 0:
                    return 1.0  # Default weight for invalid results

                return float(result)
            except (ZeroDivisionError, ValueError, TypeError, KeyError):
                return 1.0  # Default weight for errors
            except Exception:
                # Log unexpected errors in production
                return 1.0

        return weight_func

    @staticmethod
    def distance_weight(edge_attrs: dict, invert: bool = True, offset: float = 1e-6) -> float:
        """
        Weight based on distance.

        Args:
            edge_attrs: Edge attributes dictionary
            invert: If True, closer distances get higher weights (1/distance)
                   If False, farther distances get higher weights (distance)
            offset: Small offset to prevent division by zero

        Returns:
            Computed weight based on distance
        """
        distance = edge_attrs.get("distance", 1.0)

        if distance <= 0:
            distance = offset  # Handle zero or negative distances

        if invert:
            return 1.0 / (distance + offset)
        else:
            return distance

    @staticmethod
    def age_decay_weight(edge_attrs: dict, decay_factor: float = 0.1,
                         base_attr: str = "weight") -> float:
        """
        Weight that decays exponentially with age.

        Args:
            edge_attrs: Edge attributes dictionary
            decay_factor: Rate of decay (higher = faster decay)
            base_attr: Attribute to use as base weight

        Returns:
            Weight decayed by age factor
        """
        age = edge_attrs.get("age", 0)
        base_weight = edge_attrs.get(base_attr, 1.0)

        if age < 0:
            age = 0  # Handle negative ages

        return base_weight * np.exp(-decay_factor * age)

    @staticmethod
    def combined_distance_age_weight(edge_attrs: dict,
                                     distance_power: float = 1.0,
                                     age_decay: float = 0.1,
                                     base_attr: str = "weight") -> float:
        """
        Combined distance and age weighting with configurable parameters.

        Args:
            edge_attrs: Edge attributes dictionary
            distance_power: Power to apply to distance factor (higher = more distance sensitive)
            age_decay: Age decay factor (higher = faster decay)
            base_attr: Attribute to use as base weight

        Returns:
            Combined weight incorporating distance and age
        """
        distance = edge_attrs.get("distance", 1.0)
        age = edge_attrs.get("age", 0)
        base_weight = edge_attrs.get(base_attr, 1.0)

        # Handle edge cases
        if distance <= 0:
            distance = 1e-6
        if age < 0:
            age = 0

        distance_factor = (1.0 / (distance + 1e-6)) ** distance_power
        age_factor = np.exp(-age_decay * age)

        return base_weight * distance_factor * age_factor

    @staticmethod
    def linear_combination_weight(edge_attrs: dict,
                                  coefficients: Dict[str, float],
                                  default_value: float = 0.0) -> float:
        """
        Linear combination of edge attributes.

        Args:
            edge_attrs: Edge attributes dictionary
            coefficients: Dictionary mapping attribute names to coefficients
            default_value: Default value for missing attributes

        Returns:
            Linear combination of attributes

        Example:
            coefficients = {"distance": -0.1, "age": -0.05, "weight": 1.0}
            # Result: weight - 0.1*distance - 0.05*age
        """
        result = 0.0
        for attr_name, coeff in coefficients.items():
            attr_value = edge_attrs.get(attr_name, default_value)
            result += coeff * attr_value

        return max(result, 1e-6)  # Ensure positive weight

    @staticmethod
    def threshold_weight(edge_attrs: dict,
                         attribute: str,
                         threshold: float,
                         high_weight: float = 1.0,
                         low_weight: float = 0.1) -> float:
        """
        Threshold-based weight assignment.

        Args:
            edge_attrs: Edge attributes dictionary
            attribute: Attribute to threshold on
            threshold: Threshold value
            high_weight: Weight when attribute >= threshold
            low_weight: Weight when attribute < threshold

        Returns:
            Threshold-based weight
        """
        value = edge_attrs.get(attribute, 0.0)
        return high_weight if value >= threshold else low_weight


def compute_edge_weights(graph: Any,
                         weight_function: Union[str, Callable[[dict], float]],
                         output_attr: str = "computed_weight",
                         validate_results: bool = True) -> Any:
    """
    Compute edge weights using a callable function or formula string.

    Args:
        graph: igraph Graph with edge attributes
        weight_function: Either formula string or callable function
        output_attr: Name for the computed weight attribute
        validate_results: Whether to validate and clean results

    Returns:
        Graph with new weight attribute

    Raises:
        ValueError: If weight_function is invalid type
        GraphCreationError: If computation fails
    """
    if graph.ecount() == 0:
        return graph

    try:
        # Convert string formula to function if needed
        if isinstance(weight_function, str):
            weight_func = WeightComputer.create_weight_function(weight_function)
        elif callable(weight_function):
            weight_func = weight_function
        else:
            raise ValueError(f"weight_function must be str or callable, got {type(weight_function)}")

        computed_weights = []

        for edge in graph.es:
            edge_attrs = {k: v for k, v in edge.attributes().items()}
            weight = weight_func(edge_attrs)

            # Validate result if requested
            if validate_results:
                if not isinstance(weight, (int, float)) or np.isnan(weight) or np.isinf(weight):
                    weight = 1.0  # Fallback for invalid results
                elif weight <= 0:
                    weight = 1e-6  # Ensure positive weights

            computed_weights.append(weight)

        graph.es[output_attr] = computed_weights
        return graph

    except Exception as e:
        from graphizy.exceptions import GraphCreationError
        raise GraphCreationError(f"Failed to compute edge weights: {str(e)}")


# Enhanced convenience functions
def apply_distance_weights(graph: Any, invert: bool = True, output_attr: str = "distance_weight") -> Any:
    """Convenience function to apply distance-based weights."""
    return compute_edge_weights(
        graph,
        lambda attrs: WeightComputer.distance_weight(attrs, invert=invert),
        output_attr=output_attr
    )


def apply_age_weights(graph: Any, decay_factor: float = 0.1, output_attr: str = "age_weight") -> Any:
    """Convenience function to apply age-based weights."""
    return compute_edge_weights(
        graph,
        lambda attrs: WeightComputer.age_decay_weight(attrs, decay_factor=decay_factor),
        output_attr=output_attr
    )


def apply_combined_weights(graph: Any, distance_power: float = 1.0, age_decay: float = 0.1,
                           output_attr: str = "combined_weight") -> Any:
    """Convenience function to apply combined distance-age weights."""
    return compute_edge_weights(
        graph,
        lambda attrs: WeightComputer.combined_distance_age_weight(
            attrs, distance_power=distance_power, age_decay=age_decay
        ),
        output_attr=output_attr
    )


# Example usage and testing functions
def example_weight_usage():
    """Example of how to use the weight computation system."""
    import igraph as ig

    # Create sample graph
    graph = ig.Graph()
    graph.add_vertices(3)
    graph.vs["id"] = [1, 2, 3]
    graph.add_edges([(0, 1), (1, 2)])

    # Add sample attributes
    graph.es["distance"] = [5.0, 10.0]
    graph.es["age"] = [1, 5]
    graph.es["weight"] = [1.0, 2.0]

    # Method 1: Formula strings
    graph = compute_edge_weights(graph, "1/distance", "inv_distance_weight")
    graph = compute_edge_weights(graph, "weight * exp(-0.1 * age)", "age_decayed_weight")

    # Method 2: Predefined functions
    graph = apply_distance_weights(graph, invert=True)
    graph = apply_age_weights(graph, decay_factor=0.15)
    graph = apply_combined_weights(graph, distance_power=1.5, age_decay=0.1)

    # Method 3: Custom functions
    def custom_weight(attrs):
        return attrs.get("weight", 1.0) * np.log(attrs.get("distance", 1.0) + 1)

    graph = compute_edge_weights(graph, custom_weight, "custom_weight")

    # Print results
    for attr in graph.es.attributes():
        if "weight" in attr:
            print(f"{attr}: {graph.es[attr]}")

    return graph


if __name__ == "__main__":
    example_weight_usage()