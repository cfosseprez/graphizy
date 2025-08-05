"""
Enhanced weight system for Graphizy

This module provides a comprehensive and flexible system for computing edge weights
in graphs. It supports distance-based weights, temporal weights (age), custom
formulas, and advanced weight computation strategies.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional, Callable
from scipy.spatial.distance import cdist
from graphizy.exceptions import GraphCreationError, IgraphMethodError
import time

DEFAULT_DISTANCE_KEY = "distance"
DEFAULT_AGE_KEY = "age"
DEFAULT_WEIGHT_KEY = "weight"


# ============================================================================
# DISTANCE COMPUTATION FUNCTIONS
# ============================================================================


def handle_add_distance(graph, add_distance: Union[bool, Dict[str, Any]]) -> Any:
    """
    Compute and add edge distances to a graph based on node coordinates.

    Enhanced version with better error handling and configuration options.
    """
    if not add_distance:
        return graph

    # Parse configuration
    if isinstance(add_distance, dict):
        metric = add_distance.get("metric", "euclidean")
        attribute = add_distance.get("attribute", DEFAULT_DISTANCE_KEY)
    else:
        metric = "euclidean"
        attribute = DEFAULT_DISTANCE_KEY
        logging.debug("Using default distance configuration")

    # Validate metric
    if not isinstance(metric, str):
        raise ValueError(f"Expected 'metric' to be a string, got {type(metric).__name__}")

    return add_edge_distances(graph, metric=metric, attribute=attribute)

def add_edge_distances(graph: Any, metric: str = "euclidean", attribute: str = DEFAULT_DISTANCE_KEY) -> Any:
    """
    Compute and assign distances between connected nodes in an igraph Graph,
    using the 'x' and 'y' vertex attributes.

    Args:
        graph: igraph Graph object with 'x' and 'y' attributes for vertices.
        metric: Distance metric - 'euclidean', 'manhattan', or 'chebyshev'.

    Returns:
        Modified graph with 'distance' added to each edge.
    """
    if graph.ecount() == 0:
        return graph

    # Get vertex coordinates as numpy array
    coords = np.column_stack((graph.vs["x"], graph.vs["y"]))

    # Extract source and target coordinates
    sources = np.array([edge.source for edge in graph.es])
    targets = np.array([edge.target for edge in graph.es])
    source_coords = coords[sources]
    target_coords = coords[targets]

    # Compute distances
    if metric == "euclidean":
        distances = np.linalg.norm(source_coords - target_coords, axis=1)
    elif metric == "manhattan":
        distances = np.sum(np.abs(source_coords - target_coords), axis=1)
    elif metric == "chebyshev":
        distances = np.max(np.abs(source_coords - target_coords), axis=1)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Assign distances to edges
    graph.es["distance"] = distances.tolist()
    return graph

def add_edge_distances_square(
    graph: Any,
    data_points: np.ndarray,
    metric: str = "euclidean",
    attribute: str = DEFAULT_DISTANCE_KEY
) -> Any:
    """
    Add pairwise distances between connected nodes as edge attributes.

    Enhanced version with better error handling and support for different coordinate systems.
    """
    try:
        # Validate inputs
        if graph is None or data_points is None:
            raise ValueError("Graph and data_points cannot be None")

        if data_points.shape[1] < 3:
            raise ValueError("data_points must have at least 3 columns [id, x, y]")

        # Create ID to coordinate mapping
        id_to_coord = {int(row[0]): row[1:3] for row in data_points}

        coords = []
        edge_pairs = []
        missing_count = 0

        for e in graph.es:
            try:
                source_id = graph.vs[e.source]['id']
                target_id = graph.vs[e.target]['id']

                if source_id in id_to_coord and target_id in id_to_coord:
                    coords.append([id_to_coord[source_id], id_to_coord[target_id]])
                    edge_pairs.append(e)
                else:
                    missing_count += 1
                    logging.debug(f"Missing coordinates for edge ({source_id}, {target_id})")

            except KeyError as e:
                missing_count += 1
                logging.debug(f"KeyError for edge {e.index}: {e}")

        if missing_count > 0:
            logging.warning(f"Missing coordinates for {missing_count} edges")

        if not coords:
            logging.warning("No valid coordinates found for any edges")
            return graph

        # Compute distances
        coords = np.array(coords)
        if metric == "euclidean":
            distances = np.linalg.norm(coords[:, 0] - coords[:, 1], axis=1)
        elif metric == "manhattan":
            distances = np.sum(np.abs(coords[:, 0] - coords[:, 1]), axis=1)
        elif metric == "chebyshev":
            distances = np.max(np.abs(coords[:, 0] - coords[:, 1]), axis=1)
        else:
            # Use scipy for other metrics
            distances = []
            for coord_pair in coords:
                dist = cdist([coord_pair[0]], [coord_pair[1]], metric=metric)[0, 0]
                distances.append(dist)

        # Assign distances to edges
        for edge, dist in zip(edge_pairs, distances):
            edge[attribute] = float(dist)

        logging.debug(f"Added distances to {len(edge_pairs)} edges using metric '{metric}'")
        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to add edge distances: {str(e)}")


# ============================================================================
# CORE WEIGHT COMPUTATION CLASS
# ============================================================================

class WeightComputer:
    """
    Advanced weight computation system for graph edges.

    This class provides a flexible framework for computing any edge attribute using
    various strategies. It can compute multiple attributes (distance, weight, custom)
    and use existing edge attributes in formulas. Optimized for real-time applications.

    Examples:
        # Compute distance and weight separately
        computer = WeightComputer(method="distance", target_attribute="distance")
        graph = computer.compute_weights(graph, data_points)

        computer = WeightComputer(method="formula", formula="1/distance", target_attribute="weight")
        graph = computer.compute_weights(graph, data_points)

        # Chain multiple computations
        computer.compute_attribute(graph, data_points, "distance", method="distance")
        computer.compute_attribute(graph, data_points, "weight", method="formula", formula="1/(distance + 0.1)")
        computer.compute_attribute(graph, data_points, "strength", method="formula", formula="weight * age")
    """

    def __init__(self,
                 method: str = "distance",
                 auto_add_distance: bool = True,
                 distance_metric: str = "euclidean",
                 formula: Optional[str] = None,
                 custom_function: Optional[Callable] = None,
                 normalize: bool = False,
                 target_attribute: Optional[str] = None,
                 **method_params):
        """
        Initialize the WeightComputer for flexible attribute computation.

        Args:
            method: Computation method:
                   - "distance": Compute distance between connected nodes
                   - "age": Use temporal age (requires memory system)
                   - "formula": Use custom formula with existing edge attributes
                   - "function": Use custom function
                   - "combined": Use multiple factors
            auto_add_distance: Whether to automatically compute distance if needed by formulas
            distance_metric: Metric for distance computation ("euclidean", "manhattan", etc.)
            formula: Custom formula string using edge attribute names (e.g., "1/distance", "distance * age")
            custom_function: Custom function taking (graph, **params) -> List[float]
            normalize: Whether to normalize results to [0,1] range
            target_attribute: Name of edge attribute to store results. If None, uses method-specific defaults:
                             - "distance" method -> "distance"
                             - "age" method -> "age_weight"
                             - "formula" method -> "weight"
                             - others -> "weight"
            **method_params: Additional parameters for specific methods

        Examples:
            # Compute distances and store in "distance" attribute
            WeightComputer(method="distance", target_attribute="distance")

            # Compute weights using distance formula, store in "weight" attribute
            WeightComputer(method="formula", formula="1/(distance + 0.1)", target_attribute="weight")

            # Compute custom metric using multiple attributes
            WeightComputer(method="formula", formula="distance * age / 100", target_attribute="importance")
        """
        self.method = method
        self.auto_add_distance = auto_add_distance
        self.distance_metric = distance_metric
        self.formula = formula
        self.custom_function = custom_function
        self.normalize = normalize
        self.method_params = method_params

        # Set default target attribute based on method
        if target_attribute is None:
            if method == "distance":
                self.target_attribute = DEFAULT_DISTANCE_KEY
            elif method == "age":
                self.target_attribute = "age_weight"
            else:
                self.target_attribute = DEFAULT_WEIGHT_KEY
        else:
            self.target_attribute = target_attribute

        # Validate configuration
        self._validate_config()

        logging.info(f"WeightComputer initialized: method='{method}', target='{self.target_attribute}'")

    def _validate_config(self):
        """Validate the weight computer configuration."""
        valid_methods = ["distance", "age", "formula", "function", "combined"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}'. Must be one of {valid_methods}")

        if self.method == "formula" and not self.formula:
            raise ValueError("Formula method requires 'formula' parameter")

        if self.method == "function" and not self.custom_function:
            raise ValueError("Function method requires 'custom_function' parameter")

    def compute_weights(self, graph: Any) -> Any:
        """
        Compute and assign values to target edge attribute.

        Args:
            graph: igraph Graph object

        Returns:
            Graph with computed values assigned to target attribute
        """
        return self.compute_attribute(graph,
                                    target_attribute=self.target_attribute,
                                    method=self.method,
                                    formula=self.formula,
                                    custom_function=self.custom_function,
                                    normalize=self.normalize,
                                    **self.method_params)

    def compute_attribute(self, graph: Any,
                         target_attribute: str,
                         method: Optional[str] = None,
                         formula: Optional[str] = None,
                         custom_function: Optional[Callable] = None,
                         normalize: bool = False,
                         do_timing: bool = False,
                         **method_params) -> Any:
        """
        Compute and assign values to any specified edge attribute.

        This is the core method that allows computing multiple different attributes
        on the same graph. Perfect for real-time applications where you need both
        distance and weight attributes.

        Args:
            graph: igraph Graph object
            target_attribute: Name of edge attribute to store results
            method: Override method for this computation
            formula: Override formula for this computation
            custom_function: Override function for this computation
            normalize: Whether to normalize results for this computation
            do_timing: Whether to print the performances
            **method_params: Additional parameters for this computation

        Returns:
            Graph with computed values in target_attribute

        Examples:
            # Compute distance and store in "distance" attribute
            computer.compute_attribute(graph, data, "distance", method="distance")

            # Then compute weight using the distance attribute
            computer.compute_attribute(graph, data, "weight", method="formula", formula="1/distance")

            # Compute importance using multiple attributes
            computer.compute_attribute(graph, data, "importance", method="formula",
                                     formula="weight * age / max_age", max_age=100)
        """
        try:
            if do_timing:
                start_time = time.perf_counter()

            # Use instance defaults if not overridden
            method = method or self.method
            formula = formula or self.formula
            custom_function = custom_function or self.custom_function

            # Merge method params
            final_params = self.method_params.copy()
            final_params.update(method_params)

            # Step 1: Ensure required attributes exist
            if self._needs_distance_for_method(method, formula) and self.auto_add_distance:
                graph = self._ensure_distance_attributes(graph)

            # Step 2: Compute values based on method
            if method == "distance":
                values = self._compute_distance_values(graph)
            elif method == "age":
                values = self._compute_age_values(graph, **final_params)
            elif method == "formula":
                values = self._compute_formula_values(graph, formula, **final_params)
            elif method == "function":
                values = self._compute_function_values(graph, custom_function, **final_params)
            elif method == "combined":
                values = self._compute_combined_values(graph, **final_params)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Step 3: Handle NaN/inf values
            values = self._clean_values(values, **final_params)

            # Step 4: Normalize if requested
            if normalize:
                values = self._normalize_values(values)

            # Step 5: Assign to graph
            graph.es[target_attribute] = values

            if do_timing:
                duration = time.perf_counter() - start_time
                print(
                    f"Attribute '{target_attribute}' computed in {duration:.4f} seconds using method '{method}'.")


            logging.debug(f"Computed {len(values)} values for '{target_attribute}' using method '{method}'")
            return graph

        except Exception as e:
            raise GraphCreationError(f"Failed to compute attribute '{target_attribute}': {str(e)}")

    def _needs_distance_for_method(self, method: str, formula: Optional[str]) -> bool:
        """Check if the method needs distance attributes."""
        distance_methods = ["distance", "combined"]
        if method in distance_methods:
            return True
        if method == "formula" and formula and DEFAULT_DISTANCE_KEY in formula:
            return True
        return False

    def _ensure_distance_attributes(self, graph: Any,) -> Any:
        """Ensure distance attributes exist on edges."""
        if DEFAULT_DISTANCE_KEY not in graph.es.attributes():
            graph = add_edge_distances(
                graph,
                metric=self.distance_metric,
                attribute=DEFAULT_DISTANCE_KEY
            )
        return graph

    def _compute_distance_values(self, graph: Any) -> List[float]:
        """Compute actual distance values (not weights)."""
        # Handle graphs with no edges gracefully
        if graph.ecount() == 0:
            return []

        # Force distance computation if not present
        if DEFAULT_DISTANCE_KEY not in graph.es.attributes():
            graph = add_edge_distances(
                graph,
                metric=self.distance_metric,
                attribute=DEFAULT_DISTANCE_KEY
            )

        # This check is now safe because we know there are edges
        if DEFAULT_DISTANCE_KEY not in graph.es.attributes():
             # This can happen if add_edge_distances fails silently
             logging.warning("Distance attribute still missing after computation attempt.")
             return [0.0] * graph.ecount()

        return list(graph.es[DEFAULT_DISTANCE_KEY])

    def _compute_age_values(self, graph: Any, **params) -> List[float]:
        """Compute values based on edge age."""
        if DEFAULT_AGE_KEY not in graph.es.attributes():
            raise ValueError(f"Graph missing '{DEFAULT_AGE_KEY}' attribute. Requires memory system.")

        ages = graph.es[DEFAULT_AGE_KEY]
        mode = params.get('age_mode', 'direct')  # 'direct', 'inverse', 'exponential'

        if mode == 'direct':
            return list(ages)
        elif mode == 'inverse':
            max_age = max(ages) if ages else 1
            return [max_age - age + 1 for age in ages]
        elif mode == 'exponential':
            decay = params.get('decay_rate', 0.1)
            return [np.exp(-decay * age) for age in ages]
        else:
            raise ValueError(f"Unknown age_mode: {mode}")

    def _compute_formula_values(self, graph: Any, formula: str, **params) -> List[float]:
        """Compute values using custom formula with edge attributes."""
        if not formula:
            raise ValueError("Formula method requires formula parameter")

        values = []
        safe_functions = {
            'np': np, 'numpy': np, 'exp': np.exp, 'log': np.log,
            'sqrt': np.sqrt, 'abs': abs, 'min': min, 'max': max,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan
        }

        # Add any additional parameters to context
        safe_functions.update(params)

        for edge in graph.es:
            context = dict(edge.attributes())
            context.update(safe_functions)

            try:
                value = eval(formula, {"__builtins__": {}}, context)
                values.append(float(value))
            except Exception as e:
                logging.warning(f"Formula '{formula}' failed for edge {edge.index}: {e}")
                default_value = params.get('default_value', 0.0)
                values.append(default_value)

        return values

    def _compute_function_values(self, graph: Any, custom_function: Optional[Callable], **params) -> List[float]:
        """Compute values using custom function."""
        if not custom_function:
            raise ValueError("Function method requires custom_function parameter")

        try:
            result = custom_function(graph, **params)
            if isinstance(result, (list, np.ndarray)):
                return list(result)
            else:
                # Single value - broadcast to all edges
                return [result] * graph.ecount()
        except Exception as e:
            logging.error(f"Custom function failed: {e}")
            default_value = params.get('default_value', 1.0)
            return [default_value] * graph.ecount()

    def _compute_combined_values(self, graph: Any, **params) -> List[float]:
        """Compute values using multiple factors."""
        factors = params.get('factors', ['distance'])
        weights_dict = params.get('weights', {})

        # Default weights for each factor
        default_weights = {'distance': 0.5, 'age': 0.5}

        combined_values = [0.0] * graph.ecount()

        for factor in factors:
            factor_weight = weights_dict.get(factor, default_weights.get(factor, 1.0))

            if factor == 'distance':
                factor_values = self._compute_distance_values(graph, np.array([]))
            elif factor == 'age':
                factor_values = self._compute_age_values(graph, **params)
            else:
                logging.warning(f"Unknown factor '{factor}', skipping")
                continue

            # Normalize factor values to [0,1] before combining
            if factor_values:
                min_val, max_val = min(factor_values), max(factor_values)
                if max_val > min_val:
                    factor_values = [(v - min_val) / (max_val - min_val) for v in factor_values]

            # Add weighted contribution
            for i, val in enumerate(factor_values):
                combined_values[i] += factor_weight * val

        return combined_values

    def _clean_values(self, values: List[float], **params) -> List[float]:
        """Clean NaN and inf values."""
        default_value = params.get('default_value', 1.0)

        cleaned = []
        for v in values:
            if np.isnan(v) or np.isinf(v):
                cleaned.append(default_value)
            else:
                cleaned.append(v)

        return cleaned

    def _normalize_values(self, values: List[float]) -> List[float]:
        """Normalize values to [0,1] range."""
        if not values:
            return values

        values_array = np.array(values)
        min_v, max_v = np.nanmin(values_array), np.nanmax(values_array)

        if max_v > min_v:
            normalized = (values_array - min_v) / (max_v - min_v)
            return normalized.tolist()
        else:
            # All values are the same
            return [0.5] * len(values)

    def get_attribute_info(self, graph: Any, attribute: str) -> Dict[str, Any]:
        """Get information about computed attribute values."""
        if attribute not in graph.es.attributes():
            return {"error": f"No '{attribute}' attribute found"}

        values = graph.es[attribute]
        values_array = np.array(values)

        return {
            "attribute": attribute,
            "count": len(values),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "mean": float(np.mean(values_array)),
            "std": float(np.std(values_array)),
            "has_nan": bool(np.any(np.isnan(values_array))),
            "has_inf": bool(np.any(np.isinf(values_array)))
        }


# ============================================================================
# REAL-TIME OPTIMIZED FUNCTIONS
# ============================================================================

class FastAttributeComputer:
    """
    Optimized version for real-time applications.

    Pre-compiles formulas and caches computation strategies for maximum performance.
    Ideal for applications that need to compute the same attributes repeatedly.
    """

    def __init__(self):
        self._compiled_formulas = {}
        self._distance_computer = None

    def setup_distance_computation(self, metric: str = "euclidean"):
        """Pre-setup distance computation for speed."""
        self._distance_computer = {
            'metric': metric,
            'coord_cache': {}  # Cache coordinate lookups
        }

    def compute_distance_fast(self, graph: Any, ) -> Any:
        """Fast distance computation with caching."""
        if DEFAULT_DISTANCE_KEY in graph.es.attributes():
            return graph  # Already computed

        # Use cached version if available
        if not self._distance_computer:
            self.setup_distance_computation()

        return add_edge_distances(graph,
                                       metric=self._distance_computer['metric'])

    def compute_weight_from_distance_fast(self, graph: Any, epsilon: float = 1e-10) -> Any:
        """Fast weight computation: weight = 1/distance."""
        if DEFAULT_DISTANCE_KEY not in graph.es.attributes():
            raise ValueError("Distance attribute required. Call compute_distance_fast first.")

        distances = graph.es[DEFAULT_DISTANCE_KEY]
        weights = [1.0 / (d + epsilon) for d in distances]
        graph.es[DEFAULT_WEIGHT_KEY] = weights
        return graph

    def compute_multiple_attributes_fast(self, graph: Any,
                                       attributes_config: Dict[str, Dict]) -> Any:
        """
        Compute multiple attributes in one pass for maximum efficiency.

        Args:
            graph: igraph Graph
            data_points: np.ndarray with coordinates
            attributes_config: Dict of {attribute_name: {method, formula, params}}

        Example:
            config = {
                "distance": {"method": "distance"},
                "weight": {"method": "formula", "formula": "1/(distance + 0.1)"},
                "strength": {"method": "formula", "formula": "weight * 2"},
            }
            fast_computer.compute_multiple_attributes_fast(graph, data, config)
        """
        # Step 1: Compute distance if any attribute needs it
        needs_distance = any(
            conf.get('method') == 'distance' or
            ('formula' in conf and DEFAULT_DISTANCE_KEY in conf.get('formula', ''))
            for conf in attributes_config.values()
        )

        if needs_distance:
            graph = self.compute_distance_fast(graph)

        # Step 2: Compute attributes in dependency order
        computed = set()

        # Simple dependency resolution - compute in order of dependencies
        max_iterations = len(attributes_config) * 2
        iteration = 0

        while len(computed) < len(attributes_config) and iteration < max_iterations:
            iteration += 1

            for attr_name, config in attributes_config.items():
                if attr_name in computed:
                    continue

                # Check if dependencies are satisfied
                if config.get('method') == 'formula':
                    formula = config.get('formula', '')
                    # Check if all required attributes exist
                    required_attrs = self._extract_attributes_from_formula(formula)
                    if not all(attr in graph.es.attributes() or attr in computed for attr in required_attrs):
                        continue  # Skip for now, dependencies not ready

                # Compute this attribute
                try:
                    method = config.get('method', 'formula')

                    if method == 'distance':
                        # Already computed above
                        values = list(graph.es[DEFAULT_DISTANCE_KEY])
                    elif method == 'formula':
                        formula = config['formula']
                        values = self._compute_formula_fast(graph, formula, config.get('params', {}))
                    else:
                        # Use regular WeightComputer for other methods
                        temp_computer = WeightComputer(method=method, target_attribute=attr_name, **config.get('params', {}))
                        graph = temp_computer.compute_attribute(graph, attr_name, method=method)
                        computed.add(attr_name)
                        continue

                    # Apply normalization if requested
                    if config.get('normalize', False):
                        values = self._normalize_fast(values)

                    # Assign to graph
                    graph.es[attr_name] = values
                    computed.add(attr_name)

                except Exception as e:
                    logging.error(f"Failed to compute attribute '{attr_name}': {e}")
                    # Set default values to avoid breaking the chain
                    graph.es[attr_name] = [1.0] * graph.ecount()
                    computed.add(attr_name)

        return graph

    def _extract_attributes_from_formula(self, formula: str) -> List[str]:
        """Extract attribute names from formula string."""
        import re
        # Simple regex to find potential attribute names
        # This is a simplified version - could be enhanced
        potential_attrs = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)

        # Filter out known functions and operators
        functions = {'np', 'exp', 'log', 'sqrt', 'abs', 'min', 'max', 'sin', 'cos', 'tan'}
        return [attr for attr in potential_attrs if attr not in functions]

    def _compute_formula_fast(self, graph: Any, formula: str, params: Dict) -> List[float]:
        """Fast formula computation with minimal overhead."""
        # Check if we've compiled this formula before
        if formula not in self._compiled_formulas:
            # Pre-compile safe functions context
            self._compiled_formulas[formula] = {
                'safe_functions': {
                    'np': np, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                    'abs': abs, 'min': min, 'max': max, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan
                }
            }
            self._compiled_formulas[formula]['safe_functions'].update(params)

        context_base = self._compiled_formulas[formula]['safe_functions']
        values = []

        for edge in graph.es:
            context = dict(edge.attributes())
            context.update(context_base)

            try:
                value = eval(formula, {"__builtins__": {}}, context)
                values.append(float(value))
            except Exception:
                values.append(params.get('default_value', 0.0))

        return values

    def _normalize_fast(self, values: List[float]) -> List[float]:
        """Fast normalization without numpy overhead for small lists."""
        if not values:
            return values

        min_val = min(values)
        max_val = max(values)

        if max_val > min_val:
            range_val = max_val - min_val
            return [(v - min_val) / range_val for v in values]
        else:
            return [0.5] * len(values)


# ============================================================================
# CONVENIENCE FUNCTIONS FOR REAL-TIME SCENARIOS
# ============================================================================

def compute_distance_and_weight_fast(graph: Any, data_points: np.ndarray,
                                    weight_formula: str = "1/(distance + 0.1)",
                                    distance_metric: str = "euclidean") -> Any:
    """
    Ultra-fast computation of both distance and weight attributes.

    Optimized for real-time applications where you need both attributes computed efficiently.

    Args:
        graph: igraph Graph
        data_points: coordinate data
        weight_formula: formula for weight computation using 'distance'
        distance_metric: distance computation metric

    Returns:
        Graph with both 'distance' and 'weight' attributes

    Example:
        # Most common case - distance and inverse weight
        graph = compute_distance_and_weight_fast(graph, data)

        # Custom weight formula
        graph = compute_distance_and_weight_fast(graph, data, "exp(-distance/50)")
    """
    fast_computer = FastAttributeComputer()

    config = {
        "distance": {"method": "distance"},
        "weight": {"method": "formula", "formula": weight_formula}
    }

    return fast_computer


# ============================================================================
# CONVENIENCE CONSTRUCTORS
# ============================================================================

def create_distance_computer(metric: str = "euclidean") -> WeightComputer:
    """Create a WeightComputer for distance computation."""
    return WeightComputer(method="distance", distance_metric=metric, target_attribute="distance")


def create_weight_from_distance_computer(formula: str = "1/(distance + 0.1)",
                                        normalize: bool = False) -> WeightComputer:
    """Create a WeightComputer for weight from distance."""
    return WeightComputer(method="formula", formula=formula, target_attribute="weight", normalize=normalize)


def create_custom_attribute_computer(attribute_name: str, formula: str,
                                   normalize: bool = False) -> WeightComputer:
    """Create a WeightComputer for any custom attribute."""
    return WeightComputer(method="formula", formula=formula, target_attribute=attribute_name, normalize=normalize)


def create_age_weight_computer(age_mode: str = "exponential",
                              decay_rate: float = 0.1,
                              normalize: bool = True) -> WeightComputer:
    """Create a WeightComputer for age-based weights."""
    return WeightComputer(
        method="age",
        normalize=normalize,
        age_mode=age_mode,
        decay_rate=decay_rate,
        target_attribute="weight"
    )


def create_combined_weight_computer(factors: List[str],
                                   weights: Dict[str, float],
                                   normalize: bool = True) -> WeightComputer:
    """Create a WeightComputer for combined weight factors."""
    return WeightComputer(
        method="combined",
        normalize=normalize,
        factors=factors,
        weights=weights
    )


# ============================================================================
# ADVANCED WEIGHT STRATEGIES
# ============================================================================

def threshold_weight_computer(threshold: float,
                             high: float = 1.0,
                             low: float = 0.0) -> WeightComputer:
    """Create a threshold-based weight computer."""
    def threshold_function(graph, threshold=threshold, high=high, low=low):
        if DEFAULT_DISTANCE_KEY not in graph.es.attributes():
            raise ValueError("Graph requires distance attribute for threshold weights")
        distances = graph.es[DEFAULT_DISTANCE_KEY]
        return [high if d <= threshold else low for d in distances]

    return WeightComputer(method="function", custom_function=threshold_function)


def linear_combination_weight_computer(alpha: float = 0.5) -> WeightComputer:
    """Create a linear combination weight computer for distance and age."""
    return create_combined_weight_computer(
        factors=['distance', 'age'],
        weights={'distance': alpha, 'age': 1 - alpha}
    )



def setup_realtime_weight_computer(distance_metric: str = "euclidean",
                                  weight_formula: str = "1/(distance + 0.1)",
                                  additional_attributes: Optional[Dict] = None) -> FastAttributeComputer:
    """
    Setup a pre-configured fast computer for real-time applications.

    Args:
        distance_metric: metric for distance computation
        weight_formula: formula for weight computation
        additional_attributes: dict of {attr_name: {method, formula, params}}

    Returns:
        Configured FastAttributeComputer ready for repeated use

    Example:
        # Setup once
        fast_comp = setup_realtime_weight_computer(
            weight_formula="1/(distance + 0.01)",
            additional_attributes={
                "importance": {"method": "formula", "formula": "weight * 2"},
                "strength": {"method": "formula", "formula": "distance * weight"}
            }
        )

        # Use repeatedly in real-time loop
        for new_data in data_stream:
            graph = make_graph(new_data)
            graph = fast_comp.compute_multiple_attributes_fast(graph, new_data, fast_comp._default_config)
    """
    fast_computer = FastAttributeComputer()
    fast_computer.setup_distance_computation(distance_metric)

    # Build default configuration
    config = {
        "distance": {"method": "distance"},
        "weight": {"method": "formula", "formula": weight_formula}
    }

    if additional_attributes:
        config.update(additional_attributes)

    # Store config for repeated use
    fast_computer._default_config = config