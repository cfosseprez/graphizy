# src/graphizy/analysis.py

"""
Analysis result objects for Graphizy.

This module provides classes that encapsulate the results of graph analysis,
offering a more intuitive, object-oriented API for accessing and exploring
graph metrics.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import igraph as ig
import numpy as np
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .exceptions import IgraphMethodError

if TYPE_CHECKING:
    from .main import Graphing  # To avoid circular import, for type hinting only


class GraphAnalysisResult:
    """
    A lazy-loading object holding the results of a graph analysis.

    This object behaves like both a standard object (e.g., `results.density`)
    and a dictionary (e.g., `results['density']`), providing maximum flexibility.

    Metrics are computed on-demand the first time they are accessed and then
    cached for subsequent calls.
    """

    def __init__(self, graph: ig.Graph, grapher: 'Graphing'):
        """
        Initialize the result object. This is a lightweight operation.

        Args:
            graph: The igraph.Graph object to be analyzed.
            grapher: The Graphing instance used for the analysis.
        """
        self._graph = graph
        self._grapher = grapher
        self._metric_cache: Dict[str, Any] = {}

    # --- Properties for common metrics (computed lazily) ---

    @property
    def vertex_count(self) -> int:
        """Returns the number of vertices. (Cached on first access)"""
        return self.get_metric('vcount', default_value=0)

    @property
    def edge_count(self) -> int:
        """Returns the number of edges. (Cached on first access)"""
        return self.get_metric('ecount', default_value=0)

    @property
    def density(self) -> float:
        """Returns the graph density. (Cached on first access)"""
        return self._grapher.density(self._graph)

    @property
    def is_connected(self) -> bool:
        """Returns True if the graph is fully connected. (Cached on first access)"""
        return self.get_metric('is_connected', default_value=False)

    @property
    def num_components(self) -> int:
        """Returns the number of disconnected components. (Cached on first access)"""
        components = self.get_metric('connected_components', return_format='raw')
        return len(components) if components else 0

    @property
    def average_path_length(self) -> Optional[float]:
        """Returns the average shortest path length of the largest component. (Cached)"""
        return self.get_metric('average_path_length', component_mode="largest", default_value=None)

    @property
    def diameter(self) -> Optional[int]:
        """Returns the diameter of the largest component. (Cached)"""
        return self.get_metric('diameter', component_mode="largest", default_value=None)

    @property
    def transitivity(self) -> Optional[float]:
        """Returns the global clustering coefficient (transitivity). (Cached)"""
        return self.get_metric('transitivity_undirected', default_value=None)

    # --- Core On-the-fly Metric Computation ---

    def get_metric(self, metric_name: str, **kwargs) -> Any:
        """
        Computes any igraph metric on the fly using the robust `call_method_safe`.
        Results are cached to avoid re-computation.
        """
        cache_key = f"{metric_name}_{sorted(kwargs.items())}"
        if cache_key in self._metric_cache:
            return self._metric_cache[cache_key]

        result = self._grapher.call_method_safe(self._graph, metric_name, **kwargs)
        self._metric_cache[cache_key] = result
        return result

    # --- Helper Methods for Common Statistical Tasks ---

    def get_top_n_by(self, metric_name: str, n: int = 5, **kwargs) -> List[tuple]:
        """
        Returns the top N nodes sorted by a given per-vertex metric.
        Handles None values by treating them as the lowest possible value.
        """
        kwargs['return_format'] = 'dict'
        metric_dict = self.get_metric(metric_name, **kwargs)

        if not isinstance(metric_dict, dict):
            raise TypeError(f"Metric '{metric_name}' did not return a dictionary.")

        sorted_items = sorted(
            metric_dict.items(),
            key=lambda item: item[1] if item[1] is not None else -float('inf'),
            reverse=True
        )
        return sorted_items[:n]

    def get_metric_stats(self, metric_name: str, **kwargs) -> Dict[str, float]:
        """
        Computes descriptive statistics for a numeric metric.
        Handles None values by ignoring them in the calculation.
        """
        kwargs['return_format'] = 'list'
        values = self.get_metric(metric_name, **kwargs)

        if not isinstance(values, list):
            raise TypeError(f"Metric '{metric_name}' did not return a list of values.")

        numeric_values = [v for v in values if v is not None]

        if not numeric_values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0, 'count': 0}

        values_arr = np.array(numeric_values)
        return {
            'mean': float(np.mean(values_arr)),
            'std': float(np.std(values_arr)),
            'min': float(np.min(values_arr)),
            'max': float(np.max(values_arr)),
            'median': float(np.median(values_arr)),
            'count': len(values_arr)
        }

    # --- Dictionary-like Access & Representation ---

    def summary(self) -> str:
        """Provides a clean, readable summary of the key metrics."""
        lines = [
            f"Graph Analysis Summary:",
            f"  - Vertices: {self.vertex_count}",
            f"  - Edges: {self.edge_count}",
            f"  - Density: {self.density:.4f}",
            f"  - Connected: {self.is_connected}",
        ]
        if not self.is_connected:
            lines.append(f"  - Components: {self.num_components}")

        if self.average_path_length is not None:
            lines.append(f"  - Avg. Path Length (largest comp): {self.average_path_length:.2f}")
        if self.diameter is not None:
            lines.append(f"  - Diameter (largest comp): {self.diameter}")
        if self.transitivity is not None:
            lines.append(f"  - Clustering (Transitivity): {self.transitivity:.4f}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"<GraphAnalysisResult: {self.vertex_count} vertices, {self.edge_count} edges>"

    def __getitem__(self, key: str) -> Any:
        """Allows dictionary-style access, e.g., `results['density']`."""
        if hasattr(self, key):
            return getattr(self, key)

        # Check the cache directly before calling get_metric.
        cache_key = f"{key}_{sorted({}.items())}"
        if cache_key in self._metric_cache:
            return self._metric_cache[cache_key]

        try:
            # If not a property and not in cache, compute it.
            return self.get_metric(key)
        except IgraphMethodError as e:
            raise KeyError(f"Metric or property '{key}' not found.") from e

    def __contains__(self, key: str) -> bool:
        """
        Allows using the 'in' operator, e.g., `'density' in results`.
        This is primarily for backward compatibility with tests.
        """
        # Check if it's a defined property on the class.
        return hasattr(self, key)
