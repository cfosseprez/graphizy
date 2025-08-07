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
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Tuple
from dataclasses import dataclass

from .exceptions import IgraphMethodError

if TYPE_CHECKING:
    from .main import Graphing  # To avoid circular import, for type hinting only


@dataclass
class PercolationResult:
    """Results from percolation analysis"""
    interaction_ranges: List[float]
    largest_cluster_sizes: List[int]
    total_clusters: List[int]
    percolation_probabilities: List[float]
    critical_range: Optional[float]
    
    def get_critical_probability(self) -> float:
        """Get percolation probability at critical range"""
        if self.critical_range is None:
            return 0.0
        
        # Find closest range to critical
        closest_idx = min(range(len(self.interaction_ranges)),
                         key=lambda i: abs(self.interaction_ranges[i] - self.critical_range))
        return self.percolation_probabilities[closest_idx]


@dataclass 
class SocialRole:
    """Individual social role classification"""
    roles: List[str]
    stats: Dict[str, float]
    
    def is_bridge(self) -> bool:
        return 'bridge' in self.roles
    
    def is_hub(self) -> bool:
        return 'hub' in self.roles
    
    def is_peripheral(self) -> bool:
        return 'peripheral' in self.roles


@dataclass
class AccessibilityResult:
    """Results from accessibility analysis"""
    service_type: str
    service_distance: float
    population_count: int
    service_count: int
    coverage_statistics: Dict[str, float]
    underserved_areas: List[Dict]
    
    def get_coverage_percentage(self) -> float:
        return self.coverage_statistics.get('served_percentage', 0.0)
    
    def get_equity_score(self) -> float:
        """Calculate spatial equity score (higher = more equitable)"""
        served_pct = self.coverage_statistics.get('served_percentage', 0.0)
        well_served_pct = self.coverage_statistics.get('well_served_percentage', 0.0)
        
        # Balance between basic coverage and quality of coverage
        return (served_pct * 0.6 + well_served_pct * 0.4) / 100.0


class PercolationAnalyzer:
    """
    Analyzer for percolation phenomena and critical behavior in spatial networks.
    
    This class provides tools for studying phase transitions, cluster formation,
    and critical thresholds in spatial systems.
    """
    
    def __init__(self, grapher: 'Graphing'):
        self.grapher = grapher
    
    def analyze_percolation_threshold(self, positions: np.ndarray, 
                                    interaction_ranges: List[float]) -> PercolationResult:
        """
        Analyze percolation behavior as a function of interaction range.
        
        Args:
            positions: Array of particle positions in format [id, x, y]
            interaction_ranges: List of interaction ranges to test
            
        Returns:
            PercolationResult with threshold analysis
        """
        largest_cluster_sizes = []
        total_clusters = []
        percolation_probabilities = []
        
        for interaction_range in interaction_ranges:
            try:
                # Create proximity graph
                graph = self.grapher.make_graph("proximity", positions, 
                                              proximity_thresh=interaction_range)
                
                if graph.vcount() > 0:
                    # Find connected components (clusters)
                    clusters = graph.connected_components()
                    cluster_sizes = [len(cluster) for cluster in clusters]
                    
                    largest_cluster = max(cluster_sizes) if cluster_sizes else 0
                    largest_cluster_sizes.append(largest_cluster)
                    total_clusters.append(len(cluster_sizes))
                    
                    # Percolation probability
                    percolation_prob = largest_cluster / len(positions)
                    percolation_probabilities.append(percolation_prob)
                else:
                    largest_cluster_sizes.append(0)
                    total_clusters.append(0)
                    percolation_probabilities.append(0)
                    
            except Exception:
                largest_cluster_sizes.append(0)
                total_clusters.append(0)
                percolation_probabilities.append(0)
        
        # Estimate critical threshold
        critical_range = self._estimate_critical_threshold(
            interaction_ranges, percolation_probabilities
        )
        
        return PercolationResult(
            interaction_ranges=interaction_ranges,
            largest_cluster_sizes=largest_cluster_sizes,
            total_clusters=total_clusters,
            percolation_probabilities=percolation_probabilities,
            critical_range=critical_range
        )
    
    def _estimate_critical_threshold(self, ranges: List[float], 
                                   probabilities: List[float]) -> Optional[float]:
        """Estimate critical percolation threshold from steepest gradient"""
        if not probabilities or max(probabilities) < 0.1:
            return ranges[-1] if ranges else None
        
        # Find steepest increase in percolation probability
        derivatives = np.gradient(probabilities, ranges)
        critical_idx = np.argmax(derivatives)
        
        return ranges[critical_idx]
    
    def detect_phase_transition(self, result: PercolationResult, 
                              gradient_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect phase transition characteristics from percolation results.
        
        Args:
            result: PercolationResult from analyze_percolation_threshold
            gradient_threshold: Minimum gradient to identify transition
            
        Returns:
            Dictionary with phase transition analysis
        """
        if not result.percolation_probabilities:
            return {'has_transition': False}
        
        # Calculate gradient
        gradients = np.gradient(result.percolation_probabilities, result.interaction_ranges)
        max_gradient_idx = np.argmax(gradients)
        max_gradient = gradients[max_gradient_idx]
        
        has_transition = max_gradient > gradient_threshold
        
        return {
            'has_transition': has_transition,
            'transition_range': result.interaction_ranges[max_gradient_idx] if has_transition else None,
            'transition_sharpness': max_gradient,
            'pre_transition_prob': result.percolation_probabilities[max_gradient_idx-1] if max_gradient_idx > 0 else 0,
            'post_transition_prob': result.percolation_probabilities[max_gradient_idx+1] if max_gradient_idx < len(result.percolation_probabilities)-1 else 1
        }


class SocialNetworkAnalyzer:
    """
    Analyzer for social network patterns and role identification.
    
    This class provides tools for identifying social roles, tracking temporal
    changes in network position, and analyzing social dynamics.
    """
    
    def __init__(self, grapher: 'Graphing'):
        self.grapher = grapher
    
    def identify_social_roles(self, graph: ig.Graph, 
                            betweenness_threshold: float = 0.8,
                            degree_threshold: float = 0.8) -> Dict[int, SocialRole]:
        """
        Identify social roles based on network centrality measures.
        
        Args:
            graph: igraph Graph object
            betweenness_threshold: Percentile threshold for bridge identification
            degree_threshold: Percentile threshold for hub identification
            
        Returns:
            Dictionary mapping node IDs to SocialRole objects
        """
        if graph.vcount() == 0:
            return {}
        
        social_roles = {}
        
        try:
            # Calculate centrality measures
            betweenness = graph.betweenness()
            degree = graph.degree()
            
            # Calculate thresholds
            betweenness_cutoff = np.percentile(betweenness, betweenness_threshold * 100)
            degree_cutoff = np.percentile(degree, degree_threshold * 100)
            
            # Assign roles
            for i, vertex in enumerate(graph.vs):
                node_id = vertex["id"] if "id" in vertex.attributes() else i
                
                roles = []
                stats = {
                    'betweenness': betweenness[i],
                    'degree': degree[i]
                }
                
                # Bridge/Broker: High betweenness centrality
                if betweenness[i] >= betweenness_cutoff:
                    roles.append('bridge')
                
                # Hub/Popular: High degree centrality  
                if degree[i] >= degree_cutoff:
                    roles.append('hub')
                
                # Peripheral: Low on all measures
                if (betweenness[i] < np.percentile(betweenness, 20) and
                    degree[i] < np.percentile(degree, 20)):
                    roles.append('peripheral')
                
                social_roles[node_id] = SocialRole(
                    roles=roles if roles else ['regular'],
                    stats=stats
                )
        
        except Exception:
            # Return empty roles if calculation fails
            for i, vertex in enumerate(graph.vs):
                node_id = vertex["id"] if "id" in vertex.attributes() else i
                social_roles[node_id] = SocialRole(
                    roles=['regular'],
                    stats={'betweenness': 0.0, 'degree': 0.0}
                )
        
        return social_roles
    
    def track_temporal_roles(self, graph_sequence: List[ig.Graph]) -> Dict[int, Dict[str, List]]:
        """
        Track how social roles evolve over time.
        
        Args:
            graph_sequence: List of graphs representing temporal sequence
            
        Returns:
            Dictionary with temporal role tracking for each individual
        """
        temporal_roles = {}
        
        for timestep, graph in enumerate(graph_sequence):
            roles = self.identify_social_roles(graph)
            
            for node_id, role in roles.items():
                if node_id not in temporal_roles:
                    temporal_roles[node_id] = {
                        'timesteps': [],
                        'roles': [],
                        'betweenness': [],
                        'degree': []
                    }
                
                temporal_roles[node_id]['timesteps'].append(timestep)
                temporal_roles[node_id]['roles'].append(role.roles)
                temporal_roles[node_id]['betweenness'].append(role.stats['betweenness'])
                temporal_roles[node_id]['degree'].append(role.stats['degree'])
        
        return temporal_roles
    
    def get_role_stability(self, temporal_roles: Dict[int, Dict[str, List]]) -> Dict[int, float]:
        """
        Calculate role stability for each individual.
        
        Args:
            temporal_roles: Output from track_temporal_roles
            
        Returns:
            Dictionary mapping node IDs to stability scores (0-1)
        """
        stability_scores = {}
        
        for node_id, data in temporal_roles.items():
            if len(data['roles']) < 2:
                stability_scores[node_id] = 1.0
                continue
            
            # Count role transitions
            transitions = 0
            prev_roles = set(data['roles'][0])
            
            for current_roles in data['roles'][1:]:
                current_roles_set = set(current_roles)
                if current_roles_set != prev_roles:
                    transitions += 1
                prev_roles = current_roles_set
            
            # Stability = 1 - (transitions / max_possible_transitions)
            max_transitions = len(data['roles']) - 1
            stability_scores[node_id] = 1.0 - (transitions / max_transitions) if max_transitions > 0 else 1.0
        
        return stability_scores


class AccessibilityAnalyzer:
    """
    Analyzer for spatial accessibility and service coverage.
    
    This class provides tools for analyzing urban accessibility, service coverage,
    and spatial equity in service distribution.
    """
    
    def __init__(self, grapher: 'Graphing'):
        self.grapher = grapher
    
    def analyze_service_accessibility(self, population_points: np.ndarray,
                                    service_points: np.ndarray,
                                    service_type: str,
                                    service_distance: float) -> AccessibilityResult:
        """
        Analyze accessibility to urban services.
        
        Args:
            population_points: Array of population locations [id, x, y]
            service_points: Array of service locations [id, x, y]
            service_type: Type of service being analyzed
            service_distance: Maximum acceptable distance to service
            
        Returns:
            AccessibilityResult with coverage analysis
        """
        served_count = 0
        well_served_count = 0
        underserved_areas = []
        
        for pop_point in population_points:
            accessible_services = 0
            
            # Check distance to each service
            for service_point in service_points:
                distance = np.linalg.norm(pop_point[1:3] - service_point[1:3])
                if distance <= service_distance:
                    accessible_services += 1
            
            if accessible_services > 0:
                served_count += 1
            if accessible_services >= 2:
                well_served_count += 1
            
            if accessible_services == 0:
                underserved_areas.append({
                    'position': pop_point[1:3].tolist(),
                    'accessible_services': accessible_services,
                    'id': pop_point[0]
                })
        
        # Coverage statistics
        coverage_statistics = {
            'served_population': served_count,
            'served_percentage': (served_count / len(population_points)) * 100,
            'well_served_population': well_served_count,
            'well_served_percentage': (well_served_count / len(population_points)) * 100,
            'underserved_count': len(underserved_areas)
        }
        
        return AccessibilityResult(
            service_type=service_type,
            service_distance=service_distance,
            population_count=len(population_points),
            service_count=len(service_points),
            coverage_statistics=coverage_statistics,
            underserved_areas=underserved_areas
        )
    
    def compare_accessibility(self, results_list: List[AccessibilityResult]) -> Dict[str, Any]:
        """
        Compare accessibility across multiple services or scenarios.
        
        Args:
            results_list: List of AccessibilityResult objects
            
        Returns:
            Dictionary with comparative analysis
        """
        comparison = {
            'services': [],
            'coverage_percentages': [],
            'equity_scores': [],
            'best_service': None,
            'worst_service': None,
            'average_coverage': 0.0
        }
        
        for result in results_list:
            comparison['services'].append(result.service_type)
            comparison['coverage_percentages'].append(result.get_coverage_percentage())
            comparison['equity_scores'].append(result.get_equity_score())
        
        if comparison['coverage_percentages']:
            best_idx = np.argmax(comparison['coverage_percentages'])
            worst_idx = np.argmin(comparison['coverage_percentages'])
            
            comparison['best_service'] = comparison['services'][best_idx]
            comparison['worst_service'] = comparison['services'][worst_idx]
            comparison['average_coverage'] = np.mean(comparison['coverage_percentages'])
        
        return comparison
    
    def identify_service_gaps(self, accessibility_result: AccessibilityResult,
                            cluster_distance: float = 200.0) -> List[Dict]:
        """
        Identify spatial clusters of underserved areas (service gaps).
        
        Args:
            accessibility_result: Result from analyze_service_accessibility
            cluster_distance: Distance threshold for clustering underserved areas
            
        Returns:
            List of service gap clusters with statistics
        """
        if not accessibility_result.underserved_areas:
            return []
        
        # Extract positions of underserved areas
        underserved_positions = np.array([
            [i] + area['position'] for i, area in enumerate(accessibility_result.underserved_areas)
        ])
        
        try:
            # Create graph of underserved areas
            gap_graph = self.grapher.make_graph("proximity", underserved_positions,
                                              proximity_thresh=cluster_distance)
            
            if gap_graph.vcount() == 0:
                return []
            
            # Find clusters of underserved areas
            clusters = gap_graph.connected_components()
            service_gaps = []
            
            for i, cluster in enumerate(clusters):
                cluster_positions = [underserved_positions[idx][1:3] for idx in cluster]
                cluster_center = np.mean(cluster_positions, axis=0)
                
                service_gaps.append({
                    'gap_id': i,
                    'size': len(cluster),
                    'center': cluster_center.tolist(),
                    'severity': len(cluster) / len(accessibility_result.underserved_areas),
                    'affected_population_ids': [accessibility_result.underserved_areas[idx]['id'] for idx in cluster]
                })
            
            # Sort by severity (largest gaps first)
            service_gaps.sort(key=lambda x: x['severity'], reverse=True)
            
            return service_gaps
            
        except Exception:
            # Fallback: treat each underserved area as individual gap
            return [
                {
                    'gap_id': i,
                    'size': 1,
                    'center': area['position'],
                    'severity': 1 / len(accessibility_result.underserved_areas),
                    'affected_population_ids': [area['id']]
                }
                for i, area in enumerate(accessibility_result.underserved_areas)
            ]


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
        
        # Initialize advanced analyzers
        self._percolation_analyzer = None
        self._social_analyzer = None
        self._accessibility_analyzer = None

    # --- Advanced Analysis Properties ---
    
    @property
    def percolation_analyzer(self) -> PercolationAnalyzer:
        """Get percolation analyzer instance (lazy-loaded)"""
        if self._percolation_analyzer is None:
            self._percolation_analyzer = PercolationAnalyzer(self._grapher)
        return self._percolation_analyzer
    
    @property
    def social_analyzer(self) -> SocialNetworkAnalyzer:
        """Get social network analyzer instance (lazy-loaded)"""
        if self._social_analyzer is None:
            self._social_analyzer = SocialNetworkAnalyzer(self._grapher)
        return self._social_analyzer
    
    @property
    def accessibility_analyzer(self) -> AccessibilityAnalyzer:
        """Get accessibility analyzer instance (lazy-loaded)"""
        if self._accessibility_analyzer is None:
            self._accessibility_analyzer = AccessibilityAnalyzer(self._grapher)
        return self._accessibility_analyzer

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

        For dictionary results, entries with `None` values are filtered out.
        """
        cache_key = f"{metric_name}_{sorted(kwargs.items())}"
        if cache_key in self._metric_cache:
            return self._metric_cache[cache_key]

        result = self._grapher.call_method_safe(self._graph, metric_name, **kwargs)

        # For per-vertex metrics that can fail on some nodes (e.g., pagerank
        # on disconnected graphs), filter out None values for robustness.
        if kwargs.get('return_format') == 'dict' and isinstance(result, dict):
            result = {k: v for k, v in result.items() if v is not None}

        self._metric_cache[cache_key] = result
        return result

    # --- Helper Methods for Common Statistical Tasks ---

    def get_top_n_by(self, metric_name: str, n: int = 5, **kwargs) -> List[tuple]:
        """
        Returns the top N nodes sorted by a given per-vertex metric.
        Handles None values by treating them as the lowest possible value.
        """
        # First, check if the metric_name is a direct vertex attribute.
        if metric_name in self._graph.vs.attributes():
            # It's a vertex attribute. Create the dictionary manually.
            # We use the 'name' attribute which is set to the ID for good summaries.
            ids = self._graph.vs["name"]
            values = self._graph.vs[metric_name]
            metric_dict = dict(zip(ids, values))
        else:
            # If not, assume it's a computable metric (igraph method).
            kwargs['return_format'] = 'dict'
            metric_dict = self.get_metric(metric_name, **kwargs)

        if not isinstance(metric_dict, dict):
            raise TypeError(f"Metric '{metric_name}' did not return a dictionary.")

        # Filter out non-numeric values before sorting for robustness
        valid_items = [
            item for item in metric_dict.items()
            if isinstance(item[1], (int, float, np.number))
        ]

        sorted_items = sorted(
            valid_items,
            key=lambda item: item[1] if item[1] is not None else -float('inf'),
            reverse=True
        )
        return sorted_items[:n]

    def get_metric_stats(self, metric_name: str, **kwargs) -> Dict[str, float]:
        """
        Computes descriptive statistics for a numeric metric.
        Handles None values by ignoring them in the calculation.
        """
        # First, check if the metric_name is a direct vertex attribute.
        if metric_name in self._graph.vs.attributes():
            values = self._graph.vs[metric_name]
        else:
            # If not, assume it's a computable metric (igraph method).
            kwargs['return_format'] = 'list'
            values = self.get_metric(metric_name, **kwargs)

        if not isinstance(values, list):
            raise TypeError(f"Metric '{metric_name}' did not return a list of values.")

        # Ensure we only process numeric types, ignoring strings or other objects.
        numeric_values = [
            v for v in values if v is not None and isinstance(v, (int, float, np.number))
        ]

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
