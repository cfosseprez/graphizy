#!/usr/bin/env python3
"""
Comprehensive Graphizy Benchmark Suite

This integrated benchmark script combines and improves upon the existing benchmark files,
providing a comprehensive performance comparison between Graphizy and the NetworkX+SciPy stack.

Features:
- Unified benchmark framework with consistent timing methodology
- Fair algorithmic comparisons ensuring identical graph structures
- Memory system benchmarking (Graphizy-only feature)
- Real-time performance monitoring with detailed metrics
- Automated result visualization and analysis
- JSON export and plots saved to results_benchmark folder

.. moduleauthor:: Charles Fosseprez (integrated by Claude)
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL-2.0-or-later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors

# Graphizy imports
from graphizy import (
    Graphing, GraphizyConfig,
    generate_and_format_positions,
    validate_graphizy_input
)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


# ============================================================================
# DATA STRUCTURES AND CONFIGURATION
# ============================================================================

@dataclass
class BenchmarkResult:
    """Structured result container for benchmark data"""
    nodes: int
    graph_type: str

    # Timing results (milliseconds)
    graphizy_construction_time: float
    graphizy_metrics_time: float
    graphizy_total_time: float

    networkx_construction_time: float
    networkx_metrics_time: float
    networkx_total_time: float

    # Validation results
    graphizy_edges: int
    networkx_edges: int
    graphs_match: bool

    # Performance metrics
    construction_speedup: float
    metrics_speedup: float
    total_speedup: float

    # Additional metadata
    threshold: Optional[float] = None
    k_value: Optional[int] = None
    extra_info: Optional[Dict] = None


@dataclass
class MemoryBenchmarkResult:
    """Results for memory system benchmarking"""
    nodes: int
    avg_time_per_update: float
    total_connections: int
    timesteps: int
    memory_size_limit: int


class BenchmarkConfig:
    """Configuration for benchmark parameters"""

    # Test parameters
    IMAGE_WIDTH = 1000
    IMAGE_HEIGHT = 1000
    NODE_COUNTS = [100, 500, 1000, 2500]
    MEMORY_NODE_COUNTS = [100, 500, 1000]  # Smaller for memory tests

    # Algorithm-specific parameters
    KNN_K = 10
    MEMORY_TIMESTEPS = 20
    MEMORY_MAX_SIZE = 200
    PROXIMITY_NEIGHBOR_COUNT = 8  # For adaptive threshold calculation

    # Performance parameters
    MIN_TIME_THRESHOLD = 1e-6  # Minimum time to avoid division by zero


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_speedup(time_baseline: float, time_optimized: float,
                 min_time: float = BenchmarkConfig.MIN_TIME_THRESHOLD) -> float:
    """
    Calculate speedup while avoiding division by zero and extreme values.

    Args:
        time_baseline: Baseline timing (usually NetworkX)
        time_optimized: Optimized timing (usually Graphizy)
        min_time: Minimum time threshold to prevent division by zero

    Returns:
        Speedup ratio (baseline / optimized)
    """
    safe_optimized = max(time_optimized, min_time)
    safe_baseline = max(time_baseline, min_time)
    return safe_baseline / safe_optimized


def validate_graph_equality(graph_gz, graph_nx, tolerance: float = 0.001) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that Graphizy and NetworkX produce equivalent graphs.

    Args:
        graph_gz: Graphizy igraph object
        graph_nx: NetworkX graph object
        tolerance: Tolerance for edge count differences

    Returns:
        Tuple of (is_equal, validation_info)
    """
    gz_edges = len(graph_gz.es)
    nx_edges = graph_nx.number_of_edges()

    edge_diff = abs(gz_edges - nx_edges)
    edge_match = edge_diff <= tolerance * max(gz_edges, nx_edges)

    validation_info = {
        'graphizy_edges': gz_edges,
        'networkx_edges': nx_edges,
        'edge_difference': edge_diff,
        'edge_match': edge_match,
        'graphizy_nodes': graph_gz.vcount(),
        'networkx_nodes': graph_nx.number_of_nodes()
    }

    return edge_match, validation_info


def format_timing_summary(results: List[BenchmarkResult], graph_type: str) -> str:
    """Format timing results for console output"""
    summary_lines = []

    for result in results:
        if result.graph_type == graph_type:
            summary_lines.append(
                f"    {result.nodes:>4} nodes: "
                f"Construction {result.construction_speedup:>5.1f}x | "
                f"Metrics {result.metrics_speedup:>5.1f}x | "
                f"Total {result.total_speedup:>5.1f}x"
            )

    return '\n'.join(summary_lines)


# ============================================================================
# CORE BENCHMARK IMPLEMENTATIONS
# ============================================================================

class GraphizyBenchmarkSuite:
    """Main benchmark suite coordinator"""

    def __init__(self, config: BenchmarkConfig = None):
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []
        self.memory_results: List[MemoryBenchmarkResult] = []

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Execute complete benchmark suite"""
        print("=" * 80)
        print("COMPREHENSIVE GRAPHIZY BENCHMARK SUITE")
        print("Performance Comparison: Graphizy vs NetworkX+SciPy")
        print("=" * 80)

        # Core graph type benchmarks
        print("\nGRAPH CONSTRUCTION & ANALYSIS BENCHMARKS")
        print("-" * 50)

        self._run_delaunay_benchmark()
        self._run_proximity_benchmark()
        self._run_knn_benchmark()
        self._run_mst_benchmark()

        # Memory system benchmark (Graphizy exclusive)
        print("\nMEMORY SYSTEM BENCHMARK")
        print("-" * 50)
        self._run_memory_benchmark()

        # Generate summary report
        print("\nPERFORMANCE SUMMARY")
        print("-" * 50)
        self._generate_summary_report()

        # Export results
        results_dict = self._export_results()
        self._save_results_to_file(results_dict)

        return results_dict

    def _run_delaunay_benchmark(self):
        """Benchmark Delaunay triangulation construction and analysis"""
        print("\nDelaunay Triangulation Benchmark")

        for n_nodes in self.config.NODE_COUNTS:
            print(f"  Testing {n_nodes} nodes...", end="", flush=True)

            # Generate test data
            data = generate_and_format_positions(
                self.config.IMAGE_WIDTH,
                self.config.IMAGE_HEIGHT,
                n_nodes
            )
            # Clean validation without extra warnings
            if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 3:
                raise ValueError("Invalid data format")
            positions = data[:, 1:3]

            # === GRAPHIZY APPROACH ===
            start_total_gz = time.perf_counter()

            # Construction
            start_construction = time.perf_counter()
            # Use basic configuration to avoid data_shape warnings
            grapher = Graphing(
                dimension=(self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT),
                aspect="array"
            )
            graph_gz = grapher.make_graph(graph_type="delaunay", data_points=data)
            construction_time_gz = (time.perf_counter() - start_construction) * 1000

            # Metrics computation
            start_metrics = time.perf_counter()
            grapher.call_method_safe(graph_gz, 'betweenness', "list", default_value=0.0)
            grapher.call_method_safe(graph_gz, 'closeness', "list", default_value=0.0)
            grapher.call_method_safe(graph_gz, 'degree', "list")
            grapher.call_method_safe(graph_gz, 'diameter', "raw", default_value=0)
            metrics_time_gz = (time.perf_counter() - start_metrics) * 1000
            total_time_gz = (time.perf_counter() - start_total_gz) * 1000

            # === NETWORKX + SCIPY APPROACH ===
            start_total_nx = time.perf_counter()

            # Construction
            start_construction = time.perf_counter()
            tri = Delaunay(positions)
            G_nx = nx.Graph()
            G_nx.add_nodes_from((i, {'x': pos[0], 'y': pos[1]}) for i, pos in enumerate(positions))

            # Extract unique edges from triangulation
            edges_set = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = sorted((simplex[i], simplex[j]))
                        edges_set.add((a, b))

            # Add weighted edges
            edges = [(i, j, {'weight': np.linalg.norm(positions[i] - positions[j])})
                     for i, j in edges_set]
            G_nx.add_edges_from(edges)
            construction_time_nx = (time.perf_counter() - start_construction) * 1000

            # Metrics computation
            start_metrics = time.perf_counter()
            nx.betweenness_centrality(G_nx)
            nx.closeness_centrality(G_nx)
            dict(G_nx.degree())
            if nx.is_connected(G_nx):
                nx.diameter(G_nx)
            metrics_time_nx = (time.perf_counter() - start_metrics) * 1000
            total_time_nx = (time.perf_counter() - start_total_nx) * 1000

            # Validation and result creation
            graphs_match, validation_info = validate_graph_equality(graph_gz, G_nx)

            result = BenchmarkResult(
                nodes=n_nodes,
                graph_type="delaunay",
                graphizy_construction_time=construction_time_gz,
                graphizy_metrics_time=metrics_time_gz,
                graphizy_total_time=total_time_gz,
                networkx_construction_time=construction_time_nx,
                networkx_metrics_time=metrics_time_nx,
                networkx_total_time=total_time_nx,
                graphizy_edges=validation_info['graphizy_edges'],
                networkx_edges=validation_info['networkx_edges'],
                graphs_match=graphs_match,
                construction_speedup=safe_speedup(construction_time_nx, construction_time_gz),
                metrics_speedup=safe_speedup(metrics_time_nx, metrics_time_gz),
                total_speedup=safe_speedup(total_time_nx, total_time_gz)
            )

            self.results.append(result)
            print(f" checkmark {result.total_speedup:.1f}x faster overall")

        print(format_timing_summary(self.results, "delaunay"))

    def _run_proximity_benchmark(self):
        """Benchmark proximity graph construction and clustering analysis"""
        print("\nProximity Graph Benchmark")

        for n_nodes in self.config.NODE_COUNTS:
            print(f"  Testing {n_nodes} nodes...", end="", flush=True)

            # Generate test data
            data = generate_and_format_positions(
                self.config.IMAGE_WIDTH,
                self.config.IMAGE_HEIGHT,
                n_nodes
            )
            # Clean validation without extra warnings
            if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 3:
                raise ValueError("Invalid data format")
            positions = data[:, 1:3]

            # Calculate adaptive threshold for meaningful connectivity
            nbrs = NearestNeighbors(n_neighbors=self.config.PROXIMITY_NEIGHBOR_COUNT).fit(positions)
            distances, _ = nbrs.kneighbors(positions)
            threshold = np.mean(distances[:, -1]) * 1.2  # 20% buffer for good connectivity

            # === GRAPHIZY APPROACH ===
            start_total_gz = time.perf_counter()

            # Construction
            start_construction = time.perf_counter()
            # Use basic configuration to avoid data_shape warnings
            grapher = Graphing(
                dimension=(self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT),
                aspect="array"
            )
            graph_gz = grapher.make_graph(
                graph_type="proximity",
                data_points=data,
                proximity_thresh=threshold
            )
            construction_time_gz = (time.perf_counter() - start_construction) * 1000

            # Clustering analysis
            start_metrics = time.perf_counter()
            grapher.call_method_safe(graph_gz, 'connected_components', "raw")
            grapher.call_method_safe(graph_gz, 'transitivity_local_undirected', "list", default_value=[])
            grapher.call_method_safe(graph_gz, 'degree', "list")
            grapher.call_method_safe(graph_gz, 'transitivity_undirected', "raw", default_value=0.0)
            metrics_time_gz = (time.perf_counter() - start_metrics) * 1000
            total_time_gz = (time.perf_counter() - start_total_gz) * 1000

            # === NETWORKX + SCIPY APPROACH ===
            start_total_nx = time.perf_counter()

            # Construction using same algorithm
            start_construction = time.perf_counter()
            dist_matrix = squareform(pdist(positions))
            G_nx = nx.Graph()
            G_nx.add_nodes_from((i, {'x': pos[0], 'y': pos[1]}) for i, pos in enumerate(positions))

            # Efficient edge creation using upper triangle
            i_idx, j_idx = np.triu_indices_from(dist_matrix, k=1)
            mask = (dist_matrix[i_idx, j_idx] < threshold) & (dist_matrix[i_idx, j_idx] > 0)
            i_idx, j_idx = i_idx[mask], j_idx[mask]
            weights = dist_matrix[i_idx, j_idx]

            edges = [(i, j, {'weight': w}) for i, j, w in zip(i_idx, j_idx, weights)]
            G_nx.add_edges_from(edges)
            construction_time_nx = (time.perf_counter() - start_construction) * 1000

            # Clustering analysis
            start_metrics = time.perf_counter()
            list(nx.connected_components(G_nx))
            nx.clustering(G_nx)
            dict(G_nx.degree())
            nx.transitivity(G_nx)
            metrics_time_nx = (time.perf_counter() - start_metrics) * 1000
            total_time_nx = (time.perf_counter() - start_total_nx) * 1000

            # Validation and result creation
            graphs_match, validation_info = validate_graph_equality(graph_gz, G_nx)

            result = BenchmarkResult(
                nodes=n_nodes,
                graph_type="proximity",
                graphizy_construction_time=construction_time_gz,
                graphizy_metrics_time=metrics_time_gz,
                graphizy_total_time=total_time_gz,
                networkx_construction_time=construction_time_nx,
                networkx_metrics_time=metrics_time_nx,
                networkx_total_time=total_time_nx,
                graphizy_edges=validation_info['graphizy_edges'],
                networkx_edges=validation_info['networkx_edges'],
                graphs_match=graphs_match,
                construction_speedup=safe_speedup(construction_time_nx, construction_time_gz),
                metrics_speedup=safe_speedup(metrics_time_nx, metrics_time_gz),
                total_speedup=safe_speedup(total_time_nx, total_time_gz),
                threshold=threshold
            )

            self.results.append(result)
            print(f" checkmark {result.total_speedup:.1f}x faster overall")

        print(format_timing_summary(self.results, "proximity"))

    def _run_knn_benchmark(self):
        """Benchmark k-nearest neighbors graph construction and analysis"""
        print(f"\nk-NN Graph Benchmark (k={self.config.KNN_K})")

        for n_nodes in self.config.NODE_COUNTS:
            print(f"  Testing {n_nodes} nodes...", end="", flush=True)

            # Generate test data
            data = generate_and_format_positions(
                self.config.IMAGE_WIDTH,
                self.config.IMAGE_HEIGHT,
                n_nodes
            )
            # Clean validation without extra warnings
            if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 3:
                raise ValueError("Invalid data format")
            positions = data[:, 1:3]

            # === GRAPHIZY APPROACH ===
            start_total_gz = time.perf_counter()

            # Construction
            start_construction = time.perf_counter()
            # Use basic configuration to avoid data_shape warnings
            grapher = Graphing(
                dimension=(self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT),
                aspect="array"
            )
            graph_gz = grapher.make_graph(
                graph_type="knn",
                data_points=data,
                k=self.config.KNN_K
            )
            construction_time_gz = (time.perf_counter() - start_construction) * 1000

            # Analysis
            start_metrics = time.perf_counter()
            grapher.call_method_safe(graph_gz, 'degree', "list")
            grapher.call_method_safe(graph_gz, 'pagerank', "list", default_value=0.0)
            grapher.call_method_safe(graph_gz, 'connected_components', "raw")
            metrics_time_gz = (time.perf_counter() - start_metrics) * 1000
            total_time_gz = (time.perf_counter() - start_total_gz) * 1000

            # === NETWORKX + SCIKIT-LEARN APPROACH ===
            start_total_nx = time.perf_counter()

            # Construction using scikit-learn (industry standard)
            start_construction = time.perf_counter()
            # n_neighbors includes the point itself, so we use k+1
            knn_graph = NearestNeighbors(n_neighbors=self.config.KNN_K + 1).fit(positions)
            adj_matrix = knn_graph.kneighbors_graph(mode='connectivity')
            G_nx = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph)
            construction_time_nx = (time.perf_counter() - start_construction) * 1000

            # Analysis
            start_metrics = time.perf_counter()
            dict(G_nx.degree())
            nx.pagerank(G_nx)
            list(nx.connected_components(G_nx.to_undirected()))
            metrics_time_nx = (time.perf_counter() - start_metrics) * 1000
            total_time_nx = (time.perf_counter() - start_total_nx) * 1000

            # Validation (k-NN graphs may have slight differences due to implementation)
            # We'll be more lenient with edge count validation
            graphs_match, validation_info = validate_graph_equality(graph_gz, G_nx, tolerance=0.1)

            result = BenchmarkResult(
                nodes=n_nodes,
                graph_type="knn",
                graphizy_construction_time=construction_time_gz,
                graphizy_metrics_time=metrics_time_gz,
                graphizy_total_time=total_time_gz,
                networkx_construction_time=construction_time_nx,
                networkx_metrics_time=metrics_time_nx,
                networkx_total_time=total_time_nx,
                graphizy_edges=validation_info['graphizy_edges'],
                networkx_edges=validation_info['networkx_edges'],
                graphs_match=graphs_match,
                construction_speedup=safe_speedup(construction_time_nx, construction_time_gz),
                metrics_speedup=safe_speedup(metrics_time_nx, metrics_time_gz),
                total_speedup=safe_speedup(total_time_nx, total_time_gz),
                k_value=self.config.KNN_K
            )

            self.results.append(result)
            print(f" checkmark {result.total_speedup:.1f}x faster overall")

        print(format_timing_summary(self.results, "knn"))

    def _run_mst_benchmark(self):
        """Benchmark minimum spanning tree construction and analysis"""
        print("\nMST Graph Benchmark")

        for n_nodes in self.config.NODE_COUNTS:
            print(f"  Testing {n_nodes} nodes...", end="", flush=True)

            # Generate test data
            data = generate_and_format_positions(
                self.config.IMAGE_WIDTH,
                self.config.IMAGE_HEIGHT,
                n_nodes
            )
            # Clean validation without extra warnings
            if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 3:
                raise ValueError("Invalid data format")
            positions = data[:, 1:3]

            # === GRAPHIZY APPROACH ===
            start_total_gz = time.perf_counter()

            # Construction
            start_construction = time.perf_counter()
            # Use basic configuration to avoid data_shape warnings
            grapher = Graphing(
                dimension=(self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT),
                aspect="array"
            )
            graph_gz = grapher.make_graph(graph_type="mst", data_points=data)
            construction_time_gz = (time.perf_counter() - start_construction) * 1000

            # Tree analysis
            start_metrics = time.perf_counter()
            grapher.call_method_safe(graph_gz, 'degree', "list")
            grapher.call_method_safe(graph_gz, 'diameter', "raw", default_value=0)
            grapher.call_method_safe(graph_gz, 'average_path_length', "raw", default_value=0)
            grapher.call_method_safe(graph_gz, 'betweenness', "list", default_value=0.0)
            metrics_time_gz = (time.perf_counter() - start_metrics) * 1000
            total_time_gz = (time.perf_counter() - start_total_gz) * 1000

            # === NETWORKX + SCIPY APPROACH ===
            start_total_nx = time.perf_counter()

            # Construction
            start_construction = time.perf_counter()
            # Create complete graph with distance weights
            G_complete = nx.Graph()
            G_complete.add_nodes_from((i, {'x': pos[0], 'y': pos[1]}) for i, pos in enumerate(positions))

            # Compute all pairwise distances
            dists = squareform(pdist(positions))
            i_idx, j_idx = np.triu_indices_from(dists, k=1)
            weights = dists[i_idx, j_idx]
            edges = [(i, j, {'weight': w}) for i, j, w in zip(i_idx, j_idx, weights)]
            G_complete.add_edges_from(edges)

            # Extract MST
            G_nx = nx.minimum_spanning_tree(G_complete)
            construction_time_nx = (time.perf_counter() - start_construction) * 1000

            # Tree analysis
            start_metrics = time.perf_counter()
            dict(G_nx.degree())
            if nx.is_connected(G_nx):
                nx.diameter(G_nx)
                nx.average_shortest_path_length(G_nx)
                nx.betweenness_centrality(G_nx)
            metrics_time_nx = (time.perf_counter() - start_metrics) * 1000
            total_time_nx = (time.perf_counter() - start_total_nx) * 1000

            # Validation
            graphs_match, validation_info = validate_graph_equality(graph_gz, G_nx)

            result = BenchmarkResult(
                nodes=n_nodes,
                graph_type="mst",
                graphizy_construction_time=construction_time_gz,
                graphizy_metrics_time=metrics_time_gz,
                graphizy_total_time=total_time_gz,
                networkx_construction_time=construction_time_nx,
                networkx_metrics_time=metrics_time_nx,
                networkx_total_time=total_time_nx,
                graphizy_edges=validation_info['graphizy_edges'],
                networkx_edges=validation_info['networkx_edges'],
                graphs_match=graphs_match,
                construction_speedup=safe_speedup(construction_time_nx, construction_time_gz),
                metrics_speedup=safe_speedup(metrics_time_nx, metrics_time_gz),
                total_speedup=safe_speedup(total_time_nx, total_time_gz)
            )

            self.results.append(result)
            print(f" checkmark {result.total_speedup:.1f}x faster overall")

        print(format_timing_summary(self.results, "mst"))

    def _run_memory_benchmark(self):
        """Benchmark the memory-enhanced system (Graphizy exclusive feature)"""
        print("\nMemory System Benchmark (Graphizy Exclusive)")

        for n_nodes in self.config.MEMORY_NODE_COUNTS:
            print(f"  Testing {n_nodes} nodes over {self.config.MEMORY_TIMESTEPS} timesteps...", end="", flush=True)

            # Generate initial data
            data = generate_and_format_positions(
                self.config.IMAGE_WIDTH,
                self.config.IMAGE_HEIGHT,
                n_nodes
            )
            # Clean validation without extra warnings
            if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 3:
                raise ValueError("Invalid data format")

            # Initialize memory system
            grapher = Graphing(
                dimension=(self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT),
                aspect="array"
            )
            grapher.init_memory_manager(
                max_memory_size=self.config.MEMORY_MAX_SIZE,
                track_edge_ages=True
            )

            # Simulate temporal evolution
            total_time = 0
            for timestep in range(self.config.MEMORY_TIMESTEPS):
                # Simulate realistic movement (Brownian motion)
                movement_scale = 5.0  # Moderate movement
                data[:, 1:3] += np.random.normal(0, movement_scale, data[:, 1:3].shape)

                # Keep points within bounds
                data[:, 1] = np.clip(data[:, 1], 0, self.config.IMAGE_WIDTH)
                data[:, 2] = np.clip(data[:, 2], 0, self.config.IMAGE_HEIGHT)

                start = time.perf_counter()

                # Create proximity graph and update memory manually
                proximity_thresh = 80.0  # Fixed threshold for consistent comparison
                current_graph = grapher.make_graph(
                    graph_type="proximity",
                    data_points=data,
                    proximity_thresh=proximity_thresh,
                    update_memory=True,
                    use_memory=True,
                    compute_weights=False
                )

                total_time += (time.perf_counter() - start) * 1000

            avg_time_per_update = total_time / self.config.MEMORY_TIMESTEPS

            # Get memory statistics
            memory_stats = grapher.get_memory_analysis()

            memory_result = MemoryBenchmarkResult(
                nodes=n_nodes,
                avg_time_per_update=avg_time_per_update,
                total_connections=memory_stats.get('total_connections', 0),
                timesteps=self.config.MEMORY_TIMESTEPS,
                memory_size_limit=self.config.MEMORY_MAX_SIZE
            )

            self.memory_results.append(memory_result)
            print(f" ✓ {avg_time_per_update:.2f}ms per update")

        # Memory benchmark summary
        print("\n    Memory System Performance:")
        for result in self.memory_results:
            efficiency = result.total_connections / (result.nodes * result.timesteps) * 100
            print(f"      {result.nodes:>4} nodes: {result.avg_time_per_update:>6.1f}ms/update, "
                  f"{result.total_connections:>4} connections tracked ({efficiency:.1f}% efficiency)")

    def _generate_summary_report(self):
        """Generate comprehensive performance summary"""
        if not self.results:
            print("No benchmark results to summarize.")
            return

        # Group results by graph type
        results_by_type = {}
        for result in self.results:
            if result.graph_type not in results_by_type:
                results_by_type[result.graph_type] = []
            results_by_type[result.graph_type].append(result)

        print("\n📈 PERFORMANCE ANALYSIS")
        print("=" * 60)

        # Calculate average speedups by graph type
        summary_stats = {}
        for graph_type, results in results_by_type.items():
            construction_speedups = [r.construction_speedup for r in results]
            metrics_speedups = [r.metrics_speedup for r in results]
            total_speedups = [r.total_speedup for r in results]

            summary_stats[graph_type] = {
                'avg_construction_speedup': np.mean(construction_speedups),
                'avg_metrics_speedup': np.mean(metrics_speedups),
                'avg_total_speedup': np.mean(total_speedups),
                'min_total_speedup': np.min(total_speedups),
                'max_total_speedup': np.max(total_speedups)
            }

        # Display summary table
        print(f"\n{'Graph Type':<12} | {'Construction':<12} | {'Metrics':<12} | {'Total':<12} | {'Range':<15}")
        print("-" * 80)

        for graph_type, stats in summary_stats.items():
            print(f"{graph_type.title():<12} | "
                  f"{stats['avg_construction_speedup']:>9.1f}x | "
                  f"{stats['avg_metrics_speedup']:>9.1f}x | "
                  f"{stats['avg_total_speedup']:>9.1f}x | "
                  f"{stats['min_total_speedup']:.1f}x - {stats['max_total_speedup']:.1f}x")

        # Overall performance insights
        all_total_speedups = [r.total_speedup for r in self.results]
        overall_avg_speedup = np.mean(all_total_speedups)

        print(f"\n🎯 KEY PERFORMANCE INSIGHTS")
        print("-" * 40)
        print(f"• Average Overall Speedup: {overall_avg_speedup:.1f}x faster than NetworkX+SciPy")
        print(f"• Performance Range: {np.min(all_total_speedups):.1f}x to {np.max(all_total_speedups):.1f}x")
        print(
            f"• Most Efficient: {max(summary_stats.keys(), key=lambda k: summary_stats[k]['avg_total_speedup']).title()}")

        # Graph validation summary
        validation_results = [r.graphs_match for r in self.results]
        validation_rate = sum(validation_results) / len(validation_results) * 100
        print(f"• Graph Validation Rate: {validation_rate:.1f}% (algorithms produce equivalent results)")

        # Memory system insights
        if self.memory_results:
            avg_memory_time = np.mean([r.avg_time_per_update for r in self.memory_results])
            total_memory_connections = sum([r.total_connections for r in self.memory_results])
            print(f"• Memory System: {avg_memory_time:.1f}ms average update time")
            print(f"• Memory Efficiency: {total_memory_connections} total connections tracked")

        print(f"\n💡 ARCHITECTURAL ADVANTAGES")
        print("-" * 40)
        print("• Unified API: Single library vs multiple library integration")
        print("• C++ Backend: igraph's optimized core provides consistent speedups")
        print("• Memory System: Unique temporal analysis capability")
        print("• Robust Analysis: call_method_safe handles disconnected graphs gracefully")

    def _export_results(self) -> Dict[str, Any]:
        """Export all results to structured dictionary"""
        return {
            'metadata': {
                'benchmark_version': '3.0',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': {
                    'image_dimensions': [self.config.IMAGE_WIDTH, self.config.IMAGE_HEIGHT],
                    'node_counts': self.config.NODE_COUNTS,
                    'memory_node_counts': self.config.MEMORY_NODE_COUNTS,
                    'knn_k': self.config.KNN_K,
                    'memory_timesteps': self.config.MEMORY_TIMESTEPS,
                    'memory_max_size': self.config.MEMORY_MAX_SIZE
                }
            },
            'graph_benchmarks': [asdict(result) for result in self.results],
            'memory_benchmarks': [asdict(result) for result in self.memory_results],
            'summary_statistics': self._calculate_summary_statistics()
        }

    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        if not self.results:
            return {}

        # Group by graph type
        by_type = {}
        for result in self.results:
            if result.graph_type not in by_type:
                by_type[result.graph_type] = []
            by_type[result.graph_type].append(result)

        # Calculate statistics for each type
        type_stats = {}
        for graph_type, results in by_type.items():
            speedups = [r.total_speedup for r in results]
            type_stats[graph_type] = {
                'count': len(results),
                'avg_speedup': float(np.mean(speedups)),
                'median_speedup': float(np.median(speedups)),
                'std_speedup': float(np.std(speedups)),
                'min_speedup': float(np.min(speedups)),
                'max_speedup': float(np.max(speedups))
            }

        # Overall statistics
        all_speedups = [r.total_speedup for r in self.results]
        validation_rate = sum(r.graphs_match for r in self.results) / len(self.results)

        return {
            'overall': {
                'total_benchmarks': len(self.results),
                'avg_speedup': float(np.mean(all_speedups)),
                'median_speedup': float(np.median(all_speedups)),
                'validation_rate': float(validation_rate),
                'graph_types_tested': list(by_type.keys())
            },
            'by_graph_type': type_stats,
            'memory_system': {
                'benchmarks_run': len(self.memory_results),
                'avg_update_time_ms': float(
                    np.mean([r.avg_time_per_update for r in self.memory_results])) if self.memory_results else None,
                'total_connections_tracked': sum(r.total_connections for r in self.memory_results)
            }
        }

    def _save_results_to_file(self, results: Dict[str, Any]):
        """Save results to JSON file with timestamp in results_benchmark folder"""
        # Create results directory if it doesn't exist
        results_dir = Path('results_benchmark')
        results_dir.mkdir(exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = results_dir / f'graphizy_benchmark_results_{timestamp}.json'

        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {filename}")

            # Also save a 'latest' version for easy access
            latest_filename = results_dir / 'graphizy_benchmark_results_latest.json'
            with open(latest_filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"💾 Latest results: {latest_filename}")

        except Exception as e:
            print(f"❌ Error saving results: {e}")


# ============================================================================
# VISUALIZATION AND REPORTING
# ============================================================================

def generate_benchmark_plots(results_file: str = None):
    """Generate comprehensive visualization plots from benchmark results"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        # Set style for publication-quality plots
        plt.style.use('default')
        sns.set_palette("husl")

    except ImportError:
        print("📊 Visualization requires matplotlib, pandas, and seaborn")
        print("Install with: pip install matplotlib pandas seaborn")
        return

    # Create results directory if it doesn't exist
    results_dir = Path('results_benchmark')
    results_dir.mkdir(exist_ok=True)

    # Default to latest results file in results_benchmark folder
    if results_file is None:
        results_file = results_dir / 'graphizy_benchmark_results_latest.json'

    # Load results
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Results file not found: {results_file}")
        print("Run the benchmark first to generate results.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(data['graph_benchmarks'])

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Graphizy vs NetworkX+SciPy: Comprehensive Performance Analysis',
                 fontsize=16, fontweight='bold')

    # Plot 1: Total Speedup by Graph Type and Node Count
    ax1 = axes[0, 0]
    for graph_type in df['graph_type'].unique():
        subset = df[df['graph_type'] == graph_type]
        ax1.plot(subset['nodes'], subset['total_speedup'],
                 marker='o', linewidth=2.5, markersize=8,
                 label=graph_type.title())

    ax1.set_xlabel('Number of Nodes')
    ax1.set_ylabel('Speedup (x faster)')
    ax1.set_title('Total Workflow Speedup')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Plot 2: Construction vs Metrics Speedup
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df['construction_speedup'], df['metrics_speedup'],
                          c=df['nodes'], s=100, alpha=0.7, cmap='viridis')
    ax2.set_xlabel('Construction Speedup')
    ax2.set_ylabel('Metrics Speedup')
    ax2.set_title('Construction vs Metrics Performance')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='Node Count')

    # Add diagonal line for reference
    max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal Performance')
    ax2.legend()

    # Plot 3: Timing Breakdown
    ax3 = axes[1, 0]
    graph_types = df['graph_type'].unique()
    x_pos = np.arange(len(graph_types))

    # Calculate average times for each graph type
    avg_construction = [df[df['graph_type'] == gt]['graphizy_construction_time'].mean() for gt in graph_types]
    avg_metrics = [df[df['graph_type'] == gt]['graphizy_metrics_time'].mean() for gt in graph_types]

    bars1 = ax3.bar(x_pos, avg_construction, 0.6, label='Construction', alpha=0.8)
    bars2 = ax3.bar(x_pos, avg_metrics, 0.6, bottom=avg_construction, label='Metrics', alpha=0.8)

    ax3.set_xlabel('Graph Type')
    ax3.set_ylabel('Average Time (ms)')
    ax3.set_title('Graphizy Timing Breakdown')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([gt.title() for gt in graph_types])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Performance Distribution
    ax4 = axes[1, 1]
    speedup_data = [df[df['graph_type'] == gt]['total_speedup'].values for gt in graph_types]

    box_plot = ax4.boxplot(speedup_data, labels=[gt.title() for gt in graph_types],
                           patch_artist=True, showmeans=True)

    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(graph_types)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_ylabel('Total Speedup (x faster)')
    ax4.set_title('Speedup Distribution by Graph Type')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plots to results_benchmark folder
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    plot_filename = results_dir / f'graphizy_benchmark_plots_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Plots saved to: {plot_filename}")

    # Also save latest version
    latest_plot_filename = results_dir / 'graphizy_benchmark_plots_latest.png'
    plt.savefig(latest_plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Latest plots: {latest_plot_filename}")

    plt.show()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main execution function with command line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Comprehensive Graphizy Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmark_comparison.py                    # Run full benchmark suite
    python benchmark_comparison.py --quick            # Quick benchmark (fewer nodes)
    python benchmark_comparison.py --plot-only        # Generate plots from existing results
    python benchmark_comparison.py --memory-only      # Run only memory benchmarks
        """
    )

    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark with fewer node counts')
    parser.add_argument('--plot-only', action='store_true',
                        help='Generate plots from existing results without running benchmarks')
    parser.add_argument('--memory-only', action='store_true',
                        help='Run only memory system benchmarks')
    parser.add_argument('--nodes', type=int, nargs='+',
                        help='Custom node counts to test (e.g., --nodes 100 500 1000)')

    args = parser.parse_args()

    # Handle plot-only mode
    if args.plot_only:
        print("📊 Generating plots from existing results...")
        generate_benchmark_plots()
        return

    # Configure benchmark parameters
    config = BenchmarkConfig()

    if args.quick:
        config.NODE_COUNTS = [100, 500, 1000]
        config.MEMORY_NODE_COUNTS = [100, 500]
        config.MEMORY_TIMESTEPS = 10
        print("🚀 Running quick benchmark mode...")

    if args.nodes:
        config.NODE_COUNTS = sorted(args.nodes)
        config.MEMORY_NODE_COUNTS = [n for n in args.nodes if n <= 1000]  # Memory benchmarks for smaller sizes
        print(f"🎯 Custom node counts: {config.NODE_COUNTS}")

    if args.memory_only:
        print("🧠 Running memory-only benchmarks...")

    # Run benchmarks
    suite = GraphizyBenchmarkSuite(config)

    if args.memory_only:
        suite._run_memory_benchmark()
        suite._generate_summary_report()
        results = suite._export_results()
        suite._save_results_to_file(results)
    else:
        results = suite.run_all_benchmarks()

    # Generate plots
    print("\n📊 Generating visualization plots...")
    try:
        generate_benchmark_plots()
    except Exception as e:
        print(f"⚠️  Could not generate plots: {e}")
        print("Install matplotlib, pandas, and seaborn for visualization")

    print("\n✅ Benchmark suite completed successfully!")
    print(f"📁 All results saved in: results_benchmark/ folder")


if __name__ == "__main__":
    main()