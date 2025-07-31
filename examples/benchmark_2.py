# C:/Users/nakag/Desktop/code/graphizy/examples/benchmark_comparison.py
"""
Comprehensive spatial benchmark comparing Graphizy vs. the NetworkX + SciPy/Scikit-learn stack.
This script provides a fair, "apples-to-apples" comparison for common spatial graph
workflows, timing both the initial graph construction and subsequent analysis metrics.

Version 2.0:
- Added k-Nearest Neighbors (k-NN) benchmark.
- Made the script self-contained by including helper functions.
- Refined output for clarity.
"""

import time
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from graphizy import Graphing


# ============================================================================
# HELPER FUNCTIONS (to make the script self-contained)
# ============================================================================

def generate_and_format_positions(size_x: int, size_y: int, num_particles: int) -> np.ndarray:
    """Generates random particle data in the format Graphizy expects."""
    ids = np.arange(num_particles).reshape(-1, 1)
    positions = np.random.rand(num_particles, 2) * [size_x, size_y]
    return np.hstack([ids, positions])


def validate_graphizy_input(data: np.ndarray):
    """Basic validation for Graphizy input data."""
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Input data must be a NumPy array with at least 3 columns [id, x, y].")


def safe_speedup(time_nx: float, time_graphizy: float, min_time: float = 1e-6) -> float:
    """Calculate speedup while avoiding division by zero for very fast operations."""
    # Use a small minimum time to prevent astronomical speedups from rounding errors
    safe_graphizy = max(time_graphizy, min_time)
    safe_nx = max(time_nx, min_time)
    return safe_nx / safe_graphizy


# ============================================================================
# BENCHMARK DEFINITIONS
# ============================================================================

def benchmark_delaunay_triangulation(n_nodes: int, width: int, height: int) -> dict:
    """Compare Delaunay triangulation construction and analysis."""
    print(f"\n--- Benchmarking Delaunay Triangulation ({n_nodes} nodes) ---")
    data = generate_and_format_positions(width, height, n_nodes)
    positions = data[:, 1:3]

    # === GRAPHIZY APPROACH ===
    start_total_gz = time.perf_counter()
    grapher = Graphing()
    graph_gz = grapher.make_graph(graph_type="delaunay", data_points=data)
    construction_time_gz = (time.perf_counter() - start_total_gz) * 1000

    start_metrics_gz = time.perf_counter()
    grapher.call_method_safe(graph_gz, 'betweenness', "list", default_value=0.0)
    grapher.call_method_safe(graph_gz, 'closeness', "list", default_value=0.0)
    grapher.call_method_safe(graph_gz, 'diameter', "raw", default_value=0)
    metrics_time_gz = (time.perf_counter() - start_metrics_gz) * 1000
    total_time_gz = (time.perf_counter() - start_total_gz) * 1000

    # === NETWORKX + SCIPY APPROACH ===
    start_total_nx = time.perf_counter()
    tri = Delaunay(positions)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edges.add(tuple(sorted((simplex[i], simplex[j]))))
    G_nx = nx.Graph()
    G_nx.add_nodes_from(range(n_nodes))
    G_nx.add_edges_from(list(edges))
    construction_time_nx = (time.perf_counter() - start_total_nx) * 1000

    start_metrics_nx = time.perf_counter()
    nx.betweenness_centrality(G_nx)
    nx.closeness_centrality(G_nx)
    if nx.is_connected(G_nx):
        nx.diameter(G_nx)
    metrics_time_nx = (time.perf_counter() - start_metrics_nx) * 1000
    total_time_nx = (time.perf_counter() - start_total_nx) * 1000

    print(f"  Construction: Graphizy {construction_time_gz:.2f}ms vs NetworkX {construction_time_nx:.2f}ms "
          f"(Speedup: {safe_speedup(construction_time_nx, construction_time_gz):.1f}x)")
    print(f"  Metrics:      Graphizy {metrics_time_gz:.2f}ms vs NetworkX {metrics_time_nx:.2f}ms "
          f"(Speedup: {safe_speedup(metrics_time_nx, metrics_time_gz):.1f}x)")

    return {
        'nodes': n_nodes, 'type': 'Delaunay',
        'construction_speedup': safe_speedup(construction_time_nx, construction_time_gz),
        'metrics_speedup': safe_speedup(metrics_time_nx, metrics_time_gz),
        'total_speedup': safe_speedup(total_time_nx, total_time_gz)
    }


def benchmark_proximity_graphs(n_nodes: int, width: int, height: int) -> dict:
    """Compare proximity graph construction and analysis."""
    print(f"\n--- Benchmarking Proximity Graph ({n_nodes} nodes) ---")
    data = generate_and_format_positions(width, height, n_nodes)
    positions = data[:, 1:3]

    # Use an adaptive threshold for meaningful connectivity
    nbrs = NearestNeighbors(n_neighbors=8).fit(positions)
    distances, _ = nbrs.kneighbors(positions)
    threshold = np.mean(distances[:, -1])

    # === GRAPHIZY APPROACH ===
    start_total_gz = time.perf_counter()
    grapher = Graphing()
    graph_gz = grapher.make_graph(graph_type="proximity", data_points=data, proximity_thresh=threshold)
    construction_time_gz = (time.perf_counter() - start_total_gz) * 1000

    start_metrics_gz = time.perf_counter()
    grapher.call_method_safe(graph_gz, 'connected_components', "raw")
    grapher.call_method_safe(graph_gz, 'transitivity_local_undirected', "list", default_value=[])
    metrics_time_gz = (time.perf_counter() - start_metrics_gz) * 1000
    total_time_gz = (time.perf_counter() - start_total_gz) * 1000

    # === NETWORKX + SCIPY APPROACH ===
    start_total_nx = time.perf_counter()
    dist_matrix = squareform(pdist(positions))
    adj_matrix = dist_matrix < threshold
    np.fill_diagonal(adj_matrix, False)
    G_nx = nx.from_numpy_array(adj_matrix)
    construction_time_nx = (time.perf_counter() - start_total_nx) * 1000

    start_metrics_nx = time.perf_counter()
    list(nx.connected_components(G_nx))
    nx.clustering(G_nx)
    metrics_time_nx = (time.perf_counter() - start_metrics_nx) * 1000
    total_time_nx = (time.perf_counter() - start_total_nx) * 1000

    print(f"  Construction: Graphizy {construction_time_gz:.2f}ms vs NetworkX {construction_time_nx:.2f}ms "
          f"(Speedup: {safe_speedup(construction_time_nx, construction_time_gz):.1f}x)")
    print(f"  Metrics:      Graphizy {metrics_time_gz:.2f}ms vs NetworkX {metrics_time_nx:.2f}ms "
          f"(Speedup: {safe_speedup(metrics_time_nx, metrics_time_gz):.1f}x)")

    return {
        'nodes': n_nodes, 'type': 'Proximity',
        'construction_speedup': safe_speedup(construction_time_nx, construction_time_gz),
        'metrics_speedup': safe_speedup(metrics_time_nx, metrics_time_gz),
        'total_speedup': safe_speedup(total_time_nx, total_time_gz)
    }


def benchmark_knn_graphs(n_nodes: int, width: int, height: int, k: int = 10) -> dict:
    """(NEW) Compare k-NN graph construction and analysis."""
    print(f"\n--- Benchmarking k-NN Graph ({n_nodes} nodes, k={k}) ---")
    data = generate_and_format_positions(width, height, n_nodes)
    positions = data[:, 1:3]

    # === GRAPHIZY APPROACH ===
    start_total_gz = time.perf_counter()
    grapher = Graphing()
    graph_gz = grapher.make_graph(graph_type="knn", data_points=data, k=k)
    construction_time_gz = (time.perf_counter() - start_total_gz) * 1000

    start_metrics_gz = time.perf_counter()
    grapher.call_method_safe(graph_gz, 'degree', "list")
    grapher.call_method_safe(graph_gz, 'pagerank', "list", default_value=0.0)
    metrics_time_gz = (time.perf_counter() - start_metrics_gz) * 1000
    total_time_gz = (time.perf_counter() - start_total_gz) * 1000

    # === NETWORKX + SCIKIT-LEARN APPROACH ===
    start_total_nx = time.perf_counter()
    # Scikit-learn is the standard for fast k-NN searches
    # n_neighbors includes the point itself, so we use k+1
    adj_matrix = NearestNeighbors(n_neighbors=k + 1).fit(positions).kneighbors_graph(mode='connectivity')
    G_nx = nx.from_scipy_sparse_array(adj_matrix, create_using=nx.DiGraph)
    construction_time_nx = (time.perf_counter() - start_total_nx) * 1000

    start_metrics_nx = time.perf_counter()
    dict(G_nx.degree())
    nx.pagerank(G_nx)
    metrics_time_nx = (time.perf_counter() - start_metrics_nx) * 1000
    total_time_nx = (time.perf_counter() - start_total_nx) * 1000

    print(f"  Construction: Graphizy {construction_time_gz:.2f}ms vs Scikit-learn {construction_time_nx:.2f}ms "
          f"(Speedup: {safe_speedup(construction_time_nx, construction_time_gz):.1f}x)")
    print(f"  Metrics:      Graphizy {metrics_time_gz:.2f}ms vs NetworkX {metrics_time_nx:.2f}ms "
          f"(Speedup: {safe_speedup(metrics_time_nx, metrics_time_gz):.1f}x)")

    return {
        'nodes': n_nodes, 'type': 'k-NN',
        'construction_speedup': safe_speedup(construction_time_nx, construction_time_gz),
        'metrics_speedup': safe_speedup(metrics_time_nx, metrics_time_gz),
        'total_speedup': safe_speedup(total_time_nx, total_time_gz)
    }


def benchmark_memory_system(n_nodes: int, width: int, height: int) -> dict:
    """Benchmark the unique memory-enhanced system (Graphizy only)."""
    print(f"\n--- Benchmarking Memory System ({n_nodes} nodes) ---")
    data = generate_and_format_positions(width, height, n_nodes)
    grapher = Graphing()
    # Use a realistic memory size
    grapher.init_memory_manager(max_memory_size=n_nodes * 5, track_edge_ages=True)

    total_time = 0
    n_timesteps = 10
    for _ in range(n_timesteps):
        data[:, 1:3] += np.random.normal(0, 5, data[:, 1:3].shape)
        start = time.perf_counter()
        # This single call creates a proximity graph and updates memory
        grapher.make_graph(graph_type="proximity", data_points=data, proximity_thresh=80.0, update_memory=True)
        total_time += (time.perf_counter() - start) * 1000

    avg_time_per_update = total_time / n_timesteps
    memory_stats = grapher.get_memory_analysis()
    print(f"  Average time per memory update: {avg_time_per_update:.2f}ms")
    return {
        'nodes': n_nodes,
        'avg_time_per_update': avg_time_per_update,
        'total_connections': memory_stats.get('total_connections', 0)
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_comprehensive_benchmark():
    """Run all benchmarks and summarize results."""
    print("=" * 60)
    print("COMPREHENSIVE SPATIAL GRAPH BENCHMARK (v2.0)")
    print("Graphizy vs. NetworkX + SciPy/Scikit-learn")
    print("=" * 60)

    all_results = []
    IMAGE_WIDTH, IMAGE_HEIGHT = 1000, 1000
    NODE_COUNTS = [100, 500, 1000, 2500]

    for n in NODE_COUNTS:
        all_results.append(benchmark_delaunay_triangulation(n, IMAGE_WIDTH, IMAGE_HEIGHT))
        all_results.append(benchmark_proximity_graphs(n, IMAGE_WIDTH, IMAGE_HEIGHT))
        all_results.append(benchmark_knn_graphs(n, IMAGE_WIDTH, IMAGE_HEIGHT))

    memory_results = [benchmark_memory_system(n, IMAGE_WIDTH, IMAGE_HEIGHT) for n in NODE_COUNTS]

    # --- Summary ---
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY (Average Speedup: Graphizy vs. Competitor)")
    print("=" * 60)

    summary = {}
    for r in all_results:
        g_type = r['type']
        if g_type not in summary:
            summary[g_type] = {'construction': [], 'metrics': [], 'total': []}
        summary[g_type]['construction'].append(r['construction_speedup'])
        summary[g_type]['metrics'].append(r['metrics_speedup'])
        summary[g_type]['total'].append(r['total_speedup'])

    print(f"{'Graph Type':<15} | {'Construction':<15} | {'Metrics':<15} | {'Total Workflow':<15}")
    print("-" * 70)
    for g_type, data in summary.items():
        avg_const = np.mean(data['construction'])
        avg_metrics = np.mean(data['metrics'])
        avg_total = np.mean(data['total'])
        print(f"{g_type:<15} | {avg_const:>12.1f}x | {avg_metrics:>12.1f}x | {avg_total:>12.1f}x")

    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("1. Unified API: Graphizy provides a single, consistent interface for all graph types.")
    print("2. Performance: Leveraging igraph's C++ backend provides a significant speed advantage.")
    print("3. Temporal Analysis: The built-in memory system is a unique feature for dynamic graphs.")
    print("4. Ease of Use: `call_method_safe` simplifies analysis on potentially disconnected graphs.")

    return {'benchmarks': all_results, 'memory_benchmarks': memory_results}


if __name__ == "__main__":
    results = run_comprehensive_benchmark()

    try:
        import json

        with open('benchmark_results_v2.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nBenchmark complete! Results saved to 'benchmark_results_v2.json'")
    except ImportError:
        print("\nBenchmark complete! Could not save results to JSON.")