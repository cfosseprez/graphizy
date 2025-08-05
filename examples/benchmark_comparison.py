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
        'graphizy_total_time': total_time_gz,
        'networkx_total_time': total_time_nx,
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
        'graphizy_total_time': total_time_gz,
        'networkx_total_time': total_time_nx,
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
        'graphizy_total_time': total_time_gz,
        'networkx_total_time': total_time_nx,
        'construction_speedup': safe_speedup(construction_time_nx, construction_time_gz),
        'metrics_speedup': safe_speedup(metrics_time_nx, metrics_time_gz),
        'total_speedup': safe_speedup(total_time_nx, total_time_gz)
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
        all_results.append(benchmark_delaunay_triangulation(n, IMAGE_
    results = []

    # Parameters following README.md pattern
    IMAGE_WIDTH, IMAGE_HEIGHT = 1000, 1000

    for n_nodes in [100, 500, 1000]:
        # 1. Generate random points using Graphizy's method
        data = generate_and_format_positions(size_x=IMAGE_WIDTH, size_y=IMAGE_HEIGHT, num_particles=n_nodes)
        validate_graphizy_input(data)

        # Extract positions for NetworkX (data format: [id, x, y])
        positions = data[:, 1:3]  # Extract x, y columns

        # === GRAPHIZY APPROACH ===
        start_total = time.perf_counter()

        # Graph construction
        start_construction = time.perf_counter()
        grapher = Graphing()
        graph = grapher.make_graph(graph_type="delaunay", data_points=data)
        graphizy_construction_time = (time.perf_counter() - start_construction) * 1000

        # Graph metrics computation using enhanced call_method
        start_metrics = time.perf_counter()

        # Basic metrics
        info = grapher.get_graph_info(graph)

        # Advanced metrics using explicit format control
        degree_centrality = grapher.call_method_safe(graph, 'degree', "list")
        betweenness_centrality = grapher.call_method_safe(graph, 'betweenness', "list")
        closeness_centrality = grapher.call_method_safe(graph, 'closeness', "list")
        clustering_coeff = grapher.call_method_safe(graph, 'transitivity_local_undirected', "list")

        # Graph-level metrics (scalars)
        diameter = grapher.call_method_safe(graph, 'diameter', "raw", default_value=0)
        avg_path_length = info.get('average_path_length', 0)

        graphizy_metrics_time = (time.perf_counter() - start_metrics) * 1000
        graphizy_total_time = (time.perf_counter() - start_total) * 1000

        # === NETWORKX + SCIPY APPROACH ===
        start_total = time.perf_counter()

        # Graph construction
        start_construction = time.perf_counter()
        tri = Delaunay(positions)
        G = nx.Graph()

        G.add_nodes_from((i, {'x': pos[0], 'y': pos[1]}) for i, pos in enumerate(positions))

        # Extract all unique edges from triangulation
        edges_set = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = sorted((simplex[i], simplex[j]))
                    edges_set.add((a, b))

        # Create weighted edge list using NumPy vectorization
        edges = [
            (i, j, {'weight': np.linalg.norm(positions[i] - positions[j])})
            for i, j in edges_set
        ]

        # Add all edges to the graph at once
        G.add_edges_from(edges)

        networkx_construction_time = (time.perf_counter() - start_construction) * 1000

        # Graph metrics computation using NetworkX
        start_metrics = time.perf_counter()

        # Basic metrics
        num_edges = G.number_of_edges()
        density = nx.density(G)
        is_connected = nx.is_connected(G)

        # Advanced metrics
        degree_centrality_nx = list(nx.degree_centrality(G).values())
        betweenness_centrality_nx = list(nx.betweenness_centrality(G).values())
        closeness_centrality_nx = list(nx.closeness_centrality(G).values())
        clustering_coeff_nx = list(nx.clustering(G).values())

        # Graph-level metrics (only if connected)
        if is_connected:
            diameter_nx = nx.diameter(G)
            avg_path_length_nx = nx.average_shortest_path_length(G)
        else:
            diameter_nx = 0
            avg_path_length_nx = 0

        networkx_metrics_time = (time.perf_counter() - start_metrics) * 1000
        networkx_total_time = (time.perf_counter() - start_total) * 1000

        # Get basic stats for verification
        graphizy_edges = len(graph.es)
        networkx_edges = G.number_of_edges()

        results.append({
            'nodes': n_nodes,
            'graphizy_construction_time': graphizy_construction_time,
            'graphizy_metrics_time': graphizy_metrics_time,
            'graphizy_total_time': graphizy_total_time,
            'networkx_construction_time': networkx_construction_time,
            'networkx_metrics_time': networkx_metrics_time,
            'networkx_total_time': networkx_total_time,
            'graphizy_edges': graphizy_edges,
            'networkx_edges': networkx_edges,
            'construction_speedup': safe_speedup(networkx_construction_time, graphizy_construction_time),
            'metrics_speedup': safe_speedup(networkx_metrics_time, graphizy_metrics_time),
            'total_speedup': safe_speedup(networkx_total_time, graphizy_total_time)
        })

        print(f"  {n_nodes} nodes:")
        print(f"    Construction: Graphizy {graphizy_construction_time:.3f}ms vs NetworkX {networkx_construction_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_construction_time, graphizy_construction_time):.1f}x)")
        print(f"    Metrics:      Graphizy {graphizy_metrics_time:.3f}ms vs NetworkX {networkx_metrics_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_metrics_time, graphizy_metrics_time):.1f}x)")
        print(f"    Total:        Graphizy {graphizy_total_time:.3f}ms vs NetworkX {networkx_total_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_total_time, graphizy_total_time):.1f}x)")

    return results


def benchmark_proximity_graphs():
    """Compare proximity graph construction + clustering analysis"""
    print("\nBenchmarking Proximity Graphs + Clustering Metrics...")
    results = []

    # Parameters following README.md pattern
    IMAGE_WIDTH, IMAGE_HEIGHT = 1000, 1000

    for n_nodes in [100, 500, 1000]:
        # 1. Generate random points using Graphizy's method
        data = generate_and_format_positions(size_x=IMAGE_WIDTH, size_y=IMAGE_HEIGHT, num_particles=n_nodes)
        validate_graphizy_input(data)

        # Extract positions for NetworkX and threshold calculation
        positions = data[:, 1:3]  # Extract x, y columns

        # Calculate adaptive threshold for meaningful connectivity
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=6).fit(positions)  # Use 6 neighbors instead of 4
        distances, _ = nbrs.kneighbors(positions)
        threshold = np.mean(distances[:, 3]) * 1.2  # Use 4th nearest neighbor * 1.2 (denser graphs)

        print(f"  {n_nodes} nodes (adaptive threshold: {threshold:.1f}):")

        # === GRAPHIZY APPROACH ===
        start_total = time.perf_counter()

        # Graph construction
        start_construction = time.perf_counter()
        grapher = Graphing()
        graph = grapher.make_graph(graph_type="proximity", data_points=data, proximity_thresh=threshold)
        graphizy_construction_time = (time.perf_counter() - start_construction) * 1000

        # Clustering and connectivity analysis
        start_metrics = time.perf_counter()

        # Connected components analysis
        components = grapher.call_method_safe(graph, 'connected_components', "raw")
        n_components = len(components)
        largest_component_size = max(len(comp) for comp in components) if components else 0

        # Clustering metrics - using explicit list format
        global_clustering = grapher.call_method_safe(graph, 'transitivity_undirected', "raw", default_value=0.0)
        local_clustering = grapher.call_method_safe(graph, 'transitivity_local_undirected', "list", default_value=[])
        flat_clustering = [val for val in local_clustering if isinstance(val, (int, float))]
        avg_clustering = np.mean(flat_clustering) if flat_clustering else 0

        # Degree distribution - using explicit list format
        degrees = grapher.call_method_safe(graph, 'degree', "list")
        avg_degree = np.mean(degrees) if degrees else 0
        degree_std = np.std(degrees) if degrees else 0

        graphizy_metrics_time = (time.perf_counter() - start_metrics) * 1000
        graphizy_total_time = (time.perf_counter() - start_total) * 1000

        # === NETWORKX + SCIPY APPROACH (FAIR COMPARISON) ===
        start_total = time.perf_counter()

        # Graph construction using SAME algorithm as Graphizy
        start_construction = time.perf_counter()

        # Use scipy's optimized distance calculation (same as Graphizy)

        square_dist = squareform(pdist(positions))

        # Create NetworkX graph
        G = nx.Graph()
        # Add nodes with positions
        G.add_nodes_from((i, {'x': pos[0], 'y': pos[1]}) for i, pos in enumerate(positions))

        # Use upper triangle of the distance matrix to avoid duplicate edges
        i_idx, j_idx = np.triu_indices_from(square_dist, k=1)

        # Apply threshold to filter valid edges
        mask = (square_dist[i_idx, j_idx] < threshold) & (square_dist[i_idx, j_idx] > 0)
        i_idx, j_idx = i_idx[mask], j_idx[mask]
        weights = square_dist[i_idx, j_idx]

        # Build edge list with weights
        edges = [(i, j, {'weight': w}) for i, j, w in zip(i_idx, j_idx, weights)]

        # Add all edges at once
        G.add_edges_from(edges)

        networkx_construction_time = (time.perf_counter() - start_construction) * 1000

        # Clustering and connectivity analysis
        start_metrics = time.perf_counter()

        # Connected components analysis
        components_nx = list(nx.connected_components(G))
        n_components_nx = len(components_nx)
        largest_component_size_nx = max(len(comp) for comp in components_nx) if components_nx else 0

        # Clustering metrics
        global_clustering_nx = nx.transitivity(G)
        local_clustering_nx = list(nx.clustering(G).values())
        avg_clustering_nx = np.mean(local_clustering_nx) if local_clustering_nx else 0

        # Degree distribution
        degrees_nx = list(dict(G.degree()).values())
        avg_degree_nx = np.mean(degrees_nx) if degrees_nx else 0
        degree_std_nx = np.std(degrees_nx) if degrees_nx else 0

        networkx_metrics_time = (time.perf_counter() - start_metrics) * 1000
        networkx_total_time = (time.perf_counter() - start_total) * 1000

        # Validation
        graphizy_edges = len(graph.es)
        networkx_edges = G.number_of_edges()
        graphs_match = (graphizy_edges == networkx_edges and n_components == n_components_nx)

        results.append({
            'nodes': n_nodes,
            'threshold': threshold,
            'graphizy_construction_time': graphizy_construction_time,
            'graphizy_metrics_time': graphizy_metrics_time,
            'graphizy_total_time': graphizy_total_time,
            'networkx_construction_time': networkx_construction_time,
            'networkx_metrics_time': networkx_metrics_time,
            'networkx_total_time': networkx_total_time,
            'graphizy_edges': graphizy_edges,
            'networkx_edges': networkx_edges,
            'graphs_match': graphs_match,
            'construction_speedup': safe_speedup(networkx_construction_time, graphizy_construction_time),
            'metrics_speedup': safe_speedup(networkx_metrics_time, graphizy_metrics_time),
            'total_speedup': safe_speedup(networkx_total_time, graphizy_total_time),
            # Verify results match
            'n_components_match': n_components == n_components_nx,
            'avg_degree_diff': abs(avg_degree - avg_degree_nx)
        })

        print(f"    Graph validation: Graphizy={graphizy_edges} edges, NetworkX={networkx_edges} edges, Match={graphs_match}")
        print(f"    Construction: Graphizy {graphizy_construction_time:.3f}ms vs NetworkX {networkx_construction_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_construction_time, graphizy_construction_time):.1f}x)")
        print(f"    Metrics:      Graphizy {graphizy_metrics_time:.3f}ms vs NetworkX {networkx_metrics_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_metrics_time, graphizy_metrics_time):.1f}x)")
        print(f"    Total:        Graphizy {graphizy_total_time:.3f}ms vs NetworkX {networkx_total_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_total_time, graphizy_total_time):.1f}x)")

        if not graphs_match:
            print(f"    ⚠️  WARNING: Graphs don't match! Results may not be comparable.")

    return results


def benchmark_mst_graphs():
    """Compare Minimum Spanning Tree graph construction + analysis"""
    print("\nBenchmarking Minimum Spanning Tree + Network Analysis...")
    results = []

    # Parameters following README.md pattern
    IMAGE_WIDTH, IMAGE_HEIGHT = 1000, 1000

    for n_nodes in [100, 500, 1000]:
        # 1. Generate random points using Graphizy's method
        data = generate_and_format_positions(size_x=IMAGE_WIDTH, size_y=IMAGE_HEIGHT, num_particles=n_nodes)
        validate_graphizy_input(data)

        # Extract positions for NetworkX (data format: [id, x, y])
        positions = data[:, 1:3]  # Extract x, y columns

        # === GRAPHIZY APPROACH ===
        start_total = time.perf_counter()

        # Graph construction
        start_construction = time.perf_counter()
        grapher = Graphing()
        graph = grapher.make_graph(graph_type="mst", data_points=data)
        graphizy_construction_time = (time.perf_counter() - start_construction) * 1000

        # Network analysis
        start_metrics = time.perf_counter()

        # Connectivity analysis (MST is always connected and acyclic)
        is_connected = grapher.call_method_safe(graph, 'is_connected', "raw")
        n_components = len(grapher.call_method_safe(graph, 'connected_components', "raw"))

        # Degree analysis - using explicit list format
        degrees = grapher.call_method_safe(graph, 'degree', "list")
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        # Tree properties
        diameter = grapher.call_method_safe(graph, 'diameter', "raw", default_value=0)

        # Path-based metrics for trees
        avg_path_length = grapher.call_method_safe(graph, 'average_path_length', "raw", default_value=0)

        graphizy_metrics_time = (time.perf_counter() - start_metrics) * 1000
        graphizy_total_time = (time.perf_counter() - start_total) * 1000

        # === NETWORKX + SCIPY APPROACH ===
        start_total = time.perf_counter()

        # Graph construction
        start_construction = time.perf_counter()

        # Create complete graph with distances
        G_complete = nx.Graph()
        # Add nodes with positions
        G_complete.add_nodes_from((i, {'x': pos[0], 'y': pos[1]}) for i, pos in enumerate(positions))

        # Compute pairwise distances using pdist
        dists = squareform(pdist(positions))  # shape (n_nodes, n_nodes)

        # Get upper triangle indices (i < j) to avoid duplicates
        i_idx, j_idx = np.triu_indices_from(dists, k=1)
        weights = dists[i_idx, j_idx]

        # Create edge list with weights
        edges = [(i, j, {'weight': w}) for i, j, w in zip(i_idx, j_idx, weights)]

        # Add all edges at once
        G_complete.add_edges_from(edges)

        # Get minimum spanning tree
        G = nx.minimum_spanning_tree(G_complete)

        networkx_construction_time = (time.perf_counter() - start_construction) * 1000

        # Network analysis
        start_metrics = time.perf_counter()

        # Connectivity analysis
        is_connected_nx = nx.is_connected(G)
        n_components_nx = nx.number_connected_components(G)

        # Degree analysis
        degrees_nx = list(dict(G.degree()).values())
        avg_degree_nx = np.mean(degrees_nx) if degrees_nx else 0
        max_degree_nx = max(degrees_nx) if degrees_nx else 0

        # Tree properties
        diameter_nx = nx.diameter(G) if is_connected_nx else 0
        avg_path_length_nx = nx.average_shortest_path_length(G) if is_connected_nx else 0

        networkx_metrics_time = (time.perf_counter() - start_metrics) * 1000
        networkx_total_time = (time.perf_counter() - start_total) * 1000

        results.append({
            'nodes': n_nodes,
            'graphizy_construction_time': graphizy_construction_time,
            'graphizy_metrics_time': graphizy_metrics_time,
            'graphizy_total_time': graphizy_total_time,
            'networkx_construction_time': networkx_construction_time,
            'networkx_metrics_time': networkx_metrics_time,
            'networkx_total_time': networkx_total_time,
            'graphizy_edges': len(graph.es),
            'networkx_edges': G.number_of_edges(),
            'construction_speedup': safe_speedup(networkx_construction_time, graphizy_construction_time),
            'metrics_speedup': safe_speedup(networkx_metrics_time, graphizy_metrics_time),
            'total_speedup': safe_speedup(networkx_total_time, graphizy_total_time)
        })

        print(f"  {n_nodes} nodes:")
        print(f"    Construction: Graphizy {graphizy_construction_time:.3f}ms vs NetworkX {networkx_construction_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_construction_time, graphizy_construction_time):.1f}x)")
        print(f"    Metrics:      Graphizy {graphizy_metrics_time:.3f}ms vs NetworkX {networkx_metrics_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_metrics_time, graphizy_metrics_time):.1f}x)")
        print(f"    Total:        Graphizy {graphizy_total_time:.3f}ms vs NetworkX {networkx_total_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_total_time, graphizy_total_time):.1f}x)")

    return results


def benchmark_memory_system():
    """Benchmark the unique memory-enhanced system (no NetworkX equivalent)"""
    print("\nBenchmarking Memory-Enhanced System (Graphizy only)...")
    results = []

    # Parameters following README.md pattern
    IMAGE_WIDTH, IMAGE_HEIGHT = 1000, 1000

    for n_nodes in [100, 500, 1000]:
        # 1. Generate initial random points using Graphizy's method
        data = generate_and_format_positions(size_x=IMAGE_WIDTH, size_y=IMAGE_HEIGHT, num_particles=n_nodes)
        validate_graphizy_input(data)

        grapher = Graphing()
        grapher.init_memory_manager(max_memory_size=50, track_edge_ages=True)

        # Simulate temporal evolution
        total_time = 0
        n_timesteps = 20

        for timestep in range(n_timesteps):
            # Simulate movement (modify existing data)
            data[:, 1:3] += np.random.normal(0, 5, data[:, 1:3].shape)  # Update x, y positions

            start = time.perf_counter()

            # Update memory with current proximity graph
            current_graph = grapher.make_graph(graph_type="proximity", data_points=data, proximity_thresh=80.0)
            grapher.update_memory_with_graph(current_graph)

            # Create memory-enhanced graph
            memory_graph = grapher.make_memory_graph(data)

            total_time += (time.perf_counter() - start) * 1000

        avg_time_per_update = total_time / n_timesteps

        # Get memory statistics
        memory_stats = grapher.get_memory_analysis()

        results.append({
            'nodes': n_nodes,
            'avg_time_per_update': avg_time_per_update,
            'total_connections': memory_stats['total_connections'],
            'timesteps': n_timesteps
        })

        print(f"  {n_nodes} nodes: {avg_time_per_update:.3f}ms per memory update "
              f"({memory_stats['total_connections']} total connections tracked)")

    return results


def run_comprehensive_benchmark():
    """Run all benchmarks and summarize results"""
    print("=" * 60)
    print("COMPREHENSIVE SPATIAL GRAPH BENCHMARK")
    print("Including Graph Construction + Analysis Metrics")
    print("Graphizy vs NetworkX+SciPy")
    print("=" * 60)

    # Run individual benchmarks
    delaunay_results = benchmark_delaunay_triangulation()
    proximity_results = benchmark_proximity_graphs()
    mst_results = benchmark_mst_graphs()
    memory_results = benchmark_memory_system()

    # Summary table
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    print("\nAverage Speedup (Graphizy vs NetworkX+SciPy):")

    # Calculate average speedups for different phases
    avg_delaunay_construction = np.mean([r['construction_speedup'] for r in delaunay_results])
    avg_delaunay_metrics = np.mean([r['metrics_speedup'] for r in delaunay_results])
    avg_delaunay_total = np.mean([r['total_speedup'] for r in delaunay_results])

    avg_proximity_construction = np.mean([r['construction_speedup'] for r in proximity_results])
    avg_proximity_metrics = np.mean([r['metrics_speedup'] for r in proximity_results])
    avg_proximity_total = np.mean([r['total_speedup'] for r in proximity_results])

    avg_mst_construction = np.mean([r['construction_speedup'] for r in mst_results])
    avg_mst_metrics = np.mean([r['metrics_speedup'] for r in mst_results])
    avg_mst_total = np.mean([r['total_speedup'] for r in mst_results])

    print(f"\nGraph Construction Phase:")
    print(f"  Delaunay Triangulation: {avg_delaunay_construction:.1f}x faster")
    print(f"  Proximity Graphs:       {avg_proximity_construction:.1f}x faster")
    print(f"  Minimum Spanning Tree:  {avg_mst_construction:.1f}x faster")

    print(f"\nGraph Analysis Metrics Phase:")
    print(f"  Delaunay Analysis:      {avg_delaunay_metrics:.1f}x faster")
    print(f"  Proximity Analysis:     {avg_proximity_metrics:.1f}x faster")
    print(f"  MST Analysis:           {avg_mst_metrics:.1f}x faster")

    print(f"\nComplete Workflow (Construction + Analysis):")
    print(f"  Delaunay Complete:      {avg_delaunay_total:.1f}x faster")
    print(f"  Proximity Complete:     {avg_proximity_total:.1f}x faster")
    print(f"  MST Complete:           {avg_mst_total:.1f}x faster")
    print(f"  Memory-Enhanced System: No equivalent in NetworkX")

    # Detailed results table with better precision
    print(f"\nDetailed Results (Construction | Metrics | Total):")
    print(f"{'Graph Type':<15} {'Nodes':<6} "
          f"{'Graphizy Const.':<16} {'NetworkX Const.':<16} "
          f"{'Graphizy Metrics':<17} {'NetworkX Metrics':<17} "
          f"{'Graphizy Total':<16} {'NetworkX Total':<17} {'Total Speedup':<12}")
    print("-" * 150)

    for result in delaunay_results:
        graphizy_total = result['graphizy_construction_time'] + result['graphizy_metrics_time']
        networkx_total = result['networkx_construction_time'] + result['networkx_metrics_time']
        print(f"{'Delaunay':<15} {result['nodes']:<6} "
              f"{result['graphizy_construction_time']:<16.3f} {result['networkx_construction_time']:<16.3f} "
              f"{result['graphizy_metrics_time']:<17.3f} {result['networkx_metrics_time']:<17.3f} "
              f"{graphizy_total:<16.3f} {networkx_total:<17.3f} {result['total_speedup']:<12.1f}")

    for result in proximity_results:
        graphizy_total = result['graphizy_construction_time'] + result['graphizy_metrics_time']
        networkx_total = result['networkx_construction_time'] + result['networkx_metrics_time']
        print(f"{'Proximity':<15} {result['nodes']:<6} "
              f"{result['graphizy_construction_time']:<16.3f} {result['networkx_construction_time']:<16.3f} "
              f"{result['graphizy_metrics_time']:<17.3f} {result['networkx_metrics_time']:<17.3f} "
              f"{graphizy_total:<16.3f} {networkx_total:<17.3f} {result['total_speedup']:<12.1f}")

    for result in mst_results:
        graphizy_total = result['graphizy_construction_time'] + result['graphizy_metrics_time']
        networkx_total = result['networkx_construction_time'] + result['networkx_metrics_time']
        print(f"{'MST':<15} {result['nodes']:<6} "
              f"{result['graphizy_construction_time']:<16.3f} {result['networkx_construction_time']:<16.3f} "
              f"{result['graphizy_metrics_time']:<17.3f} {result['networkx_metrics_time']:<17.3f} "
              f"{graphizy_total:<16.3f} {networkx_total:<17.3f} {result['total_speedup']:<12.1f}")

    print(f"\nMemory System Performance (Graphizy only):")
    for result in memory_results:
        print(f"  {result['nodes']} nodes: {result['avg_time_per_update']:.3f}ms per update "
              f"({result['total_connections']} connections tracked)")

    # Key insights
    print(f"\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)

    print(f"1. Graph Construction: Graphizy shows {avg_delaunay_construction:.1f}-{avg_proximity_construction:.1f}x speedup")
    print(f"2. Graph Analysis: Efficient igraph backend provides {avg_delaunay_metrics:.1f}-{avg_proximity_metrics:.1f}x speedup")
    print(f"3. Complete Workflow: {avg_delaunay_total:.1f}-{avg_proximity_total:.1f}x faster end-to-end")
    print(f"4. Memory System: Unique temporal analysis capability not available elsewhere")
    print(f"5. API Simplicity: Single library vs. multiple library integration")

    return {
        'delaunay': delaunay_results,
        'proximity': proximity_results,
        'mst': mst_results,
        'memory': memory_results
    }


if __name__ == "__main__":
    # Run the comprehensive benchmark
    results = run_comprehensive_benchmark()

    # Save results for paper
    import json

    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark complete! Results saved to 'benchmark_results.json'")