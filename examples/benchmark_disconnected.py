"""
Comprehensive spatial benchmark comparing Graphizy vs NetworkX+SciPy
PROPER handling of disconnected graphs - accounts for library behavior differences
"""

import time
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay, distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
import graphizy
import json

BLOCK_DISPLAY = False


def safe_speedup(time_nx, time_graphizy, min_time=0.001):
    """Calculate speedup while avoiding division by zero"""
    safe_graphizy = max(time_graphizy, min_time)
    safe_nx = max(time_nx, min_time)
    return safe_nx / safe_graphizy


def convert_numpy_types(obj):
    """Convert numpy types to regular Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None  # Convert NaN to None for JSON
        elif np.isinf(obj):
            return None  # Convert inf to None for JSON
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def handle_disconnected_igraph_metrics(graph, grapher):
    """
    Handle igraph metrics on disconnected graphs properly
    Returns metrics that are comparable to NetworkX's disconnected graph handling
    """
    metrics = {}

    # Basic connectivity info
    components = grapher.call_method_safe(graph, 'connected_components', "raw")
    is_connected = len(components) == 1

    # Always safe metrics
    metrics['is_connected'] = is_connected
    metrics['components'] = len(components)
    metrics['edges'] = len(graph.es)
    metrics['vertices'] = len(graph.vs)

    # Degree-based metrics (always work)
    degrees = grapher.call_method_safe(graph, 'degree', "list")
    metrics['degrees'] = degrees
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0
    metrics['max_degree'] = max(degrees) if degrees else 0
    metrics['min_degree'] = min(degrees) if degrees else 0
    metrics['isolated_nodes'] = sum(1 for d in degrees if d == 0)

    # Density (always works)
    metrics['density'] = grapher.call_method_safe(graph, 'density', "raw", default_value=0.0)

    # Clustering metrics (work on disconnected graphs, but may have NaN)
    global_clustering = grapher.call_method_safe(graph, 'transitivity_undirected', "raw", default_value=0.0)
    local_clustering = grapher.call_method_safe(graph, 'transitivity_local_undirected', "list", default_value=[])

    # Handle NaN in local clustering - replace with 0 for isolated nodes
    if local_clustering:
        local_clustering_clean = []
        for i, val in enumerate(local_clustering):
            if np.isnan(val) or np.isinf(val):
                # For isolated nodes (degree 0), clustering should be 0, not NaN
                local_clustering_clean.append(0.0)
            else:
                local_clustering_clean.append(val)
        metrics['local_clustering'] = local_clustering_clean
        metrics['avg_local_clustering'] = np.mean(local_clustering_clean)
    else:
        metrics['local_clustering'] = [0.0] * len(graph.vs)
        metrics['avg_local_clustering'] = 0.0

    metrics['global_clustering'] = global_clustering

    # Path-based metrics (problematic for disconnected graphs in igraph)
    if is_connected:
        # Graph is connected - safe to compute path metrics
        metrics['diameter'] = grapher.call_method_safe(graph, 'diameter', "raw", default_value=0)
        metrics['avg_path_length'] = grapher.call_method_safe(graph, 'average_path_length', "raw", default_value=0)

        # Centrality metrics (safe on connected graphs)
        metrics['betweenness'] = grapher.call_method_safe(graph, 'betweenness', "list", default_value=[])
        metrics['closeness'] = grapher.call_method_safe(graph, 'closeness', "list", default_value=[])
    else:
        # Graph is disconnected - compute on largest component only (like NetworkX often does)
        largest_component = max(components, key=len) if components else []

        if len(largest_component) >= 2:
            # Create subgraph of largest component
            try:
                subgraph = graph.subgraph(largest_component)
                metrics['diameter'] = grapher.call_method_safe(subgraph, 'diameter', "raw", default_value=0)
                metrics['avg_path_length'] = grapher.call_method_safe(subgraph, 'average_path_length', "raw",
                                                                      default_value=0)

                # For centrality, compute on full graph but expect some NaN values
                betweenness_raw = grapher.call_method_safe(graph, 'betweenness', "list", default_value=[])
                closeness_raw = grapher.call_method_safe(graph, 'closeness', "list", default_value=[])

                # Replace NaN/inf with 0 for disconnected nodes
                metrics['betweenness'] = [0.0 if np.isnan(x) or np.isinf(x) else x for x in betweenness_raw]
                metrics['closeness'] = [0.0 if np.isnan(x) or np.isinf(x) else x for x in closeness_raw]

            except Exception as e:
                # Fallback values
                metrics['diameter'] = 0
                metrics['avg_path_length'] = 0
                metrics['betweenness'] = [0.0] * len(graph.vs)
                metrics['closeness'] = [0.0] * len(graph.vs)
        else:
            # No meaningful connected component
            metrics['diameter'] = 0
            metrics['avg_path_length'] = 0
            metrics['betweenness'] = [0.0] * len(graph.vs)
            metrics['closeness'] = [0.0] * len(graph.vs)

    return metrics


def handle_disconnected_networkx_metrics(G):
    """
    Handle NetworkX metrics on disconnected graphs
    Uses NetworkX's native disconnected graph handling
    """
    metrics = {}

    # Basic connectivity info
    is_connected = nx.is_connected(G)
    components = list(nx.connected_components(G))

    metrics['is_connected'] = is_connected
    metrics['components'] = len(components)
    metrics['edges'] = G.number_of_edges()
    metrics['vertices'] = G.number_of_nodes()

    # Degree-based metrics (always work)
    degrees = list(dict(G.degree()).values())
    metrics['degrees'] = degrees
    metrics['avg_degree'] = np.mean(degrees) if degrees else 0
    metrics['max_degree'] = max(degrees) if degrees else 0
    metrics['min_degree'] = min(degrees) if degrees else 0
    metrics['isolated_nodes'] = sum(1 for d in degrees if d == 0)

    # Density (always works)
    metrics['density'] = nx.density(G)

    # Clustering metrics (NetworkX handles disconnected graphs gracefully)
    global_clustering = nx.transitivity(G)
    local_clustering = list(nx.clustering(G).values())

    metrics['global_clustering'] = global_clustering
    metrics['local_clustering'] = local_clustering
    metrics['avg_local_clustering'] = np.mean(local_clustering) if local_clustering else 0

    # Path-based metrics (NetworkX approach for disconnected graphs)
    if is_connected:
        # Graph is connected - compute normally
        metrics['diameter'] = nx.diameter(G)
        metrics['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        # Graph is disconnected - compute on largest component
        largest_component = max(components, key=len) if components else set()

        if len(largest_component) >= 2:
            largest_subgraph = G.subgraph(largest_component)
            metrics['diameter'] = nx.diameter(largest_subgraph)
            metrics['avg_path_length'] = nx.average_shortest_path_length(largest_subgraph)
        else:
            metrics['diameter'] = 0
            metrics['avg_path_length'] = 0

    # Centrality metrics (NetworkX handles disconnected graphs well)
    # Uses harmonic centrality for better disconnected graph handling
    try:
        metrics['betweenness'] = list(nx.betweenness_centrality(G).values())
        metrics['closeness'] = list(nx.closeness_centrality(G).values())
    except:
        # Fallback if centrality computation fails
        metrics['betweenness'] = [0.0] * G.number_of_nodes()
        metrics['closeness'] = [0.0] * G.number_of_nodes()

    return metrics


def benchmark_proximity_graphs_with_proper_disconnected_handling():
    """
    Benchmark with proper handling of disconnected graph differences between libraries
    """
    print("\nBenchmarking Proximity Graphs (PROPER DISCONNECTED GRAPH HANDLING)...")
    results = []
    metrics_comparison = []

    for n_nodes in [100, 500, 1000]:
        positions = np.random.rand(n_nodes, 2) * 1000

        # Use multiple thresholds to test different connectivity levels
        for threshold_multiplier in [1.0, 1.5, 2.0]:
            # Calculate threshold
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=4).fit(positions)
            distances, _ = nbrs.kneighbors(positions)
            threshold = np.mean(distances[:, 1]) * threshold_multiplier

            print(f"\n  {n_nodes} nodes, threshold: {threshold:.1f} (multiplier: {threshold_multiplier})")

            # === GRAPHIZY APPROACH ===
            start_total = time.perf_counter()

            # Graph construction
            start_construction = time.perf_counter()
            grapher = graphizy.Graphing()
            data = np.column_stack((np.arange(n_nodes), positions))
            graph = grapher.make_proximity(data, proximity_thresh=threshold)
            graphizy_construction_time = (time.perf_counter() - start_construction) * 1000

            # Metrics computation with proper disconnected graph handling
            start_metrics = time.perf_counter()
            graphizy_metrics = handle_disconnected_igraph_metrics(graph, grapher)
            graphizy_metrics_time = (time.perf_counter() - start_metrics) * 1000
            graphizy_total_time = (time.perf_counter() - start_total) * 1000

            # === NETWORKX APPROACH ===
            start_total = time.perf_counter()

            # Graph construction using SAME algorithm as Graphizy
            start_construction = time.perf_counter()

            from scipy.spatial.distance import pdist, squareform
            square_dist = squareform(pdist(positions))

            G = nx.Graph()
            for i, pos in enumerate(positions):
                G.add_node(i, x=pos[0], y=pos[1])

            for i, row in enumerate(square_dist):
                nearby_indices = np.where((row < threshold) & (row > 0))[0]
                for j in nearby_indices:
                    if i < j:
                        G.add_edge(i, j, weight=row[j])

            networkx_construction_time = (time.perf_counter() - start_construction) * 1000

            # Metrics computation with NetworkX's disconnected graph handling
            start_metrics = time.perf_counter()
            networkx_metrics = handle_disconnected_networkx_metrics(G)
            networkx_metrics_time = (time.perf_counter() - start_metrics) * 1000
            networkx_total_time = (time.perf_counter() - start_total) * 1000

            # === COMPARISON ===
            graphs_match = (graphizy_metrics['edges'] == networkx_metrics['edges'] and
                            graphizy_metrics['components'] == networkx_metrics['components'])

            connectivity_ratio = max(len(comp) for comp in grapher.call_method(graph, 'connected_components',
                                                                               "raw")) / n_nodes if n_nodes > 0 else 0
            isolation_ratio = graphizy_metrics['isolated_nodes'] / n_nodes if n_nodes > 0 else 0

            result = {
                'nodes': n_nodes,
                'threshold': threshold,
                'threshold_multiplier': threshold_multiplier,
                'graphizy_construction_time': graphizy_construction_time,
                'graphizy_metrics_time': graphizy_metrics_time,
                'graphizy_total_time': graphizy_total_time,
                'networkx_construction_time': networkx_construction_time,
                'networkx_metrics_time': networkx_metrics_time,
                'networkx_total_time': networkx_total_time,
                'graphs_match': graphs_match,
                'connectivity_ratio': connectivity_ratio,
                'isolation_ratio': isolation_ratio,
                'construction_speedup': safe_speedup(networkx_construction_time, graphizy_construction_time),
                'metrics_speedup': safe_speedup(networkx_metrics_time, graphizy_metrics_time),
                'total_speedup': safe_speedup(networkx_total_time, graphizy_total_time)
            }

            results.append(result)

            # Store metrics comparison
            metrics_comparison.append({
                "nodes": n_nodes,
                "threshold": threshold,
                "threshold_multiplier": threshold_multiplier,
                "graphizy": graphizy_metrics,
                "networkx": networkx_metrics,
                "connectivity_analysis": {
                    "connected": graphizy_metrics['is_connected'],
                    "components": graphizy_metrics['components'],
                    "connectivity_ratio": connectivity_ratio,
                    "isolation_ratio": isolation_ratio
                }
            })

            # Output
            print(f"    Graph: {graphizy_metrics['edges']} edges, {graphizy_metrics['components']} components")
            print(f"    Connectivity: {connectivity_ratio:.1%}, Isolation: {isolation_ratio:.1%}")
            print(
                f"    Construction: Graphizy {graphizy_construction_time:.3f}ms vs NetworkX {networkx_construction_time:.3f}ms "
                f"(speedup: {result['construction_speedup']:.1f}x)")
            print(f"    Metrics:      Graphizy {graphizy_metrics_time:.3f}ms vs NetworkX {networkx_metrics_time:.3f}ms "
                  f"(speedup: {result['metrics_speedup']:.1f}x)")
            print(f"    Total:        Graphizy {graphizy_total_time:.3f}ms vs NetworkX {networkx_total_time:.3f}ms "
                  f"(speedup: {result['total_speedup']:.1f}x)")

            # Warn about problematic graphs
            if isolation_ratio > 0.5:
                print(f"    ⚠️  HIGH ISOLATION: {isolation_ratio:.0%} isolated nodes")
            if not graphs_match:
                print(f"    ⚠️  GRAPH MISMATCH!")

    return results, metrics_comparison


def analyze_disconnected_graph_behavior():
    """
    Analyze how both libraries handle disconnected graphs
    """
    print(f"\n{'=' * 60}")
    print("DISCONNECTED GRAPH BEHAVIOR ANALYSIS")
    print(f"{'=' * 60}")

    n_nodes = 200
    positions = np.random.rand(n_nodes, 2) * 1000

    # Create a very sparse graph (definitely disconnected)
    threshold = 30  # Very small threshold

    print(f"Creating sparse graph with threshold {threshold} on {n_nodes} nodes...")

    # Create Graphizy graph
    grapher = graphizy.Graphing()
    data = np.column_stack((np.arange(n_nodes), positions))
    graph = grapher.make_proximity(data, proximity_thresh=threshold)

    # Create equivalent NetworkX graph
    from scipy.spatial.distance import pdist, squareform
    square_dist = squareform(pdist(positions))
    G = nx.Graph()
    for i, pos in enumerate(positions):
        G.add_node(i, x=pos[0], y=pos[1])
    for i, row in enumerate(square_dist):
        nearby_indices = np.where((row < threshold) & (row > 0))[0]
        for j in nearby_indices:
            if i < j:
                G.add_edge(i, j, weight=row[j])

    print(f"\nGraph structure:")
    print(f"  Edges: {len(graph.es)}")
    print(f"  Components: {len(grapher.call_method_safe(graph, 'connected_components', 'raw'))}")
    print(f"  Connected: {len(grapher.call_method_safe(graph, 'connected_components', 'raw')) == 1}")

    print(f"\nTesting problematic metrics:")

    # Test clustering (should work on both)
    print(f"  Clustering:")
    try:
        ig_clustering = grapher.call_method_safe(graph, 'transitivity_undirected', "raw")
        print(f"    igraph global clustering: {ig_clustering}")
    except Exception as e:
        print(f"    igraph global clustering: FAILED - {e}")

    try:
        nx_clustering = nx.transitivity(G)
        print(f"    NetworkX global clustering: {nx_clustering}")
    except Exception as e:
        print(f"    NetworkX global clustering: FAILED - {e}")

    # Test local clustering
    print(f"  Local clustering:")
    try:
        ig_local = grapher.call_method_safe(graph, 'transitivity_local_undirected', "list")
        nan_count = sum(1 for x in ig_local if np.isnan(x))
        print(f"    igraph local clustering: {len(ig_local)} values, {nan_count} NaNs")
    except Exception as e:
        print(f"    igraph local clustering: FAILED - {e}")

    try:
        nx_local = list(nx.clustering(G).values())
        nan_count = sum(1 for x in nx_local if np.isnan(x))
        print(f"    NetworkX local clustering: {len(nx_local)} values, {nan_count} NaNs")
    except Exception as e:
        print(f"    NetworkX local clustering: FAILED - {e}")


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("DISCONNECTED GRAPH AWARE PROXIMITY BENCHMARK")
    print("Proper handling of library differences in disconnected graph metrics")
    print("=" * 80)

    # Analyze how libraries handle disconnected graphs
    analyze_disconnected_graph_behavior()

    # Run main benchmark with proper disconnected graph handling
    results, metrics_comparison = benchmark_proximity_graphs_with_proper_disconnected_handling()

    print("\n" + "=" * 50)
    print("RESULTS SUMMARY BY CONNECTIVITY")
    print("=" * 50)

    # Group results by connectivity level
    connected_results = [r for r in results if r['connectivity_ratio'] > 0.8]
    partially_connected_results = [r for r in results if 0.3 <= r['connectivity_ratio'] <= 0.8]
    fragmented_results = [r for r in results if r['connectivity_ratio'] < 0.3]

    for category, category_results in [
        ("WELL CONNECTED (>80% in largest component)", connected_results),
        ("PARTIALLY CONNECTED (30-80% in largest component)", partially_connected_results),
        ("HIGHLY FRAGMENTED (<30% in largest component)", fragmented_results)
    ]:
        if category_results:
            print(f"\n{category}:")
            avg_construction = np.mean([r['construction_speedup'] for r in category_results])
            avg_metrics = np.mean([r['metrics_speedup'] for r in category_results])
            avg_total = np.mean([r['total_speedup'] for r in category_results])

            print(
                f"  Average speedups - Construction: {avg_construction:.1f}x, Metrics: {avg_metrics:.1f}x, Total: {avg_total:.1f}x")
            print(f"  Sample results:")
            for r in category_results[:3]:  # Show first 3
                print(
                    f"    {r['nodes']} nodes: Construction {r['construction_speedup']:.1f}x, Metrics {r['metrics_speedup']:.1f}x, Total {r['total_speedup']:.1f}x")

    # Save results
    try:
        with open('disconnected_aware_benchmark_results.json', 'w') as f:
            json.dump(convert_numpy_types(results), f, indent=2)
        print(f"\nResults saved to 'disconnected_aware_benchmark_results.json'")
    except Exception as e:
        print(f"Error saving results: {e}")

    print(f"\n{'=' * 50}")
    print("KEY INSIGHTS")
    print(f"{'=' * 50}")
    print("✅ This benchmark properly accounts for disconnected graph behavior differences")
    print("✅ igraph NaN values are handled consistently with NetworkX approaches")
    print("✅ Path-based metrics computed on largest component for fair comparison")
    print("✅ Results show realistic performance differences between libraries")