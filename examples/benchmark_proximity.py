"""
Comprehensive spatial benchmark comparing Graphizy vs NetworkX+SciPy
Complete version with metrics comparison and connectivity analysis
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


def debug_timing_issue():
    """Diagnose the 0.0ms timing issue"""
    import time
    import timeit
    import graphizy
    import numpy as np

    print("=== TIMING DIAGNOSTIC ===")

    # Create a substantial graph
    n_nodes = 1000
    positions = np.random.rand(n_nodes, 2) * 1000
    grapher = graphizy.Graphing()
    data = np.column_stack((np.arange(n_nodes), positions))
    graph = grapher.make_proximity(data, proximity_thresh=200)  # Dense graph

    print(f"Created graph with {len(graph.es)} edges, {len(graph.vs)} vertices")

    # Test different timing methods
    print("\n--- Testing timing resolution ---")

    # Test 1: time.time() resolution
    start = time.time()
    degrees1 = grapher.call_method_brutal(graph, 'degree', "list")
    time_time_duration = (time.time() - start) * 1000

    # Test 2: time.perf_counter() resolution
    start = time.perf_counter()
    degrees2 = grapher.call_method_brutal(graph, 'degree', "list")
    perf_counter_duration = (time.perf_counter() - start) * 1000

    # Test 3: timeit for accurate measurement
    def degree_call():
        return grapher.call_method_brutal(graph, 'degree', "list")

    timeit_duration = timeit.timeit(degree_call, number=1) * 1000

    print(f"time.time():         {time_time_duration:.3f}ms")
    print(f"time.perf_counter(): {perf_counter_duration:.3f}ms")
    print(f"timeit:              {timeit_duration:.3f}ms")

    # Test 4: Direct igraph call (bypass call_method_brutal)
    start = time.perf_counter()
    degrees_direct = graph.degree()
    direct_duration = (time.perf_counter() - start) * 1000

    print(f"Direct igraph call:  {direct_duration:.3f}ms")

    # Test 5: Check if results are identical (caching test)
    print(f"\nResults identical: {degrees1 == degrees2 == list(degrees_direct)}")

    # Test 6: Multiple calls to check caching
    print("\n--- Testing for caching ---")
    times = []
    for i in range(5):
        start = time.perf_counter()
        degrees = grapher.call_method_brutal(graph, 'degree', "list")
        duration = (time.perf_counter() - start) * 1000
        times.append(duration)
        print(f"Call {i + 1}: {duration:.3f}ms")

    print(f"Average: {np.mean(times):.3f}ms, Std: {np.std(times):.3f}ms")

    # Test 7: Check call_method_brutal implementation
    print("\n--- Testing call_method_brutal path ---")

    # Direct method call
    start = time.perf_counter()
    result_direct = graph.degree()
    direct_time = (time.perf_counter() - start) * 1000

    # Through call_igraph_method
    start = time.perf_counter()
    from graphizy.algorithms import call_igraph_method
    result_call = call_igraph_method(graph, "degree")
    call_time = (time.perf_counter() - start) * 1000

    # Through call_method_brutal
    start = time.perf_counter()
    result_method = grapher.call_method_brutal(graph, "degree", "list")
    method_time = (time.perf_counter() - start) * 1000

    print(f"Direct graph.degree():     {direct_time:.3f}ms")
    print(f"call_igraph_method():      {call_time:.3f}ms")
    print(f"grapher.call_method_brutal():     {method_time:.3f}ms")

    print(f"Results match: {list(result_direct) == list(result_call) == result_method}")


def benchmark_proximity_graphs_diagnostic():
    """Fixed diagnostic with better timing"""
    print("\nProximity Graph Diagnostic Analysis (FIXED TIMING)...")

    n_nodes = 500
    positions = np.random.rand(n_nodes, 2) * 1000

    # Test different thresholds
    thresholds = [50, 100, 150, 200, 300]

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")

        # Create Graphizy graph
        grapher = graphizy.Graphing()
        data = np.column_stack((np.arange(n_nodes), positions))

        start_time = time.perf_counter()
        graph = grapher.make_proximity(data, proximity_thresh=threshold)
        construction_time = (time.perf_counter() - start_time) * 1000

        # Analyze graph structure
        edges = len(graph.es)
        components = grapher.call_method_brutal(graph, 'connected_components', "raw")
        n_components = len(components)
        largest_component = max(len(comp) for comp in components) if components else 0
        degrees = grapher.call_method_brutal(graph, 'degree', "list")
        isolated_nodes = sum(1 for d in degrees if d == 0)

        # Time different metrics WITH BETTER TIMING
        metrics_times = {}

        start = time.perf_counter()
        clustering = grapher.call_method_safe(graph, 'transitivity_undirected', "raw", default_value=0.0)
        metrics_times['global_clustering'] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        local_clustering = grapher.call_method_safe(graph, 'transitivity_local_undirected', "list", default_value=[])
        metrics_times['local_clustering'] = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        degree_calc = grapher.call_method_safe(graph, 'degree', "list")
        metrics_times['degrees'] = (time.perf_counter() - start) * 1000

        print(f"  Construction: {construction_time:.3f}ms")
        print(f"  Graph: {edges} edges, {n_components} components, largest={largest_component}")
        print(f"  Isolated nodes: {isolated_nodes}/{n_nodes} ({isolated_nodes/n_nodes*100:.0f}%)")
        print(f"  Metric times: global_clustering={metrics_times['global_clustering']:.3f}ms, "
              f"local_clustering={metrics_times['local_clustering']:.3f}ms, "
              f"degrees={metrics_times['degrees']:.3f}ms")


def analyze_connectivity_impact():
    """Analyze how graph connectivity affects performance metrics"""
    print(f"\n{'=' * 60}")
    print("CONNECTIVITY IMPACT ANALYSIS")
    print(f"{'=' * 60}")

    n_nodes = 500
    positions = np.random.rand(n_nodes, 2) * 1000

    # Test range of thresholds to see connectivity impact
    thresholds = [30, 50, 80, 120, 200]

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold} ---")

        grapher = graphizy.Graphing()
        data = np.column_stack((np.arange(n_nodes), positions))

        # Create graph
        graph = grapher.make_proximity(data, proximity_thresh=threshold)

        # Analyze connectivity
        components = grapher.call_method_safe(graph, 'connected_components', "raw")
        n_components = len(components)
        largest_component = max(len(comp) for comp in components) if components else 0
        edges = len(graph.es)
        degrees = grapher.call_method_safe(graph, 'degree', "list")
        isolated_nodes = sum(1 for d in degrees if d == 0)

        # Time critical metrics
        start = time.perf_counter()
        clustering = grapher.call_method_safe(graph, 'transitivity_undirected', "raw", default_value=0.0)
        clustering_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        local_clustering = grapher.call_method_safe(graph, 'transitivity_local_undirected', "list", default_value=[])
        local_clustering_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        degree_calc = grapher.call_method_safe(graph, 'degree', "list")
        degree_time = (time.perf_counter() - start) * 1000

        connectivity_ratio = largest_component / n_nodes
        isolation_ratio = isolated_nodes / n_nodes

        print(f"  Edges: {edges}")
        print(f"  Components: {n_components}")
        print(f"  Connectivity ratio: {connectivity_ratio:.1%}")
        print(f"  Isolation ratio: {isolation_ratio:.1%}")
        print(f"  Timing:")
        print(f"    Global clustering: {clustering_time:.3f}ms")
        print(f"    Local clustering: {local_clustering_time:.3f}ms")
        print(f"    Degrees: {degree_time:.3f}ms")

        # Flag concerning patterns
        if isolation_ratio > 0.7:
            print(f"    🚨 HIGH ISOLATION: {isolation_ratio:.0%} isolated nodes")
        if clustering_time < 0.1 and edges > 100:
            print(f"    🚨 SUSPICIOUSLY FAST: {clustering_time:.3f}ms for {edges} edges")


def benchmark_proximity_graphs_fair():
    """Fixed version with proper timing resolution and complete metrics comparison"""
    print("\nBenchmarking Proximity Graphs (FAIR COMPARISON - FIXED)...")
    results = []
    metrics_comparison = []

    for n_nodes in [100, 500, 1000]:
        positions = np.random.rand(n_nodes, 2) * 1000

        # Calculate meaningful threshold
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=4).fit(positions)
        distances, _ = nbrs.kneighbors(positions)
        adaptive_threshold = np.mean(distances[:, 1]) * 1.5

        print(f"  {n_nodes} nodes, threshold: {adaptive_threshold:.1f}")

        # === GRAPHIZY APPROACH ===
        start_total = time.perf_counter()

        # Graph construction
        start_construction = time.perf_counter()
        grapher = graphizy.Graphing()
        data = np.column_stack((np.arange(n_nodes), positions))
        graph = grapher.make_proximity(data, proximity_thresh=adaptive_threshold)
        graphizy_construction_time = (time.perf_counter() - start_construction) * 1000

        # Draw graph (optional visualization)
        image = grapher.draw_graph(graph, direct_show=True, kwargs_show={"block": BLOCK_DISPLAY})

        # VALIDATE GRAPH CONNECTIVITY
        graphizy_edges = len(graph.es)
        graphizy_components = len(grapher.call_method_safe(graph, 'connected_components', "raw"))

        # Clustering and connectivity analysis
        start_metrics = time.perf_counter()

        # === COMPREHENSIVE METRICS COLLECTION (GRAPHIZY) ===
        components = grapher.call_method_safe(graph, 'connected_components', "raw")
        n_components = len(components)
        largest_component_size = max(len(comp) for comp in components) if components else 0
        largest_component_ratio = largest_component_size / n_nodes if n_nodes > 0 else 0

        # Clustering metrics
        global_clustering = grapher.call_method_safe(graph, 'transitivity_undirected', "raw", default_value=0.0)
        local_clustering = grapher.call_method_safe(graph, 'transitivity_local_undirected', "list", default_value=[])
        avg_clustering = np.mean(local_clustering) if local_clustering else 0

        # Degree distribution
        degrees = grapher.call_method_safe(graph, 'degree', "list")
        avg_degree = np.mean(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        degree_std = np.std(degrees) if degrees else 0
        isolated_nodes = sum(1 for d in degrees if d == 0)

        # Graph density
        density = grapher.call_method_safe(graph, 'density', "raw", default_value=0.0)

        # Store Graphizy metrics
        graphizy_metrics = {
            "edges": graphizy_edges,
            "components": n_components,
            "largest_component_size": largest_component_size,
            "largest_component_ratio": largest_component_ratio,
            "global_clustering": global_clustering,
            "avg_local_clustering": avg_clustering,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "min_degree": min_degree,
            "degree_std": degree_std,
            "isolated_nodes": isolated_nodes,
            "isolated_ratio": isolated_nodes / n_nodes if n_nodes > 0 else 0,
            "density": density
        }

        graphizy_metrics_time = (time.perf_counter() - start_metrics) * 1000
        graphizy_total_time = (time.perf_counter() - start_total) * 1000

        # === NETWORKX APPROACH ===
        start_total = time.perf_counter()

        # Graph construction using SAME algorithm as Graphizy
        start_construction = time.perf_counter()

        # Use scipy's optimized distance calculation (same as Graphizy)
        from scipy.spatial.distance import pdist, squareform
        square_dist = squareform(pdist(positions))

        # Create NetworkX graph
        G = nx.Graph()
        for i, pos in enumerate(positions):
            G.add_node(i, x=pos[0], y=pos[1])

        # Add edges using same logic as Graphizy
        for i, row in enumerate(square_dist):
            nearby_indices = np.where((row < adaptive_threshold) & (row > 0))[0]
            for j in nearby_indices:
                if i < j:  # Avoid duplicates
                    G.add_edge(i, j, weight=row[j])

        networkx_construction_time = (time.perf_counter() - start_construction) * 1000

        # VALIDATE GRAPH CONNECTIVITY
        networkx_edges = G.number_of_edges()
        networkx_components = nx.number_connected_components(G)

        # Clustering and connectivity analysis
        start_metrics = time.perf_counter()

        # === COMPREHENSIVE METRICS COLLECTION (NETWORKX) ===
        components_nx = list(nx.connected_components(G))
        n_components_nx = len(components_nx)
        largest_component_size_nx = max(len(comp) for comp in components_nx) if components_nx else 0
        largest_component_ratio_nx = largest_component_size_nx / n_nodes if n_nodes > 0 else 0

        # Clustering metrics
        global_clustering_nx = nx.transitivity(G)
        local_clustering_nx = list(nx.clustering(G).values())
        avg_clustering_nx = np.mean(local_clustering_nx) if local_clustering_nx else 0

        # Degree distribution
        degrees_nx = list(dict(G.degree()).values())
        avg_degree_nx = np.mean(degrees_nx) if degrees_nx else 0
        max_degree_nx = max(degrees_nx) if degrees_nx else 0
        min_degree_nx = min(degrees_nx) if degrees_nx else 0
        degree_std_nx = np.std(degrees_nx) if degrees_nx else 0
        isolated_nodes_nx = sum(1 for d in degrees_nx if d == 0)

        # Graph density
        density_nx = nx.density(G)

        # Store NetworkX metrics
        networkx_metrics = {
            "edges": networkx_edges,
            "components": n_components_nx,
            "largest_component_size": largest_component_size_nx,
            "largest_component_ratio": largest_component_ratio_nx,
            "global_clustering": global_clustering_nx,
            "avg_local_clustering": avg_clustering_nx,
            "avg_degree": avg_degree_nx,
            "max_degree": max_degree_nx,
            "min_degree": min_degree_nx,
            "degree_std": degree_std_nx,
            "isolated_nodes": isolated_nodes_nx,
            "isolated_ratio": isolated_nodes_nx / n_nodes if n_nodes > 0 else 0,
            "density": density_nx
        }

        networkx_metrics_time = (time.perf_counter() - start_metrics) * 1000
        networkx_total_time = (time.perf_counter() - start_total) * 1000

        # === METRICS COMPARISON ===
        metrics_match = {}
        metrics_differences = {}

        for key in graphizy_metrics.keys():
            graphizy_val = graphizy_metrics[key]
            networkx_val = networkx_metrics[key]

            # Check if values match (with tolerance for floating point)
            if isinstance(graphizy_val, (int, float)) and isinstance(networkx_val, (int, float)):
                matches = abs(graphizy_val - networkx_val) < 1e-10
                difference = abs(graphizy_val - networkx_val)
            else:
                matches = graphizy_val == networkx_val
                difference = 0 if matches else 1

            metrics_match[key] = matches
            metrics_differences[key] = difference

        # Store complete metrics comparison
        metrics_comparison.append({
            "nodes": n_nodes,
            "threshold": adaptive_threshold,
            "graphizy": graphizy_metrics,
            "networkx": networkx_metrics,
            "matches": metrics_match,
            "differences": metrics_differences,
            "graphs_identical": all(metrics_match.values())
        })

        # VALIDATION: Check graphs are identical
        graphs_match = (graphizy_edges == networkx_edges and
                        graphizy_components == networkx_components)

        results.append({
            'nodes': n_nodes,
            'threshold': adaptive_threshold,
            'graphizy_construction_time': graphizy_construction_time,
            'graphizy_metrics_time': graphizy_metrics_time,
            'graphizy_total_time': graphizy_total_time,
            'networkx_construction_time': networkx_construction_time,
            'networkx_metrics_time': networkx_metrics_time,
            'networkx_total_time': networkx_total_time,
            'graphizy_edges': graphizy_edges,
            'networkx_edges': networkx_edges,
            'graphizy_components': graphizy_components,
            'networkx_components': networkx_components,
            'graphs_match': graphs_match,
            'construction_speedup': safe_speedup(networkx_construction_time, graphizy_construction_time),
            'metrics_speedup': safe_speedup(networkx_metrics_time, graphizy_metrics_time),
            'total_speedup': safe_speedup(networkx_total_time, graphizy_total_time)
        })

        # === DETAILED OUTPUT ===
        print(f"    Graph validation: Graphizy={graphizy_edges} edges, NetworkX={networkx_edges} edges, Match={graphs_match}")
        print(f"    Connectivity: {n_components} components, largest={largest_component_size}/{n_nodes} ({largest_component_ratio:.1%})")
        print(f"    Isolated nodes: {isolated_nodes}/{n_nodes} ({isolated_nodes / n_nodes * 100:.0f}%)")
        print(f"    Construction: Graphizy {graphizy_construction_time:.3f}ms vs NetworkX {networkx_construction_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_construction_time, graphizy_construction_time):.1f}x)")
        print(f"    Metrics:      Graphizy {graphizy_metrics_time:.3f}ms vs NetworkX {networkx_metrics_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_metrics_time, graphizy_metrics_time):.1f}x)")
        print(f"    Total:        Graphizy {graphizy_total_time:.3f}ms vs NetworkX {networkx_total_time:.3f}ms "
              f"(speedup: {safe_speedup(networkx_total_time, graphizy_total_time):.1f}x)")

        # Check for concerning graph characteristics
        if isolated_nodes / n_nodes > 0.5:
            print(f"    ⚠️  WARNING: {isolated_nodes / n_nodes * 100:.0f}% of nodes are isolated!")
        if n_components > n_nodes / 2:
            print(f"    ⚠️  WARNING: Graph is highly fragmented ({n_components} components)!")
        if not graphs_match:
            print(f"    ⚠️  WARNING: Graphs don't match! Results may not be comparable.")

        # Show metrics differences if any
        mismatched_metrics = [k for k, v in metrics_match.items() if not v]
        if mismatched_metrics:
            print(f"    ⚠️  METRICS MISMATCH: {mismatched_metrics}")
            for metric in mismatched_metrics:
                print(f"        {metric}: Graphizy={graphizy_metrics[metric]:.6f}, NetworkX={networkx_metrics[metric]:.6f}")

    # === SUMMARY METRICS ANALYSIS ===
    print(f"\n{'=' * 60}")
    print("METRICS VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    for i, comparison in enumerate(metrics_comparison):
        nodes = comparison["nodes"]
        print(f"\n{nodes} nodes:")
        print(f"  Threshold: {comparison['threshold']:.1f}")
        print(f"  Graphs identical: {comparison['graphs_identical']}")
        print(f"  Graph structure:")
        print(f"    Edges: {comparison['graphizy']['edges']}")
        print(f"    Components: {comparison['graphizy']['components']}")
        print(f"    Largest component: {comparison['graphizy']['largest_component_size']}/{nodes} ({comparison['graphizy']['largest_component_ratio']:.1%})")
        print(f"    Isolated nodes: {comparison['graphizy']['isolated_nodes']} ({comparison['graphizy']['isolated_ratio']:.1%})")
        print(f"    Density: {comparison['graphizy']['density']:.6f}")
        print(f"    Avg degree: {comparison['graphizy']['avg_degree']:.2f}")
        print(f"    Global clustering: {comparison['graphizy']['global_clustering']:.6f}")

    return results, metrics_comparison


def convert_numpy_types(obj):
    """Convert numpy types to regular Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE PROXIMITY GRAPH BENCHMARK")
    print("Graphizy vs NetworkX with detailed metrics comparison")
    print("=" * 80)

    # Run timing diagnostic first
    debug_timing_issue()

    # Run diagnostic analysis
    benchmark_proximity_graphs_diagnostic()

    # Run connectivity impact analysis
    analyze_connectivity_impact()

    # Run main fair comparison benchmark
    fair_results, metrics_comparison = benchmark_proximity_graphs_fair()

    print("\n" + "=" * 50)
    print("FAIR COMPARISON RESULTS SUMMARY")
    print("=" * 50)

    for result in fair_results:
        print(f"Nodes: {result['nodes']}, Edges: {result['graphizy_edges']}, "
              f"Construction Speedup: {result['construction_speedup']:.1f}x, "
              f"Metrics Speedup: {result['metrics_speedup']:.1f}x, "
              f"Total Speedup: {result['total_speedup']:.1f}x")

    # Save detailed results
    print(f"\n" + "=" * 50)
    print("SAVING RESULTS")
    print("=" * 50)

    # Save benchmark results
    with open('proximity_benchmark_results.json', 'w') as f:
        json.dump(convert_numpy_types(fair_results), f, indent=2)
    print("Benchmark results saved to 'proximity_benchmark_results.json'")

    # Save detailed metrics comparison
    with open('proximity_metrics_comparison.json', 'w') as f:
        json.dump(convert_numpy_types(metrics_comparison), f, indent=2)
    print("Detailed metrics comparison saved to 'proximity_metrics_comparison.json'")

    # Final analysis
    print(f"\n" + "=" * 50)
    print("FINAL ANALYSIS")
    print("=" * 50)

    avg_construction_speedup = np.mean([r['construction_speedup'] for r in fair_results])
    avg_metrics_speedup = np.mean([r['metrics_speedup'] for r in fair_results])
    avg_total_speedup = np.mean([r['total_speedup'] for r in fair_results])

    print(f"Average speedups:")
    print(f"  Construction: {avg_construction_speedup:.1f}x")
    print(f"  Metrics:      {avg_metrics_speedup:.1f}x")
    print(f"  Total:        {avg_total_speedup:.1f}x")

    # Check for any concerning patterns
    all_graphs_identical = all(comp['graphs_identical'] for comp in metrics_comparison)
    print(f"\nAll graphs identical between libraries: {all_graphs_identical}")

    if not all_graphs_identical:
        print("⚠️  Some graphs differ between libraries - check metrics comparison file!")

    # Check connectivity patterns
    isolation_ratios = [comp['graphizy']['isolated_ratio'] for comp in metrics_comparison]
    avg_isolation = np.mean(isolation_ratios)

    if avg_isolation > 0.3:
        print(f"⚠️  High average isolation ratio: {avg_isolation:.1%} - graphs may be too sparse!")
    else:
        print(f"✅ Reasonable connectivity - average isolation ratio: {avg_isolation:.1%}")

    print(f"\nBenchmark complete! Check the JSON files for detailed analysis.")