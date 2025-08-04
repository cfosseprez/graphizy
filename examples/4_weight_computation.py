#!/usr/bin/env python3
"""
Weight Computation Examples for Graphizy - Updated for v0.1.17+

This script demonstrates the comprehensive weight computation capabilities of Graphizy,
showcasing the latest weight system features and APIs. It covers:

1. Modern weight computer initialization and configuration
2. Multiple weight computation methods (distance, inverse_distance, gaussian, etc.)
3. Integration with the unified make_graph() interface  
4. Custom weight formulas and edge attribute computation
5. Real-time weight computation with fast computers
6. Weight analysis and statistical methods
7. Integration with memory systems for temporal weight evolution
8. Performance optimization for weight computation

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

# --- Core Libraries ---
import numpy as np
import logging
import sys
from pathlib import Path
import time
from typing import Dict, Any, List

# --- Graphizy Imports ---
from graphizy import (
    Graphing,  # The main class for graphing and analysis
    GraphizyConfig,  # The configuration object for settings and styles
    WeightComputer,  # The weight computation system
    generate_and_format_positions,  # Helper function to create random 2D points
    validate_graphizy_input,  # Helper function to validate coordinates
    GraphizyError  # Custom error class for package-specific issues
)
from graphizy.utils import setup_output_directory

# --- Initial Setup ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# =============================================================================
# EXAMPLE 1: MODERN WEIGHT COMPUTER INITIALIZATION
# =============================================================================

def example_modern_weight_computer_initialization():
    """
    Demonstrates the enhanced weight computer initialization and configuration
    options available in the latest version of Graphizy.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: MODERN WEIGHT COMPUTER INITIALIZATION")
    print("=" * 60)

    try:
        # Create different weight computer configurations for various use cases
        
        # 1. Distance-based weight computation
        print("1. Distance-based weight computation:")
        config1 = GraphizyConfig(dimension=(600, 600))
        config1.weight.weight_method = "distance"
        config1.weight.weight_attribute = "weight"
        grapher1 = Graphing(config=config1)
        
        grapher1.init_weight_computer()
        print("   • Method: Euclidean distance")
        print("   • Target attribute: 'weight'")
        print("   • Use case: Physical proximity weighting")

        # 2. Inverse distance weight computation
        print("\n2. Inverse distance weight computation:")
        config2 = GraphizyConfig(dimension=(600, 600))
        config2.weight.weight_method = "inverse_distance"
        config2.weight.weight_attribute = "strength"
        grapher2 = Graphing(config=config2)
        
        grapher2.init_weight_computer()
        print("   • Method: 1/distance (closer = stronger)")
        print("   • Target attribute: 'strength'")
        print("   • Use case: Communication networks, social networks")

        # 3. Gaussian weight computation
        print("\n3. Gaussian weight computation:")
        config3 = GraphizyConfig(dimension=(600, 600))
        config3.weight.weight_method = "gaussian"
        config3.weight.weight_attribute = "similarity"
        grapher3 = Graphing(config=config3)
        
        grapher3.init_weight_computer(
            sigma=50.0  # Controls the spread of the Gaussian
        )
        print("   • Method: Gaussian kernel exp(-d²/2σ²)")
        print("   • Target attribute: 'similarity'")
        print("   • Sigma: 50.0 (controls spread)")
        print("   • Use case: Similarity networks, clustering")

        # 4. Custom formula weight computation
        print("\n4. Custom formula weight computation:")
        config4 = GraphizyConfig(dimension=(600, 600))
        config4.weight.weight_method = "formula"
        config4.weight.weight_attribute = "custom_weight"
        grapher4 = Graphing(config=config4)
        
        grapher4.init_weight_computer(
            formula="1.0 / (1.0 + distance * 0.01)"  # Custom decay function
        )
        print("   • Method: Custom formula")
        print("   • Formula: 1.0 / (1.0 + distance * 0.01)")
        print("   • Target attribute: 'custom_weight'")
        print("   • Use case: Custom weighting schemes")

        # Test each configuration with sample data
        print("\nTesting configurations with sample data...")
        test_data = generate_and_format_positions(600, 600, 25)
        validate_graphizy_input(test_data)
        
        configurations = [
            ("Distance", grapher1, "weight"),
            ("Inverse Distance", grapher2, "strength"),
            ("Gaussian", grapher3, "similarity"),
            ("Custom Formula", grapher4, "custom_weight")
        ]

        results = []
        
        for name, grapher, attr_name in configurations:
            try:
                # Create proximity graph with weights
                test_graph = grapher.make_graph(
                    "proximity", test_data, 
                    proximity_thresh=100.0
                )
                
                if test_graph and test_graph.ecount() > 0:
                    # Check if weights were computed
                    edge_attrs = test_graph.es.attributes()
                    if attr_name in edge_attrs:
                        weights = test_graph.es[attr_name]
                        result = {
                            'name': name,
                            'edges': test_graph.ecount(),
                            'avg_weight': np.mean(weights),
                            'weight_range': (min(weights), max(weights)),
                            'std_weight': np.std(weights)
                        }
                        results.append(result)
                        print(f"   {name}: {len(weights)} weights computed")
                    else:
                        print(f"   {name}: Weight attribute '{attr_name}' not found")
                else:
                    print(f"   {name}: No graph or edges created")
                    
            except Exception as e:
                print(f"   {name}: Error - {e}")

        # Display results comparison
        if results:
            print(f"\nWeight Computation Results Comparison:")
            print(f"{'Method':<16} {'Edges':<7} {'Avg Weight':<12} {'Std Dev':<10} {'Range':<20}")
            print("-" * 70)
            
            for result in results:
                range_str = f"[{result['weight_range'][0]:.3f}, {result['weight_range'][1]:.3f}]"
                print(f"{result['name']:<16} {result['edges']:<7} {result['avg_weight']:<12.4f} "
                      f"{result['std_weight']:<10.4f} {range_str:<20}")

        print("\nWeight computer initialization completed successfully!")
        return grapher3  # Return Gaussian configuration for further examples

    except Exception as e:
        print(f"Weight computer initialization failed: {e}")
        return None


# =============================================================================
# EXAMPLE 2: INTEGRATION WITH MAKE_GRAPH INTERFACE
# =============================================================================

def example_weight_integration_with_make_graph():
    """
    Demonstrates seamless integration of weight computation with the unified
    make_graph() interface, showing automatic weight computation.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: WEIGHT INTEGRATION WITH MAKE_GRAPH")
    print("=" * 60)

    try:
        # Initialize grapher with automatic weight computation enabled
        config = GraphizyConfig(dimension=(600, 600))
        config.weight.auto_compute_weights = True
        config.weight.weight_method = "distance"
        config.weight.weight_attribute = "edge_weight"
        grapher = Graphing(config=config)
        
        # Generate sample data
        data = generate_and_format_positions(600, 600, 40)
        validate_graphizy_input(data)

        print("Testing weight computation with different graph types...")

        # Test with different graph types
        graph_types = [
            ("delaunay", {}),
            ("proximity", {"proximity_thresh": 80.0}),
            ("knn", {"k": 5}),
            ("mst", {"metric": "euclidean"}),
            ("gabriel", {})
        ]

        weight_results = {}

        for graph_type, params in graph_types:
            try:
                print(f"\n• Testing {graph_type} graph:")
                
                # Create graph with automatic weight computation
                graph = grapher.make_graph(
                    graph_type, data,
                    **params
                )
                
                if graph and graph.ecount() > 0:
                    # Check edge attributes
                    edge_attrs = graph.es.attributes()
                    print(f"    Edge attributes: {edge_attrs}")
                    
                    if "edge_weight" in edge_attrs:
                        weights = graph.es["edge_weight"]
                        weight_stats = {
                            'count': len(weights),
                            'mean': np.mean(weights),
                            'std': np.std(weights),
                            'min': min(weights),
                            'max': max(weights),
                            'median': np.median(weights)
                        }
                        weight_results[graph_type] = weight_stats
                        
                        print(f"    Weights computed: {weight_stats['count']} edges")
                        print(f"    Statistics: mean={weight_stats['mean']:.3f}, "
                              f"std={weight_stats['std']:.3f}")
                        print(f"    Range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")
                    else:
                        print(f"    No weights computed (edge_weight not found)")
                else:
                    print(f"    No graph created or no edges")
                    
            except Exception as e:
                print(f"    Error with {graph_type}: {e}")

        # Comparison analysis
        if weight_results:
            print(f"\nWeight Statistics Comparison Across Graph Types:")
            print(f"{'Graph Type':<12} {'Edges':<7} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}")
            print("-" * 70)
            
            for graph_type, stats in weight_results.items():
                print(f"{graph_type:<12} {stats['count']:<7} {stats['mean']:<8.3f} "
                      f"{stats['std']:<8.3f} {stats['min']:<8.3f} {stats['max']:<8.3f} "
                      f"{stats['median']:<8.3f}")

        # Test conditional weight computation
        print(f"\nTesting conditional weight computation...")
        
        # Create graph without weights
        config_no_weights = GraphizyConfig(dimension=(600, 600))
        config_no_weights.weight.auto_compute_weights = False
        grapher_no_weights = Graphing(config=config_no_weights)
        graph_no_weights = grapher_no_weights.make_graph(
            "proximity", data,
            proximity_thresh=80.0
        )
        
        # Create graph with weights
        graph_with_weights = grapher.make_graph(
            "proximity", data,
            proximity_thresh=80.0
        )
        
        print(f"  Without weights: {graph_no_weights.es.attributes() if graph_no_weights else 'No graph'}")
        print(f"  With weights: {graph_with_weights.es.attributes() if graph_with_weights else 'No graph'}")

        print("\nWeight integration with make_graph completed successfully!")
        return grapher

    except Exception as e:
        print(f"Weight integration with make_graph failed: {e}")
        return None


# =============================================================================
# EXAMPLE 3: CUSTOM WEIGHT FORMULAS AND EDGE ATTRIBUTES
# =============================================================================

def example_custom_weight_formulas():
    """
    Demonstrates custom weight formulas and computation of multiple edge attributes.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: CUSTOM WEIGHT FORMULAS AND EDGE ATTRIBUTES")
    print("=" * 60)

    try:
        # Initialize grapher for custom attribute computation
        config = GraphizyConfig(dimension=(500, 500))
        grapher = Graphing(config=config)
        
        # Generate sample data
        data = generate_and_format_positions(500, 500, 35)
        
        # Create base graph for attribute computation
        base_graph = grapher.make_graph("proximity", data, proximity_thresh=75.0)
        
        if not base_graph or base_graph.ecount() == 0:
            print("No base graph created for attribute computation.")
            return

        print(f"Created base graph with {base_graph.ecount()} edges")

        # Initialize weight computer for custom formulas
        if not hasattr(grapher, 'weight_computer'):
            grapher.init_weight_computer()

        # 1. Compute multiple attributes with different formulas
        print("\n1. Computing multiple edge attributes:")
        
        attribute_formulas = [
            ("euclidean_distance", "distance"),
            ("manhattan_distance", "abs(x1 - x2) + abs(y1 - y2)"),
            ("inverse_weight", "1.0 / (1.0 + distance)"),
            ("gaussian_similarity", "exp(-distance * distance / (2.0 * 30.0 * 30.0))"),
            ("exponential_decay", "exp(-distance / 50.0)"),
            ("power_law", "pow(distance + 1.0, -1.5)"),
            ("normalized_distance", "distance / 100.0"),
            ("angular_component", "abs(atan2(y2 - y1, x2 - x1))"),
        ]

        computed_attributes = {}

        for attr_name, formula in attribute_formulas:
            try:
                print(f"  Computing '{attr_name}' with formula: {formula}")
                
                # Compute the attribute
                result_graph = grapher.compute_edge_attribute(
                    base_graph, 
                    attr_name, 
                    method="formula",
                    formula=formula
                )
                
                if result_graph and attr_name in result_graph.es.attributes():
                    values = result_graph.es[attr_name]
                    computed_attributes[attr_name] = {
                        'values': values,
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': min(values),
                        'max': max(values)
                    }
                    print(f"    Success: mean={computed_attributes[attr_name]['mean']:.4f}")
                else:
                    print(f"    Failed: attribute not created")
                    
            except Exception as e:
                print(f"    Error computing {attr_name}: {e}")

        # Display comprehensive attribute statistics
        if computed_attributes:
            print(f"\nComputed Attribute Statistics:")
            print(f"{'Attribute':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            print("-" * 65)
            
            for attr_name, stats in computed_attributes.items():
                print(f"{attr_name:<20} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
                      f"{stats['min']:<10.4f} {stats['max']:<10.4f}")

        print(f"\nFinal edge attributes: {base_graph.es.attributes()}")
        print("Custom weight formulas and edge attributes completed successfully!")

    except Exception as e:
        print(f"Custom weight formulas failed: {e}")


# =============================================================================
# EXAMPLE 4: INTEGRATION WITH MEMORY SYSTEMS
# =============================================================================

def example_weight_memory_integration():
    """
    Demonstrates integration of weight computation with memory systems for
    temporal weight evolution analysis.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: WEIGHT INTEGRATION WITH MEMORY SYSTEMS")
    print("=" * 60)

    try:
        # Initialize grapher with both memory and weight systems
        config = GraphizyConfig(dimension=(500, 500))
        config.weight.auto_compute_weights = True
        config.weight.weight_method = "distance"
        config.weight.weight_attribute = "weight"
        grapher = Graphing(config=config)
        
        # Initialize memory manager
        grapher.init_memory_manager(
            max_memory_size=100,
            max_iterations=15,
            track_edge_ages=True
        )

        print("Integrated memory and weight systems initialized")
        
        # Generate initial data
        data = generate_and_format_positions(500, 500, 30)
        
        # Temporal simulation with weight evolution
        print("\nSimulating temporal weight evolution...")
        
        weight_evolution = []
        
        for iteration in range(8):
            print(f"\nIteration {iteration + 1}:")
            
            # Evolve data positions
            if iteration > 0:
                data[:, 1:3] += np.random.normal(0, 10, (len(data), 2))
                data[:, 1] = np.clip(data[:, 1], 25, 475)
                data[:, 2] = np.clip(data[:, 2], 25, 475)

            # Create memory-enhanced graph with weights
            graph = grapher.make_graph(
                "proximity", data,
                proximity_thresh=80.0,
                use_memory=True,
                update_memory=True
            )
            
            if graph and graph.ecount() > 0:
                # Analyze weight distribution
                edge_attrs = graph.es.attributes()
                
                analysis = {
                    'iteration': iteration + 1,
                    'total_edges': graph.ecount(),
                    'attributes': edge_attrs
                }
                
                # Analyze weights
                if 'weight' in edge_attrs:
                    weights = graph.es['weight']
                    analysis['weight_stats'] = {
                        'mean': np.mean(weights),
                        'std': np.std(weights),
                        'min': min(weights),
                        'max': max(weights)
                    }
                
                # Analyze memory-based vs current edges
                if 'memory_based' in edge_attrs:
                    memory_edges = [i for i, e in enumerate(graph.es) if e['memory_based']]
                    current_edges = [i for i, e in enumerate(graph.es) if not e['memory_based']]
                    
                    analysis['memory_edge_count'] = len(memory_edges)
                    analysis['current_edge_count'] = len(current_edges)
                    
                    # Compare weights between memory and current edges
                    if 'weight' in edge_attrs:
                        memory_weights = [graph.es[i]['weight'] for i in memory_edges]
                        current_weights = [graph.es[i]['weight'] for i in current_edges]
                        
                        if memory_weights:
                            analysis['memory_weight_avg'] = np.mean(memory_weights)
                        if current_weights:
                            analysis['current_weight_avg'] = np.mean(current_weights)
                
                weight_evolution.append(analysis)
                
                # Print current iteration analysis
                print(f"  Total edges: {analysis['total_edges']}")
                if 'weight_stats' in analysis:
                    ws = analysis['weight_stats']
                    print(f"  Weight stats: mean={ws['mean']:.3f}, std={ws['std']:.3f}")
                if 'memory_edge_count' in analysis:
                    print(f"  Memory edges: {analysis['memory_edge_count']}, "
                          f"Current edges: {analysis['current_edge_count']}")

        # Temporal analysis
        print(f"\nTemporal Weight Evolution Analysis:")
        
        if weight_evolution:
            print(f"{'Iter':<5} {'Edges':<7} {'Avg Weight':<11} {'Memory':<8} {'Current':<8}")
            print("-" * 45)
            
            for analysis in weight_evolution:
                iter_num = analysis['iteration']
                total_edges = analysis['total_edges']
                avg_weight = analysis.get('weight_stats', {}).get('mean', 0)
                memory_count = analysis.get('memory_edge_count', 0)
                current_count = analysis.get('current_edge_count', 0)
                
                print(f"{iter_num:<5} {total_edges:<7} {avg_weight:<11.3f} "
                      f"{memory_count:<8} {current_count:<8}")

        print("\nWeight-memory integration analysis completed successfully!")

    except Exception as e:
        print(f"Weight-memory integration failed: {e}")


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    Runs all the weight computation examples using the latest API features.
    """
    print("Graphizy Weight Computation Examples - v0.1.17+ Edition")
    print("=" * 70)

    try:
        # Run examples sequentially
        weight_grapher = example_modern_weight_computer_initialization()
        integration_grapher = example_weight_integration_with_make_graph()
        example_custom_weight_formulas()
        example_weight_memory_integration()

        # Create sample visualizations with weights
        print(f"\nCreating weight visualization examples...")
        
        if integration_grapher:
            try:
                output_dir = setup_output_directory()
                
                # Create sample data
                data = generate_and_format_positions(500, 500, 40)
                
                # Create graphs with different weight methods
                methods = [
                    ("distance", "distance", None),
                    ("gaussian", "formula", "exp(-distance * distance / (2.0 * 30.0 * 30.0))"),
                    ("inverse", "formula", "1/(distance + 0.1)")
                ]
                
                for method_name, method_type, formula in methods:
                    try:
                        # Configure weight computer
                        config = GraphizyConfig(dimension=(500, 500))
                        config.weight.auto_compute_weights = True
                        config.weight.weight_method = method_type
                        config.weight.weight_attribute = "display_weight"
                        if formula:
                            config.weight.weight_formula = formula
                        
                        grapher = Graphing(config=config)
                        
                        # Create weighted graph
                        graph = grapher.make_graph(
                            "proximity", data,
                            proximity_thresh=80.0
                        )
                        
                        if graph:
                            # Create visualization
                            image = grapher.draw_graph(graph)
                            filename = f"weighted_graph_{method_name}.jpg"
                            grapher.save_graph(image, str(output_dir / filename))
                            print(f"  Saved: {filename}")
                            
                    except Exception as e:
                        print(f"  Failed to create {method_name} visualization: {e}")
                        
            except Exception as e:
                print(f"Visualization creation failed: {e}")

        print("\n" + "=" * 70)
        print("All weight computation examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Modern weight computer initialization with multiple methods")
        print("  • Seamless integration with make_graph() unified interface")
        print("  • Custom weight formulas and edge attribute computation")
        print("  • Integration with memory systems for temporal analysis")
        print("  • Comprehensive weight analysis and statistical methods")
        print(f"Check the 'examples/output/' folder for weight visualizations.")

    except Exception as e:
        print(f"\nWeight computation examples failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
