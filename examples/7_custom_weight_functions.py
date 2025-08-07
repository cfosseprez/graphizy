#!/usr/bin/env python3
"""
Example showing how to use custom callable functions for weight computation in Graphizy.
"""

import numpy as np
from graphizy import Graphing, GraphizyConfig, WeightComputer, generate_and_format_positions

# Example 1: Using a custom function with WeightComputer directly
def custom_weight_function(graph, **params):
    """
    Custom weight function that takes a graph and returns a list of weights.
    
    Args:
        graph: igraph Graph object
        **params: Additional parameters passed to the function
    
    Returns:
        List of weights, one per edge
    """
    # Access edge distances if they exist
    if 'distance' in graph.es.attributes():
        distances = graph.es['distance']
    else:
        # Compute distances if not present
        distances = []
        for edge in graph.es:
            source_x, source_y = graph.vs[edge.source]['x'], graph.vs[edge.source]['y']
            target_x, target_y = graph.vs[edge.target]['x'], graph.vs[edge.target]['y']
            dist = np.sqrt((target_x - source_x)**2 + (target_y - source_y)**2)
            distances.append(dist)
    
    # Custom weight calculation with parameters
    decay_rate = params.get('decay_rate', 0.05)
    threshold = params.get('threshold', 50.0)
    
    weights = []
    for d in distances:
        if d <= threshold:
            # Strong connection for close nodes
            weight = np.exp(-decay_rate * d)
        else:
            # Weak connection for distant nodes
            weight = 0.1 * np.exp(-decay_rate * d)
        weights.append(weight)
    
    return weights


# Example 2: Using custom function with init_weight_computer
def example_custom_weight_function():
    """Demonstrates using custom callable functions for weight computation."""
    print("\n" + "=" * 60)
    print("CUSTOM WEIGHT FUNCTION EXAMPLE")
    print("=" * 60)
    
    # Method 1: Direct WeightComputer usage
    print("\n1. Using WeightComputer directly with custom function:")
    
    # Create sample data
    data = generate_and_format_positions(400, 400, 30)
    
    # Create a basic graph first
    config = GraphizyConfig(dimension=(400, 400))
    grapher = Graphing(config=config)
    graph = grapher.make_graph("proximity", data, proximity_thresh=80.0)
    
    if graph and graph.ecount() > 0:
        # Create weight computer with custom function
        weight_computer = WeightComputer(
            method="function",
            custom_function=custom_weight_function,
            target_attribute="custom_weight"
        )
        
        # Compute weights with custom parameters
        graph = weight_computer.compute_weights(graph)
        
        # Check results
        if 'custom_weight' in graph.es.attributes():
            weights = graph.es['custom_weight']
            print(f"   ✓ Custom weights computed: {len(weights)} edges")
            print(f"   Weight stats: mean={np.mean(weights):.3f}, std={np.std(weights):.3f}")
            print(f"   Weight range: [{min(weights):.3f}, {max(weights):.3f}]")
        else:
            print("   ✗ Custom weight attribute not created")
    
    # Method 2: Using init_weight_computer with custom function
    print("\n2. Using init_weight_computer with custom function:")
    
    # Create new grapher
    config2 = GraphizyConfig(dimension=(400, 400))
    grapher2 = Graphing(config=config2)
    
    # Initialize weight computer with custom function
    grapher2.init_weight_computer(
        method="function",
        custom_function=custom_weight_function,
        target_attribute="my_weight"
    )
    
    # Create graph - weights will be computed automatically if auto_compute_weights is True
    config2.weight.auto_compute_weights = True
    graph2 = grapher2.make_graph("proximity", data, proximity_thresh=80.0)
    
    if graph2 and 'my_weight' in graph2.es.attributes():
        weights2 = graph2.es['my_weight']
        print(f"   ✓ Auto-computed custom weights: {len(weights2)} edges")
        print(f"   Weight stats: mean={np.mean(weights2):.3f}, std={np.std(weights2):.3f}")
    
    # Method 3: More complex custom function with edge attributes
    print("\n3. Custom function using multiple edge attributes:")
    
    def multi_attribute_weight(graph, **params):
        """Weight based on multiple factors."""
        weights = []
        
        for edge in graph.es:
            # Get edge attributes using proper igraph syntax
            distance = edge['distance'] if 'distance' in graph.es.attributes() else 1.0
            
            # If graph has memory system, we might have age
            age = edge['age'] if 'age' in graph.es.attributes() else 0
            
            # Multi-factor weight
            distance_factor = 1.0 / (1.0 + distance * 0.01)
            age_factor = np.exp(-0.1 * age) if age > 0 else 1.0
            
            # Combine factors
            alpha = params.get('alpha', 0.7)  # Weight for distance vs age
            weight = alpha * distance_factor + (1 - alpha) * age_factor
            
            weights.append(weight)
        
        return weights
    
    # Use the multi-attribute function
    grapher3 = Graphing(config=GraphizyConfig(dimension=(400, 400)))
    grapher3.init_weight_computer(
        method="function",
        custom_function=multi_attribute_weight,
        target_attribute="multi_weight",
        alpha=0.8  # More weight on distance
    )
    
    graph3 = grapher3.make_graph("proximity", data, proximity_thresh=80.0)
    if graph3 and 'multi_weight' in graph3.es.attributes():
        print(f"   ✓ Multi-attribute weights computed successfully")
    
    # Method 4: Using compute_edge_attribute with custom function
    print("\n4. Using compute_edge_attribute with custom function:")
    
    if hasattr(grapher, 'compute_edge_attribute'):
        # Define inline custom function
        def edge_importance(graph, base_importance=1.0):
            # Compute importance based on node degrees
            importance = []
            for edge in graph.es:
                source_degree = graph.degree(edge.source)
                target_degree = graph.degree(edge.target)
                # Higher importance for edges connecting high-degree nodes
                imp = base_importance * np.sqrt(source_degree * target_degree)
                importance.append(imp)
            return importance
        
        # Apply custom function
        graph = grapher.compute_edge_attribute(
            graph,
            "importance",
            method="function",
            custom_function=edge_importance,
            base_importance=0.5
        )
        
        if 'importance' in graph.es.attributes():
            imp_values = graph.es['importance']
            print(f"   ✓ Edge importance computed: mean={np.mean(imp_values):.3f}")
    
    print("\nCustom weight function examples completed!")


# Example 3: Advanced custom weight scenarios
def example_advanced_custom_weights():
    """Shows advanced custom weight computation scenarios."""
    print("\n" + "=" * 60)
    print("ADVANCED CUSTOM WEIGHT SCENARIOS")
    print("=" * 60)
    
    data = generate_and_format_positions(300, 300, 25)
    
    # Scenario 1: Time-varying weights
    print("\n1. Time-varying weight function:")
    
    def time_varying_weight(graph, time_step=0, frequency=0.1):
        """Weights that change over time."""
        weights = []
        for edge in graph.es:
            distance = edge['distance'] if 'distance' in graph.es.attributes() else 1.0
            # Oscillating weight based on time
            time_factor = 0.5 + 0.5 * np.sin(frequency * time_step)
            weight = (1.0 / (1.0 + distance * 0.01)) * time_factor
            weights.append(weight)
        return weights
    
    grapher = Graphing(config=GraphizyConfig(dimension=(300, 300)))
    graph = grapher.make_graph("proximity", data, proximity_thresh=70.0)
    
    # Compute weights at different time steps
    for t in [0, 10, 20]:
        weight_comp = WeightComputer(
            method="function",
            custom_function=time_varying_weight,
            target_attribute=f"weight_t{t}",
            time_step=t
        )
        graph = weight_comp.compute_weights(graph)
        
        if f"weight_t{t}" in graph.es.attributes():
            weights = graph.es[f"weight_t{t}"]
            print(f"   Time {t}: mean weight = {np.mean(weights):.3f}")
    
    # Scenario 2: Threshold-based custom weight
    print("\n2. Complex threshold-based weight:")
    
    def complex_threshold_weight(graph, thresholds=[30, 60, 90], values=[1.0, 0.5, 0.2, 0.1]):
        """Multi-level threshold weight function."""
        weights = []
        for edge in graph.es:
            distance = edge['distance'] if 'distance' in graph.es.attributes() else 0
            
            # Find appropriate weight based on thresholds
            weight = values[-1]  # Default to last value
            for i, thresh in enumerate(thresholds):
                if distance <= thresh:
                    weight = values[i]
                    break
            
            weights.append(weight)
        return weights
    
    weight_comp2 = WeightComputer(
        method="function",
        custom_function=complex_threshold_weight,
        target_attribute="threshold_weight"
    )
    graph = weight_comp2.compute_weights(graph)
    
    if 'threshold_weight' in graph.es.attributes():
        weights = graph.es['threshold_weight']
        unique_weights = set(weights)
        print(f"   ✓ Threshold weights computed")
        print(f"   Unique weight values: {sorted(unique_weights)}")
    
    print("\nAdvanced custom weight examples completed!")


if __name__ == "__main__":
    example_custom_weight_function()
    example_advanced_custom_weights()
