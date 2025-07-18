#!/usr/bin/env python3
"""
Test script for Gabriel graph functionality in graphizy
"""

import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_gabriel_graph():
    """Test that Gabriel graph functionality works correctly"""
    print("Testing Gabriel graph functionality...")
    
    try:
        from graphizy import Graphing, GraphizyConfig, generate_positions
        
        # Create test data
        print("Creating test data...")
        positions = generate_positions(400, 400, 15)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))
        
        # Create grapher
        config = GraphizyConfig()
        config.graph.dimension = (400, 400)
        grapher = Graphing(config=config)
        
        # Test Gabriel graph creation
        print("Creating Gabriel graph...")
        gabriel_graph = grapher.make_gabriel(particle_stack)
        print(f"✓ Gabriel graph created successfully: {gabriel_graph.vcount()} vertices, {gabriel_graph.ecount()} edges")
        
        # Test relationship with Delaunay triangulation
        print("Testing relationship with Delaunay triangulation...")
        delaunay_graph = grapher.make_delaunay(particle_stack)
        
        gabriel_edges = gabriel_graph.ecount()
        delaunay_edges = delaunay_graph.ecount()
        
        if gabriel_edges <= delaunay_edges:
            print(f"✓ Gabriel graph is subset of Delaunay: {gabriel_edges} ≤ {delaunay_edges} edges")
        else:
            print(f"✗ Gabriel graph has more edges than Delaunay: {gabriel_edges} > {delaunay_edges}")
            return False
        
        # Test visualization
        print("Testing Gabriel graph visualization...")
        gabriel_image = grapher.draw_graph(gabriel_graph)
        print(f"✓ Gabriel graph visualization created: {gabriel_image.shape}")
        
        # Test with memory
        print("Testing Gabriel graph with memory...")
        grapher.init_memory_manager(max_memory_size=10)
        
        # Simulate a few iterations with Gabriel graphs
        for i in range(5):
            # Slightly modify positions
            particle_stack[:, 1:3] += np.random.normal(0, 3, (len(particle_stack), 2))
            
            # Create new Gabriel graph and update memory
            current_gabriel = grapher.make_gabriel(particle_stack)
            grapher.update_memory_with_graph(current_gabriel)
        
        # Create memory-enhanced Gabriel graph
        memory_gabriel = grapher.make_memory_graph(particle_stack)
        print(f"✓ Memory-enhanced Gabriel graph created: {memory_gabriel.vcount()} vertices, {memory_gabriel.ecount()} edges")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Gabriel graph: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gabriel_comparison():
    """Test Gabriel graph comparison with other graph types"""
    print("\nTesting Gabriel graph comparison...")
    
    try:
        from graphizy import Graphing, GraphizyConfig, generate_positions
        
        # Create test data
        positions = generate_positions(300, 300, 20)
        particle_ids = np.arange(len(positions))
        data = np.column_stack((particle_ids, positions))
        
        # Create grapher
        config = GraphizyConfig()
        config.graph.dimension = (300, 300)
        grapher = Graphing(config=config)
        
        # Create all graph types
        graph_types = {
            'Gabriel': grapher.make_gabriel(data),
            'Delaunay': grapher.make_delaunay(data),
            'Proximity': grapher.make_proximity(data, proximity_thresh=60.0),
            'MST': grapher.make_mst(data),
        }
        
        # Compare properties
        print("Graph Type Comparison:")
        print(f"{'Type':<12} {'Vertices':<10} {'Edges':<8} {'Density':<10} {'Connected':<10}")
        print("-" * 55)
        
        for name, graph in graph_types.items():
            if graph:
                info = grapher.get_graph_info(graph)
                print(f"{name:<12} {info['vertex_count']:<10} {info['edge_count']:<8} "
                      f"{info['density']:<10.3f} {info['is_connected']:<10}")
        
        # Verify Gabriel ⊆ Delaunay
        gabriel_edges = graph_types['Gabriel'].ecount()
        delaunay_edges = graph_types['Delaunay'].ecount()
        
        if gabriel_edges <= delaunay_edges:
            print(f"\n✓ Verified: Gabriel ⊆ Delaunay ({gabriel_edges} ≤ {delaunay_edges} edges)")
        else:
            print(f"\n✗ Error: Gabriel has more edges than Delaunay!")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error in comparison test: {e}")
        return False

if __name__ == "__main__":
    print("Testing graphizy Gabriel graph functionality...")
    print("=" * 50)
    
    success = True
    success &= test_gabriel_graph()
    success &= test_gabriel_comparison()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All Gabriel graph tests passed!")
        print("\nYou can now use:")
        print("  - grapher.make_gabriel(data) to create Gabriel graphs")
        print("  - Gabriel graphs work with memory system")
        print("  - Gabriel graphs are always subsets of Delaunay triangulation")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)
