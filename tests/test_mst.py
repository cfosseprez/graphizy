#!/usr/bin/env python3
"""
Test script for MST functionality in graphizy
"""

import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_mst_functionality():
    """Test that MST functionality works correctly"""
    print("Testing MST functionality...")
    
    try:
        from graphizy import Graphing, GraphizyConfig, generate_positions
        
        # Create test data
        print("Creating test data...")
        positions = generate_positions(800, 600, 20)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))
        
        # Create grapher
        config = GraphizyConfig()
        config.graph.dimension = (800, 600)
        grapher = Graphing(config=config)
        
        # Test MST creation
        print("Creating MST graph...")
        mst_graph = grapher.make_mst(particle_stack)
        print(f"✓ MST created successfully: {mst_graph.vcount()} vertices, {mst_graph.ecount()} edges")
        
        # Verify MST properties
        # An MST of n vertices should have exactly n-1 edges
        expected_edges = mst_graph.vcount() - 1
        actual_edges = mst_graph.ecount()
        
        if actual_edges == expected_edges:
            print(f"✓ MST has correct number of edges: {actual_edges} (expected: {expected_edges})")
        else:
            print(f"✗ MST has wrong number of edges: {actual_edges} (expected: {expected_edges})")
            return False
        
        # Test that the graph is connected
        is_connected = grapher.call_method(mst_graph, 'is_connected')
        if is_connected:
            print("✓ MST is connected")
        else:
            print("✗ MST is not connected")
            return False
        
        # Test drawing the MST
        print("Testing MST visualization...")
        mst_image = grapher.draw_graph(mst_graph)
        print(f"✓ MST visualization created: {mst_image.shape}")
        
        # Test MST with memory
        print("Testing MST with memory...")
        grapher.init_memory_manager(max_memory_size=10)
        
        # Simulate a few iterations
        for i in range(5):
            # Slightly modify positions
            particle_stack[:, 1:3] += np.random.normal(0, 5, (len(particle_stack), 2))
            
            # Create new MST and update memory
            current_mst = grapher.make_mst(particle_stack)
            grapher.update_memory_with_graph(current_mst)
        
        # Create memory-enhanced graph
        memory_graph = grapher.make_memory_graph(particle_stack)
        print(f"✓ Memory-enhanced MST created: {memory_graph.vcount()} vertices, {memory_graph.ecount()} edges")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing MST: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_graph_types():
    """Test all graph types including MST"""
    print("\nTesting all graph types...")
    
    try:
        from graphizy import Graphing, GraphizyConfig, generate_positions
        
        # Create test data
        positions = generate_positions(400, 400, 15)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))
        
        # Create grapher
        config = GraphizyConfig()
        config.graph.dimension = (400, 400)
        grapher = Graphing(config=config)
        
        # Test all graph types
        graph_types = {
            'Proximity': lambda: grapher.make_proximity(particle_stack, proximity_thresh=100),
            'Delaunay': lambda: grapher.make_delaunay(particle_stack),
            'MST': lambda: grapher.make_mst(particle_stack),
        }
        
        results = {}
        for name, create_func in graph_types.items():
            try:
                graph = create_func()
                results[name] = {
                    'vertices': graph.vcount(),
                    'edges': graph.ecount(),
                    'connected': grapher.call_method(graph, 'is_connected')
                }
                print(f"✓ {name}: {results[name]['vertices']}v, {results[name]['edges']}e, connected={results[name]['connected']}")
            except Exception as e:
                print(f"✗ {name} failed: {e}")
                results[name] = None
        
        # Print comparison
        print("\nGraph Type Comparison:")
        print(f"{'Type':<12} {'Vertices':<10} {'Edges':<8} {'Connected':<10}")
        print("-" * 45)
        for name, stats in results.items():
            if stats:
                print(f"{name:<12} {stats['vertices']:<10} {stats['edges']:<8} {stats['connected']:<10}")
        
        return all(results.values())
        
    except Exception as e:
        print(f"✗ Error testing graph types: {e}")
        return False

if __name__ == "__main__":
    print("Testing graphizy MST functionality...")
    print("=" * 50)
    
    success = True
    success &= test_mst_functionality()
    success &= test_all_graph_types()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All MST tests passed! The MST functionality works correctly.")
        print("\nYou can now use:")
        print("  - grapher.make_mst(data) to create MST graphs")
        print("  - Graph type 4 in the interactive Brownian motion viewer")
        print("  - MST with memory modifier: python improved_brownian.py 4 --memory")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)
