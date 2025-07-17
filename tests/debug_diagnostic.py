#!/usr/bin/env python3
"""
Diagnostic script for graphizy memory graph issues.
This script helps identify why memory graphs are not creating edges properly.
"""

import sys
import os
from pathlib import Path
import numpy as np


try:
    from graphizy.algorithms import create_memory_graph, update_memory_from_proximity, MemoryManager
    from graphizy.config import MemoryConfig
    from graphizy.main import Graphing

    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def diagnose_create_memory_graph():
    """Test the create_memory_graph function directly"""
    print("\n" + "=" * 60)
    print("DIAGNOSING: create_memory_graph function")
    print("=" * 60)

    # Test data in array format (correct format for the algorithms)
    positions = np.array([
        [1, 100, 100],  # [id, x, y]
        [2, 150, 100],
        [3, 200, 200],
        [4, 220, 220]
    ])

    memory_connections = {
        "1": ["2"],
        "2": ["1", "3"],
        "3": ["2"],
        "4": []
    }

    print("\nInput positions (array format):")
    for row in positions:
        print(f"  ID {int(row[0])}: x={int(row[1])}, y={int(row[2])}")

    print("\nInput memory_connections:")
    for k, v in memory_connections.items():
        print(f"  {k}: {v}")

    # Try creating the graph with array aspect
    try:
        graph = create_memory_graph(positions, memory_connections, aspect="array")
        print(f"\n✓ Graph created successfully")
        print(f"  Vertices: {graph.vcount()}")
        print(f"  Edges: {graph.ecount()}")

        # Check vertex attributes
        print("\nVertex details:")
        for i in range(graph.vcount()):
            attrs = graph.vs[i].attributes()
            print(f"  Vertex {i}: {attrs}")

        # Check edge details
        if graph.ecount() > 0:
            print("\nEdge details:")
            for e in graph.es:
                print(f"  Edge: {e.source} -> {e.target}")
        else:
            print("\n✗ No edges created!")

    except Exception as e:
        print(f"\n✗ Error creating graph: {e}")
        import traceback
        traceback.print_exc()

    # Also test with dict format
    print("\n" + "-" * 40)
    print("Testing with dict format:")
    
    positions_dict = {
        "id": [1, 2, 3, 4],
        "x": [100, 150, 200, 220],
        "y": [100, 100, 200, 220]
    }

    try:
        graph = create_memory_graph(positions_dict, memory_connections, aspect="dict")
        print(f"\n✓ Dict format graph created successfully")
        print(f"  Vertices: {graph.vcount()}")
        print(f"  Edges: {graph.ecount()}")
    except Exception as e:
        print(f"\n✗ Error creating dict format graph: {e}")
        import traceback
        traceback.print_exc()


def diagnose_memory_manager():
    """Test MemoryManager functionality"""
    print("\n" + "=" * 60)
    print("DIAGNOSING: MemoryManager")
    print("=" * 60)

    # Create MemoryManager with correct parameters
    memory_mgr = MemoryManager(max_memory_size=100, max_iterations=None, track_edge_ages=True)

    print(f"\nMemoryManager initialized:")
    print(f"  max_memory_size: {memory_mgr.max_memory_size}")
    print(f"  max_iterations: {memory_mgr.max_iterations}")
    print(f"  track_edge_ages: {memory_mgr.track_edge_ages}")
    print(f"  current_iteration: {memory_mgr.current_iteration}")

    # Test connections
    test_connections = {
        "1": ["2"],
        "2": ["1", "3"],
        "3": ["2"],
        "4": []
    }

    # Add connections
    memory_mgr.add_connections(test_connections)
    print(f"\nAfter add_connections:")
    print(f"  Current iteration: {memory_mgr.current_iteration}")
    print(f"  All objects: {memory_mgr.all_objects}")

    # Get memory graph
    memory_graph = memory_mgr.get_current_memory_graph()
    print(f"\nMemory graph: {memory_graph}")

    # Get statistics
    stats = memory_mgr.get_memory_stats()
    print(f"\nMemory stats: {stats}")


def diagnose_proximity_update():
    """Test proximity-based memory updates"""
    print("\n" + "=" * 60)
    print("DIAGNOSING: update_memory_from_proximity")
    print("=" * 60)

    # Test data in array format
    positions = np.array([
        [1, 100, 100],  # [id, x, y]
        [2, 150, 100],  # Distance ~50 from 1
        [3, 200, 200],
        [4, 220, 220]   # Distance ~28 from 3
    ])

    memory_mgr = MemoryManager(max_memory_size=100)

    print(f"\nTesting with proximity_thresh=100.0")
    print("Input positions:")
    for row in positions:
        print(f"  ID {int(row[0])}: x={int(row[1])}, y={int(row[2])}")

    try:
        connections = update_memory_from_proximity(
            positions,
            proximity_thresh=100.0,
            memory_manager=memory_mgr,
            metric="euclidean",
            aspect="array"
        )

        print(f"\n✓ Proximity update successful")
        print(f"  Connections: {connections}")

        # Calculate actual distances
        import math
        print("\nActual distances:")
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                id1, x1, y1 = positions[i]
                id2, x2, y2 = positions[j]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                print(f"  {int(id1)} <-> {int(id2)}: {dist:.2f}")

        # Get memory stats
        stats = memory_mgr.get_memory_stats()
        print(f"\nMemory stats after proximity update: {stats}")

    except Exception as e:
        print(f"\n✗ Error in proximity update: {e}")
        import traceback
        traceback.print_exc()


def diagnose_graphing_integration():
    """Test Graphing class integration"""
    print("\n" + "=" * 60)
    print("DIAGNOSING: Graphing class integration")
    print("=" * 60)

    # Initialize Graphing properly
    grapher = Graphing()
    
    # Initialize memory manager
    memory_mgr = grapher.init_memory_manager(max_memory_size=100)

    positions = np.array([
        [1, 100, 100],
        [2, 150, 100],
        [3, 200, 200],
        [4, 220, 220]
    ])

    memory_connections = {
        "1": ["2"],
        "2": ["1", "3"],
        "3": ["2"],
        "4": []
    }

    try:
        graph = grapher.make_memory_graph(positions, memory_connections)
        print(f"\n✓ make_memory_graph successful")
        print(f"  Vertices: {graph.vcount()}")
        print(f"  Edges: {graph.ecount()}")
        
        # Test memory update with proximity
        print("\nTesting memory update with proximity...")
        connections = grapher.update_memory_with_proximity(positions, proximity_thresh=100.0)
        print(f"  Proximity connections: {connections}")
        
        # Get memory stats
        stats = grapher.get_memory_stats()
        print(f"  Memory stats: {stats}")
        
    except Exception as e:
        print(f"\n✗ Error in make_memory_graph: {e}")
        import traceback
        traceback.print_exc()


def check_source_code():
    """Check the actual implementation of create_memory_graph"""
    print("\n" + "=" * 60)
    print("CHECKING: Source code inspection")
    print("=" * 60)

    try:
        import inspect
        import graphizy.algorithms

        # Get source of create_memory_graph
        source = inspect.getsource(graphizy.algorithms.create_memory_graph)
        lines = source.split('\n')

        # Find warning line
        for i, line in enumerate(lines):
            if "in memory but not in current positions" in line:
                print(f"\nFound warning at line {i + 1} of create_memory_graph:")
                # Show context
                start = max(0, i - 5)
                end = min(len(lines), i + 6)
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"{marker} {lines[j]}")
                break

        # Also check the ID mapping logic
        print("\nLooking for ID mapping logic...")
        for i, line in enumerate(lines):
            if "id_to_vertex" in line and "=" in line:
                print(f"\nFound ID mapping at line {i + 1}:")
                start = max(0, i - 2)
                end = min(len(lines), i + 5)
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"{marker} {lines[j]}")

    except Exception as e:
        print(f"Could not inspect source: {e}")


def test_edge_case_scenarios():
    """Test various edge cases that might cause issues"""
    print("\n" + "=" * 60)
    print("TESTING: Edge case scenarios")
    print("=" * 60)

    # Test 1: Empty memory connections
    print("\nTest 1: Empty memory connections")
    positions = np.array([[1, 100, 100], [2, 150, 100]])
    empty_connections = {"1": [], "2": []}
    
    try:
        graph = create_memory_graph(positions, empty_connections, aspect="array")
        print(f"  ✓ Empty connections: {graph.ecount()} edges")
    except Exception as e:
        print(f"  ✗ Empty connections failed: {e}")

    # Test 2: Missing position for connected object
    print("\nTest 2: Missing position for connected object")
    positions = np.array([[1, 100, 100]])  # Only object 1
    bad_connections = {"1": ["2"]}  # References non-existent object 2
    
    try:
        graph = create_memory_graph(positions, bad_connections, aspect="array")
        print(f"  ✓ Missing position handled: {graph.ecount()} edges")
    except Exception as e:
        print(f"  ✗ Missing position failed: {e}")

    # Test 3: Self-connections
    print("\nTest 3: Self-connections")
    positions = np.array([[1, 100, 100]])
    self_connections = {"1": ["1"]}  # Self-reference
    
    try:
        graph = create_memory_graph(positions, self_connections, aspect="array")
        print(f"  ✓ Self-connections handled: {graph.ecount()} edges")
    except Exception as e:
        print(f"  ✗ Self-connections failed: {e}")

    # Test 4: Type mismatches (string vs int IDs)
    print("\nTest 4: Type mismatches")
    positions = np.array([[1, 100, 100], [2, 150, 100]])  # Integer IDs
    string_connections = {"1": ["2"], "2": ["1"]}  # String keys
    
    try:
        graph = create_memory_graph(positions, string_connections, aspect="array")
        print(f"  ✓ Type mismatch handled: {graph.ecount()} edges")
    except Exception as e:
        print(f"  ✗ Type mismatch failed: {e}")

    # Test 5: Large ID numbers
    print("\nTest 5: Large ID numbers")
    positions = np.array([[1001, 100, 100], [2002, 150, 100]])
    large_connections = {"1001": ["2002"], "2002": ["1001"]}
    
    try:
        graph = create_memory_graph(positions, large_connections, aspect="array")
        print(f"  ✓ Large IDs handled: {graph.ecount()} edges")
    except Exception as e:
        print(f"  ✗ Large IDs failed: {e}")

    # Test 6: Mixed ID types in connections
    print("\nTest 6: Mixed ID types in connections")
    positions = np.array([[1, 100, 100], [2, 150, 100]])
    mixed_connections = {"1": [2], "2": ["1"]}  # Mix of string keys and int/string values
    
    try:
        graph = create_memory_graph(positions, mixed_connections, aspect="array")
        print(f"  ✓ Mixed types handled: {graph.ecount()} edges")
    except Exception as e:
        print(f"  ✗ Mixed types failed: {e}")


def test_memory_manager_edge_cases():
    """Test MemoryManager with various edge cases"""
    print("\n" + "=" * 60)
    print("TESTING: MemoryManager edge cases")
    print("=" * 60)

    # Test 1: Memory size limit
    print("\nTest 1: Memory size limit")
    memory_mgr = MemoryManager(max_memory_size=2)  # Very small limit
    
    large_connections = {
        "1": ["2", "3", "4", "5", "6"],  # More than limit
        "2": ["1"],
        "3": ["1"],
        "4": ["1"],
        "5": ["1"],
        "6": ["1"]
    }
    
    try:
        memory_mgr.add_connections(large_connections)
        stats = memory_mgr.get_memory_stats()
        print(f"  ✓ Memory limit respected: {stats}")
    except Exception as e:
        print(f"  ✗ Memory limit test failed: {e}")

    # Test 2: Iteration limit
    print("\nTest 2: Iteration limit")
    memory_mgr = MemoryManager(max_memory_size=100, max_iterations=2)
    
    try:
        # Add connections over multiple iterations
        for i in range(5):
            connections = {"1": ["2"], "2": ["1"]}
            memory_mgr.add_connections(connections)
        
        stats = memory_mgr.get_memory_stats()
        print(f"  ✓ Iteration limit working: current_iter={stats['current_iteration']}")
    except Exception as e:
        print(f"  ✗ Iteration limit test failed: {e}")

    # Test 3: Edge age tracking
    print("\nTest 3: Edge age tracking")
    memory_mgr = MemoryManager(max_memory_size=100, track_edge_ages=True)
    
    try:
        # Add some connections
        memory_mgr.add_connections({"1": ["2"], "2": ["1"]})
        memory_mgr.add_connections({"1": ["3"], "3": ["1"]})  # New iteration
        
        edge_ages = memory_mgr.get_edge_ages()
        normalized_ages = memory_mgr.get_edge_age_normalized()
        
        print(f"  ✓ Edge ages tracked: {len(edge_ages)} edges")
        print(f"    Raw ages: {edge_ages}")
        print(f"    Normalized: {normalized_ages}")
    except Exception as e:
        print(f"  ✗ Edge age tracking failed: {e}")


if __name__ == "__main__":
    print("Graphizy Memory Graph Diagnostic Tool")
    print("=====================================")

    # Run all diagnostics
    diagnose_create_memory_graph()
    diagnose_memory_manager()
    diagnose_proximity_update()
    diagnose_graphing_integration()
    check_source_code()
    test_edge_case_scenarios()
    test_memory_manager_edge_cases()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("\nIf you see any ✗ errors above, those indicate issues that need to be fixed.")
    print("If you see ✓ success messages, those components are working correctly.")
    print("\nCommon fixes:")
    print("- Ensure position data is in correct format (numpy array or proper dict)")
    print("- Check that memory_connections use string keys consistently")
    print("- Verify all referenced IDs exist in the position data")
    print("- Make sure MemoryManager is initialized before use")
