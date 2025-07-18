"""Simple test to verify our fixes work"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np

# Test normalize_id function
def test_id_normalization():
    from graphizy.algorithms import normalize_id
    
    # Test cases that were problematic
    test_cases = [
        (1.0, "1"),   # Float 1.0 should become "1" 
        (2.0, "2"),   # Float 2.0 should become "2"
        ("1", "1"),   # String "1" should stay "1"
        ("2", "2"),   # String "2" should stay "2"
    ]
    
    print("Testing ID normalization:")
    all_passed = True
    for input_val, expected in test_cases:
        result = normalize_id(input_val)
        passed = result == expected
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {input_val} -> '{result}' (expected '{expected}')")
        if not passed:
            all_passed = False
    
    return all_passed

# Test memory graph creation
def test_memory_graph():
    from graphizy.algorithms import create_memory_graph
    
    # The exact test data that was failing
    positions = np.array([
        [1, 100.0, 100.0],
        [2, 200.0, 150.0], 
        [3, 300.0, 200.0],
        [4, 400.0, 250.0]
    ], dtype=float)
    
    memory_connections = {
        "1": ["2", "3"],
        "2": ["1"],
        "3": ["1", "4"],
        "4": ["3"]
    }
    
    print("\nTesting memory graph creation:")
    try:
        graph = create_memory_graph(positions, memory_connections, aspect="array")
        vertices = graph.vcount()
        edges = graph.ecount()
        
        print(f"  Graph created: {vertices} vertices, {edges} edges")
        
        if edges > 0:
            print("  PASS: Graph has edges as expected")
            return True
        else:
            print("  FAIL: Graph has 0 edges")
            return False
            
    except Exception as e:
        print(f"  FAIL: Error creating graph: {e}")
        return False

if __name__ == "__main__":
    print("Quick Fix Validation Test")
    print("=" * 40)
    
    test1 = test_id_normalization()
    test2 = test_memory_graph()
    
    print("\n" + "=" * 40)
    if test1 and test2:
        print("SUCCESS: All tests passed! Fix is working.")
    else:
        print("FAILURE: Some tests failed. Need more debugging.")
