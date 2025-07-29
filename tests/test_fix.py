#!/usr/bin/env python3
"""
Quick test to validate the ID normalization fix for real-time applications
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from graphizy.algorithms import normalize_id, create_memory_graph

def test_normalize_id():
    """Test the normalize_id function"""
    test_cases = [
        (1.0, "1"), (2.0, "2"), (3.14, "3.14"),
        (4, "4"), ("5", "5"), (np.float64(6.0), "6"),
        (np.int64(7), "7"),
    ]
    for input_val, expected in test_cases:
        result = normalize_id(input_val)
        assert result == expected


def test_memory_graph_fix():
    """Test the specific failing test case"""
    positions = np.array([
        [1, 100.0, 100.0], [2, 200.0, 150.0],
        [3, 300.0, 200.0], [4, 400.0, 250.0]
    ], dtype=float)

    memory_connections = {
        "1": ["2", "3"], "2": ["1"],
        "3": ["1", "4"], "4": ["3"]
    }

    graph = create_memory_graph(positions, memory_connections, aspect="array")
    assert graph.ecount() > 0, "Graph should have edges but has 0"

def main():
    """Run all tests"""
    print("Testing ID Normalization Fix for Real-Time Applications")
    print("=" * 60)
    
    test1_passed = test_normalize_id()
    test2_passed = test_memory_graph_fix()
    
    print("\\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 ALL TESTS PASSED! The fix should work.")
        print("   - ID normalization working correctly")
        print("   - Memory graphs creating edges as expected")
        print("   - Ready for real-time applications")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        if not test1_passed:
            print("   - ID normalization has issues")
        if not test2_passed:
            print("   - Memory graph creation still failing")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
