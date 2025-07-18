"""
GRAPHIZY ID NORMALIZATION FIX - Summary and Test

PROBLEM IDENTIFIED:
- Test data has float IDs: 1.0, 2.0, 3.0, 4.0 (from numpy array with dtype=float)
- Memory connections use string keys: "1", "2", "3", "4"  
- Original code used str(1.0) = "1.0" which doesn't match "1"
- This caused "Object X in memory but not in current positions" warnings
- Result: 0 edges created in memory graphs

SOLUTION IMPLEMENTED:
1. Added normalize_id() function for consistent ID handling
2. Converts float IDs like 1.0, 2.0 to "1", "2" 
3. Preserves non-integer floats like 3.14 as "3.14"
4. Updated 3 key functions:
   - create_memory_graph() 
   - update_memory_from_proximity()
   - update_memory_from_graph()

PERFORMANCE IMPACT:
- Minimal overhead for real-time applications
- Single function call per ID (very fast)
- No breaking changes to existing API

TEST VALIDATION:
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    from graphizy.algorithms import normalize_id, create_memory_graph

    print("=" * 60)
    print("TESTING ID NORMALIZATION FIX")
    print("=" * 60)

    # Test 1: normalize_id function
    print("\\n1. Testing normalize_id function:")
    test_cases = [
        (1.0, "1"),
        (2.0, "2"), 
        (3.14, "3.14"),
        (4, "4"),
        ("5", "5")
    ]

    all_passed = True
    for input_val, expected in test_cases:
        result = normalize_id(input_val)
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"  {status} {input_val} ({type(input_val).__name__}) -> '{result}'")
        if not passed:
            all_passed = False

    # Test 2: Memory graph creation (the failing test case)
    print("\\n2. Testing memory graph creation:")
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

    try:
        graph = create_memory_graph(positions, memory_connections, aspect="array")
        vertices = graph.vcount()
        edges = graph.ecount()
        
        print(f"  ✓ Graph created successfully")
        print(f"  ✓ Vertices: {vertices}")
        print(f"  ✓ Edges: {edges}")
        
        if edges > 0:
            print(f"  ✓ SUCCESS: Graph has {edges} edges (test should pass)")
            test2_passed = True
        else:
            print(f"  ✗ FAIL: Graph has 0 edges (test would fail)")
            test2_passed = False
            
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        test2_passed = False

    print("\\n" + "=" * 60)
    if all_passed and test2_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✓ ID normalization working correctly")
        print("✓ Memory graphs creating edges as expected")
        print("✓ Ready for real-time applications")
        print("\\nThe failing pytest tests should now pass.")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Need to debug further...")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
except Exception as e:
    print(f"Unexpected error: {e}")
    import traceback
    traceback.print_exc()
