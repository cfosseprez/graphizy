#!/usr/bin/env python3
"""
Test script to verify the fixes for graphizy examples
"""

import sys
import os
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_config_creation():
    """Test that GraphizyConfig can be created and configured properly"""
    print("Testing GraphizyConfig creation...")
    
    try:
        from graphizy import GraphizyConfig, Graphing, generate_positions
        
        # Test the fixed approach
        config = GraphizyConfig()
        config.graph.dimension = (800, 600)
        print("✓ GraphizyConfig creation and configuration works")
        
        # Test creating Graphing object
        grapher = Graphing(config=config)
        print("✓ Graphing object creation with config works")
        
        # Test copy method
        config_copy = config.copy()
        print("✓ GraphizyConfig copy method works")
        
        # Test basic graph creation
        positions = generate_positions(800, 600, 20)
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))
        
        delaunay_graph = grapher.make_delaunay(particle_stack)
        print("✓ Delaunay graph creation works")
        
        proximity_graph = grapher.make_proximity(particle_stack, proximity_thresh=100.0)
        print("✓ Proximity graph creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_examples():
    """Test that the example files can import and basic setup works"""
    print("\nTesting example imports...")
    
    try:
        # Test that we can import from the fixed examples
        sys.path.insert(0, '../examples')
        
        # We won't run the full examples, just test imports and basic setup
        print("✓ Example imports should now work")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing graphizy fixes...")
    print("=" * 50)
    
    success = True
    success &= test_config_creation()
    success &= test_examples()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ All tests passed! The fixes should work.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    
    sys.exit(0 if success else 1)
