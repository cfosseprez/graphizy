#!/usr/bin/env python3
"""
Diagnostic script to identify and fix graphizy issues

This script will:
1. Test basic imports
2. Check coordinate system handling  
3. Test memory functionality
4. Provide detailed error reporting
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that all imports work"""
    print("🔍 Testing imports...")
    
    try:
        from graphizy import (
            Graphing, GraphizyConfig, MemoryConfig, MemoryManager,
            generate_positions, create_memory_graph
        )
        print("✅ All imports successful")
        # ✅ Proper pytest style - no return value
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False, f"Import failed: {e}"  # ✅ Use assert instead of return False

def test_coordinate_system():
    """Test coordinate system with simple case"""
    print("\n🔍 Testing coordinate system...")
    
    try:
        from graphizy import Graphing, generate_positions
        
        # Simple test case
        WIDTH, HEIGHT = 100, 100
        NUM_POINTS = 10
        
        print(f"   Generating {NUM_POINTS} points in {WIDTH}x{HEIGHT} canvas...")
        positions = generate_positions(WIDTH, HEIGHT, NUM_POINTS)
        
        # Check bounds
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        print(f"   Generated positions:")
        print(f"     X range: {x_min:.1f} to {x_max:.1f} (expected: 0 to {WIDTH-1})")
        print(f"     Y range: {y_min:.1f} to {y_max:.1f} (expected: 0 to {HEIGHT-1})")
        
        # Check if within bounds
        x_valid = (0 <= x_min) and (x_max < WIDTH)
        y_valid = (0 <= y_min) and (y_max < HEIGHT)
        
        assert x_valid and y_valid, f"Coordinates out of bounds: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}]"
        
        # Test triangulation
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))
        
        grapher = Graphing(dimension=(WIDTH, HEIGHT))
        graph = grapher.make_delaunay(particle_stack)
        
        print(f"   ✅ Triangulation successful: {graph.vcount()} vertices, {graph.ecount()} edges")
        
    except Exception as e:
        print(f"   ❌ Coordinate system test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        assert False, f"Coordinate system test failed: {e}"

def test_memory_functionality():
    """Test memory functionality"""
    print("\n🔍 Testing memory functionality...")
    
    try:
        from graphizy import MemoryManager, Graphing
        
        # Test MemoryManager
        memory_mgr = MemoryManager(max_memory_size=10)
        
        # Add some connections
        connections = {"A": ["B", "C"], "B": ["A"], "C": ["A"]}
        memory_mgr.add_connections(connections)
        
        # Get memory graph
        memory_graph = memory_mgr.get_current_memory_graph()
        stats = memory_mgr.get_memory_stats()
        
        print(f"   Memory connections: {memory_graph}")
        print(f"   Memory stats: {stats}")
        
        # Test with Graphing class
        grapher = Graphing()
        mgr = grapher.init_memory_manager(max_memory_size=5)
        
        assert mgr is not None, "Memory manager initialization returned None"
        
        print("   ✅ Memory functionality working")
        
    except Exception as e:
        print(f"   ❌ Memory functionality test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        assert False, f"Memory functionality test failed: {e}"

def test_specific_error_scenario():
    """Test the specific scenario that's causing issues"""
    print("\n🔍 Testing specific error scenario...")
    
    try:
        from graphizy import Graphing, GraphizyConfig, generate_positions
        
        # Recreate the problematic scenario from your code
        config = GraphizyConfig()
        config.drawing.line_color = (0, 0, 255)
        config.drawing.line_thickness = 2
        config.drawing.point_color = (255, 255, 0)
        config.drawing.point_radius = 12
        config.drawing.point_thickness = 3
        
        # FIXED: Proper coordinate alignment
        WIDTH, HEIGHT = 400, 300  # Smaller for testing
        config.graph.dimension = (WIDTH, HEIGHT)
        config.graph.proximity_threshold = 80.0
        
        print(f"   Config dimension: {config.graph.dimension}")
        
        # Generate positions correctly aligned
        positions = generate_positions(WIDTH, HEIGHT, 20)  # Fewer points for testing
        print(f"   Position ranges: X[{positions[:, 0].min():.1f}, {positions[:, 0].max():.1f}], Y[{positions[:, 1].min():.1f}, {positions[:, 1].max():.1f}]")
        
        particle_ids = np.arange(len(positions))
        particle_stack = np.column_stack((particle_ids, positions))
        
        # Create grapher
        grapher = Graphing(config=config)
        
        # Try to create Delaunay
        print("   Creating Delaunay triangulation...")
        delaunay_graph = grapher.make_delaunay(particle_stack)
        
        print(f"   ✅ Delaunay successful: {delaunay_graph.vcount()} vertices, {delaunay_graph.ecount()} edges")
        
        # Try to create proximity graph
        print("   Creating proximity graph...")
        proximity_graph = grapher.make_proximity(particle_stack)
        
        print(f"   ✅ Proximity successful: {proximity_graph.vcount()} vertices, {proximity_graph.ecount()} edges")
        
    except Exception as e:
        print(f"   ❌ Specific scenario failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print("   📋 Full traceback:")
        traceback.print_exc()
        assert False, f"Specific scenario failed: {e}"

def diagnose_opencv_subdivision():
    """Diagnose OpenCV subdivision issues specifically"""
    print("\n🔍 Diagnosing OpenCV subdivision...")
    
    try:
        import cv2
        print(f"   OpenCV version: {cv2.__version__}")
        
        # Test basic subdivision
        rect = (0, 0, 100, 100)
        subdiv = cv2.Subdiv2D(rect)
        
        # Test point insertion
        test_points = [(10.0, 10.0), (50.0, 50.0), (90.0, 90.0)]
        
        for i, point in enumerate(test_points):
            try:
                subdiv.insert(point)
                print(f"   ✅ Point {i} inserted: {point}")
            except cv2.error as e:
                print(f"   ❌ Point {i} failed: {point}, error: {e}")
                return False
        
        # Test triangulation
        triangles = subdiv.getTriangleList()
        print(f"   ✅ Triangulation successful: {len(triangles)} triangles")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenCV subdivision test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🏥 Graphizy Diagnostic Tool")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Coordinate System", test_coordinate_system),
        ("Memory Functionality", test_memory_functionality),
        ("OpenCV Subdivision", diagnose_opencv_subdivision),
        ("Specific Error Scenario", test_specific_error_scenario),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: CRASHED - {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Graphizy should be working correctly.")
    else:
        print("🔧 Some tests failed. Check the error messages above for details.")
        print("\n💡 Common fixes:")
        print("   1. Ensure OpenCV is installed: pip install opencv-python")
        print("   2. Ensure coordinates are properly aligned (generate_positions params)")
        print("   3. Check that canvas dimensions match position generation")
        
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
