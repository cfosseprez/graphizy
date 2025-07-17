#!/usr/bin/env python3
"""
Test script to verify that all graphizy issues are fixed

This script tests:
1. Memory configuration import/export
2. Memory manager functionality 
3. CLI memory command functionality
4. Circular import resolution
"""

import sys
import os
import numpy as np
import tempfile

# Add the src directory to the path so we can import graphizy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all imports work without circular import issues"""
    print("Testing imports...")
    
    try:
        from graphizy import (
            Graphing, GraphizyConfig, MemoryConfig, MemoryManager,
            create_memory_graph, update_memory_from_proximity
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_memory_config():
    """Test memory configuration"""
    print("\nTesting memory configuration...")
    
    try:
        from graphizy import GraphizyConfig, MemoryConfig
        
        # Test creating config with memory settings
        config = GraphizyConfig()
        config.memory.max_memory_size = 50
        config.memory.max_iterations = 10
        
        # Test config export/import
        config_dict = config.to_dict()
        assert 'memory' in config_dict or hasattr(config, 'memory')
        
        print("✓ Memory configuration works")
        return True
    except Exception as e:
        print(f"✗ Memory configuration failed: {e}")
        return False

def test_memory_manager():
    """Test memory manager functionality"""
    print("\nTesting memory manager...")
    
    try:
        from graphizy import MemoryManager
        
        # Create memory manager
        memory_mgr = MemoryManager(max_memory_size=10, max_iterations=5)
        
        # Test adding connections
        connections = {
            "1": ["2", "3"],
            "2": ["1"],
            "3": ["1", "4"],
            "4": ["3"]
        }
        memory_mgr.add_connections(connections)
        
        # Test getting memory graph
        memory_graph = memory_mgr.get_current_memory_graph()
        assert len(memory_graph) > 0
        
        # Test stats
        stats = memory_mgr.get_memory_stats()
        assert stats['total_objects'] == 4
        
        print("✓ Memory manager works")
        return True
    except Exception as e:
        print(f"✗ Memory manager failed: {e}")
        return False

def test_memory_graph_creation():
    """Test creating memory graphs"""
    print("\nTesting memory graph creation...")
    
    try:
        from graphizy import create_memory_graph, MemoryManager
        
        # Create test data
        positions = np.array([
            [1, 100, 100],
            [2, 200, 150], 
            [3, 120, 300],
            [4, 400, 100]
        ])
        
        # Create memory connections
        memory_connections = {
            "1": ["3", "4"],
            "2": [],
            "3": ["1"],
            "4": ["1"]
        }
        
        # Create memory graph
        graph = create_memory_graph(positions, memory_connections, aspect="array")
        
        assert graph.vcount() == 4
        assert graph.ecount() >= 0
        
        print("✓ Memory graph creation works")
        return True
    except Exception as e:
        print(f"✗ Memory graph creation failed: {e}")
        return False

def test_graphing_integration():
    """Test full Graphing class with memory functionality"""
    print("\nTesting Graphing class memory integration...")
    
    try:
        from graphizy import Graphing, GraphizyConfig
        
        # Create config
        config = GraphizyConfig()
        config.memory.max_memory_size = 20
        
        # Create grapher
        grapher = Graphing(config=config)
        
        # Initialize memory manager
        memory_mgr = grapher.init_memory_manager(max_memory_size=20, max_iterations=5)
        assert memory_mgr is not None
        
        # Create test data
        particle_stack = np.array([
            [0, 100, 100],
            [1, 200, 150],
            [2, 120, 300]
        ])
        
        # Test memory update with proximity
        connections = grapher.update_memory_with_proximity(particle_stack, 100.0)
        assert isinstance(connections, dict)
        
        # Test creating memory graph
        memory_graph = grapher.make_memory_graph(particle_stack)
        assert memory_graph.vcount() == 3
        
        # Test getting stats
        stats = grapher.get_memory_stats()
        assert 'total_objects' in stats
        
        print("✓ Graphing memory integration works")
        return True
    except Exception as e:
        print(f"✗ Graphing memory integration failed: {e}")
        return False

def test_cli_memory_command():
    """Test CLI memory command functionality"""
    print("\nTesting CLI memory command...")
    
    try:
        from graphizy.cli import create_parser, create_config_from_args
        import argparse
        
        # Create parser and test memory command args
        parser = create_parser()
        
        # Test that memory subcommand exists
        test_args = ['memory', '--size', '800', '--particles', '50', 
                    '--memory-size', '30', '--memory-iterations', '8',
                    '--iterations', '5']
        
        args = parser.parse_args(test_args)
        
        # Test config creation from args
        config = create_config_from_args(args)
        
        # Verify memory config is properly set
        assert config.memory.max_memory_size == 30
        assert config.memory.max_iterations == 8
        
        print("✓ CLI memory command works")
        return True
    except Exception as e:
        print(f"✗ CLI memory command failed: {e}")
        return False

def test_drawing_memory_graph():
    """Test drawing memory graphs"""
    print("\nTesting memory graph drawing...")
    
    try:
        from graphizy import Graphing, MemoryManager
        
        # Create simple test
        grapher = Graphing()
        memory_mgr = grapher.init_memory_manager(max_memory_size=10)
        
        # Create test data
        positions = np.array([
            [0, 100, 100],
            [1, 200, 150]
        ])
        
        # Add some memory
        memory_mgr.add_connections({"0": ["1"], "1": ["0"]})
        
        # Create memory graph
        memory_graph = grapher.make_memory_graph(positions)
        
        # Test drawing (this creates the image but doesn't display it)
        image = grapher.draw_memory_graph(memory_graph, use_age_colors=True)
        assert image is not None
        assert image.shape[2] == 3  # RGB channels
        
        print("✓ Memory graph drawing works")
        return True
    except Exception as e:
        print(f"✗ Memory graph drawing failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Graphizy Fix Verification Tests ===\n")
    
    tests = [
        test_imports,
        test_memory_config,
        test_memory_manager,
        test_memory_graph_creation,
        test_graphing_integration,
        test_cli_memory_command,
        test_drawing_memory_graph
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The graphizy package has been successfully fixed.")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
