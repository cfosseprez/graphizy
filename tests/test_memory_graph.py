"""
Unit tests for memory graph functionality

Tests cover:
- MemoryManager class
- Memory graph creation
- Memory updates from different sources
- Integration with main Graphing class
"""

import unittest
import numpy as np
from collections import defaultdict, deque

# Import the memory graph components
from graphizy.algorithms import (
    MemoryManager, 
    create_memory_graph,
    update_memory_from_proximity,
    create_graph_array
)
from graphizy.main import Graphing
from graphizy.exceptions import GraphCreationError


class TestMemoryManager(unittest.TestCase):
    """Test basic MemoryManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.memory_mgr = MemoryManager(max_memory_size=5, max_iterations=3)
    
    def test_initialization(self):
        """Test MemoryManager initialization"""
        self.assertEqual(self.memory_mgr.max_memory_size, 5)
        self.assertEqual(self.memory_mgr.max_iterations, 3)
        self.assertEqual(self.memory_mgr.current_iteration, 0)
        self.assertEqual(len(self.memory_mgr.all_objects), 0)
    
    def test_add_connections_basic(self):
        """Test basic connection addition"""
        connections = {
            "A": ["B", "C"],
            "B": ["A"],
            "C": ["A"],
            "D": []
        }
        
        self.memory_mgr.add_connections(connections)
        
        self.assertEqual(self.memory_mgr.current_iteration, 1)
        self.assertEqual(len(self.memory_mgr.all_objects), 4)
        self.assertIn("A", self.memory_mgr.all_objects)
        self.assertIn("D", self.memory_mgr.all_objects)
    
    def test_bidirectional_connections(self):
        """Test that connections are stored bidirectionally"""
        connections = {"A": ["B"], "B": []}
        self.memory_mgr.add_connections(connections)
        
        memory_graph = self.memory_mgr.get_current_memory_graph()
        
        # Both A and B should have each other in their connections
        self.assertIn("B", memory_graph["A"])
        self.assertIn("A", memory_graph["B"])
    
    def test_memory_size_limit(self):
        """Test memory size limiting"""
        # Add more connections than the limit
        for i in range(10):
            connections = {"A": [f"obj_{i}"]}
            self.memory_mgr.add_connections(connections)
        
        # Should not exceed max_memory_size
        self.assertLessEqual(len(self.memory_mgr.memory["A"]), 5)
    
    def test_get_memory_stats(self):
        """Test memory statistics"""
        connections = {
            "A": ["B", "C"],
            "B": ["A"], 
            "C": ["A"],
            "D": []
        }
        self.memory_mgr.add_connections(connections)
        
        stats = self.memory_mgr.get_memory_stats()
        
        self.assertEqual(stats["total_objects"], 4)
        self.assertEqual(stats["current_iteration"], 1)
        self.assertEqual(stats["max_memory_size"], 5)
        self.assertEqual(stats["max_iterations"], 3)
        self.assertGreater(stats["objects_with_memory"], 0)


class TestMemoryGraphCreation(unittest.TestCase):
    """Test memory graph creation functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.positions = np.array([
            [1, 100.0, 100.0],
            [2, 200.0, 150.0],
            [3, 300.0, 200.0],
            [4, 400.0, 250.0]
        ], dtype=float)
        
        self.memory_connections = {
            "1": ["2", "3"],
            "2": ["1"],
            "3": ["1", "4"], 
            "4": ["3"]
        }
    
    def test_create_memory_graph_array(self):
        """Test memory graph creation with array data"""
        graph = create_memory_graph(self.positions, self.memory_connections, aspect="array")
        
        # Check basic graph properties
        print(f"Graph vertexes={graph.vcount()}")
        self.assertEqual(graph.vcount(), 4)
        print(f"Graph edges={graph.ecount()}")
        self.assertGreater(graph.ecount(), 0)
        
        # Check that all vertices have correct IDs
        vertex_ids = set(graph.vs["id"])
        expected_ids = {1.0, 2.0, 3.0, 4.0}
        self.assertEqual(vertex_ids, expected_ids)
    
    def test_create_memory_graph_dict(self):
        """Test memory graph creation with dictionary data"""
        positions_dict = {
            "id": [1, 2, 3, 4],
            "x": [100, 200, 300, 400],
            "y": [100, 150, 200, 250]
        }
        
        graph = create_memory_graph(positions_dict, self.memory_connections, aspect="dict")

        print(f"Graph vertexes={graph.vcount()}")
        self.assertEqual(graph.vcount(), 4)
        print(f"Graph edges={graph.ecount()}")
        self.assertGreater(graph.ecount(), 0)
    
    def test_empty_memory_connections(self):
        """Test handling of empty memory connections"""
        empty_memory = {"1": [], "2": [], "3": [], "4": []}
        
        graph = create_memory_graph(self.positions, empty_memory, aspect="array")
        
        # Should have vertices but no edges
        self.assertEqual(graph.vcount(), 4)
        self.assertEqual(graph.ecount(), 0)


class TestMemoryUpdates(unittest.TestCase):
    """Test memory update functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.positions = np.array([
            [1, 100.0, 100.0],
            [2, 150.0, 100.0],  # Close to object 1
            [3, 300.0, 300.0],  # Far from others
            [4, 320.0, 320.0]   # Close to object 3
        ], dtype=float)
        
        self.memory_mgr = MemoryManager(max_memory_size=10)
    
    def test_update_memory_from_proximity(self):
        """Test proximity-based memory updates"""
        connections = update_memory_from_proximity(
            self.positions, 
            proximity_thresh=100.0,
            memory_manager=self.memory_mgr,
            aspect="array"
        )
        
        # Objects 1-2 should be connected (distance ~50)
        # Objects 3-4 should be connected (distance ~28)
        self.assertGreater(len(connections), 0)
        
        # Check that close objects are connected
        obj1_connections = connections.get("1", [])
        self.assertIn("2", obj1_connections)


class TestGraphingIntegration(unittest.TestCase):
    """Test integration with main Graphing class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.grapher = Graphing(dimension=(500, 400))
        self.positions = np.array([
            [1, 100.0, 100.0],
            [2, 200.0, 150.0],
            [3, 300.0, 200.0],
            [4, 400.0, 250.0]
        ], dtype=float)
    
    def test_memory_manager_initialization(self):
        """Test memory manager initialization in Graphing class"""
        memory_mgr = self.grapher.init_memory_manager(max_memory_size=20, max_iterations=10)
        
        self.assertIsNotNone(self.grapher.memory_manager)
        self.assertEqual(memory_mgr.max_memory_size, 20)
        self.assertEqual(memory_mgr.max_iterations, 10)
        self.assertIs(memory_mgr, self.grapher.memory_manager)
    
    def test_make_memory_graph_without_manager(self):
        """Test memory graph creation without manager should fail"""
        with self.assertRaises(GraphCreationError):
            self.grapher.make_memory_graph(self.positions)
    
    def test_make_memory_graph_with_explicit_connections(self):
        """Test memory graph creation with explicit connections"""
        memory_connections = {
            "1": ["2"],
            "2": ["1", "3"],
            "3": ["2"],
            "4": []
        }
        
        graph = self.grapher.make_memory_graph(self.positions, memory_connections)
        
        self.assertEqual(graph.vcount(), 4)
        self.assertGreater(graph.ecount(), 0)
    
    def test_update_memory_with_proximity(self):
        """Test proximity-based memory updates through Graphing class"""
        self.grapher.init_memory_manager()
        
        connections = self.grapher.update_memory_with_proximity(self.positions, proximity_thresh=150)
        
        self.assertIsInstance(connections, dict)
        self.assertEqual(len(connections), 4)
    
    def test_get_memory_stats_without_manager(self):
        """Test memory stats when no manager is initialized"""
        stats = self.grapher.get_memory_stats()
        
        self.assertIn("error", stats)
    
    def test_get_memory_stats_with_manager(self):
        """Test memory stats with initialized manager"""
        self.grapher.init_memory_manager()
        self.grapher.update_memory_with_proximity(self.positions)
        
        stats = self.grapher.get_memory_stats()
        
        self.assertIn("total_objects", stats)
        self.assertIn("total_connections", stats)
        self.assertIn("current_iteration", stats)
    
    def test_memory_graph_workflow(self):
        """Test complete memory graph workflow"""
        # Initialize memory manager
        self.grapher.init_memory_manager(max_memory_size=50)
        
        # Update memory multiple times
        for i in range(3):
            # Small movements
            if i > 0:
                movement = np.random.normal(0, 10, (len(self.positions), 2))
                self.positions[:, 1:3] += movement
            
            # Update memory
            connections = self.grapher.update_memory_with_proximity(
                self.positions, proximity_thresh=120
            )
            
            self.assertIsInstance(connections, dict)
        
        # Create final memory graph
        memory_graph = self.grapher.make_memory_graph(self.positions)
        
        self.assertEqual(memory_graph.vcount(), 4)
        
        # Get statistics
        stats = self.grapher.get_memory_stats()
        self.assertEqual(stats["current_iteration"], 3)


class TestMemoryVisualization(unittest.TestCase):
    """Test memory visualization features"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.grapher = Graphing(dimension=(300, 300))
        self.positions = np.array([
            [1, 50.0, 50.0],
            [2, 150.0, 50.0],
            [3, 250.0, 50.0]
        ], dtype=float)
    
    def test_standard_memory_graph_drawing(self):
        """Test standard memory graph drawing"""
        memory_connections = {
            "1": ["2"],
            "2": ["1", "3"],
            "3": ["2"]
        }
        
        graph = self.grapher.make_memory_graph(self.positions, memory_connections)
        image = self.grapher.draw_graph(graph)
        
        # Check image properties
        self.assertEqual(image.shape, (300, 300, 3))
        self.assertEqual(image.dtype, np.uint8)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in memory graph functionality"""
    
    def test_invalid_positions_data(self):
        """Test handling of invalid position data"""
        memory_mgr = MemoryManager()
        
        # Test with None positions
        with self.assertRaises(GraphCreationError):
            update_memory_from_proximity(None, 100, memory_mgr)
    
    def test_invalid_memory_connections(self):
        """Test handling of invalid memory connections"""
        positions = np.array([[1, 100, 100]], dtype=float)
        
        # Test with None memory connections
        with self.assertRaises(GraphCreationError):
            create_memory_graph(positions, None)
        
        # Test with empty memory connections should work
        graph = create_memory_graph(positions, {})
        self.assertEqual(graph.vcount(), 1)
        self.assertEqual(graph.ecount(), 0)
    
    def test_memory_manager_none_handling(self):
        """Test handling when memory manager is None"""
        positions = np.array([[1, 100, 100]], dtype=float)
        
        with self.assertRaises(GraphCreationError):
            update_memory_from_proximity(positions, 100, None)


class TestPerformance(unittest.TestCase):
    """Test performance aspects of memory graph functionality"""
    
    def test_large_memory_performance(self):
        """Test performance with large memory datasets"""
        # Create medium dataset for testing
        n_objects = 50
        positions = np.random.rand(n_objects, 3) * 500
        positions[:, 0] = np.arange(n_objects)  # Set IDs
        positions = positions.astype(float)
        
        memory_mgr = MemoryManager(max_memory_size=100, max_iterations=10)
        
        # Time the memory update
        import time
        start_time = time.time()
        
        connections = update_memory_from_proximity(
            positions, proximity_thresh=100, memory_manager=memory_mgr
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete in reasonable time (less than 2 seconds)
        self.assertLess(elapsed, 2.0)
        self.assertEqual(len(connections), n_objects)
    
    def test_memory_cleanup_performance(self):
        """Test performance of memory cleanup operations"""
        memory_mgr = MemoryManager(max_memory_size=50, max_iterations=10)
        
        # Add many iterations to trigger cleanup
        for i in range(20):
            connections = {f"obj_{i}": [f"target_{j}" for j in range(5)]}
            memory_mgr.add_connections(connections)
        
        # Should maintain reasonable memory size
        total_memory_items = sum(len(deque_obj) for deque_obj in memory_mgr.memory.values())
        
        # Total items should be limited by cleanup
        expected_max = len(memory_mgr.all_objects) * 50  # max_memory_size
        self.assertLessEqual(total_memory_items, expected_max)


def create_test_suite():
    """Create comprehensive test suite for memory graph functionality"""
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMemoryManager,
        TestMemoryGraphCreation,
        TestMemoryUpdates,
        TestGraphingIntegration,
        TestMemoryVisualization,
        TestErrorHandling,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    return suite


if __name__ == "__main__":
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    suite = create_test_suite()
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
