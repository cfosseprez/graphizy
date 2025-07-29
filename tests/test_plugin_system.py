"""
Test for the plugin system functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from graphizy.plugins_logic import graph_type_plugin, GraphTypePlugin, GraphTypeInfo, register_graph_type, get_graph_registry


def test_plugin_system():
    """Test the plugin system functionality"""
    print("Testing Graphizy Plugin System...")
    
    # Test 1: Decorator-based plugin
    @graph_type_plugin(
        name="test_random",
        description="Test random graph for plugin system",
        parameters={
            "probability": {"type": float, "default": 0.2, "description": "Connection probability"}
        },
        category="test"
    )
    def create_test_random(data_points, aspect, dimension, probability=0.2):
        from graphizy.algorithms import create_graph_array
        import random
        
        graph = create_graph_array(data_points)
        num_points = len(data_points)
        
        for i in range(num_points):
            for j in range(i + 1, num_points):
                if random.random() < probability:
                    graph.add_edge(i, j)
        
        return graph
    
    # Test 2: Class-based plugin
    class TestStarPlugin(GraphTypePlugin):
        @property
        def info(self):
            return GraphTypeInfo(
                name="test_star",
                description="Test star graph for plugin system",
                parameters={},
                category="test",
                author="Test Suite",
                version="1.0.0"
            )
        
        def create_graph(self, data_points, aspect, dimension, **kwargs):
            from graphizy.algorithms import create_graph_array
            
            graph = create_graph_array(data_points)
            num_points = len(data_points)
            
            # Connect first point to all others (star topology)
            for i in range(1, num_points):
                graph.add_edge(0, i)
            
            return graph
    
    # Register the class-based plugin
    register_graph_type(TestStarPlugin())
    
    # Test 3: Registry functionality
    registry = get_graph_registry()
    
    print("✅ Plugins registered successfully")
    
    # List all plugins
    all_plugins = registry.list_plugins()
    test_plugins = registry.list_plugins(category="test")
    
    print(f"✅ Total plugins: {len(all_plugins)}")
    print(f"✅ Test plugins: {len(test_plugins)}")
    
    assert "test_random" in test_plugins
    assert "test_star" in test_plugins
    
    # Test 4: Plugin usage
    from graphizy import Graphing, generate_positions
    
    # Generate test data
    positions = generate_positions(200, 200, 10)
    data = np.column_stack((np.arange(len(positions)), positions))
    
    # Create grapher
    grapher = Graphing(dimension=(200, 200))
    
    # Test using the new plugin system
    try:
        # Use decorator-based plugin
        random_graph = grapher.make_graph('test_random', data, probability=0.3)
        print(f"✅ Random graph: {random_graph.vcount()} vertices, {random_graph.ecount()} edges")
        
        # Use class-based plugin
        star_graph = grapher.make_graph('test_star', data)
        print(f"✅ Star graph: {star_graph.vcount()} vertices, {star_graph.ecount()} edges")
        
        # Verify star topology (should have 9 edges for 10 vertices)
        assert star_graph.ecount() == 9, f"Star graph should have 9 edges, got {star_graph.ecount()}"
        
        # Test built-in types still work
        delaunay_graph = grapher.make_graph('delaunay', data)
        print(f"✅ Delaunay graph: {delaunay_graph.vcount()} vertices, {delaunay_graph.ecount()} edges")
        
    except Exception as e:
        print(f"❌ Error testing graph creation: {e}")
        return False
    
    # Test 5: Information retrieval
    try:
        info = grapher.get_graph_info('test_random')
        assert info['info']['description'] == "Test random graph for plugin system"
        assert 'probability' in info['parameters']
        print("✅ Plugin information retrieval works")
        
    except Exception as e:
        print(f"❌ Error getting plugin info: {e}")
        return False
    
    # Test 6: List graph types
    try:
        all_types = grapher.list_graph_types()
        test_types = grapher.list_graph_types(category="test")
        
        assert 'test_random' in all_types
        assert 'test_star' in all_types
        assert len(test_types) == 2
        print("✅ Graph type listing works")
        
    except Exception as e:
        print(f"❌ Error listing graph types: {e}")
        return False
    
    print("\n🎉 All plugin system tests passed!")
    print("\nPlugin System Benefits Demonstrated:")
    print("✅ Easy registration with decorators and classes")
    print("✅ Automatic parameter handling")
    print("✅ Discovery through list_graph_types()")
    print("✅ Same API as built-in graph types")
    print("✅ Built-in documentation system")
    print("✅ Zero core file modifications needed")
    
    return True


if __name__ == "__main__":
    success = test_plugin_system()
    if success:
        print("\n🚀 Plugin system is working perfectly!")
    else:
        print("\n💥 Plugin system test failed!")
        sys.exit(1)
