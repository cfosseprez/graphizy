"""
Quick test to verify the memory graph fixes
"""
from graphizy import Graphing, create_memory_graph
import numpy as np

def test_memory_graph_fix():
    """Test that the ID matching fix works"""
    
    print("Testing memory graph ID matching fix...")
    
    # Test positions (numeric IDs)
    positions = np.array([
        [1, 100.0, 100.0],
        [2, 200.0, 150.0],
        [3, 300.0, 200.0],
        [4, 400.0, 250.0]
    ], dtype=float)
    
    # Test memory connections (string IDs like in the tests)
    memory_connections = {
        "1": ["2", "3"],
        "2": ["1"],
        "3": ["1", "4"], 
        "4": ["3"]
    }
    
    print("Creating memory graph...")
    graph = create_memory_graph(positions, memory_connections, aspect="array")
    
    print(f"Graph vertices: {graph.vcount()}")
    print(f"Graph edges: {graph.ecount()}")
    
    assert graph.ecount() > 0, "Memory graph creation failed - no edges created"
    
    print("✓ Memory graph creation successful!")
    print("Edge connections:")
    for edge in graph.es:
        v1_id = graph.vs[edge.tuple[0]]["id"]
        v2_id = graph.vs[edge.tuple[1]]["id"]
        print(f"  {v1_id} <-> {v2_id}")

def test_proximity_memory():
    """Test proximity memory updates"""
    
    print("\nTesting proximity memory updates...")
    
    # Test positions with close objects
    positions = np.array([
        [1, 100.0, 100.0],
        [2, 150.0, 100.0],  # Close to object 1 (distance = 50)
        [3, 300.0, 300.0],  # Far from others
        [4, 320.0, 320.0]   # Close to object 3 (distance ~28)
    ], dtype=float)
    
    grapher = Graphing(dimension=(400, 400))
    grapher.init_memory_manager(max_memory_size=10, track_edge_ages=True)
    
    connections = grapher.update_memory_with_proximity(positions, proximity_thresh=100.0)
    
    print(f"Proximity connections: {connections}")
    
    # Check if object 1 is connected to object 2
    obj1_connections = connections.get("1", [])
    assert "2" in obj1_connections, f"Object 1 should be connected to object 2. Got connections: {obj1_connections}"
    
    print("✓ Proximity memory update successful!")

if __name__ == "__main__":
    try:
        test_memory_graph_fix()
        test_proximity_memory()
        print("\n🎉 All tests passed! Memory graph functionality is working.")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
