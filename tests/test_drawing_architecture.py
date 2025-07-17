"""
Quick test to verify the memory graph drawing functions work correctly
"""

from graphizy import Graphing
import numpy as np

def test_memory_graph_drawing():
    """Test that memory graph drawing works with the new architecture"""
    
    print("Testing memory graph drawing architecture...")
    
    # Initialize grapher
    grapher = Graphing(dimension=(400, 300))
    grapher.init_memory_manager(max_memory_size=50, track_edge_ages=True)
    
    # Create test positions
    positions = np.array([
        [1, 50.0, 50.0],
        [2, 150.0, 50.0], 
        [3, 250.0, 50.0],
        [4, 100.0, 150.0],
        [5, 200.0, 150.0]
    ], dtype=float)
    
    # Update memory a few times
    for i in range(3):
        grapher.update_memory_with_proximity(positions, proximity_thresh=120)
        # Small movement
        if i < 2:
            positions[:, 1:3] += np.random.normal(0, 10, (len(positions), 2))
            positions[:, 1] = np.clip(positions[:, 1], 10, 390)
            positions[:, 2] = np.clip(positions[:, 2], 10, 290)
    
    # Create memory graph
    memory_graph = grapher.make_memory_graph(positions)
    
    # Test 1: Standard drawing (should work)
    print("Test 1: Standard graph drawing...")
    standard_image = grapher.draw_graph(memory_graph)
    print(f"✓ Standard drawing: {standard_image.shape}")
    
    # Test 2: Memory graph drawing with age colors
    print("Test 2: Memory graph with age colors...")
    aged_image = grapher.draw_memory_graph(memory_graph, use_age_colors=True)
    print(f"✓ Age-based drawing: {aged_image.shape}")
    
    # Test 3: Memory graph drawing without age colors
    print("Test 3: Memory graph without age colors...")
    normal_memory_image = grapher.draw_memory_graph(memory_graph, use_age_colors=False)
    print(f"✓ Normal memory drawing: {normal_memory_image.shape}")
    
    # Test 4: Direct drawing module usage
    print("Test 4: Direct drawing module usage...")
    from graphizy.drawing import create_memory_graph_image
    direct_image = create_memory_graph_image(
        graph=memory_graph,
        memory_manager=grapher.memory_manager,
        dimension=(400, 300),
        use_age_colors=True
    )
    print(f"✓ Direct drawing module: {direct_image.shape}")
    
    # Get memory stats
    stats = grapher.get_memory_stats()
    print(f"Memory stats: {stats}")
    
    print("All tests passed! ✓")
    return True

if __name__ == "__main__":
    test_memory_graph_drawing()
