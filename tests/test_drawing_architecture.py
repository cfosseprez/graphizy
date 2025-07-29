"""
Quick test to verify the memory graph drawing functions work correctly
"""

from graphizy import Graphing
import numpy as np


def test_memory_graph_drawing():
    """Test that memory graph drawing works with the new architecture"""
    grapher = Graphing(dimension=(400, 300))
    grapher.init_memory_manager(max_memory_size=50, track_edge_ages=True)

    positions = np.array([
        [1, 50.0, 50.0], [2, 150.0, 50.0], [3, 250.0, 50.0],
        [4, 100.0, 150.0], [5, 200.0, 150.0]
    ], dtype=float)

    for i in range(3):
        grapher.update_memory_with_proximity(positions, proximity_thresh=120)
        if i < 2:
            positions[:, 1:3] += np.random.normal(0, 10, (len(positions), 2))

    memory_graph = grapher.make_memory_graph(positions)

    # Use assert statements instead of returning a value
    standard_image = grapher.draw_graph(memory_graph)
    assert standard_image.shape == (300, 400, 3)

    aged_image = grapher.draw_memory_graph(memory_graph, use_age_colors=True)
    assert aged_image.shape == (300, 400, 3)
    
if __name__ == "__main__":
    test_memory_graph_drawing()
