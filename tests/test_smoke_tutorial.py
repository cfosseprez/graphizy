"""
Smoke test to ensure basic tutorial functionality works.

This test covers the core object creation and method calls from the basic
tutorial to catch any breaking changes in the main API.
"""

import pytest
from graphizy import Graphing, GraphizyConfig, generate_and_format_positions


def test_basic_tutorial_smoke_test():
    """
    Walks through the basic steps of the tutorial to ensure no exceptions are raised.
    """
    # 1. Test configuration creation
    try:
        config = GraphizyConfig(dimension=(1000, 800))
    except Exception as e:
        pytest.fail(f"GraphizyConfig creation failed: {e}")

    # 2. Test Graphing object creation
    try:
        grapher = Graphing(config=config)
    except Exception as e:
        pytest.fail(f"Graphing object creation failed: {e}")

    # 3. Test update_config method
    try:
        grapher.update_config(
            drawing={
                "point_color": (255, 100, 100),
                "line_color": (100, 100, 255),
                "point_radius": 8,
                "line_thickness": 2
            }
        )
    except Exception as e:
        pytest.fail(f"Drawing config update failed: {e}")

    # 4. Test memory manager initialization
    try:
        grapher.init_memory_manager(max_memory_size=100, track_edge_ages=True)
    except Exception as e:
        pytest.fail(f"Memory manager initialization failed: {e}")

    # 5. Test data generation
    try:
        data = generate_and_format_positions(1000, 800, 20)
        assert data.shape == (20, 3), "Generated data has incorrect shape"
    except Exception as e:
        pytest.fail(f"Data generation failed: {e}")

    # 6. Test graph creation
    try:
        graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)
        assert graph.vcount() == 20, "Graph has incorrect number of vertices"
    except Exception as e:
        pytest.fail(f"Graph creation failed: {e}")