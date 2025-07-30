"""
Tests for the MemoryManager and memory-based graph functionality.
"""
import pytest
import numpy as np
from graphizy.algorithms import MemoryManager, create_memory_graph, normalize_id
from graphizy import Graphing
from graphizy.exceptions import GraphCreationError

@pytest.fixture
def mem_manager():
    """Provides a MemoryManager instance."""
    return MemoryManager(max_memory_size=5, max_iterations=3, track_edge_ages=True)

def test_memory_manager_add_connections(mem_manager):
    """Test adding connections and iteration counting."""
    connections = {"1": ["2", "3"], "2": ["1"]}
    mem_manager.add_connections(connections)
    assert mem_manager.current_iteration == 1
    assert "1" in mem_manager.all_objects
    assert "3" in mem_manager.all_objects

    mem_graph = mem_manager.get_current_memory_graph()
    assert "2" in mem_graph["1"]
    assert "1" in mem_graph["2"]  # Check bidirectionality

def test_memory_manager_aging(mem_manager):
    """Test edge aging and statistics."""
    mem_manager.add_connections({"1": ["2"]})  # iter 1
    mem_manager.add_connections({"1": ["2"], "2": ["3"]})  # iter 2
    mem_manager.add_connections({"1": ["2"], "2": ["3"]})  # iter 3

    stats = mem_manager.get_memory_stats()
    assert stats["current_iteration"] == 3
    assert stats["edge_age_stats"]["total_aged_edges"] == 2

    ages = mem_manager.get_edge_age_normalized()
    # Edge (1,2) was seen first at iter 1, age is 3-1=2
    # Edge (2,3) was seen first at iter 2, age is 3-2=1
    # Max age is 2.
    assert ages[('1', '2')] == 1.0  # Normalized age is age/max_age = 2/2
    assert ages[('2', '3')] == 0.5  # Normalized age is 1/2

def test_create_memory_graph_with_id_normalization(sample_array_data):
    """Test that memory graph creation handles mixed float/string IDs."""
    # sample_array_data has float IDs (1.0, 2.0, ...)
    # memory_connections has string IDs ("1", "2", ...)
    memory_connections = {"1": ["2"], "3": ["4"]}
    graph = create_memory_graph(sample_array_data, memory_connections)
    assert graph.ecount() == 2

def test_graphing_memory_integration(grapher, sample_array_data):
    """Test the full memory workflow through the Graphing class."""
    grapher.init_memory_manager(max_memory_size=10, track_edge_ages=True)
    assert grapher.memory_manager is not None

    # Update memory
    grapher.update_memory_with_proximity(sample_array_data, proximity_thresh=50.0)
    grapher.update_memory_with_proximity(sample_array_data, proximity_thresh=110.0)

    # Create memory graph
    mem_graph = grapher.make_memory_graph(sample_array_data)
    assert mem_graph.ecount() > 0

    stats = grapher.get_memory_stats()
    assert stats["current_iteration"] == 2