"""
Ultra-Simplified Memory System for Graphizy

This implementation uses the simplest possible approach:
1. Store edges as a list (allowing duplicates)
2. Add all edges to graph (including duplicates)
3. Use igraph's simplify() to handle deduplication
4. Minimal overhead, maximum performance

The key insight: Let igraph's C++ implementation handle the complexity
instead of managing it in Python.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Optional
import igraph as ig
from collections import deque
import time

from graphizy.exceptions import GraphCreationError


class MemoryManager:
    """
    Ultra-simplified MemoryManager using igraph's built-in deduplication

    This implementation is radically simplified:
    - No complex data structures
    - No manual deduplication
    - Just store edges and let igraph handle everything
    """

    def __init__(self,
                 max_memory_size: int = 10000,
                 max_iterations: int = None,
                 track_edge_ages: bool = True):
        """
        Initialize simplified memory manager

        Args:
            max_memory_size: Maximum number of edges to store (before cleanup)
            max_iterations: Not used in this implementation (kept for API compatibility)
            track_edge_ages: Whether to track when edges were added
        """
        self.max_memory_size = max_memory_size
        self.max_iterations = max_iterations  # Kept for API compatibility
        self.track_edge_ages = track_edge_ages
        self.current_iteration = 0

        # Use deque for efficient, fixed-size storage.
        # It automatically discards the oldest items when full.
        self._edges = deque(maxlen=self.max_memory_size)
        if self.track_edge_ages:
            self._edge_iterations = deque(maxlen=self.max_memory_size)
        else:
            # Keep the attribute for consistent API, even if unused
            self._edge_iterations = None

        # Compatibility attributes
        self.all_objects = set()   # Track all vertex IDs seen

        # Performance tracking
        self._stats = {
            'update_times': [],
            'graph_creation_times': []
        }

    def add_graph_vectorized(self,
                           graph: ig.Graph,
                           return_memory_graph: bool = False) -> Union[Dict[str, Any], ig.Graph]:
        """
        Add graph edges to memory and optionally return memory-enhanced graph

        This is the main method - extremely simplified:
        1. Extract edges from graph
        2. Add to our edge list
        3. If requested, create memory graph using simplify()
        """
        start_time = time.perf_counter()
        self.current_iteration += 1

        # Extract vertex IDs once
        vertex_ids = graph.vs["id"]

        # Update tracked objects
        self.all_objects.update(str(vid) for vid in vertex_ids)

        # Extract edges - no deduplication, just collect them
        edge_count = 0
        for edge in graph.es:
            v1_idx, v2_idx = edge.tuple
            v1_id = str(vertex_ids[v1_idx])
            v2_id = str(vertex_ids[v2_idx])

            # Store normalized edge (smaller ID first)
            self._edges.append((min(v1_id, v2_id), max(v1_id, v2_id)))

            if self.track_edge_ages:
                self._edge_iterations.append(self.current_iteration)

            edge_count += 1


        update_time = time.perf_counter() - start_time
        self._stats['update_times'].append(update_time)

        if return_memory_graph:
            graph_start = time.perf_counter()
            memory_graph = self._create_memory_graph_ultra_simple(graph)
            graph_time = time.perf_counter() - graph_start
            self._stats['graph_creation_times'].append(graph_time)

            logging.debug(f"Memory update: {update_time*1000:.1f}ms, "
                         f"Graph creation: {graph_time*1000:.1f}ms")
            return memory_graph
        else:
            return {
                'edges_processed': edge_count,
                'total_time_ms': update_time * 1000,
                'memory_size': len(self._edges),
                'iteration': self.current_iteration
            }

    def _create_memory_graph_ultra_simple(self, current_graph: ig.Graph) -> ig.Graph:
        """
        Create memory graph using igraph's simplify() method

        This is the key optimization - we add ALL edges (including duplicates)
        and let igraph's C++ simplify() method handle deduplication efficiently.
        """
        # Copy current graph
        memory_graph = current_graph.copy()

        if not self._edges:
            return memory_graph

        # Get vertex mapping
        vertex_ids = [str(vid) for vid in memory_graph.vs["id"]]
        vertex_id_to_idx = {vid: i for i, vid in enumerate(vertex_ids)}

        # Collect all edges to add (including duplicates!)
        edges_to_add = []
        edge_ages = []

        for i, (v1_id, v2_id) in enumerate(self._edges):
            # Only add if both vertices exist in current graph
            if v1_id in vertex_id_to_idx and v2_id in vertex_id_to_idx:
                v1_idx = vertex_id_to_idx[v1_id]
                v2_idx = vertex_id_to_idx[v2_id]
                edges_to_add.append((v1_idx, v2_idx))

                if self.track_edge_ages:
                    age = self.current_iteration - self._edge_iterations[i]
                    edge_ages.append(age)

        if edges_to_add:
            # Store counts for attribute initialization
            n_original = memory_graph.ecount()
            n_to_add = len(edges_to_add)

            # Add ALL edges at once (including duplicates)
            memory_graph.add_edges(edges_to_add)

            # Initialize attributes for all edges
            # Current edges are not memory-based
            memory_graph.es["memory_based"] = [False] * n_original + [True] * n_to_add

            # Set weights (will be summed for duplicates)
            memory_graph.es["weight"] = [1.0] * n_original + [1.0] * n_to_add

            # Set ages
            if self.track_edge_ages:
                memory_graph.es["age"] = [0] * n_original + edge_ages
            else:
                memory_graph.es["age"] = [0] * (n_original + n_to_add)

            # Now the magic: use igraph's simplify to merge duplicates
            # This is implemented in C++ and is very fast
            memory_graph.simplify(
                multiple=True,  # Remove multiple edges
                loops=True,     # Remove self-loops
                combine_edges={
                    "memory_based": "min",   # If an edge is current (False), it's not memory-based.
                    "weight": "sum",         # Sum weights of duplicates
                    "age": "min"            # Keep the oldest age
                }
            )

        return memory_graph

    def create_memory_graph(self, data_points: Union[np.ndarray, Dict[str, Any]]) -> ig.Graph:
        """
        Create a memory-only graph from current positions

        This creates a graph with vertices from data_points but edges only from memory.
        """
        from graphizy.algorithms import create_graph_array

        # Create graph with vertices only
        if isinstance(data_points, dict):
            data_array = np.column_stack((
                data_points["id"],
                data_points["x"],
                data_points["y"]
            ))
        else:
            data_array = data_points

        graph = create_graph_array(data_array)

        # Add memory edges
        return self._create_memory_graph_ultra_simple(graph)

    def get_current_memory_graph(self) -> Dict[str, List[str]]:
        """
        COMPATIBILITY METHOD: Return memory in dict format

        This is kept for backward compatibility but is less efficient.
        """
        # Count unique edges and build adjacency
        adjacency = {str(obj_id): set() for obj_id in self.all_objects}

        for v1_id, v2_id in self._edges:
            if v1_id in adjacency and v2_id in adjacency:
                adjacency[v1_id].add(v2_id)
                adjacency[v2_id].add(v1_id)

        # Convert sets to lists
        return {obj_id: list(neighbors) for obj_id, neighbors in adjacency.items()}

    def add_connections(self, connections: Dict[Any, List[Any]]) -> None:
        """
        COMPATIBILITY METHOD: Add connections in dict format

        This is kept for backward compatibility.
        """
        self.current_iteration += 1

        for obj_id, connected_ids in connections.items():
            obj_id_str = str(obj_id)
            self.all_objects.add(obj_id_str)

            for connected_id in connected_ids:
                connected_id_str = str(connected_id)
                self.all_objects.add(connected_id_str)

                # Add normalized edge
                edge = (min(obj_id_str, connected_id_str),
                       max(obj_id_str, connected_id_str))
                self._edges.append(edge)

                if self.track_edge_ages:
                    self._edge_iterations.append(self.current_iteration)

        # Cleanup if needed
        if len(self._edges) > self.max_memory_size * 2:
            keep_from = len(self._edges) - self.max_memory_size
            self._edges = self._edges[keep_from:]
            if self.track_edge_ages:
                self._edge_iterations = self._edge_iterations[keep_from:]

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        # Count unique edges
        unique_edges = len(set(self._edges))

        stats = {
            "total_objects": len(self.all_objects),
            "total_connections": len(self._edges),
            "unique_connections": unique_edges,
            "duplicate_ratio": 1 - (unique_edges / len(self._edges)) if self._edges else 0,
            "current_iteration": self.current_iteration,
            "max_memory_size": self.max_memory_size,
            "edge_aging_enabled": self.track_edge_ages
        }

        if self._stats['update_times']:
            stats["avg_update_time_ms"] = np.mean(self._stats['update_times']) * 1000
            stats["avg_graph_creation_time_ms"] = (
                np.mean(self._stats['graph_creation_times']) * 1000
                if self._stats['graph_creation_times'] else 0
            )

        if self.track_edge_ages and self._edge_iterations:
            ages = [self.current_iteration - iter_num for iter_num in self._edge_iterations]
            stats["edge_age_stats"] = {
                "min_age": min(ages),
                "max_age": max(ages),
                "avg_age": float(np.mean(ages)),
                "total_aged_edges": len(ages)
            }

        return stats

    def clear(self) -> None:
        """Clear all memory"""
        self._edges.clear()
        self._edge_iterations.clear()
        self.all_objects.clear()
        self.current_iteration = 0
        self._stats = {
            'update_times': [],
            'graph_creation_times': []
        }


# Standalone compatibility functions
def create_memory_graph(current_positions: Union[np.ndarray, Dict[str, Any]],
                       memory_connections: Dict[Any, List[Any]],
                       aspect: str = "array") -> Any:
    """
    STANDALONE FUNCTION: Create memory graph

    This uses the ultra-simple approach: create graph, add all edges, simplify.
    """
    try:
        from graphizy.algorithms import create_graph_array, normalize_id

        # Create base graph
        if aspect == "array":
            if not isinstance(current_positions, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")
            graph = create_graph_array(current_positions)
        elif aspect == "dict":
            if isinstance(current_positions, dict):
                required_keys = ["id", "x", "y"]
                if not all(k in current_positions for k in required_keys):
                    raise GraphCreationError(f"Dict must contain keys: {required_keys}")
                data_array = np.column_stack(tuple(current_positions[k] for k in required_keys))
                graph = create_graph_array(data_array)
            elif isinstance(current_positions, np.ndarray):
                graph = create_graph_array(current_positions)
            else:
                raise GraphCreationError("Dict aspect requires dict or array")
        else:
            raise GraphCreationError(f"Unknown aspect '{aspect}'")

        # Get vertex mapping
        id_to_vertex = {normalize_id(v["id"]): v.index for v in graph.vs}

        # Add all edges (including duplicates)
        edges_to_add = []
        for obj_id, connected_ids in memory_connections.items():
            norm_obj_id = normalize_id(obj_id)
            if norm_obj_id not in id_to_vertex:
                continue

            v_from = id_to_vertex[norm_obj_id]
            for connected_id in connected_ids:
                norm_conn_id = normalize_id(connected_id)
                if norm_conn_id in id_to_vertex:
                    v_to = id_to_vertex[norm_conn_id]
                    edges_to_add.append((v_from, v_to))

        # Add edges and simplify
        if edges_to_add:
            graph.add_edges(edges_to_add)
            graph.es["memory_based"] = [True] * len(edges_to_add)
            graph.simplify()  # Remove duplicates

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create memory graph: {str(e)}")


def update_memory_from_graph(graph: Any, memory_manager: MemoryManager) -> Dict[str, List[str]]:
    """STANDALONE FUNCTION: Update memory from graph"""
    try:
        if graph is None or memory_manager is None:
            raise GraphCreationError("Graph and memory manager cannot be None")

        memory_manager.add_graph_vectorized(graph, return_memory_graph=False)
        return memory_manager.get_current_memory_graph()

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from graph: {str(e)}")


def update_memory_from_custom_function(data_points: Union[np.ndarray, Dict[str, Any]],
                                     memory_manager: MemoryManager,
                                     connection_function: callable,
                                     aspect: str = "array",
                                     **kwargs) -> Dict[str, List[str]]:
    """STANDALONE FUNCTION: Update memory using custom function"""
    try:
        raw_connections = connection_function(data_points, **kwargs)

        connections_dict = {}

        # Initialize empty connections
        if aspect == "array":
            if not isinstance(data_points, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")
            for obj_id in data_points[:, 0]:
                from graphizy.algorithms import normalize_id
                connections_dict[normalize_id(obj_id)] = []
        elif aspect == "dict":
            if isinstance(data_points, dict):
                for obj_id in data_points["id"]:
                    from graphizy.algorithms import normalize_id
                    connections_dict[normalize_id(obj_id)] = []
            else:
                raise GraphCreationError("Expected dictionary for 'dict' aspect")

        # Add connections
        for connection in raw_connections:
            if len(connection) >= 2:
                from graphizy.algorithms import normalize_id
                id1, id2 = normalize_id(connection[0]), normalize_id(connection[1])
                if id1 in connections_dict:
                    connections_dict[id1].append(id2)
                if id2 in connections_dict:
                    connections_dict[id2].append(id1)

        memory_manager.add_connections(connections_dict)
        return connections_dict

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from custom function: {str(e)}")


# Example usage
if __name__ == "__main__":
    # Demonstrate the ultra-simple approach
    print("=== ULTRA-SIMPLE MEMORY SYSTEM ===\n")

    # Create test data
    positions = np.array([
        [1, 100, 100],
        [2, 200, 150],
        [3, 120, 300],
        [4, 400, 100],
        [5, 250, 250]
    ])

    # Initialize memory manager
    memory_mgr = MemoryManager(max_memory_size=1000, track_edge_ages=True)

    # Create test graph
    import igraph as ig
    graph = ig.Graph()
    graph.add_vertices(5)
    graph.vs["id"] = ["1", "2", "3", "4", "5"]
    graph.vs["x"] = positions[:, 1]
    graph.vs["y"] = positions[:, 2]
    graph.add_edges([(0, 1), (1, 2), (2, 3)])

    # Test performance
    import time

    # Update memory 100 times
    start = time.perf_counter()
    for i in range(100):
        memory_mgr.add_graph_vectorized(graph, return_memory_graph=False)
    update_time = (time.perf_counter() - start) / 100 * 1000

    # Create memory graph
    start = time.perf_counter()
    memory_graph = memory_mgr.add_graph_vectorized(graph, return_memory_graph=True)
    create_time = (time.perf_counter() - start) * 1000

    print(f"Average update time: {update_time:.2f}ms")
    print(f"Graph creation time: {create_time:.2f}ms")
    print(f"Memory edges: {memory_graph.ecount() - graph.ecount()}")
    print(f"\nStats: {memory_mgr.get_memory_stats()}")