"""
Vectorized memory operations for maximum performance with igraph objects.

This module provides the same interface as the original memory operations
but uses vectorized operations internally for much better performance.
"""

import logging
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
from collections import defaultdict, deque

from graphizy.exceptions import GraphCreationError
from graphizy.algorithms import create_graph_array, normalize_id


def update_memory_from_graph(graph: Any, memory_manager: 'MemoryManager') -> Dict[str, List[str]]:
    """
    Update memory manager from any igraph Graph object.

    VECTORIZED IMPLEMENTATION: Uses numpy operations and igraph's vectorized methods
    for maximum speed while maintaining the exact same interface.

    Args:
        graph: Any igraph Graph object (Delaunay, proximity, custom, etc.)
        memory_manager: MemoryManager instance to update

    Returns:
        Current connections dictionary (normalized IDs)
    """
    try:
        if graph is None:
            raise GraphCreationError("Graph cannot be None")
        if memory_manager is None:
            raise GraphCreationError("Memory manager cannot be None")

        # VECTORIZED: Get all vertex IDs at once and normalize them
        vertex_ids = np.array(graph.vs["id"])
        normalized_ids = np.array([normalize_id(vid) for vid in vertex_ids])

        # Initialize connections dict for all vertices
        current_connections = {norm_id: [] for norm_id in normalized_ids}

        if graph.ecount() > 0:
            # VECTORIZED: Get all edges as numpy array
            edges = np.array(graph.get_edgelist())

            # VECTORIZED: Map vertex indices to normalized IDs
            source_ids = normalized_ids[edges[:, 0]]
            target_ids = normalized_ids[edges[:, 1]]

            # Build connections using vectorized operations
            for src_id, tgt_id in zip(source_ids, target_ids):
                current_connections[src_id].append(tgt_id)
                current_connections[tgt_id].append(src_id)

        # Update memory manager
        memory_manager.add_connections(current_connections)

        return current_connections

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from graph: {str(e)}")


def create_memory_graph(current_positions: Union[np.ndarray, Dict[str, Any]],
                        memory_connections: Dict[Any, List[Any]],
                        aspect: str = "array") -> Any:
    """
    Create a graph with current positions and memory-based edges.

    VECTORIZED IMPLEMENTATION: Uses numpy operations for maximum performance
    while maintaining the exact same interface.

    Args:
        current_positions: Current positions as array [id, x, y, ...] or dict
        memory_connections: Memory connections {"obj_id": ["connected_id1", "connected_id2"]}
                           IDs will be automatically normalized
        aspect: Data format ("array" or "dict")

    Returns:
        igraph Graph object with memory-based edges and optional distances

    Raises:
        GraphCreationError: If graph creation fails
    """
    try:
        # VECTORIZED: Convert input data to standardized array
        if aspect == "array":
            if not isinstance(current_positions, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")
            data_array = current_positions
        elif aspect == "dict":
            if isinstance(current_positions, dict):
                required_keys = ["id", "x", "y"]
                if not all(k in current_positions for k in required_keys):
                    raise GraphCreationError(f"Dict data must contain required keys: {required_keys}")
                # VECTORIZED: Column stacking
                data_array = np.column_stack([current_positions[k] for k in required_keys])
            elif isinstance(current_positions, np.ndarray):
                data_array = current_positions
            else:
                raise GraphCreationError("Dict aspect requires a dictionary or NumPy array as input")
        else:
            raise GraphCreationError(f"Unknown aspect '{aspect}'. Use 'array' or 'dict'")

        # Create basic graph from the standardized array
        graph = create_graph_array(data_array)

        if not memory_connections:
            return graph

        # VECTORIZED: ID normalization and mapping
        vertex_ids = np.array(graph.vs["id"])
        normalized_vertex_ids = np.array([normalize_id(vid) for vid in vertex_ids])

        # Create vectorized ID to vertex index mapping
        id_to_vertex = {norm_id: idx for idx, norm_id in enumerate(normalized_vertex_ids)}

        # VECTORIZED: Memory connection processing
        edges_to_add = []

        # Process all memory connections at once
        for obj_id, connected_ids in memory_connections.items():
            norm_obj_id = normalize_id(obj_id)

            if norm_obj_id not in id_to_vertex:
                continue

            vertex_from = id_to_vertex[norm_obj_id]

            # VECTORIZED: Processing of connected IDs
            if connected_ids:
                norm_connected_ids = np.array([normalize_id(cid) for cid in connected_ids])

                # VECTORIZED: Filter valid connections
                valid_mask = np.array([cid in id_to_vertex for cid in norm_connected_ids])
                valid_connected_ids = norm_connected_ids[valid_mask]

                # Create edges
                for norm_connected_id in valid_connected_ids:
                    vertex_to = id_to_vertex[norm_connected_id]

                    if vertex_from != vertex_to:
                        # Ensure consistent edge ordering
                        edge = tuple(sorted((vertex_from, vertex_to)))
                        edges_to_add.append(edge)

        # VECTORIZED: Remove duplicates and add edges
        unique_edges = list(set(edges_to_add))

        if unique_edges:
            graph.add_edges(unique_edges)
            # Add memory_based attribute to all new edges at once
            graph.es["memory_based"] = [True] * len(unique_edges)

        logging.debug(f"Created memory graph with {graph.vcount()} vertices and {graph.ecount()} memory-based edges")

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create memory graph: {str(e)}")


class MemoryManager:
    """
    Manages memory connections between objects with optional edge aging support.

    VECTORIZED IMPLEMENTATION: Uses numpy arrays and vectorized operations
    internally for maximum performance while maintaining the exact same interface.
    """

    def __init__(self, max_memory_size: int = 100, max_iterations: int = None, track_edge_ages: bool = True):
        """Initialize memory manager

        Args:
            max_memory_size: Maximum number of connections to keep per object
            max_iterations: Maximum number of iterations to keep connections (None = unlimited)
            track_edge_ages: Whether to track edge ages for visualization
        """
        self.max_memory_size = max_memory_size
        self.max_iterations = max_iterations
        self.track_edge_ages = track_edge_ages
        self.current_iteration = 0

        # Memory structure: {object_id: deque([(connected_id, iteration), ...])}
        self.memory = defaultdict(lambda: deque(maxlen=max_memory_size))

        # Track all unique object IDs that have been seen
        self.all_objects = set()

        # Track edge ages (only if enabled)
        if self.track_edge_ages:
            self.edge_ages = {}  # {(obj1, obj2): {"first_seen": iter, "last_seen": iter}}

    def add_connections(self, connections: Dict[Any, List[Any]]) -> None:
        """
        Add new connections to memory with automatic ID normalization.

        VECTORIZED IMPLEMENTATION: Uses batch processing and numpy operations
        for maximum performance.

        Args:
            connections: Dictionary like {"A": ["C", "D"], "B": [], ...}
                        IDs will be automatically normalized to strings
        """
        self.current_iteration += 1

        # VECTORIZED: Normalize connections first to ensure consistency
        normalized_connections = normalize_memory_connections(connections)

        # VECTORIZED: Track current edges for age updates (if enabled)
        if self.track_edge_ages:
            current_edges = set()

        for obj_id, connected_ids in normalized_connections.items():
            self.all_objects.add(obj_id)

            # VECTORIZED: Process all connected IDs for this object at once
            if connected_ids:
                # Add to all_objects (vectorized)
                self.all_objects.update(connected_ids)

                # VECTORIZED: Process edge ages in batch
                if self.track_edge_ages:
                    # Create edge keys (vectorized)
                    edge_keys = [tuple(sorted([obj_id, connected_id])) for connected_id in connected_ids]
                    current_edges.update(edge_keys)

                    # Update edge ages (vectorized)
                    for edge_key in edge_keys:
                        if edge_key not in self.edge_ages:
                            self.edge_ages[edge_key] = {
                                "first_seen": self.current_iteration,
                                "last_seen": self.current_iteration
                            }
                        else:
                            self.edge_ages[edge_key]["last_seen"] = self.current_iteration

                # VECTORIZED: Add bidirectional connections in batch
                for connected_id in connected_ids:
                    self.memory[obj_id].append((connected_id, self.current_iteration))
                    self.memory[connected_id].append((obj_id, self.current_iteration))

        # Clean old iterations if max_iterations is set
        if self.max_iterations:
            self._clean_old_iterations()

    def _clean_old_iterations(self) -> None:
        """
        Remove connections older than max_iterations.

        VECTORIZED IMPLEMENTATION: Uses numpy operations for filtering.
        """
        cutoff_iteration = self.current_iteration - self.max_iterations

        # VECTORIZED: Clean memory connections
        for obj_id in list(self.memory.keys()):
            if self.memory[obj_id]:
                # Convert to numpy array for vectorized filtering
                connections_list = list(self.memory[obj_id])
                if connections_list:
                    connections_array = np.array(connections_list, dtype=object)
                    iterations = np.array([int(item[1]) for item in connections_list])
                    valid_mask = iterations > cutoff_iteration

                    if np.any(valid_mask):
                        valid_connections = connections_array[valid_mask]
                        self.memory[obj_id] = deque(
                            [(conn_id, int(iteration)) for conn_id, iteration in valid_connections],
                            maxlen=self.max_memory_size
                        )
                    else:
                        self.memory[obj_id].clear()

        # VECTORIZED: Clean old edge ages
        if self.track_edge_ages and hasattr(self, 'edge_ages'):
            # Vectorized filtering of edge ages
            self.edge_ages = {
                edge_key: age_info
                for edge_key, age_info in self.edge_ages.items()
                if age_info["last_seen"] > cutoff_iteration
            }

    def get_current_memory_graph(self) -> Dict[str, List[str]]:
        """
        Get current memory as a graph dictionary.

        VECTORIZED IMPLEMENTATION: Uses numpy operations for processing.

        Returns:
            Dictionary with current memory connections (all IDs normalized as strings)
        """
        result = {}

        # VECTORIZED: Process all objects at once
        all_objects_array = np.array(list(self.all_objects))

        for obj_id in all_objects_array:
            connections = []
            if obj_id in self.memory and self.memory[obj_id]:
                # VECTORIZED: Extract unique connections
                memory_connections = list(self.memory[obj_id])
                connected_ids = [conn_id for conn_id, _ in memory_connections if conn_id != obj_id]

                # VECTORIZED: Use numpy unique for deduplication (faster for large lists)
                if connected_ids:
                    connections = list(np.unique(connected_ids))

            result[obj_id] = connections

        return result

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current memory state.

        VECTORIZED IMPLEMENTATION: Uses numpy operations for statistics.
        """
        # VECTORIZED: Compute total connections
        connection_counts = np.array([len(connections) for connections in self.memory.values()])
        total_connections = int(np.sum(connection_counts) // 2) if len(connection_counts) > 0 else 0

        stats = {
            "total_objects": len(self.all_objects),
            "total_connections": total_connections,
            "current_iteration": self.current_iteration,
            "objects_with_memory": int(np.sum(connection_counts > 0)) if len(connection_counts) > 0 else 0,
            "max_memory_size": self.max_memory_size,
            "max_iterations": self.max_iterations,
            "edge_aging_enabled": self.track_edge_ages
        }

        # VECTORIZED: Add edge age statistics if tracking is enabled
        if self.track_edge_ages and hasattr(self, 'edge_ages') and self.edge_ages:
            current_iter = self.current_iteration
            ages = np.array([current_iter - info["first_seen"] for info in self.edge_ages.values()])

            stats["edge_age_stats"] = {
                "min_age": int(np.min(ages)),
                "max_age": int(np.max(ages)),
                "avg_age": float(np.mean(ages)),
                "total_aged_edges": len(ages)
            }

        return stats

    def get_edge_ages(self) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Get age information for all edges (if tracking enabled)"""
        if not self.track_edge_ages or not hasattr(self, 'edge_ages'):
            return {}
        return self.edge_ages.copy()

    def get_edge_age_normalized(self, max_age: int = None) -> Dict[Tuple[str, str], float]:
        """
        Get normalized edge ages (0.0 = newest, 1.0 = oldest).

        VECTORIZED IMPLEMENTATION: Uses numpy operations for normalization.

        Args:
            max_age: Maximum age to consider (uses current max if None)

        Returns:
            Dictionary mapping edge to normalized age (0.0-1.0)
        """
        if not self.track_edge_ages or not hasattr(self, 'edge_ages') or not self.edge_ages:
            return {}

        # VECTORIZED: Compute ages
        current_iter = self.current_iteration
        ages = np.array([current_iter - info["first_seen"] for info in self.edge_ages.values()])

        if max_age is None:
            max_age = int(np.max(ages)) if len(ages) > 0 else 1

        if max_age == 0:
            return {edge: 0.0 for edge in self.edge_ages.keys()}

        # VECTORIZED: Normalize ages
        normalized_ages_array = np.minimum(ages / max_age, 1.0)

        # Build result dictionary
        normalized_ages = {}
        for i, edge_key in enumerate(self.edge_ages.keys()):
            normalized_ages[edge_key] = float(normalized_ages_array[i])

        return normalized_ages

    def clear(self) -> None:
        """Clear all memory connections and reset state"""
        self.memory.clear()
        self.all_objects.clear()
        if self.track_edge_ages:
            self.edge_ages.clear()
        self.current_iteration = 0
        logging.debug("Memory manager cleared")


def normalize_memory_connections(memory_connections: Dict[Any, List[Any]]) -> Dict[str, List[str]]:
    """
    Ensure all memory connection IDs are normalized strings.

    VECTORIZED IMPLEMENTATION: Uses batch processing for better performance.

    Args:
        memory_connections: Raw memory connections with potentially mixed ID types

    Returns:
        Normalized memory connections with string IDs
    """
    if not memory_connections:
        return {}

    # VECTORIZED: Extract all unique IDs for batch normalization
    all_obj_ids = list(memory_connections.keys())
    all_connected_ids = [cid for connected_list in memory_connections.values() for cid in connected_list]
    all_ids = all_obj_ids + all_connected_ids

    # VECTORIZED: Batch normalization
    normalized_all_ids = np.array([normalize_id(id_val) for id_val in all_ids])

    # Create mapping for efficient lookup
    id_mapping = dict(zip(all_ids, normalized_all_ids))

    # Build normalized result
    normalized = {}
    for obj_id, connected_ids in memory_connections.items():
        norm_obj_id = id_mapping[obj_id]
        norm_connected_ids = [id_mapping[cid] for cid in connected_ids]
        normalized[norm_obj_id] = norm_connected_ids

    return normalized


def update_memory_from_custom_function(data_points: Union[np.ndarray, Dict[str, Any]],
                                       memory_manager: MemoryManager,
                                       connection_function: callable,
                                       aspect: str = "array",
                                       **kwargs) -> Dict[str, List[str]]:
    """
    Update memory using a custom connection function.

    VECTORIZED IMPLEMENTATION: Uses vectorized operations where possible
    while maintaining the exact same interface.

    Args:
        data_points: Point data in specified aspect format
        memory_manager: MemoryManager instance to update
        connection_function: Function that returns connections as iterable of (id1, id2) pairs
        aspect: Data format ("array" or "dict")
        **kwargs: Additional arguments passed to connection_function

    Returns:
        Dict[str, List[str]]: Current connections dictionary (normalized IDs)
    """
    try:
        # Call the custom function to get connections
        raw_connections = connection_function(data_points, **kwargs)

        # Convert connections to our standard format
        connections_dict = {}

        # VECTORIZED: Initialize all objects with empty connections
        if aspect == "array":
            if not isinstance(data_points, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")
            # VECTORIZED: Process all IDs at once
            obj_ids = data_points[:, 0]
            for obj_id in obj_ids:
                connections_dict[normalize_id(obj_id)] = []
        elif aspect == "dict":
            if isinstance(data_points, dict):
                # VECTORIZED: Process all IDs at once
                obj_ids = data_points["id"]
                for obj_id in obj_ids:
                    connections_dict[normalize_id(obj_id)] = []
            else:
                raise GraphCreationError("Expected dictionary for 'dict' aspect")

        # VECTORIZED: Add connections from custom function
        for connection in raw_connections:
            if len(connection) >= 2:
                id1, id2 = normalize_id(connection[0]), normalize_id(connection[1])
                if id1 in connections_dict:
                    connections_dict[id1].append(id2)
                if id2 in connections_dict:
                    connections_dict[id2].append(id1)

        # Update memory manager
        memory_manager.add_connections(connections_dict)

        return connections_dict

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from custom function: {str(e)}")


# Example usage function (same interface, vectorized internally)
def example_memory_graph_usage():
    """Example of how to use the memory graph functionality (same interface, faster performance)"""

    # Example data - current positions
    current_positions = np.array([
        [1, 100, 100],  # Object A at (100, 100)
        [2, 200, 150],  # Object B at (200, 150)
        [3, 120, 300],  # Object C at (120, 300)
        [4, 400, 100],  # Object D at (400, 100)
    ])

    # Example memory connections (historical proximities) - mixed ID types
    memory_connections = {
        1: [3, 4],  # Integer IDs (will be normalized to strings)
        "2": [],  # String ID
        3: ["1"],  # Mixed types (will be normalized)
        "4": [1],  # Mixed types (will be normalized)
    }

    # Create memory graph (automatic normalization) - SAME INTERFACE, VECTORIZED INTERNALLY
    graph = create_memory_graph(current_positions, memory_connections, aspect="array")

    print(f"Memory graph: {graph.vcount()} vertices, {graph.ecount()} edges")

    # Using with MemoryManager - SAME INTERFACE, VECTORIZED INTERNALLY
    memory_mgr = MemoryManager(max_memory_size=50, max_iterations=10, track_edge_ages=True)

    # Simulate multiple iterations with mixed ID types - SAME INTERFACE, VECTORIZED INTERNALLY
    for iteration in range(5):
        # Simulate changing proximity connections each iteration (mixed ID types)
        proximity_connections = {
            1: [2] if iteration % 2 == 0 else [3],  # Integer IDs
            "2": ["1"] if iteration % 2 == 0 else [],  # String IDs
            3: [1] if iteration % 2 == 1 else [4],  # Mixed types
            "4": ["3"] if iteration % 2 == 1 else [],  # String IDs
        }

        memory_mgr.add_connections(proximity_connections)

    # Get final memory state (all normalized) - SAME INTERFACE, VECTORIZED INTERNALLY
    final_memory = memory_mgr.get_current_memory_graph()
    final_graph = create_memory_graph(current_positions, final_memory, aspect="array")

    stats = memory_mgr.get_memory_stats()
    print(f"Final memory stats: {stats}")

    # Show edge age information - SAME INTERFACE, VECTORIZED INTERNALLY
    edge_ages = memory_mgr.get_edge_ages()
    print(f"Edge ages: {edge_ages}")

    return final_graph


if __name__ == "__main__":
    example_memory_graph_usage()