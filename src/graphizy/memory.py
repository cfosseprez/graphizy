

import logging
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
from collections import defaultdict, deque

from graphizy.exceptions import (
    InvalidPointArrayError, SubdivisionError, TriangulationError,
    GraphCreationError, PositionGenerationError, DependencyError,
    IgraphMethodError
)
from graphizy.algorithms import (create_graph_array,
DataInterface,
create_graph_dict,
normalize_id,
normalize_id,
get_distance,
make_subdiv,
graph_delaunay,)


class MemoryManager:
    """Manages memory connections between objects with optional edge aging support"""

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

    def add_connections(self, connections: Dict[str, List[str]]) -> None:
        """Add new connections to memory

        Args:
            connections: Dictionary like {"A": ["C", "D"], "B": [], ...}
        """
        self.current_iteration += 1

        # Track current edges for age updates (if enabled)
        if self.track_edge_ages:
            current_edges = set()

        for obj_id, connected_ids in connections.items():
            self.all_objects.add(obj_id)

            # Add each connection with current iteration timestamp
            for connected_id in connected_ids:
                self.all_objects.add(connected_id)

                # Track edge age (if enabled)
                if self.track_edge_ages:
                    edge_key = tuple(sorted([obj_id, connected_id]))
                    current_edges.add(edge_key)

                    if edge_key not in self.edge_ages:
                        self.edge_ages[edge_key] = {
                            "first_seen": self.current_iteration,
                            "last_seen": self.current_iteration
                        }
                    else:
                        self.edge_ages[edge_key]["last_seen"] = self.current_iteration

                # Add bidirectional connections
                self.memory[obj_id].append((connected_id, self.current_iteration))
                self.memory[connected_id].append((obj_id, self.current_iteration))

        # Clean old iterations if max_iterations is set
        if self.max_iterations:
            self._clean_old_iterations()

    def _clean_old_iterations(self) -> None:
        """Remove connections older than max_iterations"""
        cutoff_iteration = self.current_iteration - self.max_iterations

        for obj_id in self.memory:
            # Filter connections to keep only recent ones
            self.memory[obj_id] = deque(
                [(connected_id, iteration) for connected_id, iteration in self.memory[obj_id]
                 if iteration > cutoff_iteration],
                maxlen=self.max_memory_size
            )

        # Clean old edge ages (if tracking enabled)
        if self.track_edge_ages and hasattr(self, 'edge_ages'):
            self.edge_ages = {
                edge_key: age_info
                for edge_key, age_info in self.edge_ages.items()
                if age_info["last_seen"] > cutoff_iteration
            }

    def get_current_memory_graph(self) -> Dict[str, List[str]]:
        """Get current memory as a graph dictionary

        Returns:
            Dictionary with current memory connections
        """
        result = {}

        # Include all objects, even those with no connections
        for obj_id in self.all_objects:
            connections = []
            if obj_id in self.memory:
                # Get unique connections (remove duplicates and self-connections)
                unique_connections = set()
                for connected_id, _ in self.memory[obj_id]:
                    if connected_id != obj_id:
                        unique_connections.add(connected_id)
                connections = list(unique_connections)

            result[obj_id] = connections

        return result

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about current memory state"""
        total_connections = sum(len(connections) for connections in self.memory.values())

        stats = {
            "total_objects": len(self.all_objects),
            "total_connections": total_connections // 2,  # Divide by 2 because connections are bidirectional
            "current_iteration": self.current_iteration,
            "objects_with_memory": len([obj for obj in self.all_objects if obj in self.memory and self.memory[obj]]),
            "max_memory_size": self.max_memory_size,
            "max_iterations": self.max_iterations,
            "edge_aging_enabled": self.track_edge_ages
        }

        # Add edge age statistics if tracking is enabled
        if self.track_edge_ages and hasattr(self, 'edge_ages') and self.edge_ages:
            current_iter = self.current_iteration
            ages = [current_iter - info["first_seen"] for info in self.edge_ages.values()]

            stats["edge_age_stats"] = {
                "min_age": min(ages),
                "max_age": max(ages),
                "avg_age": sum(ages) / len(ages),
                "total_aged_edges": len(ages)
            }

        return stats

    def get_edge_ages(self) -> Dict[Tuple[str, str], Dict[str, int]]:
        """Get age information for all edges (if tracking enabled)"""
        if not self.track_edge_ages or not hasattr(self, 'edge_ages'):
            return {}
        return self.edge_ages.copy()

    def get_edge_age_normalized(self, max_age: int = None) -> Dict[Tuple[str, str], float]:
        """Get normalized edge ages (0.0 = newest, 1.0 = oldest)

        Args:
            max_age: Maximum age to consider (uses current max if None)

        Returns:
            Dictionary mapping edge to normalized age (0.0-1.0)
        """
        if not self.track_edge_ages or not hasattr(self, 'edge_ages') or not self.edge_ages:
            return {}

        if max_age is None:
            ages = [self.current_iteration - info["first_seen"] for info in self.edge_ages.values()]
            max_age = max(ages) if ages else 1

        if max_age == 0:
            return {edge: 0.0 for edge in self.edge_ages.keys()}

        normalized_ages = {}
        for edge_key, age_info in self.edge_ages.items():
            age = self.current_iteration - age_info["first_seen"]
            normalized_age = min(age / max_age, 1.0)
            normalized_ages[edge_key] = normalized_age

        return normalized_ages


def create_memory_graph(current_positions: Union[np.ndarray, Dict[str, Any]],
                        memory_connections: Dict[str, List[str]],
                        aspect: str = "array") -> Any:
    """Create a graph with current positions and memory-based edges

    Args:
        current_positions: Current positions as array [id, x, y, ...] or dict
        memory_connections: Memory connections {"obj_id": ["connected_id1", "connected_id2"]}
        aspect: Data format ("array" or "dict")

    Returns:
        igraph Graph object with memory-based edges

    Raises:
        GraphCreationError: If graph creation fails
    """
    try:
        # Create basic graph with positions
        if aspect == "array":
            if not isinstance(current_positions, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")
            graph = create_graph_array(current_positions)

        elif aspect == "dict":
            if isinstance(current_positions, np.ndarray):
                # Convert array to dict format if needed
                data_interface = DataInterface([("id", int), ("x", int), ("y", int)])
                current_positions = data_interface.convert(current_positions)

            if not isinstance(current_positions, dict):
                raise GraphCreationError("Expected dictionary for 'dict' aspect")
            graph = create_graph_dict(current_positions)
        else:
            raise GraphCreationError("Aspect must be 'array' or 'dict'")

        # Create mapping from normalized object ID to vertex index
        id_to_vertex = {}
        for i, obj_id in enumerate(graph.vs["id"]):
            normalized_id = normalize_id(obj_id)
            id_to_vertex[normalized_id] = i

        # Add memory-based edges
        edges_to_add = []
        for obj_id, connected_ids in memory_connections.items():
            # Normalize the source ID
            obj_id_norm = normalize_id(obj_id)
            if obj_id_norm not in id_to_vertex:
                logging.warning(f"Object {obj_id} (normalized: {obj_id_norm}) in memory but not in current positions")
                continue

            vertex_from = id_to_vertex[obj_id_norm]

            for connected_id in connected_ids:
                # Normalize the target ID
                connected_id_norm = normalize_id(connected_id)
                if connected_id_norm not in id_to_vertex:
                    logging.warning(
                        f"Connected object {connected_id} (normalized: {connected_id_norm}) in memory but not in current positions")
                    continue

                vertex_to = id_to_vertex[connected_id_norm]

                # Avoid self-loops and ensure consistent edge ordering
                if vertex_from != vertex_to:
                    edge = tuple(sorted([vertex_from, vertex_to]))
                    edges_to_add.append(edge)

        # Remove duplicates and add edges
        unique_edges = list(set(edges_to_add))
        if unique_edges:
            graph.add_edges(unique_edges)

            # Add memory attribute to edges
            graph.es["memory_based"] = [True] * len(unique_edges)

        logging.debug(f"Created memory graph with {graph.vcount()} vertices and {graph.ecount()} memory-based edges")

        return graph

    except Exception as e:
        raise GraphCreationError(f"Failed to create memory graph: {str(e)}")


def update_memory_from_proximity(current_positions: Union[np.ndarray, Dict[str, Any]],
                                 proximity_thresh: float,
                                 memory_manager: MemoryManager,
                                 metric: str = "euclidean",
                                 aspect: str = "array") -> Dict[str, List[str]]:
    """Update memory manager with current proximity connections

    Args:
        current_positions: Current positions
        proximity_thresh: Distance threshold for proximity
        memory_manager: MemoryManager instance to update
        metric: Distance metric
        aspect: Data format

    Returns:
        Current proximity connections dictionary
    """
    try:

        # Extract position data and create ID mapping
        if aspect == "array":
            if not isinstance(current_positions, np.ndarray):
                raise GraphCreationError("Expected numpy array for 'array' aspect")

            # Normalize IDs consistently
            object_ids = [normalize_id(obj_id) for obj_id in current_positions[:, 0]]
            positions_2d = current_positions[:, 1:3].astype(float)

        elif aspect == "dict":
            if isinstance(current_positions, np.ndarray):
                data_interface = DataInterface([("id", int), ("x", int), ("y", int)])
                current_positions = data_interface.convert(current_positions)

            # Normalize IDs consistently
            object_ids = [normalize_id(obj_id) for obj_id in current_positions["id"]]
            positions_2d = np.column_stack([current_positions["x"], current_positions["y"]])

        else:
            raise GraphCreationError("Aspect must be 'array' or 'dict'")

        # Get proximity connections
        proximity_indices = get_distance(positions_2d, proximity_thresh, metric)

        # Convert indices to object IDs
        current_connections = {}
        for i, nearby_indices in enumerate(proximity_indices):
            obj_id = object_ids[i]
            connected_ids = [object_ids[j] for j in nearby_indices]
            current_connections[obj_id] = connected_ids

        # Ensure all objects are represented (even those with no connections)
        for obj_id in object_ids:
            if obj_id not in current_connections:
                current_connections[obj_id] = []

        # Update memory manager
        memory_manager.add_connections(current_connections)

        return current_connections

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from proximity: {str(e)}")


def update_memory_from_graph(graph: Any, memory_manager: MemoryManager) -> Dict[str, List[str]]:
    """Update memory manager from any igraph Graph object

    Args:
        graph: Any igraph Graph object (Delaunay, proximity, custom, etc.)
        memory_manager: MemoryManager instance to update

    Returns:
        Current connections dictionary
    """
    try:
        if graph is None:
            raise GraphCreationError("Graph cannot be None")
        if memory_manager is None:
            raise GraphCreationError("Memory manager cannot be None")

        # Extract connections from the graph
        current_connections = {}

        # Initialize all vertices with empty connections
        for vertex in graph.vs:
            obj_id = normalize_id(vertex["id"])  # FIXED: Added normalization
            current_connections[obj_id] = []

        # Add edges as bidirectional connections
        for edge in graph.es:
            vertex1_id = normalize_id(graph.vs[edge.tuple[0]]["id"])  # FIXED: Added normalization
            vertex2_id = normalize_id(graph.vs[edge.tuple[1]]["id"])  # FIXED: Added normalization

            current_connections[vertex1_id].append(vertex2_id)
            current_connections[vertex2_id].append(vertex1_id)

        # Update memory manager
        memory_manager.add_connections(current_connections)

        return current_connections

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from graph: {str(e)}")


def update_memory_from_delaunay(current_positions: Union[np.ndarray, Dict[str, Any]],
                                memory_manager: MemoryManager,
                                aspect: str = "array",
                                dimension: Tuple[int, int] = (1200, 1200)) -> Dict[str, List[str]]:
    """Update memory manager with Delaunay triangulation connections

    Args:
        current_positions: Current positions
        memory_manager: MemoryManager instance to update
        aspect: Data format
        dimension: Canvas dimensions for triangulation

    Returns:
        Current Delaunay connections dictionary
    """
    try:
        # Create temporary graph for Delaunay triangulation
        if aspect == "array":
            temp_graph = create_graph_array(current_positions)
            pos_array = np.stack((
                current_positions[:, 1],  # x positions
                current_positions[:, 2]  # y positions
            ), axis=1)
        elif aspect == "dict":
            if isinstance(current_positions, np.ndarray):
                data_interface = DataInterface([("id", int), ("x", int), ("y", int)])
                current_positions = data_interface.convert(current_positions)
            temp_graph = create_graph_dict(current_positions)
            pos_array = np.column_stack([current_positions["x"], current_positions["y"]])
        else:
            raise GraphCreationError("Aspect must be 'array' or 'dict'")

        # Create Delaunay triangulation
        subdiv = make_subdiv(pos_array, dimension)
        tri_list = subdiv.getTriangleList()
        delaunay_graph = graph_delaunay(temp_graph, subdiv, tri_list)

        # Update memory from the Delaunay graph
        return update_memory_from_graph(delaunay_graph, memory_manager)

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from Delaunay: {str(e)}")


def update_memory_from_custom_function(current_positions: Union[np.ndarray, Dict[str, Any]],
                                       memory_manager: MemoryManager,
                                       connection_function: callable,
                                       aspect: str = "array",
                                       **kwargs) -> Dict[str, List[str]]:
    """Update memory using a custom connection function

    Args:
        current_positions: Current positions
        memory_manager: MemoryManager instance to update
        connection_function: Function that takes positions and returns connections
        aspect: Data format
        **kwargs: Additional arguments for the connection function

    Returns:
        Current connections dictionary
    """
    try:
        # Call the custom function to get connections
        custom_connections = connection_function(current_positions, **kwargs)

        # Handle different return types from custom function
        if hasattr(custom_connections, 'vs'):  # It's an igraph object
            return update_memory_from_graph(custom_connections, memory_manager)

        elif isinstance(custom_connections, (list, tuple)):  # List of (id1, id2) tuples
            # Convert connections list to connection dictionary
            connections_dict = {}

            # Initialize all object IDs
            if aspect == "array":
                all_ids = [normalize_id(obj_id) for obj_id in current_positions[:, 0]]
            else:  # dict aspect
                all_ids = [normalize_id(obj_id) for obj_id in current_positions["id"]]

            for obj_id in all_ids:
                connections_dict[obj_id] = []

            # Add connections from the list
            for connection in custom_connections:
                if len(connection) >= 2:
                    id1 = normalize_id(connection[0])
                    id2 = normalize_id(connection[1])

                    if id1 in connections_dict and id2 in connections_dict:
                        connections_dict[id1].append(id2)
                        connections_dict[id2].append(id1)

            # Update memory manager
            memory_manager.add_connections(connections_dict)
            return connections_dict

        elif isinstance(custom_connections, dict):  # Already a connections dictionary
            memory_manager.add_connections(custom_connections)
            return custom_connections

        else:
            raise GraphCreationError(f"Custom function returned unsupported type: {type(custom_connections)}")

    except Exception as e:
        raise GraphCreationError(f"Failed to update memory from custom function: {str(e)}")


# Example usage function
def example_memory_graph_usage():
    """Example of how to use the memory graph functionality"""

    # Example data - current positions
    current_positions = np.array([
        [1, 100, 100],  # Object A at (100, 100)
        [2, 200, 150],  # Object B at (200, 150)
        [3, 120, 300],  # Object C at (120, 300)
        [4, 400, 100],  # Object D at (400, 100)
    ])

    # Example memory connections (historical proximities)
    memory_connections = {
        "1": ["3", "4"],  # A was connected to C and D
        "2": [],  # B has no memory connections
        "3": ["1"],  # C was connected to A
        "4": ["1"],  # D was connected to A
    }

    # Create memory graph
    graph = create_memory_graph(current_positions, memory_connections, aspect="array")

    print(f"Memory graph: {graph.vcount()} vertices, {graph.ecount()} edges")

    # Using with MemoryManager
    memory_mgr = MemoryManager(max_memory_size=50, max_iterations=10)

    # Simulate multiple iterations
    for iteration in range(5):
        # Simulate changing proximity connections each iteration
        proximity_connections = {
            "1": ["2"] if iteration % 2 == 0 else ["3"],
            "2": ["1"] if iteration % 2 == 0 else [],
            "3": ["1"] if iteration % 2 == 1 else ["4"],
            "4": ["3"] if iteration % 2 == 1 else [],
        }

        memory_mgr.add_connections(proximity_connections)

    # Get final memory state
    final_memory = memory_mgr.get_current_memory_graph()
    final_graph = create_memory_graph(current_positions, final_memory, aspect="array")

    stats = memory_mgr.get_memory_stats()
    print(f"Final memory stats: {stats}")

    return final_graph