"""
Plugin system examples and documentation

This demonstrates how to easily add new graph types to Graphizy using the plugin system.
"""

import numpy as np
from typing import Union, Dict, Any

# Import the plugin system
from graphizy.plugins import GraphTypePlugin, GraphTypeInfo, register_graph_type, graph_type_plugin


# Example 1: Creating a plugin using the class-based approach
class KNearestNeighborsPlugin(GraphTypePlugin):
    """K-Nearest Neighbors graph plugin"""
    
    @property
    def info(self):
        return GraphTypeInfo(
            name="knn",
            description="Connects each point to its k nearest neighbors",
            parameters={
                "k": {
                    "type": int,
                    "default": 3,
                    "description": "Number of nearest neighbors to connect"
                },
                "metric": {
                    "type": str,
                    "default": "euclidean",
                    "description": "Distance metric to use"
                }
            },
            category="community",
            author="Example Developer",
            version="1.0.0",
            requires_external_deps=True,
            external_deps=["scipy"]
        )
    
    def create_graph(self, data_points: Union[np.ndarray, Dict[str, Any]], 
                     aspect: str, dimension: tuple, **kwargs):
        """Create k-nearest neighbors graph"""
        try:
            from scipy.spatial.distance import cdist
            from graphizy.algorithms import create_graph_array, create_graph_dict
            
            k = kwargs.get("k", 3)
            metric = kwargs.get("metric", "euclidean")
            
            # Create initial graph
            if aspect == "array":
                graph = create_graph_array(data_points)
                pos_array = data_points[:, 1:3]  # x, y coordinates
            else:
                graph = create_graph_dict(data_points)
                pos_array = np.column_stack([data_points["x"], data_points["y"]])
            
            # Calculate distances
            distances = cdist(pos_array, pos_array, metric=metric)
            
            # Add edges to k nearest neighbors
            for i in range(len(distances)):
                # Get indices of k+1 nearest neighbors (excluding self)
                nearest_indices = np.argsort(distances[i])[1:k+1]
                
                for j in nearest_indices:
                    if not graph.are_connected(i, j):
                        graph.add_edge(i, j)
            
            return graph
            
        except ImportError:
            raise ImportError("K-NN graph requires scipy. Install with: pip install scipy")
        except Exception as e:
            raise Exception(f"Failed to create KNN graph: {str(e)}")


# Example 2: Creating a plugin using the decorator approach
@graph_type_plugin(
    name="grid",
    description="Connects points in a grid pattern based on distance threshold",
    parameters={
        "grid_size": {
            "type": float,
            "default": 100.0,
            "description": "Size of grid cells"
        },
        "connect_diagonal": {
            "type": bool,
            "default": False,
            "description": "Whether to connect diagonal neighbors"
        }
    },
    category="experimental",
    author="Grid Expert",
    version="0.1.0"
)
def create_grid_graph(data_points, aspect, dimension, grid_size=100.0, connect_diagonal=False):
    """Create a grid-based graph"""
    from graphizy.algorithms import create_graph_array, create_graph_dict
    import numpy as np
    
    # Create initial graph
    if aspect == "array":
        graph = create_graph_array(data_points)
        positions = data_points[:, 1:3]  # x, y coordinates
    else:
        graph = create_graph_dict(data_points)
        positions = np.column_stack([data_points["x"], data_points["y"]])
    
    # Create grid connections
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pos1, pos2 = positions[i], positions[j]
            
            # Calculate grid distance
            grid_dist_x = abs(pos1[0] - pos2[0])
            grid_dist_y = abs(pos1[1] - pos2[1])
            
            # Connect if within grid size
            if grid_dist_x <= grid_size and grid_dist_y <= grid_size:
                # Check if it's a valid connection
                if connect_diagonal or grid_dist_x == 0 or grid_dist_y == 0:
                    graph.add_edge(i, j)
    
    return graph


# Example 3: Advanced plugin with custom validation
class RadialGraphPlugin(GraphTypePlugin):
    """Radial graph that connects points in concentric circles"""
    
    @property
    def info(self):
        return GraphTypeInfo(
            name="radial",
            description="Connects points based on radial distance from center",
            parameters={
                "center_x": {
                    "type": float,
                    "default": None,
                    "description": "X coordinate of center (auto-calculated if None)"
                },
                "center_y": {
                    "type": float,
                    "default": None,
                    "description": "Y coordinate of center (auto-calculated if None)"
                },
                "radius_threshold": {
                    "type": float,
                    "default": 150.0,
                    "description": "Maximum radius for connections"
                },
                "ring_connections": {
                    "type": bool,
                    "default": True,
                    "description": "Connect points at similar radial distances"
                }
            },
            category="experimental",
            author="Radial Graph Expert",
            version="1.0.0"
        )
    
    def validate_parameters(self, **kwargs):
        """Custom parameter validation"""
        processed = super().validate_parameters(**kwargs)
        
        # Validate radius threshold
        if processed.get("radius_threshold", 0) <= 0:
            raise ValueError("radius_threshold must be positive")
        
        # Validate center coordinates if provided
        center_x = processed.get("center_x")
        center_y = processed.get("center_y")
        
        if (center_x is None) != (center_y is None):
            raise ValueError("Both center_x and center_y must be provided or both None")
        
        return processed
    
    def create_graph(self, data_points, aspect, dimension, **kwargs):
        """Create radial graph"""
        from graphizy.algorithms import create_graph_array, create_graph_dict
        import numpy as np
        
        center_x = kwargs.get("center_x")
        center_y = kwargs.get("center_y")
        radius_threshold = kwargs.get("radius_threshold", 150.0)
        ring_connections = kwargs.get("ring_connections", True)
        
        # Create initial graph
        if aspect == "array":
            graph = create_graph_array(data_points)
            positions = data_points[:, 1:3]
        else:
            graph = create_graph_dict(data_points)
            positions = np.column_stack([data_points["x"], data_points["y"]])
        
        # Calculate center if not provided
        if center_x is None:
            center_x = np.mean(positions[:, 0])
            center_y = np.mean(positions[:, 1])
        
        center = np.array([center_x, center_y])
        
        # Calculate radial distances
        radial_distances = np.linalg.norm(positions - center, axis=1)
        
        # Create connections
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist1, dist2 = radial_distances[i], radial_distances[j]
                
                # Connect if both within radius threshold
                if dist1 <= radius_threshold and dist2 <= radius_threshold:
                    # Ring connections: connect points at similar distances
                    if ring_connections and abs(dist1 - dist2) <= 30.0:
                        graph.add_edge(i, j)
                    # Radial connections: connect points along radii
                    elif not ring_connections:
                        # Check if points are roughly on the same radial line
                        angle1 = np.arctan2(positions[i][1] - center_y, positions[i][0] - center_x)
                        angle2 = np.arctan2(positions[j][1] - center_y, positions[j][0] - center_x)
                        angle_diff = abs(angle1 - angle2)
                        
                        if angle_diff <= 0.2 or angle_diff >= (2 * np.pi - 0.2):  # Similar angles
                            graph.add_edge(i, j)
        
        return graph


# Register the plugins
def register_example_plugins():
    """Register all example plugins"""
    register_graph_type(KNearestNeighborsPlugin())
    register_graph_type(RadialGraphPlugin())
    # The grid plugin is auto-registered by the decorator


if __name__ == "__main__":
    # Example usage
    from graphizy import Graphing, generate_positions
    from graphizy.plugins import get_graph_registry
    
    # Register our example plugins
    register_example_plugins()
    
    # Generate test data
    positions = generate_positions(400, 400, 50)
    data = np.column_stack((np.arange(len(positions)), positions))
    
    # Create grapher
    grapher = Graphing(dimension=(400, 400))
    
    # List all available graph types
    print("Available graph types:")
    registry = get_graph_registry()
    all_types = registry.list_plugins()
    
    for name, info in all_types.items():
        print(f"  {name}: {info.description} (category: {info.category})")
    
    # Create graphs using the new plugin system
    print("\nCreating graphs using plugin system:")
    
    # Built-in graph types
    delaunay_graph = grapher.make_graph('delaunay', data)
    print(f"✅ Delaunay: {delaunay_graph.vcount()} vertices, {delaunay_graph.ecount()} edges")
    
    proximity_graph = grapher.make_graph('proximity', data, proximity_thresh=60.0)
    print(f"✅ Proximity: {proximity_graph.vcount()} vertices, {proximity_graph.ecount()} edges")
    
    # Custom graph types
    try:
        knn_graph = grapher.make_graph('knn', data, k=4)
        print(f"✅ KNN: {knn_graph.vcount()} vertices, {knn_graph.ecount()} edges")
    except ImportError as e:
        print(f"⚠️  KNN: {e}")
    
    grid_graph = grapher.make_graph('grid', data, grid_size=80.0, connect_diagonal=True)
    print(f"✅ Grid: {grid_graph.vcount()} vertices, {grid_graph.ecount()} edges")
    
    radial_graph = grapher.make_graph('radial', data, radius_threshold=120.0)
    print(f"✅ Radial: {radial_graph.vcount()} vertices, {radial_graph.ecount()} edges")
    
    # Get information about a specific graph type
    print("\nKNN Graph Information:")
    try:
        knn_info = grapher.get_graph_info('knn')
        print(f"Description: {knn_info['info']['description']}")
        print("Parameters:")
        for param_name, param_info in knn_info['parameters'].items():
            print(f"  {param_name}: {param_info['description']} (default: {param_info.get('default', 'None')})")
    except Exception as e:
        print(f"Could not get KNN info: {e}")
