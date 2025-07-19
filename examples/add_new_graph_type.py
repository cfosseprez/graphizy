"""
Simple example: Adding a new graph type to Graphizy

This demonstrates how a developer can easily add a new graph type 
without modifying any core Graphizy files.
"""

import numpy as np
from graphizy import (
    Graphing, generate_positions, graph_type_plugin, 
    register_graph_type, GraphTypePlugin, GraphTypeInfo
)


# Method 1: Using the simple decorator approach
@graph_type_plugin(
    name="random_edges",
    description="Randomly connects points with a specified probability",
    parameters={
        "connection_probability": {
            "type": float,
            "default": 0.1,
            "description": "Probability of connecting any two points (0-1)"
        },
        "min_connections": {
            "type": int,
            "default": 1,
            "description": "Minimum connections per point"
        }
    },
    category="community",
    author="Random Graph Developer",
    version="1.0.0"
)
def create_random_graph(data_points, aspect, dimension, 
                       connection_probability=0.1, min_connections=1):
    """Create a graph with random connections"""
    from graphizy.algorithms import create_graph_array, create_graph_dict
    import random
    
    # Create initial graph
    if aspect == "array":
        graph = create_graph_array(data_points)
        num_points = len(data_points)
    else:
        graph = create_graph_dict(data_points)
        num_points = len(data_points["id"])
    
    # Add random connections
    for i in range(num_points):
        connections_made = 0
        
        for j in range(i + 1, num_points):
            if random.random() < connection_probability:
                graph.add_edge(i, j)
                connections_made += 1
        
        # Ensure minimum connections
        if connections_made < min_connections:
            available_targets = [j for j in range(num_points) 
                               if j != i and not graph.are_connected(i, j)]
            
            while connections_made < min_connections and available_targets:
                target = random.choice(available_targets)
                graph.add_edge(i, target)
                available_targets.remove(target)
                connections_made += 1
    
    return graph


# Method 2: Using the class-based approach for more complex plugins
class StarGraphPlugin(GraphTypePlugin):
    """Creates a star graph with one central hub connected to all other points"""
    
    @property
    def info(self):
        return GraphTypeInfo(
            name="star",
            description="Creates a star topology with one central hub",
            parameters={
                "hub_index": {
                    "type": int,
                    "default": None,
                    "description": "Index of hub point (auto-select if None)"
                },
                "hub_selection": {
                    "type": str,
                    "default": "center",
                    "description": "How to select hub: 'center', 'random', 'first'"
                }
            },
            category="topology",
            author="Star Graph Expert",
            version="2.0.0"
        )
    
    def validate_parameters(self, **kwargs):
        """Validate parameters"""
        processed = super().validate_parameters(**kwargs)
        
        hub_selection = processed.get("hub_selection", "center")
        if hub_selection not in ["center", "random", "first"]:
            raise ValueError("hub_selection must be 'center', 'random', or 'first'")
        
        return processed
    
    def create_graph(self, data_points, aspect, dimension, **kwargs):
        """Create star graph"""
        from graphizy.algorithms import create_graph_array, create_graph_dict
        import numpy as np
        
        hub_index = kwargs.get("hub_index")
        hub_selection = kwargs.get("hub_selection", "center")
        
        # Create initial graph
        if aspect == "array":
            graph = create_graph_array(data_points)
            positions = data_points[:, 1:3]
            num_points = len(data_points)
        else:
            graph = create_graph_dict(data_points)
            positions = np.column_stack([data_points["x"], data_points["y"]])
            num_points = len(data_points["id"])
        
        # Select hub
        if hub_index is None:
            if hub_selection == "center":
                # Find point closest to center
                center = np.mean(positions, axis=0)
                distances = np.linalg.norm(positions - center, axis=1)
                hub_index = np.argmin(distances)
            elif hub_selection == "random":
                hub_index = np.random.randint(0, num_points)
            else:  # "first"
                hub_index = 0
        
        # Connect hub to all other points
        for i in range(num_points):
            if i != hub_index:
                graph.add_edge(hub_index, i)
        
        return graph


def main():
    """Demonstrate the new graph types"""
    print("🚀 Demonstrating easy graph type extension in Graphizy")
    print("=" * 60)
    
    # Register the class-based plugin
    register_graph_type(StarGraphPlugin())
    
    # Generate test data
    positions = generate_positions(300, 300, 20)
    data = np.column_stack((np.arange(len(positions)), positions))
    
    # Create grapher
    grapher = Graphing(dimension=(300, 300))
    
    # Show all available graph types
    print("📋 Available graph types:")
    all_types = grapher.list_graph_types()
    
    for name, info in all_types.items():
        print(f"  • {name}: {info.description}")
        if info.category not in ["built-in"]:
            print(f"    └─ Category: {info.category}, Author: {info.author}")
    
    print("\n" + "=" * 60)
    print("🎯 Creating graphs with new types:")
    
    # Test the new graph types
    try:
        # Random graph
        random_graph = grapher.make_graph('random_edges', data, 
                                        connection_probability=0.15,
                                        min_connections=2)
        print(f"✅ Random Graph: {random_graph.vcount()} vertices, {random_graph.ecount()} edges")
        
        # Star graph
        star_graph = grapher.make_graph('star', data, hub_selection='center')
        print(f"✅ Star Graph: {star_graph.vcount()} vertices, {star_graph.ecount()} edges")
        
        # Compare with built-in types
        delaunay_graph = grapher.make_graph('delaunay', data)
        print(f"✅ Delaunay Graph: {delaunay_graph.vcount()} vertices, {delaunay_graph.ecount()} edges")
        
        proximity_graph = grapher.make_graph('proximity', data, proximity_thresh=50.0)
        print(f"✅ Proximity Graph: {proximity_graph.vcount()} vertices, {proximity_graph.ecount()} edges")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("📖 Get detailed info about our new graph types:")
    
    # Get detailed information
    try:
        star_info = grapher.get_graph_info('star')
        print(f"\n🌟 Star Graph Details:")
        print(f"   Description: {star_info['info']['description']}")
        print(f"   Author: {star_info['info']['author']}")
        print(f"   Version: {star_info['info']['version']}")
        print("   Parameters:")
        for param, details in star_info['parameters'].items():
            print(f"     • {param}: {details['description']}")
            print(f"       Default: {details.get('default', 'None')}")
    
    except Exception as e:
        print(f"❌ Could not get star info: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Success! New graph types added with minimal code!")
    print("\nKey Benefits:")
    print("✅ No core files modified")
    print("✅ Automatic parameter validation")
    print("✅ Discoverable through list_graph_types()")
    print("✅ Integrated documentation")
    print("✅ Same API as built-in types")


if __name__ == "__main__":
    main()
