Plugin System for Graph Types
=============================

The Graphizy plugin system makes it incredibly easy to add new graph types without modifying any core files. This guide shows developers how to extend Graphizy with custom graph algorithms.

## Why Use the Plugin System?

**Before (Traditional Approach):**
❌ Modify multiple core files  
❌ Edit `algorithms.py`, `main.py`, `__init__.py`  
❌ Manually update imports and exports  
❌ Risk breaking existing functionality  
❌ Difficult to maintain third-party extensions  
❌ No standardized interface  

**After (Plugin System):**
✅ **Zero core file modifications**  
✅ **Single file per graph type**  
✅ **Automatic registration and discovery**  
✅ **Standardized parameter validation**  
✅ **Built-in documentation system**  
✅ **Easy distribution as separate packages**  

## Quick Start: Adding a New Graph Type

### Method 1: Simple Decorator Approach

For simple graph algorithms, use the `@graph_type_plugin` decorator:

.. code-block:: python

   from graphizy import graph_type_plugin
   import numpy as np
   
   @graph_type_plugin(
       name="random_graph",
       description="Randomly connects points with specified probability",
       parameters={
           "probability": {
               "type": float,
               "default": 0.1,
               "description": "Connection probability (0-1)"
           }
       },
       category="experimental"
   )
   def create_random_graph(data_points, aspect, dimension, probability=0.1):
       """Create a graph with random connections"""
       from graphizy.algorithms import create_graph_array, create_graph_dict
       import random
       
       # Create base graph
       if aspect == "array":
           graph = create_graph_array(data_points)
           num_points = len(data_points)
       else:
           graph = create_graph_dict(data_points)
           num_points = len(data_points["id"])
       
       # Add random connections
       for i in range(num_points):
           for j in range(i + 1, num_points):
               if random.random() < probability:
                   graph.add_edge(i, j)
       
       return graph

**That's it!** Your new graph type is automatically:
- ✅ Registered in the system
- ✅ Available via `grapher.make_graph('random_graph', data)`
- ✅ Listed in `grapher.list_graph_types()`
- ✅ Documented with parameter info

### Method 2: Class-Based Approach

For more complex algorithms with custom validation:

.. code-block:: python

   from graphizy import GraphTypePlugin, GraphTypeInfo, register_graph_type
   import numpy as np
   
   class ClusterGraphPlugin(GraphTypePlugin):
       """Creates connections within predefined clusters"""
       
       @property
       def info(self):
           return GraphTypeInfo(
               name="cluster",
               description="Connects points within spatial clusters",
               parameters={
                   "num_clusters": {
                       "type": int,
                       "default": 3,
                       "description": "Number of clusters to create"
                   },
                   "cluster_radius": {
                       "type": float,
                       "default": 100.0,
                       "description": "Maximum distance within clusters"
                   }
               },
               category="clustering",
               author="Your Name",
               version="1.0.0",
               requires_external_deps=True,
               external_deps=["scikit-learn"]
           )
       
       def validate_parameters(self, **kwargs):
           """Custom parameter validation"""
           processed = super().validate_parameters(**kwargs)
           
           if processed.get("num_clusters", 1) < 1:
               raise ValueError("num_clusters must be >= 1")
           
           if processed.get("cluster_radius", 0) <= 0:
               raise ValueError("cluster_radius must be positive")
           
           return processed
       
       def create_graph(self, data_points, aspect, dimension, **kwargs):
           """Create cluster-based graph"""
           try:
               from sklearn.cluster import KMeans
               from graphizy.algorithms import create_graph_array, create_graph_dict
               
               num_clusters = kwargs.get("num_clusters", 3)
               cluster_radius = kwargs.get("cluster_radius", 100.0)
               
               # Create base graph
               if aspect == "array":
                   graph = create_graph_array(data_points)
                   positions = data_points[:, 1:3]
               else:
                   graph = create_graph_dict(data_points)
                   positions = np.column_stack([data_points["x"], data_points["y"]])
               
               # Perform clustering
               kmeans = KMeans(n_clusters=num_clusters, random_state=42)
               clusters = kmeans.fit_predict(positions)
               
               # Connect points within same cluster
               for i in range(len(positions)):
                   for j in range(i + 1, len(positions)):
                       if clusters[i] == clusters[j]:  # Same cluster
                           distance = np.linalg.norm(positions[i] - positions[j])
                           if distance <= cluster_radius:
                               graph.add_edge(i, j)
               
               return graph
               
           except ImportError:
               raise ImportError("Cluster graph requires scikit-learn: pip install scikit-learn")
   
   # Register the plugin
   register_graph_type(ClusterGraphPlugin())

## Using Your New Graph Types

Once registered, your graph types work exactly like built-in ones:

.. code-block:: python

   from graphizy import Graphing, generate_positions
   import numpy as np
   
   # Generate test data
   positions = generate_positions(400, 400, 50)
   data = np.column_stack((np.arange(len(positions)), positions))
   
   # Create grapher
   grapher = Graphing(dimension=(400, 400))
   
   # Use your custom graph types
   random_graph = grapher.make_graph('random_graph', data, probability=0.15)
   cluster_graph = grapher.make_graph('cluster', data, num_clusters=4)
   
   # Mix with built-in types
   delaunay_graph = grapher.make_graph('delaunay', data)
   proximity_graph = grapher.make_graph('proximity', data, proximity_thresh=60.0)

## Discovery and Documentation

### List Available Graph Types

.. code-block:: python

   # List all graph types
   all_types = grapher.list_graph_types()
   for name, info in all_types.items():
       print(f"{name}: {info.description}")
   
   # List by category
   experimental = grapher.list_graph_types(category="experimental")
   built_in = grapher.list_graph_types(category="built-in")

### Get Detailed Information

.. code-block:: python

   # Get detailed info about a graph type
   info = grapher.get_graph_info('cluster')
   
   print(f"Description: {info['info']['description']}")
   print(f"Author: {info['info']['author']}")
   print(f"Version: {info['info']['version']}")
   
   print("Parameters:")
   for param, details in info['parameters'].items():
       print(f"  {param}: {details['description']}")
       print(f"    Default: {details.get('default', 'None')}")
       print(f"    Type: {details.get('type', 'Any')}")

## Best Practices

### 1. Clear Naming

.. code-block:: python

   # ✅ Good
   name="k_nearest_neighbors"
   name="minimum_spanning_tree"
   name="community_detection"
   
   # ❌ Avoid
   name="knn"  # Too cryptic
   name="my_algorithm"  # Too generic

### 2. Comprehensive Documentation

.. code-block:: python

   description="Connects each point to its k nearest neighbors using spatial distance"
   
   parameters={
       "k": {
           "description": "Number of nearest neighbors to connect to each point",
           "type": int,
           "default": 3
       }
   }

### 3. Handle Both Data Formats

.. code-block:: python

   def create_graph(self, data_points, aspect, dimension, **kwargs):
       # Handle both array and dict formats
       if aspect == "array":
           graph = create_graph_array(data_points)
           positions = data_points[:, 1:3]
       else:
           graph = create_graph_dict(data_points)
           positions = np.column_stack([data_points["x"], data_points["y"]])
       
       # Common algorithm logic...

## Real-World Example

See `examples/add_new_graph_type.py` for a complete working example that demonstrates:

- **Random Graph** - Simple decorator-based plugin
- **Star Graph** - Class-based plugin with validation
- Automatic discovery and documentation
- Integration with existing Graphizy workflow

## Summary

The Graphizy plugin system transforms graph type development from:

❌ **Complex**: Modify multiple files, manual integration  
✅ **Simple**: Single file, automatic integration  

❌ **Fragile**: Risk breaking existing code  
✅ **Safe**: Zero impact on core functionality  

❌ **Undocumented**: Manual documentation updates  
✅ **Self-Documenting**: Built-in parameter and usage docs  

With just a few lines of code, developers can now add sophisticated graph algorithms that integrate seamlessly with the entire Graphizy ecosystem! 🚀
