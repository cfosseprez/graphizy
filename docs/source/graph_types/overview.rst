Graph Types Overview
===================

Graphizy supports a comprehensive collection of graph types designed for different use cases in computational geometry, network analysis, and spatial data processing. All graph types use the modern unified `make_graph()` interface and are compatible with memory and weight systems.

Available Graph Types
---------------------

Built-in Graph Types
~~~~~~~~~~~~~~~~~~~~~

**Geometric Graphs**
  - **Delaunay Triangulation**: Creates optimal triangular meshes with excellent geometric properties
  - **Gabriel Graph**: Subset of Delaunay with specific geometric constraints for wireless networks

**Proximity-Based Graphs**  
  - **Proximity Graphs**: Connect points within a specified distance threshold
  - **K-Nearest Neighbors (KNN)**: Connect each point to its k closest neighbors

**Optimization-Based Graphs**
  - **Minimum Spanning Tree (MST)**: Minimal connected graph with shortest total edge length

**Memory-Enhanced Graphs**
  - **Memory Graphs**: Any base graph type enhanced with temporal connection tracking

**Custom Graphs**
  - **Plugin System**: Extensible framework for domain-specific algorithms

Unified Interface
-----------------

All graph types use the same unified interface:

.. code-block:: python

   # Basic usage
   graph = grapher.make_graph("graph_type", data, **parameters)
   
   # With memory and weights (smart defaults)
   graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)
   # Automatically: use_memory=True, update_memory=True, compute_weights=True
   
   # Explicit control
   graph = grapher.make_graph("delaunay", data, 
                            use_memory=False, 
                            update_memory=True,
                            compute_weights=True)

Performance Characteristics
---------------------------

.. list-table:: Performance Comparison
   :header-rows: 1
   :widths: 15 15 15 20 35

   * - Graph Type
     - Time Complexity
     - Space Complexity
     - Typical Edges
     - Best Use Cases
   * - Delaunay
     - O(n log n)
     - O(n)
     - ~3n
     - Mesh generation, spatial analysis
   * - Proximity
     - O(n²)
     - O(n²)
     - Variable
     - Local neighborhoods, clustering
   * - KNN
     - O(n² log n)
     - O(kn)
     - kn
     - Machine learning, recommendations
   * - MST
     - O(n² log n)
     - O(n)
     - n-1
     - Optimization, minimal connectivity
   * - Gabriel
     - O(n³)
     - O(n)
     - ⊆ Delaunay
     - Wireless networks, geometric constraints

Selection Guidelines
--------------------

**For Spatial Analysis:**
  - Dense regular patterns → Delaunay Triangulation
  - Sparse irregular patterns → Proximity Graphs  
  - Fixed connectivity requirements → K-Nearest Neighbors
  - Minimal connectivity needs → Minimum Spanning Tree

**For Network Properties:**
  - Always connected graphs → Delaunay or MST
  - Local neighborhood analysis → Proximity or KNN
  - Minimal edge count → MST
  - Controlled degree distribution → KNN

**For Dynamic Analysis:**
  - Any base type + Memory system
  - Temporal pattern analysis → Memory-enhanced graphs
  - Evolution tracking → Memory with age visualization

Compatibility Matrix
--------------------

.. list-table:: System Compatibility
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Graph Type  
     - Memory
     - Weights
     - Streaming
     - Plugins
     - Special Features
   * - Delaunay
     - ✅
     - ✅  
     - ✅
     - N/A
     - Always connected, planar
   * - Proximity
     - ✅
     - ✅
     - ✅
     - N/A
     - Adjustable density
   * - KNN
     - ✅
     - ✅
     - ✅
     - N/A
     - Fixed out-degree
   * - MST
     - ✅
     - ✅
     - ✅
     - N/A
     - Minimal edges, tree structure
   * - Gabriel
     - ✅
     - ✅
     - ✅
     - N/A
     - Geometric constraints
   * - Custom
     - ✅
     - ✅
     - ✅
     - ✅
     - User-defined algorithms

Next Steps
----------

- :doc:`delaunay` - Detailed Delaunay triangulation documentation
- :doc:`proximity` - Proximity graphs and distance metrics
- :doc:`knn` - K-nearest neighbors implementation
- :doc:`mst` - Minimum spanning tree algorithms
- :doc:`gabriel` - Gabriel graph properties and applications
- :doc:`custom_plugins` - Creating custom graph types
- :doc:`selection_guide` - Detailed selection criteria