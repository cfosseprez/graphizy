Graph Types
===========

Graphizy supports multiple graph construction algorithms, each optimized for different use cases. This section provides comprehensive coverage of all available graph types, their properties, and when to use them.

Overview
--------

.. list-table:: Graph Types Comparison
   :header-rows: 1
   :widths: 20 15 15 25 25

   * - Graph Type
     - Connectivity
     - Edge Count
     - Use Case
     - Memory Compatible
   * - Delaunay
     - Always
     - ~3n
     - Natural triangulation
     - ✅
   * - Proximity  
     - Variable
     - ~distance²
     - Local neighborhoods
     - ✅
   * - K-NN
     - Variable
     - k×n
     - Fixed degree networks
     - ✅
   * - MST
     - Always
     - n-1
     - Minimal connectivity
     - ✅
   * - Gabriel
     - Variable
     - Subset of Delaunay
     - Geometric proximity
     - ✅
   * - Memory
     - Variable
     - Historical
     - Temporal analysis
     - N/A (is the modifier)

Delaunay Triangulation
----------------------

Delaunay triangulation creates an optimal triangular mesh where no point lies inside the circumcircle of any triangle. This produces the "most equilateral" triangulation possible.

**Mathematical Properties:**

- **Connectivity**: Always produces a connected graph
- **Planarity**: Edges never cross
- **Optimality**: Maximizes minimum angle of all triangles
- **Edge Count**: Typically ~3n edges for n vertices

**Algorithm:**

1. Create OpenCV Subdiv2D structure
2. Insert all points into subdivision
3. Extract triangle list
4. Convert triangles to graph edges
5. Map OpenCV indices back to original IDs

.. code-block:: python

   # Create Delaunay triangulation
   delaunay_graph = grapher.make_delaunay(data)
   
   # Properties
   info = grapher.get_graph_info(delaunay_graph)
   print(f"Delaunay: {info['vertex_count']} vertices, {info['edge_count']} edges")
   print(f"Always connected: {info['is_connected']}")  # True
   print(f"Planar embedding: edges never cross")

**Use Cases:**

- **Mesh generation** for finite element analysis
- **Natural neighbor interpolation**  
- **Spatial analysis** where triangle quality matters
- **Geographic information systems** (GIS)
- **Computer graphics** mesh generation

Proximity Graphs
-----------------

Proximity graphs connect points within a specified distance threshold. This creates local neighborhood structures based on spatial proximity.

.. code-block:: python

   # Create proximity graph
   proximity_graph = grapher.make_proximity(
       data, 
       proximity_thresh=50.0,    # Distance threshold
       metric="euclidean"        # Distance metric
   )
   
   # Analyze connectivity
   components = grapher.call_method_raw(proximity_graph, 'connected_components')
   print(f"Proximity graph has {len(components)} connected components")

**Distance Metrics:**

- **Euclidean**: √((x₁-x₂)² + (y₁-y₂)²) - Standard Cartesian distance
- **Manhattan**: |x₁-x₂| + |y₁-y₂| - City block distance  
- **Chebyshev**: max(|x₁-x₂|, |y₁-y₂|) - Chessboard distance

K-Nearest Neighbors (KNN)
--------------------------

K-Nearest Neighbors graphs connect each point to its k closest neighbors, creating a directed graph that can be made undirected by including reverse edges.

.. code-block:: python

   # Create KNN graph (requires scipy)
   knn_graph = grapher.make_knn(data, k=4)
   
   # Analyze degree distribution
   degrees = grapher.call_method(knn_graph, 'degree')
   degree_values = list(degrees.values())
   print(f"Average degree: {np.mean(degree_values):.2f}")

Minimum Spanning Tree (MST)
----------------------------

Minimum Spanning Tree creates the minimal connected graph by selecting the shortest edges that connect all vertices without creating cycles.

.. code-block:: python

   # Create minimum spanning tree
   mst_graph = grapher.make_mst(data, metric="euclidean")
   
   # Verify MST properties  
   info = grapher.get_graph_info(mst_graph)
   n_vertices = info['vertex_count']
   n_edges = info['edge_count']
   
   print(f"Tree property: {n_edges == n_vertices - 1}")  # Should be True
   print(f"Connected: {info['is_connected']}")           # Always True

Gabriel Graph
-------------

Gabriel graph connects two points if no other point lies within the circle having the two points as diameter endpoints. It's a subset of the Delaunay triangulation with interesting geometric properties.

**Mathematical Properties:**

- **Connectivity**: May be disconnected for sparse point sets
- **Subset Relationship**: Always a subset of the Delaunay triangulation
- **Local Property**: Connections based on local geometric criteria
- **Edge Count**: Generally fewer edges than Delaunay triangulation

**Algorithm:**

1. For each pair of points, create a circle with the pair as diameter
2. Check if any other point lies strictly inside this circle
3. If no point is inside, the pair forms a Gabriel edge
4. Add all valid Gabriel edges to the graph

.. code-block:: python

   # Create Gabriel graph
   gabriel_graph = grapher.make_gabriel(data)
   
   # Properties
   info = grapher.get_graph_info(gabriel_graph)
   print(f"Gabriel: {info['vertex_count']} vertices, {info['edge_count']} edges")
   print(f"Subset of Delaunay: edges ≤ Delaunay edges")
   print(f"Connected: {info['is_connected']}")  # May be False

**Use Cases:**

- **Wireless sensor networks** with interference-free communication
- **Geographic analysis** where direct line-of-sight matters
- **Computational geometry** applications requiring local proximity
- **Pattern recognition** in point cloud analysis
- **Network topology** design with geometric constraints

**Comparison with Other Graph Types:**

.. code-block:: python

   # Compare Gabriel with related graph types
   gabriel_graph = grapher.make_gabriel(data)
   delaunay_graph = grapher.make_delaunay(data)
   proximity_graph = grapher.make_proximity(data, 50.0)
   
   gabriel_info = grapher.get_graph_info(gabriel_graph)
   delaunay_info = grapher.get_graph_info(delaunay_graph)
   proximity_info = grapher.get_graph_info(proximity_graph)
   
   print(f"Gabriel edges: {gabriel_info['edge_count']}")
   print(f"Delaunay edges: {delaunay_info['edge_count']}")
   print(f"Proximity edges: {proximity_info['edge_count']}")
   
   # Gabriel is always a subset of Delaunay
   assert gabriel_info['edge_count'] <= delaunay_info['edge_count']

Memory-Enhanced Graphs
----------------------

Memory graphs are not a separate graph type but a **modifier** that can be applied to any base graph type. They track connections over time, creating temporal analysis capabilities.

.. code-block:: python

   # Initialize memory system
   grapher.init_memory_manager(
       max_memory_size=50,      # Max connections per object
       max_iterations=None,     # Keep all history (or set limit)
       track_edge_ages=True     # Enable age-based visualization
   )
   
   # Evolution simulation
   for iteration in range(100):
       # Update positions (simulate movement)
       data[:, 1:3] += np.random.normal(0, 2, (len(data), 2))
       
       # Create current graph (any type)
       current_graph = grapher.make_proximity(data, proximity_thresh=60.0)
       
       # Update memory with current connections
       grapher.update_memory_with_graph(current_graph)
       
       # Create memory-enhanced graph
       memory_graph = grapher.make_memory_graph(data)

Graph Type Selection Guide
---------------------------

Choosing the right graph type depends on your specific requirements:

**For Spatial Analysis:**
   - **Dense regular patterns** → Delaunay Triangulation
   - **Sparse irregular patterns** → Proximity Graphs
   - **Fixed connectivity needs** → K-Nearest Neighbors
   - **Minimal connectivity** → Minimum Spanning Tree

**For Network Properties:**
   - **Always connected** → Delaunay or MST
   - **Local neighborhoods** → Proximity or KNN  
   - **Minimal edges** → MST
   - **Regular degree** → KNN

**For Dynamic Analysis:**
   - **Any of the above + Memory modifier**
   - **Temporal patterns** → Memory-enhanced graphs
   - **Evolution tracking** → Memory with age visualization
