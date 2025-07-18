Graph Analysis Methods
======================

Graphizy provides comprehensive graph analysis capabilities through its integration with igraph and custom analysis methods. This section documents all available analysis interfaces and their usage.

Analysis Method Categories
--------------------------

Graphizy offers three levels of analysis access:

1. **High-Level API**: Convenient methods for common analysis tasks
2. **Direct igraph Access**: Full access to 200+ igraph algorithms
3. **Custom Analysis**: User-defined analysis functions

.. note::
   All analysis methods work with any graph type (Delaunay, Proximity, K-NN, MST, Memory-enhanced).

High-Level Analysis API
-----------------------

**Graph Information Summary:**

.. code-block:: python

   # Get comprehensive graph overview
   info = grapher.get_graph_info(graph)
   
   # Returns dictionary with:
   # - vertex_count: Number of nodes
   # - edge_count: Number of edges  
   # - density: Edge density (0-1)
   # - is_connected: Connectivity status
   # - average_path_length: Mean shortest path length
   # - diameter: Maximum shortest path length
   # - transitivity: Global clustering coefficient

**Example Output:**

.. code-block:: python

   {
       'vertex_count': 100,
       'edge_count': 287,
       'density': 0.058,
       'is_connected': True,
       'average_path_length': 3.42,
       'diameter': 8,
       'transitivity': 0.156
   }

**Connections Per Node:**

.. code-block:: python

   # Get degree (number of connections) for each node
   connections = grapher.get_connections_per_object(graph)
   
   # Returns: {node_id: degree_count, ...}
   # Example: {0: 5, 1: 3, 2: 7, ...}
   
   # Usage examples
   most_connected = max(connections.items(), key=lambda x: x[1])
   avg_connections = sum(connections.values()) / len(connections)
   
   print(f"Most connected node: {most_connected[0]} with {most_connected[1]} connections")
   print(f"Average connections per node: {avg_connections:.2f}")

Centrality Analysis Methods
---------------------------

**Available Centrality Measures:**

.. list-table:: Centrality Methods
   :header-rows: 1
   :widths: 25 25 50

   * - Method
     - igraph Function
     - Description
   * - ``degree``
     - ``degree()``
     - Number of direct connections
   * - ``betweenness``
     - ``betweenness()``
     - How often node lies on shortest paths
   * - ``closeness``
     - ``closeness()``
     - Average distance to all other nodes
   * - ``eigenvector``
     - ``eigenvector_centrality()``
     - Influence based on neighbor importance
   * - ``pagerank``
     - ``pagerank()``
     - Google PageRank algorithm
   * - ``authority``
     - ``authority_score()``
     - HITS authority score
   * - ``hub``
     - ``hub_score()``
     - HITS hub score

**Centrality Analysis Examples:**

.. code-block:: python

   # Degree centrality (most connected nodes)
   degree_centrality = grapher.call_method(graph, 'degree')
   
   # Betweenness centrality (bridge nodes)
   betweenness_centrality = grapher.call_method(graph, 'betweenness')
   
   # Closeness centrality (broadcaster nodes)  
   closeness_centrality = grapher.call_method(graph, 'closeness')
   
   # PageRank (influential nodes)
   pagerank_scores = grapher.call_method(graph, 'pagerank')
   
   # Find top nodes by different measures
   def top_nodes(centrality_dict, n=5):
       return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]
   
   print("Top 5 nodes by betweenness centrality:")
   for node, score in top_nodes(betweenness_centrality):
       print(f"  Node {node}: {score:.4f}")

**Centrality Comparison:**

.. code-block:: python

   # Compare different centrality measures
   centralities = {
       'degree': grapher.call_method(graph, 'degree'),
       'betweenness': grapher.call_method(graph, 'betweenness'),
       'closeness': grapher.call_method(graph, 'closeness'),
       'pagerank': grapher.call_method(graph, 'pagerank')
   }
   
   # Create centrality comparison table
   nodes = list(centralities['degree'].keys())
   print(f"{'Node':<6} {'Degree':<8} {'Between':<8} {'Close':<8} {'PageRank':<8}")
   print("-" * 50)
   
   for node in nodes[:10]:  # Show first 10 nodes
       deg = centralities['degree'][node]
       bet = centralities['betweenness'][node]
       clo = centralities['closeness'][node]  
       pgr = centralities['pagerank'][node]
       print(f"{node:<6} {deg:<8} {bet:<8.3f} {clo:<8.3f} {pgr:<8.3f}")

Connectivity Analysis Methods
-----------------------------

**Basic Connectivity:**

.. code-block:: python

   # Check if graph is connected
   is_connected = grapher.call_method(graph, 'is_connected')
   print(f"Graph is connected: {is_connected}")
   
   # Get connected components
   components = grapher.call_method_raw(graph, 'connected_components')
   print(f"Number of components: {len(components)}")
   
   # Analyze component sizes
   component_sizes = [len(comp) for comp in components]
   print(f"Component sizes: {component_sizes}")
   
   if len(components) > 1:
       largest_component = max(component_sizes)
       print(f"Largest component: {largest_component} nodes ({largest_component/graph.vcount():.1%})")

**Path Analysis:**

.. code-block:: python

   # Average path length (for connected graphs)
   if grapher.call_method(graph, 'is_connected'):
       avg_path = grapher.call_method(graph, 'average_path_length')
       diameter = grapher.call_method(graph, 'diameter')
       radius = grapher.call_method(graph, 'radius')
       
       print(f"Average path length: {avg_path:.2f}")
       print(f"Diameter (max path): {diameter}")
       print(f"Radius (min eccentricity): {radius}")

**Shortest Paths:**

.. code-block:: python

   # Shortest paths between specific nodes
   node1, node2 = 0, 10  # Example nodes
   
   # Get shortest path
   path = grapher.call_method_raw(graph, 'get_shortest_paths', node1, node2)
   print(f"Shortest path from {node1} to {node2}: {path[0]}")
   
   # Get all shortest path lengths from one node
   distances = grapher.call_method_raw(graph, 'shortest_paths', [node1])
   print(f"Distances from node {node1}: {distances[0][:10]}...")  # First 10

Clustering Analysis Methods
---------------------------

**Clustering Coefficients:**

.. code-block:: python

   # Global clustering coefficient (transitivity)
   global_clustering = grapher.call_method(graph, 'transitivity_undirected')
   print(f"Global clustering coefficient: {global_clustering:.4f}")
   
   # Local clustering coefficients for each node
   local_clustering = grapher.call_method(graph, 'transitivity_local_undirected')
   
   # Average local clustering
   avg_local_clustering = sum(local_clustering.values()) / len(local_clustering)
   print(f"Average local clustering: {avg_local_clustering:.4f}")
   
   # Find most clustered nodes
   top_clustered = sorted(local_clustering.items(), key=lambda x: x[1], reverse=True)[:5]
   print("Most clustered nodes:")
   for node, clustering in top_clustered:
       print(f"  Node {node}: {clustering:.4f}")

**Assortativity Analysis:**

.. code-block:: python

   # Degree assortativity (do high-degree nodes connect to each other?)
   degree_assortativity = grapher.call_method(graph, 'assortativity_degree')
   print(f"Degree assortativity: {degree_assortativity:.4f}")
   
   # Interpretation:
   # > 0: Similar degree nodes prefer to connect (assortative)
   # < 0: Different degree nodes prefer to connect (disassortative)  
   # ≈ 0: Random mixing

Community Detection Methods
---------------------------

**Available Community Detection Algorithms:**

.. code-block:: python

   # Leiden algorithm (high quality, recommended)
   communities_leiden = grapher.call_method_raw(graph, 'community_leiden')
   
   # Louvain algorithm (fast, good quality)
   communities_louvain = grapher.call_method_raw(graph, 'community_multilevel')
   
   # Walktrap algorithm (random walk based)
   communities_walktrap = grapher.call_method_raw(graph, 'community_walktrap')
   
   # Fast greedy algorithm (agglomerative)
   communities_fastgreedy = grapher.call_method_raw(graph, 'community_fastgreedy')

**Community Analysis:**

.. code-block:: python

   # Analyze community structure
   communities = grapher.call_method_raw(graph, 'community_leiden')
   
   print(f"Number of communities: {len(communities)}")
   print(f"Modularity: {communities.modularity:.4f}")
   
   # Community sizes
   community_sizes = [len(comm) for comm in communities]
   print(f"Community sizes: {sorted(community_sizes, reverse=True)}")
   
   # Largest communities
   largest_communities = sorted(enumerate(communities), key=lambda x: len(x[1]), reverse=True)
   
   print("Largest communities:")
   for i, (comm_id, members) in enumerate(largest_communities[:5]):
       print(f"  Community {comm_id}: {len(members)} members")
       if len(members) <= 10:
           print(f"    Members: {list(members)}")
       else:
           print(f"    Sample members: {list(members)[:10]}...")

Direct igraph Access Methods
----------------------------

**Using call_method() - User-Friendly Interface:**

.. code-block:: python

   # Returns dictionaries mapping node IDs to values (for node-level metrics)
   # Returns single values (for graph-level metrics)
   
   # Node-level metrics (returns dict)
   degrees = grapher.call_method(graph, 'degree')
   # Returns: {node_id: degree_value, ...}
   
   betweenness = grapher.call_method(graph, 'betweenness')
   # Returns: {node_id: betweenness_value, ...}
   
   # Graph-level metrics (returns single value)
   density = grapher.call_method(graph, 'density')
   # Returns: float value
   
   is_connected = grapher.call_method(graph, 'is_connected')
   # Returns: boolean value

**Using call_method_raw() - Direct igraph Interface:**

.. code-block:: python

   # Returns raw igraph output (lists, objects, etc.)
   
   # Get raw degree list
   degree_list = grapher.call_method_raw(graph, 'degree')
   # Returns: [degree_node0, degree_node1, ...]
   
   # Get connected components object
   components = grapher.call_method_raw(graph, 'connected_components')
   # Returns: igraph VertexClustering object
   
   # Get shortest paths matrix
   paths = grapher.call_method_raw(graph, 'shortest_paths')
   # Returns: 2D array of distances

**Common igraph Methods:**

.. list-table:: Frequently Used igraph Methods
   :header-rows: 1
   :widths: 30 20 50

   * - Method Name
     - Return Type
     - Description
   * - ``vcount()``
     - int
     - Number of vertices
   * - ``ecount()``
     - int
     - Number of edges
   * - ``degree()``
     - list
     - Degree of each vertex
   * - ``neighbors(vertex)``
     - list
     - Neighbors of specific vertex
   * - ``shortest_paths()``
     - matrix
     - All shortest path lengths
   * - ``get_shortest_paths()``
     - list
     - Actual shortest paths
   * - ``subgraph(vertices)``
     - graph
     - Extract subgraph
   * - ``induced_subgraph()``
     - graph
     - Create induced subgraph
   * - ``edge_betweenness()``
     - list
     - Betweenness of each edge

Memory-Enhanced Analysis
------------------------

**Memory-Specific Methods:**

.. code-block:: python

   # Memory statistics (only available with memory manager)
   if hasattr(grapher, 'memory_manager') and grapher.memory_manager:
       stats = grapher.get_memory_stats()
       
       print("Memory Analysis:")
       print(f"  Total tracked objects: {stats['total_objects']}")
       print(f"  Total connections in memory: {stats['total_connections']}")
       print(f"  Current iteration: {stats['current_iteration']}")
       
       # Edge age analysis
       if 'edge_age_stats' in stats:
           age_stats = stats['edge_age_stats']
           print(f"  Edge age range: {age_stats['min_age']}-{age_stats['max_age']}")
           print(f"  Average edge age: {age_stats['avg_age']:.1f}")

**Temporal Analysis:**

.. code-block:: python

   # Compare current vs memory graphs
   current_graph = grapher.make_proximity(data, 50.0)
   memory_graph = grapher.make_memory_graph(data)
   
   current_info = grapher.get_graph_info(current_graph)
   memory_info = grapher.get_graph_info(memory_graph)
   
   print("Current vs Memory Comparison:")
   print(f"  Current edges: {current_info['edge_count']}")
   print(f"  Memory edges: {memory_info['edge_count']}")
   print(f"  Memory enhancement: {memory_info['edge_count'] / current_info['edge_count']:.1f}x")

External igraph Resources
-------------------------

For more advanced analysis capabilities, refer to the comprehensive igraph documentation:

**Official igraph Documentation:**
   - **Python Tutorial**: https://igraph.org/python/tutorial/latest/
   - **API Reference**: https://igraph.org/python/api/latest/
   - **Graph Analysis Guide**: https://igraph.org/python/tutorial/latest/analysis.html

**Additional igraph Methods:**
   - **200+ algorithms** available through ``call_method_raw()``
   - **Graph generators** for testing and comparison
   - **Import/export** functions for various formats
   - **Statistical analysis** and random graph models

**Integration Examples:**

.. code-block:: python

   # Use any igraph method not explicitly wrapped
   def use_advanced_igraph_features(graph, grapher):
       
       # Graph isomorphism
       graph2 = grapher.make_delaunay(other_data)
       are_isomorphic = grapher.call_method_raw(graph, 'isomorphic', graph2)
       
       # Random walks
       walk = grapher.call_method_raw(graph, 'random_walk', start=0, steps=100)
       
       # Network flow
       max_flow = grapher.call_method_raw(graph, 'maxflow', source=0, target=10)
       
       return {
           'isomorphic': are_isomorphic,
           'random_walk': walk,
           'max_flow': max_flow
       }

This comprehensive analysis interface makes Graphizy a powerful platform for network analysis, combining ease of use with the full power of igraph's extensive algorithm library.
