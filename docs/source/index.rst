Graphizy Documentation
======================

.. image:: https://raw.githubusercontent.com/lesptizami/graphizy/main/images/detection_to_graph.png
   :alt: Detection to Graph
   :align: center

*A powerful real-time graph maker for computational geometry and network visualization*

Graphizy specializes in creating multiple graph types from point data, with advanced memory-enhanced analysis for temporal patterns. Built on OpenCV and igraph, it provides real-time graph construction and comprehensive analytics.

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from graphizy import Graphing, GraphizyConfig, generate_positions

   # Generate random points positions
   positions = generate_positions(800, 800, 100)
   # The postions can be in different format
   # Here it is and array with column id, x, y
   data = np.column_stack((np.arange(len(positions)), positions))

   # Create the grapher config
   # Note: The grapher require the image dimensions
   config = GraphizyConfig()
   config.graph.dimension = (800, 800)
   # Create the grapher
   grapher = Graphing(config=config)

   # Generate different graph types from the data
   delaunay_graph = grapher.make_delaunay(data)
   proximity_graph = grapher.make_proximity(data, proximity_thresh=50.0)
   mst_graph = grapher.make_mst(data)  # Minimum spanning tree
   gabriel_graph = grapher.make_gabriel(data)  # Gabriel graph

   # Visualize
   image = grapher.draw_graph(delaunay_graph)
   grapher.show_graph(image, "Delaunay Triangulation")
   
   # Graph-level metrics calculation
   graph_info = grapher.get_graph_info(delaunay_graph)
   print(f"Graph metrics:")
   print(f"  Vertices: {graph_info['vertex_count']}")
   print(f"  Edges: {graph_info['edge_count']}")
   print(f"  Density: {graph_info['density']:.3f}")
   print(f"  Connected: {graph_info['is_connected']}")
   print(f"  Average path length: {graph_info['average_path_length']:.2f}")
   print(f"  Clustering coefficient: {graph_info['transitivity']:.3f}")
   
   # Individual-level metrics calculation
   # Betweenness centrality (measures how often a node acts as a bridge)
   betweenness = grapher.call_method(delaunay_graph, 'betweenness')
   
   # Find the top 5 most central nodes
   top_central = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
   print(f"\nTop 5 most central nodes (betweenness):")
   for node_id, centrality in top_central:
       print(f"  Node {node_id}: {centrality:.3f}")
   
   # Degree centrality (number of connections per node)
   degrees = grapher.call_method(delaunay_graph, 'degree')
   avg_degree = sum(degrees.values()) / len(degrees)
   max_degree_node = max(degrees.items(), key=lambda x: x[1])
   print(f"\nDegree statistics:")
   print(f"  Average degree: {avg_degree:.2f}")
   print(f"  Most connected node: {max_degree_node[0]} with {max_degree_node[1]} connections")

Key Features
------------

🎯 **Graph Types**
   - **Delaunay Triangulation**: Optimal triangular meshes
   - **Proximity Graphs**: Distance-based connections  
   - **K-Nearest Neighbors**: Fixed-degree networks
   - **Minimum Spanning Tree**: Minimal connectivity
   - **Gabriel Graph**: Geometric proximity (subset of Delaunay)

🧠 **Temporal memory**
   - **Memory-Enhanced**: Any graph type can have a memory in number of past interaction (edges) or in number of past frames

🧮 **Analysis**
   - Full igraph integration with 200+ graph analysis algorithms
   - Real-time statistics and centrality measures
   - Memory system for temporal pattern analysis

🎨 **Visualization**
   - Interactive OpenCV display
   - Age-based edge coloring for memory graphs
   - Configurable styling and output formats

🔧 **Architecture**
   - Type-safe dataclass configuration
   - Robust error handling with detailed exceptions
   - Performance monitoring and optimization

User Guide
==========

.. toctree::
   :maxdepth: 2

   user_guide/installation
   user_guide/graph_types
   user_guide/graph_analysis
   user_guide/memory_system
   user_guide/configuration
   user_guide/examples

API Reference
=============

.. toctree::
   :maxdepth: 2

   modules

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
