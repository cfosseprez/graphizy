Basic Usage
===========

This guide covers the fundamental operations in Graphizy, from creating your first graph to performing basic analysis. By the end, you'll understand all the core features and be ready for advanced applications.

The Graphing Class
-----------------

The `Graphing` class is your main interface to Graphizy. It creates graphs, performs analysis, and handles visualization.

**Basic Initialization:**

.. code-block:: python

   from graphizy import Graphing, GraphizyConfig
   
   # Simple initialization
   grapher = Graphing(dimension=(800, 600))
   
   # With custom configuration
   config = GraphizyConfig(dimension=(800, 600))
   grapher = Graphing(config=config)

**Configuration Options:**

.. code-block:: python

   from graphizy import GraphizyConfig
   
   # Create detailed configuration
   config = GraphizyConfig(
       dimension=(800, 600),
       drawing={
           "point_color": (255, 100, 100),  # Red points
           "line_color": (100, 100, 255),   # Blue lines
           "point_radius": 6,
           "line_thickness": 2
       }
   )
   
   grapher = Graphing(config=config)

Creating Graphs
--------------

Graphizy uses a unified `make_graph()` interface for all graph types:

**Unified Interface:**

.. code-block:: python

   from graphizy import generate_and_format_positions
   
   # Generate sample data
   data = generate_and_format_positions(800, 600, 100)
   
   # Create different graph types
   proximity_graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)
   delaunay_graph = grapher.make_graph("delaunay", data)
   knn_graph = grapher.make_graph("knn", data, k=4)
   mst_graph = grapher.make_graph("mst", data)
   gabriel_graph = grapher.make_graph("gabriel", data)

**Graph Type Details:**

*Proximity Graph*
   Connects points within a specified distance:

   .. code-block:: python

      # Connect points within 60 units
      graph = grapher.make_graph("proximity", data, proximity_thresh=60.0)
      
      # Different distance metrics
      graph = grapher.make_graph("proximity", data, 
                                proximity_thresh=60.0, 
                                metric="euclidean")  # default
      graph = grapher.make_graph("proximity", data,
                                proximity_thresh=60.0,
                                metric="manhattan")

*Delaunay Triangulation*
   Creates optimal triangular mesh:

   .. code-block:: python

      # No additional parameters needed
      delaunay_graph = grapher.make_graph("delaunay", data)

*K-Nearest Neighbors*
   Connects each point to its k closest neighbors:

   .. code-block:: python

      # Each point connects to 4 nearest neighbors
      knn_graph = grapher.make_graph("knn", data, k=4)
      
      # Different k values change connectivity
      sparse_graph = grapher.make_graph("knn", data, k=2)    # Sparse
      dense_graph = grapher.make_graph("knn", data, k=8)     # Dense

*Minimum Spanning Tree*
   Creates minimal connected graph:

   .. code-block:: python

      # Minimal edges to connect all points
      mst_graph = grapher.make_graph("mst", data)

*Gabriel Graph*
   Geometric proximity graph (subset of Delaunay):

   .. code-block:: python

      # Geometric connectivity rules
      gabriel_graph = grapher.make_graph("gabriel", data)

Data Formats
-----------

Graphizy expects data in a simple 3-column format:

**Standard Format:**

.. code-block:: python

   import numpy as np
   
   # Manual data creation
   data = np.array([
       [0, 100, 150],  # Point ID=0 at (100, 150)
       [1, 200, 200],  # Point ID=1 at (200, 200)  
       [2, 150, 100],  # Point ID=2 at (150, 100)
       [3, 250, 180],  # Point ID=3 at (250, 180)
   ])

**Data Generation Helpers:**

.. code-block:: python

   from graphizy import generate_and_format_positions, generate_positions
   
   # Generate random points (recommended)
   data = generate_and_format_positions(
       size_x=800,        # Canvas width
       size_y=600,        # Canvas height  
       num_particles=50   # Number of points
   )
   
   # Just positions (you add IDs)
   positions = generate_positions(800, 600, 50)
   data = np.column_stack((np.arange(50), positions))

**Data Validation:**

.. code-block:: python

   from graphizy import validate_graphizy_input
   
   # Check your data format
   is_valid = validate_graphizy_input(data, verbose=True)
   
   if not is_valid:
       print("Data format issues detected!")

Graph Analysis
-------------

The new `GraphAnalysisResult` provides comprehensive analysis:

**Basic Analysis:**

.. code-block:: python

   # Get analysis result
   result = grapher.get_graph_info(graph)
   
   # Basic properties (computed lazily)
   print(f"Vertices: {result.vertex_count}")
   print(f"Edges: {result.edge_count}")
   print(f"Density: {result.density:.3f}")
   print(f"Connected: {result.is_connected}")
   print(f"Components: {result.num_components}")
   
   # Advanced properties (when available)
   if result.transitivity is not None:
       print(f"Clustering: {result.transitivity:.3f}")
   
   if result.diameter is not None:
       print(f"Diameter: {result.diameter}")

**Summary Report:**

.. code-block:: python

   # Get formatted summary
   print(result.summary())
   
   # Output:
   # Graph Analysis Summary:
   #   - Vertices: 100
   #   - Edges: 245
   #   - Density: 0.0495
   #   - Connected: True
   #   - Clustering (Transitivity): 0.3421

**Top Nodes Analysis:**

.. code-block:: python

   # Find most central nodes
   top_degree = result.get_top_n_by('degree', n=5)
   top_betweenness = result.get_top_n_by('betweenness', n=5)
   
   print("Top 5 most connected nodes:")
   for node_id, degree in top_degree:
       print(f"  Node {node_id}: {degree} connections")
   
   print("Top 5 bridge nodes:")
   for node_id, betweenness in top_betweenness:
       print(f"  Node {node_id}: {betweenness:.3f}")

**Statistical Analysis:**

.. code-block:: python

   # Get statistics for any metric
   degree_stats = result.get_metric_stats('degree')
   betweenness_stats = result.get_metric_stats('betweenness')
   
   print("Degree Distribution:")
   print(f"  Mean: {degree_stats['mean']:.2f}")
   print(f"  Std: {degree_stats['std']:.2f}")
   print(f"  Range: {degree_stats['min']:.0f} - {degree_stats['max']:.0f}")

Advanced Analysis Tools
----------------------

Graphizy now includes research-grade analysis tools:

**Percolation Analysis:**

.. code-block:: python

   # Analyze critical thresholds
   ranges = [20, 30, 40, 50, 60, 70, 80]
   
   percolation_result = result.percolation_analyzer.analyze_percolation_threshold(
       data, ranges
   )
   
   print(f"Critical threshold: {percolation_result.critical_range}")
   print(f"Max cluster size: {max(percolation_result.largest_cluster_sizes)}")
   
   # Detect phase transition
   transition = result.percolation_analyzer.detect_phase_transition(percolation_result)
   
   if transition['has_transition']:
       print(f"Phase transition at: {transition['transition_range']}")
       print(f"Transition sharpness: {transition['transition_sharpness']:.3f}")

**Social Network Analysis:**

.. code-block:: python

   # Identify social roles
   social_roles = result.social_analyzer.identify_social_roles(graph)
   
   # Find different role types
   bridges = [node_id for node_id, role in social_roles.items() if role.is_bridge()]
   hubs = [node_id for node_id, role in social_roles.items() if role.is_hub()]
   peripheral = [node_id for node_id, role in social_roles.items() if role.is_peripheral()]
   
   print(f"Bridge nodes (connectors): {bridges}")
   print(f"Hub nodes (popular): {hubs}")
   print(f"Peripheral nodes: {peripheral}")
   
   # Analyze specific roles
   for node_id, role in list(social_roles.items())[:5]:
       print(f"Node {node_id}: {role.roles}")
       print(f"  Betweenness: {role.stats['betweenness']:.3f}")
       print(f"  Degree: {role.stats['degree']}")

**Accessibility Analysis:**

.. code-block:: python

   # For spatial applications (urban planning, etc.)
   population_data = data  # Your population points
   service_data = service_locations  # Your service points
   
   accessibility = result.accessibility_analyzer.analyze_service_accessibility(
       population_data, 
       service_data,
       service_type="hospital",
       service_distance=500.0  # 500m walking distance
   )
   
   print(f"Coverage: {accessibility.get_coverage_percentage():.1f}%")
   print(f"Equity score: {accessibility.get_equity_score():.3f}")
   print(f"Underserved areas: {len(accessibility.underserved_areas)}")
   
   # Identify service gaps
   gaps = result.accessibility_analyzer.identify_service_gaps(accessibility)
   print(f"Service gaps: {len(gaps)}")

Visualization
------------

Create beautiful visualizations of your networks:

**Basic Visualization:**

.. code-block:: python

   # Draw the graph
   image = grapher.draw_graph(graph)
   
   # Display on screen
   grapher.show_graph(image, "My Network")
   
   # Save to file
   grapher.save_graph(image, "network.png")

**Custom Styling:**

.. code-block:: python

   # Update drawing configuration
   grapher.update_config(drawing={
       "point_color": (255, 100, 100),  # Red points (BGR format)
       "line_color": (100, 255, 100),   # Green lines
       "point_radius": 8,               # Larger points
       "line_thickness": 2              # Thicker lines
   })
   
   # Draw with new style
   styled_image = grapher.draw_graph(graph)
   grapher.save_graph(styled_image, "styled_network.png")

**Multiple Visualizations:**

.. code-block:: python

   # Compare different graph types visually
   graph_types = {
       'Proximity': grapher.make_graph("proximity", data, proximity_thresh=50.0),
       'Delaunay': grapher.make_graph("delaunay", data),
       'MST': grapher.make_graph("mst", data)
   }
   
   for name, graph in graph_types.items():
       image = grapher.draw_graph(graph)
       grapher.save_graph(image, f"{name.lower()}_graph.png")
       print(f"Saved {name} visualization")

Memory System (Temporal Analysis)
--------------------------------

Track how networks evolve over time:

**Basic Memory Setup:**

.. code-block:: python

   # Initialize memory system
   grapher.init_memory_manager(
       max_memory_size=50,      # Remember last 50 connections
       track_edge_ages=True     # Track connection ages
   )

**Temporal Evolution:**

.. code-block:: python

   # Simulate network evolution
   original_data = data.copy()
   
   for timestep in range(30):
       # Simulate movement (your own function)
       data = simulate_movement(data, timestep)
       
       # Create current network
       current_graph = grapher.make_graph("proximity", data, proximity_thresh=60.0)
       
       # Update memory with current connections
       grapher.update_memory_with_graph(current_graph)
       
       # Analyze memory periodically
       if timestep % 10 == 0:
           memory_stats = grapher.get_memory_stats()
           print(f"Timestep {timestep}: {memory_stats['total_connections']} total connections")

**Memory Visualization:**

.. code-block:: python

   # Create memory-enhanced graph
   memory_graph = grapher.make_memory_graph(data)
   
   # Visualize with age-based coloring
   memory_image = grapher.draw_memory_graph(
       memory_graph,
       use_age_colors=True,      # Color by connection age
       alpha_range=(0.3, 1.0)   # Fade old connections
   )
   
   grapher.save_graph(memory_image, "memory_network.png")

**Memory Analysis:**

.. code-block:: python

   # Get detailed memory statistics
   memory_stats = grapher.get_memory_stats()
   
   print("Memory Statistics:")
   print(f"  Total connections tracked: {memory_stats['total_connections']}")
   print(f"  Active connections: {memory_stats['active_connections']}")
   
   # Get edge age information
   if hasattr(grapher.memory_manager, 'get_edge_ages'):
       edge_ages = grapher.memory_manager.get_edge_ages()
       
       # Find most persistent connections
       persistent = []
       for edge, age_info in edge_ages.items():
           duration = age_info['last_seen'] - age_info['first_seen']
           persistent.append((edge, duration))
       
       # Sort by persistence
       persistent.sort(key=lambda x: x[1], reverse=True)
       
       print("Most persistent connections:")
       for (node1, node2), duration in persistent[:5]:
           print(f"  {node1} <-> {node2}: {duration} timesteps")

Performance Tips
---------------

**For Large Datasets:**

.. code-block:: python

   # Use efficient graph types for large data
   large_data = generate_and_format_positions(1000, 1000, 500)
   
   # MST is efficient for large datasets
   mst_graph = grapher.make_graph("mst", large_data)
   
   # Use reasonable proximity thresholds
   prox_graph = grapher.make_graph("proximity", large_data, proximity_thresh=30.0)
   
   # Smaller k for KNN
   knn_graph = grapher.make_graph("knn", large_data, k=4)

**Memory Optimization:**

.. code-block:: python

   # For large temporal datasets
   grapher.init_memory_manager(
       max_memory_size=20,       # Smaller memory
       track_edge_ages=False     # Disable for performance
   )

**Batch Processing:**

.. code-block:: python

   # Process multiple datasets efficiently
   datasets = [generate_and_format_positions(400, 300, n) for n in [20, 50, 100]]
   
   results = []
   for i, data in enumerate(datasets):
       graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)
       result = grapher.get_graph_info(graph)
       
       results.append({
           'dataset': i,
           'vertices': result.vertex_count,
           'edges': result.edge_count,
           'density': result.density
       })

Common Workflows
---------------

**Research Analysis Pipeline:**

.. code-block:: python

   def analyze_network(data, graph_type="proximity", **kwargs):
       """Complete analysis pipeline"""
       
       # 1. Create graph
       graph = grapher.make_graph(graph_type, data, **kwargs)
       
       # 2. Basic analysis
       result = grapher.get_graph_info(graph)
       
       # 3. Advanced analysis
       percolation = None
       social_roles = None
       
       if graph_type == "proximity":
           # Percolation analysis for proximity graphs
           ranges = [kwargs.get('proximity_thresh', 50) * factor 
                    for factor in [0.5, 0.75, 1.0, 1.25, 1.5]]
           percolation = result.percolation_analyzer.analyze_percolation_threshold(data, ranges)
       
       # Social analysis for all graph types
       social_roles = result.social_analyzer.identify_social_roles(graph)
       
       # 4. Visualization
       image = grapher.draw_graph(graph)
       
       return {
           'graph': graph,
           'analysis': result,
           'percolation': percolation,
           'social_roles': social_roles,
           'image': image
       }
   
   # Use the pipeline
   analysis = analyze_network(data, "proximity", proximity_thresh=60.0)
   
   print(f"Network has {analysis['analysis'].vertex_count} nodes")
   print(f"Found {len(analysis['social_roles'])} nodes with roles")

**Comparative Analysis:**

.. code-block:: python

   def compare_graph_types(data):
       """Compare different graph types on same data"""
       
       graph_configs = [
           ("proximity", {"proximity_thresh": 50.0}),
           ("delaunay", {}),
           ("mst", {}),
           ("knn", {"k": 4})
       ]
       
       results = {}
       
       for graph_type, params in graph_configs:
           try:
               graph = grapher.make_graph(graph_type, data, **params)
               analysis = grapher.get_graph_info(graph)
               
               results[graph_type] = {
                   'vertices': analysis.vertex_count,
                   'edges': analysis.edge_count,
                   'density': analysis.density,
                   'connected': analysis.is_connected,
                   'clustering': analysis.transitivity
               }
               
               # Save visualization
               image = grapher.draw_graph(graph)
               grapher.save_graph(image, f"comparison_{graph_type}.png")
               
           except Exception as e:
               print(f"Failed to create {graph_type}: {e}")
               results[graph_type] = None
       
       return results
   
   # Compare graph types
   comparison = compare_graph_types(data)
   
   # Print comparison table
   print(f"{'Type':<12} {'Edges':<8} {'Density':<10} {'Connected':<10} {'Clustering':<10}")
   print("-" * 60)
   
   for graph_type, stats in comparison.items():
       if stats:
           clustering = stats['clustering']
           clustering_str = f"{clustering:.3f}" if clustering is not None else "N/A"
           print(f"{graph_type:<12} {stats['edges']:<8} {stats['density']:<10.3f} "
                 f"{str(stats['connected']):<10} {clustering_str:<10}")

Error Handling
-------------

Handle common issues gracefully:

.. code-block:: python

   from graphizy import GraphCreationError, InvalidDataShapeError
   
   try:
       # Validate data first
       is_valid = validate_graphizy_input(data, verbose=True)
       
       if not is_valid:
           print("Data validation failed")
           return
       
       # Create graph
       graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)
       
       # Analyze graph
       result = grapher.get_graph_info(graph)
       
   except InvalidDataShapeError as e:
       print(f"Data format error: {e}")
   except GraphCreationError as e:
       print(f"Graph creation failed: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")

**Handling Empty Graphs:**

.. code-block:: python

   # Check for empty graphs
   if graph.vcount() == 0:
       print("Warning: Graph has no vertices")
       return
   
   if graph.ecount() == 0:
       print("Warning: Graph has no edges (isolated points)")
   
   # Safe analysis
   result = grapher.get_graph_info(graph)
   
   if result.vertex_count > 0:
       print(f"Graph analysis successful: {result.vertex_count} vertices")
   else:
       print("Cannot analyze empty graph")

Integration with Other Libraries
------------------------------

**NetworkX Integration:**

.. code-block:: python

   # Export to NetworkX for advanced analysis
   try:
       import networkx as nx
       
       # Convert Graphizy graph to NetworkX
       nx_graph = grapher.to_networkx(graph)
       
       # Use NetworkX algorithms
       communities = nx.community.greedy_modularity_communities(nx_graph)
       centrality = nx.betweenness_centrality(nx_graph)
       
       print(f"Found {len(communities)} communities")
       print(f"Average betweenness: {sum(centrality.values()) / len(centrality):.3f}")
       
   except ImportError:
       print("NetworkX not available")

**Pandas Integration:**

.. code-block:: python

   # Export analysis results to DataFrame
   try:
       import pandas as pd
       
       # Convert graph analysis to DataFrame
       analysis_data = []
       
       for node_id in range(result.vertex_count):
           node_data = {
               'node_id': node_id,
               'degree': graph.degree(node_id),
               'betweenness': graph.betweenness(vertices=[node_id])[0]
           }
           analysis_data.append(node_data)
       
       df = pd.DataFrame(analysis_data)
       print(df.describe())
       
       # Save to CSV
       df.to_csv('network_analysis.csv', index=False)
       
   except ImportError:
       print("Pandas not available")

Next Steps
----------

Now that you understand the basics, explore these advanced topics:

1. **Advanced Analysis**: Learn the research-grade analysis tools
2. **Memory Systems**: Master temporal network analysis
3. **Weight Computation**: Add sophisticated edge weights
4. **Plugin System**: Create custom graph types
5. **Research Applications**: Use domain-specific tutorials
6. **Performance Optimization**: Handle large datasets efficiently

**Recommended Learning Path:**

1. Master basic graph creation and analysis
2. Experiment with different graph types
3. Learn the advanced analysis tools
4. Explore memory systems for temporal data
5. Try the research tutorials for your domain
6. Develop custom analysis pipelines

You now have a solid foundation in Graphizy's core functionality and are ready to tackle advanced spatial network analysis!
