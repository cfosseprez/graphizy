Quick Start Guide
================

This guide will get you up and running with Graphizy in just a few minutes. You'll learn the basic concepts and see how to create your first spatial networks.

Installation
-----------

Install Graphizy using pip:

.. code-block:: bash

   pip install graphizy

For development or the latest features:

.. code-block:: bash

   git clone https://github.com/cfosseprez/graphizy.git
   cd graphizy
   pip install -e .

Basic Concepts
--------------

**What is Graphizy?**

Graphizy creates networks (graphs) from spatial coordinate data. Think of it as connecting dots based on their positions in space, with different connection rules creating different types of networks.

**Key Components:**

- **Graphing**: The main class that creates and analyzes networks
- **Data Format**: Your spatial coordinates in a simple format
- **Graph Types**: Different ways to connect points (proximity, triangulation, etc.)
- **Analysis Tools**: Advanced analyzers for research applications

Your First Graph
---------------

Let's create a simple network from random points:

.. code-block:: python

   from graphizy import Graphing, generate_and_format_positions
   
   # Step 1: Generate some random points
   data = generate_and_format_positions(
       size_x=400,      # Canvas width
       size_y=300,      # Canvas height  
       num_particles=30 # Number of points
   )
   
   # Step 2: Create a Graphing object
   grapher = Graphing(dimension=(400, 300))
   
   # Step 3: Create a proximity network
   # Points within 60 units of each other get connected
   graph = grapher.make_graph("proximity", data, proximity_thresh=60.0)
   
   # Step 4: Visualize the result
   image = grapher.draw_graph(graph)
   grapher.show_graph(image, "My First Network")

That's it! You've created your first spatial network.

Understanding the Data Format
----------------------------

Graphizy expects data in a simple format: each row represents one point with an ID and coordinates.

.. code-block:: python

   import numpy as np
   
   # Manual data creation
   data = np.array([
       [0, 100, 150],  # Point 0 at position (100, 150)
       [1, 200, 200],  # Point 1 at position (200, 200)
       [2, 150, 100],  # Point 2 at position (150, 100)
       # ... more points
   ])
   
   # Or use the helper function
   data = generate_and_format_positions(400, 300, 30)

**Data Requirements:**
- Column 0: Point ID (unique identifier)
- Column 1: X coordinate
- Column 2: Y coordinate

Different Graph Types
-------------------

Graphizy offers several ways to connect your points:

**Proximity Graph** - Connect nearby points:

.. code-block:: python

   # Connect points within 50 units of each other
   proximity_graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)

**Delaunay Triangulation** - Optimal triangular mesh:

.. code-block:: python

   # Create triangular mesh (no overlapping triangles)
   delaunay_graph = grapher.make_graph("delaunay", data)

**K-Nearest Neighbors** - Connect to closest neighbors:

.. code-block:: python

   # Each point connects to its 4 closest neighbors
   knn_graph = grapher.make_graph("knn", data, k=4)

**Minimum Spanning Tree** - Minimal connected network:

.. code-block:: python

   # Minimum edges needed to connect all points
   mst_graph = grapher.make_graph("mst", data)

**Gabriel Graph** - Geometric proximity network:

.. code-block:: python

   # Special geometric connectivity rules
   gabriel_graph = grapher.make_graph("gabriel", data)

Analyzing Your Networks
----------------------

Once you have a graph, you can analyze its properties:

.. code-block:: python

   # Get comprehensive analysis
   result = grapher.get_graph_info(graph)
   
   # Basic properties
   print(f"Number of points: {result.vertex_count}")
   print(f"Number of connections: {result.edge_count}")
   print(f"Network density: {result.density:.3f}")
   print(f"Is connected: {result.is_connected}")
   
   # Advanced properties
   if result.transitivity is not None:
       print(f"Clustering coefficient: {result.transitivity:.3f}")
   
   if result.diameter is not None:
       print(f"Network diameter: {result.diameter}")

**Quick Summary Report:**

.. code-block:: python

   # Get a formatted summary
   print(result.summary())

Advanced Analysis Tools (New!)
-----------------------------

Graphizy now includes powerful research-grade analysis tools:

**Percolation Analysis** - Find critical thresholds:

.. code-block:: python

   # Test different connection distances
   ranges = [20, 30, 40, 50, 60, 70]
   
   # Analyze percolation behavior
   percolation_result = result.percolation_analyzer.analyze_percolation_threshold(
       data, ranges
   )
   
   print(f"Critical threshold: {percolation_result.critical_range}")
   print(f"Largest cluster: {max(percolation_result.largest_cluster_sizes)}")

**Social Network Analysis** - Identify roles:

.. code-block:: python

   # Identify social roles in the network
   social_roles = result.social_analyzer.identify_social_roles(graph)
   
   # Find bridges (connectors between groups)
   bridges = [node_id for node_id, role in social_roles.items() 
             if role.is_bridge()]
   
   # Find hubs (highly connected nodes)  
   hubs = [node_id for node_id, role in social_roles.items()
          if role.is_hub()]
   
   print(f"Network bridges: {bridges}")
   print(f"Network hubs: {hubs}")

**Accessibility Analysis** - Study spatial coverage:

.. code-block:: python

   # For spatial planning applications
   population_points = data  # Your population data
   service_points = service_data  # Your service locations
   
   accessibility_result = result.accessibility_analyzer.analyze_service_accessibility(
       population_points, service_points, 
       service_type="hospital", 
       service_distance=500.0  # 500m walking distance
   )
   
   print(f"Coverage: {accessibility_result.get_coverage_percentage():.1f}%")
   print(f"Equity score: {accessibility_result.get_equity_score():.3f}")

Memory System for Temporal Analysis
----------------------------------

Track how networks change over time:

.. code-block:: python

   # Initialize memory system
   grapher.init_memory_manager(
       max_memory_size=50,    # Remember last 50 connections
       track_edge_ages=True   # Track how long connections last
   )
   
   # Simulate evolution over time
   for timestep in range(20):
       # Update positions (your simulation)
       data = update_positions(data, timestep)
       
       # Create current network
       current_graph = grapher.make_graph("proximity", data, proximity_thresh=60.0)
       
       # Update memory with current connections
       grapher.update_memory_with_graph(current_graph)
       
       # Visualize memory-enhanced network
       if timestep % 5 == 0:
           memory_graph = grapher.make_memory_graph(data)
           memory_image = grapher.draw_memory_graph(
               memory_graph, 
               use_age_colors=True,      # Color by connection age
               alpha_range=(0.3, 1.0)   # Fade old connections
           )
           grapher.save_graph(memory_image, f"evolution_{timestep:02d}.png")

Customizing Visualization
------------------------

Make your networks look exactly how you want:

.. code-block:: python

   # Update drawing style
   grapher.update_config(drawing={
       "point_color": (255, 100, 100),  # Red points (B, G, R format)
       "line_color": (100, 255, 100),   # Green lines
       "point_radius": 8,               # Larger points
       "line_thickness": 2              # Thicker lines
   })
   
   # Draw with custom style
   styled_image = grapher.draw_graph(graph)
   grapher.save_graph(styled_image, "styled_network.png")

Saving and Loading
-----------------

Save your work for later:

.. code-block:: python

   # Save graph visualization
   grapher.save_graph(image, "my_network.png")
   
   # Save analysis results
   import json
   
   analysis_data = {
       'vertex_count': result.vertex_count,
       'edge_count': result.edge_count,
       'density': result.density,
       'is_connected': result.is_connected
   }
   
   with open('analysis_results.json', 'w') as f:
       json.dump(analysis_data, f, indent=2)

**Export to Other Tools:**

.. code-block:: python

   # Export to NetworkX for advanced analysis
   try:
       import networkx as nx
       nx_graph = grapher.to_networkx(graph)
       
       # Now use NetworkX methods
       communities = nx.community.greedy_modularity_communities(nx_graph)
       print(f"Found {len(communities)} communities")
   except ImportError:
       print("NetworkX not available")

Common Patterns
--------------

**Compare Multiple Graph Types:**

.. code-block:: python

   # Create different graph types from same data
   graph_types = {
       'Proximity': grapher.make_graph("proximity", data, proximity_thresh=50.0),
       'Delaunay': grapher.make_graph("delaunay", data),
       'MST': grapher.make_graph("mst", data),
       'KNN': grapher.make_graph("knn", data, k=4)
   }
   
   # Compare properties
   print(f"{'Type':<12} {'Edges':<8} {'Density':<10} {'Connected':<10}")
   print("-" * 45)
   
   for name, graph in graph_types.items():
       info = grapher.get_graph_info(graph)
       print(f"{name:<12} {info.edge_count:<8} {info.density:<10.3f} {info.is_connected}")

**Batch Processing:**

.. code-block:: python

   # Process multiple datasets
   datasets = [
       generate_and_format_positions(400, 300, 20),
       generate_and_format_positions(400, 300, 50), 
       generate_and_format_positions(400, 300, 100)
   ]
   
   results = []
   for i, data in enumerate(datasets):
       graph = grapher.make_graph("proximity", data, proximity_thresh=60.0)
       info = grapher.get_graph_info(graph)
       
       results.append({
           'dataset': i,
           'points': info.vertex_count,
           'connections': info.edge_count,
           'density': info.density
       })
       
       # Save visualization
       image = grapher.draw_graph(graph)
       grapher.save_graph(image, f"dataset_{i}_network.png")

What's Next?
-----------

Now that you know the basics, explore these advanced topics:

1. **Memory Systems**: Track temporal evolution in dynamic networks
2. **Weight Computation**: Add sophisticated edge weights and attributes  
3. **Plugin System**: Create custom graph types for your specific needs
4. **Research Applications**: Use the specialized analysis tools for your domain
5. **Performance Optimization**: Handle large datasets efficiently

**Recommended Learning Path:**

1. **Start Here**: Complete this quickstart guide
2. **Basic Usage**: Learn all graph types and analysis methods
3. **Advanced Analysis**: Explore the new research-grade analyzers
4. **Memory Systems**: Add temporal analysis to your networks
5. **Research Applications**: Check out domain-specific tutorials
6. **Custom Development**: Create plugins and custom analysis methods

**Get Help:**

- **Documentation**: Comprehensive guides and API reference
- **Examples**: Working code for common use cases
- **Tutorials**: Research-grade applications in multiple domains
- **GitHub Issues**: Report bugs or request features

**Example Research Tutorials:**

- **Particle Physics**: Percolation analysis and phase transitions
- **Animal Behavior**: Social network dynamics and role identification  
- **Urban Planning**: Accessibility analysis and spatial equity

You're now ready to create sophisticated spatial networks and perform advanced analysis with Graphizy!
