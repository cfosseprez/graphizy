Examples
========

This section provides practical examples demonstrating the key features of Graphizy across different use cases and applications.

Basic Graph Creation
--------------------

**Simple Delaunay Triangulation:**

.. code-block:: python

   import numpy as np
   from graphizy import Graphing, GraphizyConfig, generate_positions
   
   # Generate sample data
   positions = generate_positions(400, 400, 30)
   data = np.column_stack((np.arange(len(positions)), positions))
   
   # Create grapher
   config = GraphizyConfig()
   config.graph.dimension = (400, 400)
   grapher = Graphing(config=config)
   
   # Create and visualize Delaunay triangulation
   delaunay_graph = grapher.make_delaunay(data)
   image = grapher.draw_graph(delaunay_graph)
   grapher.show_graph(image, "Delaunay Triangulation")
   
   # Save to file
   grapher.save_graph(image, "delaunay_example.jpg")

**Comparing Graph Types:**

.. code-block:: python

   # Create multiple graph types from the same data
   graph_types = {
       'Delaunay': grapher.make_delaunay(data),
       'Proximity': grapher.make_proximity(data, proximity_thresh=60.0),
       'MST': grapher.make_mst(data),
       'KNN': grapher.make_knn(data, k=4)  # Requires scipy
   }
   
   # Compare properties
   print("Graph Type Comparison:")
   print(f"{'Type':<12} {'Vertices':<10} {'Edges':<8} {'Density':<10} {'Connected':<10}")
   print("-" * 55)
   
   for name, graph in graph_types.items():
       info = grapher.get_graph_info(graph)
       print(f"{name:<12} {info['vertex_count']:<10} {info['edge_count']:<8} "
             f"{info['density']:<10.3f} {info['is_connected']:<10}")

Graph Analysis
--------------

**Centrality Analysis:**

.. code-block:: python

   # Create a proximity graph for analysis
   graph = grapher.make_proximity(data, proximity_thresh=80.0)
   
   # Calculate different centrality measures
   degree_cent = grapher.call_method(graph, 'degree')
   betweenness_cent = grapher.call_method(graph, 'betweenness') 
   closeness_cent = grapher.call_method(graph, 'closeness')
   
   # Find most central nodes
   def top_nodes(centrality_dict, n=5):
       return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]
   
   print("Top 5 nodes by different centrality measures:")
   print("\nDegree Centrality (most connected):")
   for node, value in top_nodes(degree_cent):
       print(f"  Node {node}: {value} connections")
   
   print("\nBetweenness Centrality (best bridges):")
   for node, value in top_nodes(betweenness_cent):
       print(f"  Node {node}: {value:.3f}")
   
   print("\nCloseness Centrality (best broadcasters):")
   for node, value in top_nodes(closeness_cent):
       print(f"  Node {node}: {value:.3f}")

**Community Detection:**

.. code-block:: python

   # Find communities in the graph
   communities = grapher.call_method_raw(graph, 'community_leiden')
   
   print(f"Found {len(communities)} communities:")
   for i, community in enumerate(communities):
       print(f"  Community {i+1}: {len(community)} nodes - {list(community)[:5]}{'...' if len(community) > 5 else ''}")
   
   # Calculate modularity (quality of community division)
   modularity = communities.modularity
   print(f"Modularity: {modularity:.3f}")

Memory System Examples
----------------------

**Basic Memory Tracking:**

.. code-block:: python

   # Initialize memory system
   grapher.init_memory_manager(
       max_memory_size=30,
       max_iterations=50,
       track_edge_ages=True
   )
   
   # Simulate dynamic system
   original_data = data.copy()
   
   for iteration in range(100):
       # Add small random movements
       data[:, 1:3] += np.random.normal(0, 3, (len(data), 2))
       
       # Keep particles in bounds
       data[:, 1] = np.clip(data[:, 1], 0, 400)
       data[:, 2] = np.clip(data[:, 2], 0, 400)
       
       # Update memory with current proximity graph
       grapher.update_memory_with_proximity(data, proximity_thresh=50.0)
       
       # Visualize every 20 iterations
       if iteration % 20 == 0:
           memory_graph = grapher.make_memory_graph(data)
           memory_image = grapher.draw_memory_graph(
               memory_graph, 
               use_age_colors=True,
               alpha_range=(0.4, 1.0)
           )
           grapher.save_graph(memory_image, f"memory_evolution_{iteration:03d}.jpg")
           
           # Print memory statistics
           stats = grapher.get_memory_stats()
           print(f"Iteration {iteration}: {stats['total_connections']} total connections")

**Memory Persistence Analysis:**

.. code-block:: python

   # Analyze which connections persisted longest
   edge_ages = grapher.memory_manager.get_edge_ages()
   
   # Calculate connection durations
   persistent_connections = []
   for edge, age_info in edge_ages.items():
       duration = age_info['last_seen'] - age_info['first_seen']
       persistent_connections.append((edge, duration, age_info))
   
   # Sort by persistence
   persistent_connections.sort(key=lambda x: x[1], reverse=True)
   
   print("Most persistent connections:")
   for (node1, node2), duration, age_info in persistent_connections[:10]:
       print(f"  {node1} <-> {node2}: lasted {duration} iterations "
             f"(first seen: {age_info['first_seen']}, last seen: {age_info['last_seen']})")

Real-World Applications
-----------------------

**Social Network Analysis:**

.. code-block:: python

   # Simulate a social network with evolving friendships
   def simulate_social_network():
       # Create initial social positions (e.g., workplace layout)
       social_positions = generate_positions(200, 200, 25)
       social_data = np.column_stack((np.arange(len(social_positions)), social_positions))
       
       grapher_social = Graphing(dimension=(200, 200))
       grapher_social.init_memory_manager(max_memory_size=50, track_edge_ages=True)
       
       # Simulate friendship formation over time
       for week in range(20):
           # People move slightly (changing office positions, etc.)
           social_data[:, 1:3] += np.random.normal(0, 2, (len(social_data), 2))
           
           # Friendships form based on proximity (people working near each other)
           grapher_social.update_memory_with_proximity(
               social_data, 
               proximity_thresh=30.0  # Friendship distance
           )
       
       # Analyze the social network
       friendship_graph = grapher_social.make_memory_graph(social_data)
       
       # Find social hubs (people with many friendships)
       degrees = grapher_social.call_method(friendship_graph, 'degree')
       social_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
       
       print("Social network analysis:")
       print("Top 5 social hubs (most friendships):")
       for person, num_friends in social_hubs:
           print(f"  Person {person}: {num_friends} friends")
       
       # Find friendship brokers (high betweenness)
       betweenness = grapher_social.call_method(friendship_graph, 'betweenness')
       brokers = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]
       
       print("\nTop 3 friendship brokers (connect different groups):")
       for person, broker_score in brokers:
           print(f"  Person {person}: {broker_score:.3f}")
       
       return friendship_graph, grapher_social
   
   friendship_graph, social_grapher = simulate_social_network()

**Sensor Network Reliability:**

.. code-block:: python

   def analyze_sensor_network():
       # Create sensor network layout
       sensor_positions = generate_positions(500, 500, 40)
       sensor_data = np.column_stack((np.arange(len(sensor_positions)), sensor_positions))
       
       sensor_grapher = Graphing(dimension=(500, 500))
       sensor_grapher.init_memory_manager(
           max_memory_size=20,    # Recent connections only
           max_iterations=100,    # Sliding window
           track_edge_ages=True
       )
       
       # Simulate sensor communication over time
       for time_step in range(200):
           # Sensors occasionally fail or have interference
           active_sensors = sensor_data.copy()
           
           # Random sensor failures (5% chance)
           failure_mask = np.random.random(len(active_sensors)) > 0.05
           active_sensors = active_sensors[failure_mask]
           
           # Communication based on signal strength (proximity)
           if len(active_sensors) > 0:
               sensor_grapher.update_memory_with_proximity(
                   active_sensors,
                   proximity_thresh=80.0  # Communication range
               )
       
       # Analyze network reliability
       reliability_graph = sensor_grapher.make_memory_graph(sensor_data)
       memory_stats = sensor_grapher.get_memory_stats()
       
       # Find most reliable communication links
       edge_ages = sensor_grapher.memory_manager.get_edge_ages()
       reliable_links = [
           (edge, age_info['last_seen'] - age_info['first_seen'])
           for edge, age_info in edge_ages.items()
           if age_info['last_seen'] - age_info['first_seen'] > 50
       ]
       
       print("Sensor network reliability analysis:")
       print(f"Total sensors: {len(sensor_data)}")
       print(f"Reliable communication links: {len(reliable_links)}")
       print(f"Network connectivity: {sensor_grapher.call_method(reliability_graph, 'is_connected')}")
       
       # Find critical sensors (high betweenness = network bridges)
       betweenness = sensor_grapher.call_method(reliability_graph, 'betweenness')
       critical_sensors = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
       
       print("Critical sensors (network bridges):")
       for sensor, criticality in critical_sensors:
           print(f"  Sensor {sensor}: criticality {criticality:.3f}")
       
       return reliability_graph
   
   sensor_graph = analyze_sensor_network()

Performance Optimization
------------------------

**Large Dataset Handling:**

.. code-block:: python

   def handle_large_dataset():
       # Generate large dataset
       large_positions = generate_positions(1000, 1000, 500)
       large_data = np.column_stack((np.arange(len(large_positions)), large_positions))
       
       large_grapher = Graphing(dimension=(1000, 1000))
       
       # For large datasets, use efficient graph types
       print("Performance comparison on large dataset:")
       
       import time
       
       # MST is efficient for large datasets
       start_time = time.time()
       mst_graph = large_grapher.make_mst(large_data)
       mst_time = time.time() - start_time
       
       # Proximity with reasonable threshold
       start_time = time.time()
       prox_graph = large_grapher.make_proximity(large_data, proximity_thresh=50.0)
       prox_time = time.time() - start_time
       
       # KNN with small k
       start_time = time.time()
       try:
           knn_graph = large_grapher.make_knn(large_data, k=4)
           knn_time = time.time() - start_time
       except:
           knn_time = float('inf')
           print("KNN failed (scipy not available)")
       
       print(f"MST creation: {mst_time:.3f} seconds")
       print(f"Proximity creation: {prox_time:.3f} seconds") 
       print(f"KNN creation: {knn_time:.3f} seconds")
       
       # Memory optimization for large datasets
       large_grapher.init_memory_manager(
           max_memory_size=10,     # Smaller memory
           max_iterations=25,      # Shorter history
           track_edge_ages=False   # Disable for performance
       )
       
       return large_data, large_grapher
   
   large_data, large_grapher = handle_large_dataset()

**Batch Processing:**

.. code-block:: python

   def batch_analysis():
       # Analyze multiple datasets in batch
       results = []
       
       for dataset_size in [50, 100, 200, 300]:
           positions = generate_positions(400, 400, dataset_size)
           data = np.column_stack((np.arange(len(positions)), positions))
           
           batch_grapher = Graphing(dimension=(400, 400))
           
           # Test different graph types
           for graph_type, create_func in [
               ('delaunay', lambda d: batch_grapher.make_delaunay(d)),
               ('proximity', lambda d: batch_grapher.make_proximity(d, 60.0)),
               ('mst', lambda d: batch_grapher.make_mst(d))
           ]:
               try:
                   graph = create_func(data)
                   info = batch_grapher.get_graph_info(graph)
                   
                   results.append({
                       'dataset_size': dataset_size,
                       'graph_type': graph_type,
                       'vertices': info['vertex_count'],
                       'edges': info['edge_count'],
                       'density': info['density'],
                       'connected': info['is_connected'],
                       'avg_path_length': info.get('average_path_length', 0),
                       'clustering': info.get('transitivity', 0)
                   })
               except Exception as e:
                   print(f"Failed {graph_type} for size {dataset_size}: {e}")
       
       # Print results summary
       print("\nBatch Analysis Results:")
       print(f"{'Size':<6} {'Type':<10} {'Edges':<8} {'Density':<8} {'Connected':<10} {'Clustering':<10}")
       print("-" * 60)
       
       for result in results:
           print(f"{result['dataset_size']:<6} {result['graph_type']:<10} "
                 f"{result['edges']:<8} {result['density']:<8.3f} "
                 f"{str(result['connected']):<10} {result['clustering']:<10.3f}")
   
   batch_analysis()

Custom Configuration
--------------------

**Styling and Visualization:**

.. code-block:: python

   # Create custom styled visualizations
   def create_styled_graph():
       positions = generate_positions(300, 300, 25)
       data = np.column_stack((np.arange(len(positions)), positions))
       
       # Create custom configuration
       custom_config = GraphizyConfig()
       custom_config.graph.dimension = (300, 300)
       custom_config.drawing.line_color = (255, 0, 0)      # Red lines
       custom_config.drawing.point_color = (0, 255, 255)   # Yellow points
       custom_config.drawing.line_thickness = 3
       custom_config.drawing.point_radius = 10
       
       styled_grapher = Graphing(config=custom_config)
       
       # Create and style different graph types
       graphs = {
           'Delaunay': styled_grapher.make_delaunay(data),
           'Proximity': styled_grapher.make_proximity(data, 50.0),
           'MST': styled_grapher.make_mst(data)
       }
       
       # Save styled visualizations
       for name, graph in graphs.items():
           image = styled_grapher.draw_graph(graph)
           styled_grapher.save_graph(image, f"styled_{name.lower()}.jpg")
           print(f"Saved styled {name} visualization")
   
   create_styled_graph()

Interactive Examples
--------------------

**Real-time Graph Evolution:**

.. code-block:: python

   def interactive_evolution():
       """
       Run the interactive Brownian motion demo with different graph types.
       This example shows how to use the interactive features.
       """
       print("Interactive Examples:")
       print("Run these commands to see graphs evolve in real-time:")
       print()
       print("# Basic proximity graph simulation")
       print("python examples/improved_brownian.py 1")
       print()
       print("# Delaunay triangulation with memory")
       print("python examples/improved_brownian.py 2 --memory")
       print()
       print("# Minimum spanning tree evolution")
       print("python examples/improved_brownian.py 4 --memory --particles 100")
       print()
       print("# Compare all graph types")
       print("python examples/improved_brownian.py 5 --memory")
       print()
       print("Interactive controls:")
       print("  ESC - Exit")
       print("  SPACE - Pause/Resume")
       print("  R - Reset simulation")
       print("  M - Toggle memory on/off")
       print("  1-5 - Switch graph types")
       print("  +/- - Adjust memory size")
   
   interactive_evolution()

These examples demonstrate the versatility and power of Graphizy across different domains and use cases. From basic graph creation to complex temporal analysis, the library provides the tools needed for comprehensive network analysis.
