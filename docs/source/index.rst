Graphizy Documentation
======================

.. image:: https://raw.githubusercontent.com/cfosseprez/graphizy/main/images/detection_to_graph.png
   :alt: Detection to Graph
   :align: center

*A powerful, fast, and flexible Python library for building and analyzing graphs from 2D spatial data*

Graphizy specializes in creating comprehensive graph types from point data, with advanced memory-enhanced analysis for temporal patterns, sophisticated weight computation systems, and real-time streaming capabilities. Built on OpenCV and igraph, it provides a modern unified API for graph construction and comprehensive analytics.

Quick Start
-----------

.. code-block:: python

   from graphizy import Graphing, GraphizyConfig, generate_and_format_positions

   # Generate sample data
   data = generate_and_format_positions(size_x=800, size_y=600, num_particles=100)

   # Configure and create grapher
   config = GraphizyConfig(dimension=(800, 600))  
   grapher = Graphing(config=config)

   # Create different graph types using unified interface
   delaunay_graph = grapher.make_graph("delaunay", data)
   proximity_graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)
   knn_graph = grapher.make_graph("knn", data, k=4)

   # Advanced analysis with new API
   graph_info = grapher.get_graph_info(delaunay_graph)
   print(f"Density: {graph_info.density:.3f}")
   print(f"Connected: {graph_info.is_connected}")
   
   # Use advanced analyzers
   social_roles = graph_info.social_analyzer.identify_social_roles(delaunay_graph)
   percolation_result = graph_info.percolation_analyzer.analyze_percolation_threshold(
       data, [20, 30, 40, 50, 60]
   )

Key Features
------------

- **One API for All Graphs**
Create Delaunay, k-NN, MST, Gabriel, Proximity, and even custom graphs with a single make_graph() call. Plugin-friendly, smart defaults, and fully type-safe.

- **Temporal Memory System**
Track how connections evolve over time. Use built-in memory features for persistence-aware analysis, temporal filtering, and age-based visualization.

- **Rich Graph Types, Easily Extended**
From spatial graphs to domain-specific topologies: support includes Delaunay triangulations, proximity graphs, k-nearest neighbors, MSTs, and custom plugins.

- **Instant Network Analysis**
Access over 200 igraph algorithms with real-time stats: clustering, centrality, components, and more. All robust to disconnections. NetworkX compatible.

- **Custom Weights, Real-Time Ready**
Define weights using distance, inverse, Gaussian, or custom formulas. Memory-aware weight updates and vectorized for performance.

- **Advanced Tools for Spatial & Temporal Insights**
Includes percolation thresholds, service accessibility, social dynamics, and time-aware community tracking — all tailored for dynamic networks.

- **Visualization & Streaming**
Visualize network memory with age-based coloring and transparency. Stream updates in real time, or export static snapshots. Comes with CLI tools and interactive demos.


User Guide
==========

.. toctree::
   :maxdepth: 2

   user_guide/installation
   user_guide/quickstart
   user_guide/data_formats
   user_guide/basic_usage
   user_guide/advanced_analysis
   user_guide/configuration

Core Systems
============

.. toctree::
   :maxdepth: 2

   graph_types/index
   memory/index  
   weight/index
   advanced_analysis/index

Advanced Features
=================

.. toctree::
   :maxdepth: 2

   advanced/streaming
   advanced/plugin_system
   advanced/performance
   advanced/networkx_bridge

Research Applications
====================

.. toctree::
   :maxdepth: 2

   research/particle_physics
   research/behavioral_ecology
   research/urban_planning
   research/overview

API Reference
=============

.. toctree::
   :maxdepth: 2

   api/graphing
   api/config
   api/algorithms
   api/analysis
   api/memory
   api/weight
   api/drawing
   api/exceptions

Examples & Tutorials
====================

.. toctree::
   :maxdepth: 2

   examples/basic_usage
   examples/advanced_analysis
   examples/graph_types
   examples/memory_system
   examples/weight_computation
   examples/streaming
   examples/custom_plugins

Development
===========

.. toctree::
   :maxdepth: 2

   development/contributing
   development/testing
   development/performance
   development/release_notes

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

