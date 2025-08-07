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

🔄 **Unified Graph Creation Interface**
   - Modern API with single `make_graph()` method for all graph types
   - Plugin system for easily adding custom graph algorithms
   - Smart defaults with intelligent parameter handling 
   - Type-safe runtime configuration validation

📊 **Comprehensive Graph Types**
   - **Delaunay Triangulation**: Optimal triangular meshes from point sets
   - **Proximity Graphs**: Connect nearby points based on distance thresholds  
   - **K-Nearest Neighbors**: Connect each point to its k closest neighbors
   - **Minimum Spanning Tree**: Minimal connected graph with shortest total edge length
   - **Gabriel Graph**: Geometric proximity graph (subset of Delaunay triangulation)
   - **Custom Graphs**: Extensible plugin system for domain-specific algorithms

🔬 **Advanced Analysis Tools** *(New!)*
   - **Percolation Analysis**: Critical threshold detection and phase transition analysis
   - **Social Network Analysis**: Role identification, temporal tracking, stability metrics
   - **Accessibility Analysis**: Service coverage, spatial equity, gap identification
   - **Temporal Dynamics**: Evolution tracking with memory-enhanced analysis

🧠 **Advanced Memory Systems**
   - Temporal analysis tracking connections across time steps
   - Smart integration with automatic memory updates and configurable retention
   - Age-based visualization showing connection persistence over time
   - Performance optimized with vectorized operations for real-time applications

⚖️ **Sophisticated Weight Computation**  
   - Multiple methods: distance, inverse distance, Gaussian, and custom formulas
   - Real-time computation with optimized fast computers for high-performance applications
   - Compute any edge attribute using mathematical expressions
   - Memory integration for weight computation on memory-enhanced structures

📈 **Comprehensive Graph Analysis**
   - Full igraph integration with access to 200+ graph analysis algorithms
   - Resilient methods providing robust analysis that handles disconnected graphs gracefully
   - Real-time statistics: vertex count, edge count, connectivity, clustering, centrality
   - Component analysis with detailed connectivity and community structure analysis

🎨 **Advanced Visualization & Real-Time Processing**
   - Memory visualization with age-based coloring and transparency effects
   - Real-time streaming with high-performance streaming and async support
   - Flexible configuration using runtime-configurable parameters with type-safe dataclasses
   - Interactive demos and built-in CLI tools

What's New in v0.1.18
---------------------

✨ **Advanced Analysis API**
   - New `PercolationAnalyzer` for automated critical threshold detection
   - New `SocialNetworkAnalyzer` for role identification and temporal tracking
   - New `AccessibilityAnalyzer` for spatial equity and service coverage analysis
   - All analyzers accessible through `GraphAnalysisResult.analyzer_name`

🚀 **Enhanced Tutorials**
   - Updated research tutorials demonstrating real-world applications
   - Particle physics with percolation analysis
   - Animal behavior with social role tracking
   - Urban planning with accessibility analysis

🔧 **API Improvements**
   - Simplified graph creation with `make_graph()` unified interface
   - Enhanced `GraphAnalysisResult` with lazy-loaded properties
   - Better error handling and user-friendly messages
   - Improved type safety throughout

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
