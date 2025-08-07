Advanced Analysis Tools
=======================

Graphizy provides three powerful advanced analysis tools that enable sophisticated research applications across multiple domains. These tools are seamlessly integrated with the main API and provide automated analysis capabilities that were previously complex to implement.

Overview
--------

The advanced analysis tools are accessible through the `GraphAnalysisResult` object:

.. code-block:: python

   from graphizy import Graphing, generate_and_format_positions
   
   # Create your graph
   data = generate_and_format_positions(800, 600, 100)
   grapher = Graphing()
   graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)
   
   # Get analysis result with advanced tools
   result = grapher.get_graph_info(graph)
   
   # Access advanced analyzers
   percolation = result.percolation_analyzer
   social = result.social_analyzer  
   accessibility = result.accessibility_analyzer

The Three Analysis Tools
------------------------

1. **PercolationAnalyzer**: For studying phase transitions and critical phenomena
2. **SocialNetworkAnalyzer**: For identifying roles and tracking temporal dynamics
3. **AccessibilityAnalyzer**: For spatial equity and service coverage analysis

Each analyzer provides specialized methods optimized for specific research domains while maintaining a consistent, user-friendly API.

Percolation Analysis
-------------------

Studies critical thresholds and phase transitions in spatial networks.

**Key Features:**
- Automated critical threshold detection
- Phase transition characterization
- Cluster size analysis
- Transition sharpness metrics

**Example Usage:**

.. code-block:: python

   # Generate particle positions
   data = generate_and_format_positions(600, 600, 150)
   
   # Define interaction ranges to test
   ranges = [15, 20, 25, 30, 35, 40, 45, 50]
   
   # Get percolation analyzer
   graph = grapher.make_graph("proximity", data, proximity_thresh=ranges[0])
   result = grapher.get_graph_info(graph)
   
   # Analyze percolation threshold
   percolation_result = result.percolation_analyzer.analyze_percolation_threshold(
       data, ranges
   )
   
   print(f"Critical range: {percolation_result.critical_range}")
   print(f"Max cluster size: {max(percolation_result.largest_cluster_sizes)}")
   
   # Detect phase transition
   transition = result.percolation_analyzer.detect_phase_transition(percolation_result)
   print(f"Phase transition detected: {transition['has_transition']}")
   print(f"Transition sharpness: {transition['transition_sharpness']:.3f}")

**Research Applications:**
- Particle physics simulations
- Material science percolation studies
- Network robustness analysis
- Critical phenomena research

Social Network Analysis
----------------------

Identifies social roles and tracks temporal dynamics in networks.

**Key Features:**
- Automated role classification (bridge, hub, peripheral)
- Temporal role tracking across time steps
- Role stability analysis
- Leadership pattern detection

**Example Usage:**

.. code-block:: python

   # Create temporal sequence of networks
   trajectory = []
   for t in range(20):
       # Simulate movement
       positions = simulate_movement(t)  # Your simulation function
       data = format_positions(positions)
       trajectory.append(data)
   
   # Create graphs for each timestep
   graph_sequence = []
   for data in trajectory:
       graph = grapher.make_graph("proximity", data, proximity_thresh=80.0)
       graph_sequence.append(graph)
   
   # Get social analyzer
   result = grapher.get_graph_info(graph_sequence[0])
   social_analyzer = result.social_analyzer
   
   # Identify roles in a single graph
   roles = social_analyzer.identify_social_roles(graph_sequence[0])
   
   for node_id, role in roles.items():
       print(f"Node {node_id}: {role.roles} (betweenness: {role.stats['betweenness']:.3f})")
   
   # Track temporal evolution
   temporal_roles = social_analyzer.track_temporal_roles(graph_sequence)
   stability_scores = social_analyzer.get_role_stability(temporal_roles)
   
   print("Most stable individuals:")
   for node_id, stability in sorted(stability_scores.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
       print(f"  Node {node_id}: {stability:.3f}")

**Research Applications:**
- Animal behavior and social dynamics
- Human social network analysis
- Leadership emergence studies
- Group formation research

Accessibility Analysis  
---------------------

Analyzes spatial accessibility, service coverage, and equity.

**Key Features:**
- Service coverage calculation
- Spatial equity scoring
- Service gap identification
- Comparative accessibility analysis

**Example Usage:**

.. code-block:: python

   # Define population and service locations
   population_data = generate_residential_areas(300)  # Your function
   school_data = generate_service_locations(12, "school")  # Your function
   
   # Get accessibility analyzer
   graph = grapher.make_graph("proximity", population_data[:10], proximity_thresh=100)
   result = grapher.get_graph_info(graph)
   accessibility_analyzer = result.accessibility_analyzer
   
   # Analyze school accessibility
   accessibility_result = accessibility_analyzer.analyze_service_accessibility(
       population_data, 
       school_data,
       service_type="school",
       service_distance=400.0  # 400m walking distance
   )
   
   print(f"Coverage: {accessibility_result.get_coverage_percentage():.1f}%")
   print(f"Equity score: {accessibility_result.get_equity_score():.3f}")
   print(f"Underserved areas: {len(accessibility_result.underserved_areas)}")
   
   # Identify service gaps
   service_gaps = accessibility_analyzer.identify_service_gaps(
       accessibility_result, cluster_distance=200.0
   )
   
   print(f"Service gaps identified: {len(service_gaps)}")
   for gap in service_gaps[:3]:  # Show top 3 gaps
       print(f"  Gap {gap['gap_id']}: {gap['size']} affected, "
             f"severity {gap['severity']:.3f}")
   
   # Compare multiple services
   hospital_result = accessibility_analyzer.analyze_service_accessibility(
       population_data, hospital_data, "hospital", 1500.0
   )
   
   comparison = accessibility_analyzer.compare_accessibility([
       accessibility_result, hospital_result
   ])
   
   print(f"Best service: {comparison['best_service']}")
   print(f"Average coverage: {comparison['average_coverage']:.1f}%")

**Research Applications:**
- Urban planning and policy
- Public health accessibility
- Transportation planning
- Environmental justice studies

Integration Examples
-------------------

**Multi-Domain Analysis:**

.. code-block:: python

   # Example: Combining all three analyzers
   def comprehensive_analysis(data):
       # Create base graph
       graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)
       result = grapher.get_graph_info(graph)
       
       # 1. Check for percolation behavior
       ranges = [20, 30, 40, 50, 60, 70]
       percolation = result.percolation_analyzer.analyze_percolation_threshold(data, ranges)
       
       # 2. Identify social structure
       roles = result.social_analyzer.identify_social_roles(graph)
       
       # 3. Assess spatial equity (if applicable)
       if has_service_data:
           accessibility = result.accessibility_analyzer.analyze_service_accessibility(
               data, service_data, "service", 100.0
           )
       
       return {
           'percolation': percolation,
           'social_roles': roles,
           'accessibility': accessibility if has_service_data else None
       }

**Performance Considerations:**

.. code-block:: python

   # For large datasets, sample data for analysis
   def analyze_large_dataset(large_data):
       # Sample for percolation analysis (computationally intensive)
       sample_size = min(200, len(large_data))
       sample_indices = np.random.choice(len(large_data), sample_size, replace=False)
       sample_data = large_data[sample_indices]
       
       # Use sample for analysis
       graph = grapher.make_graph("proximity", sample_data, proximity_thresh=50.0)
       result = grapher.get_graph_info(graph)
       
       # Percolation analysis on sample
       percolation = result.percolation_analyzer.analyze_percolation_threshold(
           sample_data, [20, 30, 40, 50]
       )
       
       # Social analysis can handle larger graphs
       full_graph = grapher.make_graph("proximity", large_data, proximity_thresh=50.0)
       full_result = grapher.get_graph_info(full_graph)
       roles = full_result.social_analyzer.identify_social_roles(full_graph)
       
       return percolation, roles

Best Practices
--------------

1. **Choose Appropriate Analyzers:**
   - Use `PercolationAnalyzer` for studying critical phenomena and phase transitions
   - Use `SocialNetworkAnalyzer` for role identification and temporal dynamics
   - Use `AccessibilityAnalyzer` for spatial planning and equity analysis

2. **Performance Optimization:**
   - Sample large datasets for computationally intensive analyses
   - Use appropriate graph types (proximity graphs for spatial analysis)
   - Consider memory usage for temporal analysis

3. **Parameter Selection:**
   - Choose meaningful distance thresholds based on your domain
   - Use domain knowledge to set realistic service standards
   - Test multiple parameter values to find critical ranges

4. **Result Interpretation:**
   - Consider statistical significance of results
   - Validate findings with domain expertise
   - Use visualization to communicate insights effectively

Advanced Topics
--------------

.. toctree::
   :maxdepth: 2

   percolation_analysis
   social_network_analysis  
   accessibility_analysis
   integration_examples
   performance_optimization
