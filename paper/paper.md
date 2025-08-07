---
title: 'Graphizy: Memory-Enhanced Graph Construction for Computational Geometry and Temporal Network Analysis'
tags:
  - Python
  - graph theory
  - computational geometry
  - spatial analysis
  - temporal networks
  - Delaunay triangulation
  - network visualization
authors:
  - name: Charles Fosseprez
    orcid: 0009-0000-4524-0399 
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 28 July 2025
bibliography: paper.bib
---

# Graphizy: Memory-Enhanced Graph Construction for Computational Geometry and Temporal Network Analysis
---
title: 'Graphizy: Memory-Enhanced Graph Construction for Computational Geometry and Temporal Network Analysis'
tags:
  - Python
  - graph theory
  - computational geometry
  - spatial analysis
  - temporal networks
  - Delaunay triangulation
  - network visualization
authors:
  - name: Charles Fosseprez
    orcid: 0009-0000-4524-0399 
    affiliation: 1
affiliations:
 - name: Research Center of Mathematics for Social Creativity, Japan
   index: 1
date: 07 August 2025
bibliography: paper.bib
---


# Summary

Graphizy is a Python package that enables researchers to construct and analyze dynamic networks from spatial coordinate data with built-in support for temporal memory tracking. The package addresses the challenge of studying evolving spatial relationships by providing a unified interface for creating geometric graphs (Delaunay triangulation, proximity graphs, k-nearest neighbors, minimum spanning trees, Gabriel graphs) while maintaining a memory of historical connections across time steps. Unlike traditional graph libraries that treat each time point independently, Graphizy's temporal memory system allows researchers to track connection persistence, analyze network stability, and visualize the evolution of spatial relationships over time. The package is optimized for real-time applications and integrates seamlessly with existing scientific Python tools including igraph and NetworkX.

# Statement of Need

Analysis of spatial networks plays a crucial role across diverse scientific disciplines, from tracking animal social interactions [@krause2002living] to studying particle dynamics in physics and analyzing urban mobility patterns. However, existing graph analysis tools fall short in three critical areas that limit their effectiveness for spatial-temporal research.

First, widely-used libraries like NetworkX [@hagberg2008networkx] and igraph [@csardi2006igraph] treat spatial coordinates merely as node attributes rather than fundamental properties that determine graph structure. Researchers must manually implement geometric algorithms to construct spatially-meaningful networks, leading to fragmented workflows and potential inconsistencies across studies.

Second, these libraries lack native support for temporal dynamics. When analyzing evolving systems—such as moving particles, migrating animals, or changing social groups—researchers must repeatedly reconstruct graphs from scratch at each time step, discarding valuable information about connection persistence and network evolution. This approach prevents the analysis of temporal patterns that are often central to understanding dynamic systems.

Third, the computational geometry workflow typically requires integrating multiple specialized libraries (SciPy for triangulation, matplotlib for visualization, NetworkX for analysis), creating barriers for researchers and increasing the likelihood of implementation errors.

Graphizy addresses these limitations by providing a unified framework that treats spatial relationships as first-class citizens and incorporates a novel temporal memory system. The package enables researchers to track which connections persist over time, identify stable network structures, and analyze the evolution of spatial relationships—capabilities that are essential for understanding dynamic systems but difficult to achieve with existing tools.

The temporal memory system is particularly valuable for applications where connection stability matters more than instantaneous network structure. For example, in studying animal social behavior, brief separations should not be interpreted as broken social bonds, and in particle physics simulations, temporary disconnections due to noise should be distinguished from genuine structural changes.

# Key Features

Graphizy provides a comprehensive toolkit for spatial-temporal network analysis through several key innovations:

**Unified Spatial Graph Construction**: The package implements multiple geometric algorithms (Delaunay triangulation, proximity graphs, k-nearest neighbors, minimum spanning trees, Gabriel graphs) through a consistent API, eliminating the need to integrate multiple specialized libraries.

**Temporal Memory System**: The core innovation is a configurable memory layer that tracks edge persistence across time steps. This system allows researchers to distinguish between stable, long-term connections and transient interactions, enabling analysis of network stability and evolution patterns that are impossible with traditional snapshot-based approaches.

**Performance Optimization**: Built on OpenCV and optimized NumPy operations, the package achieves real-time performance with sub-50ms processing times for networks with 1000+ nodes, making it suitable for live analysis applications.

**Extensible Plugin Architecture**: A flexible plugin system allows researchers to implement custom graph construction algorithms while benefiting from the package's memory system and visualization capabilities.

**Seamless Integration**: The package works alongside existing tools rather than replacing them, providing easy conversion to NetworkX and igraph formats for advanced analysis while offering specialized capabilities for spatial-temporal research.

# Research Applications

The package provides detailed documentation for characterizing topological interactions within Paramecium populations in real-time, showcasing how the temporal memory system can track collective behavior patterns and spatial organization in microbial communities. This allows us to study (and perturb in real time at 20 FPS) swarms topological dynamics.

Graphizy includes comprehensive tutorials and examples demonstrating its application to real biological systems. These tutorials demonstrate the package's capability to handle high-frequency spatial data and extract meaningful interaction patterns from complex biological systems.

The package has been designed to support research across multiple domains where spatial relationships and their temporal evolution are critical, including behavioral ecology (animal social networks), physics and materials science (particle interaction networks), and urban planning (mobility pattern analysis). The comprehensive example suite and plugin architecture make it accessible for researchers across these diverse fields.

# Performance and Validation

Comprehensive benchmarking demonstrates significant performance advantages over general-purpose graph libraries for spatial applications. When compared to NetworkX implementations, Graphizy achieves 21× speedup for Delaunay triangulation analysis and 12× speedup for minimum spanning tree construction on 1000-node networks. The temporal memory system adds minimal computational overhead (typically <5ms per update), making it suitable for real-time applications.

The package includes extensive validation through unit tests (>50% code coverage), integration tests across multiple graph types, and performance benchmarks that demonstrate consistent scalability. All algorithms have been validated against established implementations to ensure correctness.

# Acknowledgements

We acknowledge the Sasakawa Science Research Grant number 2023-5025 (C.F.) and MEXT KAKENHI Grant numbers 21H05308 (Y.N.), 21H05310 (C.F.) for financial support. We thank the contributors to the igraph, NetworkX, NumPy, and OpenCV projects, whose excellent libraries form the foundation of Graphizy.

# References