# Graphizy Examples

This directory contains comprehensive examples demonstrating the capabilities of the Graphizy package.

## Example Structure

### 1. Basic Usage (`1_basic_usage.py`)
**Fundamental graph operations and visualizations**

Learn the core functionality:
- Creating Delaunay triangulations 
- Building proximity graphs
- Generating k-nearest neighbor graphs
- Configuration and styling options
- Basic graph analysis and statistics

```bash
python 1_basic_usage.py
```

**Key concepts covered:**
- Position generation with proper coordinate system
- Different graph construction methods
- Drawing configuration and customization
- Basic error handling

---

### 2. Graph Metrics (`2_graph_metrics.py`)
**Comprehensive graph analysis and metrics computation**

Master graph analysis techniques:
- Basic metrics (vertices, edges, density, connectivity)
- Centrality measures (degree, betweenness, closeness, PageRank)
- Structural analysis (clustering, assortativity, modularity)
- Using both graphizy API and direct igraph interface
- Comparing metrics across different graph types

```bash
python 2_graph_metrics.py
```

**Key concepts covered:**
- `grapher.get_graph_info()` for quick analysis
- `grapher.call_method()` for node-wise results
- `grapher.call_method_raw()` for direct igraph access
- Custom metric calculations
- Comparative analysis techniques

---

### 3. Advanced Memory (`3_advanced_memory.py`)
**Memory functionality and temporal graph analysis**

Explore memory-based graph evolution:
- Memory manager configuration and usage
- Different memory update strategies (proximity, Delaunay, custom)
- Edge aging and temporal visualization
- Memory statistics and analysis
- Advanced memory patterns and behaviors

```bash
python 3_advanced_memory.py
```

**Key concepts covered:**
- `MemoryManager` class usage
- Memory update strategies
- Temporal graph evolution
- Edge aging visualization
- Memory efficiency analysis

---

### 4. Integrated Brownian Motion (`4_integrated_brownian.py`)
**Complete simulation with dynamic networks and movie generation**

Experience advanced simulations:
- Brownian motion particle physics
- Real-time graph generation
- Movie creation from frame sequences
- Interactive command-line options
- Statistical analysis of dynamic networks

```bash
# Basic simulation
python 4_integrated_brownian.py --iterations 100 --particles 30

# Advanced options
python 4_integrated_brownian.py --no-memory --fps 15 --size 1000 800
python 4_integrated_brownian.py --memory-size 50 --iterations 200 --save-freq 2
```

**Key concepts covered:**
- Physics simulation integration
- Dynamic graph evolution
- Movie generation with OpenCV
- Command-line interface design
- Performance optimization for long simulations

---

## Quick Start

1. **Run all examples in sequence:**
```bash
cd examples
python 1_basic_usage.py
python 2_graph_metrics.py  
python 3_advanced_memory.py
python 4_integrated_brownian.py --iterations 50 --particles 20  # Quick demo
```

2. **Check output:**
```bash
ls output/  # View generated images and movies
```

## Requirements

All examples require:
- **Core**: numpy, opencv-python, python-igraph, scipy
- **Optional**: For movie creation in example 4, ensure OpenCV is properly installed

## Learning Path

**Beginner**: Start with `1_basic_usage.py` to understand core concepts

**Intermediate**: Move to `2_graph_metrics.py` to learn analysis techniques

**Advanced**: Explore `3_advanced_memory.py` for temporal graphs

**Expert**: Run `4_integrated_brownian.py` for complete simulations

## Output Files

Each example creates output in the `examples/output/` directory:
- **Images**: Individual graph visualizations (.jpg)
- **Movies**: Dynamic simulations (.mp4) 
- **Data**: Statistics and analysis results

## Troubleshooting

**Coordinate system issues**: Examples 1 and 2 demonstrate proper coordinate alignment

**Memory errors**: Example 3 shows memory management best practices  

**Performance issues**: Example 4 includes optimization techniques for large simulations

**Import errors**: Check that all dependencies are installed:
```bash
pip install opencv-python python-igraph scipy numpy
```

## Extending the Examples

Each example is designed to be educational and extensible:

- **Modify parameters** to see how they affect results
- **Add your own metrics** using the patterns shown in example 2
- **Create custom memory strategies** following example 3
- **Implement new physics** based on example 4

## API Reference

For detailed API documentation, see the main graphizy documentation.

Key classes and methods demonstrated:
- `Graphing`: Main graph creation and analysis
- `GraphizyConfig`: Configuration management  
- `MemoryManager`: Temporal graph functionality
- `generate_positions()`: Position generation utilities

---

*Happy graphing! 📊🎨*
