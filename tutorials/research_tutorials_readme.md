# Graphizy Research Tutorials

This collection provides comprehensive, research-focused tutorials demonstrating how to use Graphizy for scientific analysis across multiple domains. Each tutorial implements established methodologies from the respective research fields and produces publication-ready results.

## 📚 Available Tutorials

### 1. Animal Movement and Social Network Analysis
**File**: `animal_movement_analysis.py`  
**Research Domain**: Behavioral Ecology, Ethology  
**Duration**: ~5-10 minutes  

**Key Methods Implemented**:
- Reynolds flocking models for realistic movement simulation
- Proximity-based social network construction  
- Social role identification (bridges, hubs, peripheral individuals)
- Temporal pattern analysis and connection persistence
- Group cohesion metrics and leadership detection

**Research Applications**:
- Analysis of GPS tracking data from wild animal populations
- Social structure identification in group-living species
- Temporal dynamics of animal social networks
- Conservation applications for population management

**Usage**:
```bash
cd tutorials
python animal_movement_analysis.py
```

**Output**: 
- Behavioral analysis with social roles identification
- Network visualizations showing social structures
- Temporal pattern analysis
- Results saved to `animal_behavior_outputs/`

---

### 2. Particle Physics and Many-Body Systems
**File**: `particle_system_dynamics.py`  
**Research Domain**: Statistical Physics, Condensed Matter, Materials Science  
**Duration**: ~3-7 minutes  

**Key Methods Implemented**:
- Percolation theory with critical threshold detection
- Clustering dynamics and aggregation analysis
- Phase transition detection using network topology
- Statistical mechanics order parameters
- Network-based critical phenomena analysis

**Research Applications**:
- Analysis of particle simulations and experimental data
- Percolation studies in porous materials
- Phase transitions in many-body systems
- Network analysis of particle interactions
- Critical behavior in condensed matter systems

**Usage**:
```bash
cd tutorials
python particle_system_dynamics.py
```

**Output**:
- Physics research analysis with percolation behavior
- Phase transition detection
- Clustering dynamics over time
- Results saved to `particle_physics_outputs/`

---

### 3. Urban Spatial Analysis and Planning
**File**: `urban_spatial_network.py`  
**Research Domain**: Urban Planning, Geography, Spatial Analysis  
**Duration**: ~4-8 minutes  

**Key Methods Implemented**:
- Multi-scale accessibility analysis (walking distance)
- Service coverage optimization analysis
- Network integration and connectivity analysis
- Urban form and spatial pattern analysis
- Evidence-based planning recommendations

**Research Applications**:
- Urban planning and policy development
- Service facility location optimization
- Transportation network analysis
- Smart city spatial analytics
- Accessibility and equity studies

**Usage**:
```bash
cd tutorials
python urban_spatial_network.py
```

**Output**:
- Urban planning analysis with accessibility metrics
- Service coverage assessment
- Network integration analysis
- Results saved to `urban_analysis_outputs/`

## 🛠️ Technical Features

### Common Research Capabilities
All tutorials demonstrate:

- **Memory-Enhanced Analysis**: Temporal tracking of network evolution
- **Statistical Rigor**: Proper statistical measures and significance testing
- **Publication Quality**: Research-grade outputs suitable for academic publication
- **Reproducible Workflows**: Complete, documented analysis pipelines
- **Domain-Specific Methods**: Established research methodologies from literature

### Advanced Graphizy Features Showcased

#### Memory Systems
```python
# Temporal network analysis with connection persistence
grapher.init_memory_manager(max_memory_size=500, track_edge_ages=True)
# Automatically tracks historical connections
```

#### Multiple Graph Types
```python
# Comparative analysis across graph types
graph_types = ["delaunay", "proximity", "knn", "mst"]
results = {gt: grapher.make_graph(gt, data) for gt in graph_types}
```

#### Network Analysis Integration
```python
# Comprehensive graph analysis
graph_info = grapher.get_graph_info(graph)
centrality_analysis = graph_info.get_centrality_analysis()
```

## 📊 Research Outputs

### Reports Generated
Each tutorial produces comprehensive research outputs:

- **Statistical Analysis**: Network metrics, centrality measures, clustering coefficients
- **Visualizations**: Research-grade figures and plots with publication quality
- **Temporal Analysis**: Evolution patterns and stability metrics
- **Domain-Specific Insights**: Research implications and future directions

### File Outputs
- Network structure visualizations (PNG format)
- Statistical analysis plots (PNG format)
- Research logs with timestamped analysis steps
- Structured data output for further analysis

## 🚀 Getting Started

### Prerequisites
```bash
# Ensure Graphizy is installed
pip install graphizy

# Additional packages for research features
pip install matplotlib pandas scipy
```

### Quick Start
```bash
# Navigate to tutorials directory
cd tutorials

# Run any tutorial
python animal_movement_analysis.py
python particle_system_dynamics.py
python urban_spatial_network.py

# Check outputs
ls *_outputs/
```

### Customization
Each tutorial is designed to be easily adapted:

```python
# Modify configuration for your research
config = AnalysisConfig(
    system_size=(2000, 1500),
    interaction_range=75.0,
    analysis_duration=500
)

# Use your own data
your_data = load_your_research_data()
results = analyzer.analyze_system(your_data)
```

## 🔬 Research Methodology

### Scientific Rigor
- **Literature-Based**: Methods based on established research practices
- **Validated Approaches**: Using peer-reviewed methodologies
- **Statistical Soundness**: Proper statistical measures and interpretations
- **Reproducible**: Complete code with clear documentation

### Performance Considerations
- **Scalability**: Tested with realistic data volumes
- **Efficiency**: Optimized for research-scale datasets
- **Memory Management**: Efficient handling of temporal data

## 📈 Use in Research

### Academic Applications
These tutorials have been designed for:
- **Graduate Research**: Thesis and dissertation projects
- **Academic Papers**: Methods sections and analysis workflows
- **Conference Presentations**: Research methodology demonstrations

### Industry Applications
- **R&D Projects**: Product development and optimization
- **Consulting**: Spatial analysis and network consulting
- **Environmental Consulting**: Ecological network analysis

### Educational Use
- **Course Material**: Network analysis and spatial statistics courses
- **Workshops**: Research methods training
- **Tutorials**: Graduate student training materials

## 🎯 Tutorial Comparison

| Tutorial | Data Type | Network Types | Analysis Focus | Output Type |
|----------|-----------|---------------|----------------|-------------|
| **Animal Behavior** | Movement trajectories | Proximity, Memory | Social roles, Temporal | Behavioral insights |
| **Particle Physics** | Particle positions | Proximity, MST | Phase transitions, Clustering | Physical phenomena |
| **Urban Planning** | Spatial features | Proximity, Accessibility | Service coverage, Equity | Policy recommendations |

## 🛠️ Advanced Usage

### Batch Processing
```python
# Process multiple datasets
datasets = ["dataset1.csv", "dataset2.csv", "dataset3.csv"]
results = {}

for dataset in datasets:
    data = load_data(dataset)
    analyzer = YourAnalyzer()
    results[dataset] = analyzer.full_analysis(data)
```

### Integration with Other Tools
```python
# Export to NetworkX for advanced analysis
import networkx as nx
nx_graph = grapher.to_networkx(your_graph)

# Export data for statistical analysis
import pandas as pd
results_df = pd.DataFrame(analysis_results)
results_df.to_csv("research_results.csv")
```

## 📚 Further Reading

### Graphizy Documentation
- [API Reference](https://graphizy.readthedocs.io/en/latest/api.html)
- [User Guide](https://graphizy.readthedocs.io/en/latest/guide.html)
- [Advanced Features](https://graphizy.readthedocs.io/en/latest/advanced.html)

### Research Methods References
- **Network Analysis**: Newman, M. E. J. (2010). Networks: An Introduction
- **Spatial Analysis**: Fotheringham, A. S. et al. (2000). Quantitative Geography  
- **Statistical Physics**: Barrat, A. et al. (2008). Dynamical Processes on Complex Networks
- **Urban Analysis**: Batty, M. (2013). The New Science of Cities
- **Animal Behavior**: Krause, J. & Ruxton, G. D. (2002). Living in groups

## 🤝 Contributing

We welcome contributions of additional research tutorials! 

### Guidelines for New Tutorials
1. **Research Focus**: Address a specific research domain with established methods
2. **Literature Base**: Reference relevant academic literature  
3. **Complete Workflow**: From data generation/loading to final results
4. **Documentation**: Comprehensive docstrings and comments
5. **Output Quality**: Research-grade results suitable for publication

### Template Structure
```python
"""
Research Tutorial: [Domain] Analysis with Graphizy

Research Questions:
1. [Primary research question]
2. [Secondary questions]

Methods Based On:
- [Citation 1]
- [Citation 2]
"""

class DomainAnalyzer:
    def __init__(self, config):
        # Setup analysis framework
        
    def analyze_method1(self, data):
        # Primary analysis method
        
    def create_visualizations(self, results):
        # Create publication-ready visualizations
        
def main():
    # Complete research workflow
```

## 🎉 Success Stories

These tutorials demonstrate Graphizy's capabilities for:
- **Cross-disciplinary research**: Same tools, different domains
- **Scalable analysis**: From small datasets to large-scale studies
- **Publication-ready results**: Professional outputs for academic use
- **Reproducible science**: Complete workflows with clear documentation

---

*These tutorials demonstrate the research capabilities of Graphizy and provide starting points for domain-specific analysis workflows.*

*For questions or contributions, please see the main Graphizy documentation or submit issues on GitHub.*
