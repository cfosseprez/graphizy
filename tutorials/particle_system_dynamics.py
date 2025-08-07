#!/usr/bin/env python3
"""
Research Tutorial: Particle System Dynamics and Network Analysis with Graphizy

This tutorial demonstrates using Graphizy for computational physics research,
specifically analyzing particle interactions, phase transitions, and emergent
network structures in many-body systems.

Research Applications:
1. Percolation theory and critical phenomena
2. Particle clustering and aggregation dynamics
3. Force network analysis in granular materials
4. Phase transition detection through network topology
5. Spatial correlation analysis in condensed matter

Based on research methods from:
- Stauffer & Aharony (1994). Introduction to Percolation Theory
- Christensen & Moloney (2005). Complexity and Criticality
- Torquato (2002). Random Heterogeneous Materials

Author: Charles Fosseprez
License: GPL-2.0-or-later
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from graphizy import (
    Graphing, GraphizyConfig, generate_and_format_positions,
    PercolationAnalyzer, PercolationResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParticleSystemConfig:
    """Configuration for particle system simulations"""
    box_size: Tuple[int, int] = (1000, 1000)
    num_particles: int = 500
    interaction_range: float = 50.0
    temperature: float = 1.0
    density: float = 0.1
    boundary_conditions: str = "periodic"


class ParticleSystemAnalyzer:
    """
    Comprehensive analyzer for particle systems and many-body physics
    
    Implements network-based analysis methods for understanding:
    - Percolation transitions
    - Clustering phenomena  
    - Force networks
    - Spatial correlations
    - Critical behavior
    """
    
    def __init__(self, config: ParticleSystemConfig):
        self.config = config
        
        # Setup Graphizy for particle analysis
        graphizy_config = GraphizyConfig(dimension=config.box_size)
        
        self.grapher = Graphing(config=graphizy_config)
        
        # Update drawing configuration with proper integer values
        try:
            self.grapher.update_config(
                drawing={
                    "point_color": (100, 255, 100),    # Green for particles
                    "line_color": (255, 200, 100),     # Orange for interactions
                    "point_radius": 6,
                    "line_thickness": 1
                }
            )
        except Exception as e:
            logger.warning(f"Could not update drawing config: {e}")
        
        # Initialize memory for temporal analysis
        try:
            self.grapher.init_memory_manager(
                max_memory_size=1000,
                track_edge_ages=True
            )
        except Exception as e:
            logger.warning(f"Could not initialize memory manager: {e}")
        
        self.analysis_history = []
        
        logger.info(f"Initialized ParticleSystemAnalyzer: {config.num_particles} particles in {config.box_size}")
    
    def generate_particle_configuration(self, config_type: str = "random") -> np.ndarray:
        """Generate different types of particle configurations"""
        logger.info(f"Generating {config_type} particle configuration")
        
        if config_type == "random":
            positions = np.random.uniform([0, 0], self.config.box_size, (self.config.num_particles, 2))
        elif config_type == "clustered":
            positions = self._generate_clustered_configuration()
        else:
            positions = np.random.uniform([0, 0], self.config.box_size, (self.config.num_particles, 2))
        
        return np.column_stack((np.arange(self.config.num_particles), positions))
    
    def _generate_clustered_configuration(self, num_clusters: int = 5) -> np.ndarray:
        """Generate clustered particle configuration"""
        positions = []
        particles_per_cluster = self.config.num_particles // num_clusters
        cluster_radius = 80.0
        
        # Generate cluster centers
        cluster_centers = np.random.uniform(
            [cluster_radius, cluster_radius],
            [self.config.box_size[0] - cluster_radius, self.config.box_size[1] - cluster_radius],
            (num_clusters, 2)
        )
        
        particle_id = 0
        for cluster_idx, center in enumerate(cluster_centers):
            cluster_size = particles_per_cluster
            if cluster_idx == num_clusters - 1:
                cluster_size = self.config.num_particles - particle_id
            
            for _ in range(cluster_size):
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, cluster_radius) * np.sqrt(np.random.random())
                
                x = center[0] + radius * np.cos(angle)
                y = center[1] + radius * np.sin(angle)
                
                positions.append([x, y])
                particle_id += 1
        
        return np.array(positions)
    
    def analyze_percolation_with_new_api(self, positions: np.ndarray, 
                                        interaction_ranges: List[float]) -> PercolationResult:
        """
        Analyze percolation using the new advanced analysis API
        """
        logger.info(f"Analyzing percolation with new API for {len(interaction_ranges)} ranges")
        
        # Get the percolation analyzer from any graph analysis result
        sample_graph = self.grapher.make_graph("proximity", positions, proximity_thresh=interaction_ranges[0])
        graph_info = self.grapher.get_graph_info(sample_graph)
        
        # Use the new percolation analyzer
        percolation_result = graph_info.percolation_analyzer.analyze_percolation_threshold(
            positions, interaction_ranges
        )
        
        # Analyze phase transition characteristics
        phase_transition = graph_info.percolation_analyzer.detect_phase_transition(percolation_result)
        
        logger.info(f"Percolation analysis complete. Critical range: {percolation_result.critical_range:.2f}")
        logger.info(f"Phase transition detected: {phase_transition['has_transition']}")
        
        return percolation_result, phase_transition
    
    def analyze_clustering_dynamics(self, positions: np.ndarray, timesteps: int = 30) -> Dict:
        """Analyze temporal clustering dynamics through network evolution"""
        logger.info(f"Analyzing clustering dynamics for {timesteps} timesteps")
        
        results = {
            'timesteps': [],
            'cluster_counts': [],
            'largest_cluster_evolution': [],
            'network_density_evolution': []
        }
        
        current_positions = positions.copy()
        
        for t in range(timesteps):
            try:
                # Create network at current timestep
                graph = self.grapher.make_graph("proximity", current_positions, 
                                              proximity_thresh=self.config.interaction_range)
                
                # Try to update memory system
                try:
                    self.grapher.update_memory_with_graph(graph)
                except Exception as e:
                    logger.warning(f"Memory update failed at timestep {t}: {e}")
                
                # Analyze clustering
                if graph.vcount() > 0:
                    clusters = graph.connected_components()
                    cluster_sizes = [len(cluster) for cluster in clusters]
                    
                    results['timesteps'].append(t)
                    results['cluster_counts'].append(len(cluster_sizes))
                    results['largest_cluster_evolution'].append(max(cluster_sizes) if cluster_sizes else 0)
                    
                    # Network density - using new API
                    graph_info = self.grapher.get_graph_info(graph)
                    results['network_density_evolution'].append(graph_info.density)
                
                # Evolve particle positions (simple brownian motion)
                if t < timesteps - 1:
                    displacement = np.random.normal(0, 1.5, (len(current_positions), 2))
                    current_positions[:, 1:3] += displacement
                    
                    # Apply boundary conditions
                    if self.config.boundary_conditions == "periodic":
                        current_positions[:, 1] = current_positions[:, 1] % self.config.box_size[0]
                        current_positions[:, 2] = current_positions[:, 2] % self.config.box_size[1]
                        
            except Exception as e:
                logger.warning(f"Clustering analysis failed at timestep {t}: {e}")
                continue
        
        logger.info("Clustering dynamics analysis complete")
        return results
    
    def create_visualizations(self, 
                            positions: np.ndarray,
                            percolation_result: PercolationResult,
                            phase_transition: Dict,
                            clustering_results: Dict,
                            output_dir: str = "particle_physics_outputs"):
        """Create comprehensive visualizations for physics analysis"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating physics visualizations in: {output_path}")
        
        try:
            # 1. Percolation visualization at critical point
            critical_range = percolation_result.critical_range
            if critical_range is not None and critical_range > 0:
                try:
                    critical_graph = self.grapher.make_graph("proximity", positions, proximity_thresh=critical_range)
                    if critical_graph.vcount() > 0:
                        critical_image = self.grapher.draw_graph(critical_graph)
                        self.grapher.save_graph(critical_image, str(output_path / "percolation_critical.png"))
                except Exception as e:
                    logger.warning(f"Could not create critical percolation visualization: {e}")
            
            # 2. Network visualizations at different scales
            try:
                ranges = percolation_result.interaction_ranges
                if len(ranges) >= 3:
                    # Small range network
                    small_range = ranges[2]
                    small_graph = self.grapher.make_graph("proximity", positions, proximity_thresh=small_range)
                    if small_graph.vcount() > 0:
                        small_image = self.grapher.draw_graph(small_graph)
                        self.grapher.save_graph(small_image, str(output_path / "network_small_range.png"))
                    
                    # Large range network
                    large_range = ranges[-3]
                    large_graph = self.grapher.make_graph("proximity", positions, proximity_thresh=large_range)
                    if large_graph.vcount() > 0:
                        large_image = self.grapher.draw_graph(large_graph)
                        self.grapher.save_graph(large_image, str(output_path / "network_large_range.png"))
                        
            except Exception as e:
                logger.warning(f"Could not create additional network visualizations: {e}")
            
            # 3. Enhanced plots with phase transition information
            if clustering_results['timesteps'] and len(clustering_results['timesteps']) > 1:
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    fig.suptitle('Particle System Clustering Dynamics', fontsize=14)
                    
                    timesteps = clustering_results['timesteps']
                    
                    # Cluster count evolution
                    axes[0,0].plot(timesteps, clustering_results['cluster_counts'], 'b-', linewidth=2)
                    axes[0,0].set_title('Total Cluster Count')
                    axes[0,0].set_xlabel('Timestep')
                    axes[0,0].set_ylabel('Number of Clusters')
                    axes[0,0].grid(True, alpha=0.3)
                    
                    # Largest cluster evolution
                    axes[0,1].plot(timesteps, clustering_results['largest_cluster_evolution'], 'r-', linewidth=2)
                    axes[0,1].set_title('Largest Cluster Size')
                    axes[0,1].set_xlabel('Timestep')
                    axes[0,1].set_ylabel('Particles in Largest Cluster')
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # Network density evolution
                    axes[1,0].plot(timesteps, clustering_results['network_density_evolution'], 'g-', linewidth=2)
                    axes[1,0].set_title('Network Density Evolution')
                    axes[1,0].set_xlabel('Timestep')
                    axes[1,0].set_ylabel('Network Density')
                    axes[1,0].grid(True, alpha=0.3)
                    
                    # Enhanced percolation plot with phase transition info
                    axes[1,1].plot(percolation_result.interaction_ranges, 
                                  percolation_result.percolation_probabilities, 'mo-', linewidth=2)
                    
                    if critical_range is not None:
                        axes[1,1].axvline(critical_range, color='red', linestyle='--', 
                                         label=f'Critical = {critical_range:.2f}')
                    
                    # Add phase transition information
                    if phase_transition['has_transition']:
                        transition_range = phase_transition['transition_range']
                        axes[1,1].axvline(transition_range, color='orange', linestyle=':', 
                                         label=f'Transition = {transition_range:.2f}')
                        
                        # Add sharpness annotation
                        sharpness = phase_transition['transition_sharpness']
                        axes[1,1].text(0.05, 0.95, f'Transition sharpness: {sharpness:.3f}', 
                                      transform=axes[1,1].transAxes, fontsize=10,
                                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    axes[1,1].set_title('Percolation Probability & Phase Transition')
                    axes[1,1].set_xlabel('Interaction Range')
                    axes[1,1].set_ylabel('P(percolation)')
                    axes[1,1].legend()
                    axes[1,1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(output_path / "physics_analysis.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    logger.warning(f"Could not create matplotlib plots: {e}")
            
            logger.info("Physics visualizations completed")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")


def main():
    """Main physics research workflow using new advanced analysis API"""
    logger.info("Starting Particle Physics Analysis Research Tutorial")
    
    try:
        # System configuration
        config = ParticleSystemConfig(
            box_size=(600, 600),
            num_particles=150,
            interaction_range=35.0,
            boundary_conditions="periodic"
        )
        
        analyzer = ParticleSystemAnalyzer(config)
        
        # Generate particle configuration
        logger.info("Generating particle configuration...")
        positions = analyzer.generate_particle_configuration("clustered")
        
        # Percolation analysis using new API
        logger.info("Analyzing percolation behavior with advanced API...")
        interaction_ranges = np.linspace(15, 60, 10)
        percolation_result, phase_transition = analyzer.analyze_percolation_with_new_api(
            positions, interaction_ranges
        )
        
        # Clustering dynamics
        logger.info("Analyzing clustering dynamics...")
        clustering_results = analyzer.analyze_clustering_dynamics(positions, timesteps=25)
        
        # Create enhanced visualizations
        logger.info("Creating physics visualizations...")
        analyzer.create_visualizations(
            positions, percolation_result, phase_transition, clustering_results, 
            "particle_physics_outputs"
        )
        
        # Enhanced summary with new API results
        print("\n" + "="*60)
        print("PARTICLE PHYSICS ANALYSIS COMPLETE")
        print("="*60)
        print(f"⚛️  Analyzed {config.num_particles} particles")
        print(f"🌐 Critical percolation range: {percolation_result.critical_range:.2f}")
        print(f"📊 Max cluster size: {max(percolation_result.largest_cluster_sizes)}")
        print(f"🔄 Phase transition detected: {phase_transition['has_transition']}")
        
        if phase_transition['has_transition']:
            print(f"📈 Transition sharpness: {phase_transition['transition_sharpness']:.3f}")
            print(f"🎯 Transition range: {phase_transition['transition_range']:.2f}")
        
        print(f"📈 Clustering timesteps: {len(clustering_results['timesteps'])}")
        print(f"🎨 Visualizations saved to: particle_physics_outputs/")
        print("\nThis analysis demonstrates network-based approaches")
        print("to computational physics and many-body systems!")
        print("🔬 New advanced analysis API provides:")
        print("   - Automated percolation threshold detection")
        print("   - Phase transition characterization")
        print("   - Enhanced visualization capabilities")
        
    except Exception as e:
        logger.error(f"Tutorial failed: {e}")
        print(f"❌ Tutorial failed: {e}")
        raise


if __name__ == "__main__":
    main()
