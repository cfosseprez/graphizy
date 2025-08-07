#!/usr/bin/env python3
"""
Research Tutorial: Animal Movement and Social Network Analysis with Graphizy

This tutorial demonstrates how to use Graphizy for behavioral ecology research,
specifically analyzing animal movement patterns and social interactions from
GPS tracking data.

Author: Charles Fosseprez
License: GPL-2.0-or-later
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any

from graphizy import (
    Graphing, GraphizyConfig, generate_and_format_positions,
    SocialNetworkAnalyzer, SocialRole
)

# Configure logging for research reproducibility
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class AnimalBehaviorAnalyzer:
    """
    Comprehensive analyzer for animal movement and social behavior
    """
    
    def __init__(self, territory_size: Tuple[int, int] = (1000, 800)):
        """Initialize analyzer for a given territory size"""
        self.territory_size = territory_size
        
        # Configure Graphizy for animal behavior analysis
        config = GraphizyConfig(dimension=territory_size)
        self.grapher = Graphing(config=config)
        
        # Try to update drawing configuration if available
        try:
            self.grapher.update_config(
                drawing={
                    "point_color": (255, 100, 100),
                    "line_color": (100, 100, 255),
                    "point_radius": 8,
                    "line_thickness": 2
                }
            )
        except Exception as e:
            logger.warning(f"Could not update drawing config: {e}")
        
        # Initialize memory system for temporal analysis
        try:
            self.grapher.init_memory_manager(
                max_memory_size=500,
                track_edge_ages=True
            )
            logger.info("Memory manager initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize memory manager: {e}")
        
        # Results storage
        self.analysis_results = {}
        
        logger.info(f"Initialized AnimalBehaviorAnalyzer for {territory_size[0]}x{territory_size[1]}m territory")
    
    def simulate_herd_movement(self, 
                              num_animals: int = 30, 
                              num_timesteps: int = 50,
                              cohesion_strength: float = 0.3,
                              exploration_rate: float = 0.1) -> List[np.ndarray]:
        """
        Simulate realistic herd movement using established behavioral models
        """
        logger.info(f"Simulating herd movement: {num_animals} animals, {num_timesteps} timesteps")
        
        # Initialize positions - animals start in a loose cluster
        center_x, center_y = self.territory_size[0] // 2, self.territory_size[1] // 2
        positions = np.random.normal([center_x, center_y], [100, 100], (num_animals, 2))
        
        # Ensure positions are within territory bounds
        positions[:, 0] = np.clip(positions[:, 0], 50, self.territory_size[0] - 50)
        positions[:, 1] = np.clip(positions[:, 1], 50, self.territory_size[1] - 50)
        
        trajectory = []
        
        for timestep in range(num_timesteps):
            # Calculate group center
            group_center = np.mean(positions, axis=0)
            
            # Individual movement with cohesion and exploration
            new_positions = positions.copy()
            
            for i in range(num_animals):
                # Vector toward group center (cohesion)
                to_center = group_center - positions[i]
                cohesion_force = to_center * cohesion_strength
                
                # Random exploration
                exploration_force = np.random.normal(0, exploration_rate * 10, 2)
                
                # Boundary avoidance
                boundary_force = np.zeros(2)
                if positions[i, 0] < 100:
                    boundary_force[0] = 50
                elif positions[i, 0] > self.territory_size[0] - 100:
                    boundary_force[0] = -50
                if positions[i, 1] < 100:
                    boundary_force[1] = 50
                elif positions[i, 1] > self.territory_size[1] - 100:
                    boundary_force[1] = -50
                
                # Combine forces
                total_force = cohesion_force + exploration_force + boundary_force
                new_positions[i] += total_force
            
            # Update positions
            positions = new_positions
            
            # Format for Graphizy (add animal IDs)
            formatted_data = np.column_stack((
                np.arange(num_animals),
                positions
            ))
            
            trajectory.append(formatted_data)
        
        logger.info(f"Generated trajectory with {len(trajectory)} timesteps")
        return trajectory
    
    def analyze_proximity_networks_with_new_api(self, 
                                               trajectory: List[np.ndarray],
                                               proximity_threshold: float = 80.0) -> Dict:
        """
        Analyze social networks using the new advanced analysis API
        """
        logger.info(f"Analyzing proximity networks with new API, threshold: {proximity_threshold}m")
        
        results = {
            'timestep_graphs': [],
            'network_metrics': [],
            'temporal_social_roles': {},
            'role_stability_scores': {},
            'group_cohesion': []
        }
        
        # Store graphs for temporal analysis
        graph_sequence = []
        
        for timestep, positions in enumerate(trajectory):
            try:
                # Create proximity graph
                graph = self.grapher.make_graph(
                    "proximity", 
                    positions, 
                    proximity_thresh=proximity_threshold
                )
                
                # Try to update memory system
                try:
                    self.grapher.update_memory_with_graph(graph)
                except Exception as e:
                    logger.warning(f"Memory update failed at timestep {timestep}: {e}")
                
                # Analyze graph properties using new API
                graph_info = self.grapher.get_graph_info(graph)
                
                # Calculate behavioral metrics using proper GraphAnalysisResult properties
                metrics = {
                    'timestep': timestep,
                    'edge_count': graph_info.edge_count,
                    'density': graph_info.density,
                    'vertex_count': graph_info.vertex_count,
                    'is_connected': graph_info.is_connected,
                    'num_components': graph_info.num_components
                }
                
                # Add clustering coefficient if available
                try:
                    metrics['clustering'] = graph_info.transitivity if graph_info.transitivity is not None else 0.0
                except Exception:
                    metrics['clustering'] = 0.0
                
                # Calculate mean degree
                if graph.vcount() > 0:
                    degrees = graph.degree()
                    metrics['mean_degree'] = np.mean(degrees) if degrees else 0.0
                else:
                    metrics['mean_degree'] = 0.0
                
                results['timestep_graphs'].append(graph)
                results['network_metrics'].append(metrics)
                graph_sequence.append(graph)
                
                # Group cohesion based on spatial clustering
                if len(positions) > 1:
                    distances = []
                    for i in range(len(positions)):
                        for j in range(i+1, len(positions)):
                            dist = np.linalg.norm(positions[i, 1:3] - positions[j, 1:3])
                            distances.append(dist)
                    
                    cohesion_metric = {
                        'mean_distance': np.mean(distances),
                        'std_distance': np.std(distances),
                        'max_distance': np.max(distances)
                    }
                    results['group_cohesion'].append(cohesion_metric)
            
            except Exception as e:
                logger.error(f"Analysis failed at timestep {timestep}: {e}")
                # Add empty results to maintain timestep alignment
                results['network_metrics'].append({
                    'timestep': timestep,
                    'edge_count': 0,
                    'density': 0.0,
                    'vertex_count': 0,
                    'is_connected': False,
                    'num_components': 0,
                    'clustering': 0.0,
                    'mean_degree': 0.0
                })
                continue
        
        # Perform temporal social role analysis using new API
        if graph_sequence:
            try:
                # Get social analyzer from first valid graph
                first_graph = next(g for g in graph_sequence if g.vcount() > 0)
                if first_graph:
                    first_graph_info = self.grapher.get_graph_info(first_graph)
                    social_analyzer = first_graph_info.social_analyzer
                    
                    # Track temporal roles
                    results['temporal_social_roles'] = social_analyzer.track_temporal_roles(graph_sequence)
                    
                    # Calculate role stability
                    results['role_stability_scores'] = social_analyzer.get_role_stability(
                        results['temporal_social_roles']
                    )
                    
                    logger.info(f"Temporal social role analysis complete for {len(results['temporal_social_roles'])} individuals")
            
            except Exception as e:
                logger.warning(f"Temporal social role analysis failed: {e}")
        
        logger.info(f"Completed proximity network analysis for {len(results['network_metrics'])} timesteps")
        return results
    
    def get_social_insights(self, results: Dict) -> Dict[str, Any]:
        """
        Extract behavioral insights from social network analysis
        """
        insights = {
            'consistent_bridges': [],
            'consistent_hubs': [],
            'stable_individuals': [],
            'dynamic_individuals': [],
            'leadership_patterns': {}
        }
        
        if not results['temporal_social_roles']:
            return insights
        
        # Analyze role consistency
        for animal_id, temporal_data in results['temporal_social_roles'].items():
            roles_over_time = temporal_data['roles']
            
            # Check for consistent roles
            bridge_count = sum(1 for roles in roles_over_time if 'bridge' in roles)
            hub_count = sum(1 for roles in roles_over_time if 'hub' in roles)
            
            bridge_consistency = bridge_count / len(roles_over_time)
            hub_consistency = hub_count / len(roles_over_time)
            
            if bridge_consistency > 0.6:
                insights['consistent_bridges'].append({
                    'animal_id': animal_id,
                    'consistency': bridge_consistency
                })
            
            if hub_consistency > 0.6:
                insights['consistent_hubs'].append({
                    'animal_id': animal_id,
                    'consistency': hub_consistency
                })
            
            # Stability analysis
            stability = results['role_stability_scores'].get(animal_id, 0.0)
            if stability > 0.8:
                insights['stable_individuals'].append({
                    'animal_id': animal_id,
                    'stability': stability
                })
            elif stability < 0.5:
                insights['dynamic_individuals'].append({
                    'animal_id': animal_id,
                    'stability': stability
                })
        
        # Sort by consistency/stability
        insights['consistent_bridges'].sort(key=lambda x: x['consistency'], reverse=True)
        insights['consistent_hubs'].sort(key=lambda x: x['consistency'], reverse=True)
        insights['stable_individuals'].sort(key=lambda x: x['stability'], reverse=True)
        insights['dynamic_individuals'].sort(key=lambda x: x['stability'])
        
        return insights
    
    def create_visualizations(self, results: Dict, insights: Dict, output_dir: str = "animal_behavior_outputs"):
        """Create enhanced visualizations using new analysis results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating enhanced visualizations in: {output_path}")
        
        try:
            # 1. Network evolution over time (sample timesteps)
            sample_timesteps = [0, len(results['timestep_graphs'])//2, len(results['timestep_graphs'])-1]
            
            for i, timestep in enumerate(sample_timesteps):
                if timestep < len(results['timestep_graphs']):
                    graph = results['timestep_graphs'][timestep]
                    if graph.vcount() > 0:
                        image = self.grapher.draw_graph(graph)
                        self.grapher.save_graph(image, str(output_path / f"network_t{timestep:03d}.png"))
            
            # 2. Enhanced plots with social role information
            if len(results['network_metrics']) > 1:
                try:
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    fig.suptitle('Animal Social Network Analysis with Role Dynamics', fontsize=14)
                    
                    timesteps = [m['timestep'] for m in results['network_metrics']]
                    densities = [m['density'] for m in results['network_metrics']]
                    edge_counts = [m['edge_count'] for m in results['network_metrics']]
                    
                    # Network density over time
                    axes[0,0].plot(timesteps, densities, 'b-', linewidth=2)
                    axes[0,0].set_title('Network Density Over Time')
                    axes[0,0].set_xlabel('Timestep')
                    axes[0,0].set_ylabel('Density')
                    axes[0,0].grid(True, alpha=0.3)
                    
                    # Number of connections over time
                    axes[0,1].plot(timesteps, edge_counts, 'r-', linewidth=2)
                    axes[0,1].set_title('Social Connections Over Time')
                    axes[0,1].set_xlabel('Timestep')
                    axes[0,1].set_ylabel('Number of Connections')
                    axes[0,1].grid(True, alpha=0.3)
                    
                    # Group cohesion
                    if results['group_cohesion']:
                        cohesion_distances = [gc['mean_distance'] for gc in results['group_cohesion']]
                        axes[0,2].plot(timesteps[:len(cohesion_distances)], cohesion_distances, 'm-', linewidth=2)
                        axes[0,2].set_title('Group Cohesion')
                        axes[0,2].set_xlabel('Timestep')
                        axes[0,2].set_ylabel('Mean Distance')
                        axes[0,2].grid(True, alpha=0.3)
                    
                    # Role consistency analysis
                    if insights['consistent_bridges'] or insights['consistent_hubs']:
                        bridge_ids = [b['animal_id'] for b in insights['consistent_bridges'][:5]]
                        bridge_consistencies = [b['consistency'] for b in insights['consistent_bridges'][:5]]
                        
                        hub_ids = [h['animal_id'] for h in insights['consistent_hubs'][:5]]
                        hub_consistencies = [h['consistency'] for h in insights['consistent_hubs'][:5]]
                        
                        x_pos = np.arange(max(len(bridge_ids), len(hub_ids)))
                        
                        if bridge_ids:
                            axes[1,0].bar(x_pos[:len(bridge_ids)] - 0.2, bridge_consistencies, 
                                         width=0.4, label='Bridges', color='orange')
                        if hub_ids:
                            axes[1,0].bar(x_pos[:len(hub_ids)] + 0.2, hub_consistencies, 
                                         width=0.4, label='Hubs', color='purple')
                        
                        axes[1,0].set_title('Role Consistency')
                        axes[1,0].set_xlabel('Individual Rank')
                        axes[1,0].set_ylabel('Consistency Score')
                        axes[1,0].legend()
                        axes[1,0].grid(True, alpha=0.3)
                    
                    # Stability scores
                    if results['role_stability_scores']:
                        stability_values = list(results['role_stability_scores'].values())
                        axes[1,1].hist(stability_values, bins=10, alpha=0.7, color='green')
                        axes[1,1].set_title('Role Stability Distribution')
                        axes[1,1].set_xlabel('Stability Score')
                        axes[1,1].set_ylabel('Number of Individuals')
                        axes[1,1].grid(True, alpha=0.3)
                    
                    # Summary statistics
                    axes[1,2].text(0.1, 0.9, f"Consistent Bridges: {len(insights['consistent_bridges'])}", 
                                  transform=axes[1,2].transAxes, fontsize=12)
                    axes[1,2].text(0.1, 0.8, f"Consistent Hubs: {len(insights['consistent_hubs'])}", 
                                  transform=axes[1,2].transAxes, fontsize=12)
                    axes[1,2].text(0.1, 0.7, f"Stable Individuals: {len(insights['stable_individuals'])}", 
                                  transform=axes[1,2].transAxes, fontsize=12)
                    axes[1,2].text(0.1, 0.6, f"Dynamic Individuals: {len(insights['dynamic_individuals'])}", 
                                  transform=axes[1,2].transAxes, fontsize=12)
                    
                    if results['network_metrics']:
                        avg_density = np.mean([m['density'] for m in results['network_metrics']])
                        axes[1,2].text(0.1, 0.4, f"Avg Network Density: {avg_density:.3f}", 
                                      transform=axes[1,2].transAxes, fontsize=12)
                    
                    axes[1,2].set_title('Analysis Summary')
                    axes[1,2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(output_path / "enhanced_temporal_analysis.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    logger.warning(f"Could not create matplotlib plots: {e}")
            
            logger.info("Enhanced visualizations completed")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")


def main():
    """Main research workflow demonstrating enhanced animal behavior analysis"""
    logger.info("Starting Enhanced Animal Behavior Analysis Research Tutorial")
    
    try:
        # Initialize analyzer
        analyzer = AnimalBehaviorAnalyzer(territory_size=(1000, 800))
        
        # Simulate realistic animal movement
        logger.info("Simulating herd movement patterns...")
        trajectory = analyzer.simulate_herd_movement(
            num_animals=20,
            num_timesteps=30,
            cohesion_strength=0.25,
            exploration_rate=0.08
        )
        
        # Analyze proximity-based social networks with new API
        logger.info("Analyzing proximity-based social networks with advanced API...")
        network_results = analyzer.analyze_proximity_networks_with_new_api(
            trajectory, 
            proximity_threshold=120.0
        )
        
        # Extract behavioral insights
        logger.info("Extracting behavioral insights...")
        social_insights = analyzer.get_social_insights(network_results)
        
        # Create enhanced visualizations
        logger.info("Creating enhanced research visualizations...")
        analyzer.create_visualizations(network_results, social_insights, "animal_behavior_outputs")
        
        # Enhanced summary with new API results
        print("\n" + "="*60)
        print("ENHANCED ANIMAL BEHAVIOR ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Analyzed {len(trajectory)} timesteps")
        print(f"🐾 Tracked {len(network_results['temporal_social_roles'])} individual animals")
        print(f"🔗 Identified {len(social_insights['consistent_bridges'])} consistent social bridges")
        print(f"⭐ Found {len(social_insights['consistent_hubs'])} consistent hub individuals")
        print(f"🎯 Detected {len(social_insights['stable_individuals'])} stable social roles")
        print(f"🔄 Identified {len(social_insights['dynamic_individuals'])} dynamic individuals")
        
        if network_results['network_metrics']:
            avg_density = np.mean([m['density'] for m in network_results['network_metrics']])
            print(f"📈 Average network density: {avg_density:.3f}")
        
        print(f"🎨 Enhanced visualizations saved to: animal_behavior_outputs/")
        print("\n🔬 New Advanced Analysis Features:")
        print("   - Temporal social role tracking")
        print("   - Role stability analysis")
        print("   - Behavioral consistency metrics")
        print("   - Leadership pattern detection")
        print("\nThis analysis demonstrates how Graphizy's advanced API")
        print("enables comprehensive behavioral ecology research!")
        
    except Exception as e:
        logger.error(f"Tutorial failed: {e}")
        print(f"❌ Tutorial failed: {e}")
        raise


if __name__ == "__main__":
    main()
