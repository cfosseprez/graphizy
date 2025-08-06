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
from typing import Dict, List, Tuple, Optional

from graphizy import Graphing, GraphizyConfig, generate_and_format_positions

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
    
    def analyze_proximity_networks(self, 
                                 trajectory: List[np.ndarray],
                                 proximity_threshold: float = 80.0) -> Dict:
        """
        Analyze social networks based on spatial proximity
        """
        logger.info(f"Analyzing proximity networks with {proximity_threshold}m threshold")
        
        results = {
            'timestep_graphs': [],
            'network_metrics': [],
            'individual_centralities': {},
            'group_cohesion': []
        }
        
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
                
                # Analyze graph properties
                graph_info = self.grapher.get_graph_info(graph)
                
                # Calculate behavioral metrics
                metrics = {
                    'timestep': timestep,
                    'edge_count': graph_info['edge_count'],
                    'density': graph_info['density'],
                    'mean_degree': graph_info['mean_degree'],
                    'clustering': graph_info.get('clustering', 0),
                    'components': graph_info['component_count']
                }
                
                # Individual centrality measures
                if graph.vcount() > 0:
                    try:
                        betweenness = graph.betweenness()
                        closeness = graph.closeness()
                        degree = graph.degree()
                        
                        # Store individual metrics
                        for i, animal_id in enumerate(graph.vs["id"]):
                            if animal_id not in results['individual_centralities']:
                                results['individual_centralities'][animal_id] = {
                                    'betweenness': [], 'closeness': [], 'degree': []
                                }
                            
                            results['individual_centralities'][animal_id]['betweenness'].append(betweenness[i])
                            results['individual_centralities'][animal_id]['closeness'].append(closeness[i])
                            results['individual_centralities'][animal_id]['degree'].append(degree[i])
                    
                    except Exception as e:
                        logger.warning(f"Centrality calculation failed at timestep {timestep}: {e}")
                
                results['timestep_graphs'].append(graph)
                results['network_metrics'].append(metrics)
                
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
                continue
        
        logger.info(f"Completed proximity network analysis for {len(results['network_metrics'])} timesteps")
        return results
    
    def identify_social_roles(self, results: Dict) -> Dict:
        """
        Identify social roles based on network position
        """
        logger.info("Identifying social roles from network centrality measures")
        
        social_roles = {}
        centralities = results['individual_centralities']
        
        if not centralities:
            logger.warning("No centrality data available for role identification")
            return social_roles
        
        # Calculate average centrality measures for each individual
        individual_stats = {}
        for animal_id, measures in centralities.items():
            individual_stats[animal_id] = {
                'avg_betweenness': np.mean(measures['betweenness']) if measures['betweenness'] else 0,
                'avg_closeness': np.mean(measures['closeness']) if measures['closeness'] else 0,
                'avg_degree': np.mean(measures['degree']) if measures['degree'] else 0
            }
        
        # Calculate thresholds (top 20% for each measure)
        all_betweenness = [stats['avg_betweenness'] for stats in individual_stats.values()]
        all_degree = [stats['avg_degree'] for stats in individual_stats.values()]
        
        betweenness_threshold = np.percentile(all_betweenness, 80) if all_betweenness else 0
        degree_threshold = np.percentile(all_degree, 80) if all_degree else 0
        
        # Assign roles based on centrality patterns
        for animal_id, stats in individual_stats.items():
            roles = []
            
            # Bridge/Broker: High betweenness centrality
            if stats['avg_betweenness'] >= betweenness_threshold:
                roles.append('bridge')
            
            # Hub/Popular: High degree centrality  
            if stats['avg_degree'] >= degree_threshold:
                roles.append('hub')
            
            # Peripheral: Low on all measures
            if (stats['avg_betweenness'] < np.percentile(all_betweenness, 20) and
                stats['avg_degree'] < np.percentile(all_degree, 20)):
                roles.append('peripheral')
            
            social_roles[animal_id] = {
                'roles': roles if roles else ['regular'],
                'stats': stats
            }
        
        logger.info(f"Identified social roles for {len(social_roles)} individuals")
        return social_roles
    
    def create_visualizations(self, results: Dict, output_dir: str = "animal_behavior_outputs"):
        """Create visualizations for the analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating visualizations in: {output_path}")
        
        try:
            # 1. Network evolution over time (sample timesteps)
            sample_timesteps = [0, len(results['timestep_graphs'])//2, len(results['timestep_graphs'])-1]
            
            for i, timestep in enumerate(sample_timesteps):
                if timestep < len(results['timestep_graphs']):
                    graph = results['timestep_graphs'][timestep]
                    image = self.grapher.draw_graph(graph)
                    self.grapher.save_graph(image, str(output_path / f"network_t{timestep:03d}.png"))
            
            # 2. Plot temporal metrics if matplotlib is available
            if len(results['network_metrics']) > 1:
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    fig.suptitle('Animal Social Network Analysis', fontsize=14)
                    
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
                        axes[1,0].plot(timesteps[:len(cohesion_distances)], cohesion_distances, 'm-', linewidth=2)
                        axes[1,0].set_title('Group Cohesion')
                        axes[1,0].set_xlabel('Timestep')
                        axes[1,0].set_ylabel('Mean Distance')
                        axes[1,0].grid(True, alpha=0.3)
                    
                    # Clear unused subplot
                    axes[1,1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(output_path / "temporal_analysis.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    logger.warning(f"Could not create matplotlib plots: {e}")
            
            logger.info("Visualizations completed")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")


def main():
    """Main research workflow demonstrating animal behavior analysis"""
    logger.info("Starting Animal Behavior Analysis Research Tutorial")
    
    try:
        # Initialize analyzer
        analyzer = AnimalBehaviorAnalyzer(territory_size=(1000, 800))
        
        # Simulate realistic animal movement
        logger.info("Simulating herd movement patterns...")
        trajectory = analyzer.simulate_herd_movement(
            num_animals=20,  # Reduced for stability
            num_timesteps=30,  # Reduced for speed
            cohesion_strength=0.25,
            exploration_rate=0.08
        )
        
        # Analyze proximity-based social networks
        logger.info("Analyzing proximity-based social networks...")
        network_results = analyzer.analyze_proximity_networks(
            trajectory, 
            proximity_threshold=120.0
        )
        
        # Identify social roles
        logger.info("Identifying individual social roles...")
        social_roles = analyzer.identify_social_roles(network_results)
        
        # Create visualizations
        logger.info("Creating research visualizations...")
        analyzer.create_visualizations(network_results, "animal_behavior_outputs")
        
        # Print summary
        print("\n" + "="*60)
        print("ANIMAL BEHAVIOR ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Analyzed {len(trajectory)} timesteps")
        print(f"🐾 Tracked {len(social_roles)} individual animals")
        print(f"🔗 Identified {len([r for r in social_roles.values() if 'bridge' in r['roles']])} social bridges")
        print(f"⭐ Found {len([r for r in social_roles.values() if 'hub' in r['roles']])} hub individuals")
        
        if network_results['network_metrics']:
            avg_density = np.mean([m['density'] for m in network_results['network_metrics']])
            print(f"📈 Average network density: {avg_density:.3f}")
        
        print(f"🎨 Visualizations saved to: animal_behavior_outputs/")
        print("\nThis analysis demonstrates how Graphizy enables")
        print("comprehensive behavioral ecology research!")
        
    except Exception as e:
        logger.error(f"Tutorial failed: {e}")
        print(f"❌ Tutorial failed: {e}")
        raise


if __name__ == "__main__":
    main()
