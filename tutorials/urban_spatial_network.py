#!/usr/bin/env python3
"""
Research Tutorial: Urban Spatial Network Analysis with Graphizy

This tutorial demonstrates using Graphizy for urban planning and spatial analysis research,
focusing on accessibility, service coverage, and spatial equity in urban environments.

Research Applications:
1. Service accessibility analysis (hospitals, schools, parks)
2. Transportation network connectivity
3. Spatial equity and environmental justice
4. Urban sprawl and density patterns
5. Emergency service coverage optimization

Based on research methods from:
- Batty (2013). The New Science of Cities
- Hillier & Hanson (1984). The Social Logic of Space
- Harvey (1973). Social Justice and the City

Author: Charles Fosseprez
License: GPL-2.0-or-later
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

from graphizy import (
    Graphing, GraphizyConfig, generate_and_format_positions
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UrbanFeatureType(Enum):
    """Types of urban features for analysis"""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    SCHOOL = "school"
    HOSPITAL = "hospital"
    PARK = "park"
    TRANSIT_STOP = "transit_stop"


@dataclass
class UrbanAnalysisConfig:
    """Configuration for urban spatial analysis"""
    city_bounds: Tuple[float, float] = (4000.0, 3200.0)
    walking_distance: float = 800.0
    service_standards: Dict[str, float] = None
    
    def __post_init__(self):
        if self.service_standards is None:
            self.service_standards = {
                "school": 400.0,
                "hospital": 2000.0,
                "park": 600.0,
                "transit_stop": 500.0
            }


class UrbanSpatialAnalyzer:
    """
    Comprehensive analyzer for urban spatial patterns and accessibility
    """
    
    def __init__(self, config: UrbanAnalysisConfig):
        self.config = config
        
        # Setup Graphizy for urban analysis
        graphizy_config = GraphizyConfig(dimension=config.city_bounds)
        
        self.grapher = Graphing(config=graphizy_config)
        
        # Update drawing configuration
        self.grapher.update_config(
            drawing={
                "point_color": (100, 150, 255),
                "line_color": (255, 150, 100),
                "point_radius": 5,
                "line_thickness": 1
            }
        )
        
        # Initialize memory for temporal urban analysis
        self.grapher.init_memory_manager(
            max_memory_size=2000,
            track_edge_ages=True
        )
        
        # Urban features storage
        self.urban_features = {}
        self.analysis_results = {}
        
        logger.info(f"Initialized UrbanSpatialAnalyzer for {config.city_bounds[0]}x{config.city_bounds[1]}m city")
    
    def generate_urban_features(self, feature_counts: Dict[str, int] = None) -> Dict[str, np.ndarray]:
        """Generate realistic urban feature distributions"""
        if feature_counts is None:
            feature_counts = {
                "residential": 400,
                "commercial": 80,
                "school": 15,
                "hospital": 5,
                "park": 25,
                "transit_stop": 60
            }
        
        logger.info(f"Generating urban features")
        
        self.urban_features = {}
        
        for feature_type, count in feature_counts.items():
            if feature_type == "residential":
                positions = self._generate_residential_pattern(count)
            elif feature_type == "commercial":
                positions = self._generate_commercial_pattern(count)
            else:
                positions = self._generate_service_pattern(count, feature_type)
            
            # Format for Graphizy
            formatted_positions = np.column_stack((
                np.arange(count),
                positions
            ))
            
            self.urban_features[feature_type] = formatted_positions
        
        logger.info(f"Generated {sum(feature_counts.values())} urban features")
        return self.urban_features
    
    def _generate_residential_pattern(self, count: int) -> np.ndarray:
        """Generate residential areas in clusters"""
        positions = []
        num_clusters = max(3, count // 100)
        
        cluster_centers = np.random.uniform(
            [200, 200], 
            [self.config.city_bounds[0]-200, self.config.city_bounds[1]-200], 
            (num_clusters, 2)
        )
        
        for i in range(count):
            cluster_idx = np.random.randint(num_clusters)
            center = cluster_centers[cluster_idx]
            
            angle = np.random.uniform(0, 2*np.pi)
            radius = np.random.exponential(150)
            
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            
            x = np.clip(x, 0, self.config.city_bounds[0])
            y = np.clip(y, 0, self.config.city_bounds[1])
            
            positions.append([x, y])
        
        return np.array(positions)
    
    def _generate_commercial_pattern(self, count: int) -> np.ndarray:
        """Generate commercial areas along corridors"""
        positions = []
        
        for i in range(count):
            if np.random.random() < 0.7:  # 70% along corridors
                corridor = np.random.randint(2)
                if corridor == 0:  # Main street
                    x = np.random.uniform(0, self.config.city_bounds[0])
                    y = np.random.normal(self.config.city_bounds[1] * 0.4, 80)
                else:  # Central business district
                    x = np.random.normal(self.config.city_bounds[0] * 0.6, 150)
                    y = np.random.normal(self.config.city_bounds[1] * 0.5, 120)
            else:  # Random distribution
                x = np.random.uniform(0, self.config.city_bounds[0])
                y = np.random.uniform(0, self.config.city_bounds[1])
            
            x = np.clip(x, 0, self.config.city_bounds[0])
            y = np.clip(y, 0, self.config.city_bounds[1])
            positions.append([x, y])
        
        return np.array(positions)
    
    def _generate_service_pattern(self, count: int, feature_type: str) -> np.ndarray:
        """Generate service locations with appropriate distribution"""
        positions = []
        
        for i in range(count):
            if feature_type in ["school", "park"]:
                # More dispersed for good coverage
                x = np.random.uniform(0, self.config.city_bounds[0])
                y = np.random.uniform(0, self.config.city_bounds[1])
            else:
                # More centralized for hospitals and transit
                x = np.random.normal(self.config.city_bounds[0] * 0.5, 
                                   self.config.city_bounds[0] * 0.25)
                y = np.random.normal(self.config.city_bounds[1] * 0.5, 
                                   self.config.city_bounds[1] * 0.25)
                
                x = np.clip(x, 0, self.config.city_bounds[0])
                y = np.clip(y, 0, self.config.city_bounds[1])
            
            positions.append([x, y])
        
        return np.array(positions)
    
    def analyze_service_accessibility(self, service_type: str) -> Dict:
        """
        Analyze accessibility to urban services
        
        Args:
            service_type: Type of service to analyze
            
        Returns:
            Dictionary with accessibility analysis results
        """
        logger.info(f"Analyzing accessibility to {service_type} services")
        
        population_points = self.urban_features.get("residential")
        service_points = self.urban_features.get(service_type)
        
        if population_points is None or service_points is None:
            logger.warning(f"Missing data for {service_type} accessibility analysis")
            return {}
        
        # Get service standard distance
        service_distance = self.config.service_standards.get(service_type, self.config.walking_distance)
        
        results = {
            'service_type': service_type,
            'service_distance': service_distance,
            'population_count': len(population_points),
            'service_count': len(service_points),
            'coverage_statistics': {},
            'underserved_areas': []
        }
        
        # Simple accessibility calculation
        served_count = 0
        well_served_count = 0
        
        for pop_point in population_points:
            accessible_services = 0
            
            # Check distance to each service
            for service_point in service_points:
                distance = np.linalg.norm(pop_point[1:3] - service_point[1:3])
                if distance <= service_distance:
                    accessible_services += 1
            
            if accessible_services > 0:
                served_count += 1
            if accessible_services >= 2:
                well_served_count += 1
            
            if accessible_services == 0:
                results['underserved_areas'].append({
                    'position': pop_point[1:3].tolist(),
                    'accessible_services': accessible_services
                })
        
        # Coverage statistics
        results['coverage_statistics'] = {
            'served_population': served_count,
            'served_percentage': (served_count / len(population_points)) * 100,
            'well_served_population': well_served_count,
            'well_served_percentage': (well_served_count / len(population_points)) * 100,
            'underserved_count': len(results['underserved_areas'])
        }
        
        logger.info(f"Accessibility analysis complete: {results['coverage_statistics']['served_percentage']:.1f}% served")
        return results
    
    def analyze_network_integration(self) -> Dict:
        """Analyze integration and connectivity of urban networks"""
        logger.info("Analyzing urban network integration")
        
        results = {
            'feature_integration': {},
            'network_properties': {}
        }
        
        # Combine all features for network analysis
        all_features = []
        for positions in self.urban_features.values():
            # Sample to avoid too large networks
            sample_size = min(50, len(positions))
            sample_indices = np.random.choice(len(positions), sample_size, replace=False)
            all_features.extend(positions[sample_indices].tolist())
        
        if not all_features:
            return results
        
        all_features = np.array(all_features)
        
        # Create network at walking scale
        graph = self.grapher.make_graph("proximity", all_features, proximity_thresh=self.config.walking_distance)
        
        if graph.vcount() > 0:
            graph_info = self.grapher.get_graph_info(graph)
            
            results['network_properties']['walking'] = {
                'node_count': graph_info['vertex_count'],
                'edge_count': graph_info['edge_count'],
                'density': graph_info['density'],
                'mean_degree': graph_info['mean_degree'],
                'clustering': graph_info.get('clustering', 0),
                'component_count': graph_info['component_count']
            }
        
        # Feature integration analysis
        for feature_type in self.urban_features.keys():
            feature_positions = self.urban_features[feature_type]
            
            if len(feature_positions) > 1:
                integration_graph = self.grapher.make_graph(
                    "proximity", 
                    feature_positions, 
                    proximity_thresh=self.config.walking_distance
                )
                
                if integration_graph.vcount() > 0:
                    integration_info = self.grapher.get_graph_info(integration_graph)
                    
                    results['feature_integration'][feature_type] = {
                        'internal_connectivity': integration_info['density'],
                        'clustering': integration_info.get('clustering', 0),
                        'component_count': integration_info['component_count'],
                        'integration_score': integration_info['density']
                    }
        
        logger.info("Network integration analysis complete")
        return results
    
    def create_urban_visualizations(self, 
                                  accessibility_results: Dict,
                                  integration_results: Dict,
                                  output_dir: str = "urban_analysis_outputs"):
        """Create comprehensive urban analysis visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating urban visualizations in: {output_path}")
        
        # 1. Urban features overview
        if self.urban_features:
            sample_features = []
            for positions in self.urban_features.values():
                sample_size = min(30, len(positions))
                sample_indices = np.random.choice(len(positions), sample_size, replace=False)
                sample_features.extend(positions[sample_indices].tolist())
            
            if sample_features:
                sample_array = np.array(sample_features)
                overview_graph = self.grapher.make_graph("proximity", sample_array, proximity_thresh=300)
                overview_image = self.grapher.draw_graph(overview_graph)
                self.grapher.save_graph(overview_image, str(output_path / "urban_overview.png"))
        
        # 2. Statistical plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Urban Spatial Analysis Results', fontsize=14)
        
        # Feature distribution
        if self.urban_features:
            feature_counts = {k: len(v) for k, v in self.urban_features.items()}
            axes[0,0].pie(feature_counts.values(), 
                         labels=[k.replace('_', ' ').title() for k in feature_counts.keys()], 
                         autopct='%1.1f%%')
            axes[0,0].set_title('Urban Feature Distribution')
        
        # Service coverage
        if accessibility_results and 'coverage_statistics' in accessibility_results:
            coverage = accessibility_results['coverage_statistics']['served_percentage']
            underserved = 100 - coverage
            
            axes[0,1].bar(['Served', 'Underserved'], [coverage, underserved], 
                         color=['green', 'red'])
            axes[0,1].set_ylabel('Percentage (%)')
            axes[0,1].set_title(f'{accessibility_results["service_type"].title()} Coverage')
            axes[0,1].set_ylim(0, 100)
        
        # Network properties
        if 'network_properties' in integration_results:
            props = integration_results['network_properties'].get('walking', {})
            if props:
                metrics = ['Density', 'Clustering']
                values = [props.get('density', 0), props.get('clustering', 0)]
                
                axes[1,0].bar(metrics, values, color='purple')
                axes[1,0].set_ylabel('Value')
                axes[1,0].set_title('Network Properties')
        
        # Integration scores
        if 'feature_integration' in integration_results:
            features = list(integration_results['feature_integration'].keys())
            scores = [integration_results['feature_integration'][f].get('integration_score', 0) 
                     for f in features]
            
            axes[1,1].barh(range(len(features)), scores, color='orange')
            axes[1,1].set_xlabel('Integration Score')
            axes[1,1].set_title('Feature Integration')
            axes[1,1].set_yticks(range(len(features)))
            axes[1,1].set_yticklabels([f.replace('_', ' ').title() for f in features])
        
        plt.tight_layout()
        plt.savefig(output_path / "urban_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Urban visualizations completed")


def main():
    """Main urban analysis research workflow"""
    logger.info("Starting Urban Spatial Analysis Research Tutorial")
    
    # Configuration
    config = UrbanAnalysisConfig(
        city_bounds=(3000.0, 2500.0),
        walking_distance=600.0,
        service_standards={
            "school": 400.0,
            "hospital": 1500.0,
            "park": 500.0,
            "transit_stop": 400.0
        }
    )
    
    analyzer = UrbanSpatialAnalyzer(config)
    
    # Generate urban features
    logger.info("Generating urban feature distribution...")
    feature_counts = {
        "residential": 300,
        "commercial": 60,
        "school": 12,
        "hospital": 4,
        "park": 20,
        "transit_stop": 45
    }
    
    urban_features = analyzer.generate_urban_features(feature_counts)
    
    # Analyze service accessibility
    logger.info("Analyzing service accessibility...")
    school_accessibility = analyzer.analyze_service_accessibility("school")
    hospital_accessibility = analyzer.analyze_service_accessibility("hospital")
    park_accessibility = analyzer.analyze_service_accessibility("park")
    
    # Analyze network integration
    logger.info("Analyzing network integration...")
    integration_results = analyzer.analyze_network_integration()
    
    # Create visualizations
    logger.info("Creating urban analysis visualizations...")
    analyzer.create_urban_visualizations(
        school_accessibility,
        integration_results,
        "urban_analysis_outputs"
    )
    
    # Print summary
    print("\n" + "="*60)
    print("URBAN SPATIAL ANALYSIS COMPLETE")
    print("="*60)
    print(f"🏙️  Analyzed {sum(feature_counts.values())} urban features")
    print(f"🏫 School accessibility: {school_accessibility.get('coverage_statistics', {}).get('served_percentage', 0):.1f}% coverage")
    print(f"🏥 Hospital accessibility: {hospital_accessibility.get('coverage_statistics', {}).get('served_percentage', 0):.1f}% coverage")
    print(f"🌳 Park accessibility: {park_accessibility.get('coverage_statistics', {}).get('served_percentage', 0):.1f}% coverage")
    
    if integration_results and 'network_properties' in integration_results:
        walking_density = integration_results['network_properties'].get('walking', {}).get('density', 0)
        print(f"🚶 Walking network density: {walking_density:.4f}")
    
    print(f"🎨 Visualizations in: urban_analysis_outputs/")
    print("\nThis analysis provides evidence-based insights")
    print("for urban planning and spatial policy development!")


if __name__ == "__main__":
    main()
