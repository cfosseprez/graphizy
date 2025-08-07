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
    Graphing, GraphizyConfig, generate_and_format_positions,
    AccessibilityAnalyzer, AccessibilityResult
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
    city_bounds: Tuple[int, int] = (4000, 3200)
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
        
        # Update drawing configuration with proper integer values
        try:
            self.grapher.update_config(
                drawing={
                    "point_color": (100, 150, 255),
                    "line_color": (255, 150, 100),
                    "point_radius": 5,
                    "line_thickness": 1
                }
            )
        except Exception as e:
            logger.warning(f"Could not update drawing config: {e}")
        
        # Initialize memory for temporal urban analysis
        try:
            self.grapher.init_memory_manager(
                max_memory_size=2000,
                track_edge_ages=True
            )
        except Exception as e:
            logger.warning(f"Could not initialize memory manager: {e}")
        
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
    
    def analyze_accessibility_with_new_api(self, service_types: List[str]) -> Dict[str, AccessibilityResult]:
        """
        Analyze accessibility using the new advanced analysis API
        """
        logger.info(f"Analyzing accessibility with new API for services: {service_types}")
        
        population_points = self.urban_features.get("residential")
        if population_points is None:
            logger.warning("No residential data available for accessibility analysis")
            return {}
        
        accessibility_results = {}
        
        for service_type in service_types:
            service_points = self.urban_features.get(service_type)
            if service_points is None:
                logger.warning(f"No {service_type} data available")
                continue
            
            service_distance = self.config.service_standards.get(service_type, self.config.walking_distance)
            
            # Get accessibility analyzer from any graph analysis result
            sample_graph = self.grapher.make_graph("proximity", population_points[:10], proximity_thresh=100)
            graph_info = self.grapher.get_graph_info(sample_graph)
            
            # Use the new accessibility analyzer
            accessibility_result = graph_info.accessibility_analyzer.analyze_service_accessibility(
                population_points, service_points, service_type, service_distance
            )
            
            # Identify service gaps
            service_gaps = graph_info.accessibility_analyzer.identify_service_gaps(
                accessibility_result, cluster_distance=200.0
            )
            
            # Store results with gap analysis
            accessibility_results[service_type] = {
                'accessibility': accessibility_result,
                'service_gaps': service_gaps
            }
            
            logger.info(f"{service_type}: {accessibility_result.get_coverage_percentage():.1f}% coverage, "
                       f"{len(service_gaps)} service gaps identified")
        
        return accessibility_results
    
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
        try:
            graph = self.grapher.make_graph("proximity", all_features, proximity_thresh=self.config.walking_distance)
            
            if graph.vcount() > 0:
                graph_info = self.grapher.get_graph_info(graph)
                
                # Calculate mean degree manually
                degrees = graph.degree()
                mean_degree = np.mean(degrees) if degrees else 0.0
                
                results['network_properties']['walking'] = {
                    'node_count': graph_info.vertex_count,
                    'edge_count': graph_info.edge_count,
                    'density': graph_info.density,
                    'mean_degree': mean_degree,
                    'clustering': graph_info.transitivity if graph_info.transitivity is not None else 0.0,
                    'component_count': graph_info.num_components
                }
        except Exception as e:
            logger.warning(f"Walking network analysis failed: {e}")
        
        # Feature integration analysis
        for feature_type in self.urban_features.keys():
            feature_positions = self.urban_features[feature_type]
            
            if len(feature_positions) > 1:
                try:
                    integration_graph = self.grapher.make_graph(
                        "proximity", 
                        feature_positions, 
                        proximity_thresh=self.config.walking_distance
                    )
                    
                    if integration_graph.vcount() > 0:
                        integration_info = self.grapher.get_graph_info(integration_graph)
                        
                        results['feature_integration'][feature_type] = {
                            'internal_connectivity': integration_info.density,
                            'clustering': integration_info.transitivity if integration_info.transitivity is not None else 0.0,
                            'component_count': integration_info.num_components,
                            'integration_score': integration_info.density
                        }
                except Exception as e:
                    logger.warning(f"Integration analysis failed for {feature_type}: {e}")
        
        logger.info("Network integration analysis complete")
        return results
    
    def perform_comparative_analysis(self, accessibility_results: Dict[str, Dict]) -> Dict[str, any]:
        """
        Perform comparative analysis across services using new API
        """
        logger.info("Performing comparative accessibility analysis")
        
        # Extract AccessibilityResult objects
        accessibility_list = [result['accessibility'] for result in accessibility_results.values()]
        
        if not accessibility_list:
            return {}
        
        # Get accessibility analyzer and perform comparison
        sample_graph = self.grapher.make_graph("proximity", 
                                             self.urban_features["residential"][:10], 
                                             proximity_thresh=100)
        graph_info = self.grapher.get_graph_info(sample_graph)
        
        comparison = graph_info.accessibility_analyzer.compare_accessibility(accessibility_list)
        
        # Add service gap analysis
        total_gaps = sum(len(result['service_gaps']) for result in accessibility_results.values())
        critical_gaps = []
        
        for service_type, result in accessibility_results.items():
            for gap in result['service_gaps']:
                if gap['severity'] > 0.1:  # More than 10% of underserved population
                    critical_gaps.append({
                        'service_type': service_type,
                        'gap_info': gap
                    })
        
        comparison['total_service_gaps'] = total_gaps
        comparison['critical_gaps'] = critical_gaps
        comparison['equity_analysis'] = {
            service_type: result['accessibility'].get_equity_score() 
            for service_type, result in accessibility_results.items()
        }
        
        return comparison
    
    def create_urban_visualizations(self, 
                                  accessibility_results: Dict,
                                  integration_results: Dict,
                                  comparative_analysis: Dict,
                                  output_dir: str = "urban_analysis_outputs"):
        """Create comprehensive urban analysis visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Creating enhanced urban visualizations in: {output_path}")
        
        try:
            # 1. Urban features overview
            if self.urban_features:
                sample_features = []
                for positions in self.urban_features.values():
                    sample_size = min(30, len(positions))
                    sample_indices = np.random.choice(len(positions), sample_size, replace=False)
                    sample_features.extend(positions[sample_indices].tolist())
                
                if sample_features:
                    sample_array = np.array(sample_features)
                    try:
                        overview_graph = self.grapher.make_graph("proximity", sample_array, proximity_thresh=300)
                        if overview_graph.vcount() > 0:
                            overview_image = self.grapher.draw_graph(overview_graph)
                            self.grapher.save_graph(overview_image, str(output_path / "urban_overview.png"))
                    except Exception as e:
                        logger.warning(f"Could not create urban overview visualization: {e}")
            
            # 2. Enhanced statistical plots
            try:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle('Enhanced Urban Spatial Analysis Results', fontsize=14)
                
                # Feature distribution
                if self.urban_features:
                    feature_counts = {k: len(v) for k, v in self.urban_features.items()}
                    axes[0,0].pie(feature_counts.values(), 
                                 labels=[k.replace('_', ' ').title() for k in feature_counts.keys()], 
                                 autopct='%1.1f%%')
                    axes[0,0].set_title('Urban Feature Distribution')
                
                # Service coverage comparison
                if accessibility_results:
                    services = list(accessibility_results.keys())
                    coverages = [accessibility_results[s]['accessibility'].get_coverage_percentage() 
                               for s in services]
                    
                    bars = axes[0,1].bar(services, coverages, color=['green', 'blue', 'orange', 'purple'])
                    axes[0,1].set_ylabel('Coverage Percentage (%)')
                    axes[0,1].set_title('Service Coverage Comparison')
                    axes[0,1].set_ylim(0, 100)
                    axes[0,1].tick_params(axis='x', rotation=45)
                    
                    # Add coverage threshold line
                    axes[0,1].axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Target: 80%')
                    axes[0,1].legend()
                
                # Equity scores
                if 'equity_analysis' in comparative_analysis:
                    equity_services = list(comparative_analysis['equity_analysis'].keys())
                    equity_scores = list(comparative_analysis['equity_analysis'].values())
                    
                    axes[0,2].barh(equity_services, equity_scores, color='teal')
                    axes[0,2].set_xlabel('Equity Score')
                    axes[0,2].set_title('Spatial Equity Analysis')
                    axes[0,2].set_xlim(0, 1)
                
                # Network properties
                if 'network_properties' in integration_results:
                    props = integration_results['network_properties'].get('walking', {})
                    if props:
                        metrics = ['Density', 'Clustering']
                        values = [props.get('density', 0), props.get('clustering', 0)]
                        
                        axes[1,0].bar(metrics, values, color='purple')
                        axes[1,0].set_ylabel('Value')
                        axes[1,0].set_title('Network Properties')
                
                # Service gaps analysis
                if accessibility_results:
                    gap_counts = [len(result['service_gaps']) for result in accessibility_results.values()]
                    services = list(accessibility_results.keys())
                    
                    bars = axes[1,1].bar(services, gap_counts, color='red', alpha=0.7)
                    axes[1,1].set_ylabel('Number of Service Gaps')
                    axes[1,1].set_title('Service Gap Distribution')
                    axes[1,1].tick_params(axis='x', rotation=45)
                
                # Summary statistics
                summary_text = []
                if 'best_service' in comparative_analysis:
                    summary_text.append(f"Best Coverage: {comparative_analysis['best_service']}")
                if 'worst_service' in comparative_analysis:
                    summary_text.append(f"Needs Improvement: {comparative_analysis['worst_service']}")
                if 'average_coverage' in comparative_analysis:
                    summary_text.append(f"Average Coverage: {comparative_analysis['average_coverage']:.1f}%")
                if 'total_service_gaps' in comparative_analysis:
                    summary_text.append(f"Total Service Gaps: {comparative_analysis['total_service_gaps']}")
                if 'critical_gaps' in comparative_analysis:
                    summary_text.append(f"Critical Gaps: {len(comparative_analysis['critical_gaps'])}")
                
                for i, text in enumerate(summary_text):
                    axes[1,2].text(0.1, 0.9 - i*0.15, text, transform=axes[1,2].transAxes, 
                                  fontsize=11, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                axes[1,2].set_title('Analysis Summary')
                axes[1,2].axis('off')
                
                plt.tight_layout()
                plt.savefig(output_path / "enhanced_urban_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logger.warning(f"Could not create matplotlib plots: {e}")
            
            logger.info("Enhanced urban visualizations completed")
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")


def main():
    """Main urban analysis research workflow using new advanced analysis API"""
    logger.info("Starting Enhanced Urban Spatial Analysis Research Tutorial")
    
    try:
        # Configuration
        config = UrbanAnalysisConfig(
            city_bounds=(3000, 2500),
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
        
        # Analyze service accessibility with new API
        logger.info("Analyzing service accessibility with advanced API...")
        accessibility_results = analyzer.analyze_accessibility_with_new_api(
            ["school", "hospital", "park", "transit_stop"]
        )
        
        # Analyze network integration
        logger.info("Analyzing network integration...")
        integration_results = analyzer.analyze_network_integration()
        
        # Perform comparative analysis
        logger.info("Performing comparative analysis...")
        comparative_analysis = analyzer.perform_comparative_analysis(accessibility_results)
        
        # Create enhanced visualizations
        logger.info("Creating enhanced urban analysis visualizations...")
        analyzer.create_urban_visualizations(
            accessibility_results,
            integration_results,
            comparative_analysis,
            "urban_analysis_outputs"
        )
        
        # Enhanced summary with new API results
        print("\n" + "="*60)
        print("ENHANCED URBAN SPATIAL ANALYSIS COMPLETE")
        print("="*60)
        print(f"🏙️  Analyzed {sum(feature_counts.values())} urban features")
        
        for service_type, result in accessibility_results.items():
            coverage = result['accessibility'].get_coverage_percentage()
            equity = result['accessibility'].get_equity_score()
            gaps = len(result['service_gaps'])
            print(f"🏢 {service_type.title()}: {coverage:.1f}% coverage, equity={equity:.3f}, {gaps} gaps")
        
        if integration_results and 'network_properties' in integration_results:
            walking_density = integration_results['network_properties'].get('walking', {}).get('density', 0)
            print(f"🚶 Walking network density: {walking_density:.4f}")
        
        print(f"\n🔬 Advanced Analysis Results:")
        if 'best_service' in comparative_analysis:
            print(f"   ✅ Best service coverage: {comparative_analysis['best_service']}")
        if 'worst_service' in comparative_analysis:
            print(f"   ❌ Needs improvement: {comparative_analysis['worst_service']}")
        if 'total_service_gaps' in comparative_analysis:
            print(f"   🕳️  Total service gaps identified: {comparative_analysis['total_service_gaps']}")
        if 'critical_gaps' in comparative_analysis:
            print(f"   🚨 Critical gaps requiring intervention: {len(comparative_analysis['critical_gaps'])}")
        
        print(f"\n🎨 Enhanced visualizations in: urban_analysis_outputs/")
        print("\n🔬 New Advanced Analysis Features:")
        print("   - Automated service gap identification")
        print("   - Spatial equity scoring")
        print("   - Comparative accessibility analysis")
        print("   - Critical gap prioritization")
        print("\nThis analysis provides evidence-based insights")
        print("for urban planning and spatial policy development!")
        
    except Exception as e:
        logger.error(f"Tutorial failed: {e}")
        print(f"❌ Tutorial failed: {e}")
        raise


if __name__ == "__main__":
    main()
