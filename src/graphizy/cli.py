"""
Command Line Interface for graphizy

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np

from .main import Graphing
from .config import GraphizyConfig
from .algorithms import generate_positions
from .exceptions import GraphizyError


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Graphizy - A graph maker for computational geometry and network visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  graphizy delaunay --size 800 --particles 100 --output delaunay.jpg
  graphizy proximity --size 1200 --particles 200 --threshold 50 --output proximity.jpg
  graphizy both --size 1000 --particles 150 --show
        """
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Graph type to create')

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--size', type=int, default=1200,
                               help='Size of the graph canvas (default: 1200)')
    common_parser.add_argument('--particles', type=int, default=200,
                               help='Number of particles/points (default: 200)')
    common_parser.add_argument('--output', '-o', type=str,
                               help='Output filename for saving the graph')
    common_parser.add_argument('--show', action='store_true',
                               help='Display the graph in a window')
    common_parser.add_argument('--config', type=str,
                               help='JSON configuration file')
    common_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Enable verbose logging')
    common_parser.add_argument('--line-color', type=str, default='0,255,0',
                               help='Line color as R,G,B (default: 0,255,0)')
    common_parser.add_argument('--point-color', type=str, default='0,0,255',
                               help='Point color as R,G,B (default: 0,0,255)')
    common_parser.add_argument('--line-thickness', type=int, default=1,
                               help='Line thickness (default: 1)')
    common_parser.add_argument('--point-radius', type=int, default=8,
                               help='Point radius (default: 8)')

    # Delaunay subcommand
    delaunay_parser = subparsers.add_parser('delaunay', parents=[common_parser],
                                            help='Create Delaunay triangulation graph')

    # Proximity subcommand
    proximity_parser = subparsers.add_parser('proximity', parents=[common_parser],
                                             help='Create proximity graph')
    proximity_parser.add_argument('--threshold', type=float, default=50.0,
                                  help='Proximity threshold distance (default: 50.0)')
    proximity_parser.add_argument('--metric', type=str, default='euclidean',
                                  choices=['euclidean', 'manhattan', 'chebyshev'],
                                  help='Distance metric (default: euclidean)')

    # Memory subcommand
    memory_parser = subparsers.add_parser('memory', parents=[common_parser],
                                          help='Create memory-based graph')
    memory_parser.add_argument('--memory-size', type=int, default=100,
                               help='Maximum memory connections per object (default: 100)')
    memory_parser.add_argument('--memory-iterations', type=int, default=None,
                               help='Maximum iterations to keep in memory (default: unlimited)')
    memory_parser.add_argument('--proximity-thresh', type=float, default=50.0,
                               help='Proximity threshold for updating memory (default: 50.0)')
    memory_parser.add_argument('--iterations', type=int, default=10,
                               help='Number of simulation iterations (default: 10)')

    # Both subcommand
    both_parser = subparsers.add_parser('both', parents=[common_parser],
                                        help='Create both Delaunay and proximity graphs')
    both_parser.add_argument('--threshold', type=float, default=50.0,
                             help='Proximity threshold distance (default: 50.0)')
    both_parser.add_argument('--metric', type=str, default='euclidean',
                             choices=['euclidean', 'manhattan', 'chebyshev'],
                             help='Distance metric (default: euclidean)')
    both_parser.add_argument('--delaunay-output', type=str,
                             help='Output filename for Delaunay graph')
    both_parser.add_argument('--proximity-output', type=str,
                             help='Output filename for proximity graph')

    # Info subcommand
    info_parser = subparsers.add_parser('info', parents=[common_parser],
                                        help='Generate graph and show statistics')
    info_parser.add_argument('--threshold', type=float, default=50.0,
                             help='Proximity threshold distance (default: 50.0)')

    return parser


def parse_color(color_str: str) -> tuple:
    """Parse color string to tuple"""
    try:
        r, g, b = map(int, color_str.split(','))
        return (b, g, r)  # OpenCV uses BGR
    except:
        raise ValueError(f"Invalid color format: {color_str}. Use R,G,B format.")


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise GraphizyError(f"Failed to load config file {config_file}: {e}")


def setup_logging(verbose: bool) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )



def create_config_from_args(args) -> GraphizyConfig:
    """Create configuration from command line arguments"""
    config = GraphizyConfig()

    # Load from file if specified
    if hasattr(args, 'config') and args.config:
        file_config = load_config(args.config)
        config.update(**file_config)

    # Override with command line arguments (only if explicitly provided)
    config.graph.dimension = (args.size, args.size)
    config.generation.size_x = args.size
    config.generation.size_y = args.size
    config.generation.num_particles = args.particles

    # Drawing configuration - only override if explicitly set (not default)
    if hasattr(args, 'line_color'):
        config.drawing.line_color = parse_color(args.line_color)
    if hasattr(args, 'point_color'):
        config.drawing.point_color = parse_color(args.point_color)
    # Only override line_thickness if it's not the default value
    if hasattr(args, 'line_thickness') and args.line_thickness != 1:
        config.drawing.line_thickness = args.line_thickness
    if hasattr(args, 'point_radius'):
        config.drawing.point_radius = args.point_radius

    # Graph-specific configuration
    if hasattr(args, 'threshold'):
        config.graph.proximity_threshold = args.threshold
    if hasattr(args, 'metric'):
        config.graph.distance_metric = args.metric

    # Memory configuration (if this is a memory command)
    if hasattr(args, 'memory_size'):
        config.memory.max_memory_size = args.memory_size
    if hasattr(args, 'memory_iterations'):
        config.memory.max_iterations = args.memory_iterations

    # Logging configuration
    config.logging.level = 'DEBUG' if args.verbose else 'INFO'

    return config


def generate_data(config: GraphizyConfig) -> np.ndarray:
    """Generate particle data"""
    position_list = generate_positions(
        config.generation.size_x,
        config.generation.size_y,
        config.generation.num_particles,
        config.generation.to_array,
        config.generation.convert_to_float
    )

    # Create particle stack with IDs
    particle_list = np.array(list(range(len(position_list))))
    particle_stack = np.column_stack((particle_list, position_list))

    return particle_stack


def cmd_delaunay(args) -> None:
    """Handle delaunay command"""
    config = create_config_from_args(args)
    setup_logging(args.verbose)

    try:
        # Generate data
        particle_stack = generate_data(config)

        # Create grapher
        grapher = Graphing(config=config)

        # Create Delaunay graph
        logging.info("Creating Delaunay triangulation...")
        graph_del = grapher.make_delaunay(particle_stack)

        # Get statistics
        info = grapher.get_graph_info(graph_del)
        print(f"Delaunay Graph Statistics:")
        print(f"  Vertices: {info['vertex_count']}")
        print(f"  Edges: {info['edge_count']}")
        print(f"  Density: {info['density']:.4f}")
        if info['average_path_length']:
            print(f"  Average Path Length: {info['average_path_length']:.4f}")

        # Draw graph
        image = grapher.draw_graph(graph_del)

        # Save if requested
        if args.output:
            grapher.save_graph(image, args.output)
            print(f"Delaunay graph saved to {args.output}")

        # Show if requested
        if args.show:
            grapher.show_graph(image, "Delaunay Triangulation")

    except Exception as e:
        logging.error(f"Failed to create Delaunay graph: {e}")
        sys.exit(1)


def cmd_proximity(args) -> None:
    """Handle proximity command"""
    config = create_config_from_args(args)
    setup_logging(args.verbose)

    try:
        # Generate data
        particle_stack = generate_data(config)

        # Create grapher
        grapher = Graphing(config=config)

        # Create proximity graph
        logging.info(f"Creating proximity graph with threshold {args.threshold}...")
        graph_prox = grapher.make_proximity(particle_stack, args.threshold, args.metric)

        # Get statistics
        info = grapher.get_graph_info(graph_prox)
        print(f"Proximity Graph Statistics:")
        print(f"  Vertices: {info['vertex_count']}")
        print(f"  Edges: {info['edge_count']}")
        print(f"  Density: {info['density']:.4f}")
        print(f"  Connected: {info['is_connected']}")
        if info['average_path_length']:
            print(f"  Average Path Length: {info['average_path_length']:.4f}")

        # Draw graph
        image = grapher.draw_graph(graph_prox)

        # Save if requested
        if args.output:
            grapher.save_graph(image, args.output)
            print(f"Proximity graph saved to {args.output}")

        # Show if requested
        if args.show:
            grapher.show_graph(image, f"Proximity Graph (threshold={args.threshold})")

    except Exception as e:
        logging.error(f"Failed to create proximity graph: {e}")
        sys.exit(1)


def cmd_memory(args):
    """Handle memory command"""
    config = create_config_from_args(args)
    setup_logging(args.verbose)

    try:
        # Generate initial data
        particle_stack = generate_data(config)

        # Create grapher with memory
        grapher = Graphing(config=config)
        grapher.init_memory_manager(args.memory_size, args.memory_iterations)

        # Simulate multiple iterations
        print(f"Simulating {args.iterations} iterations...")

        for i in range(args.iterations):
            # Add some random movement (optional)
            if i > 0:
                # Small random movements
                particle_stack[:, 1:3] += np.random.normal(0, 5, (len(particle_stack), 2))
                # Keep within bounds
                particle_stack[:, 1] = np.clip(particle_stack[:, 1], 0, config.graph.dimension[0] - 1)
                particle_stack[:, 2] = np.clip(particle_stack[:, 2], 0, config.graph.dimension[1] - 1)

            # Update memory with current proximities
            current_connections = grapher.update_memory_with_proximity(
                particle_stack, args.proximity_thresh
            )

            print(
                f"Iteration {i + 1}: {sum(len(conns) for conns in current_connections.values()) // 2} current connections")

        # Create final memory graph
        memory_graph = grapher.make_memory_graph(particle_stack)

        # Get statistics
        stats = grapher.get_memory_stats()
        graph_info = grapher.get_graph_info(memory_graph)

        print(f"\nMemory Graph Results:")
        print(f"  Total Objects: {stats['total_objects']}")
        print(f"  Memory Connections: {stats['total_connections']}")
        print(f"  Graph Vertices: {graph_info['vertex_count']}")
        print(f"  Graph Edges: {graph_info['edge_count']}")
        print(f"  Graph Density: {graph_info['density']:.4f}")

        # Draw graph
        image = grapher.draw_graph(memory_graph)

        # Save if requested
        if args.output:
            grapher.save_graph(image, args.output)
            print(f"Memory graph saved to {args.output}")

        # Show if requested
        if args.show:
            grapher.show_graph(image, f"Memory Graph ({stats['total_connections']} connections)")

    except Exception as e:
        logging.error(f"Failed to create memory graph: {e}")
        sys.exit(1)
        
def cmd_both(args) -> None:
    """Handle both command"""
    config = create_config_from_args(args)
    setup_logging(args.verbose)

    try:
        # Generate data
        particle_stack = generate_data(config)

        # Create grapher
        grapher = Graphing(config=config)

        # Create both graphs
        logging.info("Creating Delaunay triangulation...")
        graph_del = grapher.make_delaunay(particle_stack)

        logging.info(f"Creating proximity graph with threshold {args.threshold}...")
        graph_prox = grapher.make_proximity(particle_stack, args.threshold, args.metric)

        # Get statistics for both
        del_info = grapher.get_graph_info(graph_del)
        prox_info = grapher.get_graph_info(graph_prox)

        print(f"Graph Comparison:")
        print(
            f"  Delaunay - Vertices: {del_info['vertex_count']}, Edges: {del_info['edge_count']}, Density: {del_info['density']:.4f}")
        print(
            f"  Proximity - Vertices: {prox_info['vertex_count']}, Edges: {prox_info['edge_count']}, Density: {prox_info['density']:.4f}")

        # Draw graphs
        del_image = grapher.draw_graph(graph_del)
        prox_image = grapher.draw_graph(graph_prox)

        # Save if requested
        delaunay_output = args.delaunay_output or (args.output and f"delaunay_{args.output}")
        proximity_output = args.proximity_output or (args.output and f"proximity_{args.output}")

        if delaunay_output:
            grapher.save_graph(del_image, delaunay_output)
            print(f"Delaunay graph saved to {delaunay_output}")

        if proximity_output:
            grapher.save_graph(prox_image, proximity_output)
            print(f"Proximity graph saved to {proximity_output}")

        # Show if requested
        if args.show:
            grapher.show_graph(del_image, "Delaunay Triangulation")
            grapher.show_graph(prox_image, f"Proximity Graph (threshold={args.threshold})")

    except Exception as e:
        logging.error(f"Failed to create graphs: {e}")
        sys.exit(1)


def cmd_info(args) -> None:
    """Handle info command"""
    config = create_config_from_args(args)
    setup_logging(args.verbose)

    try:
        # Generate data
        particle_stack = generate_data(config)

        # Create grapher
        grapher = Graphing(config=config)

        # Create both graphs
        graph_del = grapher.make_delaunay(particle_stack)
        graph_prox = grapher.make_proximity(particle_stack, args.threshold)

        # Detailed statistics
        del_info = grapher.get_graph_info(graph_del)
        prox_info = grapher.get_graph_info(graph_prox)

        print(f"Dataset Information:")
        print(f"  Canvas Size: {args.size}x{args.size}")
        print(f"  Number of Particles: {args.particles}")
        print()

        print("Delaunay Triangulation:")
        for key, value in del_info.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")
        print()

        print(f"Proximity Graph (threshold={args.threshold}):")
        for key, value in prox_info.items():
            if value is not None:
                if isinstance(value, float):
                    print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        # Save summary if output specified
        if args.output:
            summary = {
                "dataset": {
                    "canvas_size": f"{args.size}x{args.size}",
                    "particles": args.particles
                },
                "delaunay": del_info,
                "proximity": prox_info
            }

            output_path = Path(args.output)
            json_path = output_path.with_suffix('.json')

            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to {json_path}")

    except Exception as e:
        logging.error(f"Failed to generate info: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'delaunay':
            cmd_delaunay(args)
        elif args.command == 'proximity':
            cmd_proximity(args)
        elif args.command == 'memory':
            cmd_memory(args)
        elif args.command == 'both':
            cmd_both(args)
        elif args.command == 'info':
            cmd_info(args)
        else:
            parser.print_help()
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()