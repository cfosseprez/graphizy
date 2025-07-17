#!/usr/bin/env python3
"""
Integrated Brownian Motion Example for Graphizy

This comprehensive example demonstrates a dynamic simulation where particles
undergo Brownian motion, and their spatial relationships are continuously
analyzed and visualized as evolving graphs.

It showcases:
1.  A class-based structure for managing a complex simulation.
2.  Real-time graph generation from dynamic data (Delaunay, proximity).
3.  Integration of a memory graph to track connections over time.
4.  Automated frame saving and movie creation from the simulation output.
5.  A flexible command-line interface for customizing simulation parameters.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.me@gmail.com
.. license:: MIT
.. copyright:: Copyright (C) 2023 Charles Fosseprez
"""

import numpy as np
import logging
from pathlib import Path
import sys
import argparse
import time
from typing import Optional, Dict, List

from graphizy import (
    Graphing, GraphizyConfig, MemoryManager, generate_positions
)

# Setup logging for clear, informative output during the simulation.
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')


class BrownianSimulation:
    """
    Manages a Brownian motion simulation, including particle physics,
    graph generation, and visualization.
    """

    def __init__(self, width: int = 800, height: int = 600, num_particles: int = 50,
                 use_memory: bool = True, memory_size: int = 25, memory_iterations: Optional[int] = 10):
        """Initializes the simulation environment and all its components."""
        self.width = width
        self.height = height
        self.num_particles = num_particles
        self.use_memory = use_memory

        # --- Physics Parameters ---
        # Controls the magnitude of random movements.
        self.diffusion_coefficient = 15.0
        # A buffer from the edge of the canvas to prevent objects from overlapping the border.
        self.boundary_buffer = 20

        # --- Graphing Parameters ---
        self.proximity_threshold = 100.0
        # Delaunay calculation can be more expensive, so we update it less frequently.
        self.delaunay_update_frequency = 3

        print("Initializing Brownian simulation...")

        # Sequentially set up all components of the simulation.
        self._initialize_particles()
        self._setup_graphing(memory_size, memory_iterations)
        self._setup_output_directory()

        self.iteration = 0

    def _initialize_particles(self):
        """Creates the initial set of particles with random positions."""
        positions = generate_positions(self.width, self.height, self.num_particles)
        particle_ids = np.arange(self.num_particles)
        # The particle_stack holds the state (ID, x, y) for all particles.
        self.particle_stack = np.column_stack((particle_ids, positions))
        # Velocities are used to add a momentum effect to the movement.
        self.velocities = np.zeros((self.num_particles, 2))
        print(f"Initialized {self.num_particles} particles.")

    def _setup_graphing(self, memory_size: int, memory_iterations: Optional[int]):
        """Sets up the Graphing instances, one for each visualization style."""
        base_config = GraphizyConfig(
            graph={'dimension': (self.width, self.height)},
            drawing={'point_radius': 8, 'line_thickness': 2}
        )

        # We create separate `Graphing` objects to handle different visual styles
        # for each type of graph, avoiding reconfiguration in the main loop.
        self.graphers = {
            'proximity': self._create_grapher(base_config, (255, 0, 0), (255, 255, 255)),  # Red lines
            'delaunay': self._create_grapher(base_config, (0, 255, 0), (255, 255, 0)),  # Green lines
            'memory': self._create_grapher(base_config, (0, 100, 255), (100, 255, 100))  # Blue lines
        }

        if self.use_memory:
            self.memory_manager = MemoryManager(
                max_memory_size=memory_size,
                max_iterations=memory_iterations,
                track_edge_ages=True  # Essential for age-based coloring.
            )
            # Associate the memory manager with the 'memory' grapher.
            self.graphers['memory'].memory_manager = self.memory_manager
            print("Memory manager initialized.")

    def _create_grapher(self, base_config: GraphizyConfig, line_color: tuple, point_color: tuple) -> Graphing:
        """Helper to create a styled Graphing instance."""
        config = base_config.copy()
        config.drawing.line_color = line_color
        config.drawing.point_color = point_color
        return Graphing(config=config)

    def _setup_output_directory(self):
        """Creates the necessary output directories for saving frames."""
        self.output_dir = Path("examples/output/brownian_simulation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Create subdirectories to keep frames for each graph type organized.
        for graph_type in ['proximity', 'delaunay', 'memory', 'combined']:
            (self.output_dir / graph_type).mkdir(exist_ok=True)
        print(f"Output directory set to: {self.output_dir}")

    def update_positions(self, dt: float = 1.0):
        """Updates particle positions based on a simple Brownian motion model."""
        # The core of Brownian motion: add a random vector to each particle's velocity.
        random_forces = np.random.normal(0, self.diffusion_coefficient, (self.num_particles, 2))
        # A simple momentum model to smooth out the movement.
        momentum_factor = 0.1
        self.velocities = momentum_factor * self.velocities + (1 - momentum_factor) * random_forces
        self.particle_stack[:, 1:3] += self.velocities * dt

        # Implement reflective boundary conditions to keep particles on screen.
        # When a particle hits a wall, its velocity component perpendicular to the wall is inverted.
        for i in range(self.num_particles):
            x, y = self.particle_stack[i, 1:3]
            if not (self.boundary_buffer < x < self.width - self.boundary_buffer):
                self.velocities[i, 0] *= -1
            if not (self.boundary_buffer < y < self.height - self.boundary_buffer):
                self.velocities[i, 1] *= -1
        # Clamp positions to ensure they stay within bounds.
        self.particle_stack[:, 1] = np.clip(self.particle_stack[:, 1], self.boundary_buffer,
                                            self.width - self.boundary_buffer)
        self.particle_stack[:, 2] = np.clip(self.particle_stack[:, 2], self.boundary_buffer,
                                            self.height - self.boundary_buffer)

    def generate_graphs(self) -> Dict[str, any]:
        """Generates all relevant graphs for the current particle positions."""
        graphs = {}
        # Proximity graph is generated every frame.
        graphs['proximity'] = self.graphers['proximity'].make_proximity(self.particle_stack)

        # Delaunay graph is updated periodically to save computation time.
        if self.iteration % self.delaunay_update_frequency == 0:
            graphs['delaunay'] = self.graphers['delaunay'].make_delaunay(self.particle_stack)
            self._last_delaunay = graphs['delaunay']  # Cache the last computed graph.
        else:
            graphs['delaunay'] = getattr(self, '_last_delaunay', None)

        if self.use_memory:
            # First, update the memory with the current proximity connections.
            self.graphers['memory'].update_memory_with_proximity(self.particle_stack)
            # Then, create a graph object representing the entire memory state.
            graphs['memory'] = self.graphers['memory'].make_memory_graph(self.particle_stack)

        return graphs

    def create_visualizations(self, graphs: Dict[str, any], save_frames: bool) -> Dict[str, np.ndarray]:
        """Draws all generated graphs and saves them as image frames."""
        images = {}
        for name, graph in graphs.items():
            if graph:
                grapher = self.graphers[name]
                # Memory graphs can be drawn with special age-based coloring.
                if name == 'memory' and self.use_memory:
                    image = grapher.draw_memory_graph(graph, use_age_colors=True, alpha_range=(0.4, 1.0))
                else:
                    image = grapher.draw_graph(graph)

                images[name] = image
                if save_frames:
                    filepath = self.output_dir / name / f"frame_{self.iteration:04d}.jpg"
                    grapher.save_graph(image, str(filepath))
        return images

    def create_combined_visualization(self, images: Dict[str, np.ndarray], save_frame: bool) -> Optional[np.ndarray]:
        """Stitches individual graph visualizations into a single dashboard image."""
        valid_images = [v for v in images.values() if v is not None]
        if not valid_images: return None

        # Arrange images in a 2x2 grid for a comprehensive view.
        while len(valid_images) < 4:
            valid_images.append(np.zeros_like(valid_images[0]))

        top_row = np.hstack([valid_images[0], valid_images[1]])
        bottom_row = np.hstack([valid_images[2], valid_images[3]])
        combined = np.vstack([top_row, bottom_row])

        if save_frame:
            filepath = self.output_dir / "combined" / f"combined_frame_{self.iteration:04d}.jpg"
            self.graphers['proximity'].save_graph(combined, str(filepath))
        return combined

    def run_simulation(self, num_iterations: int, save_frequency: int, dt: float, verbose: bool):
        """The main simulation loop."""
        print(f"Starting simulation for {num_iterations} iterations...")
        start_time = time.time()

        for i in range(num_iterations):
            self.iteration = i

            # --- The core sequence for each time step ---
            # 1. Update particle physics.
            self.update_positions(dt)
            # 2. Generate graphs from the new positions.
            graphs = self.generate_graphs()

            # (Optional) Collect statistics for later analysis.
            # stats = self._analyze_graphs(graphs)

            # 3. Create and save visualizations.
            if i % save_frequency == 0:
                images = self.create_visualizations(graphs, save_frames=True)
                self.create_combined_visualization(images, save_frame=True)

            if verbose and (i % 10 == 0 or i < 10):
                print(f"Completed Iteration {i + 1}/{num_iterations}")

        print(f"\nSimulation finished in {time.time() - start_time:.2f} seconds.")

    def create_movie(self, graph_type: str, fps: int):
        """Creates an MP4 movie from the saved frames using OpenCV."""
        try:
            import cv2
        except ImportError:
            print("OpenCV (cv2) is required to create movies. Please install it (`pip install opencv-python`).")
            return

        frame_dir = self.output_dir / graph_type
        frame_files = sorted(list(frame_dir.glob("*.jpg")))
        if not frame_files:
            print(f"No frames found in '{frame_dir}' to create a movie.")
            return

        print(f"Creating '{graph_type}' movie from {len(frame_files)} frames at {fps} FPS...")
        first_frame = cv2.imread(str(frame_files[0]))
        height, width, _ = first_frame.shape

        output_path = self.output_dir / f"brownian_simulation_{graph_type}.mp4"
        video_writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame_file in frame_files:
            video_writer.write(cv2.imread(str(frame_file)))
        video_writer.release()

        print(f"Movie saved to: {output_path}")


def main():
    """Parses command-line arguments and runs the simulation."""
    parser = argparse.ArgumentParser(description="Graphizy: Integrated Brownian Motion Simulation")
    # Simulation parameters
    parser.add_argument('--iterations', '-i', type=int, default=100)
    parser.add_argument('--particles', '-p', type=int, default=50)
    parser.add_argument('--size', nargs=2, type=int, default=[800, 600])
    # Memory parameters
    parser.add_argument('--no-memory', action='store_false', dest='memory')
    parser.add_argument('--memory-size', type=int, default=25)
    # Output parameters
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--no-movie', action='store_true')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    try:
        simulation = BrownianSimulation(
            width=args.size[0], height=args.size[1],
            num_particles=args.particles, use_memory=args.memory,
            memory_size=args.memory_size
        )

        simulation.run_simulation(
            num_iterations=args.iterations, save_frequency=1,
            dt=1.0, verbose=not args.quiet
        )

        if not args.no_movie:
            simulation.create_movie(graph_type='combined', fps=args.fps)
            if args.memory:
                simulation.create_movie(graph_type='memory', fps=args.fps)

        print("\nSimulation completed successfully!")
        print(f"Output files are in: {simulation.output_dir}")

    except Exception as e:
        print(f"\nAn error occurred during the simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())