#!/usr/bin/env python3
"""
Interactive Brownian Motion Viewer for Graphizy

This modified version displays graphs in real-time using OpenCV imshow
instead of saving files. Select graph type by number:

Graph Types:
    1 - Proximity Graph (Red)
    2 - Delaunay Triangulation (Green)
    3 - Memory Graph (Blue)
    4 - Combined View (All graphs side-by-side)

Usage:
    python interactive_brownian.py [graph_type] [options]

Controls:
    ESC - Exit
    SPACE - Pause/Resume
    R - Reset simulation
    1-4 - Switch graph type during simulation

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL-2.0-or-later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
import logging
import sys
import argparse
import time
import cv2
from typing import Optional, Dict, List

from graphizy import (
    Graphing, GraphizyConfig, generate_positions
)

# Setup logging for clear, informative output during the simulation.
logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(message)s')


class InteractiveBrownianSimulation:
    """
    Interactive Brownian motion simulation with real-time OpenCV display
    """

    def __init__(self, width: int = 800, height: int = 600, num_particles: int = 50,
                 use_memory: bool = True, memory_size: int = 25, memory_iterations: Optional[int] = 10):
        """Initializes the simulation environment and all its components."""
        self.width = width
        self.height = height
        self.num_particles = num_particles
        self.use_memory = use_memory

        # --- Physics Parameters ---
        self.diffusion_coefficient = 15.0
        self.boundary_buffer = 20

        # --- Graphing Parameters ---
        self.proximity_threshold = 100.0
        self.delaunay_update_frequency = 3

        # --- Display Parameters ---
        self.window_name = "Graphizy - Interactive Brownian Motion"
        self.paused = False
        self.current_graph_type = 1  # Default to proximity

        # Graph type mapping
        self.graph_type_names = {
            1: "Proximity Graph",
            2: "Delaunay Triangulation",
            3: "Memory Graph",
            4: "Combined View"
        }

        print("Initializing Interactive Brownian simulation...")
        print("Controls: ESC=Exit, SPACE=Pause/Resume, R=Reset, 1-4=Switch graph type")

        # Sequentially set up all components of the simulation.
        self._initialize_particles()
        self._setup_graphing(memory_size, memory_iterations)
        self._setup_opencv()

        self.iteration = 0

    def _initialize_particles(self):
        """Creates the initial set of particles with random positions."""
        positions = generate_positions(self.width, self.height, self.num_particles)
        particle_ids = np.arange(self.num_particles)
        self.particle_stack = np.column_stack((particle_ids, positions))
        self.velocities = np.zeros((self.num_particles, 2))
        print(f"Initialized {self.num_particles} particles.")

    def _setup_graphing(self, memory_size: int, memory_iterations: Optional[int]):
        """Sets up the Graphing instances, one for each visualization style."""
        base_config = GraphizyConfig()
        base_config.graph.dimension = (self.width, self.height)
        base_config.drawing.point_radius = 8
        base_config.drawing.line_thickness = 2

        # Create graphers with distinct colors
        self.graphers = {
            'proximity': self._create_grapher(base_config, (0, 0, 255), (255, 255, 255)),  # Red lines
            'delaunay': self._create_grapher(base_config, (0, 255, 0), (255, 255, 0)),  # Green lines
            'memory': self._create_grapher(base_config, (255, 100, 0), (100, 255, 100))  # Blue lines
        }

        if self.use_memory:
            # Initialize memory manager for the memory grapher
            self.graphers['memory'].init_memory_manager(
                max_memory_size=memory_size,
                max_iterations=memory_iterations,
                track_edge_ages=True
            )
            print("Memory manager initialized.")

    def _create_grapher(self, base_config: GraphizyConfig, line_color: tuple, point_color: tuple) -> Graphing:
        """Helper to create a styled Graphing instance."""
        config = base_config.copy()
        config.drawing.line_color = line_color
        config.drawing.point_color = point_color
        return Graphing(config=config)

    def _setup_opencv(self):
        """Setup OpenCV window and display"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        print(f"OpenCV window '{self.window_name}' created")

    def update_positions(self, dt: float = 1.0):
        """Updates particle positions based on a simple Brownian motion model."""
        random_forces = np.random.normal(0, self.diffusion_coefficient, (self.num_particles, 2))
        momentum_factor = 0.1
        self.velocities = momentum_factor * self.velocities + (1 - momentum_factor) * random_forces
        self.particle_stack[:, 1:3] += self.velocities * dt

        # Implement reflective boundary conditions
        for i in range(self.num_particles):
            x, y = self.particle_stack[i, 1:3]
            if not (self.boundary_buffer < x < self.width - self.boundary_buffer):
                self.velocities[i, 0] *= -1
            if not (self.boundary_buffer < y < self.height - self.boundary_buffer):
                self.velocities[i, 1] *= -1

        # Clamp positions to bounds
        self.particle_stack[:, 1] = np.clip(self.particle_stack[:, 1], self.boundary_buffer,
                                            self.width - self.boundary_buffer)
        self.particle_stack[:, 2] = np.clip(self.particle_stack[:, 2], self.boundary_buffer,
                                            self.height - self.boundary_buffer)

    def generate_graphs(self) -> Dict[str, any]:
        """Generates all relevant graphs for the current particle positions."""
        graphs = {}

        # Proximity graph - generated every frame
        graphs['proximity'] = self.graphers['proximity'].make_proximity(self.particle_stack,
                                                                        proximity_thresh=self.proximity_threshold)

        # Delaunay graph - updated periodically to save computation
        if self.iteration % self.delaunay_update_frequency == 0:
            graphs['delaunay'] = self.graphers['delaunay'].make_delaunay(self.particle_stack)
            self._last_delaunay = graphs['delaunay']
        else:
            graphs['delaunay'] = getattr(self, '_last_delaunay', None)

        # Memory graph
        if self.use_memory:
            self.graphers['memory'].update_memory_with_proximity(self.particle_stack)
            graphs['memory'] = self.graphers['memory'].make_memory_graph(self.particle_stack)

        return graphs

    def create_single_visualization(self, graphs: Dict[str, any], graph_type: int) -> Optional[np.ndarray]:
        """Creates visualization for a single graph type"""
        graph_map = {
            1: 'proximity',
            2: 'delaunay',
            3: 'memory'
        }

        if graph_type not in graph_map:
            return None

        graph_name = graph_map[graph_type]
        graph = graphs.get(graph_name)

        if not graph:
            return None

        grapher = self.graphers[graph_name]

        # Special handling for memory graphs
        if graph_name == 'memory' and self.use_memory:
            try:
                image = grapher.draw_memory_graph(graph, use_age_colors=True, alpha_range=(0.4, 1.0))
            except:
                # Fallback to regular drawing if memory drawing fails
                image = grapher.draw_graph(graph)
        else:
            image = grapher.draw_graph(graph)

        return image

    def create_combined_visualization(self, graphs: Dict[str, any]) -> Optional[np.ndarray]:
        """Creates a combined view showing all graph types"""
        images = []

        for graph_type in [1, 2, 3]:  # proximity, delaunay, memory
            img = self.create_single_visualization(graphs, graph_type)
            if img is not None:
                images.append(img)
            else:
                # Create blank image as placeholder
                blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                images.append(blank)

        if not images:
            return None

        # Ensure we have exactly 3 images
        while len(images) < 3:
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            images.append(blank)

        # Create a 2x2 grid (with one empty slot)
        top_row = np.hstack([images[0], images[1]])
        bottom_left = images[2]
        bottom_right = np.zeros_like(images[2])  # Empty slot
        bottom_row = np.hstack([bottom_left, bottom_right])
        combined = np.vstack([top_row, bottom_row])

        return combined

    def add_info_overlay(self, image: np.ndarray, graph_type: int) -> np.ndarray:
        """Add information overlay to the image"""
        if image is None:
            return image

        # Create a copy to avoid modifying original
        img_with_overlay = image.copy()

        # Add title
        title = self.graph_type_names.get(graph_type, f"Graph Type {graph_type}")
        cv2.putText(img_with_overlay, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Add iteration counter
        iter_text = f"Iteration: {self.iteration}"
        cv2.putText(img_with_overlay, iter_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add particle count
        particle_text = f"Particles: {self.num_particles}"
        cv2.putText(img_with_overlay, particle_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Add pause indicator
        if self.paused:
            cv2.putText(img_with_overlay, "PAUSED", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                        2)

        return img_with_overlay

    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input. Returns False if should exit."""
        if key == 27:  # ESC key
            return False
        elif key == ord(' '):  # Space key
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'resumed'}")
        elif key == ord('r') or key == ord('R'):  # Reset
            self._initialize_particles()
            self.iteration = 0
            print("Simulation reset")
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:  # Graph type selection
            self.current_graph_type = int(chr(key))
            print(f"Switched to: {self.graph_type_names.get(self.current_graph_type, 'Unknown')}")

        return True

    def run_simulation(self, graph_type: int = 1, max_iterations: int = 1000, dt: float = 1.0, fps: int = 30):
        """The main interactive simulation loop."""
        self.current_graph_type = graph_type
        print(f"Starting interactive simulation...")
        print(f"Initial graph type: {self.graph_type_names.get(graph_type, 'Unknown')}")

        frame_delay = int(1000 / fps)  # Delay in milliseconds

        while self.iteration < max_iterations:
            if not self.paused:
                # Update physics
                self.update_positions(dt)

                # Generate graphs
                graphs = self.generate_graphs()

                # Create visualization based on current graph type
                if self.current_graph_type == 4:  # Combined view
                    image = self.create_combined_visualization(graphs)
                else:
                    image = self.create_single_visualization(graphs, self.current_graph_type)

                if image is not None:
                    # Add info overlay
                    display_image = self.add_info_overlay(image, self.current_graph_type)

                    # Display the image
                    cv2.imshow(self.window_name, display_image)

                self.iteration += 1
            else:
                # Still show the last frame when paused
                cv2.imshow(self.window_name,
                           display_image if 'display_image' in locals() else np.zeros((self.height, self.width, 3),
                                                                                      dtype=np.uint8))

            # Handle keyboard input
            key = cv2.waitKey(frame_delay) & 0xFF
            if not self.handle_keyboard_input(key):
                break

        print(f"\nSimulation finished after {self.iteration} iterations.")
        cv2.destroyAllWindows()

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self._initialize_particles()
        self.iteration = 0


def main():
    """Parses command-line arguments and runs the interactive simulation."""
    parser = argparse.ArgumentParser(description="Graphizy: Interactive Brownian Motion Viewer")

    # Graph type selection
    parser.add_argument('graph_type', type=int, nargs='?', default=1,
                        help='Graph type to display (1=Proximity, 2=Delaunay, 3=Memory, 4=Combined)')

    # Simulation parameters
    parser.add_argument('--iterations', '-i', type=int, default=1000,
                        help='Maximum number of iterations (default: 1000)')
    parser.add_argument('--particles', '-p', type=int, default=50,
                        help='Number of particles (default: 50)')
    parser.add_argument('--size', nargs=2, type=int, default=[800, 600],
                        help='Canvas size [width height] (default: 800 600)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Display FPS (default: 30)')

    # Memory parameters
    parser.add_argument('--no-memory', action='store_false', dest='memory',
                        help='Disable memory graph functionality')
    parser.add_argument('--memory-size', type=int, default=25,
                        help='Memory buffer size (default: 25)')

    args = parser.parse_args()

    # Validate graph type
    if args.graph_type not in [1, 2, 3, 4]:
        print("Error: Graph type must be 1 (Proximity), 2 (Delaunay), 3 (Memory), or 4 (Combined)")
        sys.exit(1)

    try:
        simulation = InteractiveBrownianSimulation(
            width=args.size[0], height=args.size[1],
            num_particles=args.particles, use_memory=args.memory,
            memory_size=args.memory_size
        )

        simulation.run_simulation(
            graph_type=args.graph_type,
            max_iterations=args.iterations,
            dt=1.0,
            fps=args.fps
        )

        print("\nSimulation completed successfully!")

    except Exception as e:
        print(f"\nAn error occurred during the simulation: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()