"""
Simplified Interactive Brownian Motion Viewer for Graphizy

This version uses the new graph update system to dramatically simplify the code.
Memory is now handled automatically by the update system.

Graph Types:
    1 - Proximity Graph
    2 - Delaunay Triangulation
    3 - Gabriel Graph
    4 - Minimum Spanning Tree
    5 - Combined View (All graphs)

Examples:
    python simplified_brownian.py 1          # Proximity graph (no memory)
    python simplified_brownian.py 1 --memory # Proximity graph WITH memory
    python simplified_brownian.py 4 --memory # MST WITH memory
    python simplified_brownian.py 5 --memory # All graphs WITH memory

Controls:
    ESC - Exit, SPACE - Pause/Resume, R - Reset simulation
    M - Toggle memory on/off, 1-5 - Switch graph type
    + / - - Increase/decrease memory size

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
from typing import Optional, Dict, List, Tuple, Any

from graphizy import Graphing, GraphizyConfig, generate_positions


class BrownianSimulator:
    """
    Brownian motion simulation using the new graph update system
    """

    def __init__(self, width: int = 800, height: int = 600, num_particles: int = 50,
                 use_memory: bool = False, memory_size: int = 25):
        """Initialize the simulation"""
        self.width = width
        self.height = height
        self.num_particles = num_particles
        self.use_memory = use_memory
        self.memory_size = memory_size

        # Physics parameters
        self.diffusion_coefficient = 15.0
        self.boundary_buffer = 20
        self.proximity_threshold = 100.0

        # Display parameters
        self.window_name = "Graphizy - Simplified (Press m to toggle memory / 1-5 to change graph type)"
        self.paused = False
        self.current_graph_type = 1

        # Graph type definitions
        self.graph_type_names = {
            1: "Proximity Graph",
            2: "Delaunay Triangulation",
            3: "Gabriel Graph",
            4: "Minimum Spanning Tree",
            5: "Combined View"
        }

        # Map numbers to graph type strings
        self.type_map = {
            1: 'proximity',
            2: 'delaunay',
            3: 'gabriel',
            4: 'mst'
        }

        print("Initializing Simplified Brownian simulation...")
        print(f"Memory {'ENABLED' if use_memory else 'DISABLED'}")
        print("Controls: ESC=Exit, SPACE=Pause, R=Reset, M=Toggle Memory, 1-5=Graph Type, +/-=Memory Size")

        self._initialize_particles()
        self._setup_graphers()
        self._setup_opencv()
        self.iteration = 0

    def _initialize_particles(self):
        """Initialize particle positions and velocities"""
        positions = generate_positions(self.width, self.height, self.num_particles)
        particle_ids = np.arange(self.num_particles)
        self.particle_stack = np.column_stack((particle_ids, positions))
        self.velocities = np.zeros((self.num_particles, 2))
        print(f"Initialized {self.num_particles} particles.")

    def _setup_graphers(self):
        """Setup graphers using the new update system"""
        # Create distinct colored graphers for each type
        self.graphers = {}

        colors = {
            'proximity': {'line': (0, 0, 255), 'point': (255, 255, 255)},    # Red
            'delaunay': {'line': (0, 255, 0), 'point': (255, 255, 0)},       # Green
            'gabriel': {'line': (255, 100, 0), 'point': (100, 255, 255)},    # Blue
            'mst': {'line': (255, 0, 255), 'point': (255, 255, 100)}         # Purple
        }

        for graph_type, color_config in colors.items():
            # Create grapher with specific colors
            config = GraphizyConfig()
            config.graph.dimension = (self.width, self.height)
            config.drawing.point_radius = 8
            config.drawing.line_thickness = 2
            config.drawing.line_color = color_config['line']
            config.drawing.point_color = color_config['point']

            grapher = Graphing(config=config)

            # Configure the graph type using the new system
            if graph_type == 'proximity':
                grapher.set_graph_type('proximity', proximity_thresh=self.proximity_threshold)
            else:
                grapher.set_graph_type(graph_type)

            # Initialize memory if enabled
            if self.use_memory:
                grapher.init_memory_manager(
                    max_memory_size=self.memory_size,
                    track_edge_ages=True
                )

            self.graphers[graph_type] = grapher

        print(f"Graphers initialized with graph update system")

    def _setup_opencv(self):
        """Setup OpenCV window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def update_positions(self, dt: float = 1.0):
        """Update particle positions using Brownian motion"""
        random_forces = np.random.normal(0, self.diffusion_coefficient, (self.num_particles, 2))
        momentum_factor = 0.1
        self.velocities = momentum_factor * self.velocities + (1 - momentum_factor) * random_forces
        self.particle_stack[:, 1:3] += self.velocities * dt

        # Reflective boundary conditions
        for i in range(self.num_particles):
            x, y = self.particle_stack[i, 1:3]
            if not (self.boundary_buffer < x < self.width - self.boundary_buffer):
                self.velocities[i, 0] *= -1
            if not (self.boundary_buffer < y < self.height - self.boundary_buffer):
                self.velocities[i, 1] *= -1

        # Clamp positions
        self.particle_stack[:, 1] = np.clip(self.particle_stack[:, 1], self.boundary_buffer,
                                            self.width - self.boundary_buffer)
        self.particle_stack[:, 2] = np.clip(self.particle_stack[:, 2], self.boundary_buffer,
                                            self.height - self.boundary_buffer)

    def create_visualization(self, graph_type: int) -> Optional[np.ndarray]:
        """Create visualization using the simplified update system"""
        if graph_type == 5:  # Combined view
            return self._create_combined_view()

        # Get the graph type string
        graph_type_str = self.type_map.get(graph_type)
        if not graph_type_str:
            return None

        grapher = self.graphers[graph_type_str]

        # Update the graphs
        start_time_graph_update = time.perf_counter()
        graphs = grapher.update_graphs(self.particle_stack, update_memory=self.use_memory)
        end_time_graph_update = time.perf_counter() - start_time_graph_update

        # Get the generated graph
        graph = graphs.get(graph_type_str)
        if graph is None:
            return None

        # Draw the graph (automatically handles memory visualization)
        if self.use_memory and grapher.memory_manager is not None:
            try:
                return grapher.draw_memory_graph(graph, use_age_colors=True, alpha_range=(0.3, 1.0)), end_time_graph_update
            except:
                return grapher.draw_graph(graph), end_time_graph_update
        else:
            return grapher.draw_graph(graph), end_time_graph_update

    def _create_combined_view(self) -> Optional[np.ndarray]:
        """Create combined view showing all graph types (2x2 grid)"""
        images = []

        for graph_type in [1, 2, 3, 4]:  # proximity, delaunay, gabriel, mst
            img, _ = self.create_visualization(graph_type)
            if img is not None:
                images.append(img)
            else:
                # Create blank placeholder
                blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                images.append(blank)

        if not images:
            return None

        # Ensure we have 4 images for 2x2 grid
        while len(images) < 4:
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            images.append(blank)

        # Create 2x2 grid
        top_row = np.hstack([images[0], images[1]])  # Proximity + Delaunay
        bottom_row = np.hstack([images[2], images[3]])  # Gabriel + MST
        combined = np.vstack([top_row, bottom_row])

        return combined

    def add_info_overlay(self, image: np.ndarray, graph_type: int, time_graph_update: Optional[float] = None) -> np.ndarray:
        """Add information overlay to the image"""
        if image is None:
            return image

        img_with_overlay = image.copy()

        # Graph type and memory status
        title = self.graph_type_names.get(graph_type, f"Graph Type {graph_type}")
        if self.use_memory:
            title += " (with Memory)"
        cv2.putText(img_with_overlay, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Iteration counter
        cv2.putText(img_with_overlay, f"Iteration: {self.iteration} / Took: {time_graph_update*1000:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)

        # Memory info
        if self.use_memory:
            cv2.putText(img_with_overlay, f"Memory Size: {self.memory_size}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

        # Pause indicator
        if self.paused:
            cv2.putText(img_with_overlay, "PAUSED", (10, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255),
                        2)

        return img_with_overlay

    def toggle_memory(self):
        """Toggle memory on/off during simulation"""
        self.use_memory = not self.use_memory

        if self.use_memory:
            # Initialize memory for all graphers
            for grapher in self.graphers.values():
                grapher.init_memory_manager(
                    max_memory_size=self.memory_size,
                    track_edge_ages=True
                )
            print(f"Memory ENABLED (size: {self.memory_size})")
        else:
            # Clear memory managers
            for grapher in self.graphers.values():
                grapher.memory_manager = None
            print("Memory DISABLED")

    def adjust_memory_size(self, delta: int):
        """Adjust memory size during simulation"""
        if not self.use_memory:
            return

        new_size = max(5, self.memory_size + delta)  # Minimum size of 5
        if new_size != self.memory_size:
            self.memory_size = new_size
            # Reinitialize memory managers with new size
            for grapher in self.graphers.values():
                grapher.init_memory_manager(
                    max_memory_size=self.memory_size,
                    track_edge_ages=True
                )
            print(f"Memory size adjusted to: {self.memory_size}")

    def handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input. Returns False if should exit."""
        if key == 27:  # ESC
            return False
        elif key == ord(' '):  # Space - Pause/Resume
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'resumed'}")
        elif key == ord('r') or key == ord('R'):  # Reset
            self._initialize_particles()
            self.iteration = 0
            if self.use_memory:
                # Clear and reinitialize memory
                for grapher in self.graphers.values():
                    grapher.init_memory_manager(
                        max_memory_size=self.memory_size,
                        track_edge_ages=True
                    )
            print("Simulation reset")
        elif key == ord('m') or key == ord('M'):  # Toggle memory
            self.toggle_memory()
        elif key == ord('+') or key == ord('='):  # Increase memory size
            self.adjust_memory_size(5)
        elif key == ord('-') or key == ord('_'):  # Decrease memory size
            self.adjust_memory_size(-5)
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:  # Graph type
            self.current_graph_type = int(chr(key))
            graph_name = self.graph_type_names.get(self.current_graph_type, 'Unknown')
            memory_status = "with Memory" if self.use_memory else "no Memory"
            print(f"Switched to: {graph_name} ({memory_status})")

        return True

    def run_simulation(self, graph_type: int = 1, max_iterations: int = 1000, fps: int = 30):
        """Main interactive simulation loop"""
        self.current_graph_type = graph_type
        frame_delay = int(1000 / fps)

        print(f"Starting simulation with {self.graph_type_names.get(graph_type, 'Unknown')}")
        print(f"Memory: {'ENABLED' if self.use_memory else 'DISABLED'}")

        display_image = None

        while self.iteration < max_iterations:
            if not self.paused:
                # Update physics
                self.update_positions()

                # Create visualization
                image, time_graph_update = self.create_visualization(self.current_graph_type)

                if image is not None:
                    display_image = self.add_info_overlay(image, self.current_graph_type, time_graph_update)
                    cv2.imshow(self.window_name, display_image)

                self.iteration += 1
            else:
                # Show last frame when paused
                if display_image is not None:
                    cv2.imshow(self.window_name, display_image)

            # Handle input
            key = cv2.waitKey(frame_delay) & 0xFF
            if not self.handle_keyboard_input(key):
                break

        print(f"Simulation finished after {self.iteration} iterations.")
        cv2.destroyAllWindows()


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Simplified Interactive Brownian Motion with Graphizy')
    parser.add_argument('graph_type', type=int, nargs='?', default=1,
                        help='Graph type: 1=Proximity, 2=Delaunay, 3=Gabriel, 4=MST, 5=Combined')
    parser.add_argument('--memory', '-m', action='store_true',
                        help='Enable memory tracking')
    parser.add_argument('--memory-size', type=int, default=25,
                        help='Memory size (default: 25)')
    parser.add_argument('--particles', '-p', type=int, default=50,
                        help='Number of particles (default: 50)')
    parser.add_argument('--width', type=int, default=800,
                        help='Window width (default: 800)')
    parser.add_argument('--height', type=int, default=600,
                        help='Window height (default: 600)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target FPS (default: 30)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Max iterations (default: 1000)')

    args = parser.parse_args()

    if args.graph_type not in [1, 2, 3, 4, 5]:
        print("Error: Graph type must be 1-5")
        print("1=Proximity, 2=Delaunay, 3=Gabriel, 4=MST, 5=Combined")
        sys.exit(1)

    # Create and run simulator
    simulator = BrownianSimulator(
        width=args.width,
        height=args.height,
        num_particles=args.particles,
        use_memory=args.memory,
        memory_size=args.memory_size
    )

    try:
        simulator.run_simulation(
            graph_type=args.graph_type,
            max_iterations=args.iterations,
            fps=args.fps
        )
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Simulation error: {e}")
        logging.error(f"Simulation failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()