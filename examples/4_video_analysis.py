#!/usr/bin/env python3
"""
Improved Interactive Brownian Motion Viewer for Graphizy

This version treats memory as a MODIFIER that can be applied to any graph type,
including the new Minimum Spanning Tree (MST) graph type.

Graph Types:
    1 - Proximity Graph
    2 - Delaunay Triangulation
    3 - K-Nearest Neighbors
    4 - Minimum Spanning Tree (NEW!)
    5 - Combined View (All graphs)

Memory Modifier:
    --memory / -m - Apply memory tracking to the selected graph type

Examples:
    python improved_brownian.py 1          # Proximity graph (no memory)
    python improved_brownian.py 1 --memory # Proximity graph WITH memory
    python improved_brownian.py 4 --memory # MST WITH memory
    python improved_brownian.py 5 --memory # All graphs WITH memory

Controls:
    ESC - Exit
    SPACE - Pause/Resume
    R - Reset simulation
    M - Toggle memory on/off during simulation
    1-5 - Switch graph type
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

from graphizy import (
    Graphing, GraphizyConfig, generate_positions
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class ImprovedBrownianSimulation:
    """
    Improved Brownian motion simulation where memory is a modifier, not a separate graph type
    Now includes Minimum Spanning Tree (MST) as a graph type!
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
        self.delaunay_update_frequency = 3
        self.knn_k = 4

        # Display parameters
        self.window_name = "Graphizy - Improved Brownian Motion with MST"
        self.paused = False
        self.current_graph_type = 1

        # Graph type definitions (memory is NOT a separate type)
        self.graph_type_names = {
            1: "Proximity Graph",
            2: "Delaunay Triangulation",
            3: "K-Nearest Neighbors",
            4: "Minimum Spanning Tree",  # NEW!
            5: "Combined View"
        }

        print("Initializing Improved Brownian simulation with MST support...")
        print("Memory is treated as a MODIFIER that can be applied to any graph type")
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
        """Setup graphers for different visualization styles"""
        base_config = GraphizyConfig()
        base_config.graph.dimension = (self.width, self.height)
        base_config.drawing.point_radius = 8
        base_config.drawing.line_thickness = 2

        # Create graphers with distinct colors for each base type
        self.graphers = {
            'proximity': self._create_grapher(base_config, (0, 0, 255), (255, 255, 255)),  # Red
            'delaunay': self._create_grapher(base_config, (0, 255, 0), (255, 255, 0)),  # Green
            'knn': self._create_grapher(base_config, (255, 100, 0), (100, 255, 255)),  # Blue
            'mst': self._create_grapher(base_config, (255, 0, 255), (255, 255, 100)),  # Purple (NEW!)
        }

        # Initialize memory managers for each grapher if memory is enabled
        if self.use_memory:
            self._initialize_memory_managers()

    def _initialize_memory_managers(self):
        """Initialize memory managers for all graphers"""
        for grapher in self.graphers.values():
            grapher.init_memory_manager(
                max_memory_size=self.memory_size,
                max_iterations=None,  # Keep all history
                track_edge_ages=True
            )
        print(f"Memory managers initialized with size {self.memory_size}")

    def _create_grapher(self, base_config: GraphizyConfig, line_color: tuple, point_color: tuple) -> Graphing:
        """Create a styled Graphing instance"""
        config = base_config.copy()
        config.drawing.line_color = line_color
        config.drawing.point_color = point_color
        return Graphing(config=config)

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

    def create_base_graph(self, graph_type: str) -> Any:
        """Create base graph of specified type (without memory)"""
        grapher = self.graphers[graph_type]

        if graph_type == 'proximity':
            return grapher.make_proximity(self.particle_stack, proximity_thresh=self.proximity_threshold)

        elif graph_type == 'delaunay':
            # Update Delaunay less frequently for performance
            if self.iteration % self.delaunay_update_frequency == 0:
                graph = grapher.make_delaunay(self.particle_stack)
                setattr(self, f'_last_{graph_type}', graph)
                return graph
            else:
                return getattr(self, f'_last_{graph_type}', None)

        elif graph_type == 'knn':
            # Create K-nearest neighbors graph
            return self._create_knn_graph(grapher)

        elif graph_type == 'mst':
            # Create Minimum Spanning Tree graph (NEW!)
            return self._create_mst_graph(grapher)

        return None

    def create_memory_enhanced_graph(self, graph_type: str) -> Any:
        """Create graph with memory enhancement"""
        grapher = self.graphers[graph_type]

        # First create the base graph
        base_graph = self.create_base_graph(graph_type)

        if base_graph is None or grapher.memory_manager is None:
            return base_graph

        # Update memory with current base graph connections
        grapher.update_memory_with_graph(base_graph)

        # Create memory-enhanced graph (current positions + memory connections)
        memory_graph = grapher.make_memory_graph(self.particle_stack)

        return memory_graph

    def _create_knn_graph(self, grapher: Graphing) -> Any:
        """Create K-nearest neighbors graph"""
        try:
            # Simple KNN implementation
            from scipy.spatial.distance import cdist

            # Create base graph structure
            graph = grapher.make_proximity(self.particle_stack, proximity_thresh=float('inf'))  # Start with empty graph
            graph.delete_edges(graph.es)  # Remove all edges

            # Extract positions
            positions = self.particle_stack[:, 1:3]

            # Calculate distances
            distances = cdist(positions, positions)

            # Find k nearest neighbors for each point
            edges_to_add = []
            for i, row in enumerate(distances):
                # Get indices of k+1 nearest (including self), then exclude self
                nearest_indices = np.argsort(row)[:self.knn_k + 1]
                nearest_indices = nearest_indices[nearest_indices != i][:self.knn_k]

                for j in nearest_indices:
                    edge = tuple(sorted([i, j]))
                    edges_to_add.append(edge)

            # Remove duplicates and add edges
            unique_edges = list(set(edges_to_add))
            if unique_edges:
                graph.add_edges(unique_edges)

            return graph

        except ImportError:
            # Fallback to proximity if scipy not available
            print("SciPy not available for KNN, using proximity instead")
            return grapher.make_proximity(self.particle_stack, proximity_thresh=self.proximity_threshold * 0.7)
        except Exception as e:
            print(f"KNN creation failed: {e}, using proximity fallback")
            return grapher.make_proximity(self.particle_stack, proximity_thresh=self.proximity_threshold * 0.7)

    def _create_mst_graph(self, grapher: Graphing) -> Any:
        """Create Minimum Spanning Tree graph (NEW!)"""
        try:
            # Use the new MST functionality from the main Graphing class
            return grapher.make_mst(self.particle_stack)

        except Exception as e:
            print(f"MST creation failed: {e}, using proximity fallback")
            # Fallback to proximity if MST fails
            return grapher.make_proximity(self.particle_stack, proximity_thresh=self.proximity_threshold * 0.8)

    def create_visualization(self, graph_type: int) -> Optional[np.ndarray]:
        """Create visualization for specified graph type (with or without memory)"""
        if graph_type == 5:  # Combined view
            return self._create_combined_view()

        # Map graph type number to string
        type_map = {1: 'proximity', 2: 'delaunay', 3: 'knn', 4: 'mst'}
        graph_type_str = type_map.get(graph_type)

        if not graph_type_str:
            return None

        # Create graph with or without memory
        if self.use_memory:
            graph = self.create_memory_enhanced_graph(graph_type_str)
        else:
            graph = self.create_base_graph(graph_type_str)

        if graph is None:
            return None

        # Draw the graph
        grapher = self.graphers[graph_type_str]

        # Special drawing for memory-enhanced graphs
        if self.use_memory and grapher.memory_manager is not None:
            try:
                return grapher.draw_memory_graph(graph, use_age_colors=True, alpha_range=(0.3, 1.0))
            except:
                # Fallback to regular drawing
                return grapher.draw_graph(graph)
        else:
            return grapher.draw_graph(graph)

    def _create_combined_view(self) -> Optional[np.ndarray]:
        """Create combined view showing all graph types (2x2 grid)"""
        images = []

        for graph_type in [1, 2, 3, 4]:  # proximity, delaunay, knn, mst
            img = self.create_visualization(graph_type)
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
        bottom_row = np.hstack([images[2], images[3]])  # KNN + MST
        combined = np.vstack([top_row, bottom_row])

        return combined

    def add_info_overlay(self, image: np.ndarray, graph_type: int) -> np.ndarray:
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
        cv2.putText(img_with_overlay, f"Iteration: {self.iteration}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
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
            self._initialize_memory_managers()
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
            self._initialize_memory_managers()  # Reinitialize with new size
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
                self._initialize_memory_managers()
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
                image = self.create_visualization(self.current_graph_type)

                if image is not None:
                    display_image = self.add_info_overlay(image, self.current_graph_type)
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
    """Parse arguments and run simulation"""
    parser = argparse.ArgumentParser(description="Improved Graphizy Brownian Motion with MST Support")

    # Graph type selection
    parser.add_argument('graph_type', type=int, nargs='?', default=1,
                        help='Graph type (1=Proximity, 2=Delaunay, 3=KNN, 4=MST, 5=Combined)')

    # Memory modifier
    parser.add_argument('--memory', '-m', action='store_true',
                        help='Enable memory tracking for the selected graph type')
    parser.add_argument('--memory-size', type=int, default=25,
                        help='Memory buffer size (default: 25)')

    # Simulation parameters
    parser.add_argument('--iterations', '-i', type=int, default=100000,
                        help='Maximum iterations (default: 1000)')
    parser.add_argument('--particles', '-p', type=int, default=50,
                        help='Number of particles (default: 50)')
    parser.add_argument('--size', nargs=2, type=int, default=[800, 600],
                        help='Canvas size [width height] (default: 800 600)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Display FPS (default: 30)')

    args = parser.parse_args()

    # Validate graph type
    if args.graph_type not in [1, 2, 3, 4, 5]:
        print("Error: Graph type must be 1 (Proximity), 2 (Delaunay), 3 (KNN), 4 (MST), or 5 (Combined)")
        sys.exit(1)

    try:
        simulation = ImprovedBrownianSimulation(
            width=args.size[0],
            height=args.size[1],
            num_particles=args.particles,
            use_memory=args.memory,
            memory_size=args.memory_size
        )

        simulation.run_simulation(
            graph_type=args.graph_type,
            max_iterations=args.iterations,
            fps=args.fps
        )

        print("Simulation completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()