#!/usr/bin/env python3
"""
Improved Interactive Brownian Motion Viewer for Graphizy

This version treats memory as a MODIFIER that can be applied to any graph type,
including the new Minimum Spanning Tree (MST) graph type.

Graph Types:
    1 - Proximity Graph
    2 - Delaunay Triangulation
    3 - Gabriel Graph
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
    Graphing, GraphizyConfig, generate_positions, BrownianSimulator
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main():
    """Parse arguments and run simulation"""
    parser = argparse.ArgumentParser(description="Improved Graphizy Brownian Motion with MST Support")

    # Graph type selection
    parser.add_argument('graph_type', type=int, nargs='?', default=1,
                        help='Graph type (1=Proximity, 2=Delaunay, 3=Gabriel, 4=MST, 5=Combined)')

    # Memory modifier
    parser.add_argument('--memory', '-m', action='store_true',
                        help='Enable memory tracking for the selected graph type')
    parser.add_argument('--memory-size', type=int, default=3,
                        help='Memory buffer size (default: 25)')

    # Simulation parameters
    parser.add_argument('--iterations', '-i', type=int, default=100000,
                        help='Maximum iterations (default: 1000)')
    parser.add_argument('--particles', '-p', type=int, default=500,
                        help='Number of particles (default: 50)')
    parser.add_argument('--size', nargs=2, type=int, default=[800, 800],
                        help='Canvas size [width height] (default: 800 600)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Display FPS (default: 30)')

    args = parser.parse_args()

    # Validate graph type
    if args.graph_type not in [1, 2, 3, 4, 5]:
        print("Error: Graph type must be 1 (Proximity), 2 (Delaunay), 3 (Gabriel), 4 (MST), or 5 (Combined)")
        sys.exit(1)

    try:
        simulation = BrownianSimulator(
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