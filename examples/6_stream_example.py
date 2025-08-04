#!/usr/bin/env python3
"""
Real-time Streaming Examples for Graphizy - Updated for v0.1.17+

This script demonstrates the advanced streaming capabilities of Graphizy.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL2 or later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import numpy as np
import time
import logging
import sys
import asyncio
from typing import Dict, Any

from graphizy import (
    Graphing, GraphizyConfig, generate_and_format_positions,
    validate_graphizy_input, GraphizyError
)
from graphizy.utils import setup_output_directory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class DataGenerator:
    """Enhanced data generator for real-time streaming simulations."""
    
    def __init__(self, canvas_size: tuple = (800, 600), num_particles: int = 50):
        self.canvas_size = canvas_size
        self.num_particles = num_particles
        self.current_positions = None
        self.time_step = 0
        self.initialize_particles()
    
    def initialize_particles(self):
        """Initialize particle positions."""
        self.current_positions = generate_and_format_positions(
            self.canvas_size[0], self.canvas_size[1], self.num_particles
        )
        validate_graphizy_input(self.current_positions)
    
    def generate_brownian_motion_data(self, movement_scale: float = 3.0) -> np.ndarray:
        """Generate data with Brownian motion movement."""
        if self.current_positions is None:
            self.initialize_particles()
        
        # Apply Brownian motion
        movement = np.random.normal(0, movement_scale, (self.num_particles, 2))
        self.current_positions[:, 1:3] += movement
        
        # Boundary conditions with soft reflection
        margin = 50
        for i in range(self.num_particles):
            # X boundary
            if self.current_positions[i, 1] < margin:
                self.current_positions[i, 1] = margin + abs(self.current_positions[i, 1] - margin)
            elif self.current_positions[i, 1] > self.canvas_size[0] - margin:
                self.current_positions[i, 1] = self.canvas_size[0] - margin - abs(self.current_positions[i, 1] - (self.canvas_size[0] - margin))
            
            # Y boundary
            if self.current_positions[i, 2] < margin:
                self.current_positions[i, 2] = margin + abs(self.current_positions[i, 2] - margin)
            elif self.current_positions[i, 2] > self.canvas_size[1] - margin:
                self.current_positions[i, 2] = self.canvas_size[1] - margin - abs(self.current_positions[i, 2] - (self.canvas_size[1] - margin))
        
        self.time_step += 1
        return self.current_positions.copy()


def example_basic_streaming():
    """
    Demonstrates basic real-time streaming with multiple graph types
    and automatic visualization updates.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: BASIC STREAMING WITH MULTIPLE GRAPH TYPES")
    print("=" * 60)

    try:
        # Initialize graphing system
        config = GraphizyConfig(dimension=(600, 500))
        grapher = Graphing(config=config)
        
        # Set up multiple graph types for streaming
        grapher.set_graph_type(['proximity', 'delaunay'])
        grapher.update_graph_params('proximity', proximity_thresh=80.0)
        
        # Initialize memory system for temporal analysis
        grapher.init_memory_manager(
            max_memory_size=100,
            max_iterations=20,
            track_edge_ages=True
        )
        
        print("Streaming system configured:")
        print(f"  Graph types: {grapher.graph_types}")
        print(f"  Memory enabled: max_size=100, max_iterations=20")

        # Create stream manager
        stream_manager = grapher.create_stream_manager(
            buffer_size=500,        # Large buffer for smooth streaming
            update_interval=0.1,    # 10 FPS update rate
            auto_memory=True        # Automatic memory updates
        )
        
        print(f"Stream manager created: buffer_size=500, update_rate=10 FPS")

        # Set up data generator
        data_generator = DataGenerator(canvas_size=(600, 500), num_particles=35)
        
        # Visualization callback
        frame_count = 0
        output_dir = setup_output_directory()
        
        def visualization_callback(graphs: Dict[str, Any]):
            """Callback for real-time visualization updates."""
            nonlocal frame_count
            
            try:
                if 'proximity' in graphs and graphs['proximity'] is not None:
                    # Create visualization with memory effects if available
                    if hasattr(grapher, 'memory_manager') and grapher.memory_manager:
                        image = grapher.draw_memory_graph(
                            graphs['proximity'],
                            use_age_colors=True,
                            alpha_range=(0.4, 1.0)
                        )
                    else:
                        image = grapher.draw_graph(graphs['proximity'])
                    
                    # Display real-time visualization
                    grapher.show_graph(image, "Real-time Proximity Graph", block=False)
                    
                    # Save frames periodically
                    if frame_count % 30 == 0:  # Every 3 seconds at 10 FPS
                        filename = f"streaming_frame_{frame_count:04d}.jpg"
                        grapher.save_graph(image, str(output_dir / filename))
                        print(f"    Saved frame {frame_count}")
                
                frame_count += 1
                
            except Exception as e:
                logging.error(f"Visualization callback error: {e}")

        # Add callback to stream manager
        stream_manager.add_callback(visualization_callback)
        
        # Performance monitoring callback
        performance_stats = []
        
        def performance_callback(graphs: Dict[str, Any]):
            """Monitor streaming performance."""
            current_time = time.time()
            successful_graphs = sum(1 for g in graphs.values() if g is not None)
            
            performance_stats.append({
                'timestamp': current_time,
                'successful_graphs': successful_graphs,
                'total_requested': len(graphs)
            })
            
            # Print performance summary every 50 frames
            if len(performance_stats) % 50 == 0:
                recent_stats = performance_stats[-50:]
                success_rate = np.mean([s['successful_graphs'] / s['total_requested'] 
                                      for s in recent_stats if s['total_requested'] > 0])
                print(f"    Performance: {success_rate:.1%} success rate over last 50 frames")

        stream_manager.add_callback(performance_callback)

        # Start streaming
        print("\nStarting real-time streaming...")
        stream_manager.start_streaming()
        
        # Simulate real-time data for 10 seconds
        simulation_duration = 10.0  # seconds
        fps = 20  # Data generation rate
        frame_time = 1.0 / fps
        
        start_time = time.time()
        generated_frames = 0
        
        try:
            while time.time() - start_time < simulation_duration:
                # Generate new data
                new_data = data_generator.generate_brownian_motion_data(movement_scale=4.0)
                
                # Push to stream with error handling
                success = stream_manager.push_data(new_data)
                if not success:
                    print(f"    Warning: Frame {generated_frames} dropped (buffer full)")
                
                generated_frames += 1
                
                # Control data generation rate
                time.sleep(frame_time)
                
                # Status update every 2 seconds
                if generated_frames % (fps * 2) == 0:
                    elapsed = time.time() - start_time
                    print(f"    Generated {generated_frames} frames in {elapsed:.1f}s")

        except KeyboardInterrupt:
            print("\n    Streaming interrupted by user")

        # Stop streaming and get final statistics
        print("\nStopping streaming...")
        stream_manager.stop_streaming()
        
        # Final performance analysis
        if performance_stats:
            total_frames = len(performance_stats)
            overall_success_rate = np.mean([s['successful_graphs'] / s['total_requested'] 
                                          for s in performance_stats if s['total_requested'] > 0])
            
            print(f"\nBasic Streaming Results:")
            print(f"  Total frames processed: {total_frames}")
            print(f"  Generated frames: {generated_frames}")
            print(f"  Overall success rate: {overall_success_rate:.1%}")
            print(f"  Visualization frames saved: {frame_count // 30}")

        print("Basic streaming example completed successfully!")

    except Exception as e:
        print(f"Basic streaming failed: {e}")
        import traceback
        traceback.print_exc()


async def example_async_streaming():
    """
    Demonstrates asynchronous streaming for high-performance applications.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: ASYNCHRONOUS HIGH-PERFORMANCE STREAMING")
    print("=" * 60)

    try:
        # Initialize high-performance graphing system
        config = GraphizyConfig(dimension=(800, 600))
        grapher = Graphing(config=config)
        
        # Configure for high-performance streaming
        grapher.set_graph_type(['proximity', 'delaunay'])
        grapher.update_graph_params('proximity', proximity_thresh=70.0)
        
        # Initialize optimized memory system
        grapher.init_memory_manager(
            max_memory_size=50,     # Smaller for performance
            max_iterations=15,
            track_edge_ages=False   # Disable for speed
        )
        
        print("High-performance streaming system configured")

        # Create async stream manager
        async_stream_manager = grapher.create_async_stream_manager(buffer_size=1000)
        
        # Set up high-speed data generator
        data_generator = DataGenerator(canvas_size=(800, 600), num_particles=60)
        
        # Performance monitoring
        processing_times = []
        
        async def high_speed_data_producer():
            """Asynchronous data producer at high frame rate."""
            print("  Starting high-speed data producer...")
            
            frames_produced = 0
            start_time = time.time()
            
            try:
                for _ in range(200):  # Generate 200 frames
                    # Generate data
                    new_data = data_generator.generate_brownian_motion_data(movement_scale=3.0)
                    
                    # Push to async stream
                    await async_stream_manager.push_data_async(new_data)
                    frames_produced += 1
                    
                    # High-frequency generation (50 FPS)
                    await asyncio.sleep(0.02)
                    
                    # Progress reporting
                    if frames_produced % 50 == 0:
                        elapsed = time.time() - start_time
                        fps = frames_produced / elapsed
                        print(f"    Producer: {frames_produced} frames, {fps:.1f} FPS")

            except Exception as e:
                print(f"    Data producer error: {e}")
            
            print(f"  Data producer completed: {frames_produced} frames")

        async def high_speed_processor():
            """Asynchronous graph processor with performance monitoring."""
            print("  Starting high-speed processor...")
            
            processed_frames = 0
            start_time = time.time()
            
            try:
                async for graphs in async_stream_manager.process_stream_async():
                    process_start = time.perf_counter()
                    
                    # Process graphs
                    successful_processing = 0
                    
                    for graph_type, graph in graphs.items():
                        if graph is not None:
                            try:
                                # Perform real-time analysis
                                info = grapher.get_graph_info(graph)
                                successful_processing += 1
                                
                            except Exception as e:
                                logging.debug(f"Graph processing error: {e}")
                    
                    process_end = time.perf_counter()
                    processing_time = (process_end - process_start) * 1000  # milliseconds
                    processing_times.append(processing_time)
                    
                    processed_frames += 1
                    
                    # Throughput monitoring
                    if processed_frames % 25 == 0:
                        elapsed = time.time() - start_time
                        fps = processed_frames / elapsed
                        avg_processing_time = np.mean(processing_times[-25:])
                        
                        print(f"    Processor: {processed_frames} frames, {fps:.1f} FPS, "
                              f"{avg_processing_time:.2f}ms avg processing")
                    
                    # Prevent overwhelming the system
                    if processed_frames >= 180:  # Stop before producer finishes
                        break

            except Exception as e:
                print(f"    Processor error: {e}")
                
            print(f"  Processor completed: {processed_frames} frames")

        # Run concurrent producer and processor
        print("\nStarting concurrent async streaming...")
        
        await asyncio.gather(
            high_speed_data_producer(),
            high_speed_processor()
        )
        
        # Performance analysis
        print(f"\nAsynchronous Streaming Performance Analysis:")
        
        if processing_times:
            print(f"  Processing time statistics:")
            print(f"    Mean: {np.mean(processing_times):.2f} ms")
            print(f"    Median: {np.median(processing_times):.2f} ms")
            print(f"    95th percentile: {np.percentile(processing_times, 95):.2f} ms")
            print(f"    Max: {np.max(processing_times):.2f} ms")

        print("Asynchronous streaming example completed successfully!")

    except Exception as e:
        print(f"Asynchronous streaming failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Runs all the streaming examples using the latest API features.
    """
    print("Graphizy Real-time Streaming Examples - v0.1.17+ Edition")
    print("=" * 70)

    try:
        # Run streaming examples
        example_basic_streaming()
        
        # Run async example
        print("\nRunning asynchronous streaming example...")
        asyncio.run(example_async_streaming())

        print("\n" + "=" * 70)
        print("All streaming examples completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  • Real-time streaming with multiple graph types")
        print("  • Asynchronous high-performance streaming")
        print("  • Integration with memory and weight systems")
        print("  • Performance optimization and monitoring")
        print("  • Live visualization with automatic updates")

    except Exception as e:
        print(f"\nStreaming examples failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
