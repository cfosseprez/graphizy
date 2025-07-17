#!/usr/bin/env python3
"""
Example script demonstrating the fixed graphizy package with memory functionality

This shows:
1. Basic memory manager usage
2. Memory graph creation 
3. Integration with the main Graphing class
"""

import sys
import os
import numpy as np

# Add the src directory to the path 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def example_basic_memory():
    """Basic memory manager example"""
    print("=== Basic Memory Manager Example ===")
    
    from graphizy import MemoryManager
    
    # Create memory manager
    memory_mgr = MemoryManager(max_memory_size=10, max_iterations=5, track_edge_ages=True)
    
    # Simulate adding connections over several iterations
    for iteration in range(3):
        connections = {
            "A": ["B", "C"] if iteration % 2 == 0 else ["C"],
            "B": ["A"] if iteration % 2 == 0 else [],
            "C": ["A", "D"],
            "D": ["C"]
        }
        memory_mgr.add_connections(connections)
        print(f"Iteration {iteration + 1}: Added connections {connections}")
    
    # Get final memory state
    final_memory = memory_mgr.get_current_memory_graph()
    stats = memory_mgr.get_memory_stats()
    
    print(f"Final memory: {final_memory}")
    print(f"Stats: {stats}")
    print()

def example_memory_graph():
    """Memory graph creation example"""
    print("=== Memory Graph Creation Example ===")
    
    from graphizy import create_memory_graph, MemoryManager
    
    # Create test positions
    positions = np.array([
        [1, 100, 100],  # Object 1 at (100, 100)
        [2, 200, 150],  # Object 2 at (200, 150) 
        [3, 120, 300],  # Object 3 at (120, 300)
        [4, 400, 100],  # Object 4 at (400, 100)
    ])
    
    # Create memory connections (from past interactions)
    memory_connections = {
        "1": ["3", "4"],  # Object 1 remembers being connected to 3 and 4
        "2": [],          # Object 2 has no memory connections
        "3": ["1"],       # Object 3 remembers being connected to 1
        "4": ["1"]        # Object 4 remembers being connected to 1
    }
    
    # Create memory graph
    graph = create_memory_graph(positions, memory_connections, aspect="array")
    
    print(f"Created memory graph:")
    print(f"  Vertices: {graph.vcount()}")
    print(f"  Edges: {graph.ecount()}")
    print(f"  Vertex IDs: {list(graph.vs['id'])}")
    
    # Show which vertices are connected
    for edge in graph.es:
        v1_id = graph.vs[edge.tuple[0]]["id"]
        v2_id = graph.vs[edge.tuple[1]]["id"] 
        print(f"  Edge: {v1_id} <-> {v2_id}")
    print()

def example_graphing_integration():
    """Full integration with Graphing class"""
    print("=== Graphing Class Integration Example ===")
    
    from graphizy import Graphing, GraphizyConfig
    
    # Create configuration
    config = GraphizyConfig()
    config.memory.max_memory_size = 20
    config.memory.max_iterations = 10
    
    # Create grapher
    grapher = Graphing(config=config)
    
    # Initialize memory manager
    memory_mgr = grapher.init_memory_manager(max_memory_size=20, max_iterations=5)
    print(f"Initialized memory manager: {memory_mgr is not None}")
    
    # Create test data 
    particle_stack = np.array([
        [0, 100, 100],
        [1, 200, 150],
        [2, 300, 100],
        [3, 150, 200]
    ])
    
    # Simulate multiple iterations with different proximities
    for i in range(3):
        print(f"\nIteration {i + 1}:")
        
        # Add some movement (except first iteration)
        if i > 0:
            # Small random movement
            particle_stack[:, 1:3] += np.random.normal(0, 10, (len(particle_stack), 2))
            # Keep within reasonable bounds
            particle_stack[:, 1] = np.clip(particle_stack[:, 1], 50, 350)
            particle_stack[:, 2] = np.clip(particle_stack[:, 2], 50, 250)
        
        # Update memory with current proximities
        connections = grapher.update_memory_with_proximity(particle_stack, 100.0)
        
        total_connections = sum(len(conns) for conns in connections.values()) // 2
        print(f"  Current proximity connections: {total_connections}")
        print(f"  Positions: {[(int(p[1]), int(p[2])) for p in particle_stack]}")
    
    # Create final memory graph
    memory_graph = grapher.make_memory_graph(particle_stack)
    
    # Get statistics
    stats = grapher.get_memory_stats()
    
    print(f"\nFinal Results:")
    print(f"  Memory stats: {stats}")
    print(f"  Memory graph vertices: {memory_graph.vcount()}")
    print(f"  Memory graph edges: {memory_graph.ecount()}")
    
    # Test drawing the memory graph
    try:
        image = grapher.draw_memory_graph(memory_graph, use_age_colors=True)
        print(f"  Successfully created memory graph image: {image.shape}")
    except Exception as e:
        print(f"  Warning: Could not create image (likely missing OpenCV): {e}")
    
    print()

def example_cli_config():
    """Test CLI configuration handling"""
    print("=== CLI Configuration Example ===")
    
    from graphizy.cli import create_parser, create_config_from_args
    
    # Test memory command parsing
    parser = create_parser()
    
    # Simulate CLI arguments for memory command
    test_args = [
        'memory', '--size', '800', '--particles', '50',
        '--memory-size', '30', '--memory-iterations', '8',
        '--iterations', '5', '--proximity-thresh', '60'
    ]
    
    args = parser.parse_args(test_args)
    config = create_config_from_args(args)
    
    print(f"CLI parsed memory command successfully:")
    print(f"  Canvas size: {config.graph.dimension}")
    print(f"  Particles: {config.generation.num_particles}")
    print(f"  Memory size: {config.memory.max_memory_size}")
    print(f"  Memory iterations: {config.memory.max_iterations}")
    print()

def main():
    """Run all examples"""
    print("🧠 Graphizy Memory Functionality Examples\n")
    
    try:
        example_basic_memory()
        example_memory_graph()
        example_graphing_integration()
        example_cli_config()
        
        print("🎉 All examples completed successfully!")
        print("\nThe graphizy package has been successfully fixed and enhanced with memory functionality.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the graphizy root directory.")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
