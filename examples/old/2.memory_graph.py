from graphizy import Graphing
import numpy as np

def raw():
    # Current positions
    positions = np.array([
        [1, 100, 100],  # Object A
        [2, 200, 150],  # Object B
        [3, 120, 300],  # Object C
        [4, 400, 100],  # Object D
    ])

    # Memory connections (historical)
    memory = {
        "1": ["3", "4"],  # A was connected to C and D
        "2": [],          # B has no connections
        "3": ["1"],       # C was connected to A
        "4": ["1"],       # D was connected to A
    }

    # Create grapher and memory graph
    grapher = Graphing(dimension=(500, 500))
    graph = grapher.make_memory_graph(positions, memory)

    # Draw and display
    image = grapher.draw_graph(graph)
    grapher.show_graph(image, "Memory Graph")

def main():
    # Initialize grapher with memory
    grapher = Graphing(dimension=(800, 600))
    grapher.init_memory_manager(max_memory_size=50, max_iterations=20)

    # Initial positions (use float dtype to allow movements)
    positions = np.array([
        [1, 100.0, 100.0],
        [2, 200.0, 150.0],
        [3, 300.0, 200.0],
        [4, 400.0, 250.0],
    ], dtype=float)

    # Simulate 15 iterations
    for iteration in range(15):
        # Add small movements
        if iteration > 0:
            movement = np.random.normal(0, 10, (4, 2))
            positions[:, 1:3] += movement
            # Keep within bounds
            positions[:, 1] = np.clip(positions[:, 1], 0, 800)
            positions[:, 2] = np.clip(positions[:, 2], 0, 600)

        # Update memory with current proximities
        current_connections = grapher.update_memory_with_proximity(positions, proximity_thresh=100)

        print(
            f"Iteration {iteration + 1}: {sum(len(conns) for conns in current_connections.values()) // 2} connections")

    # Create final memory graph
    memory_graph = grapher.make_memory_graph(positions)

    # Get statistics
    stats = grapher.get_memory_stats()
    print(f"Final memory: {stats}")

    # Visualize
    image = grapher.draw_graph(memory_graph)
    grapher.show_graph(image, "Memory Evolution")

if __name__ == "__main__":
    main()