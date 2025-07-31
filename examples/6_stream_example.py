grapher = Graphing()
grapher.set_graph_type(['proximity', 'delaunay'])

# Set up streaming
stream_manager = grapher.create_stream_manager(
    buffer_size=500,
    update_interval=0.05,  # 20 FPS
    auto_memory=True
)

# Add visualization callback
def visualize_update(graphs):
    if 'proximity' in graphs and graphs['proximity']:
        image = grapher.draw_graph(graphs['proximity'])
        grapher.show_graph(image, "Real-time Proximity", block=False)

stream_manager.add_callback(visualize_update)
stream_manager.start_streaming()

# Simulate real-time data
for t in range(1000):
    new_data = generate_dynamic_data(t)
    success = stream_manager.push_data(new_data)
    if not success:
        print("Warning: Dropped frame due to full buffer")
    time.sleep(0.02)  # 50 FPS data generation

stream_manager.stop_streaming()