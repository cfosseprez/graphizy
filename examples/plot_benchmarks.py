#!/usr/bin/env python3
"""
Benchmark Results Plotter for Graphizy

This script reads the JSON output from the benchmark comparison and generates
a publication-quality bar chart to visualize the performance differences.

.. moduleauthor:: Charles Fosseprez
.. contact:: charles.fosseprez.pro@gmail.com
.. license:: GPL-2.0-or-later
.. copyright:: Copyright (C) 2025 Charles Fosseprez
"""

import json

import numpy as np
from pathlib import Path


def plot_benchmark_results(data_file: Path = Path('benchmark_results.json')):
    """
    Loads benchmark data and generates a comparative plot.
    """

    import pandas as pd
    import matplotlib.pyplot as plt

    if not data_file.exists():
        print(f"Error: Benchmark data file not found at '{data_file}'")
        print("Please run 'python examples/benchmark_comparison.py' first.")
        return

    with open(data_file, 'r') as f:
        results = json.load(f)

    # Convert to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(results['benchmarks'])

    # Prepare data for plotting
    df_melted = df.melt(
        id_vars=['nodes', 'type'],
        value_vars=['graphizy_total_time', 'networkx_total_time'],
        var_name='library',
        value_name='time_ms'
    )
    df_melted['library'] = df_melted['library'].apply(
        lambda x: 'Graphizy' if 'graphizy' in x else 'NetworkX+SciPy'
    )

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define positions for the bars
    node_counts = sorted(df['nodes'].unique())
    x = np.arange(len(node_counts))
    width = 0.2
    graph_types = sorted(df['type'].unique())
    num_types = len(graph_types)

    # Colors and patterns for clarity
    colors = {'Graphizy': '#4CAF50', 'NetworkX+SciPy': '#F44336'}
    hatches = {'Delaunay': '', 'k-NN': '///', 'Proximity': '...'}

    for i, g_type in enumerate(graph_types):
        offset = (i - num_types / 2 + 0.5) * width
        
        # Graphizy bars
        graphizy_times = df_melted[(df_melted['type'] == g_type) & (df_melted['library'] == 'Graphizy')].set_index('nodes').loc[node_counts]['time_ms']
        rects1 = ax.bar(x + offset - width/2, graphizy_times, width, label=f'Graphizy ({g_type})' if i == 0 else "", color=colors['Graphizy'], hatch=hatches.get(g_type, ''))

        # NetworkX bars
        networkx_times = df_melted[(df_melted['type'] == g_type) & (df_melted['library'] == 'NetworkX+SciPy')].set_index('nodes').loc[node_counts]['time_ms']
        rects2 = ax.bar(x + offset + width/2, networkx_times, width, label=f'NetworkX+SciPy ({g_type})' if i == 0 else "", color=colors['NetworkX+SciPy'], hatch=hatches.get(g_type, ''))

    # --- Formatting for Publication Quality ---
    ax.set_ylabel('Total Workflow Time (ms)')
    ax.set_xlabel('Number of Nodes')
    ax.set_title('Graphizy vs. NetworkX+SciPy: End-to-End Performance', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(node_counts)
    ax.set_yscale('log')  # Log scale is essential for visualizing large performance gaps
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['Graphizy'], label='Graphizy'),
        Patch(facecolor=colors['NetworkX+SciPy'], label='NetworkX+SciPy'),
        Patch(facecolor='white', edgecolor='black', hatch=hatches['Delaunay'], label='Delaunay'),
        Patch(facecolor='white', edgecolor='black', hatch=hatches['k-NN'], label='k-NN'),
        Patch(facecolor='white', edgecolor='black', hatch=hatches['Proximity'], label='Proximity')
    ]
    ax.legend(handles=legend_elements, loc='upper left', title='Legend')

    fig.tight_layout()

    # Save the figure
    output_filename = 'benchmark_comparison.png'
    plt.savefig(output_filename, dpi=300)
    print(f"\nPlot saved to '{output_filename}'")


def main():
    """Main function to run the plotter."""
    print("=" * 60)
    print("Generating Benchmark Comparison Plot")
    print("=" * 60)
    plot_benchmark_results()


if __name__ == "__main__":
    main()