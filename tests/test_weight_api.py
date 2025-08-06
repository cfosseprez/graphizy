"""
Tests for the weight computation API and functionality.
"""

import pytest
from graphizy import GraphizyConfig, Graphing, generate_and_format_positions


def test_weight_computer_initialization():
    """Test that the weight computer can be initialized via GraphizyConfig."""
    config = GraphizyConfig(dimension=(300, 300))
    config.weight.auto_compute_weights = True
    config.weight.weight_method = "distance"
    config.weight.weight_attribute = "test_weight"

    grapher = Graphing(config=config)
    assert grapher.weight_computer is not None
    assert grapher.weight_computer.method == "distance"


def test_auto_weight_computation_on_creation():
    """Test that weights are automatically computed when a graph is created."""
    config = GraphizyConfig(dimension=(300, 300))
    config.weight.auto_compute_weights = True
    config.weight.weight_method = "distance"
    config.weight.weight_attribute = "distance_weight"
    grapher = Graphing(config=config)

    data = generate_and_format_positions(300, 300, 20)
    graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)

    assert graph is not None
    assert graph.ecount() > 0
    assert "distance_weight" in graph.es.attributes()
    assert len(graph.es["distance_weight"]) == graph.ecount()


def test_formula_based_weight_computation():
    """Test that formula-based weights are computed correctly."""
    config = GraphizyConfig(dimension=(300, 300))
    config.weight.auto_compute_weights = True
    config.weight.weight_method = "formula"
    config.weight.weight_formula = "1 / (distance + 0.1)"
    config.weight.weight_attribute = "formula_weight"
    grapher = Graphing(config=config)

    data = generate_and_format_positions(300, 300, 20)
    graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)

    assert "formula_weight" in graph.es.attributes()


def test_manual_edge_attribute_computation():
    """Test the standalone compute_edge_attribute method."""
    grapher = Graphing(GraphizyConfig(dimension=(300, 300)))
    data = generate_and_format_positions(300, 300, 20)
    graph = grapher.make_graph("proximity", data, proximity_thresh=50.0)

    grapher.compute_edge_attribute(graph, "custom_attr", method="formula", formula="distance * 2")
    assert "custom_attr" in graph.es.attributes()
