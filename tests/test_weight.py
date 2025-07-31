# tests/test_weight.py

import pytest
import numpy as np
import igraph as ig
import logging

# Import components from the graphizy package
from graphizy import Graphing, GraphizyConfig
from graphizy.weight import (
    WeightComputer,
    FastAttributeComputer,
    create_distance_computer,
    create_weight_from_distance_computer,
    setup_realtime_weight_computer,
    create_age_weight_computer,
)
from graphizy.exceptions import GraphCreationError


# Note: Fixtures like `weight_test_graph` and `graph_with_age` are now
# automatically available from `tests/conftest.py`.

class TestWeightComputer:
    """Tests for the main WeightComputer class."""

    def test_initialization(self):
        """Test WeightComputer initialization with various configurations."""
        wc = WeightComputer()
        assert wc.method == "distance"
        assert wc.target_attribute == "distance"

        wc_custom = WeightComputer(
            method="formula", formula="1/dist", target_attribute="my_weight"
        )
        assert wc_custom.method == "formula"
        assert wc_custom.formula == "1/dist"
        assert wc_custom.target_attribute == "my_weight"

        with pytest.raises(ValueError, match="Invalid method"):
            WeightComputer(method="invalid_method")

        with pytest.raises(ValueError, match="Formula method requires 'formula' parameter"):
            WeightComputer(method="formula")

    def test_compute_distance_euclidean(self, weight_test_graph):
        """Test default Euclidean distance computation."""
        wc = WeightComputer(method="distance")
        g = wc.compute_weights(weight_test_graph)

        assert "distance" in g.es.attributes()
        distances = g.es["distance"]
        expected = [5.0, 10.0, 5.0, 10.0, 8.0622577, 8.9442719]
        np.testing.assert_allclose(distances, expected, rtol=1e-6)

    def test_compute_distance_manhattan(self, weight_test_graph):
        """Test Manhattan distance computation."""
        wc = WeightComputer(method="distance", distance_metric="manhattan")
        g = wc.compute_weights(weight_test_graph)

        assert "distance" in g.es.attributes()
        distances = g.es["distance"]
        expected = [7.0, 14.0, 7.0, 10.0, 11.0, 12.0]
        np.testing.assert_allclose(distances, expected, rtol=1e-6)

    def test_compute_formula(self, weight_test_graph):
        """Test formula-based weight computation."""
        wc_dist = WeightComputer(method="distance")
        g = wc_dist.compute_weights(weight_test_graph)

        wc_weight = WeightComputer(
            method="formula", formula="1 / (distance + 1)", target_attribute="weight"
        )
        g = wc_weight.compute_weights(g)

        assert "weight" in g.es.attributes()
        expected_distances = np.array(g.es["distance"])
        expected_weights = 1 / (expected_distances + 1)
        np.testing.assert_allclose(g.es["weight"], expected_weights, rtol=1e-6)

    def test_error_on_missing_attribute_in_formula(self, weight_test_graph, caplog):
        """Test that a formula fails gracefully if a required attribute is missing."""
        wc = WeightComputer(
            method="formula", formula="distance * age", default_value=-1.0
        )
        g = WeightComputer(method="distance").compute_weights(weight_test_graph)

        with caplog.at_level(logging.WARNING):
            g = wc.compute_weights(g)
            assert "failed for edge" in caplog.text

        assert "weight" in g.es.attributes()
        assert all(w == -1.0 for w in g.es["weight"])


class TestConvenienceConstructors:
    """Tests for the create_*_computer factory functions."""

    def test_create_distance_computer(self, weight_test_graph):
        """Test the distance computer constructor."""
        computer = create_distance_computer(metric="manhattan")
        g = computer.compute_weights(weight_test_graph)
        assert "distance" in g.es.attributes()
        expected = [7.0, 14.0, 7.0, 10.0, 11.0, 12.0]
        np.testing.assert_allclose(g.es["distance"], expected, rtol=1e-6)

    def test_create_weight_from_distance_computer(self, weight_test_graph):
        """Test the weight-from-distance computer constructor."""
        g = WeightComputer(method="distance").compute_weights(weight_test_graph)
        computer = create_weight_from_distance_computer(formula="10 - distance")
        g = computer.compute_weights(g)
        assert "weight" in g.es.attributes()
        expected_distances = np.array(g.es["distance"])
        expected_weights = 10 - expected_distances
        np.testing.assert_allclose(g.es["weight"], expected_weights, rtol=1e-6)

    def test_create_age_weight_computer(self, graph_with_age):
        """Test the age-based weight computer constructor."""
        computer = create_age_weight_computer(age_mode="inverse", normalize=False)
        g = computer.compute_weights(graph_with_age)
        assert "weight" in g.es.attributes()
        expected = [10, 6, 9, 1, 8, 7]
        np.testing.assert_allclose(g.es["weight"], expected, rtol=1e-6)




class TestIntegrationWithGraphing:
    """Tests the integration of the weight system with the main Graphing class."""

    def test_auto_compute_weights_on_init(self, weight_test_data):
        """Test that weights are computed automatically if configured."""
        config = GraphizyConfig()
        config.weight.auto_compute_weights = True
        config.weight.weight_method = "distance"
        config.weight.weight_attribute = "dist_auto"

        grapher = Graphing(config=config)
        g = grapher.make_graph("proximity", weight_test_data, proximity_thresh=12.0)

        assert "dist_auto" in g.es.attributes()

        # FIX: Sort both actual and expected values to make the test robust to edge order
        actual_distances = sorted(g.es["dist_auto"])
        expected_distances = sorted([5.0, 10.0, 5.0, 10.0, 8.0622577, 8.9442719])

        np.testing.assert_allclose(actual_distances, expected_distances, rtol=1e-6)

    def test_maybe_compute_weights_pipeline(self, weight_test_data):
        """Test the full make_graph pipeline with weight computation."""
        grapher = Graphing()
        grapher.init_weight_computer(method="formula", formula="1 / (distance + 1)")
        g = grapher.make_graph("proximity", weight_test_data, proximity_thresh=12.0, compute_weights=True)

        assert "distance" in g.es.attributes()
        assert "weight" in g.es.attributes()

        distances = np.array(g.es["distance"])
        expected_weights = 1 / (distances + 1)
        np.testing.assert_allclose(g.es["weight"], expected_weights, rtol=1e-6)