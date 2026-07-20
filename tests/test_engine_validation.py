"""Focused regression and input-validation tests for core engine kernels."""

import numpy as np
import pytest

from planx.engine import demand, equity, graphs


def test_theil_zero_observation_emits_no_runtime_warning():
    with np.errstate(all="raise"):
        assert equity.theil_t([0.0, 2.0]) == pytest.approx(np.log(2.0))


@pytest.mark.parametrize(
    ("productions", "attractions", "cost", "message"),
    [
        ([1.0, 2.0], [3.0], [[1.0, 2.0]], "cost shape"),
        ([1.0, -2.0], [3.0], [[1.0], [2.0]], "P must contain"),
        ([1.0, 2.0], [3.0], [[1.0], [np.nan]], "cost must contain"),
    ],
)
def test_gravity_rejects_invalid_inputs(productions, attractions, cost, message):
    with pytest.raises(ValueError, match=message):
        demand.gravity(np.asarray(productions), np.asarray(attractions), np.asarray(cost), 0.1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "unknown"}, "kind must be"),
        ({"beta": -0.1}, "beta must be"),
        ({"max_iter": 0}, "max_iter must be"),
        ({"tol": 0.0}, "tol must be"),
    ],
)
def test_gravity_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        demand.gravity(np.ones(2), np.ones(2), np.ones((2, 2)), **{"beta": 0.1, **kwargs})


def test_mode_split_validates_sequence_lengths_and_shapes():
    with pytest.raises(ValueError, match="at least one"):
        demand.mode_split([], [], [])
    with pytest.raises(ValueError, match="identical lengths"):
        demand.mode_split([np.ones(2)], [], [0.0])
    with pytest.raises(ValueError, match="identical shapes"):
        demand.mode_split([np.ones(2), np.ones(3)], [-0.1, -0.1], [0.0, 0.0])


@pytest.mark.parametrize("builder", [graphs.build_node_graph, graphs.build_segment_graph])
def test_graph_builders_validate_tolerance_and_geometry(builder):
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="tolerance"):
        builder([line], tolerance=0.0)
    with pytest.raises(ValueError, match="shape"):
        builder([np.array([[0.0, 0.0]])])
    with pytest.raises(ValueError, match="finite"):
        builder([np.array([[0.0, 0.0], [np.nan, 1.0]])])


def test_empty_node_graph_has_consistent_coordinate_shape():
    graph = graphs.build_node_graph([])
    assert graph.node_xy.shape == (0, 2)
    assert graph.num_nodes == graph.num_edges == 0


def test_node_graph_rejects_mismatched_cost_count():
    line = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="value per polyline"):
        graphs.build_node_graph([line], costs=[1.0, 2.0])


def test_node_graph_does_not_mutate_custom_costs():
    lines = [
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([[1.0, 0.0], [2.0, 0.0]]),
    ]
    costs = np.array([np.nan, -1.0])
    original = costs.copy()

    graph = graphs.build_node_graph(lines, costs=costs)

    np.testing.assert_equal(costs, original)
    np.testing.assert_allclose(graph.edge_cost, [1.0, 1.0])
