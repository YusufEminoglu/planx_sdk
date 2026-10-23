"""Tests for the embedded pure-Python path kernels."""

import numpy as np

from planx.engine import paths

# A four-node line: 0 -1- 1 -2- 2 -3- 3
INDptr = np.array([0, 1, 3, 5, 6], dtype=np.int64)
ADJ = np.array([1, 0, 2, 1, 3, 2], dtype=np.int64)
WEIGHTS = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=np.float64)


def test_many_to_many_matches_heap_fallback(monkeypatch):
    expected = paths.many_to_many(INDptr, ADJ, WEIGHTS, 4, [0, 3])
    monkeypatch.setattr(paths, "HAS_SCIPY", False)
    fallback = paths.many_to_many(INDptr, ADJ, WEIGHTS, 4, [0, 3])
    assert np.allclose(fallback, expected)


def test_multi_source_matches_heap_fallback(monkeypatch):
    expected = paths.multi_source(INDptr, ADJ, WEIGHTS, 4, [0, 3])
    monkeypatch.setattr(paths, "HAS_SCIPY", False)
    fallback = paths.multi_source(INDptr, ADJ, WEIGHTS, 4, [0, 3])
    assert np.array_equal(fallback[0], expected[0])
    assert np.array_equal(fallback[1], expected[1])


def test_multi_source_offset_accounts_for_entry_costs():
    dist, label = paths.multi_source_offset(INDptr, ADJ, WEIGHTS, 4, [0, 3], [5.0, 0.0])
    assert dist.tolist() == [5.0, 5.0, 3.0, 0.0]
    assert label.tolist() == [0, 1, 1, 1]


def test_multi_source_offset_skips_non_finite_offsets():
    dist, label = paths.multi_source_offset(INDptr, ADJ, WEIGHTS, 4, [0, 3], [np.nan, 0.0])
    assert dist.tolist() == [6.0, 5.0, 3.0, 0.0]
    assert label.tolist() == [1, 1, 1, 1]


def test_dijkstra_pruned_limits_by_prune_weight():
    cost, prune = paths.dijkstra_pruned(INDptr, ADJ, WEIGHTS, WEIGHTS, 4, 0, 4.0)
    assert cost.tolist() == [0.0, 1.0, 3.0, np.inf]
    assert prune.tolist() == [0.0, 1.0, 3.0, np.inf]


def test_shortest_path_tree_and_reconstruction():
    dist, pred_node, pred_edge = paths.shortest_path_tree(INDptr, ADJ, np.arange(6), WEIGHTS, 4, 0)
    assert dist.tolist() == [0.0, 1.0, 3.0, 6.0]
    assert pred_node.tolist() == [-1, 0, 1, 2]
    nodes, edges = paths.reconstruct_path(pred_node, pred_edge, 0, 3)
    assert nodes == [0, 1, 2, 3]
    assert len(edges) == 3
