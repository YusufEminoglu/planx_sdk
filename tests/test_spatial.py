# -*- coding: utf-8 -*-
"""Tests for the spatial submodule."""

import numpy as np
import pytest

from planx.spatial import (
    closeness_straightness,
    cumulative_opportunities,
    eigenvector,
    gravity_accessibility,
    many_to_many,
    multi_source,
    network_criticality,
)


@pytest.fixture
def sample_graph():
    # Simple line graph: 0 - 1 - 2
    # Node coordinates: 0: (0,0), 1: (1,0), 2: (2,0)
    # CSR representations:
    # adj_list:
    # 0 -> 1 (weight 1.5)
    # 1 -> 0 (weight 1.5), 2 (weight 2.5)
    # 2 -> 1 (weight 2.5)
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([1.5, 1.5, 2.5, 2.5], dtype=np.float64)
    node_xy = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float64)
    return indptr, adj, weights, 3, node_xy


def test_dijkstra_many_to_many(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    # Distances from node 0 to all other nodes:
    # 0 -> 0: 0.0
    # 0 -> 1: 1.5
    # 0 -> 2: 1.5 + 2.5 = 4.0
    dists = many_to_many(indptr, adj, weights, n, sources=[0])
    np.testing.assert_allclose(dists[0], [0.0, 1.5, 4.0])


def test_dijkstra_multi_source(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    # Minimum distances from sources [0, 2] to all other nodes:
    # Node 0 is near source 0 (dist 0.0)
    # Node 1 is near source 0 (dist 1.5) or source 2 (dist 2.5) -> nearest is source 0
    # Node 2 is near source 2 (dist 0.0)
    dists, labels = multi_source(indptr, adj, weights, n, sources=[0, 2])
    np.testing.assert_allclose(dists, [0.0, 1.5, 0.0])
    np.testing.assert_allclose(labels, [0, 0, 1])


def test_closeness_straightness(sample_graph):
    indptr, adj, weights, n, node_xy = sample_graph
    metrics = closeness_straightness(indptr, adj, weights, n, node_xy=node_xy)

    # Reach: each node can reach 2 other nodes
    np.testing.assert_allclose(metrics["reach"], [2.0, 2.0, 2.0])

    # Farness:
    # Node 0 farness: 0->1 (1.5) + 0->2 (4.0) = 5.5
    # Node 1 farness: 1->0 (1.5) + 1->2 (2.5) = 4.0
    # Node 2 farness: 2->1 (2.5) + 2->0 (4.0) = 6.5
    np.testing.assert_allclose(metrics["farness"], [5.5, 4.0, 6.5])


def test_eigenvector_centrality(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    ev = eigenvector(indptr, adj, n)

    # Node 1 is the center, should have the highest centrality (1.0)
    assert np.isclose(ev[1], 1.0)
    assert ev[0] > 0.0
    assert ev[2] > 0.0
    assert ev[1] > ev[0]
    assert ev[1] > ev[2]


def test_accessibility():
    dists = np.array([[1.0, 2.0, 5.0], [4.0, 1.0, 10.0]])
    weights = np.array([10.0, 20.0, 50.0])

    co = cumulative_opportunities(dists, weights, cutoff=3.0)
    np.testing.assert_allclose(co, [30.0, 20.0])

    ga_exp = gravity_accessibility(dists, weights, decay_method="exponential", beta=0.5)
    np.testing.assert_allclose(ga_exp, [17.527144, 13.820863], rtol=1e-5)


def test_network_criticality(sample_graph):
    indptr, adj, weights, n, _ = sample_graph

    # 0 - 1 - 2
    # Node 1 is between 0 and 2.
    # If origins = [0, 2], destinations = [1]
    # Path from 0 -> 1: uses edge 0 (0 -> 1)
    # Path from 2 -> 1: uses edge 3 (2 -> 1)
    # So edge 0 (0->1) and edge 3 (2->1) should each have usage count 1.
    usage, criticality = network_criticality(
        indptr, adj, weights, n, origins=[0, 2], destinations=[1]
    )
    np.testing.assert_array_equal(usage, [1, 0, 0, 1])
    np.testing.assert_allclose(criticality, [100.0, 0.0, 0.0, 100.0])
