# -*- coding: utf-8 -*-
"""Tests for the spatial submodule."""

import numpy as np
import pytest

from planx.spatial import (
    closeness_straightness,
    cumulative_opportunities,
    eigenvector,
    enhanced_2sfca,
    gravity_accessibility,
    huff_gravity_model,
    kernel_density_2sfca,
    many_to_many,
    multi_source,
    network_criticality,
    service_area_coverage,
    spatial_equity_gini,
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


def test_enhanced_2sfca():
    # 2 origins, 2 destinations
    dists = np.array([[10.0, 50.0], [30.0, 10.0]])
    supply = np.array([10.0, 20.0])
    demand = np.array([100.0, 200.0])

    # 1. Standard 2SFCA (decay_method='none')
    # Cutoff = 40.0
    # For destination 0: covered origins are 0 (dist 10) and 1 (dist 30).
    # Weighted demand = 100 + 200 = 300.
    # R_0 = 10 / 300 = 1/30.
    # For destination 1: covered origins is only 1 (dist 10).
    # Weighted demand = 200.
    # R_1 = 20 / 200 = 0.1.
    #
    # Accessibility A_0: covers only destination 0 -> A_0 = R_0 = 1/30 = 0.033333
    # Accessibility A_1: covers both destinations -> A_1 = R_0 + R_1 = 1/30 + 0.1 = 0.133333
    a = enhanced_2sfca(dists, supply, demand, cutoff=40.0, decay_method="none")
    np.testing.assert_allclose(a, [1.0 / 30.0, 1.0 / 30.0 + 0.1])

    # 2. Linear decay
    # W_00 = 1 - 10/40 = 0.75
    # W_01 = 0.0 (cutoff)
    # W_10 = 1 - 30/40 = 0.25
    # W_11 = 1 - 10/40 = 0.75
    #
    # Weighted demand at 0 = P_0*W_00 + P_1*W_10 = 100*0.75 + 200*0.25 = 75 + 50 = 125.
    # R_0 = 10 / 125 = 0.08.
    # Weighted demand at 1 = P_0*W_01 + P_1*W_11 = 0 + 200*0.75 = 150.
    # R_1 = 20 / 150 = 0.133333.
    #
    # Accessibility A_0 = R_0 * W_00 + R_1 * W_01 = 0.08 * 0.75 + 0 = 0.06.
    # Accessibility A_1 = R_0 * W_10 + R_1 * W_11
    #                   = 0.08 * 0.25 + 0.133333 * 0.75 = 0.02 + 0.1 = 0.12.
    a_linear = enhanced_2sfca(dists, supply, demand, cutoff=40.0, decay_method="linear")
    np.testing.assert_allclose(a_linear, [0.06, 0.12], rtol=1e-5)


def test_spatial_equity_gini():
    # Equal accessibility -> Gini = 0.0
    acc = np.array([5.0, 5.0, 5.0])
    pop = np.array([100.0, 200.0, 300.0])
    assert np.isclose(spatial_equity_gini(acc, pop), 0.0)

    # Some inequality
    acc2 = np.array([10.0, 0.0])
    pop2 = np.array([50.0, 50.0])
    # Gini = 0.5
    assert np.isclose(spatial_equity_gini(acc2, pop2), 0.5)


def test_service_area_coverage(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    pop = np.array([100.0, 200.0, 300.0])
    thresholds = [1.0, 2.0, 5.0]

    res = service_area_coverage(
        indptr,
        adj,
        weights,
        n,
        facilities=[0],
        thresholds=thresholds,
        node_population=pop,
    )

    # Threshold 1.0: only node 0 reachable
    assert np.array_equal(res[1.0]["reachable_nodes"], [0])
    assert np.isclose(res[1.0]["population_covered"], 100.0)
    assert np.isclose(res[1.0]["coverage_fraction"], 100.0 / 600.0)

    # Threshold 2.0: nodes 0 and 1 reachable
    assert np.array_equal(res[2.0]["reachable_nodes"], [0, 1])
    assert np.isclose(res[2.0]["population_covered"], 300.0)
    assert np.isclose(res[2.0]["coverage_fraction"], 300.0 / 600.0)

    # Threshold 5.0: all nodes reachable
    assert np.array_equal(res[5.0]["reachable_nodes"], [0, 1, 2])
    assert np.isclose(res[5.0]["population_covered"], 600.0)
    assert np.isclose(res[5.0]["coverage_fraction"], 1.0)


def test_huff_gravity_model():
    # 2 origins, 3 destinations
    dists = np.array([[1.0, 2.0, 10.0], [3.0, 1.0, 10.0]])
    weights = np.array([10.0, 20.0, 100.0])

    # Power decay with exponent=2
    # Origin 0:
    # f(d_00) = 1/1 = 1, utility = 10 * 1 = 10
    # f(d_01) = 1/4 = 0.25, utility = 20 * 0.25 = 5
    # f(d_02) = 1/100 = 0.01, utility = 100 * 0.01 = 1
    # Sum = 10 + 5 + 1 = 16
    # Probs = [10/16, 5/16, 1/16] = [0.625, 0.3125, 0.0625]
    probs = huff_gravity_model(dists, weights, decay_method="power", exponent=2.0)
    np.testing.assert_allclose(probs[0], [0.625, 0.3125, 0.0625])

    # Row sum must be 1.0
    np.testing.assert_allclose(np.sum(probs, axis=1), [1.0, 1.0])

    # Exponential decay with beta=0.1
    probs_exp = huff_gravity_model(dists, weights, decay_method="exponential", beta=0.1)
    np.testing.assert_allclose(np.sum(probs_exp, axis=1), [1.0, 1.0])

    # Error handling
    with pytest.raises(ValueError):
        huff_gravity_model(dists[0], weights)  # non-2D dists
    with pytest.raises(ValueError):
        huff_gravity_model(dists, weights[:-1])  # size mismatch


def test_kernel_density_2sfca():
    # 2 demand points, 2 supply points
    dists = np.array([[10.0, 50.0], [30.0, 10.0]])
    supply = np.array([10.0, 20.0])
    demand = np.array([100.0, 200.0])

    # Quartic kernel, cutoff=40.0
    # ratio:
    # r_00 = 10/40 = 0.25, W_00 = (15/16) * (1 - 0.25^2)^2 = (15/16) * (15/16)^2 = 0.8239746
    # r_01 = 50/40 > 1.0 -> 0.0
    # r_10 = 30/40 = 0.75, W_10 = (15/16) * (1 - 0.75^2)^2 = (15/16) * (7/16)^2 = 0.179443
    # r_11 = 10/40 = 0.25, W_11 = (15/16) * (1 - 0.25^2)^2 = 0.8239746
    #
    # Step 1: Weighted demand
    # D_0 = P_0*W_00 + P_1*W_10 = 100*0.8239746 + 200*0.179443 = 82.39746 + 35.8886 = 118.286
    # R_0 = S_0 / D_0 = 10 / 118.286 = 0.08454
    # D_1 = P_0*W_01 + P_1*W_11 = 0 + 200*0.8239746 = 164.795
    # R_1 = S_1 / D_1 = 20 / 164.795 = 0.12136
    #
    # Step 2: Sum R_j * W_ij
    # A_0 = R_0 * W_00 + R_1 * W_01 = 0.08454 * 0.8239746 + 0 = 0.069658
    # A_1 = R_0 * W_10 + R_1 * W_11
    #     = 0.08454 * 0.179443 + 0.12136 * 0.8239746
    #     = 0.01517 + 0.099998 = 0.115168
    a = kernel_density_2sfca(dists, supply, demand, cutoff=40.0, kernel="quartic")
    np.testing.assert_allclose(a, [0.069658, 0.115168], rtol=1e-4)

    # Epanechnikov kernel, cutoff=40.0
    a_epa = kernel_density_2sfca(dists, supply, demand, cutoff=40.0, kernel="epanechnikov")
    assert len(a_epa) == 2

    # Gaussian kernel, cutoff=40.0
    a_gau = kernel_density_2sfca(dists, supply, demand, cutoff=40.0, kernel="gaussian")
    assert len(a_gau) == 2
