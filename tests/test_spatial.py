# -*- coding: utf-8 -*-
"""Tests for the spatial submodule."""

import numpy as np
import pytest

from planx.spatial import (
    active_mobility_permeability,
    brandes_betweenness,
    calculate_pedestrian_route_directness,
    calculate_walk_score,
    choice_centrality_una,
    classify_level_of_traffic_stress,
    closeness_straightness,
    cumulative_opportunities,
    eigenvector,
    enhanced_2sfca,
    gravity_accessibility,
    gravity_centrality_una,
    huff_gravity_model,
    identify_low_stress_islands,
    kernel_density_2sfca,
    many_to_many,
    multi_source,
    network_criticality,
    reach_centrality_una,
    service_area_coverage,
    simulate_thermal_comfort_pet,
    spatial_equity_gini,
    thermal_comfort_routing,
    three_step_2sfca,
)
from planx.spatial import paths as spatial_paths


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


@pytest.fixture
def disconnected_graph():
    # Node 0 - Node 1 connected (weight 1.0); Node 2 fully isolated.
    indptr = np.array([0, 1, 2, 2], dtype=np.int64)
    adj = np.array([1, 0], dtype=np.int64)
    weights = np.array([1.0, 1.0], dtype=np.float64)
    return indptr, adj, weights, 3


@pytest.fixture
def triangle_graph():
    # Triangle where the direct 0-2 edge (weight 5) is never on a shortest
    # path, since routing via node 1 (1 + 1 = 2) is cheaper. This forces
    # stale/duplicate entries in the priority queues of the Dijkstra-style
    # kernels, exercising their "already settled" skip branches.
    indptr = np.array([0, 2, 4, 6], dtype=np.int64)
    adj = np.array([1, 2, 0, 2, 1, 0], dtype=np.int64)
    weights = np.array([1.0, 5.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float64)
    return indptr, adj, weights, 3


@pytest.fixture
def skewed_triangle_graph():
    # 0 reaches both 1 and 2 directly and cheaply (weight 1 each); the 1-2
    # edge is expensive (weight 10), so relaxing it never improves on the
    # direct route -- exercising a "failed relaxation" branch.
    indptr = np.array([0, 2, 4, 6], dtype=np.int64)
    adj = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    weights = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 10.0], dtype=np.float64)
    return indptr, adj, weights, 3


@pytest.fixture
def diamond_graph():
    # 0 -> {1, 2} -> 3, two equal-cost shortest paths between 0 and 3.
    indptr = np.array([0, 2, 4, 6, 8], dtype=np.int64)
    adj = np.array([1, 2, 0, 3, 0, 3, 1, 2], dtype=np.int64)
    weights = np.full(8, 1.0, dtype=np.float64)
    return indptr, adj, weights, 4


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


# --------------------------------------------------------------------------- #
# paths.py
# --------------------------------------------------------------------------- #


def test_many_to_many_cutoff(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    # Node 2 is 4.0 away from node 0, beyond the 2.0 cutoff -> unreachable.
    dists = many_to_many(indptr, adj, weights, n, sources=[0], cutoff=2.0)
    assert dists[0][0] == 0.0
    assert dists[0][1] == 1.5
    assert np.isinf(dists[0][2])


def test_many_to_many_disconnected(disconnected_graph):
    indptr, adj, weights, n = disconnected_graph
    dists = many_to_many(indptr, adj, weights, n, sources=[0])
    np.testing.assert_allclose(dists[0][:2], [0.0, 1.0])
    assert np.isinf(dists[0][2])


def test_multi_source_cutoff(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    # Both sources are farther than 1.0 from node 1, so it stays unreachable.
    dists, labels = multi_source(indptr, adj, weights, n, sources=[0, 2], cutoff=1.0)
    np.testing.assert_allclose(dists, [0.0, np.inf, 0.0])
    np.testing.assert_array_equal(labels, [0, -1, 1])


def test_multi_source_disconnected(disconnected_graph):
    indptr, adj, weights, n = disconnected_graph
    dists, labels = multi_source(indptr, adj, weights, n, sources=[0, 2])
    np.testing.assert_allclose(dists, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(labels, [0, 0, 1])


def test_many_to_many_pure_python_matches_scipy(sample_graph, monkeypatch):
    indptr, adj, weights, n, _ = sample_graph
    monkeypatch.setattr(spatial_paths, "HAS_SCIPY", False)
    dists = many_to_many(indptr, adj, weights, n, sources=[0, 2])
    np.testing.assert_allclose(dists[0], [0.0, 1.5, 4.0])
    np.testing.assert_allclose(dists[1], [4.0, 2.5, 0.0])


def test_many_to_many_pure_python_cutoff(sample_graph, monkeypatch):
    indptr, adj, weights, n, _ = sample_graph
    monkeypatch.setattr(spatial_paths, "HAS_SCIPY", False)
    dists = many_to_many(indptr, adj, weights, n, sources=[0], cutoff=2.0)
    assert dists[0][1] == 1.5
    assert np.isinf(dists[0][2])


def test_many_to_many_pure_python_cancel(sample_graph, monkeypatch):
    indptr, adj, weights, n, _ = sample_graph
    monkeypatch.setattr(spatial_paths, "HAS_SCIPY", False)
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1

    dists = many_to_many(indptr, adj, weights, n, sources=[0, 1, 2], cancel=cancel)
    # First source is processed before cancel() reports True.
    np.testing.assert_allclose(dists[0], [0.0, 1.5, 4.0])


def test_multi_source_pure_python_matches_scipy(sample_graph, monkeypatch):
    indptr, adj, weights, n, _ = sample_graph
    monkeypatch.setattr(spatial_paths, "HAS_SCIPY", False)
    dists, labels = multi_source(indptr, adj, weights, n, sources=[0, 2])
    np.testing.assert_allclose(dists, [0.0, 1.5, 0.0])
    np.testing.assert_array_equal(labels, [0, 0, 1])


def test_multi_source_pure_python_cutoff(sample_graph, monkeypatch):
    indptr, adj, weights, n, _ = sample_graph
    monkeypatch.setattr(spatial_paths, "HAS_SCIPY", False)
    dists, labels = multi_source(indptr, adj, weights, n, sources=[0, 2], cutoff=1.0)
    np.testing.assert_allclose(dists, [0.0, np.inf, 0.0])
    np.testing.assert_array_equal(labels, [0, -1, 1])


# --------------------------------------------------------------------------- #
# centrality.py: closeness_straightness
# --------------------------------------------------------------------------- #


def test_closeness_straightness_no_node_xy(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    metrics = closeness_straightness(indptr, adj, weights, n)
    assert "straightness" not in metrics
    np.testing.assert_allclose(metrics["reach"], [2.0, 2.0, 2.0])


def test_closeness_straightness_single_node():
    indptr = np.array([0, 0], dtype=np.int64)
    adj = np.array([], dtype=np.int64)
    weights = np.array([], dtype=np.float64)
    metrics = closeness_straightness(indptr, adj, weights, 1)
    np.testing.assert_allclose(metrics["reach"], [0.0])
    np.testing.assert_allclose(metrics["farness"], [0.0])
    # n == 1 means the (n > 1) branch is skipped, closeness stays 0.
    np.testing.assert_allclose(metrics["closeness"], [0.0])


def test_closeness_straightness_radius(sample_graph):
    indptr, adj, weights, n, node_xy = sample_graph
    metrics = closeness_straightness(indptr, adj, weights, n, radius=2.0)
    np.testing.assert_allclose(metrics["reach"], [1.0, 1.0, 0.0])
    np.testing.assert_allclose(metrics["farness"], [1.5, 1.5, 0.0])
    np.testing.assert_allclose(metrics["harmonic"], [1.0 / 1.5, 1.0 / 1.5, 0.0])
    # Node 2 has farness == 0 -> excluded from the closeness computation.
    assert metrics["closeness"][2] == 0.0


def test_closeness_straightness_cancel(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 1

    metrics = closeness_straightness(indptr, adj, weights, n, chunk=1, cancel=cancel)
    # Only the first chunk (node 0) should have been processed.
    assert metrics["reach"][0] == 2.0
    assert metrics["reach"][1] == 0.0
    assert metrics["reach"][2] == 0.0


def test_closeness_straightness_progress(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    seen = []
    closeness_straightness(indptr, adj, weights, n, chunk=1, progress=seen.append)
    assert seen == pytest.approx([1 / 3, 2 / 3, 1.0])


# --------------------------------------------------------------------------- #
# centrality.py: eigenvector
# --------------------------------------------------------------------------- #


def test_eigenvector_empty_graph():
    ev = eigenvector(np.array([0], dtype=np.int64), np.array([], dtype=np.int64), 0)
    assert ev.shape == (0,)


def test_eigenvector_max_iter_exhausted(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    # A single power-iteration step is far from the tol=1e-10 convergence
    # criterion, exercising the "loop completes without break" path.
    ev = eigenvector(indptr, adj, n, max_iter=1)
    np.testing.assert_allclose(ev, [2.0 / 3.0, 1.0, 2.0 / 3.0], rtol=1e-4)


# --------------------------------------------------------------------------- #
# centrality.py: brandes_betweenness
# --------------------------------------------------------------------------- #


def test_brandes_betweenness_basic(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    node_bc, edge_bc, depth = brandes_betweenness(indptr, adj, weights, n)
    # Middle node lies on the (0, 2) and (2, 0) shortest paths.
    np.testing.assert_allclose(node_bc, [0.0, 2.0, 0.0])
    assert edge_bc is None
    assert depth is None


def test_brandes_betweenness_edge_bc(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    # Edge 0 covers CSR entries 0 and 1 (0<->1); edge 1 covers entries 2 and 3 (1<->2).
    adj_edge = np.array([0, 0, 1, 1], dtype=np.int64)
    node_bc, edge_bc, depth = brandes_betweenness(
        indptr, adj, weights, n, adj_edge=adj_edge, num_edges=2
    )
    np.testing.assert_allclose(node_bc, [0.0, 2.0, 0.0])
    np.testing.assert_allclose(edge_bc, [4.0, 4.0])
    assert depth is None


def test_brandes_betweenness_radius_prune(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    adj_edge = np.array([0, 0, 1, 1], dtype=np.int64)
    node_bc, edge_bc, _ = brandes_betweenness(
        indptr,
        adj,
        weights,
        n,
        adj_edge=adj_edge,
        num_edges=2,
        w_prune=weights,
        radius=2.0,
    )
    np.testing.assert_allclose(node_bc, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(edge_bc, [2.0, 0.0])


def test_brandes_betweenness_sources_subset_scaling(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    adj_edge = np.array([0, 0, 1, 1], dtype=np.int64)
    node_bc, edge_bc, _ = brandes_betweenness(
        indptr, adj, weights, n, adj_edge=adj_edge, num_edges=2, sources=[0]
    )
    # Un-scaled contribution from source 0 alone is node_bc=[0,1,0], edge_bc=[2,1];
    # scaled by n / len(sources) == 3.
    np.testing.assert_allclose(node_bc, [0.0, 3.0, 0.0])
    np.testing.assert_allclose(edge_bc, [6.0, 3.0])


def test_brandes_betweenness_collect_depth(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    _, _, depth = brandes_betweenness(indptr, adj, weights, n, collect_depth=True)
    np.testing.assert_allclose(depth["node_count"], [3.0, 3.0, 3.0])
    np.testing.assert_allclose(depth["total_depth"], [5.5, 4.0, 6.5])


def test_brandes_betweenness_cancel(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    node_bc, _, _ = brandes_betweenness(indptr, adj, weights, n, cancel=lambda: True)
    np.testing.assert_allclose(node_bc, [0.0, 0.0, 0.0])


def test_brandes_betweenness_progress(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    seen = []
    brandes_betweenness(indptr, adj, weights, n, progress=seen.append)
    assert seen and seen[0] == 0.0


# --------------------------------------------------------------------------- #
# centrality.py: network_criticality
# --------------------------------------------------------------------------- #


def test_network_criticality_empty_destinations(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    usage, criticality = network_criticality(indptr, adj, weights, n, origins=[0], destinations=[])
    np.testing.assert_array_equal(usage, np.zeros(len(adj), dtype=np.int64))
    np.testing.assert_allclose(criticality, np.zeros(len(adj)))


def test_network_criticality_multi_hop_path(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    # Forces traversal through an already-visited neighbour (node 0 seen twice).
    usage, criticality = network_criticality(indptr, adj, weights, n, origins=[0], destinations=[2])
    np.testing.assert_array_equal(usage, [1, 0, 1, 0])
    np.testing.assert_allclose(criticality, [100.0, 0.0, 100.0, 0.0])


def test_network_criticality_unreachable_destination(disconnected_graph):
    indptr, adj, weights, n = disconnected_graph
    # Node 2 is isolated, so it can never reach destination 0.
    usage, criticality = network_criticality(indptr, adj, weights, n, origins=[2], destinations=[0])
    np.testing.assert_array_equal(usage, np.zeros(len(adj), dtype=np.int64))
    np.testing.assert_allclose(criticality, np.zeros(len(adj)))


def test_network_criticality_no_edges():
    indptr = np.array([0, 0], dtype=np.int64)
    adj = np.array([], dtype=np.int64)
    weights = np.array([], dtype=np.float64)
    usage, criticality = network_criticality(indptr, adj, weights, 1, origins=[0], destinations=[0])
    assert usage.shape == (0,)
    assert criticality.shape == (0,)


def test_network_criticality_revisits_stale_heap_entry(triangle_graph):
    indptr, adj, weights, n = triangle_graph
    # No node matches destination 999, forcing the search to drain the heap
    # fully and pop the stale (already-visited) entry for node 2.
    usage, criticality = network_criticality(
        indptr, adj, weights, n, origins=[0], destinations=[999]
    )
    np.testing.assert_array_equal(usage, np.zeros(len(adj), dtype=np.int64))
    np.testing.assert_allclose(criticality, np.zeros(len(adj)))


def test_network_criticality_failed_relaxation(skewed_triangle_graph):
    indptr, adj, weights, n = skewed_triangle_graph
    # Forces an attempted relaxation of an unvisited node that doesn't
    # improve its distance (the expensive 1-2 edge loses to the direct route).
    usage, criticality = network_criticality(
        indptr, adj, weights, n, origins=[0], destinations=[999]
    )
    np.testing.assert_array_equal(usage, np.zeros(len(adj), dtype=np.int64))
    np.testing.assert_allclose(criticality, np.zeros(len(adj)))


# --------------------------------------------------------------------------- #
# Additional branch coverage: stale heap entries / tie-breaking
# --------------------------------------------------------------------------- #


def test_many_to_many_pure_python_stale_heap_entry(triangle_graph, monkeypatch):
    indptr, adj, weights, n = triangle_graph
    monkeypatch.setattr(spatial_paths, "HAS_SCIPY", False)
    dists = many_to_many(indptr, adj, weights, n, sources=[0])
    np.testing.assert_allclose(dists[0], [0.0, 1.0, 2.0])


def test_multi_source_pure_python_stale_heap_entry(triangle_graph, monkeypatch):
    indptr, adj, weights, n = triangle_graph
    monkeypatch.setattr(spatial_paths, "HAS_SCIPY", False)
    dists, labels = multi_source(indptr, adj, weights, n, sources=[0])
    np.testing.assert_allclose(dists, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(labels, [0, 0, 0])


def test_multi_source_pure_python_duplicate_source(sample_graph, monkeypatch):
    indptr, adj, weights, n, _ = sample_graph
    monkeypatch.setattr(spatial_paths, "HAS_SCIPY", False)
    # The same node appearing twice in `sources` must not overwrite its label.
    dists, labels = multi_source(indptr, adj, weights, n, sources=[0, 0])
    np.testing.assert_allclose(dists, [0.0, 1.5, 4.0])
    np.testing.assert_array_equal(labels, [0, 0, 0])


def test_brandes_betweenness_stale_heap_entry(triangle_graph):
    indptr, adj, weights, n = triangle_graph
    adj_edge = np.array([0, 1, 0, 2, 2, 1], dtype=np.int64)
    node_bc, edge_bc, _ = brandes_betweenness(
        indptr, adj, weights, n, adj_edge=adj_edge, num_edges=3
    )
    # The expensive direct 0-2 edge (id 1) is never on a shortest path.
    np.testing.assert_allclose(node_bc, [0.0, 2.0, 0.0])
    np.testing.assert_allclose(edge_bc, [4.0, 0.0, 4.0])


def test_brandes_betweenness_equal_length_paths(diamond_graph):
    indptr, adj, weights, n = diamond_graph
    node_bc, edge_bc, _ = brandes_betweenness(indptr, adj, weights, n)
    # Fully symmetric diamond: every node sits on an equal-cost shortest
    # path between the other two "opposite" nodes, splitting credit evenly.
    np.testing.assert_allclose(node_bc, [1.0, 1.0, 1.0, 1.0])
    assert edge_bc is None


def test_brandes_betweenness_sources_subset_no_edge_bc(sample_graph):
    indptr, adj, weights, n, _ = sample_graph
    node_bc, edge_bc, _ = brandes_betweenness(indptr, adj, weights, n, sources=[0])
    np.testing.assert_allclose(node_bc, [0.0, 3.0, 0.0])
    assert edge_bc is None


# ---------------------------------------------------------------------------
# Additional accessibility.py coverage
# ---------------------------------------------------------------------------


def test_gravity_accessibility_ndim_error():
    dists = np.array([1.0, 2.0, 3.0])  # 1D, invalid
    weights = np.array([10.0, 20.0, 30.0])
    with pytest.raises(ValueError):
        gravity_accessibility(dists, weights)


def test_gravity_accessibility_weight_mismatch_error():
    dists = np.array([[1.0, 2.0, 3.0]])
    weights = np.array([10.0, 20.0])  # wrong length
    with pytest.raises(ValueError):
        gravity_accessibility(dists, weights)


def test_gravity_accessibility_cutoff_masks_far_destinations():
    dists = np.array([[1.0, 10.0], [5.0, 5.0]])
    weights = np.array([100.0, 100.0])

    # Without cutoff both destinations contribute.
    acc_no_cutoff = gravity_accessibility(dists, weights, cutoff=None)

    # With cutoff=8.0, the first origin's distance of 10.0 to the second destination is excluded.
    acc_cutoff = gravity_accessibility(dists, weights, cutoff=8.0)

    assert acc_cutoff[0] < acc_no_cutoff[0]
    assert np.isclose(acc_cutoff[1], acc_no_cutoff[1])


# ---------------------------------------------------------------------------
# Walkability and Advanced Active Mobility Tests
# ---------------------------------------------------------------------------


def test_thermal_comfort_routing():
    # 3-node path: 0 - 1 - 2
    # CSR graph representation:
    # 0 -> 1 (edge index 0, cost 1.0)
    # 1 -> 0 (edge index 1, cost 1.0)
    # 1 -> 2 (edge index 2, cost 1.0)
    # 2 -> 1 (edge index 3, cost 1.0)
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64)

    # Shade factors: 0->1 is fully shaded (1.0), 1->2 is exposed (0.0)
    shade = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)

    # 1. Simple routing check
    res = thermal_comfort_routing(
        indptr, adj, weights, n=3, start_node=0, end_node=2, shade_factors=shade, alpha=0.5
    )

    assert res["path"] == [0, 1, 2]
    assert res["shortest_path"] == [0, 1, 2]
    assert res["comfort_distance"] == 20.0
    assert res["shortest_distance"] == 20.0
    assert np.isclose(res["comfort_index"], 0.5)  # mean of 1.0 and 0.0

    # 2. Add heat weights: 0->1 has 0.0 heat weight, 1->2 has 1.0 heat weight
    heat = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    res_heat = thermal_comfort_routing(
        indptr,
        adj,
        weights,
        n=3,
        start_node=0,
        end_node=2,
        shade_factors=shade,
        heat_weights=heat,
        alpha=0.5,
    )
    assert res_heat["path"] == [0, 1, 2]

    # 3. Test dimension mismatches
    with pytest.raises(ValueError, match="shade_factors length"):
        thermal_comfort_routing(
            indptr, adj, weights, n=3, start_node=0, end_node=2, shade_factors=shade[:-1]
        )

    with pytest.raises(ValueError, match="heat_weights length"):
        thermal_comfort_routing(
            indptr,
            adj,
            weights,
            n=3,
            start_node=0,
            end_node=2,
            shade_factors=shade,
            heat_weights=heat[:-1],
        )

    with pytest.raises(ValueError, match="out of bounds"):
        thermal_comfort_routing(
            indptr, adj, weights, n=3, start_node=-1, end_node=2, shade_factors=shade
        )


def test_gravity_centrality_una():
    # 3-node path: 0 - 1 - 2
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # Destination at node 2 with weight 100.0
    destinations = np.array([2], dtype=np.int64)
    dest_weights = np.array([100.0], dtype=np.float64)

    # 1. Exponential decay
    gc_exp = gravity_centrality_una(
        indptr,
        adj,
        weights,
        n=3,
        destination_weights=dest_weights,
        destinations=destinations,
        cutoff=10.0,
        decay_method="exponential",
        beta=0.1,
    )
    # Distance from 0 to 2 is 2.0 -> decay is exp(-0.1 * 2) = exp(-0.2)
    # Distance from 1 to 2 is 1.0 -> decay is exp(-0.1 * 1) = exp(-0.1)
    # Distance from 2 to 2 is 0.0 -> decay is exp(-0.0) = 1.0
    assert np.isclose(gc_exp[0], 100.0 * np.exp(-0.2))
    assert np.isclose(gc_exp[1], 100.0 * np.exp(-0.1))
    assert np.isclose(gc_exp[2], 100.0)

    # 2. Linear decay
    gc_lin = gravity_centrality_una(
        indptr,
        adj,
        weights,
        n=3,
        destination_weights=dest_weights,
        destinations=destinations,
        cutoff=5.0,
        decay_method="linear",
    )
    # f(d) = 1.0 - d / cutoff
    assert np.isclose(gc_lin[0], 100.0 * (1.0 - 2.0 / 5.0))
    assert np.isclose(gc_lin[1], 100.0 * (1.0 - 1.0 / 5.0))

    # 3. Validation error checks
    with pytest.raises(ValueError, match="identical length"):
        gravity_centrality_una(
            indptr,
            adj,
            weights,
            n=3,
            destination_weights=dest_weights[:-1],
            destinations=destinations,
            cutoff=5.0,
        )

    with pytest.raises(ValueError, match="must be >= 0"):
        gravity_centrality_una(
            indptr,
            adj,
            weights,
            n=3,
            destination_weights=dest_weights,
            destinations=destinations,
            cutoff=-1.0,
            decay_method="linear",
        )


def test_active_mobility_permeability():
    # 3-node path: 0 - 1 - 2
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # Edge indices:
    # 0->1: index 0
    # 1->0: index 1
    # 1->2: index 2
    # 2->1: index 3
    # Both directions of the 1-2 edge are high stress (False)
    low_stress = np.array([True, True, False, False], dtype=bool)

    permeability = active_mobility_permeability(
        indptr, adj, weights, n=3, low_stress_mask=low_stress
    )

    # From node 0: on full network, reachable nodes are {1, 2}.
    # Under low-stress only, we can only reach 1 (since 1-2 is high stress).
    # So permeability is 50.0% (1 out of 2).
    assert np.isclose(permeability[0], 50.0)

    # From node 1: full reachable {0, 2}. Under low stress, reachable is {0}.
    # Permeability is 50.0%.
    assert np.isclose(permeability[1], 50.0)

    # From node 2: full reachable {0, 1}. Under low stress, neither is reachable.
    # So permeability is 0.0%.
    assert np.isclose(permeability[2], 0.0)

    # Test mismatch validation
    with pytest.raises(ValueError, match="low_stress_mask length"):
        active_mobility_permeability(indptr, adj, weights, n=3, low_stress_mask=low_stress[:-1])


def test_simulate_thermal_comfort_pet():
    air_temp = np.array([30.0, 35.0])
    rel_humidity = np.array([75.0, 50.0])
    wind_speed = np.array([1.5, 2.0])
    solar_radiation = np.array([800.0, 400.0])
    sky_view_factor = np.array([0.8, 0.4])
    canopy_cover = np.array([0.1, 0.5])

    pet = simulate_thermal_comfort_pet(
        air_temp, rel_humidity, wind_speed, solar_radiation, sky_view_factor, canopy_cover
    )

    assert len(pet) == 2
    # Verify that higher temperature + solar radiation yields higher PET
    assert pet[0] > 30.0
    assert pet[1] > 25.0


def test_reach_centrality_una():
    # Path 0 - 1 - 2
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    destinations = np.array([2], dtype=np.int64)
    dest_weights = np.array([10.0], dtype=np.float64)

    reach = reach_centrality_una(
        indptr,
        adj,
        weights,
        n=3,
        destinations=destinations,
        destination_weights=dest_weights,
        cutoff=5.0,
        decay_method="none",
    )

    # All nodes are within distance 5.0 of node 2, decay_method "none" -> factor is 1.0
    assert np.isclose(reach[0], 10.0)
    assert np.isclose(reach[1], 10.0)
    assert np.isclose(reach[2], 10.0)


def test_choice_centrality_una():
    # Star graph: Center (node 0) connected to leaves 1, 2, 3
    indptr = np.array([0, 3, 4, 5, 6], dtype=np.int64)
    adj = np.array([1, 2, 3, 0, 0, 0], dtype=np.int64)
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # Shortest paths between leaf nodes (e.g., 1->2, 1->3, etc.) must go through the center (node 0)
    origins = np.array([1, 2, 3], dtype=np.int64)
    destinations = np.array([1, 2, 3], dtype=np.int64)
    orig_w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    dest_w = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    choice = choice_centrality_una(
        indptr,
        adj,
        weights,
        n=5,
        origins=origins,
        destinations=destinations,
        origin_weights=orig_w,
        destination_weights=dest_w,
        cutoff=5.0,
    )

    # Node 0 (center) should have high choice betweenness because all inter-leaf paths cross it
    assert choice[0] > 0.0
    # Node 4 is isolated / not in paths, should be 0.0
    assert choice[4] == 0.0

    # Test parameter validation checks
    with pytest.raises(ValueError, match="origins and origin_weights"):
        choice_centrality_una(
            indptr,
            adj,
            weights,
            n=5,
            origins=origins,
            destinations=destinations,
            origin_weights=orig_w[:-1],
            destination_weights=dest_w,
            cutoff=5.0,
        )

    with pytest.raises(ValueError, match="destinations and destination_weights"):
        choice_centrality_una(
            indptr,
            adj,
            weights,
            n=5,
            origins=origins,
            destinations=destinations,
            origin_weights=orig_w,
            destination_weights=dest_w[:-1],
            cutoff=5.0,
        )


def test_classify_level_of_traffic_stress():
    speed_limit = np.array([25.0, 35.0, 45.0, 60.0])
    num_lanes = np.array([2, 2, 4, 6])
    has_bike_lane = np.array([True, True, True, False])
    has_sidewalk = np.array([True, True, False, False])
    daily_traffic = np.array([2000.0, 5000.0, 7000.0, 12000.0])

    lts = classify_level_of_traffic_stress(
        speed_limit, num_lanes, has_bike_lane, has_sidewalk, daily_traffic
    )

    assert len(lts) == 4
    # LTS 1: Comfortable for kids
    assert lts[0] == 1
    # LTS 2: Comfortable for mainstream adults
    assert lts[1] == 2
    # LTS 3: Moderate stress
    assert lts[2] == 3
    # LTS 4: High stress
    assert lts[3] == 4


def test_identify_low_stress_islands():
    # 5-node graph:
    # 0 - 1 (low stress, LTS=1)
    # 1 - 2 (high stress barrier, LTS=4)
    # 2 - 3 (low stress, LTS=1)
    # 3 - 4 (low stress, LTS=1)
    indptr = np.array([0, 1, 3, 5, 7, 8], dtype=np.int64)
    adj = np.array([1, 0, 2, 1, 3, 2, 4, 3], dtype=np.int64)
    edge_lts = np.array([1, 1, 4, 4, 1, 1, 1, 1], dtype=np.int64)

    island_labels, island_sizes, barriers = identify_low_stress_islands(
        indptr, adj, n=5, edge_lts=edge_lts
    )

    # 2 islands: {0, 1} and {2, 3, 4}
    assert island_labels[0] == island_labels[1]
    assert island_labels[2] == island_labels[3]
    assert island_labels[3] == island_labels[4]
    assert island_labels[0] != island_labels[2]

    assert island_sizes[island_labels[0]] == 2
    assert island_sizes[island_labels[2]] == 3

    # Barrier should be the edge connecting node 1 and 2 (edge indexes 2 and 3)
    assert len(barriers) > 0
    assert barriers[0][0] in (2, 3)  # index of 1-2 edge or 2-1 edge


def test_accessibility_additional_coverage():
    # Setup simple data
    dists = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.array([10.0, 20.0])

    # 1. gravity_accessibility decay methods
    # Power
    ga_power = gravity_accessibility(dists, weights, decay_method="power", beta=0.5)
    assert len(ga_power) == 2
    # Gaussian
    ga_gauss = gravity_accessibility(dists, weights, decay_method="gaussian", beta=2.0)
    assert len(ga_gauss) == 2
    # Linear
    ga_linear = gravity_accessibility(dists, weights, decay_method="linear", cutoff=10.0)
    assert len(ga_linear) == 2
    # Invalid decay
    with pytest.raises(ValueError, match="Unknown decay method"):
        gravity_accessibility(dists, weights, decay_method="invalid")
    with pytest.raises(ValueError, match="linear decay requires a positive cutoff"):
        gravity_accessibility(dists, weights, decay_method="linear", cutoff=None)

    # 2. cumulative_opportunities validation errors
    with pytest.raises(ValueError, match="must be a 2D array"):
        cumulative_opportunities(np.array([1.0, 2.0]), weights, cutoff=5.0)
    with pytest.raises(ValueError, match="length must match"):
        cumulative_opportunities(dists, np.array([10.0]), cutoff=5.0)

    # 3. enhanced_2sfca validation errors & decay methods
    supply = np.array([5.0, 5.0])
    demand = np.array([100.0, 100.0])
    with pytest.raises(ValueError, match="supply length"):
        enhanced_2sfca(dists, np.array([5.0]), demand, cutoff=5.0)
    with pytest.raises(ValueError, match="demand length"):
        enhanced_2sfca(dists, supply, np.array([100.0]), cutoff=5.0)
    with pytest.raises(ValueError, match="cutoff must be greater than 0"):
        enhanced_2sfca(dists, supply, demand, cutoff=0.0)
    with pytest.raises(ValueError, match="Unknown decay method"):
        enhanced_2sfca(dists, supply, demand, cutoff=5.0, decay_method="invalid")

    # Decay methods
    # Gaussian
    e2_gauss = enhanced_2sfca(dists, supply, demand, cutoff=5.0, decay_method="gaussian", beta=2.0)
    assert len(e2_gauss) == 2
    # Exponential
    e2_exp = enhanced_2sfca(dists, supply, demand, cutoff=5.0, decay_method="exponential", beta=0.5)
    assert len(e2_exp) == 2

    # 4. spatial_equity_gini edge cases
    with pytest.raises(ValueError, match="same length"):
        spatial_equity_gini(np.array([1.0]), np.array([1.0, 2.0]))
    # sum(p) <= 0
    assert spatial_equity_gini(np.array([1.0, 2.0]), np.array([0.0, 0.0])) == 0.0
    # mean_a <= 0
    assert spatial_equity_gini(np.array([0.0, 0.0]), np.array([10.0, 20.0])) == 0.0
    # denominator <= 0
    assert spatial_equity_gini(np.array([-1.0, -1.0]), np.array([10.0, 10.0])) == 0.0

    # 5. service_area_coverage edge cases
    indptr = np.array([0, 1, 2], dtype=np.int64)
    adj = np.array([1, 0], dtype=np.int64)
    edge_w = np.array([1.0, 1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="cannot be empty"):
        service_area_coverage(indptr, adj, edge_w, n=2, facilities=np.array([]), thresholds=[1.0])
    with pytest.raises(ValueError, match="node_population shape"):
        service_area_coverage(
            indptr,
            adj,
            edge_w,
            n=2,
            facilities=np.array([0]),
            thresholds=[1.0],
            node_population=np.array([1.0]),
        )

    # Default population & zero population
    res_def = service_area_coverage(
        indptr, adj, edge_w, n=2, facilities=np.array([0]), thresholds=[1.0], node_population=None
    )
    assert 1.0 in res_def
    res_zero = service_area_coverage(
        indptr,
        adj,
        edge_w,
        n=2,
        facilities=np.array([0]),
        thresholds=[1.0],
        node_population=np.array([0.0, 0.0]),
    )
    assert res_zero[1.0]["coverage_fraction"] == 0.0

    # 6. huff_gravity_model validation error
    with pytest.raises(ValueError, match="Unknown decay method"):
        huff_gravity_model(dists, weights, decay_method="invalid")

    # 7. kernel_density_2sfca validation errors & decay methods
    with pytest.raises(ValueError, match="supply length"):
        kernel_density_2sfca(dists, np.array([5.0]), demand, cutoff=5.0)
    with pytest.raises(ValueError, match="demand length"):
        kernel_density_2sfca(dists, supply, np.array([100.0]), cutoff=5.0)
    with pytest.raises(ValueError, match="cutoff must be greater than 0"):
        kernel_density_2sfca(dists, supply, demand, cutoff=0.0)
    with pytest.raises(ValueError, match="Unknown kernel type"):
        kernel_density_2sfca(dists, supply, demand, cutoff=5.0, kernel="invalid")


def test_walkability_additional_coverage():
    # 3-node path: 0 - 1 - 2
    indptr = np.array([0, 1, 3, 4], dtype=np.int64)
    adj = np.array([1, 0, 2, 1], dtype=np.int64)
    weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    shade = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)

    # 1. thermal_comfort_routing edge cases
    # start == end
    res_same = thermal_comfort_routing(
        indptr, adj, weights, n=3, start_node=0, end_node=0, shade_factors=shade
    )
    assert res_same["path"] == [0]
    assert res_same["comfort_index"] == 1.0

    # disconnected target
    res_disc = thermal_comfort_routing(
        np.array([0, 0, 0, 0]),
        np.array([]),
        np.array([]),
        n=3,
        start_node=0,
        end_node=2,
        shade_factors=np.array([]),
    )
    assert res_disc["path"] == []

    # 2. gravity_centrality_una decay modes and validation
    destinations = np.array([2], dtype=np.int64)
    dest_weights = np.array([100.0], dtype=np.float64)
    # Power decay
    gc_pow = gravity_centrality_una(
        indptr,
        adj,
        weights,
        n=3,
        destination_weights=dest_weights,
        destinations=destinations,
        cutoff=5.0,
        decay_method="power",
        beta=0.5,
    )
    assert len(gc_pow) == 3
    # Unknown decay
    with pytest.raises(ValueError, match="Unknown decay method"):
        gravity_centrality_una(
            indptr,
            adj,
            weights,
            n=3,
            destination_weights=dest_weights,
            destinations=destinations,
            cutoff=5.0,
            decay_method="invalid",
        )

    # 3. choice_centrality_una decay methods & parameters
    origins = np.array([0], dtype=np.int64)
    orig_weights = np.array([10.0], dtype=np.float64)
    # Power decay
    cc_pow = choice_centrality_una(
        indptr,
        adj,
        weights,
        n=3,
        origins=origins,
        destinations=destinations,
        origin_weights=orig_weights,
        destination_weights=dest_weights,
        cutoff=5.0,
        decay_method="power",
        beta=0.5,
    )
    assert len(cc_pow) == 3
    # Exponential decay
    cc_exp = choice_centrality_una(
        indptr,
        adj,
        weights,
        n=3,
        origins=origins,
        destinations=destinations,
        origin_weights=orig_weights,
        destination_weights=dest_weights,
        cutoff=5.0,
        decay_method="exponential",
        beta=0.1,
    )
    assert len(cc_exp) == 3
    # Linear decay
    cc_lin = choice_centrality_una(
        indptr,
        adj,
        weights,
        n=3,
        origins=origins,
        destinations=destinations,
        origin_weights=orig_weights,
        destination_weights=dest_weights,
        cutoff=5.0,
        decay_method="linear",
    )
    assert len(cc_lin) == 3
    # Unknown decay
    with pytest.raises(ValueError, match="Unknown decay method"):
        choice_centrality_una(
            indptr,
            adj,
            weights,
            n=3,
            origins=origins,
            destinations=destinations,
            origin_weights=orig_weights,
            destination_weights=dest_weights,
            cutoff=5.0,
            decay_method="invalid",
        )
    # Linear decay with negative/zero cutoff raises
    with pytest.raises(ValueError, match="linear decay requires a positive cutoff"):
        choice_centrality_una(
            indptr,
            adj,
            weights,
            n=3,
            origins=origins,
            destinations=destinations,
            origin_weights=orig_weights,
            destination_weights=dest_weights,
            cutoff=-1.0,
            decay_method="linear",
        )

    # 4. classify_level_of_traffic_stress alternative branches
    speeds = np.array([30.0, 40.0, 50.0])
    lanes = np.array([3, 3, 2])
    bike = np.array([False, False, True])
    sidewalk = np.array([False, False, True])
    traffic = np.array([5000.0, 1000.0, 9000.0])
    lts_alt = classify_level_of_traffic_stress(speeds, lanes, bike, sidewalk, traffic)
    assert len(lts_alt) == 3

    # 5. active_mobility_permeability count_full == 0
    # Create isolated graph
    perm_zero = active_mobility_permeability(
        np.array([0, 0, 0]),
        np.array([]),
        np.array([]),
        n=2,
        low_stress_mask=np.array([], dtype=bool),
    )
    assert np.all(perm_zero == 100.0)


def test_three_step_2sfca():
    # 2 origins, 2 destinations
    dists = np.array([[10.0, 50.0], [30.0, 10.0]])
    supply = np.array([10.0, 20.0])
    demand = np.array([100.0, 200.0])

    # 1. 3SFCA (none decay)
    A_none = three_step_2sfca(dists, supply, demand, cutoff=40.0, decay_method="none")
    assert len(A_none) == 2

    # 2. Gaussian decay
    A_gauss = three_step_2sfca(
        dists, supply, demand, cutoff=40.0, decay_method="gaussian", beta=20.0
    )
    assert len(A_gauss) == 2

    # 3. Exponential decay
    A_exp = three_step_2sfca(
        dists, supply, demand, cutoff=40.0, decay_method="exponential", beta=0.05
    )
    assert len(A_exp) == 2

    # 4. Linear decay
    A_lin = three_step_2sfca(dists, supply, demand, cutoff=40.0, decay_method="linear")
    assert len(A_lin) == 2

    # 5. Validation errors
    with pytest.raises(ValueError, match="supply length"):
        three_step_2sfca(dists, np.array([10.0]), demand, cutoff=40.0)

    with pytest.raises(ValueError, match="demand length"):
        three_step_2sfca(dists, supply, np.array([100.0]), cutoff=40.0)

    with pytest.raises(ValueError, match="cutoff must be greater than 0"):
        three_step_2sfca(dists, supply, demand, cutoff=-5.0)

    with pytest.raises(ValueError, match="Unknown decay method"):
        three_step_2sfca(dists, supply, demand, cutoff=40.0, decay_method="invalid")


def test_calculate_walk_score():
    # 2 locations, 3 amenities
    dists = np.array([[200.0, 600.0, 1500.0], [500.0, 1000.0, 3000.0]])
    weights = np.array([0.5, 0.3, 0.2])
    int_dens = np.array([220.0, 80.0])  # no penalty vs 4% penalty
    block_len = np.array([100.0, 260.0])  # no penalty vs 5% penalty

    # 1. Calculate walk scores
    scores = calculate_walk_score(dists, weights, int_dens, block_len)
    assert len(scores) == 2
    assert scores[0] > scores[1]  # location 0 is much closer and has no penalties

    # 2. Equal weights fallback when sum is 0
    scores_fallback = calculate_walk_score(dists, np.array([0.0, 0.0, 0.0]), int_dens, block_len)
    assert len(scores_fallback) == 2

    # 3. Validation errors
    with pytest.raises(ValueError, match="amenity_weights shape"):
        calculate_walk_score(dists, np.array([0.5, 0.5]), int_dens, block_len)

    with pytest.raises(ValueError, match="intersection_density and avg_block_length"):
        calculate_walk_score(dists, weights, np.array([200.0]), block_len)


def test_calculate_pedestrian_route_directness():
    # 2 origins, 2 destinations
    net_d = np.array([[10.0, 50.0], [np.inf, 20.0]])
    origins = np.array([[0.0, 0.0], [3.0, 4.0]])
    destinations = np.array([[0.0, 0.0], [3.0, 4.0]])

    # 1. Base check
    prd = calculate_pedestrian_route_directness(net_d, origins, destinations)
    assert prd.shape == (2, 2)
    # distance 10.0 from (0,0) to (0,0) is collocated -> 1.0
    assert prd[0, 0] == 1.0
    # Euclidean distance from (3,4) to (0,0) is 5.0. Network distance is inf -> nan
    assert np.isnan(prd[1, 0])
    # distance from (3,4) to (3,4) is collocated -> 1.0 (Euclidean distance 0)
    assert prd[1, 1] == 1.0

    # 2. Validation checks
    with pytest.raises(ValueError, match="origin_coords shape"):
        calculate_pedestrian_route_directness(net_d, np.array([[0.0]]), destinations)

    with pytest.raises(ValueError, match="destination_coords shape"):
        calculate_pedestrian_route_directness(net_d, origins, np.array([[0.0]]))
