# -*- coding: utf-8 -*-
"""Tests for the suitability submodule."""

import numpy as np
import pytest

from planx.suitability import (
    ahp_weights,
    capacitated_location_allocation,
    critic_weights,
    decision_matrix_from_layers,
    entropy_weights,
    greedy_lscp,
    greedy_mclp,
    greedy_p_median,
    normalize_array,
    pca_weights,
    topsis_method,
    vikor_method,
    weighted_linear_combination,
)


def test_normalize_array_benefit_minmax():
    arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    norm = normalize_array(arr, "benefit_minmax", low=0.0, high=100.0)
    np.testing.assert_allclose(norm, [0.0, 50.0, 100.0])


def test_normalize_array_cost_minmax():
    arr = np.array([0.0, 50.0, 100.0], dtype=np.float32)
    norm = normalize_array(arr, "cost_minmax", low=0.0, high=100.0)
    np.testing.assert_allclose(norm, [100.0, 50.0, 0.0])


def test_normalize_array_sigmoid():
    arr = np.array([40.0, 50.0, 60.0], dtype=np.float32)
    norm = normalize_array(arr, "benefit_sigmoid", mid=50.0, spread=10.0)
    # At mid, sigmoid is 0.5 (which scales to 50.0)
    assert np.isclose(norm[1], 50.0)
    assert norm[2] > norm[1]
    assert norm[0] < norm[1]


def test_weighted_linear_combination():
    c1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    c2 = np.array([[50.0, 60.0], [70.0, 80.0]], dtype=np.float32)

    weights = [0.4, 0.6]
    result = weighted_linear_combination([c1, c2], weights)

    expected = 0.4 * c1 + 0.6 * c2
    np.testing.assert_allclose(result, expected)


def test_wlc_with_constraint():
    c1 = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)
    c2 = np.array([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32)
    constraint = np.array([[1, 0], [1, 1]], dtype=np.uint8)

    weights = [0.5, 0.5]
    result = weighted_linear_combination(
        [c1, c2], weights, constraint_array=constraint, nodata=-999.0
    )

    expected = np.array([[50.0, 0.0], [50.0, 50.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected)


def test_greedy_mclp():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    indices, added, cum = greedy_mclp(candidates, demands, pop, max_distance=5.0, k=2)
    assert indices == [1, 0]
    np.testing.assert_allclose(added, [200.0, 100.0])
    np.testing.assert_allclose(cum, [200.0, 300.0])

    existing = np.array([[0.0, 0.0]])
    indices, added, cum = greedy_mclp(
        candidates, demands, pop, max_distance=5.0, k=2, existing_coords=existing
    )
    assert indices == [1]
    np.testing.assert_allclose(added, [200.0])
    np.testing.assert_allclose(cum, [300.0])


def test_ahp_weights():
    # 3x3 consistency comparison matrix
    matrix = np.array([[1.0, 2.0, 3.0], [0.5, 1.0, 2.0], [0.3333333, 0.5, 1.0]])
    weights, cr = ahp_weights(matrix)

    assert weights.shape == (3,)
    assert cr < 0.10
    assert np.isclose(np.sum(weights), 1.0)
    assert weights[0] > weights[1] > weights[2]


def test_decision_matrix_from_layers():
    lyr1 = np.array([[1.0, 2.0], [np.nan, 4.0]])
    lyr2 = np.array([[10.0, 20.0], [30.0, 40.0]])

    dm, mask = decision_matrix_from_layers([lyr1, lyr2])

    # Position (1, 0) is nan in lyr1, so only 3 pixels should be valid
    assert dm.shape == (3, 2)
    assert np.all(mask == [[True, True], [False, True]])
    np.testing.assert_allclose(dm, [[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])


def test_entropy_weights():
    # 4 alternatives, 3 criteria
    decision_matrix = np.array(
        [[10.0, 100.0, 0.1], [20.0, 50.0, 0.2], [15.0, 80.0, 0.15], [30.0, 20.0, 0.3]]
    )
    weights = entropy_weights(decision_matrix)

    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)


def test_critic_weights():
    # 5 alternatives, 3 criteria
    decision_matrix = np.array(
        [
            [10.0, 100.0, 1.0],
            [20.0, 80.0, 1.2],
            [15.0, 90.0, 1.1],
            [30.0, 70.0, 1.5],
            [25.0, 60.0, 1.3],
        ]
    )

    # 2 benefit criteria, 1 cost criterion (index 1 is cost)
    weights, sigmas, contrasts = critic_weights(decision_matrix, [1, -1, 1])

    assert weights.shape == (3,)
    assert sigmas.shape == (3,)
    assert contrasts.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)


def test_pca_weights():
    decision_matrix = np.array(
        [
            [10.0, 100.0, 1.0],
            [20.0, 80.0, 1.2],
            [15.0, 90.0, 1.1],
            [30.0, 70.0, 1.5],
            [25.0, 60.0, 1.3],
        ]
    )
    weights = pca_weights(decision_matrix)

    assert weights.shape == (3,)
    assert np.isclose(np.sum(weights), 1.0)


def test_greedy_p_median():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    # 1. p=2 using coordinates
    selected, costs = greedy_p_median(
        candidate_coords=candidates, demand_coords=demands, demand_pop=pop, p=2
    )
    # The first greedy choice will pick candidates[2] (20,20) because it is closest
    # to the largest pop 500.
    # The second choice will pick candidates[1] (10,10) to cover the rest.
    assert selected == [2, 1]
    assert len(costs) == 2

    # 2. p=2 using precomputed distance matrix
    dists = np.array([[1.414, 15.556, 35.355], [12.728, 1.414, 21.213], [26.870, 12.728, 7.071]])
    selected_dist, costs_dist = greedy_p_median(dists=dists, demand_pop=pop, p=2)
    assert selected_dist == [2, 1]


def test_greedy_lscp():
    candidates = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
    demands = np.array([[1.0, 1.0], [11.0, 11.0], [25.0, 25.0]])
    pop = np.array([100.0, 200.0, 500.0])

    # With max_distance = 15.0 and target_coverage = 0.8
    # Total pop = 800. 80% = 640.
    # Candidates[1] covers (1,1) (dist 14.14) and (11,11) (dist 1.414) -> 300 pop
    # Candidates[2] covers (11,11) (dist 12.72) and (25,25) (dist 7.07) -> 700 pop
    # Picking candidates[2] first covers 700 pop (which is >= 640).
    # So it should stop after picking 1 facility!
    selected, cov_frac = greedy_lscp(
        candidates, demands, demand_pop=pop, max_distance=15.0, target_coverage=0.8
    )
    assert selected == [2]
    assert cov_frac == 700.0 / 800.0


def test_capacitated_location_allocation():
    facilities = np.array([[0.0, 0.0], [10.0, 0.0]])
    capacities = np.array([150.0, 200.0])
    demands = np.array([[1.0, 0.0], [9.0, 0.0], [2.0, 0.0]])
    pop = np.array([100.0, 150.0, 80.0])

    # Minimum distances to facilities:
    # d((1,0)) to F0 is 1.0, to F1 is 9.0.
    # d((9,0)) to F0 is 9.0, to F1 is 1.0.
    # d((2,0)) to F0 is 2.0, to F1 is 8.0.
    # Sorted order of demands by min distance:
    # 1. d_idx=0 (dist 1.0)
    # 2. d_idx=1 (dist 1.0)
    # 3. d_idx=2 (dist 2.0)

    # Allocating d_idx=0 (pop 100) -> closest F0 (cap 150). Remaining F0 cap = 50.
    # Allocating d_idx=1 (pop 150) -> closest F1 (cap 200). Remaining F1 cap = 50.
    # Allocating d_idx=2 (pop 80) -> closest F0 (need 80, but remaining is 50 -> too small).
    # Next closest is F1 (need 80, but remaining is 50 -> too small).
    # So d_idx=2 is unassigned.

    allocations, unassigned, usage = capacitated_location_allocation(
        facilities, capacities, demands, pop
    )

    assert allocations[0] == [0]
    assert allocations[1] == [1]
    assert list(unassigned) == [2]
    assert np.allclose(usage, [100.0, 150.0])

    # Test max distance: F0 cannot serve d_idx=0 if max_distance < 1.0
    allocations_dist, unassigned_dist, usage_dist = capacitated_location_allocation(
        facilities, capacities, demands, pop, max_distance=0.5
    )
    assert len(allocations_dist[0]) == 0
    assert len(allocations_dist[1]) == 0
    assert len(unassigned_dist) == 3

    # Error handling
    import pytest

    with pytest.raises(ValueError, match="shape"):
        capacitated_location_allocation(np.ones((2, 3)), capacities, demands, pop)


def test_topsis_method():
    # 3 alternatives, 2 criteria (both benefit)
    # Alt 0 is clearly best, Alt 2 is clearly worst
    decision_matrix = np.array([[10.0, 100.0], [5.0, 50.0], [1.0, 10.0]])
    weights = np.array([0.5, 0.5])
    benefit_criteria = np.array([True, True])

    scores, ranks = topsis_method(decision_matrix, weights, benefit_criteria)

    assert len(scores) == 3
    assert len(ranks) == 3
    # Alt 0 should rank 1st, Alt 1 2nd, Alt 2 3rd
    assert ranks[0] == 1
    assert ranks[1] == 2
    assert ranks[2] == 3
    assert scores[0] > scores[1] > scores[2]
    # Check bounds
    assert np.all(scores >= 0.0) & np.all(scores <= 1.0)

    # Error checking
    with pytest.raises(ValueError):
        topsis_method(decision_matrix, weights[:-1], benefit_criteria)


def test_vikor_method():
    # 3 alternatives, 2 criteria (both benefit)
    decision_matrix = np.array([[10.0, 100.0], [5.0, 50.0], [1.0, 10.0]])
    weights = np.array([0.5, 0.5])
    benefit_criteria = np.array([True, True])

    scores, ranks = vikor_method(decision_matrix, weights, benefit_criteria, v=0.5)

    assert len(scores) == 3
    assert len(ranks) == 3
    # Lower is better in VIKOR compromise index Q
    # Alt 0 is closest to ideal best, so Q should be 0.0 (best)
    # Alt 2 is at ideal worst, so Q should be 1.0 (worst)
    assert np.isclose(scores[0], 0.0)
    assert np.isclose(scores[2], 1.0)
    assert ranks[0] == 1
    assert ranks[1] == 2
    assert ranks[2] == 3

    # Error checking
    with pytest.raises(ValueError):
        vikor_method(decision_matrix, weights, benefit_criteria[:-1])
