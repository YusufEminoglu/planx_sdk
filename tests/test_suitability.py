# -*- coding: utf-8 -*-
"""Tests for the suitability submodule."""

import numpy as np

from planx.suitability import (
    ahp_weights,
    critic_weights,
    decision_matrix_from_layers,
    entropy_weights,
    greedy_lscp,
    greedy_mclp,
    greedy_p_median,
    normalize_array,
    pca_weights,
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
