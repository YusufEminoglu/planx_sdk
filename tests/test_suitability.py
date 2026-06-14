# -*- coding: utf-8 -*-
"""Tests for the suitability submodule."""

import numpy as np

from planx.suitability import greedy_mclp, normalize_array, weighted_linear_combination


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
