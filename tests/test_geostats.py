# -*- coding: utf-8 -*-
"""Tests for the geostats submodule."""

import numpy as np

from planx.geostats import calculate_getis_ord, calculate_mean_center


def test_calculate_mean_center():
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 20.0, 40.0])

    mx, my = calculate_mean_center(x, y)
    assert np.isclose(mx, 10.0)
    assert np.isclose(my, 20.0)


def test_calculate_mean_center_weighted():
    x = np.array([0.0, 10.0, 20.0])
    y = np.array([0.0, 20.0, 40.0])
    weights = np.array([1.0, 2.0, 1.0])

    mx, my = calculate_mean_center(x, y, weights=weights)
    # Total weight = 4.0
    # Mean X = (0*1 + 10*2 + 20*1) / 4 = 40 / 4 = 10.0
    # Mean Y = (0*1 + 20*2 + 40*1) / 4 = 80 / 4 = 20.0
    assert np.isclose(mx, 10.0)
    assert np.isclose(my, 20.0)


def test_calculate_getis_ord():
    # Simple grid or line of 3 features
    y = np.array([10.0, 100.0, 10.0], dtype=np.float64)
    # 0 - 1 - 2
    neighbors = {0: [1], 1: [0, 2], 2: [1]}
    weights = {0: [1.0], 1: [0.5, 0.5], 2: [1.0]}
    id_order = [0, 1, 2]

    z, p, bins = calculate_getis_ord(y, neighbors, weights, id_order, star=True)

    # Feature 1 should be a significant hot spot (z-score > 0)
    assert len(z) == 3
    assert z[1] > 0
    assert (
        bins[1] >= 0
    )  # Should be identified as positive hot/warm spot or not significant depending on std
