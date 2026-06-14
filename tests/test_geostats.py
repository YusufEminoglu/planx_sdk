# -*- coding: utf-8 -*-
"""Tests for the geostats submodule."""

import numpy as np
import pytest

from planx.geostats import (
    calculate_getis_ord,
    calculate_mean_center,
    idw_to_grid,
    idw_to_points,
)


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


def test_idw_interpolation():
    # Source points
    src_coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    src_values = np.array([10.0, 20.0, 30.0, 40.0])

    # 1. Test target points
    target_coords = np.array([[5.0, 5.0], [0.0, 0.0], [5.0, 0.0]])
    interpolated = idw_to_points(src_coords, src_values, target_coords, power=2.0)

    # At (5, 5), average: (10+20+30+40)/4 = 25.0
    assert np.isclose(interpolated[0], 25.0)
    # At (0, 0), exact match = 10.0
    assert np.isclose(interpolated[1], 10.0)
    # At (5, 0), expected = 18.333333
    assert np.isclose(interpolated[2], 220.0 / 12.0)

    # 2. Test grid interpolation
    grid, x, y = idw_to_grid(src_coords, src_values, (0.0, 0.0, 10.0, 10.0), cell_size=5.0)
    assert grid.shape == (2, 2)
    assert len(x) == 2
    assert len(y) == 2

    # Point out of bounds with search_radius
    val_radius = idw_to_points(src_coords, src_values, np.array([[50.0, 50.0]]), search_radius=10.0)
    assert np.isnan(val_radius[0])

    # Argument validation
    with pytest.raises(ValueError, match="must be of shape"):
        idw_to_points(np.ones((3, 3)), src_values, target_coords)
