# -*- coding: utf-8 -*-
"""Tests for the geostats submodule."""

import numpy as np
import pytest

from planx.geostats import (
    calculate_getis_ord,
    calculate_global_geary,
    calculate_global_moran,
    calculate_local_moran,
    calculate_mean_center,
    create_distance_band_weights,
    create_knn_weights,
    idw_to_grid,
    idw_to_points,
    kriging_to_grid,
    kriging_to_points,
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


def test_spatial_weights():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    ids = [100, 101, 102]

    # 1. Test KNN Weights (k=2)
    neigh_knn, w_knn = create_knn_weights(coords, ids, k=2, row_standardized=True)
    assert set(neigh_knn[101]) == {100, 102}
    assert np.allclose(w_knn[101], [0.5, 0.5])

    # 2. Test Distance Band Weights (threshold = 1.5)
    neigh_db, w_db = create_distance_band_weights(
        coords, ids, threshold=1.5, row_standardized=True, binary=True
    )
    assert neigh_db[100] == [101]
    assert w_db[100] == [1.0]
    assert set(neigh_db[101]) == {100, 102}
    assert np.allclose(w_db[101], [0.5, 0.5])

    # Distance Band with Inverse Distance (binary = False)
    neigh_db_inv, w_db_inv = create_distance_band_weights(
        coords, ids, threshold=2.5, row_standardized=False, binary=False, power=1.0
    )
    pos_101 = neigh_db_inv[100].index(101)
    pos_102 = neigh_db_inv[100].index(102)
    assert np.isclose(w_db_inv[100][pos_101], 1.0)
    assert np.isclose(w_db_inv[100][pos_102], 0.5)

    with pytest.raises(ValueError, match="coords"):
        create_knn_weights(np.ones((3, 3)), ids, k=2)


def test_calculate_global_geary():
    # 4 observations
    y = np.array([1.0, 2.0, 3.0, 4.0])
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    weights = {0: [1.0], 1: [1.0, 1.0], 2: [1.0, 1.0], 3: [1.0]}
    id_order = [0, 1, 2, 3]

    c, ec, var, z, p = calculate_global_geary(y, neighbors, weights, id_order)

    # For this perfectly positive spatial autocorrelation case:
    # C should be exactly 0.3
    assert np.isclose(c, 0.3)
    assert np.isclose(ec, 1.0)
    assert var > 0
    # Z-score should be negative (indicating clustering/positive autocorrelation for Geary's C < 1)
    assert z < 0
    assert 0 <= p <= 1.0

    # Error handling
    with pytest.raises(ValueError):
        calculate_global_geary(y[:3], neighbors, weights, id_order[:3])


def test_kriging_interpolation():
    # Source points
    src_coords = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    src_values = np.array([10.0, 20.0, 30.0, 40.0])

    # 1. Test target points
    target_coords = np.array([[5.0, 5.0], [0.0, 0.0]])
    estimates, variances = kriging_to_points(
        src_coords, src_values, target_coords, model="spherical", nugget=0.0, sill=10.0, range_=15.0
    )

    assert len(estimates) == 2
    assert len(variances) == 2
    # At (5,5), by symmetry the estimate should be the mean: (10+20+30+40)/4 = 25.0
    assert np.isclose(estimates[0], 25.0)
    # At (0,0), it should be exactly the source value 10.0 and variance should be near 0
    assert np.isclose(estimates[1], 10.0, atol=1e-5)
    assert variances[1] < 1e-4

    # 2. Test grid interpolation
    grid, var_grid, x, y = kriging_to_grid(
        src_coords,
        src_values,
        (0.0, 0.0, 10.0, 10.0),
        cell_size=5.0,
        model="exponential",
        range_=10.0,
    )
    assert grid.shape == (2, 2)
    assert var_grid.shape == (2, 2)

    # Argument validation
    with pytest.raises(ValueError):
        kriging_to_points(np.ones((3, 3)), src_values, target_coords)
    with pytest.raises(ValueError):
        kriging_to_points(src_coords, src_values, target_coords, range_=-1)
    with pytest.raises(ValueError):
        kriging_to_points(src_coords, src_values, target_coords, nugget=2.0, sill=1.0)


def test_calculate_global_moran():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    weights = {0: [1.0], 1: [1.0, 1.0], 2: [1.0, 1.0], 3: [1.0]}
    id_order = [0, 1, 2, 3]

    moran_i, expected_i, variance, z_score, p_value = calculate_global_moran(
        y, neighbors, weights, id_order
    )

    # In this case of positive spatial autocorrelation:
    # Moran's I should be greater than the expected value (-1/3)
    assert moran_i > expected_i
    assert np.isclose(expected_i, -1.0 / 3.0)
    assert variance > 0
    assert z_score > 0
    assert 0 <= p_value <= 1.0

    # Error handling
    with pytest.raises(ValueError, match="at least 4 observations"):
        calculate_global_moran(y[:3], neighbors, weights, id_order[:3])


def test_calculate_local_moran():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    weights = {0: [1.0], 1: [1.0, 1.0], 2: [1.0, 1.0], 3: [1.0]}
    id_order = [0, 1, 2, 3]

    I_vals, z_scores, p_vals, quadrants = calculate_local_moran(y, neighbors, weights, id_order)

    assert len(I_vals) == 4
    assert len(z_scores) == 4
    assert len(p_vals) == 4
    assert len(quadrants) == 4

    # The elements should have quadrant classifications
    assert any(q in ["HH", "LL", "HL", "LH", "Not Significant"] for q in quadrants)

    # Edge cases
    # Too few elements
    I_short, _, _, _ = calculate_local_moran(y[:2], neighbors, weights, id_order[:2])
    assert len(I_short) == 2
