# -*- coding: utf-8 -*-
"""Additional coverage tests for planx.geostats.stats_engines.

These tests intentionally avoid re-testing calculate_getis_ord,
calculate_global_geary, calculate_global_moran, calculate_local_moran,
calculate_local_geary, and calculate_mean_center, which already have
dedicated coverage in tests/test_geostats.py.
"""

import numpy as np
import pytest

from planx.geostats import (
    calculate_average_nearest_neighbor,
    calculate_bivariate_lee_l,
    calculate_bivariate_moran,
    calculate_central_feature,
    calculate_distance_band_stats,
    calculate_exploratory_regression,
    calculate_general_g,
    calculate_glr,
    calculate_gwlr,
    calculate_gwpca,
    calculate_gwr,
    calculate_gwss,
    calculate_incremental_autocorrelation,
    calculate_kmeans,
    calculate_linear_directional_mean,
    calculate_local_bivariate_moran,
    calculate_local_moran_fdr,
    calculate_median_center,
    calculate_ols,
    calculate_ripleys_cross_k,
    calculate_ripleys_k,
    calculate_sde,
    calculate_similarity_search,
    calculate_spatial_gini,
    calculate_spatial_lag,
    calculate_standard_distance,
    calculate_weighted_kde,
    create_space_time_cube,
    emerging_hotspot_analysis,
    fit_spatial_error_model,
    fit_spatial_lag_model,
    fit_spatial_probit_panel,
    fit_spatial_quantile_panel,
    fit_spatial_sarma_model,
    fit_spatial_sarma_panel,
    fit_spatial_tobit_model,
    fit_spatial_tobit_panel,
    run_sensitivity_simulation,
    skater_spatial_clustering,
)

# Shared line-graph fixture used across several tests: 0 - 1 - 2 - 3
LINE_NEIGHBORS = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
LINE_WEIGHTS_UNIT = {0: [1.0], 1: [1.0, 1.0], 2: [1.0, 1.0], 3: [1.0]}
LINE_WEIGHTS_ROWSTD = {0: [1.0], 1: [0.5, 0.5], 2: [0.5, 0.5], 3: [1.0]}
LINE_ID_ORDER = [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# calculate_bivariate_lee_l
# ---------------------------------------------------------------------------


def test_calculate_bivariate_lee_l_normal():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    local_l, spatial_lag_y, classes = calculate_bivariate_lee_l(
        x, y, LINE_NEIGHBORS, LINE_WEIGHTS_ROWSTD, LINE_ID_ORDER
    )

    assert len(local_l) == 4
    assert len(spatial_lag_y) == 4
    assert classes[0] == "Low-X / Low-Y Lag"
    assert classes[1] == "Low-X / Low-Y Lag"
    assert classes[2] == "High-X / High-Y Lag"
    assert classes[3] == "High-X / High-Y Lag"
    assert np.all(local_l >= 0.0)


def test_calculate_bivariate_lee_l_too_few_observations():
    with pytest.raises(ValueError, match="at least 3 observations"):
        calculate_bivariate_lee_l(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), {0: [], 1: []}, {0: [], 1: []}, [0, 1]
        )


def test_calculate_bivariate_lee_l_isolated_feature():
    # Feature 0 has no valid neighbors, so w_sum == 0 and it should be skipped.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    neighbors = {0: [], 1: [0, 2], 2: [1, 3], 3: [2]}
    weights = {0: [], 1: [0.5, 0.5], 2: [0.5, 0.5], 3: [1.0]}

    local_l, spatial_lag_y, classes = calculate_bivariate_lee_l(
        x, y, neighbors, weights, LINE_ID_ORDER
    )

    assert local_l[0] == 0.0
    assert spatial_lag_y[0] == 0.0


# ---------------------------------------------------------------------------
# calculate_spatial_lag & bivariate moran
# ---------------------------------------------------------------------------


def test_calculate_spatial_lag():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    lag = calculate_spatial_lag(
        y, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER, row_standardize=True
    )
    # 0 -> neigh 1 (20)
    # 1 -> neigh 0, 2 ((10+30)/2 = 20)
    # 2 -> neigh 1, 3 ((20+40)/2 = 30)
    # 3 -> neigh 2 (30)
    np.testing.assert_allclose(lag, [20.0, 20.0, 30.0, 30.0])

    # Empty array edge case
    assert len(calculate_spatial_lag(np.array([]), {}, {}, [])) == 0


def test_calculate_bivariate_moran():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    biv_i, exp_i, var_i, z_score, p_val = calculate_bivariate_moran(
        x, y, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER
    )

    assert isinstance(biv_i, float)
    assert isinstance(p_val, float)
    assert 0.0 <= p_val <= 1.0

    # Error handling
    with pytest.raises(ValueError, match="at least 4 observations"):
        calculate_bivariate_moran(
            x[:2], y[:2], LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER[:2]
        )

    with pytest.raises(ValueError, match="same length"):
        calculate_bivariate_moran(x, y[:3], LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)


def test_calculate_local_bivariate_moran():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    I_vals, z_scores, p_vals, quads = calculate_local_bivariate_moran(
        x, y, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER, alpha=0.5
    )

    assert len(I_vals) == 4
    assert len(z_scores) == 4
    assert len(p_vals) == 4
    assert len(quads) == 4
    assert any(q in ("HH", "LL", "HL", "LH", "Not Significant") for q in quads)

    # Edge case: zero std
    x_flat = np.array([2.0, 2.0, 2.0, 2.0])
    I_flat, z_flat, p_flat, q_flat = calculate_local_bivariate_moran(
        x_flat, y, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER
    )
    np.testing.assert_allclose(I_flat, 0.0)


def test_fit_spatial_lag_model():
    y = np.array([10.0, 15.0, 20.0, 25.0])
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])

    res = fit_spatial_lag_model(y, X, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)

    assert "rho" in res
    assert "beta" in res
    assert len(res["beta"]) == 2
    assert len(res["fitted"]) == 4
    assert len(res["residuals"]) == 4
    assert isinstance(res["r2"], float)

    # Validation errors
    with pytest.raises(ValueError, match="Length of y must match"):
        fit_spatial_lag_model(y[:2], X, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)

    with pytest.raises(ValueError, match="greater than number of predictors"):
        fit_spatial_lag_model(y[:2], X[:2], LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER[:2])


def test_fit_spatial_error_model():
    y = np.array([10.0, 15.0, 20.0, 25.0])
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])

    res = fit_spatial_error_model(y, X, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)

    assert "lambda_param" in res
    assert "beta" in res
    assert len(res["beta"]) == 2
    assert len(res["fitted"]) == 4
    assert len(res["residuals"]) == 4
    assert isinstance(res["r2"], float)

    # Validation errors
    with pytest.raises(ValueError, match="Length of y must match"):
        fit_spatial_error_model(y[:2], X, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)

    with pytest.raises(ValueError, match="greater than number of predictors"):
        fit_spatial_error_model(y[:1], X[:1], LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER[:1])


def test_calculate_gwlr():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    res = calculate_gwlr(y, X, coords, bandwidth=2.0, kernel_type="fixed_gaussian")

    assert "coefficients" in res
    assert "std_errors" in res
    assert "probabilities" in res
    assert len(res["probabilities"]) == 4
    assert np.all(res["probabilities"] >= 0.0) & np.all(res["probabilities"] <= 1.0)

    with pytest.raises(ValueError, match="Length of y must match"):
        calculate_gwlr(y[:2], X, coords, bandwidth=2.0)

    with pytest.raises(ValueError, match="binary dependent variable"):
        calculate_gwlr(np.array([0.0, 1.0, 2.0, 3.0]), X, coords, bandwidth=2.0)


def test_fit_spatial_tobit_model():
    y = np.array([0.0, 5.0, 10.0, 15.0])
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])

    res = fit_spatial_tobit_model(y, X, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)

    assert "rho" in res
    assert "beta" in res
    assert len(res["fitted"]) == 4
    assert np.all(res["fitted"] >= 0.0)

    with pytest.raises(ValueError, match="Length of y must match"):
        fit_spatial_tobit_model(y[:2], X, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)


def test_fit_spatial_quantile_panel():
    y = np.array([10.0, 20.0, 30.0, 15.0, 25.0, 35.0])
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 2.5],
            [1.0, 3.5],
            [1.0, 4.5],
        ]
    )
    W = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )

    res = fit_spatial_quantile_panel(y, X, W, time_periods=2, quantile=0.5)

    assert "coefficients" in res
    assert "spatial_rho" in res
    assert "quantile" in res
    assert res["quantile"] == 0.5
    assert "pinball_loss" in res
    assert "pseudo_r_squared" in res
    assert "residuals" in res
    assert len(res["coefficients"]) == 2
    assert len(res["residuals"]) == 6
    assert isinstance(res["spatial_rho"], float)


def test_fit_spatial_quantile_panel_validations():
    y = np.array([10.0, 20.0, 30.0, 15.0, 25.0, 35.0])
    X = np.ones((6, 2))
    W = np.array(
        [
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        fit_spatial_quantile_panel(y, X, W, time_periods=2, quantile=1.5)

    with pytest.raises(ValueError, match="time_periods must be >= 2"):
        fit_spatial_quantile_panel(y[:3], X[:3], W, time_periods=1, quantile=0.5)

    with pytest.raises(ValueError, match="Number of spatial units must be >= 3"):
        fit_spatial_quantile_panel(y[:4], X[:4], np.eye(2), time_periods=2, quantile=0.5)

    with pytest.raises(ValueError, match="must equal N \\* T"):
        fit_spatial_quantile_panel(y[:5], X, W, time_periods=2, quantile=0.5)

    with pytest.raises(ValueError, match="Number of rows in independent_vars must equal N \\* T"):
        fit_spatial_quantile_panel(y, X[:5], W, time_periods=2, quantile=0.5)


def test_calculate_local_moran_fdr():
    x = np.array([10.0, 20.0, 30.0, 40.0])
    res = calculate_local_moran_fdr(x, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER, alpha=0.1)

    assert "local_moran" in res
    assert "fdr_p_values" in res
    assert len(res["fdr_p_values"]) == 4
    assert np.all((res["fdr_p_values"] >= 0.0) & (res["fdr_p_values"] <= 1.0))


def test_calculate_gwss():
    X = np.array([[10.0], [20.0], [30.0], [40.0]])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    res = calculate_gwss(X, coords, bandwidth=2.0, kernel_type="fixed_gaussian")

    assert "local_mean" in res
    assert "local_std" in res
    assert "local_skewness" in res
    assert res["local_mean"].shape == (4, 1)

    with pytest.raises(ValueError, match="Length of coords must match"):
        calculate_gwss(X[:2], coords, bandwidth=2.0)


def test_calculate_weighted_kde():
    events = np.array([[0.0, 0.0], [10.0, 0.0]])
    weights = np.array([1.0, 5.0])
    grid = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])

    dens = calculate_weighted_kde(events, weights, grid, bandwidth=15.0, kernel_type="quartic")

    assert len(dens) == 3
    assert dens[2] > dens[0]

    with pytest.raises(ValueError, match="event_weights length"):
        calculate_weighted_kde(events, weights[:1], grid, 15.0)


def test_calculate_ripleys_cross_k():
    pts_a = np.array([[0.0, 0.0], [10.0, 10.0]])
    pts_b = np.array([[1.0, 1.0], [100.0, 100.0]])
    radii = np.array([5.0, 20.0])

    k_vals = calculate_ripleys_cross_k(pts_a, pts_b, radii, area=10000.0)

    assert len(k_vals) == 2
    assert k_vals[0] > 0.0


def test_skater_spatial_clustering():
    X = np.array([[0.0], [0.1], [10.0], [10.1]])
    labels = skater_spatial_clustering(X, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, n_clusters=2)

    assert len(labels) == 4
    assert len(np.unique(labels)) == 2


def test_fit_spatial_sarma_model():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])

    res = fit_spatial_sarma_model(y, X, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)

    assert "rho" in res
    assert "lambda_err" in res
    assert len(res["beta"]) == 2

    with pytest.raises(ValueError, match="Length of y must match"):
        fit_spatial_sarma_model(y[:2], X, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)


def test_calculate_bivariate_lee_l_zero_variance():
    x = np.array([5.0, 5.0, 5.0, 5.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    local_l, spatial_lag_y, classes = calculate_bivariate_lee_l(
        x, y, LINE_NEIGHBORS, LINE_WEIGHTS_ROWSTD, LINE_ID_ORDER
    )

    assert np.all(local_l == 0.0)
    assert np.all(spatial_lag_y == 0.0)
    assert all(c == "Not Significant" for c in classes)


# ---------------------------------------------------------------------------
# calculate_central_feature
# ---------------------------------------------------------------------------


def test_calculate_central_feature_single_point():
    assert calculate_central_feature(np.array([1.0]), np.array([1.0])) == 0


def test_calculate_central_feature_unweighted():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    assert calculate_central_feature(x, y) == 1


def test_calculate_central_feature_weighted():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    weights = np.array([1.0, 1.0, 5.0])
    # Heavy weight on the last point should pull the central feature toward it.
    assert calculate_central_feature(x, y, weights=weights) == 2


# ---------------------------------------------------------------------------
# calculate_sde
# ---------------------------------------------------------------------------


def test_calculate_sde_too_few_points():
    result = calculate_sde(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert result == (0.5, 0.5, 0.0, 0.0, 0.0)


def test_calculate_sde_normal():
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    mean_x, mean_y, angle, semi_major, semi_minor = calculate_sde(x, y, num_std=1)

    assert np.isclose(mean_x, 0.0)
    assert np.isclose(mean_y, 0.0)
    assert np.isclose(angle, 0.0, atol=1e-9)
    assert np.isclose(semi_major, np.sqrt(50.0))
    assert np.isclose(semi_minor, 0.0, atol=1e-9)


def test_calculate_sde_num_std_scaling():
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    _, _, _, semi_major_1, _ = calculate_sde(x, y, num_std=1)
    _, _, _, semi_major_2, _ = calculate_sde(x, y, num_std=2)

    assert np.isclose(semi_major_2, 2.0 * semi_major_1)


def test_calculate_sde_y_dominant_variance():
    # Variance concentrated along Y should rotate the semi-major axis by 90 degrees.
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    y = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])

    _, _, angle, semi_major, semi_minor = calculate_sde(x, y, num_std=1)

    assert np.isclose(angle, np.pi / 2.0)
    assert np.isclose(semi_major, np.sqrt(50.0))
    assert np.isclose(semi_minor, 0.0, atol=1e-9)


def test_calculate_sde_weighted():
    x = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
    y = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    mean_x, mean_y, angle, semi_major, semi_minor = calculate_sde(x, y, weights=weights)
    assert np.isfinite(angle)
    assert semi_major >= semi_minor >= 0.0


# ---------------------------------------------------------------------------
# calculate_ols
# ---------------------------------------------------------------------------


def test_calculate_ols_normal():
    x_data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    y = 2 * x_data[:, 0] + 1 + np.array([0.5, -0.3, 0.2, -0.1, 0.4, -0.2, 0.1, -0.6])
    neighbors = {i: [j for j in (i - 1, i + 1) if 0 <= j < 8] for i in range(8)}
    weights = {i: [1.0] * len(neighbors[i]) for i in range(8)}

    result = calculate_ols(y, x_data, neighbors, weights, list(range(8)), ["x1"])

    assert result["n"] == 8
    assert result["p"] == 1
    assert result["df_err"] == 6
    assert result["variable_names"] == ["Intercept", "x1"]
    assert np.isclose(result["coefficients"][1], 1.9238, atol=1e-3)
    assert result["r2"] > 0.99
    assert 0.0 <= result["adj_r2"] <= 1.0
    jb_stat, jb_p = result["jarque_bera"]
    assert jb_stat >= 0.0
    assert 0.0 <= jb_p <= 1.0
    bp_stat, bp_p = result["breusch_pagan"]
    assert bp_stat >= 0.0
    assert 0.0 <= bp_p <= 1.0
    assert isinstance(result["residuals_moran"], float)
    assert len(result["residuals"]) == 8


def test_calculate_ols_perfect_fit_zero_residual_variance():
    # Exact linear relationship: residuals are all (numerically) zero, exercising
    # the s2_ml == 0 / g_tot == 0 branches for the diagnostic tests.
    x_data = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = 2 * x_data[:, 0] + 1.0
    neighbors = {i: [] for i in range(5)}
    weights = {i: [] for i in range(5)}

    result = calculate_ols(y, x_data, neighbors, weights, list(range(5)), ["x1"])
    assert np.isclose(result["r2"], 1.0)
    jb_stat, jb_p = result["jarque_bera"]
    assert jb_stat >= 0.0
    assert 0.0 <= jb_p <= 1.0
    bp_stat, bp_p = result["breusch_pagan"]
    assert bp_stat >= 0.0
    assert 0.0 <= bp_p <= 1.0
    assert result["residuals_moran"] == 0.0


def test_calculate_ols_insufficient_sample_size():
    y = np.array([1.0, 2.0, 3.0])
    x_data = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    with pytest.raises(ValueError, match="Sample size"):
        calculate_ols(y, x_data, {}, {}, [0, 1, 2], ["a", "b"])


# ---------------------------------------------------------------------------
# calculate_spatial_gini
# ---------------------------------------------------------------------------


def test_calculate_spatial_gini_normal():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    result = calculate_spatial_gini(values, LINE_NEIGHBORS, LINE_ID_ORDER, permutations=49, seed=1)

    assert result["n"] == 4
    assert np.isclose(result["mean"], 2.5)
    assert np.isclose(result["gini"], 0.25)
    assert np.isclose(
        result["neighbor_component"] + result["non_neighbor_component"], result["gini"]
    )
    assert result["neighbor_pair_count"] == 3
    assert result["non_neighbor_pair_count"] == 3
    assert result["total_pair_count"] == 6
    assert result["polarization"] is not None
    assert result["p_sim"] is not None
    assert 0.0 <= result["p_sim"] <= 1.0
    assert 0.0 <= result["p_low_sim"] <= 1.0
    assert result["polarization_p_sim"] is not None


def test_calculate_spatial_gini_no_permutations():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = calculate_spatial_gini(values, LINE_NEIGHBORS, LINE_ID_ORDER, permutations=0)
    assert result["permutations"] == 0
    assert result["p_sim"] is None
    assert result["expected_non_neighbor_component"] is None


def test_calculate_spatial_gini_zero_variation_short_circuits_simulation():
    flat = np.array([5.0, 5.0, 5.0, 5.0])
    result = calculate_spatial_gini(flat, LINE_NEIGHBORS, LINE_ID_ORDER, permutations=10, seed=1)
    assert result["gini"] == 0.0
    assert result["polarization"] is None
    # pair_sum <= 0 short-circuits before running any simulation.
    assert result["p_sim"] is None
    assert result["expected_non_neighbor_component"] is None


def test_calculate_spatial_gini_drops_non_finite_values():
    values = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    result = calculate_spatial_gini(values, {0: [1], 1: [0], 2: [], 3: [], 4: []}, [0, 1, 2, 3, 4])
    assert result["n"] == 4


def test_calculate_spatial_gini_zero_mean_denominator():
    values = np.array([0.0, 0.0, 0.0, 0.0])
    result = calculate_spatial_gini(values, LINE_NEIGHBORS, LINE_ID_ORDER, permutations=5)
    assert result["mean"] == 0.0
    assert result["gini"] == 0.0
    assert result["p_sim"] is None


def test_calculate_spatial_gini_negative_values_raise():
    with pytest.raises(ValueError, match="non-negative"):
        calculate_spatial_gini(np.array([1.0, -2.0, 3.0]), {}, [0, 1, 2])


def test_calculate_spatial_gini_too_few_observations():
    with pytest.raises(ValueError, match="at least 2"):
        calculate_spatial_gini(np.array([1.0]), {}, [0])


# ---------------------------------------------------------------------------
# calculate_average_nearest_neighbor
# ---------------------------------------------------------------------------


def test_calculate_average_nearest_neighbor_normal():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    observed, expected, ratio, z, p, area = calculate_average_nearest_neighbor(x, y)

    assert np.isclose(observed, 10.0)
    assert np.isclose(area, 100.0)
    assert ratio > 1.0  # dispersed square pattern -> ratio above 1
    assert 0.0 <= p <= 1.0


def test_calculate_average_nearest_neighbor_explicit_area():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    observed, expected, ratio, z, p, area = calculate_average_nearest_neighbor(
        x, y, study_area=200.0
    )
    assert area == 200.0
    assert np.isfinite(expected)
    assert np.isfinite(z)


def test_calculate_average_nearest_neighbor_too_few_points():
    with pytest.raises(ValueError, match="at least 2 points"):
        calculate_average_nearest_neighbor(np.array([1.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# calculate_standard_distance
# ---------------------------------------------------------------------------


def test_calculate_standard_distance_normal():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    mean_x, mean_y, std_distance = calculate_standard_distance(x, y)

    assert np.isclose(mean_x, 5.0)
    assert np.isclose(mean_y, 5.0)
    assert np.isclose(std_distance, np.sqrt(50.0))


def test_calculate_standard_distance_empty():
    assert calculate_standard_distance(np.array([]), np.array([])) == (0.0, 0.0, 0.0)


def test_calculate_standard_distance_weighted():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    mean_x, mean_y, std_distance = calculate_standard_distance(x, y, weights=weights)
    assert np.isclose(mean_x, 5.0)
    assert std_distance > 0.0


def test_calculate_standard_distance_zero_weights_falls_back():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    weights = np.array([0.0, 0.0, 0.0, 0.0])

    mean_x, mean_y, std_distance = calculate_standard_distance(x, y, weights=weights)
    assert np.isclose(std_distance, np.sqrt(50.0))


# ---------------------------------------------------------------------------
# calculate_gwr
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel_type,bandwidth",
    [
        ("fixed_gaussian", 2.0),
        ("fixed_bisquare", 3.0),
        ("adaptive_bisquare", 3),
        ("unrecognized_kernel", 1.0),
    ],
)
def test_calculate_gwr_kernels(kernel_type, bandwidth):
    coords = np.column_stack((np.arange(6.0), np.zeros(6)))
    x_data = np.arange(6.0).reshape(-1, 1)
    y = 2 * x_data[:, 0] + 1.0

    result = calculate_gwr(y, x_data, coords, bandwidth, kernel_type=kernel_type)

    assert result["local_beta"].shape == (6, 2)
    assert result["local_se"].shape == (6, 2)
    assert result["local_t"].shape == (6, 2)
    assert len(result["local_support"]) == 6
    assert len(result["y_pred"]) == 6
    assert result["r2"] > 0.9
    assert result["effective_df"] > 0.0


# ---------------------------------------------------------------------------
# calculate_median_center
# ---------------------------------------------------------------------------


def test_calculate_median_center_empty():
    assert calculate_median_center(np.array([]), np.array([])) == (0.0, 0.0, 0.0)


def test_calculate_median_center_single_point():
    result = calculate_median_center(np.array([5.0]), np.array([7.0]))
    assert result == (5.0, 7.0, 0.0)


def test_calculate_median_center_normal():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    cx, cy, total_dist = calculate_median_center(x, y)
    assert np.isclose(cx, 5.0, atol=1e-3)
    assert np.isclose(cy, 5.0, atol=1e-3)
    assert total_dist > 0.0


def test_calculate_median_center_zero_weights_breaks_immediately():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    cx, cy, total_dist = calculate_median_center(x, y, weights=np.zeros(4))
    # sum_inv == 0 on the first iteration breaks immediately, keeping the mean center.
    assert np.isclose(cx, 5.0)
    assert np.isclose(cy, 5.0)
    assert total_dist == 0.0


def test_calculate_median_center_weighted():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0])

    cx, cy, total_dist = calculate_median_center(x, y, weights=weights)
    assert np.isclose(cx, 5.0, atol=1e-3)
    assert np.isclose(cy, 5.0, atol=1e-3)


# ---------------------------------------------------------------------------
# calculate_general_g
# ---------------------------------------------------------------------------


def test_calculate_general_g_normal():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    observed_g, expected_g, variance, z, p = calculate_general_g(
        values, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER
    )

    assert np.isclose(expected_g, 0.5)
    assert observed_g > 0.0
    assert variance > 0.0
    assert 0.0 <= p <= 1.0


def test_calculate_general_g_dict_input():
    values = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    result = calculate_general_g(values, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)
    array_result = calculate_general_g(
        np.array([1.0, 2.0, 3.0, 4.0]), LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER
    )
    assert result == array_result


def test_calculate_general_g_zero_denominator():
    values = np.array([0.0, 0.0, 0.0, 0.0])
    result = calculate_general_g(values, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER)
    assert result == (0.0, 0.0, 0.0, 0.0, 1.0)


def test_calculate_general_g_too_few_features():
    with pytest.raises(ValueError, match="at least 4 features"):
        calculate_general_g(np.array([1.0, 2.0, 3.0]), {}, {}, [0, 1, 2])


# ---------------------------------------------------------------------------
# calculate_similarity_search
# ---------------------------------------------------------------------------


def test_calculate_similarity_search_euclidean_target_is_zero():
    full_data = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 10.0], [1.5, 2.5]])
    scores = calculate_similarity_search(full_data, [0], metric="euclidean")
    assert scores[0] == pytest.approx(0.0, abs=1e-9)
    assert scores[2] > scores[1] > scores[3]


def test_calculate_similarity_search_manhattan():
    full_data = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 10.0], [1.5, 2.5]])
    scores_manhattan = calculate_similarity_search(full_data, [0], metric="manhattan")
    scores_euclidean = calculate_similarity_search(full_data, [0], metric="euclidean")
    assert scores_manhattan[0] == pytest.approx(0.0, abs=1e-9)
    # Manhattan distance is always >= Euclidean distance for the same points.
    assert np.all(scores_manhattan >= scores_euclidean - 1e-9)


def test_calculate_similarity_search_multiple_targets_average_profile():
    full_data = np.array([[1.0, 2.0], [2.0, 3.0], [10.0, 10.0], [1.5, 2.5]])
    scores = calculate_similarity_search(full_data, [0, 3])
    assert len(scores) == 4


def test_calculate_similarity_search_empty_input():
    scores = calculate_similarity_search(np.zeros((0, 2)), [])
    assert scores.size == 0


# ---------------------------------------------------------------------------
# calculate_distance_band_stats
# ---------------------------------------------------------------------------


def test_calculate_distance_band_stats_normal():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    stats = calculate_distance_band_stats(x, y, k_neighbors=1)
    assert np.isclose(stats["min"], 10.0)
    assert np.isclose(stats["max"], 10.0)
    assert np.isclose(stats["mean"], 10.0)
    assert np.isclose(stats["median"], 10.0)


def test_calculate_distance_band_stats_k_clamped():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y = np.array([0.0, 0.0, 10.0, 10.0])

    # k_neighbors larger than n - 1 should be clamped to the farthest point.
    stats = calculate_distance_band_stats(x, y, k_neighbors=10)
    assert np.isclose(stats["max"], np.sqrt(200.0))


def test_calculate_distance_band_stats_too_few_points():
    with pytest.raises(ValueError, match="At least 2 points"):
        calculate_distance_band_stats(np.array([1.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# calculate_kmeans
# ---------------------------------------------------------------------------


def test_calculate_kmeans_separates_clusters():
    rng = np.random.default_rng(0)
    cluster_a = np.array([0.0, 0.0]) + rng.normal(0.0, 0.01, (5, 2))
    cluster_b = np.array([100.0, 100.0]) + rng.normal(0.0, 0.01, (5, 2))
    data = np.vstack([cluster_a, cluster_b])

    labels, wcss = calculate_kmeans(data, 2, seed=0)

    assert len(labels) == 10
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]
    assert wcss >= 0.0


def test_calculate_kmeans_constant_column():
    # A zero-variance column should not raise (division-by-zero guard).
    data = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [10.0, 5.0]])
    labels, wcss = calculate_kmeans(data, 2, seed=1)
    assert len(labels) == 4
    assert wcss >= 0.0


def test_calculate_kmeans_duplicate_points_no_crash():
    # All points identical: initialization probabilities and cluster reassignment
    # must fall back gracefully instead of dividing by zero.
    data = np.array([[1.0, 1.0]] * 4)
    labels, wcss = calculate_kmeans(data, 2, seed=0)
    assert len(labels) == 4
    assert np.isclose(wcss, 0.0)


def test_calculate_kmeans_too_few_points():
    with pytest.raises(ValueError, match="greater than or equal to k_clusters"):
        calculate_kmeans(np.array([[1.0, 2.0]]), 2)


# ---------------------------------------------------------------------------
# calculate_linear_directional_mean
# ---------------------------------------------------------------------------


def test_calculate_linear_directional_mean_normal():
    start_x = np.array([0.0, 1.0, 2.0])
    start_y = np.array([0.0, 1.0, 2.0])
    end_x = np.array([1.0, 2.0, 3.0])
    end_y = np.array([1.0, 2.0, 3.0])

    center_x, center_y, mean_angle, mean_length = calculate_linear_directional_mean(
        start_x, start_y, end_x, end_y
    )

    assert np.isclose(center_x, 1.5)
    assert np.isclose(center_y, 1.5)
    assert np.isclose(mean_angle, 45.0)
    assert np.isclose(mean_length, np.sqrt(2.0))


def test_calculate_linear_directional_mean_empty():
    empty = np.array([])
    assert calculate_linear_directional_mean(empty, empty, empty, empty) == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# run_sensitivity_simulation
# ---------------------------------------------------------------------------


def test_run_sensitivity_simulation_normal():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    result = run_sensitivity_simulation(
        values, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER, n_simulations=49, seed=1
    )

    assert np.isclose(result["observed_i"], 1.0 / 3.0)
    assert len(result["simulated_values"]) == 49
    assert 0.0 <= result["empirical_p"] <= 1.0
    assert result["percentile_5"] <= result["percentile_95"]


def test_run_sensitivity_simulation_zero_variance_values():
    flat = np.array([5.0, 5.0, 5.0, 5.0])
    result = run_sensitivity_simulation(
        flat, LINE_NEIGHBORS, LINE_WEIGHTS_UNIT, LINE_ID_ORDER, n_simulations=10, seed=1
    )
    assert result["observed_i"] == 0.0
    assert all(v == 0.0 for v in result["simulated_values"])
    assert result["empirical_p"] == 1.0


def test_run_sensitivity_simulation_too_few_features():
    with pytest.raises(ValueError, match="at least 4 features"):
        run_sensitivity_simulation(np.array([1.0, 2.0, 3.0]), {}, {}, [0, 1, 2], n_simulations=10)


def test_run_sensitivity_simulation_no_neighbors_raises():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    empty_neighbors = {0: [], 1: [], 2: [], 3: []}
    with pytest.raises(ValueError, match="No spatial neighbors"):
        run_sensitivity_simulation(values, empty_neighbors, empty_neighbors, LINE_ID_ORDER)


# ---------------------------------------------------------------------------
# calculate_incremental_autocorrelation
# ---------------------------------------------------------------------------


def test_calculate_incremental_autocorrelation_normal():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y_coords = np.array([0.0, 0.0, 0.0, 0.0])
    values = np.array([1.0, 2.0, 3.0, 4.0])

    results = calculate_incremental_autocorrelation(x, y_coords, values, 0.5, 1.0, 3)

    assert len(results) == 3
    # First band (0.5) is smaller than the minimum spacing (1.0): no neighbors at all.
    assert results[0]["isolated_count"] == 4
    assert results[0]["morans_i"] == 0.0
    # Subsequent bands include neighbors and produce a non-trivial statistic.
    assert results[1]["isolated_count"] == 0
    assert results[1]["morans_i"] > 0.0
    for row in results:
        assert 0.0 <= row["p_value"] <= 1.0


def test_calculate_incremental_autocorrelation_zero_variance():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y_coords = np.array([0.0, 0.0, 0.0, 0.0])
    flat_values = np.array([2.0, 2.0, 2.0, 2.0])

    results = calculate_incremental_autocorrelation(x, y_coords, flat_values, 0.5, 1.0, 2)
    assert all(row["morans_i"] == 0.0 for row in results)
    assert all(row["p_value"] == 1.0 for row in results)
    assert "min_neighbors" not in results[0]


def test_calculate_incremental_autocorrelation_too_few_features():
    with pytest.raises(ValueError, match="at least 4 features"):
        calculate_incremental_autocorrelation(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 3.0]),
            1.0,
            1.0,
            2,
        )


# ---------------------------------------------------------------------------
# calculate_ripleys_k
# ---------------------------------------------------------------------------


def test_calculate_ripleys_k_normal():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y_coords = np.array([0.0, 0.0, 0.0, 0.0])

    results = calculate_ripleys_k(x, y_coords, 0.5, 1.0, 3)

    assert len(results) == 3
    assert results[0]["observed_pairs"] == 0
    assert results[1]["observed_pairs"] > 0
    for row in results:
        assert row["expected_k"] >= 0.0
        assert "l_minus_d" in row


def test_calculate_ripleys_k_explicit_study_area():
    x = np.array([0.0, 10.0, 0.0, 10.0])
    y_coords = np.array([0.0, 0.0, 10.0, 10.0])

    results = calculate_ripleys_k(x, y_coords, 5.0, 5.0, 2, study_area=500.0)
    assert all(row["study_area"] == 500.0 for row in results)


def test_calculate_ripleys_k_too_few_points():
    with pytest.raises(ValueError, match="at least 3 features"):
        calculate_ripleys_k(np.array([0.0, 1.0]), np.array([0.0, 1.0]), 1.0, 1.0, 2)


# ---------------------------------------------------------------------------
# calculate_exploratory_regression
# ---------------------------------------------------------------------------


def test_calculate_exploratory_regression_normal():
    x_data = np.column_stack([np.arange(10.0), np.arange(10.0) ** 0.5, np.sin(np.arange(10.0))])
    y = 3 * x_data[:, 0] - 2 * x_data[:, 1] + 1.0
    names = ["a", "b", "c"]

    models = calculate_exploratory_regression(y, x_data, names)

    assert len(models) > 0
    # Best model (lowest AICc) should be first, and should be sorted ascending.
    aiccs = [m["aicc"] for m in models]
    assert aiccs == sorted(aiccs)
    best = models[0]
    assert set(best["variables"]) <= {"a", "b", "c"}
    assert "Intercept" in best["coefficients"]


def test_calculate_exploratory_regression_zero_variance_y_returns_empty():
    x_data = np.column_stack([np.arange(10.0), np.arange(10.0) ** 0.5])
    y = np.ones(10)
    assert calculate_exploratory_regression(y, x_data, ["a", "b"]) == []


def test_calculate_exploratory_regression_max_vars_limits_models():
    x_data = np.column_stack([np.arange(10.0), np.arange(10.0) ** 0.5, np.sin(np.arange(10.0))])
    y = 3 * x_data[:, 0] - 2 * x_data[:, 1] + 1.0
    names = ["a", "b", "c"]

    models = calculate_exploratory_regression(y, x_data, names, max_vars=1)
    assert all(m["n_vars"] == 1 for m in models)
    assert len(models) == 3


# ---------------------------------------------------------------------------
# calculate_glr
# ---------------------------------------------------------------------------


def test_calculate_glr_gaussian():
    x_data = np.arange(20.0).reshape(-1, 1)
    y = 2 * x_data[:, 0] + 1.0

    result = calculate_glr(y, x_data, family="gaussian")

    assert result["family"] == "gaussian"
    assert np.isclose(result["r2"], 1.0)
    assert result["converged"] is True
    assert result["iterations"] == 1


def test_calculate_glr_logistic():
    rng = np.random.default_rng(0)
    x = np.arange(30.0)
    prob = 1.0 / (1.0 + np.exp(-(0.3 * (x - 15))))
    y = (rng.random(30) < prob).astype(float)

    result = calculate_glr(y, x.reshape(-1, 1), family="logistic")

    assert result["family"] == "logistic"
    assert result["r2"] is None
    assert result["converged"] is True
    assert result["iterations"] >= 1
    assert len(result["p_values"]) == 2


def test_calculate_glr_logistic_reports_non_convergence():
    rng = np.random.default_rng(0)
    x = np.arange(30.0)
    prob = 1.0 / (1.0 + np.exp(-(0.3 * (x - 15))))
    y = (rng.random(30) < prob).astype(float)

    result = calculate_glr(y, x.reshape(-1, 1), family="logistic", max_iter=1)
    assert result["converged"] is False
    assert result["iterations"] == 1


def test_calculate_glr_logistic_non_binary_raises():
    with pytest.raises(ValueError, match="binary dependent variable"):
        calculate_glr(np.array([0.0, 2.0, 1.0]), np.array([[1.0], [2.0], [3.0]]), family="logistic")


def test_calculate_glr_poisson():
    rng = np.random.default_rng(0)
    x = np.arange(30.0)
    y = rng.poisson(lam=5, size=30).astype(float)

    result = calculate_glr(y, x.reshape(-1, 1), family="poisson")

    assert result["family"] == "poisson"
    assert result["r2"] is None
    assert result["converged"] is True


def test_calculate_glr_poisson_reports_non_convergence():
    rng = np.random.default_rng(0)
    x = np.arange(30.0)
    y = rng.poisson(lam=5, size=30).astype(float)

    result = calculate_glr(y, x.reshape(-1, 1), family="poisson", max_iter=1)
    assert result["converged"] is False
    assert result["iterations"] == 1


def test_calculate_glr_poisson_invalid_values_raises():
    with pytest.raises(ValueError, match="non-negative integer count"):
        calculate_glr(np.array([1.0, -2.0, 3.0]), np.array([[1.0], [2.0], [3.0]]), family="poisson")


def test_calculate_glr_unsupported_family_raises():
    x_data = np.arange(20.0).reshape(-1, 1)
    y = 2 * x_data[:, 0] + 1.0
    with pytest.raises(ValueError, match="Unsupported GLR family"):
        calculate_glr(y, x_data, family="unknown")


def test_calculate_glr_insufficient_observations_raises():
    with pytest.raises(ValueError, match="more observations than model parameters"):
        calculate_glr(np.array([1.0, 2.0]), np.array([[1.0], [2.0]]), family="gaussian")


def test_stats_engines_additional_coverage():
    # 1. Getis-Ord validation and edge cases
    from planx.geostats import calculate_getis_ord

    # n <= 1
    z, p, bins = calculate_getis_ord(np.array([1.0]), {}, {}, [0], star=True)
    assert len(z) == 1

    # y_std == 0
    z, p, bins = calculate_getis_ord(
        np.array([5.0, 5.0, 5.0]), {0: [1, 2]}, {0: [1, 1]}, [0, 1, 2], star=True
    )
    assert np.all(z == 0.0)

    # Gi* with star=False
    y_high = np.array([1000.0, 1.0, 1.0, 1.0], dtype=np.float64)
    z_g, p_g, bins_g = calculate_getis_ord(
        y_high,
        {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]},
        {0: [1.0], 1: [0.5, 0.5], 2: [0.5, 0.5], 3: [1.0]},
        [0, 1, 2, 3],
        star=False,
    )
    assert len(z_g) == 4

    # 2. Bivariate Lee's L with negative association classes
    x_neg = np.array([1.0, 2.0, 3.0, 4.0])
    y_neg = np.array([4.0, 3.0, 2.0, 1.0])
    local_l, spatial_lag_y, classes = calculate_bivariate_lee_l(
        x_neg, y_neg, LINE_NEIGHBORS, LINE_WEIGHTS_ROWSTD, LINE_ID_ORDER
    )
    assert any(c in ["High-X / Low-Y Lag", "Low-X / High-Y Lag"] for c in classes)

    # 3. calculate_mean_center weight sum is 0
    from planx.geostats import calculate_mean_center

    mx, my = calculate_mean_center(
        np.array([1.0, 2.0]), np.array([1.0, 2.0]), weights=np.array([0.0, 0.0])
    )
    assert np.isclose(mx, 1.5)

    # 4. calculate_central_feature with n <= 1
    assert calculate_central_feature(np.array([1.0]), np.array([1.0])) == 0


def test_calculate_gwpca():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    res = calculate_gwpca(X, coords, bandwidth=2.0, n_components=2)

    assert res["local_eigenvalues"].shape == (4, 2)
    assert res["local_variance_explained"].shape == (4, 2)
    assert len(res["winning_variable"]) == 4
    assert len(res["total_local_variance"]) == 4
    assert np.all(res["local_variance_explained"] >= 0.0)

    with pytest.raises(ValueError, match="Length of coords"):
        calculate_gwpca(X[:2], coords, bandwidth=2.0)


def test_calculate_gwpca_bisquare():
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    res = calculate_gwpca(X, coords, bandwidth=1.5, n_components=2, kernel_type="fixed_bisquare")

    assert res["local_eigenvalues"].shape == (4, 2)
    assert np.all(res["total_local_variance"] >= 0.0)


def test_emerging_hotspot_analysis_normal():
    # Setup simple line network
    coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]])
    W = np.array(
        [
            [1.0, 1.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0, 1.0, 1.0],
        ]
    )

    # Values that are high on the left and decreasing, to create a hotspot on the left
    values = np.array([[10, 10, 10, 10], [8, 9, 8, 9], [5, 5, 5, 5], [2, 2, 2, 2], [1, 1, 1, 1]])

    ts = np.array([0, 1, 2, 3])

    res = emerging_hotspot_analysis(coords, values, ts, W, significance_level=0.1)

    assert "pattern" in res
    assert "z_scores" in res
    assert "p_values_gi" in res
    assert "mann_kendall_z" in res
    assert "mann_kendall_p" in res
    assert "kendall_tau" in res
    assert "hot_spot_count" in res
    assert "cold_spot_count" in res

    assert res["pattern"].shape == (5,)
    assert res["z_scores"].shape == (5, 4)
    # The left side (index 0) should be persistent or intensifying hotspot
    assert res["hot_spot_count"][0] > 0
    # The right side (index 4) should be cold spot
    assert res["cold_spot_count"][4] > 0


def test_emerging_hotspot_analysis_validation():
    coords = np.array([[0, 0], [1, 0]])
    values = np.random.randn(2, 2)
    ts = np.arange(2)
    W = np.eye(2)

    with pytest.raises(ValueError, match="Need at least 3 time steps"):
        emerging_hotspot_analysis(coords, values, ts, W)

    ts3 = np.arange(3)
    values3 = np.random.randn(2, 3)

    with pytest.raises(ValueError, match="shape \\(N, 2\\)"):
        emerging_hotspot_analysis(coords[:, 0:1], values3, ts3, W)

    with pytest.raises(ValueError, match="length"):
        emerging_hotspot_analysis(coords, values3, np.arange(4), W)

    with pytest.raises(ValueError, match="weights_matrix"):
        emerging_hotspot_analysis(coords, values3, ts3, np.eye(3))


# ---------------------------------------------------------------------------
# create_space_time_cube
# ---------------------------------------------------------------------------


def test_create_space_time_cube_normal():
    # 3 points, spatial bin 10, 2 temporal bins
    coords = np.array([[0.0, 0.0], [5.0, 5.0], [20.0, 20.0]])
    t = np.array([0.0, 5.0, 10.0])
    vals = np.array([1.0, 2.0, 3.0])

    # x_min=0, x_max=20 -> range=20 -> n_x = ceil(20/10) = 2.
    # But points are 0, 5, 20. Wait, point at 20 goes to bin 20/10=2,
    # which would be out of bounds if n_x=2.
    # Ah, the implementation clips to n_x-1, so point at 20 goes to bin 1.
    result = create_space_time_cube(
        coordinates=coords,
        timestamps=t,
        values=vals,
        spatial_bin_size=10.0,
        temporal_bin_count=2,
        aggregation="mean",
    )

    assert result["n_spatial_bins"] == 4
    assert result["n_populated_bins"] == 3
    assert result["cube"].shape == (2, 2, 2)
    assert result["spatial_extent"]["x_max"] == 20.0

    # Check centers
    np.testing.assert_allclose(result["x_centers"], [5.0, 15.0])
    np.testing.assert_allclose(result["t_centers"], [2.5, 7.5])

    # Counts
    assert np.sum(result["bin_counts"]) == 3


def test_create_space_time_cube_count_aggregation():
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    t = np.array([1.0, 1.0, 2.0])
    vals = np.array([1.0, 2.0, 3.0])

    result = create_space_time_cube(
        coordinates=coords,
        timestamps=t,
        values=vals,
        spatial_bin_size=2.0,
        temporal_bin_count=1,
        aggregation="count",
    )

    assert result["cube"].shape == (1, 1, 1)
    assert result["cube"][0, 0, 0] == 3.0


def test_create_space_time_cube_min_max_std():
    coords = np.array([[0.0, 0.0], [0.0, 0.0]])
    t = np.array([0.0, 0.0])
    vals = np.array([1.0, 3.0])

    for agg, exp in [("min", 1.0), ("max", 3.0), ("std", 1.0), ("sum", 4.0)]:
        res = create_space_time_cube(coords, t, vals, 5.0, 1, aggregation=agg)
        assert res["cube"][0, 0, 0] == exp


def test_create_space_time_cube_validation():
    coords = np.array([[0.0, 0.0]])
    t = np.array([0.0])
    vals = np.array([1.0])

    with pytest.raises(ValueError, match="coordinates must be a 2D array"):
        create_space_time_cube(np.array([0.0]), t, vals, 5.0, 1)

    with pytest.raises(ValueError, match="timestamps must be a 1D array"):
        create_space_time_cube(coords, np.array([[0.0]]), vals, 5.0, 1)

    with pytest.raises(ValueError, match="values must be a 1D array"):
        create_space_time_cube(coords, t, np.array([[1.0]]), 5.0, 1)

    with pytest.raises(ValueError, match="spatial_bin_size must be"):
        create_space_time_cube(coords, t, vals, -1.0, 1)

    with pytest.raises(ValueError, match="temporal_bin_count must be"):
        create_space_time_cube(coords, t, vals, 5.0, 0)

    with pytest.raises(ValueError, match="aggregation must be one of"):
        create_space_time_cube(coords, t, vals, 5.0, 1, "invalid")


def test_fit_spatial_panel_model_lag():
    from planx.geostats.stats_engines import fit_spatial_panel_model

    np.random.seed(42)
    N = 4
    T = 3
    K = 2
    # dependent_var: N*T, independent_vars: N*T x K
    y = np.random.randn(N * T)
    X = np.random.randn(N * T, K)

    # Simple row-standardized spatial weights matrix for N=4
    W = np.array([[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]])

    res = fit_spatial_panel_model(
        dependent_var=y, independent_vars=X, weights_matrix=W, time_periods=T, model_type="lag"
    )

    assert res["model_type"] == "lag"
    assert res["n_spatial_units"] == N
    assert res["time_periods"] == T
    assert len(res["coefficients"]) == K
    assert len(res["std_errors"]) == K
    assert len(res["t_stat"]) == K
    assert len(res["p_values"]) == K
    assert len(res["residuals"]) == N * T
    assert isinstance(res["spatial_parameter"], float)
    assert isinstance(res["r_squared"], float)


def test_fit_spatial_panel_model_error():
    from planx.geostats.stats_engines import fit_spatial_panel_model

    np.random.seed(43)
    N = 5
    T = 2
    K = 1
    # 2D and 3D inputs
    y = np.random.randn(N, T)
    X = np.random.randn(N, T, K)

    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                W[i, j] = 1.0 / (N - 1)

    res = fit_spatial_panel_model(
        dependent_var=y, independent_vars=X, weights_matrix=W, time_periods=T, model_type="error"
    )

    assert res["model_type"] == "error"
    assert res["n_spatial_units"] == N
    assert res["time_periods"] == T
    assert len(res["coefficients"]) == K
    assert len(res["residuals"]) == N * T
    assert isinstance(res["spatial_parameter"], float)


def test_fit_spatial_panel_model_validation():
    import pytest

    from planx.geostats.stats_engines import fit_spatial_panel_model

    y = np.random.randn(10)
    X = np.random.randn(10, 2)
    W = np.ones((5, 5))

    # T < 2
    with pytest.raises(ValueError, match="time_periods must be >= 2"):
        fit_spatial_panel_model(y, X, W, 1)

    # Invalid model type
    with pytest.raises(ValueError, match="model_type must be 'lag' or 'error'"):
        fit_spatial_panel_model(y, X, W, 2, model_type="invalid")

    # Non-square W
    with pytest.raises(ValueError, match="weights_matrix must be a square 2D array"):
        fit_spatial_panel_model(y, X, np.ones((5, 4)), 2)

    # N < 3
    with pytest.raises(ValueError, match="Number of spatial units .* must be >= 3"):
        fit_spatial_panel_model(y, X, np.ones((2, 2)), 5)

    # y shape mismatch 1D
    with pytest.raises(ValueError, match="dependent_var length"):
        fit_spatial_panel_model(np.random.randn(11), X, W, 2)

    # y shape mismatch 2D
    with pytest.raises(ValueError, match="dependent_var shape"):
        fit_spatial_panel_model(np.random.randn(5, 3), X, W, 2)

    # X shape mismatch 2D
    with pytest.raises(ValueError, match="independent_vars first dim"):
        fit_spatial_panel_model(y, np.random.randn(11, 2), W, 2)

    # X shape mismatch 3D
    with pytest.raises(ValueError, match="independent_vars shape"):
        fit_spatial_panel_model(y, np.random.randn(5, 3, 2), W, 2)


# ---------------------------------------------------------------------------
# fit_spatial_tobit_panel
# ---------------------------------------------------------------------------


def test_fit_spatial_tobit_panel_normal():
    # 4 spatial units, 2 time periods -> N=4, T=2
    # Total obs = 8
    T = 2

    y = np.array([0.0, 5.0, 10.0, 15.0, 0.0, 6.0, 12.0, 18.0])
    # 2 independent variables
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 2.5],
            [1.0, 3.5],
            [1.0, 4.5],
            [1.0, 5.5],
        ]
    )

    W = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 1.0, 0.0]]
    )

    res = fit_spatial_tobit_panel(y, X, W, time_periods=T, censoring_limit=0.0)

    assert "coefficients" in res
    assert "spatial_rho" in res
    assert "std_errors" in res
    assert "t_stat" in res
    assert "p_values" in res
    assert "r_squared" in res
    assert "censored_count" in res
    assert "uncensored_count" in res
    assert "residuals" in res

    assert res["censored_count"] == 2
    assert res["uncensored_count"] == 6
    assert len(res["coefficients"]) == 2
    assert len(res["residuals"]) == 8


def test_fit_spatial_tobit_panel_validation():
    y = np.ones(8)
    X = np.ones((8, 2))
    W = np.eye(4)

    with pytest.raises(ValueError, match="time_periods must be >= 2"):
        fit_spatial_tobit_panel(y, X, W, time_periods=1)

    with pytest.raises(ValueError, match="square 2D array"):
        fit_spatial_tobit_panel(y, X, np.ones(4), time_periods=2)

    with pytest.raises(ValueError, match="must be >= 3"):
        fit_spatial_tobit_panel(np.ones(4), np.ones((4, 2)), np.eye(2), time_periods=2)

    with pytest.raises(ValueError, match="does not match"):
        fit_spatial_tobit_panel(np.ones((4, 3)), X, W, time_periods=2)

    with pytest.raises(ValueError, match="Too few uncensored"):
        fit_spatial_tobit_panel(np.zeros(8), X, W, time_periods=2, censoring_limit=0.0)


def test_fit_spatial_tobit_panel_2d_inputs():
    T = 2
    y = np.array([[0.0, 5.0, 10.0, 15.0], [0.0, 6.0, 12.0, 18.0]]).T  # shape (4, 2)

    X = np.ones((4, 2, 3))
    W = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 1.0, 0.0]]
    )

    res = fit_spatial_tobit_panel(y, X, W, time_periods=T)
    assert res["censored_count"] == 2
    assert len(res["coefficients"]) == 3


def test_fit_spatial_sarma_panel():
    y = np.array([10.0, 20.0, 30.0, 40.0, 12.0, 22.0, 32.0, 42.0])
    X = np.array(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
            [1.0, 5.0],
            [1.0, 2.5],
            [1.0, 3.5],
            [1.0, 4.5],
            [1.0, 5.5],
        ]
    )
    W = np.array(
        [[0.0, 0.5, 0.5, 0.0], [0.5, 0.0, 0.0, 0.5], [0.5, 0.0, 0.0, 0.5], [0.0, 0.5, 0.5, 0.0]]
    )

    res = fit_spatial_sarma_panel(y, X, W, time_periods=2)

    assert "spatial_rho" in res
    assert "spatial_lambda" in res
    assert "coefficients" in res
    assert len(res["coefficients"]) == 2
    assert "std_errors" in res
    assert "p_values" in res
    assert "t_stat" in res
    assert "r_squared" in res
    assert "residuals" in res
    assert len(res["residuals"]) == 8


def test_fit_spatial_sarma_panel_invalid():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    X = np.array([[1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]])
    W = np.eye(4)

    with pytest.raises(ValueError, match="time_periods must be >= 2"):
        fit_spatial_sarma_panel(y, X, W, time_periods=1)

    with pytest.raises(ValueError, match="weights_matrix must be a square 2D array"):
        fit_spatial_sarma_panel(y, X, np.array([1, 2, 3]), time_periods=2)

    with pytest.raises(ValueError, match="must be >= 3"):
        fit_spatial_sarma_panel(y[:2], X[:2], np.eye(2), time_periods=2)

    with pytest.raises(ValueError, match="dependent_var length"):
        fit_spatial_sarma_panel(y, X, W, time_periods=2)

    with pytest.raises(ValueError, match="independent_vars first dim"):
        fit_spatial_sarma_panel(np.zeros(8), X, W, time_periods=2)


# ---------------------------------------------------------------------------
# fit_spatial_probit_panel
# ---------------------------------------------------------------------------


def test_fit_spatial_probit_panel_normal():
    np.random.seed(42)
    N, T, K = 4, 3, 2
    W = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    X = np.random.randn(N * T, K)
    y = np.random.randint(0, 2, N * T)

    res = fit_spatial_probit_panel(y, X, W, T)

    assert "coefficients" in res
    assert "spatial_rho" in res
    assert "std_errors" in res
    assert "z_stat" in res
    assert "p_values" in res
    assert "pseudo_r_squared" in res
    assert "log_likelihood" in res
    assert "classification_accuracy" in res
    assert "predicted_probabilities" in res

    assert len(res["coefficients"]) == K
    assert len(res["std_errors"]) == K
    assert len(res["z_stat"]) == K
    assert len(res["p_values"]) == K
    assert len(res["predicted_probabilities"]) == N * T
    assert isinstance(res["spatial_rho"], float)
    assert isinstance(res["pseudo_r_squared"], float)
    assert isinstance(res["log_likelihood"], float)
    assert isinstance(res["classification_accuracy"], float)
    assert 0.0 <= res["classification_accuracy"] <= 1.0


def test_fit_spatial_probit_panel_invalid_t():
    with pytest.raises(ValueError, match="time_periods must be >= 2"):
        fit_spatial_probit_panel(np.array([0, 1, 0]), np.ones((3, 2)), np.eye(3), 1)


def test_fit_spatial_probit_panel_invalid_w():
    with pytest.raises(ValueError, match="weights_matrix must be a square 2D array"):
        fit_spatial_probit_panel(np.zeros(6), np.ones((6, 2)), np.ones((3, 4)), 2)


def test_fit_spatial_probit_panel_invalid_n():
    with pytest.raises(ValueError, match="Number of spatial units \\(N\\) must be >= 3"):
        fit_spatial_probit_panel(np.zeros(4), np.ones((4, 2)), np.eye(2), 2)


def test_fit_spatial_probit_panel_invalid_y_binary():
    with pytest.raises(ValueError, match="dependent_var must contain only binary 0/1 values"):
        fit_spatial_probit_panel(np.array([0, 1, 2, 0, 1, 0]), np.ones((6, 2)), np.eye(3), 2)


def test_fit_spatial_probit_panel_invalid_y_shape():
    with pytest.raises(ValueError, match="dependent_var shape"):
        fit_spatial_probit_panel(np.zeros((3, 3)), np.ones((6, 2)), np.eye(3), 2)


def test_fit_spatial_probit_panel_invalid_x_shape():
    with pytest.raises(ValueError, match="independent_vars shape"):
        fit_spatial_probit_panel(np.zeros(6), np.ones((3, 3, 2)), np.eye(3), 2)


# ---------------------------------------------------------------------------
# fit_spatial_count_panel
# ---------------------------------------------------------------------------


def test_fit_spatial_count_panel_poisson():
    from planx.geostats import fit_spatial_count_panel

    np.random.seed(42)
    N, T, K = 4, 3, 2
    W = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    X = np.random.randn(N * T, K)
    y = np.random.poisson(lam=5.0, size=N * T)

    res = fit_spatial_count_panel(y, X, W, time_periods=T, model_type="poisson")

    assert "coefficients" in res
    assert "spatial_rho" in res
    assert "dispersion_alpha" in res
    assert "log_likelihood" in res
    assert "deviance" in res
    assert "pseudo_r_squared" in res
    assert "fitted_values" in res
    assert "residuals" in res

    assert len(res["coefficients"]) == K
    assert len(res["fitted_values"]) == N * T
    assert len(res["residuals"]) == N * T
    assert res["dispersion_alpha"] == 0.0
    assert isinstance(res["spatial_rho"], float)


def test_fit_spatial_count_panel_negative_binomial():
    from planx.geostats import fit_spatial_count_panel

    np.random.seed(42)
    N, T, K = 4, 3, 2
    W = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    X = np.random.randn(N * T, K)
    y = np.random.negative_binomial(n=5, p=0.5, size=N * T)

    res = fit_spatial_count_panel(y, X, W, time_periods=T, model_type="negative_binomial")

    assert len(res["coefficients"]) == K
    assert res["dispersion_alpha"] > 0.0
    assert len(res["fitted_values"]) == N * T


def test_fit_spatial_count_panel_validation():
    from planx.geostats import fit_spatial_count_panel

    y = np.array([1, 2, 3, 4, 5, 6])
    X = np.ones((6, 2))
    W = np.eye(3)

    with pytest.raises(ValueError, match="model_type"):
        fit_spatial_count_panel(y, X, W, time_periods=2, model_type="invalid")

    with pytest.raises(ValueError, match="time_periods must be >= 2"):
        fit_spatial_count_panel(y, X, W, time_periods=1)

    with pytest.raises(ValueError, match="non-negative"):
        fit_spatial_count_panel(np.array([-1, 2, 3, 4, 5, 6]), X, W, time_periods=2)


# ---------------------------------------------------------------------------
# fit_spatial_zip_panel
# ---------------------------------------------------------------------------


def test_fit_spatial_zip_panel_normal():
    from planx.geostats import fit_spatial_zip_panel

    np.random.seed(42)
    N, T, K = 4, 3, 2
    W = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    X = np.random.randn(N * T, K)
    y = np.random.poisson(lam=3.0, size=N * T)
    y[0] = 0
    y[3] = 0
    y[7] = 0

    res = fit_spatial_zip_panel(y, X, W, time_periods=T, dist="poisson")

    assert "count_coefficients" in res
    assert "zero_coefficients" in res
    assert "spatial_rho" in res
    assert "dispersion_alpha" in res
    assert "zero_inflation_mean" in res
    assert "log_likelihood" in res
    assert "pseudo_r_squared" in res
    assert "fitted_values" in res
    assert "zero_probabilities" in res

    assert len(res["count_coefficients"]) == K
    assert len(res["zero_coefficients"]) == K
    assert len(res["fitted_values"]) == N * T
    assert len(res["zero_probabilities"]) == N * T
    assert 0.0 <= res["zero_inflation_mean"] <= 1.0


def test_fit_spatial_zip_panel_validation():
    from planx.geostats import fit_spatial_zip_panel

    y = np.array([0, 1, 2, 0, 3, 4])
    X = np.ones((6, 2))
    W = np.eye(3)

    with pytest.raises(ValueError, match="dist must be"):
        fit_spatial_zip_panel(y, X, W, time_periods=2, dist="invalid")

    with pytest.raises(ValueError, match="time_periods must be >= 2"):
        fit_spatial_zip_panel(y, X, W, time_periods=1)

    with pytest.raises(ValueError, match="non-negative"):
        fit_spatial_zip_panel(np.array([-1, 0, 1, 2, 3, 4]), X, W, time_periods=2)


# ---------------------------------------------------------------------------
# fit_spatial_dynamic_panel_gmm
# ---------------------------------------------------------------------------


def test_fit_spatial_dynamic_panel_gmm_normal():
    from planx.geostats import fit_spatial_dynamic_panel_gmm

    np.random.seed(42)
    N, T, K = 4, 3, 2
    W = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    X = np.random.randn(N * T, K)
    y = np.random.randn(N * T)

    res = fit_spatial_dynamic_panel_gmm(y, X, W, time_periods=T)

    assert "gamma_lag" in res
    assert "spatial_rho" in res
    assert "beta" in res
    assert "std_errors" in res
    assert "z_stat" in res
    assert "p_values" in res
    assert "r_squared" in res
    assert "residuals" in res

    assert isinstance(res["gamma_lag"], float)
    assert isinstance(res["spatial_rho"], float)
    assert len(res["beta"]) == K
    assert len(res["std_errors"]) == K + 2
    assert 0.0 <= res["r_squared"] <= 1.0


def test_fit_spatial_dynamic_panel_gmm_validation():
    from planx.geostats import fit_spatial_dynamic_panel_gmm

    y = np.ones(6)
    X = np.ones((6, 2))
    W = np.eye(3)

    with pytest.raises(ValueError, match="time_periods must be >= 3"):
        fit_spatial_dynamic_panel_gmm(y, X, W, time_periods=2)


def test_fit_spatial_panel_sur():
    from planx.geostats import fit_spatial_panel_sur

    np.random.seed(42)
    N, T = 5, 4
    W = np.array([
        [0.0, 0.5, 0.5, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.3, 0.3, 0.0, 0.4, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ])

    y1 = np.random.randn(N, T)
    y2 = np.random.randn(N, T)
    x1 = np.random.randn(N, T, 2)
    x2 = np.random.randn(N, T, 3)

    res = fit_spatial_panel_sur([y1, y2], [x1, x2], W)
    assert res["num_equations"] == 2
    assert len(res["coefficients"]) == 2
    assert res["cross_equation_covariance"].shape == (2, 2)
    assert len(res["r_squared_per_equation"]) == 2


def test_fit_spatial_panel_tobit_lag():
    from planx.geostats import fit_spatial_panel_tobit_lag

    np.random.seed(42)
    N, T = 4, 3
    W = np.array([
        [0.0, 0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.3, 0.3, 0.0, 0.4],
        [0.0, 0.0, 1.0, 0.0],
    ])
    x = np.random.randn(N * T, 2)
    y = np.maximum(0.0, np.random.randn(N * T))

    res = fit_spatial_panel_tobit_lag(y, x, W, time_periods=T, lower_bound=0.0)

    assert "spatial_rho" in res
    assert "beta" in res
    assert "sigma" in res
    assert "censored_ratio" in res
    assert "log_likelihood" in res
    assert 0.0 <= res["censored_ratio"] <= 1.0





