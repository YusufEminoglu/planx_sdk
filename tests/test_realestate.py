# -*- coding: utf-8 -*-
"""Unit tests for planx.realestate submodule."""

import numpy as np

from planx.realestate import (
    automated_comps_selector,
    cap_rate_spatial_interpolator,
    hedonic_price_model,
    land_value_uplift,
    transit_oriented_premium_index,
)


def test_hedonic_price_model():
    y = np.array([200.0, 250.0, 300.0, 350.0, 400.0])
    X = np.array([[50, 2], [60, 3], [70, 3], [80, 4], [90, 4]])
    W = np.eye(5)

    res_ols = hedonic_price_model(y, X, model_type="ols")
    assert "r_squared" in res_ols
    assert res_ols["r_squared"] > 0.9

    res_lag = hedonic_price_model(y, X, weights_matrix=W, model_type="spatial_lag")
    assert "spatial_autoregressive_rho" in res_lag


def test_land_value_uplift():
    base = np.array([100.0, 110.0, 105.0, 100.0])
    post = np.array([180.0, 190.0, 115.0, 110.0])
    treat = np.array([True, True, False, False])

    res = land_value_uplift(base, post, treat)
    assert "average_treatment_effect_att" in res
    assert res["percentage_uplift"] > 0.0


def test_transit_oriented_premium_index():
    prices = np.array([300.0, 400.0, 500.0])
    dt = np.array([200.0, 500.0, 1200.0])
    da = np.array([100.0, 400.0, 800.0])

    res = transit_oriented_premium_index(prices, dt, da)
    assert "transit_oriented_premium_index" in res
    assert len(res["transit_oriented_premium_index"]) == 3


def test_automated_comps_selector():
    target = np.array([10.0, 10.0, 80.0, 5.0, 3.0])
    comps = np.array(
        [
            [10.1, 10.1, 82.0, 6.0, 3.0],
            [12.0, 12.0, 120.0, 2.0, 4.0],
            [10.2, 10.2, 78.0, 4.0, 3.0],
        ]
    )

    res = automated_comps_selector(target, comps, top_k=2)
    assert len(res["selected_comps_indices"]) == 2


def test_cap_rate_spatial_interpolator():
    yields = np.array([0.05, 0.06, 0.04])
    noi = np.array([50.0, 60.0, 40.0])
    coords = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

    res = cap_rate_spatial_interpolator(yields, noi, coords)
    assert "mean_cap_rate_pct" in res
    assert res["mean_cap_rate_pct"] > 0.0
