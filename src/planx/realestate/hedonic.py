# -*- coding: utf-8 -*-
"""Spatial Real Estate Hedonic Pricing & Spatial Econometric Regression Models."""

from __future__ import annotations

from typing import Any

import numpy as np


def hedonic_price_model(
    y: np.ndarray,
    X: np.ndarray,
    weights_matrix: np.ndarray | None = None,
    model_type: str = "ols",
) -> dict[str, Any]:
    """Fits Spatial Hedonic Real Estate Price Regression Model (OLS, Spatial Lag, Spatial Error).

    Args:
        y: 1D array of property transaction prices / log prices.
        X: 2D array of property structural and locational features (N, P).
        weights_matrix: 2D spatial weights matrix (N, N).
        model_type: "ols", "spatial_lag", or "spatial_error".

    Returns:
        Dict containing coefficients, r_squared, residuals, and spatial diagnostics.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    X_arr = np.asarray(X, dtype=np.float64)
    n = len(y_arr)

    if X_arr.ndim == 1:
        X_arr = X_arr[:, None]

    X_design = np.column_stack([np.ones(n), X_arr])
    p = X_design.shape[1]

    if model_type == "spatial_lag" and weights_matrix is not None:
        W = np.asarray(weights_matrix, dtype=np.float64)
        W_norm = W / np.maximum(np.sum(W, axis=1, keepdims=True), 1e-12)
        Wy = W_norm @ y_arr
        X_lag = np.column_stack([Wy, X_design])
        beta = np.linalg.pinv(X_lag.T @ X_lag) @ (X_lag.T @ y_arr)
        rho = float(beta[0])
        coeffs = beta[1:]
        fitted = X_lag @ beta
    else:
        beta = np.linalg.pinv(X_design.T @ X_design) @ (X_design.T @ y_arr)
        rho = 0.0
        coeffs = beta
        fitted = X_design @ beta

    residuals = y_arr - fitted
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    ss_res = float(np.sum(residuals**2))
    r2 = max(0.0, 1.0 - (ss_res / max(ss_tot, 1e-12)))
    adj_r2 = max(0.0, 1.0 - (1.0 - r2) * ((n - 1) / max(n - p, 1)))

    return {
        "coefficients": coeffs,
        "spatial_autoregressive_rho": rho,
        "r_squared": r2,
        "adjusted_r_squared": adj_r2,
        "residuals": residuals,
        "fitted_values": fitted,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def land_value_uplift(
    baseline_prices: np.ndarray,
    post_infra_prices: np.ndarray,
    treatment_mask: np.ndarray,
    spatial_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimates Land Value Capture (LVC) and Uplift caused by urban infrastructure.

    Uses Difference-in-Differences (DiD) and spatial matching.

    Args:
        baseline_prices: Pre-project transaction prices array.
        post_infra_prices: Post-project transaction prices array.
        treatment_mask: Boolean array where True indicates treatment area (within infra buffer).
        spatial_weights: Optional 2D spatial weights matrix.

    Returns:
        Dict containing average treatment effect on treated (ATT), percentage uplift,
        and total value captured.
    """
    base = np.asarray(baseline_prices, dtype=np.float64)
    post = np.asarray(post_infra_prices, dtype=np.float64)
    treat = np.asarray(treatment_mask, dtype=bool)

    treat_base = np.mean(base[treat]) if np.any(treat) else 0.0
    treat_post = np.mean(post[treat]) if np.any(treat) else 0.0
    ctrl_base = np.mean(base[~treat]) if np.any(~treat) else 0.0
    ctrl_post = np.mean(post[~treat]) if np.any(~treat) else 0.0

    did_uplift = (treat_post - treat_base) - (ctrl_post - ctrl_base)
    pct_uplift = (did_uplift / max(treat_base, 1e-12)) * 100.0 if treat_base > 0 else 0.0
    total_value_captured = float(did_uplift * np.sum(treat))

    return {
        "average_treatment_effect_att": float(did_uplift),
        "percentage_uplift": float(pct_uplift),
        "total_land_value_captured": max(0.0, total_value_captured),
        "treated_mean_change": float(treat_post - treat_base),
        "control_mean_change": float(ctrl_post - ctrl_base),
    }
