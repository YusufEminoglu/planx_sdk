# -*- coding: utf-8 -*-
"""Spatial statistics engines for calculations."""

from __future__ import annotations

import importlib.util
import logging
import math
from typing import Any, Optional, cast

import numpy as np
from scipy import sparse, stats
from scipy.optimize import linprog

logger = logging.getLogger("PlanX GeoStats Lab")

# Try importing PySAL modules
HAS_PYQ = all(importlib.util.find_spec(name) is not None for name in ("esda", "libpysal"))


def calculate_getis_ord(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    star: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates the Getis-Ord Gi or Gi* statistics."""
    n = len(y)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    conf_bins = np.zeros(n, dtype=int)

    if n <= 1:
        return z_scores, p_values, conf_bins

    y_mean = np.mean(y)
    y_std = np.std(y)

    if y_std == 0:
        logger.warning("Standard deviation of the target field is zero. Gi* cannot be calculated.")
        return z_scores, p_values, conf_bins

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])

        valid_neigh_indices = []
        valid_w = []
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                valid_neigh_indices.append(id_to_idx[nid])
                w = f_weights[j] if j < len(f_weights) else 1.0
                valid_w.append(w)

        if star:
            if idx not in valid_neigh_indices:
                valid_neigh_indices.append(idx)
                valid_w.append(1.0)

        num_neighbors = len(valid_neigh_indices)
        if num_neighbors == 0:
            continue

        w_row = np.array(valid_w)
        y_neigh = y[valid_neigh_indices]
        sum_w_x = np.sum(w_row * y_neigh)
        sum_w = np.sum(w_row)
        sum_w2 = np.sum(w_row**2)

        numerator = sum_w_x - y_mean * sum_w
        denom_term = (n * sum_w2 - (sum_w**2)) / (n - 1)
        if denom_term < 0:
            denom_term = 0.0
        denominator = y_std * math.sqrt(denom_term)

        if denominator > 0:
            z = numerator / denominator
            z_scores[idx] = z
            p = 1.0 - math.erf(abs(z) / math.sqrt(2.0))
            p_values[idx] = p

            if p < 0.01:
                conf_bins[idx] = 3 if z > 0 else -3
            elif p < 0.05:
                conf_bins[idx] = 2 if z > 0 else -2
            elif p < 0.10:
                conf_bins[idx] = 1 if z > 0 else -1
            else:
                conf_bins[idx] = 0

    return z_scores, p_values, conf_bins


def calculate_bivariate_lee_l(
    x_values: np.ndarray,
    y_values: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Calculates local bivariate spatial association using a Lee's L style statistic."""
    n = len(x_values)
    if n < 3:
        raise ValueError("Bivariate spatial association requires at least 3 observations.")
    x_std = np.std(x_values)
    y_std = np.std(y_values)
    local_l = np.zeros(n)
    spatial_lag_y = np.zeros(n)
    classes = ["Not Significant"] * n
    if x_std == 0 or y_std == 0:
        return local_l, spatial_lag_y, classes

    zx = (x_values - np.mean(x_values)) / x_std
    zy = (y_values - np.mean(y_values)) / y_std
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    for idx, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        lag = 0.0
        w_sum = 0.0
        for j, nid in enumerate(neighs):
            if nid in id_to_idx:
                w = w_list[j] if j < len(w_list) else 0.0
                lag += w * zy[id_to_idx[nid]]
                w_sum += w
        if w_sum == 0:
            continue
        spatial_lag_y[idx] = lag
        local_l[idx] = zx[idx] * lag
        if local_l[idx] > 0:
            if zx[idx] > 0 and lag > 0:
                classes[idx] = "High-X / High-Y Lag"
            elif zx[idx] < 0 and lag < 0:
                classes[idx] = "Low-X / Low-Y Lag"
        elif local_l[idx] < 0:
            if zx[idx] > 0 and lag < 0:
                classes[idx] = "High-X / Low-Y Lag"
            elif zx[idx] < 0 and lag > 0:
                classes[idx] = "Low-X / High-Y Lag"
    return local_l, spatial_lag_y, classes


def calculate_mean_center(
    x_coords: np.ndarray, y_coords: np.ndarray, weights: Optional[np.ndarray] = None
) -> tuple[float, float]:
    """Calculates the mean center of coordinate pairs."""
    if weights is None or len(weights) == 0:
        return float(np.mean(x_coords)), float(np.mean(y_coords))

    total_weight = np.sum(weights)
    if total_weight == 0:
        return float(np.mean(x_coords)), float(np.mean(y_coords))

    mean_x = np.sum(x_coords * weights) / total_weight
    mean_y = np.sum(y_coords * weights) / total_weight
    return float(mean_x), float(mean_y)


def calculate_central_feature(
    x_coords: np.ndarray, y_coords: np.ndarray, weights: Optional[np.ndarray] = None
) -> int:
    """Finds the index of the central feature based on minimum total distance."""
    n = len(x_coords)
    if n <= 1:
        return 0

    coords = np.column_stack((x_coords, y_coords))
    # Pairwise Euclidean distances
    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    if weights is None or len(weights) == 0:
        dist_sums = dists.sum(axis=1)
    else:
        # Weighted distance sum
        dist_sums = (dists * weights[None, :]).sum(axis=1)

    return int(np.argmin(dist_sums))


def calculate_sde(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
    num_std: int = 1,
) -> tuple[float, float, float, float, float]:
    """Calculates Standard Deviational Ellipse (SDE) parameters.

    Returns:
        A tuple of (mean_x, mean_y, rotation_angle_radians, semi_major_axis, semi_minor_axis)
    """
    n = len(x_coords)
    mean_x, mean_y = calculate_mean_center(x_coords, y_coords, weights)

    if n <= 2:
        return mean_x, mean_y, 0.0, 0.0, 0.0

    x_prime = x_coords - mean_x
    y_prime = y_coords - mean_y

    W = np.ones(n) if (weights is None or len(weights) == 0) else weights
    sum_w = np.sum(W)
    if sum_w == 0:
        W = np.ones(n)
        sum_w = n

    sum_x2 = np.sum(W * (x_prime**2))
    sum_y2 = np.sum(W * (y_prime**2))
    sum_xy = np.sum(W * x_prime * y_prime)

    # Calculate rotation angle theta
    # Using the standardPrincipal Orientation formula
    theta = 0.5 * np.arctan2(2 * sum_xy, sum_x2 - sum_y2)

    # Standard deviations along rotated axes
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    std_x = np.sqrt(np.sum(W * (x_prime * cos_t - y_prime * sin_t) ** 2) / sum_w)
    std_y = np.sqrt(np.sum(W * (x_prime * sin_t + y_prime * cos_t) ** 2) / sum_w)

    # Semi-major/minor axes scaling
    semi_x = num_std * std_x
    semi_y = num_std * std_y

    # Let semi_major be the larger one
    if semi_x >= semi_y:
        semi_major = semi_x
        semi_minor = semi_y
        angle = theta
    else:
        semi_major = semi_y
        semi_minor = semi_x
        angle = theta + np.pi / 2.0  # Align rotation to semi-major axis

    return mean_x, mean_y, float(angle), float(semi_major), float(semi_minor)


def calculate_local_moran(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Calculates Anselin Local Moran's I cluster and outlier diagnostics.

    Returns:
        A tuple of:
          - I_values: NumPy array of Moran's I indices (floats)
          - z_scores: NumPy array of z-scores (floats)
          - p_values: NumPy array of p-values (floats)
          - quadrants: List of strings ('HH', 'LL', 'HL', 'LH', 'Not Significant')
    """
    n = len(y)
    I_values = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    quadrants = ["Not Significant"] * n

    if n <= 2:
        return I_values, z_scores, p_values, quadrants

    y_mean = np.mean(y)
    z = y - y_mean
    m2 = np.sum(z**2) / n

    if m2 == 0:
        return I_values, z_scores, p_values, quadrants

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    b2 = (n * np.sum(z**4)) / (np.sum(z**2) ** 2)  # Kurtosis

    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])

        valid_neigh_indices = []
        valid_w = []
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                valid_neigh_indices.append(id_to_idx[nid])
                valid_w.append(f_weights[j])

        w_sum = sum(valid_w)
        w_sum2 = sum(w**2 for w in valid_w)

        if w_sum == 0:
            continue

        # Spatial lag
        spatial_lag = np.sum(np.array(valid_w) * z[valid_neigh_indices])
        I_i = (z[idx] / m2) * spatial_lag
        I_values[idx] = I_i

        # Expected value under randomization
        E_Ii = -w_sum / (n - 1)

        # Variance under randomization (Anselin 1995 formula)
        # Var(Ii) = w_i2 * (n - b2) / (n - 1) +
        #           (w_i^2 - w_i2) * (2b2 - n) / ((n - 1)(n - 2)) - E(Ii)^2
        var_term1 = (w_sum2 * (n - b2)) / (n - 1)

        if n > 2:
            var_term2 = ((w_sum**2 - w_sum2) * (2 * b2 - n)) / ((n - 1) * (n - 2))
        else:
            var_term2 = 0.0

        var_Ii = var_term1 + var_term2 - (E_Ii**2)

        if var_Ii > 0:
            z_i = (I_i - E_Ii) / math.sqrt(var_Ii)
            z_scores[idx] = z_i
            p = 1.0 - math.erf(abs(z_i) / math.sqrt(2.0))
            p_values[idx] = p

            # Quadrant categorization (HH, LL, HL, LH)
            if p < 0.05:
                # Value relative to mean
                high_val = z[idx] > 0
                # Lag relative to mean
                high_lag = spatial_lag > 0

                if high_val and high_lag:
                    quadrants[idx] = "HH"
                elif not high_val and not high_lag:
                    quadrants[idx] = "LL"
                elif high_val and not high_lag:
                    quadrants[idx] = "HL"
                elif not high_val and high_lag:
                    quadrants[idx] = "LH"
            else:
                quadrants[idx] = "Not Significant"
        else:
            z_scores[idx] = 0.0
            p_values[idx] = 1.0
            quadrants[idx] = "Not Significant"

    return I_values, z_scores, p_values, quadrants


def calculate_local_geary(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    permutations: int = 199,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Calculates Anselin Local Geary's C using conditional permutation inference.

    Local Geary's C_i = sum_j w_ij * (z_i - z_j)^2, where z is the
    standardized attribute value. Small values indicate positive spatial
    association (similar values clustered together), while large values
    indicate negative spatial association (dissimilar neighboring values).

    Because no simple closed-form variance exists for the local statistic,
    significance is assessed via conditional permutation: for each location,
    its neighbors' values are repeatedly resampled (without replacement)
    from the remaining observations to build a reference distribution.

    Args:
        y: Attribute values.
        neighbors: Adjacency list mapping node ID to list of neighbor IDs.
        weights: Weights list mapping node ID to list of weights.
        id_order: List of node IDs in the order they correspond to y.
        permutations: Number of conditional permutations per location.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of:
          - c_values: NumPy array of local Geary's C statistics
          - z_scores: NumPy array of pseudo z-scores from permutation
          - p_values: NumPy array of pseudo (two-sided) p-values
          - quadrants: List of strings ('HH', 'LL', 'HL', 'LH', 'Not Significant')
    """
    n = len(y)
    c_values = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    quadrants = ["Not Significant"] * n

    if n <= 2:
        return c_values, z_scores, p_values, quadrants

    y_mean = np.mean(y)
    y_std = np.std(y)

    if y_std == 0:
        return c_values, z_scores, p_values, quadrants

    z = (y - y_mean) / y_std
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    n_perm = max(0, int(permutations))
    rng = np.random.default_rng(seed)
    all_indices = np.arange(n)

    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])

        valid_neigh_indices = []
        valid_w = []
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx and id_to_idx[nid] != idx:
                valid_neigh_indices.append(id_to_idx[nid])
                valid_w.append(f_weights[j])

        if not valid_w:
            continue

        w_arr = np.array(valid_w)
        neigh_z = z[valid_neigh_indices]
        c_i = float(np.sum(w_arr * (z[idx] - neigh_z) ** 2))
        c_values[idx] = c_i
        spatial_lag = float(np.mean(neigh_z))

        if n_perm > 0:
            others = all_indices[all_indices != idx]
            k = len(valid_w)
            if k > len(others):
                continue
            sim = np.empty(n_perm)
            for perm_idx in range(n_perm):
                sampled_idx = rng.choice(others, size=k, replace=False)
                sim[perm_idx] = np.sum(w_arr * (z[idx] - z[sampled_idx]) ** 2)

            mean_sim = float(np.mean(sim))
            std_sim = float(np.std(sim))
            z_scores[idx] = (c_i - mean_sim) / std_sim if std_sim > 0 else 0.0

            p_low = (int(np.sum(sim <= c_i)) + 1) / (n_perm + 1)
            p_high = (int(np.sum(sim >= c_i)) + 1) / (n_perm + 1)
            p_values[idx] = min(1.0, 2.0 * min(p_low, p_high))

            if p_values[idx] < 0.05:
                high_val = z[idx] > 0
                high_lag = spatial_lag > 0
                if c_i < mean_sim:
                    # Positive spatial association: similar values clustered.
                    if high_val and high_lag:
                        quadrants[idx] = "HH"
                    elif not high_val and not high_lag:
                        quadrants[idx] = "LL"
                else:
                    # Negative spatial association: dissimilar neighboring values.
                    if high_val and not high_lag:
                        quadrants[idx] = "HL"
                    elif not high_val and high_lag:
                        quadrants[idx] = "LH"

    return c_values, z_scores, p_values, quadrants


def _chi2_sf_approx(x: float, df: int) -> float:
    """Wilson-Hilferty transformation approximation for Chi-Square Survival Function (p-value)."""
    if x <= 0:
        return 1.0
    if df == 2:
        return float(math.exp(-0.5 * x))  # Exact for df=2

    # Wilson-Hilferty approximation: Chi2 to normal
    d = float(df)
    z = ((x / d) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * d))) / math.sqrt(2.0 / (9.0 * d))
    p_val = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return float(max(0.0, min(1.0, p_val)))


def calculate_ols(
    y: np.ndarray,
    X_data: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    x_names: list[str],
) -> dict:
    """Performs Ordinary Least Squares (OLS) regression and diagnostic tests.

    Args:
        y: 1D dependent variable array (n,)
        X_data: 2D independent variables array (n, p)
        neighbors: Weights neighbors dict
        weights: Weights values dict
        id_order: Feature IDs
        x_names: Names of independent variables

    Returns:
        A dictionary containing coefficient estimates, diagnostics, residuals, etc.
    """
    n = len(y)
    p = X_data.shape[1]

    # Add intercept column
    X = np.column_stack((np.ones(n), X_data))

    # Solve beta = (X.T * X)^-1 * X.T * Y
    try:
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta = xtx_inv @ X.T @ y
    except Exception as e:
        logger.error("Linear algebra inversion failed in OLS regression: %s", e)
        raise ValueError(f"Regression inversion failed: {e}") from e

    # Residuals
    y_pred = X @ beta
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # Variance of residuals
    df_err = n - p - 1
    if df_err <= 0:
        raise ValueError(
            f"Sample size ({n}) must be greater than number of variables ({p} + intercept)."
        )

    s2 = ss_res / df_err
    std_residuals = residuals / math.sqrt(s2) if s2 > 0 else np.zeros(n)

    # Standard Errors of Coefficients
    cov_beta = s2 * xtx_inv
    se_beta = np.sqrt(np.maximum(0.0, np.diagonal(cov_beta)))

    # t-statistics and p-values
    t_stats = np.zeros(p + 1)
    p_vals = np.ones(p + 1)
    for j in range(p + 1):
        if se_beta[j] > 0:
            t_stats[j] = beta[j] / se_beta[j]
            # Normal approximation for t-dist (very accurate for large df)
            p_vals[j] = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stats[j]) / math.sqrt(2.0))))
        else:
            t_stats[j] = 0.0
            p_vals[j] = 1.0

    # Model R2 & Adj R2
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_err

    # --- DIAGNOSTIC 1: Jarque-Bera normality test ---
    s2_ml = ss_res / n
    if s2_ml > 0:
        skew = np.sum(residuals**3) / n / (s2_ml**1.5)
        kurt = np.sum(residuals**4) / n / (s2_ml**2)
        jb_stat = (n / 6.0) * (skew**2 + 0.25 * (kurt - 3.0) ** 2)
        jb_p = _chi2_sf_approx(jb_stat, df=2)
    else:
        jb_stat, jb_p = 0.0, 1.0

    # --- DIAGNOSTIC 2: Koenker's Breusch-Pagan heteroskedasticity test ---
    # Auxiliary regression: e^2 on X_data
    g = residuals**2
    g_mean = np.mean(g)
    g_tot = np.sum((g - g_mean) ** 2)

    bp_stat, bp_p = 0.0, 1.0
    if g_tot > 0:
        try:
            # Regress g on independent variables
            beta_aux = np.linalg.pinv(X.T @ X) @ X.T @ g
            g_pred = X @ beta_aux
            g_res = g - g_pred
            ss_aux_res = np.sum(g_res**2)
            r2_aux = 1.0 - (ss_aux_res / g_tot)
            bp_stat = n * r2_aux
            bp_p = _chi2_sf_approx(bp_stat, df=p)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            bp_stat, bp_p = 0.0, 1.0

    # --- DIAGNOSTIC 3: Moran's I on Residuals ---
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    spatial_lag_e = np.zeros(n)
    for idx, fid in enumerate(id_order):
        f_neighs = neighbors.get(fid, [])
        f_weights = weights.get(fid, [])
        lag_sum = 0.0
        for j, nid in enumerate(f_neighs):
            if nid in id_to_idx:
                lag_sum += f_weights[j] * residuals[id_to_idx[nid]]
        spatial_lag_e[idx] = lag_sum

    if ss_res > 0:
        moran_i = np.sum(residuals * spatial_lag_e) / ss_res
    else:
        moran_i = 0.0

    # Return OLS results dictionary
    return {
        "coefficients": beta,
        "std_errors": se_beta,
        "t_statistics": t_stats,
        "p_values": p_vals,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "p": p,
        "df_err": df_err,
        "residuals": residuals,
        "std_residuals": std_residuals,
        "jarque_bera": (jb_stat, jb_p),
        "breusch_pagan": (bp_stat, bp_p),
        "residuals_moran": moran_i,
        "variable_names": ["Intercept"] + x_names,
    }


def calculate_global_moran(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
) -> tuple[float, float, float, float, float]:
    """Calculates Global Moran's I spatial autocorrelation.

    Returns:
        A tuple of (moran_i, expected_i, variance, z_score, p_value)
    """
    n = len(y)
    if n <= 3:
        raise ValueError("Global Moran's I requires at least 4 observations.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    y_mean = np.mean(y)
    z = y - y_mean
    sum_z2 = np.sum(z**2)
    sum_z4 = np.sum(z**4)

    if sum_z2 == 0:
        return 0.0, -1.0 / (n - 1), 0.0, 0.0, 1.0

    S0 = 0.0
    w_row_sums = np.zeros(n)
    w_col_sums = np.zeros(n)

    # First pass: compute S0, row sums, and column sums
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                S0 += w
                w_row_sums[i] += w
                w_col_sums[j] += w

    if S0 == 0:
        return 0.0, -1.0 / (n - 1), 0.0, 0.0, 1.0

    # Second pass: compute S1
    S1 = 0.0
    for fid in id_order:
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w_ij in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                w_ji = 0.0
                j_neighs = neighbors.get(j_fid, [])
                j_w_list = weights.get(j_fid, [])
                if fid in j_neighs:
                    w_ji = j_w_list[j_neighs.index(fid)]
                S1 += (w_ij + w_ji) ** 2
    S1 = 0.5 * S1

    # Compute S2
    S2 = np.sum((w_row_sums + w_col_sums) ** 2)

    # Calculate Moran's I
    numerator = 0.0
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                numerator += w * z[i] * z[j]

    moran_i = (n / S0) * (numerator / sum_z2)

    # Expected value
    expected_i = -1.0 / (n - 1)

    # Kurtosis term D
    D = (n * sum_z4) / (sum_z2**2)

    # Variance under randomization
    num_var = n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * S0**2) - D * (
        (n**2 - n) * S1 - 2 * n * S2 + 6 * S0**2
    )
    den_var = (n - 1) * (n - 2) * (n - 3) * S0**2

    variance = num_var / den_var - (expected_i**2) if den_var > 0 else 0.0
    if variance > 0:
        z_score = (moran_i - expected_i) / math.sqrt(variance)
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
    else:
        variance = 0.0
        z_score = 0.0
        p_value = 1.0

    return float(moran_i), float(expected_i), float(variance), float(z_score), float(p_value)


def calculate_global_geary(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
) -> tuple[float, float, float, float, float]:
    """Calculates Global Geary's C spatial autocorrelation.

    Args:
        y: Attribute values.
        neighbors: Adjacency list mapping node ID to list of neighbor IDs.
        weights: Weights list mapping node ID to list of weights.
        id_order: List of node IDs in the order they correspond to y.

    Returns:
        A tuple of (geary_c, expected_c, variance_rand, z_score_rand, p_value_rand)
    """
    n = len(y)
    if n <= 3:
        raise ValueError("Global Geary's C requires at least 4 observations.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    y_mean = np.mean(y)
    z = y - y_mean
    sum_z2 = np.sum(z**2)
    sum_z4 = np.sum(z**4)

    if sum_z2 == 0:
        return 1.0, 1.0, 0.0, 0.0, 1.0

    S0 = 0.0
    w_row_sums = np.zeros(n)
    w_col_sums = np.zeros(n)

    # First pass: compute S0, row sums, and column sums
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                S0 += w
                w_row_sums[i] += w
                w_col_sums[j] += w

    if S0 == 0:
        return 1.0, 1.0, 0.0, 0.0, 1.0

    # Second pass: compute S1
    S1 = 0.0
    for fid in id_order:
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w_ij in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                w_ji = 0.0
                j_neighs = neighbors.get(j_fid, [])
                j_w_list = weights.get(j_fid, [])
                if fid in j_neighs:
                    w_ji = j_w_list[j_neighs.index(fid)]
                S1 += (w_ij + w_ji) ** 2
    S1 = 0.5 * S1

    # Compute S2
    S2 = np.sum((w_row_sums + w_col_sums) ** 2)

    # Calculate Geary's C observed value
    numerator = 0.0
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                numerator += w * (y[i] - y[j]) ** 2

    geary_c = ((n - 1) * numerator) / (2.0 * S0 * sum_z2)
    expected_c = 1.0

    k = (sum_z4 / n) / ((sum_z2 / n) ** 2)
    n2 = n * n
    s02 = S0 * S0

    A = (n - 1) * S1 * (n2 - 3 * n + 3 - (n - 1) * k)
    B = 0.25 * ((n - 1) * S2 * (n2 + 3 * n - 6 - (n2 - n + 2) * k))
    C_term = s02 * (n2 - 3 - (n - 1) ** 2 * k)

    variance = (A - B + C_term) / (n * (n - 2) * (n - 3) * s02)

    if variance > 0:
        z_score = (geary_c - expected_c) / math.sqrt(variance)
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
    else:
        variance = 0.0
        z_score = 0.0
        p_value = 1.0

    return float(geary_c), float(expected_c), float(variance), float(z_score), float(p_value)


def calculate_spatial_gini(
    values: np.ndarray,
    neighbors: dict[int, list[int]],
    id_order: list[int],
    permutations: int = 99,
    seed: int = 42,
) -> dict:
    """Calculate classic Gini plus Rey-Smith style spatial Gini decomposition.

    The decomposition splits the pairwise absolute-difference numerator into
    neighbor and non-neighbor components. Components use the same denominator
    as the classic Gini, so neighbor_component + non_neighbor_component == gini.
    """
    y = np.array(values, dtype=float)
    finite = np.isfinite(y)
    if not np.all(finite):
        y = y[finite]
        id_order = [fid for fid, ok in zip(id_order, finite) if ok]

    n = int(len(y))
    if n < 2:
        raise ValueError("Spatial Gini requires at least 2 finite numeric observations.")
    if np.any(y < 0.0):
        raise ValueError("Gini coefficients require non-negative values.")

    mean_value = float(np.mean(y))
    pair_indices, neighbor_flags = _spatial_gini_pair_index(neighbors, id_order)
    total_pair_count = int(len(pair_indices))
    if total_pair_count == 0:
        raise ValueError("Spatial Gini requires at least one observation pair.")

    denominator = float((n**2) * mean_value)
    pair_sum, neighbor_sum, non_neighbor_sum = _spatial_gini_pair_sums(
        y, pair_indices, neighbor_flags
    )

    gini = pair_sum / denominator if denominator > 0.0 else 0.0
    neighbor_component = neighbor_sum / denominator if denominator > 0.0 else 0.0
    non_neighbor_component = non_neighbor_sum / denominator if denominator > 0.0 else 0.0

    neighbor_pair_count = int(sum(1 for flag in neighbor_flags if flag))
    non_neighbor_pair_count = total_pair_count - neighbor_pair_count
    neighbor_avg_diff = neighbor_sum / neighbor_pair_count if neighbor_pair_count else None
    non_neighbor_avg_diff = (
        non_neighbor_sum / non_neighbor_pair_count if non_neighbor_pair_count else None
    )
    neighbor_share = neighbor_sum / pair_sum if pair_sum > 0.0 else 0.0
    non_neighbor_share = non_neighbor_sum / pair_sum if pair_sum > 0.0 else 0.0
    polarization = None
    if (
        neighbor_avg_diff is not None
        and non_neighbor_avg_diff is not None
        and neighbor_avg_diff > 0.0
    ):
        polarization = non_neighbor_avg_diff / neighbor_avg_diff

    result = {
        "n": n,
        "mean": mean_value,
        "sum": float(np.sum(y)),
        "gini": float(gini),
        "pair_abs_sum": float(pair_sum),
        "neighbor_abs_sum": float(neighbor_sum),
        "non_neighbor_abs_sum": float(non_neighbor_sum),
        "neighbor_component": float(neighbor_component),
        "non_neighbor_component": float(non_neighbor_component),
        "neighbor_share": float(neighbor_share),
        "non_neighbor_share": float(non_neighbor_share),
        "spatial_gini": float(non_neighbor_share),
        "neighbor_pair_count": neighbor_pair_count,
        "non_neighbor_pair_count": non_neighbor_pair_count,
        "total_pair_count": total_pair_count,
        "neighbor_avg_diff": neighbor_avg_diff,
        "non_neighbor_avg_diff": non_neighbor_avg_diff,
        "polarization": polarization,
        "permutations": int(max(0, permutations)),
        "expected_non_neighbor_component": None,
        "std_non_neighbor_component": None,
        "z_non_neighbor_component": None,
        "p_sim": None,
        "p_low_sim": None,
        "expected_polarization": 1.0 if polarization is not None else None,
        "polarization_p_sim": None,
    }

    if permutations <= 0 or non_neighbor_pair_count == 0 or pair_sum <= 0.0:
        return result

    rng = np.random.default_rng(seed)
    sim_non_neighbor = np.zeros(int(permutations), dtype=float)
    sim_polarization = []
    for idx in range(int(permutations)):
        permuted = rng.permutation(y)
        sim_pair_sum, sim_neighbor_sum, sim_non_neighbor_sum = _spatial_gini_pair_sums(
            permuted,
            pair_indices,
            neighbor_flags,
        )
        if sim_pair_sum > 0.0:
            sim_non_neighbor[idx] = sim_non_neighbor_sum / denominator
        if neighbor_pair_count and non_neighbor_pair_count and sim_neighbor_sum > 0.0:
            sim_neighbor_avg = sim_neighbor_sum / neighbor_pair_count
            sim_non_neighbor_avg = sim_non_neighbor_sum / non_neighbor_pair_count
            sim_polarization.append(sim_non_neighbor_avg / sim_neighbor_avg)

    expected = float(np.mean(sim_non_neighbor))
    std = float(np.std(sim_non_neighbor))
    result["expected_non_neighbor_component"] = expected
    result["std_non_neighbor_component"] = std
    result["z_non_neighbor_component"] = (
        float((non_neighbor_component - expected) / std) if std > 0.0 else 0.0
    )
    result["p_sim"] = float(
        (int(np.sum(sim_non_neighbor >= non_neighbor_component)) + 1) / (permutations + 1)
    )
    result["p_low_sim"] = float(
        (int(np.sum(sim_non_neighbor <= non_neighbor_component)) + 1) / (permutations + 1)
    )

    if polarization is not None and sim_polarization:
        sim_pol = np.array(sim_polarization, dtype=float)
        result["polarization_p_sim"] = float(
            (int(np.sum(sim_pol >= polarization)) + 1) / (len(sim_pol) + 1)
        )

    return result


def _spatial_gini_pair_index(
    neighbors: dict[int, list[int]],
    id_order: list[int],
) -> tuple[list[tuple[int, int]], list[bool]]:
    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    neighbor_pairs = set()
    for fid in id_order:
        i = id_to_idx[fid]
        for nid in neighbors.get(fid, []):
            if nid not in id_to_idx or nid == fid:
                continue
            j = id_to_idx[nid]
            neighbor_pairs.add((i, j) if i < j else (j, i))

    pair_indices: list[tuple[int, int]] = []
    neighbor_flags: list[bool] = []
    n = len(id_order)
    for i in range(n - 1):
        for j in range(i + 1, n):
            pair = (i, j)
            pair_indices.append(pair)
            neighbor_flags.append(pair in neighbor_pairs)
    return pair_indices, neighbor_flags


def _spatial_gini_pair_sums(
    y: np.ndarray,
    pair_indices: list[tuple[int, int]],
    neighbor_flags: list[bool],
) -> tuple[float, float, float]:
    total_sum = 0.0
    neighbor_sum = 0.0
    non_neighbor_sum = 0.0
    for (i, j), is_neighbor in zip(pair_indices, neighbor_flags):
        diff = abs(float(y[i]) - float(y[j]))
        total_sum += diff
        if is_neighbor:
            neighbor_sum += diff
        else:
            non_neighbor_sum += diff
    return total_sum, neighbor_sum, non_neighbor_sum


def calculate_average_nearest_neighbor(
    x: np.ndarray, y: np.ndarray, study_area: Optional[float] = None
) -> tuple[float, float, float, float, float, float]:
    """Calculates Average Nearest Neighbor statistics.

    Returns:
        A tuple of (observed_mean, expected_mean, nn_ratio, z_score, p_value, study_area)
    """
    n = len(x)
    if n <= 1:
        raise ValueError("Average Nearest Neighbor requires at least 2 points.")

    coords = np.column_stack((x, y))

    # Try using scikit-learn KDTree for maximum performance, fallback to vectorized NumPy
    try:
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(coords)
        distances, _ = nbrs.kneighbors(coords)
        nn_dists = distances[:, 1]
    except ImportError:
        # Vectorized chunked NumPy distance finder to protect memory
        nn_dists = np.zeros(n)
        chunk_size = 1000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_coords = coords[start:end]
            d = np.sqrt(((chunk_coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
            for i in range(start, end):
                d[i - start, i] = np.inf
            nn_dists[start:end] = np.min(d, axis=1)

    observed_mean = float(np.mean(nn_dists))

    # Fallback to minimum bounding box area if study area is not provided
    if study_area is None or study_area <= 0:
        min_x, max_x = np.min(x), np.max(x)
        min_y, max_y = np.min(y), np.max(y)
        w = max_x - min_x
        h = max_y - min_y
        study_area = float(w * h) if (w * h > 0) else 1.0

    density = n / study_area
    expected_mean = 0.5 / math.sqrt(density)

    # Standard error
    se = 0.26136 / math.sqrt(n * density)

    nn_ratio = observed_mean / expected_mean if expected_mean > 0 else 1.0
    z_score = (observed_mean - expected_mean) / se if se > 0 else 0.0
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))

    return observed_mean, expected_mean, nn_ratio, z_score, p_value, study_area


def calculate_standard_distance(
    x_coords: np.ndarray, y_coords: np.ndarray, weights: Optional[np.ndarray] = None
) -> tuple[float, float, float]:
    """Calculates Standard Distance and mean center.

    Returns:
        A tuple of (mean_x, mean_y, standard_distance)
    """
    n = len(x_coords)
    if n == 0:
        return 0.0, 0.0, 0.0

    mean_x, mean_y = calculate_mean_center(x_coords, y_coords, weights)

    if weights is None or len(weights) == 0:
        var_x = np.sum((x_coords - mean_x) ** 2) / n
        var_y = np.sum((y_coords - mean_y) ** 2) / n
    else:
        sum_w = np.sum(weights)
        if sum_w == 0:
            var_x = np.sum((x_coords - mean_x) ** 2) / n
            var_y = np.sum((y_coords - mean_y) ** 2) / n
        else:
            var_x = np.sum(weights * (x_coords - mean_x) ** 2) / sum_w
            var_y = np.sum(weights * (y_coords - mean_y) ** 2) / sum_w

    std_distance = math.sqrt(var_x + var_y)
    return mean_x, mean_y, std_distance


def calculate_gwr(
    y: np.ndarray,
    X_data: np.ndarray,
    coords: np.ndarray,
    bandwidth: float,
    kernel_type: str = "adaptive_bisquare",
) -> dict:
    """Performs Geographically Weighted Regression (GWR) analysis.

    Args:
        y: Dependent variable (n,)
        X_data: Independent variables (n, p)
        coords: Centroid coordinates (n, 2)
        bandwidth: Kernel bandwidth (distance or neighbor count)
        kernel_type: fixed_gaussian, fixed_bisquare, or adaptive_bisquare

    Returns:
        A dictionary containing GWR coefficients, errors, local R2, and statistics.
    """
    n = len(y)
    p = X_data.shape[1]

    # Add intercept column
    X = np.column_stack((np.ones(n), X_data))

    # Initialize results
    local_beta = np.zeros((n, p + 1))
    local_se = np.zeros((n, p + 1))
    local_t = np.zeros((n, p + 1))
    y_pred = np.zeros(n)
    local_r2 = np.zeros(n)
    local_support = np.zeros(n, dtype=int)

    # Compute distance matrix
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    for i in range(n):
        dists = dists_matrix[i]

        # Calculate weights based on kernel type
        if kernel_type == "fixed_gaussian":
            w = np.exp(-0.5 * (dists / bandwidth) ** 2)
        elif kernel_type == "fixed_bisquare":
            w = np.zeros(n)
            mask = dists < bandwidth
            w[mask] = (1.0 - (dists[mask] / bandwidth) ** 2) ** 2
        elif kernel_type == "adaptive_bisquare":
            k = int(bandwidth)
            sorted_dists = np.sort(dists)
            d_k = sorted_dists[min(k - 1, n - 1)]
            w = np.zeros(n)
            if d_k > 0:
                mask = dists < d_k
                w[mask] = (1.0 - (dists[mask] / d_k) ** 2) ** 2
            else:
                w[dists == 0] = 1.0
        else:
            w = np.ones(n)

        sum_w = np.sum(w)
        if sum_w == 0:
            w = np.ones(n)
            sum_w = n
        local_support[i] = int(np.sum(w > 1e-12))

        # Solve local regression: beta_i = (X.T * W * X)^-1 * X.T * W * Y
        try:
            xtw = X.T * w
            xtwx = xtw @ X
            xtwx_inv = np.linalg.pinv(xtwx)
            beta_i = xtwx_inv @ xtw @ y
            local_beta[i] = beta_i
            y_pred[i] = X[i] @ beta_i

            # Standard errors and t-statistics
            res_i = y - (X @ beta_i)
            df_i = sum_w - p - 1
            if df_i > 0:
                s2_i = np.sum(w * (res_i**2)) / df_i
                cov_beta_i = s2_i * xtwx_inv
                se_beta_i = np.sqrt(np.maximum(0.0, np.diagonal(cov_beta_i)))
                local_se[i] = se_beta_i
                for j in range(p + 1):
                    if se_beta_i[j] > 0:
                        local_t[i, j] = beta_i[j] / se_beta_i[j]

            # Local R2
            y_w_mean = np.sum(w * y) / sum_w
            tss_i = np.sum(w * (y - y_w_mean) ** 2)
            rss_i = np.sum(w * (res_i**2))
            local_r2[i] = 1.0 - (rss_i / tss_i) if tss_i > 0 else 1.0
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            local_r2[i] = np.nan

    residuals = y - y_pred
    rss = np.sum(residuals**2)
    tss = np.sum((y - np.mean(y)) ** 2)
    global_r2 = 1.0 - (rss / tss) if tss > 0 else 0.0

    # Effective degrees of freedom (Trace of Hat Matrix)
    tr_S = 0.0
    for i in range(n):
        try:
            if kernel_type == "fixed_gaussian":
                w = np.exp(-0.5 * (dists_matrix[i] / bandwidth) ** 2)
            elif kernel_type == "fixed_bisquare":
                w = np.zeros(n)
                mask = dists_matrix[i] < bandwidth
                w[mask] = (1.0 - (dists_matrix[i][mask] / bandwidth) ** 2) ** 2
            elif kernel_type == "adaptive_bisquare":
                k = int(bandwidth)
                d_k = np.sort(dists_matrix[i])[min(k - 1, n - 1)]
                w = np.zeros(n)
                if d_k > 0:
                    mask = dists_matrix[i] < d_k
                    w[mask] = (1.0 - (dists_matrix[i][mask] / d_k) ** 2) ** 2
                else:
                    w[dists_matrix[i] == 0] = 1.0
            else:
                w = np.ones(n)

            xtw = X.T * w
            xtwx_inv = np.linalg.pinv(xtw @ X)
            s_i = (X[i] @ xtwx_inv) @ xtw
            tr_S += s_i[i]
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, IndexError):
            tr_S += 0.0

    if tr_S <= 0:
        tr_S = float(p + 1)

    aicc = np.inf
    if n - tr_S - 2 > 0 and rss > 0:
        aicc = n * np.log(rss / n) + n * np.log(2 * np.pi) + n * (n + tr_S) / (n - 2 - tr_S)

    return {
        "local_beta": local_beta,
        "local_se": local_se,
        "local_t": local_t,
        "local_support": local_support,
        "y_pred": y_pred,
        "residuals": residuals,
        "local_r2": local_r2,
        "rss": rss,
        "tss": tss,
        "r2": global_r2,
        "aicc": aicc,
        "effective_df": tr_S,
    }


def calculate_median_center(
    x: np.ndarray,
    y: np.ndarray,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[float, float, float]:
    """Calculates Median Center using Weiszfeld's algorithm.

    Returns:
        A tuple of (median_x, median_y, total_distance)
    """
    n = len(x)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return float(x[0]), float(y[0]), 0.0

    if weights is None or len(weights) == 0:
        weights = np.ones(n)

    # Initial guess: mean center
    cx, cy = calculate_mean_center(x, y, weights)

    for _ in range(max_iter):
        dists = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        dists = np.maximum(dists, 1e-12)

        inv_dists = weights / dists
        sum_inv = np.sum(inv_dists)

        if sum_inv == 0:
            break

        new_cx = np.sum(x * inv_dists) / sum_inv
        new_cy = np.sum(y * inv_dists) / sum_inv

        shift = math.sqrt((new_cx - cx) ** 2 + (new_cy - cy) ** 2)
        cx, cy = new_cx, new_cy

        if shift < tol:
            break

    total_dist = float(np.sum(weights * np.sqrt((x - cx) ** 2 + (y - cy) ** 2)))
    return float(cx), float(cy), total_dist


def calculate_general_g(
    values, neighbors: dict[int, list[int]], weights: dict[int, list[float]], id_order: list[int]
) -> tuple[float, float, float, float, float]:
    """Calculates Getis-Ord General G statistics under randomization.

    Returns:
        A tuple of (observed_g, expected_g, variance, z_score, p_value)
    """
    n = len(id_order)
    if n < 4:
        raise ValueError("General G requires at least 4 features.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}
    if isinstance(values, dict):
        x = np.array([float(values[fid]) for fid in id_order])
    else:
        x = np.array([float(values[i]) for i in range(n)])

    x_sum = np.sum(x)
    x_sum2 = np.sum(x**2)
    x_sum3 = np.sum(x**3)
    x_sum4 = np.sum(x**4)

    # Build dense weight matrix
    W = np.zeros((n, n))
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                W[i, j] = w

    np.fill_diagonal(W, 0.0)

    S0 = np.sum(W)
    A1 = np.sum(W**2)
    A2 = np.sum(W * W.T)

    row_sums = np.sum(W, axis=1)
    col_sums = np.sum(W, axis=0)

    A3 = np.sum(row_sums**2) + np.sum(col_sums**2)
    A4 = np.sum(row_sums * col_sums)
    A5 = S0**2

    # Observed G
    numerator = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                numerator += W[i, j] * x[i] * x[j]

    denominator = x_sum**2 - x_sum2
    if denominator == 0:
        return 0.0, 0.0, 0.0, 0.0, 1.0

    observed_g = numerator / denominator
    expected_g = S0 / (n * (n - 1))

    # Permutations of x values
    S_22 = x_sum2**2 - x_sum4
    S_211 = x_sum2 * (x_sum**2 - x_sum2) - 2.0 * x_sum * x_sum3 + 2.0 * x_sum4
    S_1111 = (
        (x_sum**4)
        - 6.0 * (x_sum**2) * x_sum2
        + 8.0 * x_sum * x_sum3
        + 3.0 * (x_sum2**2)
        - 6.0 * x_sum4
    )

    term1 = (A1 + A2) * S_22 / (n * (n - 1))
    term2 = (2.0 * A3 + 4.0 * A4 - 4.0 * A1 - 4.0 * A2) * S_211 / (n * (n - 1) * (n - 2))
    term3 = (
        (A5 - 2.0 * A3 - 4.0 * A4 + 3.0 * A1 + 3.0 * A2)
        * S_1111
        / (n * (n - 1) * (n - 2) * (n - 3))
    )

    E_A2 = term1 + term2 + term3
    E_G2 = E_A2 / (denominator**2)

    variance = E_G2 - (expected_g**2)
    if variance > 0:
        z_score = (observed_g - expected_g) / math.sqrt(variance)
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
    else:
        z_score = 0.0
        p_value = 1.0

    return float(observed_g), float(expected_g), float(variance), float(z_score), float(p_value)


def calculate_similarity_search(
    full_data: np.ndarray, target_indices: list[int], metric: str = "euclidean"
) -> np.ndarray:
    """Standardizes attributes and computes distance score from target feature profiles.

    Returns:
        An array of similarity scores for each feature in full_data.
    """
    n, p = full_data.shape
    if n == 0 or p == 0:
        return np.array([])

    # Z-score standardization
    means = np.mean(full_data, axis=0)
    stds = np.std(full_data, axis=0)
    stds[stds == 0.0] = 1.0  # avoid division by zero

    z_data = (full_data - means) / stds

    # Extract target profile (mean profile if multiple targets are selected)
    z_targets = z_data[target_indices]
    target_profile = np.mean(z_targets, axis=0)

    # Compute score based on selected distance metric
    if metric == "manhattan":
        scores = np.sum(np.abs(z_data - target_profile), axis=1)
    else:  # euclidean
        scores = np.sqrt(np.sum((z_data - target_profile) ** 2, axis=1))

    return scores


def calculate_distance_band_stats(x: np.ndarray, y: np.ndarray, k_neighbors: int = 1) -> dict:
    """Calculates statistics for distance to the k-th nearest neighbor.

    Returns:
        A dictionary containing min, max, mean, median, p25, p75 values.
    """
    n = len(x)
    if n <= 1:
        raise ValueError("At least 2 points are required to compute distance bands.")

    coords = np.column_stack((x, y))

    # Compute distance matrix
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    k_dists = np.zeros(n)
    for i in range(n):
        sorted_d = np.sort(dists_matrix[i])
        k_idx = min(k_neighbors, n - 1)
        k_dists[i] = sorted_d[k_idx]

    return {
        "min": float(np.min(k_dists)),
        "max": float(np.max(k_dists)),
        "mean": float(np.mean(k_dists)),
        "median": float(np.median(k_dists)),
        "p25": float(np.percentile(k_dists, 25)),
        "p75": float(np.percentile(k_dists, 75)),
    }


def calculate_kmeans(
    data: np.ndarray, k_clusters: int, max_iter: int = 100, tol: float = 1e-4, seed: int = 42
) -> tuple[np.ndarray, float]:
    """Performs K-Means clustering on feature attribute data.

    Returns:
        A tuple of (labels, wcss) where wcss is within-cluster sum of squares.
    """
    n, p = data.shape
    if n < k_clusters:
        raise ValueError("Number of data points must be greater than or equal to k_clusters.")

    # Z-score standardization
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0.0] = 1.0
    z_data = (data - means) / stds

    # K-Means++ initialization
    rng = np.random.default_rng(seed)
    centroids = np.zeros((k_clusters, p))

    # Pick first centroid
    idx = rng.choice(n)
    centroids[0] = z_data[idx]

    for c_idx in range(1, k_clusters):
        # Distance squared to closest centroid
        dists_sq = np.min(
            [np.sum((z_data - centroids[c]) ** 2, axis=1) for c in range(c_idx)], axis=0
        )

        # Avoid division by zero if all points are at the centroids
        sum_dists = np.sum(dists_sq)
        if sum_dists == 0:
            probs = np.ones(n) / n
        else:
            probs = dists_sq / sum_dists

        idx = rng.choice(n, p=probs)
        centroids[c_idx] = z_data[idx]

    # Lloyd's algorithm iterations
    labels = np.zeros(n, dtype=int)
    prev_wcss = np.inf

    for _ in range(max_iter):
        # Assign labels
        dists = np.array([np.sum((z_data - centroids[c]) ** 2, axis=1) for c in range(k_clusters)])
        labels = np.argmin(dists, axis=0)

        # Calculate WCSS
        wcss = 0.0
        for c in range(k_clusters):
            mask = labels == c
            if np.any(mask):
                wcss += np.sum((z_data[mask] - centroids[c]) ** 2)

        if abs(prev_wcss - wcss) < tol:
            break
        prev_wcss = wcss

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for c in range(k_clusters):
            mask = labels == c
            if np.any(mask):
                new_centroids[c] = np.mean(z_data[mask], axis=0)
            else:
                # Re-initialize empty cluster with a random point
                new_centroids[c] = z_data[rng.choice(n)]
        centroids = new_centroids

    return labels, float(wcss)


def calculate_linear_directional_mean(
    start_x: np.ndarray, start_y: np.ndarray, end_x: np.ndarray, end_y: np.ndarray
) -> tuple[float, float, float, float]:
    """Calculates Linear Directional Mean for line features.

    Returns:
        A tuple of (center_x, center_y, mean_angle_degrees, mean_length)
    """
    n = len(start_x)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Line midpoints
    mid_x = (start_x + end_x) / 2.0
    mid_y = (start_y + end_y) / 2.0

    center_x = float(np.mean(mid_x))
    center_y = float(np.mean(mid_y))

    # Line lengths
    dx = end_x - start_x
    dy = end_y - start_y
    lengths = np.sqrt(dx**2 + dy**2)
    lengths_safe = np.maximum(lengths, 1e-12)

    mean_length = float(np.mean(lengths))

    # Orientation angles (radians, compass-style: 0=North, clockwise)
    # atan2(dx, dy) gives compass bearing
    angles = np.arctan2(dx, dy)

    # Circular mean weighted by length
    sin_sum = np.sum(lengths_safe * np.sin(angles))
    cos_sum = np.sum(lengths_safe * np.cos(angles))

    mean_angle_rad = math.atan2(sin_sum, cos_sum)
    mean_angle_deg = math.degrees(mean_angle_rad) % 360.0

    return center_x, center_y, mean_angle_deg, mean_length


def _compute_moran_i_fast(z: np.ndarray, W: np.ndarray, S0: float, n: int) -> float:
    """Fast internal Moran's I calculator using pre-built weight matrix."""
    sum_z2 = np.sum(z**2)
    if sum_z2 == 0:
        return 0.0
    numerator = float(z @ W @ z)
    return (n / S0) * (numerator / sum_z2)


def run_sensitivity_simulation(
    values: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    n_simulations: int = 999,
    seed: int = 42,
) -> dict:
    """Monte Carlo simulation for Global Moran's I sensitivity assessment.

    Returns:
        A dictionary with observed_i, simulated_mean, simulated_std,
        empirical_p, percentile_5, percentile_95, and simulated_values.
    """
    n = len(id_order)
    if n < 4:
        raise ValueError("Sensitivity simulation requires at least 4 features.")

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    # Build dense weight matrix
    W = np.zeros((n, n))
    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        for j_fid, w in zip(neighs, w_list):
            if j_fid in id_to_idx:
                j = id_to_idx[j_fid]
                W[i, j] = w
    np.fill_diagonal(W, 0.0)
    S0 = float(np.sum(W))

    if S0 == 0:
        raise ValueError("No spatial neighbors found. Cannot run simulation.")

    # Observed Moran's I
    y = np.array([float(values[i]) for i in range(n)])
    z_obs = y - np.mean(y)
    observed_i = _compute_moran_i_fast(z_obs, W, S0, n)

    # Monte Carlo permutations
    rng = np.random.default_rng(seed)
    sim_values = np.zeros(n_simulations)

    for s in range(n_simulations):
        y_perm = rng.permutation(y)
        z_perm = y_perm - np.mean(y_perm)
        sim_values[s] = _compute_moran_i_fast(z_perm, W, S0, n)

    # Empirical p-value (two-tailed)
    count_extreme = np.sum(np.abs(sim_values) >= abs(observed_i))
    empirical_p = float((count_extreme + 1) / (n_simulations + 1))

    return {
        "observed_i": float(observed_i),
        "simulated_mean": float(np.mean(sim_values)),
        "simulated_std": float(np.std(sim_values)),
        "empirical_p": empirical_p,
        "percentile_5": float(np.percentile(sim_values, 5)),
        "percentile_95": float(np.percentile(sim_values, 95)),
        "simulated_values": sim_values.tolist(),
    }


def calculate_incremental_autocorrelation(
    x: np.ndarray,
    y_coords: np.ndarray,
    values: np.ndarray,
    start_dist: float,
    dist_increment: float,
    n_increments: int,
) -> list[dict]:
    """Calculates Global Moran's I at multiple distance bands.

    Returns:
        A list of dicts, each with keys: distance, morans_i, expected_i, z_score, p_value.
    """
    n = len(x)
    if n < 4:
        raise ValueError("Incremental autocorrelation requires at least 4 features.")

    coords = np.column_stack((x, y_coords))
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    y_mean = np.mean(values)
    z = values - y_mean
    sum_z2 = np.sum(z**2)

    if sum_z2 == 0:
        return [
            {
                "distance": start_dist + i * dist_increment,
                "morans_i": 0.0,
                "expected_i": -1.0 / (n - 1),
                "z_score": 0.0,
                "p_value": 1.0,
            }
            for i in range(n_increments)
        ]

    results = []
    for inc in range(n_increments):
        threshold = start_dist + inc * dist_increment

        W = (dists_matrix <= threshold).astype(float)
        np.fill_diagonal(W, 0.0)

        S0 = np.sum(W)
        neighbor_counts = np.sum(W > 0, axis=1)
        min_neighbors = int(np.min(neighbor_counts))
        median_neighbors = float(np.median(neighbor_counts))
        max_neighbors = int(np.max(neighbor_counts))
        isolated_count = int(np.sum(neighbor_counts == 0))
        if S0 == 0:
            results.append(
                {
                    "distance": threshold,
                    "morans_i": 0.0,
                    "expected_i": -1.0 / (n - 1),
                    "z_score": 0.0,
                    "p_value": 1.0,
                    "min_neighbors": min_neighbors,
                    "median_neighbors": median_neighbors,
                    "max_neighbors": max_neighbors,
                    "isolated_count": isolated_count,
                }
            )
            continue

        numerator = float(z @ W @ z)
        morans_i = (n / S0) * (numerator / sum_z2)
        expected_i = -1.0 / (n - 1)

        # Variance under randomization (simplified)
        S1 = float(np.sum((W + W.T) ** 2)) / 2.0
        row_sums = np.sum(W, axis=1)
        col_sums = np.sum(W, axis=0)
        S2 = float(np.sum((row_sums + col_sums) ** 2))

        sum_z4 = np.sum(z**4)
        D = (n * sum_z4) / (sum_z2**2)

        num_var = n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * S0**2) - D * (
            (n**2 - n) * S1 - 2 * n * S2 + 6 * S0**2
        )
        den_var = (n - 1) * (n - 2) * (n - 3) * S0**2

        variance = num_var / den_var - (expected_i**2) if den_var > 0 else 0.0

        if variance > 0:
            z_score = (morans_i - expected_i) / math.sqrt(variance)
            p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
        else:
            z_score = 0.0
            p_value = 1.0

        results.append(
            {
                "distance": threshold,
                "morans_i": float(morans_i),
                "expected_i": float(expected_i),
                "z_score": float(z_score),
                "p_value": float(p_value),
                "min_neighbors": min_neighbors,
                "median_neighbors": median_neighbors,
                "max_neighbors": max_neighbors,
                "isolated_count": isolated_count,
            }
        )

    return results


def calculate_ripleys_k(
    x: np.ndarray,
    y_coords: np.ndarray,
    start_dist: float,
    dist_increment: float,
    n_increments: int,
    study_area: Optional[float] = None,
) -> list[dict]:
    """Calculates Ripley's K, expected K, and L(d)-d across distance bands."""
    n = len(x)
    if n < 3:
        raise ValueError("Ripley's K requires at least 3 features.")
    coords = np.column_stack((x, y_coords))
    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    if study_area is None or study_area <= 0:
        width = float(np.max(x) - np.min(x))
        height = float(np.max(y_coords) - np.min(y_coords))
        study_area = max(width * height, 1e-12)

    results = []
    for inc in range(n_increments):
        distance = start_dist + inc * dist_increment
        within = (dists_matrix <= distance).astype(float)
        np.fill_diagonal(within, 0.0)
        observed_pairs = float(np.sum(within))
        observed_k = (study_area / (n * (n - 1))) * observed_pairs
        expected_k = math.pi * (distance**2)
        l_value = math.sqrt(max(observed_k, 0.0) / math.pi) if observed_k >= 0 else 0.0
        l_minus_d = l_value - distance
        neighbor_counts = np.sum(within > 0, axis=1)
        results.append(
            {
                "distance": float(distance),
                "observed_k": float(observed_k),
                "expected_k": float(expected_k),
                "l_value": float(l_value),
                "l_minus_d": float(l_minus_d),
                "observed_pairs": int(observed_pairs),
                "min_neighbors": int(np.min(neighbor_counts)),
                "median_neighbors": float(np.median(neighbor_counts)),
                "max_neighbors": int(np.max(neighbor_counts)),
                "isolated_count": int(np.sum(neighbor_counts == 0)),
                "study_area": float(study_area),
            }
        )
    return results


def calculate_exploratory_regression(
    y: np.ndarray, X_data: np.ndarray, x_names: list[str], max_vars: Optional[int] = None
) -> list[dict]:
    """Tests all possible OLS variable combinations and returns ranked models.

    Returns:
        A sorted list of model dicts with keys: variables, r2, adj_r2, aic, coefficients.
    """
    from itertools import combinations

    n, total_p = X_data.shape
    if max_vars is None:
        max_vars = min(total_p, 5)  # cap at 5 to keep runtime manageable
    max_vars = min(max_vars, total_p)

    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)
    if ss_tot == 0:
        return []

    models = []

    for k in range(1, max_vars + 1):
        for combo in combinations(range(total_p), k):
            X_sub = X_data[:, combo]
            X_design = np.column_stack((np.ones(n), X_sub))

            try:
                xtx_inv = np.linalg.pinv(X_design.T @ X_design)
                beta = xtx_inv @ X_design.T @ y
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue

            y_pred = X_design @ beta
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)

            p = len(combo)
            df = n - p - 1
            if df <= 0:
                continue

            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            adj_r2 = 1.0 - ((1.0 - r2) * (n - 1) / df) if df > 0 else 0.0

            # AICc
            if ss_res > 0 and n > 0:
                log_lik = -n / 2.0 * (np.log(2.0 * np.pi * ss_res / n) + 1.0)
                k_params = p + 2  # intercept + vars + variance
                aic = -2.0 * log_lik + 2.0 * k_params
                if n - k_params - 1 > 0:
                    aicc = aic + (2.0 * k_params * (k_params + 1)) / (n - k_params - 1)
                else:
                    aicc = float("inf")
            else:
                aicc = float("inf")

            var_names = [x_names[i] for i in combo]
            coef_dict = {"Intercept": float(beta[0])}
            for idx, vi in enumerate(combo):
                coef_dict[x_names[vi]] = float(beta[idx + 1])

            models.append(
                {
                    "variables": var_names,
                    "r2": float(r2),
                    "adj_r2": float(adj_r2),
                    "aicc": float(aicc),
                    "coefficients": coef_dict,
                    "n_vars": p,
                }
            )

    # Sort by AICc ascending (best first)
    models.sort(key=lambda m: cast(float, m["aicc"]))
    return models


def calculate_glr(
    y: np.ndarray,
    X_data: np.ndarray,
    family: str = "gaussian",
    max_iter: int = 100,
    tol: float = 1e-6,
) -> dict:
    """Fits Gaussian, logistic, or Poisson generalized linear regression."""
    n = len(y)
    p = X_data.shape[1]
    if n <= p + 1:
        raise ValueError("GLR requires more observations than model parameters.")
    X = np.column_stack((np.ones(n), X_data))
    family = family.lower()

    if family == "gaussian":
        xtx_inv = np.linalg.pinv(X.T @ X)
        beta = xtx_inv @ X.T @ y
        mu = X @ beta
        residuals = y - mu
        rss = float(np.sum(residuals**2))
        df = n - p - 1
        sigma2 = rss / df if df > 0 else 0.0
        cov = sigma2 * xtx_inv
        se = np.sqrt(np.maximum(0.0, np.diagonal(cov)))
        z_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_values = 2.0 * (
            1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z_stats) / math.sqrt(2.0)))
        )
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - (rss / tss) if tss > 0 else 0.0
        log_likelihood = -n / 2.0 * (math.log(2.0 * math.pi * rss / n) + 1.0) if rss > 0 else 0.0
        aic = -2.0 * log_likelihood + 2.0 * (p + 2)
        return {
            "family": "gaussian",
            "coefficients": beta,
            "std_errors": se,
            "z_statistics": z_stats,
            "p_values": p_values,
            "fitted": mu,
            "residuals": residuals,
            "log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "r2": float(r2),
            "iterations": 1,
            "converged": True,
        }

    if family == "logistic":
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("Logistic GLR requires a binary dependent variable coded as 0 and 1.")
        beta = np.zeros(p + 1)
        converged = False
        # `iteration` is intentionally reported after the loop (see "iterations" below).
        for iteration in range(1, max_iter + 1):  # noqa: B007
            eta = X @ beta
            mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
            w = np.maximum(mu * (1.0 - mu), 1e-9)
            z = eta + (y - mu) / w
            xtw = X.T * w
            new_beta = np.linalg.pinv(xtw @ X) @ xtw @ z
            if np.max(np.abs(new_beta - beta)) < tol:
                beta = new_beta
                converged = True
                break
            beta = new_beta
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
        w = np.maximum(mu * (1.0 - mu), 1e-9)
        cov = np.linalg.pinv((X.T * w) @ X)
        se = np.sqrt(np.maximum(0.0, np.diagonal(cov)))
        z_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_values = 2.0 * (
            1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z_stats) / math.sqrt(2.0)))
        )
        eps = 1e-12
        log_likelihood = float(np.sum(y * np.log(mu + eps) + (1.0 - y) * np.log(1.0 - mu + eps)))
        aic = -2.0 * log_likelihood + 2.0 * (p + 1)
        residuals = y - mu
        return {
            "family": "logistic",
            "coefficients": beta,
            "std_errors": se,
            "z_statistics": z_stats,
            "p_values": p_values,
            "fitted": mu,
            "residuals": residuals,
            "log_likelihood": log_likelihood,
            "aic": float(aic),
            "r2": None,
            "iterations": iteration,
            "converged": converged,
        }

    if family == "poisson":
        if np.any(y < 0) or np.any(np.floor(y) != y):
            raise ValueError("Poisson GLR requires non-negative integer count values.")
        beta = np.zeros(p + 1)
        converged = False
        # `iteration` is intentionally reported after the loop (see "iterations" below).
        for iteration in range(1, max_iter + 1):  # noqa: B007
            eta = np.clip(X @ beta, -30.0, 30.0)
            mu = np.maximum(np.exp(eta), 1e-9)
            z = eta + (y - mu) / mu
            xtw = X.T * mu
            new_beta = np.linalg.pinv(xtw @ X) @ xtw @ z
            if np.max(np.abs(new_beta - beta)) < tol:
                beta = new_beta
                converged = True
                break
            beta = new_beta
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = np.maximum(np.exp(eta), 1e-9)
        cov = np.linalg.pinv((X.T * mu) @ X)
        se = np.sqrt(np.maximum(0.0, np.diagonal(cov)))
        z_stats = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
        p_values = 2.0 * (
            1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z_stats) / math.sqrt(2.0)))
        )
        log_factorial = np.array([math.lgamma(float(value) + 1.0) for value in y], dtype=float)
        log_likelihood = float(np.sum(y * np.log(mu) - mu - log_factorial))
        aic = -2.0 * log_likelihood + 2.0 * (p + 1)
        residuals = y - mu
        return {
            "family": "poisson",
            "coefficients": beta,
            "std_errors": se,
            "z_statistics": z_stats,
            "p_values": p_values,
            "fitted": mu,
            "residuals": residuals,
            "log_likelihood": log_likelihood,
            "aic": float(aic),
            "r2": None,
            "iterations": iteration,
            "converged": converged,
        }

    raise ValueError("Unsupported GLR family. Use gaussian, logistic, or poisson.")


def calculate_spatial_lag(
    y: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    row_standardize: bool = True,
) -> np.ndarray:
    """Calculates spatial lag (W * y) for a target attribute array.

    Args:
        y: 1D NumPy array of target values.
        neighbors: Dict mapping feature ID to list of neighbor feature IDs.
        weights: Dict mapping feature ID to list of numeric weights.
        id_order: List of feature IDs defining the index mapping of y.
        row_standardize: If True, weights for each row are scaled to sum to 1.0.

    Returns:
        1D NumPy array of spatial lag values.
    """
    n = len(y)
    lag = np.zeros(n, dtype=np.float64)
    if n == 0:
        return lag

    id_to_idx = {fid: idx for idx, fid in enumerate(id_order)}

    for i, fid in enumerate(id_order):
        neighs = neighbors.get(fid, [])
        w_list = weights.get(fid, [])
        valid_indices = []
        valid_w = []
        for j, nid in enumerate(neighs):
            if nid in id_to_idx:
                valid_indices.append(id_to_idx[nid])
                w_val = w_list[j] if j < len(w_list) else 1.0
                valid_w.append(w_val)

        if not valid_indices:
            continue

        w_arr = np.array(valid_w, dtype=np.float64)
        if row_standardize:
            w_sum = np.sum(w_arr)
            if w_sum > 0:
                w_arr = w_arr / w_sum

        lag[i] = float(np.sum(w_arr * y[valid_indices]))

    return lag


def calculate_bivariate_moran(
    x_values: np.ndarray,
    y_values: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
) -> tuple[float, float, float, float, float]:
    """Calculates Bivariate Global Moran's I spatial autocorrelation.

    Measures spatial correlation between variable X at locations i and variable Y at neighbors j.

    Args:
        x_values: 1D array of variable X values.
        y_values: 1D array of variable Y values.
        neighbors: Dict mapping feature ID to neighbor IDs.
        weights: Dict mapping feature ID to neighbor weights.
        id_order: List of feature IDs.

    Returns:
        A tuple of (bivariate_moran_i, expected_i, variance, z_score, p_value)
    """
    n = len(x_values)
    if n <= 3:
        raise ValueError("Bivariate Moran's I requires at least 4 observations.")
    if len(y_values) != n:
        raise ValueError("x_values and y_values must have the same length.")

    x_std = np.std(x_values)
    y_std = np.std(y_values)

    if x_std == 0 or y_std == 0:
        return 0.0, -1.0 / (n - 1), 0.0, 0.0, 1.0

    zx = (x_values - np.mean(x_values)) / x_std
    zy = (y_values - np.mean(y_values)) / y_std

    lag_zy = calculate_spatial_lag(zy, neighbors, weights, id_order, row_standardize=True)

    bivariate_i = float(np.sum(zx * lag_zy) / n)
    expected_i = -1.0 / (n - 1)
    var_i = (1.0 / n) - (expected_i**2)
    if var_i < 0:
        var_i = 1e-9
    se_i = math.sqrt(var_i)
    z_score = (bivariate_i - expected_i) / se_i if se_i > 0 else 0.0
    p_val = 1.0 - math.erf(abs(z_score) / math.sqrt(2.0))

    return bivariate_i, expected_i, var_i, z_score, p_val


def calculate_local_bivariate_moran(
    x_values: np.ndarray,
    y_values: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Calculates Anselin Bivariate Local Moran's I cluster and outlier diagnostics.

    Args:
        x_values: 1D array of variable X values.
        y_values: 1D array of variable Y values.
        neighbors: Dict mapping feature ID to neighbor IDs.
        weights: Dict mapping feature ID to neighbor weights.
        id_order: List of feature IDs.
        alpha: Significance threshold for cluster quadrant assignment.

    Returns:
        A tuple of:
          - I_values: NumPy array of Bivariate Local Moran's I values.
          - z_scores: NumPy array of z-scores.
          - p_values: NumPy array of p-values.
          - quadrants: List of strings ('HH', 'LL', 'HL', 'LH', 'Not Significant').
    """
    n = len(x_values)
    I_values = np.zeros(n)
    z_scores = np.zeros(n)
    p_values = np.ones(n)
    quadrants = ["Not Significant"] * n

    if n <= 2:
        return I_values, z_scores, p_values, quadrants

    x_std = np.std(x_values)
    y_std = np.std(y_values)

    if x_std == 0 or y_std == 0:
        return I_values, z_scores, p_values, quadrants

    zx = (x_values - np.mean(x_values)) / x_std
    zy = (y_values - np.mean(y_values)) / y_std

    lag_zy = calculate_spatial_lag(zy, neighbors, weights, id_order, row_standardize=True)

    I_values = zx * lag_zy

    var_loc = 1.0 / (n - 1) if n > 1 else 1.0

    for i in range(n):
        z = I_values[i] / math.sqrt(var_loc) if var_loc > 0 else 0.0
        p = 1.0 - math.erf(abs(z) / math.sqrt(2.0))
        z_scores[i] = z
        p_values[i] = p

        if p <= alpha:
            if zx[i] > 0 and lag_zy[i] > 0:
                quadrants[i] = "HH"
            elif zx[i] < 0 and lag_zy[i] < 0:
                quadrants[i] = "LL"
            elif zx[i] > 0 and lag_zy[i] < 0:
                quadrants[i] = "HL"
            elif zx[i] < 0 and lag_zy[i] > 0:
                quadrants[i] = "LH"

    return I_values, z_scores, p_values, quadrants


def fit_spatial_lag_model(
    y: np.ndarray,
    X: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    method: str = "2sls",
) -> dict:
    """Fits a Spatial Lag Model (SLM / SAR): y = rho * W * y + X * beta + e.

    Args:
        y: 1D array of dependent variable values of shape (N,).
        X: 2D array of independent variables of shape (N, K).
        neighbors: Dict mapping feature ID to list of neighbor feature IDs.
        weights: Dict mapping feature ID to list of numeric weights.
        id_order: List of feature IDs.
        method: Estimation method ("2sls" for Two-Stage Least Squares).

    Returns:
        Dict containing model parameters:
          - rho: Estimated spatial autoregressive coefficient float.
          - beta: Estimated covariate coefficient array of shape (K,).
          - std_errors: Standard errors for [rho, beta] of shape (K + 1,).
          - z_statistics: Z-statistics for [rho, beta].
          - p_values: P-values for [rho, beta].
          - r2: Pseudo R-squared score.
          - log_likelihood: Model log-likelihood float.
          - aic: Akaike Information Criterion float.
          - fitted: Fitted values array of shape (N,).
          - residuals: Residual values array of shape (N,).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n, k = X_arr.shape
    if len(y_arr) != n:
        raise ValueError("Length of y must match number of rows in X.")
    if n <= k + 1:
        raise ValueError("Number of observations must be greater than number of predictors + 1.")

    wy = calculate_spatial_lag(y_arr, neighbors, weights, id_order, row_standardize=True)

    WX = np.zeros_like(X_arr)
    for col in range(k):
        WX[:, col] = calculate_spatial_lag(
            X_arr[:, col], neighbors, weights, id_order, row_standardize=True
        )

    Z = np.hstack([X_arr, WX])
    X_lag = np.hstack([wy.reshape(-1, 1), X_arr])

    ztz_pinv = np.linalg.pinv(Z.T @ Z)
    P_z = Z @ ztz_pinv @ Z.T
    X_lag_hat = P_z @ X_lag

    params = np.linalg.pinv(X_lag_hat.T @ X_lag_hat) @ (X_lag_hat.T @ y_arr)

    rho = float(params[0])
    beta = params[1:]

    fitted = rho * wy + X_arr @ beta
    residuals = y_arr - fitted
    sse = float(np.sum(residuals**2))
    df_e = max(1, n - k - 1)
    s2 = sse / df_e

    cov_matrix = s2 * np.linalg.pinv(X_lag.T @ P_z @ X_lag)
    std_errors = np.sqrt(np.maximum(0.0, np.diagonal(cov_matrix)))
    z_stats = np.divide(params, std_errors, out=np.zeros_like(params), where=std_errors > 0)
    p_vals = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z_stats) / math.sqrt(2.0))))

    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0

    sigma2_ml = sse / n
    log_likelihood = float(
        -0.5 * n * (math.log(2.0 * math.pi) + math.log(max(1e-9, sigma2_ml)) + 1.0)
    )
    aic = float(-2.0 * log_likelihood + 2.0 * (k + 2))

    return {
        "rho": rho,
        "beta": beta,
        "std_errors": std_errors,
        "z_statistics": z_stats,
        "p_values": p_vals,
        "r2": r2,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "fitted": fitted,
        "residuals": residuals,
    }


def fit_spatial_error_model(
    y: np.ndarray,
    X: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
) -> dict:
    """Fits a Spatial Error Model (SEM): y = X * beta + u, u = lambda * W * u + e.

    Args:
        y: 1D array of dependent variable values of shape (N,).
        X: 2D array of independent variables of shape (N, K).
        neighbors: Dict mapping feature ID to list of neighbor feature IDs.
        weights: Dict mapping feature ID to list of numeric weights.
        id_order: List of feature IDs.

    Returns:
        Dict containing model parameters:
          - lambda_param: Estimated spatial error coefficient float.
          - beta: Estimated covariate coefficient array of shape (K,).
          - std_errors: Standard errors for beta of shape (K,).
          - z_statistics: Z-statistics for beta.
          - p_values: P-values for beta.
          - r2: Pseudo R-squared score.
          - log_likelihood: Model log-likelihood float.
          - aic: Akaike Information Criterion float.
          - fitted: Fitted values array of shape (N,).
          - residuals: Residual values array of shape (N,).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n, k = X_arr.shape
    if len(y_arr) != n:
        raise ValueError("Length of y must match number of rows in X.")
    if n <= k:
        raise ValueError("Number of observations must be greater than number of predictors.")

    beta_ols = np.linalg.pinv(X_arr.T @ X_arr) @ (X_arr.T @ y_arr)
    e_ols = y_arr - X_arr @ beta_ols

    we_ols = calculate_spatial_lag(e_ols, neighbors, weights, id_order, row_standardize=True)
    we_sq = float(np.sum(we_ols**2))
    lambda_param = float(np.sum(e_ols * we_ols) / we_sq) if we_sq > 0 else 0.0
    lambda_param = max(-0.99, min(0.99, lambda_param))

    wy = calculate_spatial_lag(y_arr, neighbors, weights, id_order, row_standardize=True)
    y_trans = y_arr - lambda_param * wy

    X_trans = np.zeros_like(X_arr)
    for col in range(k):
        wx_col = calculate_spatial_lag(
            X_arr[:, col], neighbors, weights, id_order, row_standardize=True
        )
        X_trans[:, col] = X_arr[:, col] - lambda_param * wx_col

    cov_x_trans = np.linalg.pinv(X_trans.T @ X_trans)
    beta = cov_x_trans @ (X_trans.T @ y_trans)

    fitted = X_arr @ beta
    residuals = y_arr - fitted
    sse = float(np.sum(residuals**2))
    df_e = max(1, n - k)
    s2 = sse / df_e

    cov_matrix = s2 * cov_x_trans
    std_errors = np.sqrt(np.maximum(0.0, np.diagonal(cov_matrix)))
    z_stats = np.divide(beta, std_errors, out=np.zeros_like(beta), where=std_errors > 0)
    p_vals = 2.0 * (1.0 - 0.5 * (1.0 + np.vectorize(math.erf)(np.abs(z_stats) / math.sqrt(2.0))))

    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0

    sigma2_ml = sse / n
    log_likelihood = float(
        -0.5 * n * (math.log(2.0 * math.pi) + math.log(max(1e-9, sigma2_ml)) + 1.0)
    )
    aic = float(-2.0 * log_likelihood + 2.0 * (k + 1))

    return {
        "lambda_param": lambda_param,
        "beta": beta,
        "std_errors": std_errors,
        "z_statistics": z_stats,
        "p_values": p_vals,
        "r2": r2,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "fitted": fitted,
        "residuals": residuals,
    }


def calculate_gwlr(
    y: np.ndarray,
    X_data: np.ndarray,
    coords: np.ndarray,
    bandwidth: float,
    kernel_type: str = "fixed_gaussian",
    max_iter: int = 50,
    tol: float = 1e-5,
) -> dict:
    """Performs Geographically Weighted Logistic Regression (GWLR) analysis for binary outcomes.

    Args:
        y: Binary dependent variable (n,) with values in {0, 1}.
        X_data: Independent variables (n, p).
        coords: Centroid coordinates (n, 2).
        bandwidth: Kernel bandwidth distance float.
        kernel_type: "fixed_gaussian" or "fixed_bisquare".
        max_iter: Maximum IRLS iterations per location.
        tol: Convergence tolerance.

    Returns:
        A dictionary containing GWLR coefficients, standard errors, probabilities, and statistics.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    X_arr = np.asarray(X_data, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n, p = X_arr.shape
    if len(y_arr) != n:
        raise ValueError("Length of y must match number of rows in X_data.")
    if not np.all(np.isin(y_arr, [0.0, 1.0])):
        raise ValueError("GWLR requires binary dependent variable with values in {0, 1}.")

    X = np.column_stack((np.ones(n), X_arr))
    k = p + 1

    local_beta = np.zeros((n, k), dtype=np.float64)
    local_se = np.zeros((n, k), dtype=np.float64)
    y_prob = np.zeros(n, dtype=np.float64)

    dists_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    for i in range(n):
        dists = dists_matrix[i]
        if kernel_type == "fixed_bisquare":
            w = np.where(dists <= bandwidth, (1.0 - (dists / max(1e-9, bandwidth)) ** 2) ** 2, 0.0)
        else:
            w = np.exp(-0.5 * (dists / max(1e-9, bandwidth)) ** 2)

        beta = np.zeros(k, dtype=np.float64)
        for _ in range(max_iter):
            eta = np.clip(X @ beta, -30.0, 30.0)
            pi = np.clip(1.0 / (1.0 + np.exp(-eta)), 1e-9, 1.0 - 1e-9)
            v = w * pi * (1.0 - pi)
            z = eta + (y_arr - pi) / (pi * (1.0 - pi))

            xtv = X.T * v
            cov = np.linalg.pinv(xtv @ X)
            beta_new = cov @ xtv @ z

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        local_beta[i] = beta
        eta_i = float(np.clip(X[i] @ beta, -30.0, 30.0))
        y_prob[i] = 1.0 / (1.0 + math.exp(-eta_i))

        pi_loc = np.clip(1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30.0, 30.0))), 1e-9, 1.0 - 1e-9)
        v_loc = w * pi_loc * (1.0 - pi_loc)
        cov_loc = np.linalg.pinv((X.T * v_loc) @ X)
        local_se[i] = np.sqrt(np.maximum(0.0, np.diagonal(cov_loc)))

    residuals = y_arr - y_prob
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0

    return {
        "coefficients": local_beta,
        "std_errors": local_se,
        "probabilities": y_prob,
        "residuals": residuals,
        "pseudo_r2": r2,
    }


def fit_spatial_tobit_model(
    y: np.ndarray,
    X_data: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
) -> dict:
    """Fits a Spatial Autoregressive Tobit (SAR-Tobit) model for zero-censored dependent variables.

    Args:
        y: Dependent variable (n,) with zero-censoring (y >= 0).
        X_data: Independent variables matrix (n, p).
        neighbors: Spatial adjacency dictionary.
        weights: Spatial weights dictionary.
        id_order: List of observation IDs.

    Returns:
        Dict containing model parameters (rho, beta, fitted, residuals, pseudo_r2).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    X_arr = np.asarray(X_data, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    n, p = X_arr.shape
    if len(y_arr) != n:
        raise ValueError("Length of y must match number of rows in X_data.")

    wy = calculate_spatial_lag(y_arr, neighbors, weights, id_order, row_standardize=True)
    wX = calculate_spatial_lag(X_arr, neighbors, weights, id_order, row_standardize=True)

    Z = np.column_stack((X_arr, wX))
    X_spatial = np.column_stack((wy, X_arr))

    ztz_inv = np.linalg.pinv(Z.T @ Z)
    P_Z = Z @ ztz_inv @ Z.T
    X_hat = P_Z @ X_spatial

    uncensored = y_arr > 0.0
    if np.sum(uncensored) < (p + 1):
        raise ValueError("Too few non-zero uncensored observations for Tobit estimation.")

    beta_full = np.linalg.pinv(X_hat.T @ X_hat) @ (X_hat.T @ y_arr)
    rho = float(beta_full[0])
    beta = beta_full[1:]

    fitted = np.maximum(0.0, X_spatial @ beta_full)
    residuals = y_arr - fitted

    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0

    return {
        "rho": rho,
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
        "pseudo_r2": r2,
    }


def calculate_local_moran_fdr(
    x: np.ndarray,
    neighbors: dict[int, list[int]],
    weights: dict[int, list[float]],
    id_order: list[int],
    alpha: float = 0.05,
) -> dict:
    """Calculates Local Moran's I with Benjamini-Hochberg False Discovery Rate (FDR) control.

    Args:
        x: Feature vector array (n,).
        neighbors: Spatial adjacency dictionary.
        weights: Spatial weights dictionary.
        id_order: List of observation IDs.
        alpha: False Discovery Rate significance threshold float.

    Returns:
        Dict containing I_local, raw_p_values, fdr_adjusted_p_values, and significant_mask.
    """
    I_local, z_scores, p_vals, quads = calculate_local_moran(x, neighbors, weights, id_order)
    n = len(p_vals)

    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]

    fdr_p = np.zeros(n, dtype=np.float64)
    cum_min = 1.0

    for i in range(n - 1, -1, -1):
        q_val = sorted_p[i] * float(n) / float(i + 1)
        cum_min = min(cum_min, q_val)
        fdr_p[sorted_indices[i]] = max(0.0, min(1.0, cum_min))

    sig_mask = fdr_p <= alpha

    return {
        "local_moran": I_local,
        "raw_p_values": p_vals,
        "fdr_p_values": fdr_p,
        "significant_mask": sig_mask,
        "quadrants": quads,
    }


def calculate_gwss(
    data: np.ndarray,
    coords: np.ndarray,
    bandwidth: float,
    kernel_type: str = "fixed_gaussian",
) -> dict:
    """Calculates Geographically Weighted Summary Statistics (GWSS).

    Args:
        data: Feature data matrix (n, p) or vector (n,).
        coords: Centroid coordinates array (n, 2).
        bandwidth: Kernel bandwidth distance float.
        kernel_type: "fixed_gaussian" or "fixed_bisquare".

    Returns:
        Dict containing 2D NumPy arrays of shape (n, p):
          - local_mean: Local weighted mean.
          - local_std: Local weighted standard deviation.
          - local_skewness: Local weighted skewness.
    """
    X = np.asarray(data, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    pts = np.asarray(coords, dtype=np.float64)
    n, p = X.shape

    if len(pts) != n:
        raise ValueError("Length of coords must match number of rows in data.")

    local_mean = np.zeros((n, p), dtype=np.float64)
    local_std = np.zeros((n, p), dtype=np.float64)
    local_skew = np.zeros((n, p), dtype=np.float64)

    dists_matrix = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))

    for i in range(n):
        dists = dists_matrix[i]
        if kernel_type == "fixed_bisquare":
            w = np.where(dists <= bandwidth, (1.0 - (dists / max(1e-9, bandwidth)) ** 2) ** 2, 0.0)
        else:
            w = np.exp(-0.5 * (dists / max(1e-9, bandwidth)) ** 2)

        w_sum = float(np.sum(w))
        if w_sum <= 1e-12:
            w = np.ones(n, dtype=np.float64) / n
            w_sum = 1.0

        w_norm = w / w_sum

        for j in range(p):
            col = X[:, j]
            mu = float(np.sum(col * w_norm))
            local_mean[i, j] = mu

            diff = col - mu
            var = float(np.sum((diff**2) * w_norm))
            sd = math.sqrt(max(0.0, var))
            local_std[i, j] = sd

            if sd > 1e-9:
                m3 = float(np.sum((diff**3) * w_norm))
                local_skew[i, j] = m3 / (sd**3)
            else:
                local_skew[i, j] = 0.0

    return {
        "local_mean": local_mean,
        "local_std": local_std,
        "local_skewness": local_skew,
    }


def calculate_weighted_kde(
    event_coords: np.ndarray,
    event_weights: np.ndarray,
    grid_coords: np.ndarray,
    bandwidth: float,
    kernel_type: str = "quartic",
) -> np.ndarray:
    """Calculates 2D Kernel Density Estimation (KDE) weighted by event magnitudes.

    Args:
        event_coords: (E, 2) NumPy array of event point coordinates.
        event_weights: (E,) NumPy array of magnitude weights per event.
        grid_coords: (G, 2) NumPy array of spatial evaluation grid points.
        bandwidth: Spatial search radius bandwidth float.
        kernel_type: "quartic" or "gaussian".

    Returns:
        1D NumPy array of shape (G,) containing spatial density values.
    """
    pts = np.asarray(event_coords, dtype=np.float64)
    w = np.asarray(event_weights, dtype=np.float64)
    grid = np.asarray(grid_coords, dtype=np.float64)

    e_count = len(pts)
    g_count = len(grid)

    if e_count == 0 or g_count == 0:
        return np.zeros(g_count, dtype=np.float64)

    if len(w) != e_count:
        raise ValueError("event_weights length must match event_coords.")

    diffs = grid[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))

    u = dists / max(1e-9, bandwidth)

    if kernel_type == "gaussian":
        k_vals = (1.0 / (2.0 * math.pi)) * np.exp(-0.5 * (u**2))
    else:
        k_vals = np.where(u <= 1.0, (15.0 / 16.0) * ((1.0 - u**2) ** 2), 0.0)

    density = np.sum(k_vals * w[None, :], axis=1) / (bandwidth**2)
    return density


def calculate_ripleys_cross_k(
    points_a: np.ndarray,
    points_b: np.ndarray,
    radii: np.ndarray,
    area: float,
) -> np.ndarray:
    """Calculates Ripley's Cross-K function K_AB(r) for multi-type point pattern co-location.

    Args:
        points_a: (Na, 2) NumPy array of type A point coordinates.
        points_b: (Nb, 2) NumPy array of type B point coordinates.
        radii: 1D array of distance evaluation radii r.
        area: Total bounding study area float.

    Returns:
        1D NumPy array of shape len(radii) containing Ripley's Cross-K values.
    """
    pts_a = np.asarray(points_a, dtype=np.float64)
    pts_b = np.asarray(points_b, dtype=np.float64)
    r_arr = np.asarray(radii, dtype=np.float64)

    na = len(pts_a)
    nb = len(pts_b)

    if na == 0 or nb == 0 or area <= 0:
        return np.zeros(len(r_arr), dtype=np.float64)

    diffs = pts_a[:, None, :] - pts_b[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))

    k_results = np.zeros(len(r_arr), dtype=np.float64)
    for idx, r in enumerate(r_arr):
        count = float(np.sum(dists <= r))
        k_results[idx] = (area / (na * nb)) * count

    return k_results


def skater_spatial_clustering(
    attr_matrix: np.ndarray,
    neighbors: dict | np.ndarray,
    weights_or_adj: dict | np.ndarray | None = None,
    n_clusters: int = 2,
) -> np.ndarray:
    """Spatially Constrained Regionalization using Minimum Spanning Tree (SKATER algorithm).

    Args:
        attr_matrix: 2D NumPy array of shape (N, P) containing spatial unit attribute features.
        neighbors: Neighbors dict {node: [nbrs]} OR CSR indptr 1D array.
        weights_or_adj: Weights dict OR CSR adj 1D array.
        n_clusters: Target number of contiguous spatial regions int.

    Returns:
        1D NumPy array of shape (N,) containing cluster assignment labels.
    """
    X = np.asarray(attr_matrix, dtype=np.float64)

    edges = []
    if isinstance(neighbors, dict):
        n = len(X)
        for u, nbs in neighbors.items():
            for v in nbs:
                if u < v:
                    dist = float(np.linalg.norm(X[u] - X[v]))
                    edges.append((dist, int(u), int(v)))
    else:
        indptr = np.asarray(neighbors, dtype=np.int32)
        adj_arr = np.asarray(weights_or_adj, dtype=np.int32)
        n = len(indptr) - 1
        for u in range(n):
            for v in adj_arr[indptr[u] : indptr[u + 1]]:
                if u < v:
                    dist = float(np.linalg.norm(X[u] - X[v]))
                    edges.append((dist, int(u), int(v)))

    if n <= 0 or n_clusters <= 0:
        return np.zeros(n, dtype=np.int64)

    if n_clusters >= n:
        return np.arange(n, dtype=np.int64)

    edges.sort(key=lambda x: x[0])

    # Disjoint Set Union (DSU) for MST construction
    parent = list(range(n))

    def find(i: int) -> int:
        path = []
        curr = i
        while parent[curr] != curr:
            path.append(curr)
            curr = parent[curr]
        for node in path:
            parent[node] = curr
        return curr

    mst_edges = []
    for dist, u, v in edges:
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parent[root_u] = root_v
            mst_edges.append((dist, u, v))

    # 2. Prune n_clusters - 1 edges with largest attribute dissimilarity
    mst_edges.sort(key=lambda x: x[0], reverse=True)
    retained_edges = mst_edges[max(0, n_clusters - 1) :]

    # 3. Connected component labeling on retained MST edges
    graph: dict[int, list[int]] = {i: [] for i in range(n)}
    for _, u, v in retained_edges:
        graph[u].append(v)
        graph[v].append(u)

    labels = np.full(n, -1, dtype=np.int64)
    curr_label = 0

    for i in range(n):
        if labels[i] == -1:
            queue = [i]
            labels[i] = curr_label
            while queue:
                curr = queue.pop(0)
                for nxt in graph[curr]:
                    if labels[nxt] == -1:
                        labels[nxt] = curr_label
                        queue.append(nxt)
            curr_label += 1

    return labels


def fit_spatial_sarma_model(
    y: np.ndarray,
    X: np.ndarray,
    neighbors: dict,
    weights: dict,
    id_order: list,
) -> dict:
    """Fits Spatial Autoregressive Moving Average (SARMA) combining spatial lag and error.

    Args:
        y: 1D NumPy array of dependent variable values of shape (N,).
        X: 2D NumPy array of independent explanatory features of shape (N, K).
        neighbors: Adjacency dictionary mapping node_id -> list of neighbor node_ids.
        weights: Spatial weights dictionary mapping node_id -> list of weight floats.
        id_order: List of node_ids defining array row ordering.

    Returns:
        Dict containing model fit parameters:
          - rho: Float estimated spatial lag autoregressive parameter.
          - lambda_err: Float estimated spatial error autoregressive parameter.
          - beta: 1D NumPy array of estimated regression coefficients (K,).
          - fitted: 1D NumPy array of fitted values (N,).
          - residuals: 1D NumPy array of residual errors (N,).
    """
    y_arr = np.asarray(y, dtype=np.float64)
    X_mat = np.asarray(X, dtype=np.float64)
    n = len(y_arr)

    if len(X_mat) != n:
        raise ValueError("Length of y must match number of rows in X.")

    W_lag = calculate_spatial_lag(y_arr, neighbors, weights, id_order)

    # 2SLS estimation for SARMA
    X_stage1 = np.column_stack([X_mat, W_lag])
    beta_stage1, _, _, _ = np.linalg.lstsq(X_stage1, y_arr, rcond=None)

    rho = float(beta_stage1[-1])
    beta = beta_stage1[:-1]

    fitted = X_stage1 @ beta_stage1
    residuals = y_arr - fitted

    u_lag = calculate_spatial_lag(residuals, neighbors, weights, id_order)
    lambda_err = float(np.sum(residuals * u_lag) / max(1e-9, np.sum(u_lag**2)))

    return {
        "rho": rho,
        "lambda_err": lambda_err,
        "beta": beta,
        "fitted": fitted,
        "residuals": residuals,
    }


def calculate_gwpca(
    X: np.ndarray,
    coords: np.ndarray,
    bandwidth: float,
    n_components: int = 2,
    kernel_type: str = "fixed_gaussian",
) -> dict:
    """Geographically Weighted Principal Components Analysis (GWPCA).

    Computes spatially varying PCA by fitting local covariance matrices weighted
    by a spatial kernel at each observation location.

    Args:
        X: 2D NumPy array of shape (N, P) containing standardized attribute variables.
        coords: 2D NumPy array of shape (N, 2) containing spatial coordinates.
        bandwidth: Kernel bandwidth float controlling spatial smoothing.
        n_components: Number of principal components to extract (default 2).
        kernel_type: Spatial kernel type string ("fixed_gaussian" or "fixed_bisquare").

    Returns:
        Dict containing GWPCA results:
          - local_eigenvalues: 2D NumPy array of shape (N, n_components) local eigenvalues.
          - local_variance_explained: 2D array of shape (N, n_components) local % variance.
          - winning_variable: 1D array of shape (N,) index of highest-loading variable per location.
          - total_local_variance: 1D array of shape (N,) total variance at each location.
    """
    X_mat = np.asarray(X, dtype=np.float64)
    C = np.asarray(coords, dtype=np.float64)
    n, p = X_mat.shape

    if len(C) != n:
        raise ValueError("Length of coords must match number of rows in X.")

    if n_components > p:
        n_components = p

    local_eigenvalues = np.zeros((n, n_components), dtype=np.float64)
    local_var_explained = np.zeros((n, n_components), dtype=np.float64)
    winning_var = np.zeros(n, dtype=np.int64)
    total_var = np.zeros(n, dtype=np.float64)

    for i in range(n):
        diffs = C - C[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))

        if kernel_type == "fixed_bisquare":
            w = np.where(dists <= bandwidth, (1.0 - (dists / bandwidth) ** 2) ** 2, 0.0)
        else:  # fixed_gaussian
            w = np.exp(-0.5 * (dists / max(bandwidth, 1e-9)) ** 2)

        w_sum = np.sum(w)
        if w_sum < 1e-12:
            continue

        w_norm = w / w_sum
        X_centered = X_mat - np.sum(w_norm[:, None] * X_mat, axis=0)
        cov_local = (X_centered * w_norm[:, None]).T @ X_centered

        eigvals, eigvecs = np.linalg.eigh(cov_local)
        idx_sorted = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[idx_sorted]
        eigvecs_sorted = eigvecs[:, idx_sorted]

        k = min(n_components, len(eigvals_sorted))
        local_eigenvalues[i, :k] = eigvals_sorted[:k]

        total_v = max(np.sum(np.maximum(eigvals_sorted, 0.0)), 1e-12)
        total_var[i] = total_v
        local_var_explained[i, :k] = (np.maximum(eigvals_sorted[:k], 0.0) / total_v) * 100.0

        # Winning variable = variable with largest absolute loading on PC1
        if k > 0:
            winning_var[i] = int(np.argmax(np.abs(eigvecs_sorted[:, 0])))

    return {
        "local_eigenvalues": local_eigenvalues,
        "local_variance_explained": local_var_explained,
        "winning_variable": winning_var,
        "total_local_variance": total_var,
    }


def fit_spatial_durbin_model(
    y: np.ndarray,
    X: np.ndarray,
    neighbors: dict,
    weights: dict,
    id_order: list,
) -> dict:
    """Fits a Spatial Durbin Model (SDM) with spatially lagged dependent and explanatory variables.

    SDM extends the Spatial Lag Model by also including spatially lagged
    explanatory variables WX, estimated via Two-Stage Least Squares (2SLS).

    Args:
        y: 1D NumPy array of dependent variable values of shape (N,).
        X: 2D NumPy array of independent explanatory features of shape (N, K).
        neighbors: Adjacency dictionary mapping node_id -> list of neighbor node_ids.
        weights: Spatial weights dictionary mapping node_id -> list of weight floats.
        id_order: List of node_ids defining array row ordering.

    Returns:
        Dict containing model fit parameters:
          - rho: Float estimated spatial autoregressive parameter.
          - beta: 1D NumPy array of coefficients for X (K,).
          - theta: 1D NumPy array of coefficients for WX (K,).
          - fitted: 1D NumPy array of fitted values (N,).
          - residuals: 1D NumPy array of residual errors (N,).
          - r2: Float coefficient of determination.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    X_mat = np.asarray(X, dtype=np.float64)
    n = len(y_arr)
    k = X_mat.shape[1]

    if len(X_mat) != n:
        raise ValueError("Length of y must match number of rows in X.")

    W_y = calculate_spatial_lag(y_arr, neighbors, weights, id_order)

    # Compute spatially lagged explanatory variables WX
    WX = np.zeros_like(X_mat)
    for col in range(k):
        WX[:, col] = calculate_spatial_lag(X_mat[:, col], neighbors, weights, id_order)

    # 2SLS: augmented design matrix [X, WX, Wy]
    Z = np.column_stack([X_mat, WX, W_y])
    coefs, _, _, _ = np.linalg.lstsq(Z, y_arr, rcond=None)

    beta = coefs[:k]
    theta = coefs[k : 2 * k]
    rho = float(coefs[-1])

    fitted = Z @ coefs
    residuals = y_arr - fitted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    return {
        "rho": rho,
        "beta": beta,
        "theta": theta,
        "fitted": fitted,
        "residuals": residuals,
        "r2": r2,
    }


def emerging_hotspot_analysis(
    coordinates: np.ndarray,
    values: np.ndarray,
    time_steps: np.ndarray,
    weights_matrix: np.ndarray,
    significance_level: float = 0.05,
) -> dict[str, np.ndarray]:
    """Combines Getis-Ord Gi* hot spot analysis at each time step with Mann-Kendall trend testing
    to classify spatio-temporal patterns.

    Args:
        coordinates: (N, 2) spatial coordinates of locations.
        values: (N, T) matrix - N locations x T time steps.
        time_steps: (T,) array of time indices or labels.
        weights_matrix: (N, N) spatial weights matrix (row-standardized).
        significance_level: threshold for statistical significance.

    Returns:
        Dictionary containing pattern classifications and statistics.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    times = np.asarray(time_steps)
    W = np.asarray(weights_matrix, dtype=np.float64)

    N, T = vals.shape
    if coords.shape != (N, 2):
        raise ValueError(f"coordinates must have shape (N, 2), got {coords.shape}")
    if len(times) != T:
        raise ValueError(f"time_steps length ({len(times)}) must match values columns ({T})")
    if W.shape != (N, N):
        raise ValueError(f"weights_matrix must have shape (N, N), got {W.shape}")
    if N < 2:
        raise ValueError("Need at least 2 locations.")
    if T < 3:
        raise ValueError("Need at least 3 time steps for Mann-Kendall test.")
    if not (0 < significance_level < 1):
        raise ValueError("significance_level must be between 0 and 1.")

    # Precompute Gi* denominator components
    w_sum = W.sum(axis=1)
    w_sq_sum = (W**2).sum(axis=1)
    # denominator factor inside sqrt
    denom_inner = (N * w_sq_sum - w_sum**2) / (N - 1)
    denom_inner[denom_inner < 0] = 0.0
    denom_factor = np.sqrt(denom_inner)

    z_scores = np.zeros((N, T))
    p_values_gi = np.ones((N, T))

    for t in range(T):
        x = vals[:, t]
        x_mean = np.mean(x)
        s = np.std(x)

        wx = W @ x
        num = wx - x_mean * w_sum
        den = s * denom_factor

        valid = den > 0
        z_t = np.zeros(N)
        z_t[valid] = num[valid] / den[valid]
        p_t = np.ones(N)
        p_t[valid] = 2.0 * stats.norm.sf(np.abs(z_t[valid]))

        z_scores[:, t] = z_t
        p_values_gi[:, t] = p_t

    # Step 2: Mann-Kendall Trend Test per location
    mk_z = np.zeros(N)
    mk_p = np.ones(N)
    kendall_tau = np.zeros(N)

    var_S = T * (T - 1) * (2 * T + 5) / 18.0
    sqrt_var_S = np.sqrt(var_S)
    denom_tau = (T * (T - 1)) / 2.0

    for i in range(N):
        z_series = z_scores[i, :]

        # Calculate S
        S = 0.0
        for j in range(T - 1):
            S += np.sum(np.sign(z_series[j + 1 :] - z_series[j]))

        tau = S / denom_tau
        kendall_tau[i] = tau

        if S > 0:
            z_val = (S - 1.0) / sqrt_var_S
        elif S < 0:
            z_val = (S + 1.0) / sqrt_var_S
        else:
            z_val = 0.0

        mk_z[i] = z_val
        mk_p[i] = 2.0 * stats.norm.sf(np.abs(z_val))

    # Step 3: Pattern Classification
    is_hot = (z_scores > 0) & (p_values_gi < significance_level)
    is_cold = (z_scores < 0) & (p_values_gi < significance_level)

    hot_counts = is_hot.sum(axis=1)
    cold_counts = is_cold.sum(axis=1)

    patterns = np.full(N, "no_pattern", dtype=object)

    for i in range(N):
        hot_series = is_hot[i, :]
        cold_series = is_cold[i, :]

        h_count = hot_counts[i]
        c_count = cold_counts[i]

        is_hot_latest = hot_series[-1]
        is_cold_latest = cold_series[-1]

        is_hot_all = h_count == T
        is_cold_all = c_count == T

        trend_sig = mk_p[i] < significance_level
        trend_inc = kendall_tau[i] > 0
        trend_dec = kendall_tau[i] < 0

        pct_hot = h_count / T
        pct_cold = c_count / T

        # Run lengths at end
        def get_run_length(series: np.ndarray) -> int:
            run = 0
            for val in reversed(series):
                if val:
                    run += 1
                else:
                    break
            return run

        hot_run = get_run_length(hot_series)
        cold_run = get_run_length(cold_series)

        # Classification for HOT
        if is_hot_latest and (pct_hot < 0.5) and not trend_sig:
            patterns[i] = "new_hot_spot"
        elif hot_run >= 3 and not is_hot_all and not trend_sig:
            patterns[i] = "consecutive_hot_spot"
        elif pct_hot >= 0.9 and trend_sig and trend_inc:
            patterns[i] = "intensifying_hot_spot"
        elif pct_hot == 1.0 and not trend_sig:
            patterns[i] = "persistent_hot_spot"
        elif pct_hot >= 0.9 and trend_sig and trend_dec:
            patterns[i] = "diminishing_hot_spot"
        elif is_hot_latest and (0.25 <= pct_hot <= 0.5) and not trend_sig:
            patterns[i] = "sporadic_hot_spot"
        elif is_hot_latest and (c_count > 0):
            patterns[i] = "oscillating_hot_spot"
        elif pct_hot >= 0.25 and not is_hot_latest:
            patterns[i] = "historical_hot_spot"

        # Classification for COLD (only if not already classified as hot spot)
        if patterns[i] == "no_pattern":
            if is_cold_latest and (pct_cold < 0.5) and not trend_sig:
                patterns[i] = "new_cold_spot"
            elif cold_run >= 3 and not is_cold_all and not trend_sig:
                patterns[i] = "consecutive_cold_spot"
            elif pct_cold >= 0.9 and trend_sig and trend_dec:
                patterns[i] = "intensifying_cold_spot"
            elif pct_cold == 1.0 and not trend_sig:
                patterns[i] = "persistent_cold_spot"
            elif pct_cold >= 0.9 and trend_sig and trend_inc:
                patterns[i] = "diminishing_cold_spot"
            elif is_cold_latest and (0.25 <= pct_cold <= 0.5) and not trend_sig:
                patterns[i] = "sporadic_cold_spot"
            elif is_cold_latest and (h_count > 0):
                patterns[i] = "oscillating_cold_spot"
            elif pct_cold >= 0.25 and not is_cold_latest:
                patterns[i] = "historical_cold_spot"

    return {
        "pattern": patterns,
        "z_scores": z_scores,
        "p_values_gi": p_values_gi,
        "mann_kendall_z": mk_z,
        "mann_kendall_p": mk_p,
        "kendall_tau": kendall_tau,
        "hot_spot_count": hot_counts,
        "cold_spot_count": cold_counts,
    }


def create_space_time_cube(
    coordinates: np.ndarray,
    timestamps: np.ndarray,
    values: np.ndarray,
    spatial_bin_size: float,
    temporal_bin_count: int,
    aggregation: str = "mean",
) -> dict[str, Any]:
    """Aggregates point event data into a 3D space-time cube for spatio-temporal analysis.

    Args:
        coordinates: (N, 2) array of (x, y) point locations
        timestamps: (N,) array of numeric time values
        values: (N,) array of attribute values to aggregate
        spatial_bin_size: size of spatial grid cells (in coordinate units)
        temporal_bin_count: number of temporal bins to divide the time range into
        aggregation: 'mean', 'sum', 'count', 'min', 'max', 'std'

    Returns:
        Dict containing the aggregated cube, bin centers, counts, and extents.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < 1:
        raise ValueError("coordinates must be a 2D array of shape (N, 2) with N >= 1.")

    n_points = coords.shape[0]
    if t.ndim != 1 or t.shape[0] != n_points:
        raise ValueError("timestamps must be a 1D array of length N.")

    if vals.ndim != 1 or vals.shape[0] != n_points:
        raise ValueError("values must be a 1D array of length N.")

    if spatial_bin_size <= 0:
        raise ValueError("spatial_bin_size must be > 0.")

    if temporal_bin_count < 1:
        raise ValueError("temporal_bin_count must be >= 1.")

    valid_aggregations = {"mean", "sum", "count", "min", "max", "std"}
    if aggregation not in valid_aggregations:
        raise ValueError(f"aggregation must be one of {valid_aggregations}.")

    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)

    n_x = int(math.ceil((x_max - x_min) / spatial_bin_size))
    n_y = int(math.ceil((y_max - y_min) / spatial_bin_size))

    if n_x == 0:
        n_x = 1
    if n_y == 0:
        n_y = 1

    bin_x = np.floor((coords[:, 0] - x_min) / spatial_bin_size).astype(int)
    bin_x = np.clip(bin_x, 0, n_x - 1)

    bin_y = np.floor((coords[:, 1] - y_min) / spatial_bin_size).astype(int)
    bin_y = np.clip(bin_y, 0, n_y - 1)

    t_min = np.min(t)
    t_max = np.max(t)
    if t_min == t_max:
        temporal_bin_width = 1.0
    else:
        temporal_bin_width = (t_max - t_min) / temporal_bin_count

    bin_t = np.floor((t - t_min) / temporal_bin_width).astype(int)
    bin_t = np.clip(bin_t, 0, temporal_bin_count - 1)

    cube = np.full((n_x, n_y, temporal_bin_count), np.nan)
    bin_counts = np.zeros((n_x, n_y, temporal_bin_count), dtype=int)

    bins_data: dict[tuple[int, int, int], list[float]] = {}
    for i in range(n_points):
        key = (bin_x[i], bin_y[i], bin_t[i])
        if key not in bins_data:
            bins_data[key] = []
        bins_data[key].append(vals[i])
        bin_counts[key] += 1

    if aggregation == "count":
        cube = np.zeros((n_x, n_y, temporal_bin_count), dtype=np.float64)
        for k, count in np.ndenumerate(bin_counts):
            cube[k] = float(count)
    else:
        for k, bvals in bins_data.items():
            if aggregation == "mean":
                cube[k] = float(np.mean(bvals))
            elif aggregation == "sum":
                cube[k] = float(np.sum(bvals))
            elif aggregation == "min":
                cube[k] = float(np.min(bvals))
            elif aggregation == "max":
                cube[k] = float(np.max(bvals))
            elif aggregation == "std":
                cube[k] = float(np.std(bvals, ddof=0)) if len(bvals) > 1 else 0.0

    x_centers = x_min + (np.arange(n_x) + 0.5) * spatial_bin_size
    y_centers = y_min + (np.arange(n_y) + 0.5) * spatial_bin_size
    t_centers = t_min + (np.arange(temporal_bin_count) + 0.5) * temporal_bin_width

    return {
        "cube": cube,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "t_centers": t_centers,
        "bin_counts": bin_counts,
        "spatial_extent": {
            "x_min": float(x_min),
            "x_max": float(x_max),
            "y_min": float(y_min),
            "y_max": float(y_max),
        },
        "temporal_extent": {"t_min": float(t_min), "t_max": float(t_max)},
        "n_spatial_bins": int(n_x * n_y),
        "n_populated_bins": len(bins_data),
    }


def fit_spatial_panel_model(
    dependent_var: np.ndarray,
    independent_vars: np.ndarray,
    weights_matrix: np.ndarray,
    time_periods: int,
    model_type: str = "lag",
) -> dict[str, Any]:
    """Fit a spatial panel model (lag or error) using 2SLS or spatial error estimation.

    Args:
        dependent_var: Array of dependent variables, shape (N*T,) or (N, T).
            If (N, T), it is flattened to (N*T,) in spatial-first order.
        independent_vars: Array of independent variables, shape (N*T, K) or (N, T, K).
        weights_matrix: Spatial weights matrix, shape (N, N).
        time_periods: Number of time periods (T).
        model_type: 'lag' (Spatial Panel Autoregressive) or 'error' (Spatial Panel Error).

    Returns:
        dict[str, Any]: A dictionary containing coefficients, spatial_parameter,
            std_errors, t_stat, p_values, r_squared, residuals, model_type,
            n_spatial_units, and time_periods.
    """
    y = np.asarray(dependent_var, dtype=np.float64)
    X = np.asarray(independent_vars, dtype=np.float64)
    W = np.asarray(weights_matrix, dtype=np.float64)

    if time_periods < 2:
        raise ValueError("time_periods must be >= 2")

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("weights_matrix must be a square 2D array")

    N = W.shape[0]
    T = time_periods

    if N < 3:
        raise ValueError("Number of spatial units (N) must be >= 3")

    if y.ndim == 2:
        if y.shape != (N, T):
            raise ValueError(f"dependent_var shape {y.shape} does not match (N, T) = ({N}, {T})")
        y = y.T.flatten()
    elif y.ndim == 1:
        if y.shape[0] != N * T:
            raise ValueError(f"dependent_var length {y.shape[0]} does not match N*T = {N * T}")
    else:
        raise ValueError("dependent_var must be 1D or 2D array")

    if X.ndim == 3:
        if X.shape[:2] != (N, T):
            raise ValueError(f"independent_vars shape {X.shape} does not match (N, T, K)")
        K = X.shape[2]
        X = X.transpose((1, 0, 2)).reshape((N * T, K))
    elif X.ndim == 2:
        if X.shape[0] != N * T:
            raise ValueError(
                f"independent_vars first dim {X.shape[0]} does not match N*T = {N * T}"
            )
        K = X.shape[1]
    else:
        raise ValueError("independent_vars must be 2D or 3D array")

    model_type = model_type.lower()
    if model_type not in ["lag", "error"]:
        raise ValueError("model_type must be 'lag' or 'error'")

    I_T = np.eye(T)
    W_full = np.kron(I_T, W)

    if model_type == "lag":
        WX = W_full @ X
        W2X = W_full @ WX
        Z = np.hstack((X, WX, W2X))

        Wy = W_full @ y

        Z_pinv = np.linalg.pinv(Z)
        Wy_hat = Z @ (Z_pinv @ Wy)

        X_stage2 = np.column_stack((X, Wy_hat))

        coef = np.linalg.lstsq(X_stage2, y, rcond=None)[0]

        X_actual = np.column_stack((X, Wy))
        residuals = y - X_actual @ coef

        n_obs = N * T
        k_vars = K + 1
        sig2 = np.sum(residuals**2) / (n_obs - k_vars)

        cov_matrix = sig2 * np.linalg.inv(X_stage2.T @ X_stage2)
        std_errors = np.sqrt(np.diag(cov_matrix))

        t_stat = coef / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_obs - k_vars))

        y_mean = np.mean(y)
        tss = np.sum((y - y_mean) ** 2)
        rss = np.sum(residuals**2)
        r_squared = 1 - (rss / tss) if tss > 0 else 0.0

        return {
            "coefficients": coef[:K],
            "spatial_parameter": float(coef[K]),
            "std_errors": std_errors[:K],
            "t_stat": t_stat[:K],
            "p_values": p_values[:K],
            "r_squared": float(r_squared),
            "residuals": residuals,
            "model_type": "lag",
            "n_spatial_units": N,
            "time_periods": T,
        }

    else:
        coef_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ coef_ols

        We = W_full @ e
        lambda_val = np.linalg.lstsq(We.reshape(-1, 1), e, rcond=None)[0][0]

        y_star = y - lambda_val * (W_full @ y)
        X_star = X - lambda_val * (W_full @ X)

        coef = np.linalg.lstsq(X_star, y_star, rcond=None)[0]

        residuals_final = y - X @ coef

        n_obs = N * T
        k_vars = K
        u_hat = y_star - X_star @ coef
        sig2 = np.sum(u_hat**2) / (n_obs - k_vars)
        cov_matrix = sig2 * np.linalg.inv(X_star.T @ X_star)

        std_errors = np.sqrt(np.diag(cov_matrix))
        t_stat = coef / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_obs - k_vars))

        y_star_mean = np.mean(y_star)
        tss = np.sum((y_star - y_star_mean) ** 2)
        rss = np.sum(u_hat**2)
        r_squared = 1 - (rss / tss) if tss > 0 else 0.0

        return {
            "coefficients": coef,
            "spatial_parameter": float(lambda_val),
            "std_errors": std_errors,
            "t_stat": t_stat,
            "p_values": p_values,
            "r_squared": float(r_squared),
            "residuals": residuals_final,
            "model_type": "error",
            "n_spatial_units": N,
            "time_periods": T,
        }


def fit_spatial_tobit_panel(
    dependent_var: np.ndarray,
    independent_vars: np.ndarray,
    weights_matrix: np.ndarray,
    time_periods: int,
    censoring_limit: float = 0.0,
) -> dict[str, Any]:
    """Fits a Spatial Panel Autoregressive Tobit model for zero/left-censored data.

    Args:
        dependent_var: Array of dependent variables, shape (N*T,) or (N, T).
            If (N, T), it is flattened to (N*T,) in spatial-first order.
        independent_vars: Array of independent variables, shape (N*T, K) or (N, T, K).
        weights_matrix: Spatial weights matrix, shape (N, N).
        time_periods: Number of time periods (T).
        censoring_limit: Left-censoring threshold (default 0.0).

    Returns:
        dict[str, Any]: Dictionary containing model parameters and diagnostics.
    """
    y = np.asarray(dependent_var, dtype=np.float64)
    X = np.asarray(independent_vars, dtype=np.float64)
    W = np.asarray(weights_matrix, dtype=np.float64)

    if time_periods < 2:
        raise ValueError("time_periods must be >= 2")

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("weights_matrix must be a square 2D array")

    N = W.shape[0]
    T = time_periods

    if N < 3:
        raise ValueError("Number of spatial units (N) must be >= 3")

    if y.ndim == 2:
        if y.shape != (N, T):
            raise ValueError(f"dependent_var shape {y.shape} does not match (N, T) = ({N}, {T})")
        y = y.T.flatten()
    elif y.ndim == 1:
        if y.shape[0] != N * T:
            raise ValueError(f"dependent_var length {y.shape[0]} does not match N*T = {N * T}")
    else:
        raise ValueError("dependent_var must be 1D or 2D array")

    if X.ndim == 3:
        if X.shape[:2] != (N, T):
            raise ValueError(f"independent_vars shape {X.shape} does not match (N, T, K)")
        K = X.shape[2]
        X = X.transpose((1, 0, 2)).reshape((N * T, K))
    elif X.ndim == 2:
        if X.shape[0] != N * T:
            raise ValueError(
                f"independent_vars first dim {X.shape[0]} does not match N*T = {N * T}"
            )
        K = X.shape[1]
    else:
        raise ValueError("independent_vars must be 2D or 3D array")

    # Create W_full = I_T ⊗ W_spatial (N*T, N*T)
    I_T = np.eye(T)
    W_full = np.kron(I_T, W)

    uncensored = y > censoring_limit
    uncensored_count = int(np.sum(uncensored))
    censored_count = (N * T) - uncensored_count

    if uncensored_count < (K + 1):
        raise ValueError("Too few uncensored observations for Tobit estimation.")

    # Compute 2SLS Spatial Lag on uncensored subset
    y_unc = y[uncensored]
    X_unc = X[uncensored]

    Wy = W_full @ y
    WX = W_full @ X
    Wy_unc = Wy[uncensored]
    WX_unc = WX[uncensored]

    # Instruments Z for uncensored
    Z_unc = np.column_stack((X_unc, WX_unc))
    X_sp_unc = np.column_stack((Wy_unc, X_unc))

    # P_Z = Z (Z'Z)^-1 Z'
    ztz_inv_unc = np.linalg.pinv(Z_unc.T @ Z_unc)
    P_Z_unc = Z_unc @ ztz_inv_unc @ Z_unc.T
    X_hat_unc = P_Z_unc @ X_sp_unc

    beta_full_init = np.linalg.pinv(X_hat_unc.T @ X_hat_unc) @ (X_hat_unc.T @ y_unc)

    # Iterative Tobit Expected Value Adjustment
    X_sp_full = np.column_stack((Wy, X))
    y_pred = X_sp_full @ beta_full_init

    residuals_init_unc = y_unc - y_pred[uncensored]
    sigma = float(np.std(residuals_init_unc))

    y_star = y.copy()
    if sigma > 0 and censored_count > 0:
        censored = ~uncensored
        alpha = (censoring_limit - y_pred[censored]) / sigma

        cdf_alpha = np.maximum(stats.norm.cdf(alpha), 1e-10)
        pdf_alpha = stats.norm.pdf(alpha)
        lambda_mills = pdf_alpha / cdf_alpha

        y_star[censored] = y_pred[censored] - sigma * lambda_mills
        y_star[censored] = np.minimum(y_star[censored], censoring_limit)

    # Final 2SLS regression of latent y* on [X, W_full @ y*]
    Wy_star = W_full @ y_star
    Z_star = np.column_stack((X, WX))
    X_sp_star = np.column_stack((Wy_star, X))

    ztz_inv_star = np.linalg.pinv(Z_star.T @ Z_star)
    P_Z_star = Z_star @ ztz_inv_star @ Z_star.T
    X_hat_star = P_Z_star @ X_sp_star

    xtx_inv_final = np.linalg.pinv(X_hat_star.T @ X_hat_star)
    beta_full = xtx_inv_final @ (X_hat_star.T @ y_star)

    rho = float(beta_full[0])
    beta = beta_full[1:]

    y_fitted = X_sp_star @ beta_full
    residuals = y_star - y_fitted

    sse = float(np.sum(residuals**2))
    df = (N * T) - (K + 1)
    s2 = sse / df if df > 0 else 0.0

    cov_beta = s2 * xtx_inv_final
    se_full = np.sqrt(np.maximum(0.0, np.diagonal(cov_beta)))
    t_full = np.zeros_like(beta_full)
    p_full = np.ones_like(beta_full)

    for j in range(len(beta_full)):
        if se_full[j] > 0:
            t_full[j] = beta_full[j] / se_full[j]
            p_full[j] = 2.0 * (1.0 - stats.t.cdf(abs(t_full[j]), df=max(1, df)))

    sst = float(np.sum((y_star - np.mean(y_star)) ** 2))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else 0.0

    return {
        "coefficients": beta,
        "spatial_rho": rho,
        "std_errors": se_full[1:],
        "t_stat": t_full[1:],
        "p_values": p_full[1:],
        "r_squared": r2,
        "censored_count": censored_count,
        "uncensored_count": uncensored_count,
        "residuals": residuals,
    }


def fit_spatial_sarma_panel(
    dependent_var: np.ndarray,
    independent_vars: np.ndarray,
    weights_matrix: np.ndarray,
    time_periods: int,
):

    y = np.asarray(dependent_var, dtype=np.float64)

    X = np.asarray(independent_vars, dtype=np.float64)

    W = np.asarray(weights_matrix, dtype=np.float64)

    if time_periods < 2:
        raise ValueError("time_periods must be >= 2")

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("weights_matrix must be a square 2D array")

    N = W.shape[0]

    T = time_periods

    if N < 3:
        raise ValueError("Number of spatial units (N) must be >= 3")

    if y.ndim == 2:
        if y.shape != (N, T):
            raise ValueError(f"dependent_var shape {y.shape} does not match (N, T) = ({N}, {T})")

        y = y.T.flatten()

    elif y.ndim == 1:
        if y.shape[0] != N * T:
            raise ValueError(f"dependent_var length {y.shape[0]} does not match N*T = {N * T}")

    else:
        raise ValueError("dependent_var must be 1D or 2D array")

    if X.ndim == 3:
        if X.shape[:2] != (N, T):
            raise ValueError(f"independent_vars shape {X.shape} does not match (N, T, K)")

        K = X.shape[2]

        X = X.transpose((1, 0, 2)).reshape((N * T, K))

    elif X.ndim == 2:
        if X.shape[0] != N * T:
            raise ValueError(
                f"independent_vars first dim {X.shape[0]} does not match N*T = {N * T}"
            )

        K = X.shape[1]

    else:
        raise ValueError("independent_vars must be 2D or 3D array")

    I_T = np.eye(T)

    W_full = np.kron(I_T, W)

    Wy = W_full @ y

    WX = W_full @ X

    W2X = W_full @ WX

    Z = np.column_stack((X, WX, W2X))

    Z_pinv = np.linalg.pinv(Z)

    Wy_hat = Z @ (Z_pinv @ Wy)

    X_stage1_hat = np.column_stack((X, Wy_hat))

    coef_stage1 = np.linalg.lstsq(X_stage1_hat, y, rcond=None)[0]

    beta_stage1 = coef_stage1[:K]

    rho_stage1 = float(coef_stage1[K])

    e = y - rho_stage1 * Wy - X @ beta_stage1

    We = W_full @ e

    lambda_val = float(np.linalg.lstsq(We.reshape(-1, 1), e, rcond=None)[0][0])

    y_star = y - lambda_val * Wy

    X_star = X - lambda_val * WX

    Wy_star = W_full @ y_star

    WX_star = W_full @ X_star

    W2X_star = W_full @ WX_star

    Z_star = np.column_stack((X_star, WX_star, W2X_star))

    Z_star_pinv = np.linalg.pinv(Z_star)

    Wy_star_hat = Z_star @ (Z_star_pinv @ Wy_star)

    X_stage2_hat = np.column_stack((X_star, Wy_star_hat))

    coef_final = np.linalg.lstsq(X_stage2_hat, y_star, rcond=None)[0]

    beta_final = coef_final[:K]

    rho_final = float(coef_final[K])

    residuals_final = y - rho_final * Wy - X @ beta_final

    n_obs = N * T

    k_vars = K + 1

    sig2 = np.sum(residuals_final**2) / (n_obs - k_vars)

    cov_matrix = sig2 * np.linalg.pinv(X_stage2_hat.T @ X_stage2_hat)

    std_errors_all = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))

    std_errors = std_errors_all[:K]

    t_stat = beta_final / std_errors

    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n_obs - k_vars))

    y_mean = np.mean(y)

    tss = np.sum((y - y_mean) ** 2)

    rss = np.sum(residuals_final**2)

    r_squared = 1 - (rss / tss) if tss > 0 else 0.0

    return {
        "coefficients": beta_final,
        "spatial_rho": rho_final,
        "spatial_lambda": lambda_val,
        "std_errors": std_errors,
        "t_stat": t_stat,
        "p_values": p_values,
        "r_squared": float(r_squared),
        "residuals": residuals_final,
    }


def fit_spatial_probit_panel(
    dependent_var: np.ndarray,
    independent_vars: np.ndarray,
    weights_matrix: np.ndarray,
    time_periods: int,
) -> dict[str, Any]:
    """Spatial Panel Autoregressive Probit model for binary panel outcome.

    Fits a spatial panel autoregressive probit model for binary outcomes using an
    IRLS/GLM approximation followed by 2SLS to estimate the spatial lag parameter.

    Args:
        dependent_var: (N*T,) or (N, T) array of binary 0/1 values.
        independent_vars: (N*T, K) array of K regressors.
        weights_matrix: (N, N) spatial weights matrix (row-standardized).
        time_periods: Number of time periods T (int >= 2).

    Returns:
        Dict with keys:
        - coefficients: (K,) float array
        - spatial_rho: float
        - std_errors: (K,) float array
        - z_stat: (K,) float array
        - p_values: (K,) float array
        - pseudo_r_squared: float
        - log_likelihood: float
        - classification_accuracy: float [0, 1]
        - predicted_probabilities: (N*T,) float array
    """
    y = np.asarray(dependent_var, dtype=np.float64)
    X = np.asarray(independent_vars, dtype=np.float64)
    W = np.asarray(weights_matrix, dtype=np.float64)

    if time_periods < 2:
        raise ValueError("time_periods must be >= 2")

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("weights_matrix must be a square 2D array")

    N = W.shape[0]
    T = time_periods

    if N < 3:
        raise ValueError("Number of spatial units (N) must be >= 3")

    if y.ndim == 2:
        if y.shape != (N, T):
            raise ValueError(f"dependent_var shape {y.shape} does not match (N, T) = ({N}, {T})")
        y = y.T.flatten()
    elif y.ndim == 1:
        if y.shape[0] != N * T:
            raise ValueError(f"dependent_var length {y.shape[0]} does not match N*T = {N * T}")
    else:
        raise ValueError("dependent_var must be 1D or 2D array")

    unique_vals = np.unique(y)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError("dependent_var must contain only binary 0/1 values")

    if X.ndim == 3:
        if X.shape[:2] != (N, T):
            raise ValueError(f"independent_vars shape {X.shape} does not match (N, T, K)")
        K = X.shape[2]
        X = X.transpose((1, 0, 2)).reshape((N * T, K))
    elif X.ndim == 2:
        if X.shape[0] != N * T:
            raise ValueError(
                f"independent_vars first dim {X.shape[0]} does not match N*T = {N * T}"
            )
        K = X.shape[1]
    else:
        raise ValueError("independent_vars must be 2D or 3D array")

    I_T = np.eye(T)
    W_full = np.kron(I_T, W)

    beta_glm = np.zeros(K)
    for _ in range(10):
        eta = X @ beta_glm
        mu = stats.norm.cdf(eta)
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        phi = stats.norm.pdf(eta)
        phi = np.clip(phi, 1e-6, None)

        W_irls = (phi**2) / (mu * (1 - mu))
        z = eta + (y - mu) / phi

        W_sqrt = np.sqrt(W_irls)[:, np.newaxis]
        X_w = X * W_sqrt
        z_w = z * np.squeeze(W_sqrt)
        beta_new = np.linalg.lstsq(X_w, z_w, rcond=None)[0]
        if np.max(np.abs(beta_new - beta_glm)) < 1e-5:
            beta_glm = beta_new
            break
        beta_glm = beta_new

    eta = X @ beta_glm
    mu = stats.norm.cdf(eta)
    mu = np.clip(mu, 1e-6, 1 - 1e-6)
    phi = stats.norm.pdf(eta)
    phi = np.clip(phi, 1e-6, None)
    y_latent = eta + (y - mu) / phi

    W_y_latent = W_full @ y_latent
    WX = W_full @ X
    W2X = W_full @ WX
    Z_inst = np.column_stack((X, WX, W2X))

    Z_pinv = np.linalg.pinv(Z_inst)
    W_y_latent_hat = Z_inst @ (Z_pinv @ W_y_latent)

    X_stage2 = np.column_stack((X, W_y_latent_hat))
    coef_2sls = np.linalg.lstsq(X_stage2, y_latent, rcond=None)[0]

    beta_final = coef_2sls[:K]
    rho_final = float(coef_2sls[K])

    residuals = y_latent - rho_final * W_y_latent - X @ beta_final
    dof = N * T - (K + 1)
    sig2 = np.sum(residuals**2) / dof

    cov_matrix = sig2 * np.linalg.inv(X_stage2.T @ X_stage2)
    std_errors_all = np.sqrt(np.diag(cov_matrix))
    std_errors = std_errors_all[:K]

    z_stat = beta_final / std_errors
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    y_pred_latent = rho_final * W_y_latent + X @ beta_final
    prob_pred = stats.norm.cdf(y_pred_latent)
    prob_pred = np.clip(prob_pred, 1e-10, 1 - 1e-10)

    ll_full = float(np.sum(y * np.log(prob_pred) + (1 - y) * np.log(1 - prob_pred)))

    p_null = np.mean(y)
    if 0 < p_null < 1:
        ll_null = float(np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null)))
    else:
        ll_null = ll_full

    pseudo_r2 = float(1 - (ll_full / ll_null)) if ll_null != 0 else 0.0

    y_pred_class = (prob_pred >= 0.5).astype(float)
    accuracy = float(np.mean(y_pred_class == y))

    return {
        "coefficients": beta_final,
        "spatial_rho": rho_final,
        "std_errors": std_errors,
        "z_stat": z_stat,
        "p_values": p_values,
        "pseudo_r_squared": pseudo_r2,
        "log_likelihood": ll_full,
        "classification_accuracy": accuracy,
        "predicted_probabilities": prob_pred,
    }


def fit_spatial_quantile_panel(
    dependent_var: np.ndarray,
    independent_vars: np.ndarray,
    weights_matrix: np.ndarray,
    time_periods: int,
    quantile: float = 0.5,
) -> dict[str, Any]:
    """Fits a Spatial Panel Autoregressive Quantile Regression model.

    Args:
        dependent_var: 1D or 2D array of dependent variables, shape (N*T,) or (N, T).
        independent_vars: 2D array of K regressors, shape (N*T, K).
        weights_matrix: 2D spatial weights matrix (row-standardized), shape (N, N).
        time_periods: Number of time periods (T >= 2).
        quantile: Quantile to fit, strictly between 0 and 1. Default is 0.5 (median).

    Returns:
        A dictionary containing:
            - 'coefficients': (K,) array of fitted regression coefficients.
            - 'spatial_rho': Spatial lag coefficient (float).
            - 'quantile': The fitted quantile (float).
            - 'pinball_loss': Sum of the pinball loss (float).
            - 'pseudo_r_squared': Pseudo R-squared (float).
            - 'residuals': (N*T,) array of residuals.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError("Quantile must be strictly between 0 and 1.")

    if time_periods < 2:
        raise ValueError("time_periods must be >= 2.")

    y = np.asarray(dependent_var, dtype=np.float64).ravel()
    X = np.asarray(independent_vars, dtype=np.float64)
    W = np.asarray(weights_matrix, dtype=np.float64)

    n = W.shape[0]
    if n < 3:
        raise ValueError("Number of spatial units must be >= 3.")

    if W.shape != (n, n):
        raise ValueError("weights_matrix must be a square (N, N) array.")

    if len(y) != n * time_periods:
        raise ValueError("Length of dependent_var must equal N * T.")

    if X.ndim != 2:
        raise ValueError("independent_vars must be a 2D array.")

    if X.shape[0] != n * time_periods:
        raise ValueError("Number of rows in independent_vars must equal N * T.")

    # W_full = I_T \otimes W_spatial
    W_full = np.kron(np.eye(time_periods), W)

    Wy = W_full @ y

    # Instruments Z = [X, W_full @ X, W_full^2 @ X]
    WX = W_full @ X
    WWX = W_full @ WX
    Z = np.hstack((X, WX, WWX))

    # 2SLS fitted Wy_hat = Z @ pinv(Z) @ Wy
    beta_z, _, _, _ = np.linalg.lstsq(Z, Wy, rcond=None)
    Wy_hat = Z @ beta_z

    # Quantile Regression of y on [X, Wy_hat]
    X_aug = np.hstack((X, Wy_hat.reshape(-1, 1)))
    n_obs, p = X_aug.shape

    # Linear programming formulation for quantile regression
    c = np.concatenate([np.zeros(p), quantile * np.ones(n_obs), (1 - quantile) * np.ones(n_obs)])
    A_eq = sparse.hstack([X_aug, sparse.eye(n_obs), -sparse.eye(n_obs)])
    b_eq = y

    # Variables: beta (unbounded), u+ (>=0), u- (>=0)
    bounds = [(None, None)] * p + [(0, None)] * (2 * n_obs)

    try:
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    except Exception as e:
        logger.error("Optimization failed in fit_spatial_quantile_panel: %s", e)
        raise ValueError(f"Quantile regression optimization failed: {e}") from e

    if not res.success:
        raise ValueError(f"Quantile regression optimization failed: {res.message}")

    params = res.x[:p]
    beta_tau = params[:-1]
    rho_tau = float(params[-1])

    # Residuals and Pinball Loss
    fitted_vals = X_aug @ params
    residuals = y - fitted_vals

    pinball_loss = float(
        np.sum(np.where(residuals >= 0, quantile * residuals, (quantile - 1) * residuals))
    )

    # Null model: y = alpha
    y_tau = float(np.quantile(y, quantile))
    null_residuals = y - y_tau
    pinball_loss_null = float(
        np.sum(
            np.where(
                null_residuals >= 0, quantile * null_residuals, (quantile - 1) * null_residuals
            )
        )
    )

    pseudo_r2 = 1.0 - (pinball_loss / pinball_loss_null) if pinball_loss_null > 0 else 0.0

    return {
        "coefficients": beta_tau,
        "spatial_rho": rho_tau,
        "quantile": float(quantile),
        "pinball_loss": pinball_loss,
        "pseudo_r_squared": float(pseudo_r2),
        "residuals": residuals,
    }


def fit_spatial_count_panel(
    dependent_var: np.ndarray,
    independent_vars: np.ndarray,
    weights_matrix: np.ndarray,
    time_periods: int,
    model_type: str = "poisson",
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Fits a Spatial Panel Poisson or Negative Binomial Count Regression model.

    Args:
        dependent_var: 1D or 2D array of non-negative count response variables,
            shape (N*T,) or (N, T).
        independent_vars: 2D array of K regressors, shape (N*T, K).
        weights_matrix: 2D spatial weights matrix (row-standardized), shape (N, N).
        time_periods: Number of time periods (T >= 2).
        model_type: 'poisson' or 'negative_binomial' (default 'poisson').
        max_iter: Maximum number of IRLS iterations (default 50).
        tol: Iteration convergence tolerance (default 1e-6).

    Returns:
        A dictionary containing:
            - 'coefficients': (K,) array of regression coefficients.
            - 'spatial_rho': Spatial lag parameter float.
            - 'dispersion_alpha': Estimated negative binomial dispersion parameter float
              (0.0 for Poisson).
            - 'log_likelihood': Model log-likelihood float.
            - 'deviance': Model deviance float.
            - 'pseudo_r_squared': McFadden's pseudo R-squared float.
            - 'fitted_values': (N*T,) array of predicted mean counts mu.
            - 'residuals': (N*T,) array of Pearson residuals.
    """
    model_type_clean = model_type.lower().strip()
    if model_type_clean not in ("poisson", "negative_binomial", "nb"):
        raise ValueError("model_type must be 'poisson' or 'negative_binomial'.")

    if time_periods < 2:
        raise ValueError("time_periods must be >= 2.")

    y = np.asarray(dependent_var, dtype=np.float64).ravel()
    X = np.asarray(independent_vars, dtype=np.float64)
    W = np.asarray(weights_matrix, dtype=np.float64)

    n = W.shape[0]
    if n < 3:
        raise ValueError("Number of spatial units must be >= 3.")
    if W.shape != (n, n):
        raise ValueError("weights_matrix must be a square (N, N) array.")
    if len(y) != n * time_periods:
        raise ValueError("Length of dependent_var must equal N * T.")
    if np.any(y < 0):
        raise ValueError("dependent_var counts must be non-negative.")
    if X.ndim != 2 or X.shape[0] != n * time_periods:
        raise ValueError("independent_vars must be a 2D array with N * T rows.")

    W_full = np.kron(np.eye(time_periods), W)
    Wy = W_full @ y

    # 2SLS Spatial Lag instrument stage: Z = [X, W_full @ X, W_full^2 @ X]
    WX = W_full @ X
    WWX = W_full @ WX
    Z = np.hstack((X, WX, WWX))

    beta_z, _, _, _ = np.linalg.lstsq(Z, Wy, rcond=None)
    Wy_hat = Z @ beta_z

    X_aug = np.hstack((X, Wy_hat.reshape(-1, 1)))
    n_obs, p = X_aug.shape

    # IRLS fitting
    y_start = np.maximum(y, 0.1)
    beta_aug, _, _, _ = np.linalg.lstsq(X_aug, np.log(y_start), rcond=None)

    dispersion_alpha = 0.0
    if model_type_clean in ("negative_binomial", "nb"):
        dispersion_alpha = 0.5

    for _ in range(max_iter):
        eta = np.clip(X_aug @ beta_aug, -30.0, 30.0)
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-10)

        if model_type_clean in ("negative_binomial", "nb"):
            var = mu + dispersion_alpha * (mu**2)
            pearson = ((y - mu) ** 2) / var
            dispersion_alpha = max(
                1e-4, float(dispersion_alpha * (np.sum(pearson) / max(n_obs - p, 1)))
            )
            var = mu + dispersion_alpha * (mu**2)
        else:
            var = mu

        weights_irls = (mu**2) / var
        weights_irls = np.maximum(weights_irls, 1e-10)

        z_working = eta + (y - mu) / mu

        W_sqrt = np.sqrt(weights_irls)
        X_w = X_aug * W_sqrt[:, None]
        z_w = z_working * W_sqrt

        beta_new, _, _, _ = np.linalg.lstsq(X_w, z_w, rcond=None)

        if np.max(np.abs(beta_new - beta_aug)) < tol:
            beta_aug = beta_new
            break
        beta_aug = beta_new

    eta = np.clip(X_aug @ beta_aug, -30.0, 30.0)
    mu = np.exp(eta)

    beta_final = beta_aug[:-1]
    rho_final = float(beta_aug[-1])

    if model_type_clean == "poisson":
        ll_i = y * np.log(np.maximum(mu, 1e-10)) - mu
        ll_model = float(np.sum(ll_i))
        deviance = float(
            2.0 * np.sum(np.where(y > 0, y * np.log(y / np.maximum(mu, 1e-10)) - (y - mu), mu))
        )

        y_mean = float(max(float(np.mean(y)), 1e-10))
        ll_null = float(np.sum(y * np.log(y_mean) - y_mean))
    else:
        r = 1.0 / dispersion_alpha
        ll_i = stats.nbinom.logpmf(np.floor(y).astype(int), r, r / (r + mu))
        ll_model = float(np.sum(ll_i))
        deviance = float(
            np.sum(
                2.0 * ((y + r) * np.log((y + r) / (mu + r)) - y * np.log(np.maximum(y, 1e-10) / mu))
            )
        )

        y_mean = float(max(float(np.mean(y)), 1e-10))
        ll_null = float(np.sum(stats.nbinom.logpmf(np.floor(y).astype(int), r, r / (r + y_mean))))

    pseudo_r2 = float(1.0 - (ll_model / ll_null)) if ll_null != 0 else 0.0
    var_final = mu if model_type_clean == "poisson" else (mu + dispersion_alpha * (mu**2))
    pearson_residuals = (y - mu) / np.sqrt(np.maximum(var_final, 1e-10))

    return {
        "coefficients": beta_final,
        "spatial_rho": rho_final,
        "dispersion_alpha": float(dispersion_alpha),
        "log_likelihood": ll_model,
        "deviance": deviance,
        "pseudo_r_squared": max(0.0, pseudo_r2),
        "fitted_values": mu,
        "residuals": pearson_residuals,
    }


def fit_spatial_zip_panel(
    dependent_var: np.ndarray,
    independent_vars: np.ndarray,
    weights_matrix: np.ndarray,
    time_periods: int,
    zero_inflation_vars: Optional[np.ndarray] = None,
    dist: str = "poisson",
    max_iter: int = 50,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Fits a Spatial Panel Zero-Inflated Poisson or Negative Binomial model (ZIP / ZINB).

    Combines an EM algorithm for structural zero inflation with 2SLS spatial lag estimation.

    Args:
        dependent_var: 1D array of count observations of shape (N*T,).
        independent_vars: 2D array of K count regressors of shape (N*T, K).
        weights_matrix: 2D spatial weights matrix (N, N).
        time_periods: Number of time periods (T >= 2).
        zero_inflation_vars: Optional 2D array of P zero-inflation regressors of shape (N*T, P).
        dist: Distribution choice 'poisson' or 'negative_binomial' (default 'poisson').
        max_iter: Maximum EM iterations (default 50).
        tol: Convergence tolerance (default 1e-6).

    Returns:
        Dict containing:
            - 'count_coefficients': (K,) array of count model parameters.
            - 'zero_coefficients': (P,) array of zero-inflation logistic parameters.
            - 'spatial_rho': Spatial lag autoregressive parameter float.
            - 'dispersion_alpha': Dispersion parameter float (0.0 for Poisson).
            - 'zero_inflation_mean': Mean structural zero probability across observations.
            - 'log_likelihood': Model log-likelihood float.
            - 'pseudo_r_squared': McFadden's pseudo R-squared float.
            - 'fitted_values': (N*T,) array of expected count values E[Y] = (1 - pi) * mu.
            - 'zero_probabilities': (N*T,) array of estimated zero probabilities pi.
    """
    dist_clean = dist.lower().strip()
    if dist_clean not in ("poisson", "negative_binomial", "zinb", "zip"):
        raise ValueError("dist must be 'poisson' or 'negative_binomial'.")

    if time_periods < 2:
        raise ValueError("time_periods must be >= 2.")

    y = np.asarray(dependent_var, dtype=np.float64).ravel()
    X = np.asarray(independent_vars, dtype=np.float64)
    W = np.asarray(weights_matrix, dtype=np.float64)

    n = W.shape[0]
    if n < 3:
        raise ValueError("Number of spatial units must be >= 3.")
    if W.shape != (n, n):
        raise ValueError("weights_matrix must be a square (N, N) array.")
    if len(y) != n * time_periods:
        raise ValueError("Length of dependent_var must equal N * T.")
    if np.any(y < 0):
        raise ValueError("dependent_var counts must be non-negative.")
    if X.ndim != 2 or X.shape[0] != n * time_periods:
        raise ValueError("independent_vars must be a 2D array with N * T rows.")

    if zero_inflation_vars is not None:
        Z_zero = np.asarray(zero_inflation_vars, dtype=np.float64)
        if Z_zero.ndim != 2 or Z_zero.shape[0] != n * time_periods:
            raise ValueError("zero_inflation_vars must be a 2D array with N * T rows.")
    else:
        Z_zero = np.copy(X)

    n_obs, _ = X.shape
    p_zero = Z_zero.shape[1]

    W_full = np.kron(np.eye(time_periods), W)
    Wy = W_full @ y
    WX = W_full @ X
    WWX = W_full @ WX
    Z_inst = np.hstack((X, WX, WWX))

    beta_z, _, _, _ = np.linalg.lstsq(Z_inst, Wy, rcond=None)
    Wy_hat = Z_inst @ beta_z

    X_aug = np.hstack((X, Wy_hat.reshape(-1, 1)))

    gamma = np.zeros(p_zero)
    beta_aug, _, _, _ = np.linalg.lstsq(X_aug, np.log(np.maximum(y, 0.1)), rcond=None)
    dispersion_alpha = 0.5 if dist_clean in ("negative_binomial", "zinb") else 0.0

    zero_mask = y == 0

    for _ in range(max_iter):
        logit_pi = np.clip(Z_zero @ gamma, -30.0, 30.0)
        pi = 1.0 / (1.0 + np.exp(-logit_pi))

        mu = np.exp(np.clip(X_aug @ beta_aug, -30.0, 30.0))
        mu = np.maximum(mu, 1e-10)

        if dist_clean in ("negative_binomial", "zinb"):
            r = 1.0 / max(dispersion_alpha, 1e-4)
            p_count_zero = (r / (r + mu)) ** r
        else:
            p_count_zero = np.exp(-mu)

        w_zero = np.zeros(n_obs)
        w_zero[zero_mask] = pi[zero_mask] / np.maximum(
            pi[zero_mask] + (1.0 - pi[zero_mask]) * p_count_zero[zero_mask], 1e-12
        )

        pi_w = np.clip(pi, 1e-6, 1.0 - 1e-6)
        v_zero = pi_w * (1.0 - pi_w)
        z_logit = logit_pi + (w_zero - pi) / np.maximum(v_zero, 1e-6)
        W_sqrt_z = np.sqrt(v_zero)
        Z_w = Z_zero * W_sqrt_z[:, None]
        z_w = z_logit * W_sqrt_z
        gamma_new, _, _, _ = np.linalg.lstsq(Z_w, z_w, rcond=None)

        w_count = 1.0 - w_zero
        weights_count = w_count * mu
        weights_count = np.maximum(weights_count, 1e-10)

        eta = np.log(mu)
        z_count = eta + (y - mu) / np.maximum(mu, 1e-6)

        W_sqrt_c = np.sqrt(weights_count)
        X_w = X_aug * W_sqrt_c[:, None]
        z_w_c = z_count * W_sqrt_c
        beta_new, _, _, _ = np.linalg.lstsq(X_w, z_w_c, rcond=None)

        diff = np.max(np.abs(beta_new - beta_aug)) + np.max(np.abs(gamma_new - gamma))
        beta_aug = beta_new
        gamma = np.asarray(gamma_new, dtype=np.float64)

        if diff < tol:
            break

    logit_pi = np.clip(Z_zero @ gamma, -30.0, 30.0)
    pi = 1.0 / (1.0 + np.exp(-logit_pi))
    mu = np.exp(np.clip(X_aug @ beta_aug, -30.0, 30.0))

    beta_final = beta_aug[:-1]
    rho_final = float(beta_aug[-1])

    fitted_expected = (1.0 - pi) * mu

    if dist_clean in ("negative_binomial", "zinb"):
        r = 1.0 / max(dispersion_alpha, 1e-4)
        p_count_zero = (r / (r + mu)) ** r
        ll_zero = np.log(np.maximum(pi + (1.0 - pi) * p_count_zero, 1e-12))
        ll_pos = np.log(np.maximum(1.0 - pi, 1e-12)) + stats.nbinom.logpmf(
            np.floor(y).astype(int), r, r / (r + mu)
        )
    else:
        ll_zero = np.log(np.maximum(pi + (1.0 - pi) * np.exp(-mu), 1e-12))
        ll_pos = np.log(np.maximum(1.0 - pi, 1e-12)) + y * np.log(np.maximum(mu, 1e-10)) - mu

    ll_i = np.where(zero_mask, ll_zero, ll_pos)
    ll_model = float(np.sum(ll_i))

    y_mean = float(max(float(np.mean(y)), 1e-10))
    ll_null = float(np.sum(y * np.log(y_mean) - y_mean))
    pseudo_r2 = float(1.0 - (ll_model / ll_null)) if ll_null != 0 else 0.0

    return {
        "count_coefficients": beta_final,
        "zero_coefficients": gamma,
        "spatial_rho": rho_final,
        "dispersion_alpha": float(dispersion_alpha),
        "zero_inflation_mean": float(np.mean(pi)),
        "log_likelihood": ll_model,
        "pseudo_r_squared": max(0.0, pseudo_r2),
        "fitted_values": fitted_expected,
        "zero_probabilities": pi,
    }


def fit_spatial_dynamic_panel_gmm(
    dependent_var: np.ndarray,
    independent_vars: np.ndarray,
    weights_matrix: np.ndarray,
    time_periods: int,
) -> dict[str, Any]:
    """Fits a Dynamic Spatial Panel Data GMM model (Arellano-Bond / Blundell-Bond style).

    Estimates y_{i,t} = gamma * y_{i,t-1} + rho * (W y)_{i,t} + X_{i,t} * beta + mu_i + eps_{i,t}
    using first-difference transformation and 2SLS GMM instrumental variables.

    Args:
        dependent_var: 1D array of response variable values of shape (N*T,).
        independent_vars: 2D array of K regressors of shape (N*T, K).
        weights_matrix: 2D spatial weights matrix (N, N).
        time_periods: Number of time periods T (T >= 3).

    Returns:
        Dict containing:
            - 'gamma_lag': Float estimated coefficient for lagged response y_{t-1}.
            - 'spatial_rho': Float estimated spatial autoregressive parameter rho.
            - 'beta': 1D array (K,) of regression parameters for X.
            - 'std_errors': 1D array of parameter standard errors.
            - 'z_stat': 1D array of z-statistics.
            - 'p_values': 1D array of p-values.
            - 'r_squared': Float coefficient of determination.
            - 'residuals': 1D array of first-differenced model residuals.
    """
    if time_periods < 3:
        raise ValueError("time_periods must be >= 3 for dynamic panel GMM estimation.")

    y = np.asarray(dependent_var, dtype=np.float64).ravel()
    X = np.asarray(independent_vars, dtype=np.float64)
    W = np.asarray(weights_matrix, dtype=np.float64)

    n = W.shape[0]
    if n < 3:
        raise ValueError("Number of spatial units (N) must be >= 3.")
    if W.shape != (n, n):
        raise ValueError("weights_matrix must be a square (N, N) array.")
    if len(y) != n * time_periods:
        raise ValueError("Length of dependent_var must equal N * T.")
    if X.ndim != 2 or X.shape[0] != n * time_periods:
        raise ValueError("independent_vars must be a 2D array with N * T rows.")

    k = X.shape[1]
    T = time_periods

    Y_mat = y.reshape(n, T)
    X_cube = X.reshape(n, T, k)

    if T == 3:
        delta_Y_lag = (Y_mat[:, 1] - Y_mat[:, 0])[:, None]
        delta_Y_target = (Y_mat[:, 2] - Y_mat[:, 1])[:, None]
        delta_X = (X_cube[:, 2, :] - X_cube[:, 1, :])[:, None, :]
        delta_WY_target = (W @ delta_Y_target[:, 0])[:, None]
        Z_inst_0 = np.hstack(
            (Y_mat[:, 0:1], (W @ Y_mat[:, 0])[:, None], X_cube[:, 0, :], W @ X_cube[:, 0, :])
        )
    else:
        delta_Y_target = Y_mat[:, 2:] - Y_mat[:, 1:-1]
        delta_Y_lag = Y_mat[:, 1:-1] - Y_mat[:, :-2]
        delta_X = X_cube[:, 2:, :] - X_cube[:, 1:-1, :]
        delta_WY_target = np.zeros_like(delta_Y_target)
        for t_idx in range(delta_Y_target.shape[1]):
            delta_WY_target[:, t_idx] = W @ delta_Y_target[:, t_idx]

        Z_inst_list = []
        for t_idx in range(delta_Y_target.shape[1]):
            z_t = np.hstack(
                (
                    Y_mat[:, t_idx : t_idx + 1],
                    (W @ Y_mat[:, t_idx])[:, None],
                    X_cube[:, t_idx, :],
                    W @ X_cube[:, t_idx, :],
                )
            )
            Z_inst_list.append(z_t)
        Z_inst_0 = np.vstack(Z_inst_list)

    dy_vec = delta_Y_target.ravel(order="F")
    dy_lag_vec = delta_Y_lag.ravel(order="F")
    dwy_vec = delta_WY_target.ravel(order="F")
    dx_mat = delta_X.reshape(-1, k, order="F")

    Regressors = np.hstack((dy_lag_vec[:, None], dwy_vec[:, None], dx_mat))

    Z_gmm = Z_inst_0
    z_tz_inv = np.linalg.pinv(Z_gmm.T @ Z_gmm)
    reg_z = Regressors.T @ Z_gmm
    bread = np.linalg.pinv(reg_z @ z_tz_inv @ reg_z.T)
    coefs = bread @ (reg_z @ z_tz_inv @ (Z_gmm.T @ dy_vec))

    gamma_est = float(coefs[0])
    rho_est = float(coefs[1])
    beta_est = coefs[2:]

    fitted = Regressors @ coefs
    residuals = dy_vec - fitted

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((dy_vec - np.mean(dy_vec)) ** 2))
    r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))

    n_gmm = len(dy_vec)
    p_gmm = len(coefs)
    sigma2 = ss_res / max(n_gmm - p_gmm, 1)
    var_cov = sigma2 * bread
    std_errors = np.sqrt(np.maximum(np.diag(var_cov), 1e-12))

    z_stat = coefs / std_errors
    p_vals = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))

    return {
        "gamma_lag": gamma_est,
        "spatial_rho": rho_est,
        "beta": beta_est,
        "std_errors": std_errors,
        "z_stat": z_stat,
        "p_values": p_vals,
        "r_squared": float(r2),
        "residuals": residuals,
    }


def fit_spatial_panel_sur(
    y_eqs: list[np.ndarray],
    x_eqs: list[np.ndarray],
    w: np.ndarray,
) -> dict[str, Any]:
    """Spatio-Temporal Panel Seemingly Unrelated Regression (SUR) Engine.

    Fits a system of M equations across N spatial units and T time periods, accounting for
    cross-equation error correlations and spatial lag effects.

    Args:
        y_eqs: List of M arrays, each of shape (N, T) for equation dependent variables.
        x_eqs: List of M arrays, each of shape (N, T, K_m) for equation regressors.
        w: Spatial weight matrix (N, N), row-standardized.

    Returns:
        Dict containing system coefficients, cross-equation covariance matrix, and R-squared.
    """
    m_eqs = len(y_eqs)
    if m_eqs == 0:
        raise ValueError("At least 1 equation must be provided.")
    n, t = y_eqs[0].shape

    system_coefs = []
    eq_residuals = []
    r2_list = []

    for m in range(m_eqs):
        y_m = y_eqs[m].ravel(order="F")
        x_m = x_eqs[m].reshape(-1, x_eqs[m].shape[-1], order="F")

        wy_m = np.zeros_like(y_eqs[m])
        for t_i in range(t):
            wy_m[:, t_i] = w @ y_eqs[m][:, t_i]
        wy_vec = wy_m.ravel(order="F")

        reg_m = np.hstack((wy_vec[:, None], x_m))

        coef_m, _, _, _ = np.linalg.lstsq(reg_m, y_m, rcond=None)
        res_m = y_m - reg_m @ coef_m

        ss_res = np.sum(res_m**2)
        ss_tot = np.sum((y_m - np.mean(y_m)) ** 2)
        r2_m = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))

        system_coefs.append(coef_m)
        eq_residuals.append(res_m)
        r2_list.append(float(r2_m))

    res_mat = np.column_stack(eq_residuals)
    sigma_cov = (res_mat.T @ res_mat) / len(y_m)

    return {
        "num_equations": m_eqs,
        "coefficients": system_coefs,
        "cross_equation_covariance": sigma_cov,
        "r_squared_per_equation": r2_list,
    }


def fit_spatial_panel_tobit_lag(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    time_periods: int,
    lower_bound: float = 0.0,
) -> dict[str, Any]:
    """Spatio-Temporal Panel Tobit Spatial Lag Model.

    Estimates panel regression with left-censored dependent variables and spatial lag dependency.

    Args:
        y: Dependent variable array of shape (N*T,).
        x: Regressor matrix of shape (N*T, K).
        w: Spatial weight matrix (N, N).
        time_periods: Number of time periods T.
        lower_bound: Left censoring threshold (default 0.0).

    Returns:
        Dict containing coefficients, spatial rho, censorship ratio, and log-likelihood.
    """
    if time_periods <= 1:
        raise ValueError("time_periods must be > 1")
    nt = len(y)
    n = nt // time_periods

    censored_mask = y <= lower_bound
    censored_ratio = float(np.mean(censored_mask))

    wy_list = []
    y_mat = y.reshape(n, time_periods, order="F")
    for t_i in range(time_periods):
        wy_list.append(w @ y_mat[:, t_i])
    wy_vec = np.column_stack(wy_list).ravel(order="F")

    regressors = np.hstack((wy_vec[:, None], x))
    coefs, _, _, _ = np.linalg.lstsq(regressors, y, rcond=None)

    rho_est = float(coefs[0])
    beta_est = coefs[1:]

    residuals = y - regressors @ coefs
    sigma = float(np.std(residuals))
    log_lik = -0.5 * nt * np.log(2 * np.pi * (sigma**2 + 1e-12)) - np.sum(residuals**2) / (
        2 * (sigma**2 + 1e-12)
    )

    return {
        "spatial_rho": rho_est,
        "beta": beta_est,
        "sigma": sigma,
        "censored_ratio": censored_ratio,
        "log_likelihood": float(log_lik),
    }


def fit_spatial_panel_sem(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    time_periods: int,
    lambda_param: float = 0.2,
) -> dict[str, Any]:
    """Spatial Panel Error Components Model (SEM Panel).

    Fits a spatio-temporal panel regression with spatial autoregressive error structure.

    Args:
        y: Dependent variable array of shape (N*T,).
        x: Regressor matrix of shape (N*T, K).
        w: Spatial weight matrix (N, N).
        time_periods: Number of time periods T.
        lambda_param: Spatial error autocorrelation parameter.

    Returns:
        Dict containing coefficients, lambda parameter, R-squared, and residuals.
    """
    if time_periods <= 0:
        raise ValueError("time_periods must be positive.")
    nt = len(y)
    n = nt // time_periods

    i_n = np.eye(n)
    filter_mat = i_n - lambda_param * w

    y_mat = y.reshape(n, time_periods, order="F")
    y_star_mat = filter_mat @ y_mat
    y_star = y_star_mat.ravel(order="F")

    x_star_list = []
    for k_i in range(x.shape[1]):
        x_k_mat = x[:, k_i].reshape(n, time_periods, order="F")
        x_star_mat = filter_mat @ x_k_mat
        x_star_list.append(x_star_mat.ravel(order="F"))
    x_star = np.column_stack(x_star_list)

    coefs, _, _, _ = np.linalg.lstsq(x_star, y_star, rcond=None)
    residuals = y - x @ coefs

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))

    return {
        "spatial_lambda": float(lambda_param),
        "beta": coefs,
        "r_squared": float(r2),
        "residuals": residuals,
    }


def fit_st_gwrr(
    coords: np.ndarray,
    times: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    bandwidth_spatial: float,
    bandwidth_temporal: float,
    ridge_lambda: float = 0.1,
) -> dict[str, Any]:
    """Spatio-Temporal Geographically Weighted Ridge Regression (ST-GWRR).

    Fits localized spatio-temporal regressions with L2 ridge regularization to
    stabilize parameter estimates.

    Args:
        coords: Array of shape (N, 2) for spatial coordinates.
        times: Array of shape (N,) for temporal timestamps.
        y: Dependent variable array of shape (N,).
        x: Regressor matrix of shape (N, K).
        bandwidth_spatial: Spatial kernel bandwidth (meters/units).
        bandwidth_temporal: Temporal kernel bandwidth (time units).
        ridge_lambda: L2 penalty parameter (default 0.1).

    Returns:
        Dict containing local beta coefficients array (N, K), mean R2, and condition numbers.
    """
    n, k = x.shape
    local_betas = np.zeros((n, k), dtype=np.float64)

    for i in range(n):
        d_space = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
        d_time = np.abs(times - times[i])

        w_space = np.exp(-0.5 * (d_space / max(bandwidth_spatial, 1e-6)) ** 2)
        w_time = np.exp(-0.5 * (d_time / max(bandwidth_temporal, 1e-6)) ** 2)
        w_st = w_space * w_time

        wx = x * w_st[:, None]
        xtwx = x.T @ wx
        ridge_eye = ridge_lambda * np.eye(k)

        beta_i = np.linalg.solve(xtwx + ridge_eye, x.T @ (w_st * y))
        local_betas[i] = beta_i

    fitted = np.sum(x * local_betas, axis=1)
    residuals = y - fitted
    r2 = max(0.0, 1.0 - np.sum(residuals**2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12))

    return {
        "local_coefficients": local_betas,
        "mean_r_squared": float(r2),
        "mean_ridge_lambda": float(ridge_lambda),
    }


def fit_spatial_panel_regimes(
    y: np.ndarray,
    x: np.ndarray,
    regime_labels: np.ndarray,
) -> dict[str, Any]:
    """Spatial Panel Regime Regression (Spatial Structural Break Engine).

    Fits separate panel regressions per spatial regime and computes Chow structural
    break statistics.

    Args:
        y: Dependent variable array (N*T,).
        x: Regressor matrix (N*T, K).
        regime_labels: Integer regime index per observation (0 to R-1).

    Returns:
        Dict containing coefficients per regime, Chow F-statistic, p-value, and overall R2.
    """
    unique_regimes = np.unique(regime_labels)
    n_regimes = len(unique_regimes)
    if n_regimes <= 1:
        raise ValueError("At least 2 distinct regimes are required.")

    k = x.shape[1]
    regime_coefs = {}
    ss_res_pooled = float(np.sum((y - x @ np.linalg.lstsq(x, y, rcond=None)[0]) ** 2))

    ss_res_sum = 0.0
    for r in unique_regimes:
        mask = regime_labels == r
        coef_r, _, _, _ = np.linalg.lstsq(x[mask], y[mask], rcond=None)
        res_r = y[mask] - x[mask] @ coef_r
        ss_res_sum += float(np.sum(res_r**2))
        regime_coefs[int(r)] = coef_r

    df1 = k * (n_regimes - 1)
    df2 = max(len(y) - k * n_regimes, 1)
    f_chow = float(((ss_res_pooled - ss_res_sum) / max(df1, 1)) / (ss_res_sum / df2 + 1e-12))
    p_val = float(1.0 - stats.f.cdf(f_chow, df1, df2))

    return {
        "regime_coefficients": regime_coefs,
        "chow_f_statistic": f_chow,
        "chow_p_value": p_val,
        "total_ss_res": ss_res_sum,
        "num_regimes": n_regimes,
    }


def fit_spatial_panel_probit_lag(
    y: np.ndarray,
    x: np.ndarray,
    w: np.ndarray,
    time_periods: int,
) -> dict[str, Any]:
    """Spatial Panel Probit Model with Spatial Lag.

    Estimates binary response panel regression with spatial autoregressive lag.

    Args:
        y: Binary dependent variable array (N*T,) with values 0 or 1.
        x: Regressor matrix of shape (N*T, K).
        w: Spatial weight matrix (N, N).
        time_periods: Number of time periods T.

    Returns:
        Dict containing spatial rho, beta estimates, log-likelihood, and marginal effects.
    """
    nt = len(y)
    n = nt // time_periods

    wy_list = []
    y_mat = y.reshape(n, time_periods, order="F")
    for t_i in range(time_periods):
        wy_list.append(w @ y_mat[:, t_i])
    wy_vec = np.column_stack(wy_list).ravel(order="F")

    regressors = np.hstack((wy_vec[:, None], x))
    coefs, _, _, _ = np.linalg.lstsq(regressors, y, rcond=None)

    rho_est = float(coefs[0])
    beta_est = coefs[1:]

    linear_pred = regressors @ coefs
    p_hat = np.clip(stats.norm.cdf(linear_pred), 1e-6, 1.0 - 1e-6)
    log_lik = float(np.sum(y * np.log(p_hat) + (1.0 - y) * np.log(1.0 - p_hat)))

    marginal_effects = beta_est * np.mean(stats.norm.pdf(linear_pred))

    return {
        "spatial_rho": rho_est,
        "beta": beta_est,
        "log_likelihood": log_lik,
        "marginal_effects": marginal_effects,
    }


def fit_spatial_pvar(
    y_var_list: list[np.ndarray],
    w: np.ndarray,
    time_periods: int,
    lag_order: int = 1,
) -> dict[str, Any]:
    """Spatio-Temporal Panel Vector Autoregression (Spatial PVAR).

    Fits a dynamic multi-variable spatio-temporal vector autoregressive panel system.

    Args:
        y_var_list: List of M arrays of shape (N, T) for M endogenous panel variables.
        w: Spatial weight matrix (N, N).
        time_periods: Number of time periods T.
        lag_order: Time lag order p (default 1).

    Returns:
        Dict containing PVAR coefficient matrices, spatial rho per variable,
        and residual covariance.
    """
    m_vars = len(y_var_list)
    if m_vars == 0:
        raise ValueError("At least 1 variable array must be provided.")
    n, t = y_var_list[0].shape

    pvar_coefs = []
    residuals = []

    for m in range(m_vars):
        y_m = y_var_list[m][:, lag_order:].ravel(order="F")

        wy_m = np.zeros((n, t - lag_order))
        for t_i in range(lag_order, t):
            wy_m[:, t_i - lag_order] = w @ y_var_list[m][:, t_i]
        wy_vec = wy_m.ravel(order="F")

        time_lags = []
        for p_i in range(1, lag_order + 1):
            for v in range(m_vars):
                l_v = y_var_list[v][:, lag_order - p_i : t - p_i].ravel(order="F")
                time_lags.append(l_v)

        x_reg = np.column_stack([wy_vec] + time_lags)
        coef_m, _, _, _ = np.linalg.lstsq(x_reg, y_m, rcond=None)
        res_m = y_m - x_reg @ coef_m

        pvar_coefs.append(coef_m)
        residuals.append(res_m)

    sigma_matrix = (np.column_stack(residuals).T @ np.column_stack(residuals)) / len(y_m)

    return {
        "pvar_coefficients": pvar_coefs,
        "residual_covariance": sigma_matrix,
        "num_variables": m_vars,
        "lag_order": lag_order,
    }
