# -*- coding: utf-8 -*-
"""Multi-Criteria Decision Analysis (MCDA) mathematical engines."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


def normalize_array(
    arr: np.ndarray,
    method: str,
    low: float = 0.0,
    high: float = 100.0,
    mid: float = 50.0,
    spread: float = 10.0,
    nodata: Optional[float] = None,
) -> np.ndarray:
    """Normalizes an input array to the range [0, 100] using specified criteria.

    Args:
        arr: Input array.
        method: One of 'benefit_minmax', 'cost_minmax', 'benefit_sigmoid',
                'cost_sigmoid', 'benefit_gaussian'.
        low: Lower bound for Min-Max normalization.
        high: Upper bound for Min-Max normalization.
        mid: Midpoint for Sigmoid or Gaussian normalization.
        spread: Spread (>0) for Sigmoid or Gaussian normalization.
        nodata: Nodata value to exclude from normalization and preserve in output.

    Returns:
        Normalized array in the range [0.0, 100.0] (with nodata preserved).
    """
    x = arr.astype(np.float32)
    valid = np.isfinite(x)
    if nodata is not None:
        valid &= x != nodata

    if high <= low:
        raise ValueError("high must be greater than low")
    if spread <= 0:
        raise ValueError("spread must be greater than 0")

    norm = np.zeros_like(x, dtype=np.float32)
    method_lower = method.lower().replace(" ", "_").replace("-", "_")

    if method_lower in ("benefit_minmax", "minmax_benefit", "fuzzy_linear_benefit"):
        norm = (x - low) / (high - low)
    elif method_lower in ("cost_minmax", "minmax_cost", "fuzzy_linear_cost"):
        norm = (high - x) / (high - low)
    elif method_lower in ("benefit_sigmoid", "sigmoid_benefit"):
        z = np.clip((x - mid) / spread, -60.0, 60.0)
        norm = 1.0 / (1.0 + np.exp(-z))
    elif method_lower in ("cost_sigmoid", "sigmoid_cost"):
        z = np.clip((x - mid) / spread, -60.0, 60.0)
        norm = 1.0 - (1.0 / (1.0 + np.exp(-z)))
    elif method_lower in ("benefit_gaussian", "gaussian_benefit"):
        norm = np.exp(-0.5 * ((x - mid) / spread) ** 2)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    norm = np.clip(norm, 0.0, 1.0) * 100.0
    if nodata is not None:
        norm[~valid] = nodata
    else:
        norm[~valid] = np.nan
    return norm


def weighted_linear_combination(
    criteria_arrays: List[np.ndarray],
    weights: Union[List[float], np.ndarray],
    constraint_array: Optional[np.ndarray] = None,
    nodata: float = -9999.0,
    criteria_nodatas: Optional[List[Optional[float]]] = None,
) -> np.ndarray:
    """Computes Weighted Linear Combination (WLC) on a stack of normalized arrays.

    Args:
        criteria_arrays: List of NumPy arrays (each representing a criterion, e.g. 0-100).
        weights: List or array of weights summing to 1.0. Length must match criteria_arrays.
        constraint_array: Optional binary array (0/1 or False/True) acting as constraint.
        nodata: Output nodata value to write for invalid pixels.
        criteria_nodatas: Optional list of nodata values for each criterion layer.

    Returns:
        NumPy array containing the suitability score in range [0, 100] (or nodata).
    """
    if not criteria_arrays:
        raise ValueError("At least one criterion array must be provided.")

    n = len(criteria_arrays)
    if len(weights) != n:
        raise ValueError(
            f"Number of weights ({len(weights)}) does not match number of criteria ({n})"
        )

    weights = np.asarray(weights, dtype=np.float32)
    # Re-normalize weights if they don't sum to 1.0 (with tolerance)
    w_sum = float(np.sum(weights))
    if not np.isclose(w_sum, 1.0) and w_sum > 0:
        weights = weights / w_sum

    shape = criteria_arrays[0].shape
    valid = np.ones(shape, dtype=bool)

    # Process criteria nodatas
    clipped_arrs = []
    for i, a in enumerate(criteria_arrays):
        if a.shape != shape:
            raise ValueError("All criteria arrays must have identical shapes.")

        a_float = a.astype(np.float32)
        # Handle nan/inf
        m = np.isfinite(a_float)

        # Handle custom nodata value
        if criteria_nodatas is not None and i < len(criteria_nodatas):
            nd = criteria_nodatas[i]
            if nd is not None:
                m &= a_float != nd

        valid &= m
        clipped_arrs.append(np.clip(a_float, 0.0, 100.0))

    stack = np.stack(clipped_arrs, axis=0)
    # Multiply by weights using broadcasting
    result = np.sum(stack * weights[:, None, None], axis=0).astype(np.float32)
    result = np.clip(result, 0.0, 100.0)

    # Apply constraints
    if constraint_array is not None:
        if constraint_array.shape != shape:
            raise ValueError("Constraint array shape must match criteria grid dimensions.")
        c_val = constraint_array.astype(np.float32)
        c_valid = np.isfinite(c_val)
        pass_mask = (c_val > 0.5) & c_valid
        result = np.where(pass_mask, result, 0.0)
        valid &= c_valid

    # Set invalid cells to output nodata
    result = np.where(valid, result, nodata).astype(np.float32)
    return result


def topsis_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    benefit_criteria: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates suitability ranking scores using the TOPSIS method.

    TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
    ranks alternatives by their relative closeness to the ideal best and worst solutions.

    Args:
        decision_matrix: NumPy array of shape (M, N) for M alternatives and N criteria.
        weights: 1D array of shape (N,) representing importance of each criterion.
        benefit_criteria: 1D boolean array of shape (N,) where True represents a benefit
            criterion (higher is better) and False represents a cost criterion (lower is better).

    Returns:
        A tuple of:
          - scores: 1D NumPy array of shape (M,) containing similarity scores in range [0.0, 1.0].
            Higher score indicates a better alternative.
          - ranks: 1D NumPy array of shape (M,) containing integer ranks (1 = best, M = worst).
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    is_benefit = np.asarray(benefit_criteria, dtype=bool)

    m, n = X.shape
    if w.shape != (n,):
        raise ValueError(f"weights length ({w.shape[0]}) must match number of criteria ({n})")
    if is_benefit.shape != (n,):
        raise ValueError(
            f"benefit_criteria length ({is_benefit.shape[0]}) must match number of criteria ({n})"
        )

    # Step 1: Normalize decision matrix using vector normalization
    norm_denom = np.sqrt(np.sum(X**2, axis=0))
    norm_denom = np.where(norm_denom > 0, norm_denom, 1e-9)
    R = X / norm_denom[None, :]

    # Step 2: Weighted normalized decision matrix
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum
    V = R * w[None, :]

    # Step 3: Determine ideal best and worst solutions
    v_best = np.zeros(n)
    v_worst = np.zeros(n)

    for j in range(n):
        col = V[:, j]
        if is_benefit[j]:
            v_best[j] = np.max(col)
            v_worst[j] = np.min(col)
        else:
            v_best[j] = np.min(col)
            v_worst[j] = np.max(col)

    # Step 4: Calculate separation measures
    S_best = np.sqrt(np.sum((V - v_best[None, :]) ** 2, axis=1))
    S_worst = np.sqrt(np.sum((V - v_worst[None, :]) ** 2, axis=1))

    # Step 5: Relative closeness to ideal solution
    denom = S_best + S_worst
    with np.errstate(divide="ignore", invalid="ignore"):
        C = S_worst / np.where(denom > 0, denom, 1e-9)
        C = np.where(denom > 0, C, 0.5)

    # Step 6: Rank the alternatives (1-indexed)
    ranks = np.argsort(-C)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, m + 1)

    return C, rank_order


def vikor_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    benefit_criteria: np.ndarray,
    v: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates compromise ranking scores using the VIKOR method.

    VIKOR determines a compromise ranking list and compromise solution for conflicting criteria.
    Lower score represents a better alternative (closer to ideal solution).

    Args:
        decision_matrix: NumPy array of shape (M, N) containing M alternatives and N criteria.
        weights: 1D array of shape (N,) representing importance of each criterion.
        benefit_criteria: 1D boolean array of shape (N,) where True represents a benefit
            criterion and False represents a cost criterion.
        v: Weight of the strategy of "majority of criteria" (usually 0.5).

    Returns:
        A tuple of:
          - scores: 1D NumPy array of shape (M,) containing Q_i compromise
            index values (lower is better).
          - ranks: 1D NumPy array of shape (M,) containing integer ranks
            (1 = best, M = worst).
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    is_benefit = np.asarray(benefit_criteria, dtype=bool)

    m, n = X.shape
    if w.shape != (n,):
        raise ValueError(f"weights length ({w.shape[0]}) must match number of criteria ({n})")
    if is_benefit.shape != (n,):
        raise ValueError(
            f"benefit_criteria length ({is_benefit.shape[0]}) must match number of criteria ({n})"
        )

    # Normalize weights to sum to 1.0 if not already
    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum

    # Step 1: Find best and worst for each criterion
    f_best = np.zeros(n)
    f_worst = np.zeros(n)

    for j in range(n):
        col = X[:, j]
        if is_benefit[j]:
            f_best[j] = np.max(col)
            f_worst[j] = np.min(col)
        else:
            f_best[j] = np.min(col)
            f_worst[j] = np.max(col)

    # Step 2: Compute S_i (utility) and R_i (regret)
    S = np.zeros(m)
    R = np.zeros(m)

    diff = f_best - f_worst
    diff = np.where(diff > 0, diff, 1e-9)

    for i in range(m):
        term = w * (f_best - X[i, :]) / diff
        S[i] = np.sum(term)
        R[i] = np.max(term)

    # Step 3: Compute Q_i
    S_star, S_minus = np.min(S), np.max(S)
    R_star, R_minus = np.min(R), np.max(R)

    S_range = S_minus - S_star
    R_range = R_minus - R_star

    with np.errstate(divide="ignore", invalid="ignore"):
        term_S = (S - S_star) / np.where(S_range > 0, S_range, 1e-9)
        term_S = np.where(S_range > 0, term_S, 0.0)

        term_R = (R - R_star) / np.where(R_range > 0, R_range, 1e-9)
        term_R = np.where(R_range > 0, term_R, 0.0)

    Q = v * term_S + (1.0 - v) * term_R

    # Step 4: Rank by Q (ascending, 1-indexed)
    ranks = np.argsort(Q)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, m + 1)

    return Q, rank_order
