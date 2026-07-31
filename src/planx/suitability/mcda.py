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


def promethee_ii_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    benefit_criteria: np.ndarray,
    preference_thresholds: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates suitability ranking scores using the PROMETHEE II method.

    PROMETHEE II (Preference Ranking Organization METHod for Enrichment Evaluations)
    ranks alternatives based on net outranking flows (leaving flow minus entering flow).

    Args:
        decision_matrix: NumPy array of shape (M, N) for M alternatives and N criteria.
        weights: 1D array of shape (N,) representing importance of each criterion.
        benefit_criteria: 1D boolean array of shape (N,) where True represents a benefit
            criterion and False represents a cost criterion.
        preference_thresholds: Optional 1D array of shape (N,) representing preference threshold p_j
            for each criterion. If None, linear preference over criterion range is used.

    Returns:
        A tuple of:
          - net_flows: 1D NumPy array of shape (M,) containing net outranking flows in [-1.0, 1.0].
            Higher score indicates a better alternative.
          - ranks: 1D NumPy array of shape (M,) containing integer ranks (1 = best, M = worst).
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    is_benefit = np.asarray(benefit_criteria, dtype=bool)

    m, n = X.shape
    if m < 2:
        return np.zeros(m, dtype=np.float64), np.ones(m, dtype=int)

    if w.shape != (n,):
        raise ValueError(f"weights length ({w.shape[0]}) must match number of criteria ({n})")
    if is_benefit.shape != (n,):
        raise ValueError(
            f"benefit_criteria length ({is_benefit.shape[0]}) must match number of criteria ({n})"
        )

    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum

    if preference_thresholds is not None:
        p_thresh = np.asarray(preference_thresholds, dtype=np.float64)
        if p_thresh.shape != (n,):
            raise ValueError(
                f"preference_thresholds length ({p_thresh.shape[0]}) must match criteria ({n})"
            )
    else:
        ranges = np.max(X, axis=0) - np.min(X, axis=0)
        p_thresh = np.where(ranges > 0, ranges, 1.0)

    p_matrix = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        for k in range(m):
            if i == k:
                continue
            pair_pref = 0.0
            for j in range(n):
                diff = (X[i, j] - X[k, j]) if is_benefit[j] else (X[k, j] - X[i, j])
                if diff > 0:
                    p_val = min(1.0, diff / p_thresh[j]) if p_thresh[j] > 0 else 1.0
                    pair_pref += w[j] * p_val
            p_matrix[i, k] = pair_pref

    leaving_flow = np.sum(p_matrix, axis=1) / (m - 1)
    entering_flow = np.sum(p_matrix, axis=0) / (m - 1)
    net_flow = leaving_flow - entering_flow

    ranks = np.argsort(-net_flow)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, m + 1)

    return net_flow, rank_order


def electre_i_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    benefit_criteria: np.ndarray,
    concordance_threshold: float = 0.7,
    discordance_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Calculates outranking relation using the ELECTRE I method.

    ELECTRE I evaluates concordance and discordance matrices to find a kernel subset
    of non-dominated alternatives.

    Args:
        decision_matrix: NumPy array of shape (M, N) for M alternatives and N criteria.
        weights: 1D array of shape (N,) representing importance of each criterion.
        benefit_criteria: 1D boolean array of shape (N,) for criterion optimization direction.
        concordance_threshold: Minimum concordance required for outranking (default 0.7).
        discordance_threshold: Maximum discordance allowed for outranking (default 0.3).

    Returns:
        A tuple of:
          - concordance_matrix: (M, M) NumPy array of pairwise concordance indices C(a, b).
          - discordance_matrix: (M, M) NumPy array of pairwise discordance indices D(a, b).
          - non_dominated_kernel: List of alternative indices belonging to the non-dominated kernel.
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    is_benefit = np.asarray(benefit_criteria, dtype=bool)

    m, n = X.shape
    if m == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), []
    if w.shape != (n,):
        raise ValueError(f"weights length ({w.shape[0]}) must match criteria count ({n})")
    if is_benefit.shape != (n,):
        raise ValueError(
            f"benefit_criteria length ({is_benefit.shape[0]}) must match criteria ({n})"
        )

    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum

    ranges = np.max(X, axis=0) - np.min(X, axis=0)
    ranges = np.where(ranges > 0, ranges, 1.0)

    C = np.zeros((m, m), dtype=np.float64)
    D = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        for k in range(m):
            if i == k:
                continue
            c_weight = 0.0
            max_disc = 0.0
            for j in range(n):
                diff = (X[i, j] - X[k, j]) if is_benefit[j] else (X[k, j] - X[i, j])
                if diff >= 0:
                    c_weight += w[j]
                else:
                    disc_val = abs(diff) / ranges[j]
                    if disc_val > max_disc:
                        max_disc = disc_val
            C[i, k] = c_weight
            D[i, k] = max_disc

    outranks = (C >= concordance_threshold) & (D <= discordance_threshold)
    np.fill_diagonal(outranks, False)

    dominated = set()
    for i in range(m):
        for k in range(m):
            if i != k and outranks[i, k]:
                dominated.add(k)

    non_dominated = [idx for idx in range(m) if idx not in dominated]
    return C, D, non_dominated


def electre_iii_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    benefit_criteria: np.ndarray,
    q_thresholds: np.ndarray,
    p_thresholds: np.ndarray,
    v_thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates outranking ranking using the ELECTRE III method with pseudo-criteria thresholds.

    Args:
        decision_matrix: NumPy array of shape (M, N) for M alternatives and N criteria.
        weights: 1D array of shape (N,) representing importance of each criterion.
        benefit_criteria: 1D boolean array of shape (N,).
        q_thresholds: 1D array of shape (N,) representing indifference thresholds q_j.
        p_thresholds: 1D array of shape (N,) representing preference thresholds p_j.
        v_thresholds: 1D array of shape (N,) representing veto thresholds v_j.

    Returns:
        A tuple of:
          - credibility_matrix: (M, M) NumPy array of fuzzy credibility outranking indices S(a, b).
          - ranks: 1D NumPy array of shape (M,) containing integer ranks (1 = best, M = worst).
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    is_benefit = np.asarray(benefit_criteria, dtype=bool)
    q = np.asarray(q_thresholds, dtype=np.float64)
    p = np.asarray(p_thresholds, dtype=np.float64)
    v = np.asarray(v_thresholds, dtype=np.float64)

    m, n = X.shape
    if m < 2:
        return np.ones((m, m), dtype=np.float64), np.ones(m, dtype=int)
    if (
        w.shape != (n,)
        or is_benefit.shape != (n,)
        or q.shape != (n,)
        or p.shape != (n,)
        or v.shape != (n,)
    ):
        raise ValueError(
            "All threshold vectors and weights must have length equal to criteria count."
        )

    w_sum = np.sum(w)
    if w_sum > 0:
        w = w / w_sum

    C = np.zeros((m, m), dtype=np.float64)
    S = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        for k in range(m):
            if i == k:
                C[i, k] = 1.0
                S[i, k] = 1.0
                continue

            c_sum = 0.0
            d_factors = []
            for j in range(n):
                diff = (X[k, j] - X[i, j]) if is_benefit[j] else (X[i, j] - X[k, j])
                if diff <= q[j]:
                    c_j = 1.0
                elif diff >= p[j]:
                    c_j = 0.0
                else:
                    c_j = (p[j] - diff) / max(1e-9, p[j] - q[j])
                c_sum += w[j] * c_j

                if diff <= p[j]:
                    d_j = 0.0
                elif diff >= v[j]:
                    d_j = 1.0
                else:
                    d_j = (diff - p[j]) / max(1e-9, v[j] - p[j])
                d_factors.append(d_j)

            C[i, k] = c_sum
            cred = c_sum
            for d_j in d_factors:
                if d_j > c_sum:
                    cred *= (1.0 - d_j) / max(1e-9, 1.0 - c_sum)
            S[i, k] = cred

    scores = np.sum(S, axis=1) - np.sum(S, axis=0)
    ranks = np.argsort(-scores)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, m + 1)

    return S, rank_order


def mcda_sensitivity_monte_carlo(
    decision_matrix: np.ndarray,
    base_weights: np.ndarray,
    directions: Optional[np.ndarray] = None,
    noise_level: float = 0.1,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict:
    """Performs Monte Carlo criteria weight sensitivity analysis on MCDA suitability rankings.

    Args:
        decision_matrix: (M, N) NumPy array of alternatives and criteria.
        base_weights: (N,) NumPy array of baseline criterion weights.
        directions: (N,) optional array indicating benefit (1) or cost (-1) criteria.
        noise_level: Standard deviation of Gaussian noise added to weights float.
        n_simulations: Number of Monte Carlo simulation iterations.
        seed: Random seed for reproducibility.

    Returns:
        Dict containing sensitivity statistics:
          - mean_ranks: (M,) average rank of each alternative across simulations.
          - std_ranks: (M,) standard deviation of ranks for each alternative.
          - rank_first_probability: (M,) empirical probability of finishing 1st.
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w_base = np.asarray(base_weights, dtype=np.float64)
    m, n = X.shape

    if len(w_base) != n:
        raise ValueError("base_weights length must match number of columns in decision_matrix.")

    if directions is None:
        dirs = np.ones(n, dtype=np.float64)
    else:
        dirs = np.asarray(directions, dtype=np.float64)

    # Normalize decision matrix
    X_norm = np.zeros_like(X)
    mins, maxs = np.min(X, axis=0), np.max(X, axis=0)
    denom = np.where(maxs - mins <= 1e-12, 1.0, maxs - mins)

    for j in range(n):
        if dirs[j] >= 0:
            X_norm[:, j] = (X[:, j] - mins[j]) / denom[j]
        else:
            X_norm[:, j] = (maxs[j] - X[:, j]) / denom[j]

    rank_records = np.zeros((n_simulations, m), dtype=np.float64)
    np.random.seed(seed)

    for sim in range(n_simulations):
        noise = np.random.normal(0.0, noise_level, size=n)
        w_sim = np.maximum(0.0, w_base + noise)
        w_sum = np.sum(w_sim)
        if w_sum > 0:
            w_sim = w_sim / w_sum
        else:
            w_sim = np.ones(n, dtype=np.float64) / n

        scores = X_norm @ w_sim
        ranks = np.argsort(-scores)
        sim_ranks = np.empty_like(ranks, dtype=np.float64)
        sim_ranks[ranks] = np.arange(1, m + 1, dtype=np.float64)
        rank_records[sim] = sim_ranks

    mean_r = np.mean(rank_records, axis=0)
    std_r = np.std(rank_records, axis=0)
    p_first = np.mean(rank_records == 1.0, axis=0)

    return {
        "mean_ranks": mean_r,
        "std_ranks": std_r,
        "rank_first_probability": p_first,
    }


def marcos_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    directions: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ranks alternatives using the MARCOS method.

    MARCOS = Measurement of Alternatives and Ranking according to COmpromise Solution.

    Args:
        decision_matrix: 2D NumPy array of shape (M, N) for M alternatives and N criteria.
        weights: 1D NumPy array of shape (N,) containing criteria weights.
        directions: 1D array indicating benefit (1) or cost (-1) criteria.

    Returns:
        A tuple of:
          - utility_scores: 1D NumPy array of shape (M,) containing final utility values f(K_i).
          - rank_order: 1D NumPy array of shape (M,) containing alternative ranks (1 = best).
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m, n = X.shape

    if len(w) != n:
        raise ValueError("weights length must match number of columns in decision_matrix.")

    if directions is None:
        dirs = np.ones(n, dtype=np.float64)
    else:
        dirs = np.asarray(directions, dtype=np.float64)

    # Ideal (AAI) and Anti-Ideal (AI) solutions
    ai = np.where(dirs >= 0, np.min(X, axis=0), np.max(X, axis=0))
    aai = np.where(dirs >= 0, np.max(X, axis=0), np.min(X, axis=0))

    # Extended decision matrix X_ext of shape (M + 2, N)
    X_ext = np.vstack([ai[None, :], X, aai[None, :]])

    # Normalization
    N_mat = np.zeros_like(X_ext)
    for j in range(n):
        if dirs[j] >= 0:
            N_mat[:, j] = X_ext[:, j] / max(1e-9, aai[j])
        else:
            N_mat[:, j] = aai[j] / np.maximum(1e-9, X_ext[:, j])

    # Weighted normalized matrix V
    V = N_mat * w

    # Utility degrees K_i^- and K_i^+
    S = np.sum(V, axis=1)
    S_ai = S[0]
    S_aai = S[-1]

    S_alt = S[1:-1]
    K_minus = S_alt / max(1e-9, S_ai)
    K_plus = S_alt / max(1e-9, S_aai)

    # Utility functions f(K_i^-) and f(K_i^+)
    f_k_minus = K_plus / np.maximum(1e-9, K_plus + K_minus)
    f_k_plus = K_minus / np.maximum(1e-9, K_plus + K_minus)

    # Overall utility score f(K_i)
    denom_f_k = (
        1.0
        + (1.0 - f_k_plus) / np.maximum(1e-9, f_k_plus)
        + (1.0 - f_k_minus) / np.maximum(1e-9, f_k_minus)
    )
    f_k = (K_plus + K_minus) / denom_f_k

    ranks = np.argsort(-f_k)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, m + 1)

    return f_k, rank_order


def waspas_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    directions: Optional[np.ndarray] = None,
    lambda_param: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Ranks alternatives using the WASPAS method.

    WASPAS = Weighted Aggregated Sum Product Assessment.

    Args:
        decision_matrix: 2D NumPy array of shape (M, N) for M alternatives and N criteria.
        weights: 1D NumPy array of shape (N,) containing criteria weights.
        directions: 1D array indicating benefit (1) or cost (-1) criteria.
        lambda_param: Joint trade-off parameter float in [0.0, 1.0].

    Returns:
        A tuple of:
          - Q_scores: 1D NumPy array of shape (M,) containing overall WASPAS score Q_i.
          - rank_order: 1D NumPy array of shape (M,) containing alternative ranks (1 = best).
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m, n = X.shape

    if len(w) != n:
        raise ValueError("weights length must match number of columns in decision_matrix.")

    if not (0.0 <= lambda_param <= 1.0):
        raise ValueError("lambda_param must be between 0.0 and 1.0.")

    if directions is None:
        dirs = np.ones(n, dtype=np.float64)
    else:
        dirs = np.asarray(directions, dtype=np.float64)

    # Linear Max/Min Normalization X_norm
    X_norm = np.zeros_like(X)
    for j in range(n):
        if dirs[j] >= 0:
            max_val = max(1e-9, np.max(X[:, j]))
            X_norm[:, j] = X[:, j] / max_val
        else:
            min_val = np.min(X[:, j])
            X_norm[:, j] = min_val / np.maximum(1e-9, X[:, j])

    # 1. Weighted Sum Model Q_1 (WSM)
    Q1 = X_norm @ w

    # 2. Weighted Product Model Q_2 (WPM)
    Q2 = np.prod(X_norm ** w[None, :], axis=1)

    # Joint WASPAS score Q_i
    Q = lambda_param * Q1 + (1.0 - lambda_param) * Q2

    ranks = np.argsort(-Q)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, m + 1)

    return Q, rank_order


def aras_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    directions: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ranks alternatives using the ARAS method.

    ARAS = Additive Ratio Assessment.

    Args:
        decision_matrix: 2D NumPy array of shape (M, N) for M alternatives and N criteria.
        weights: 1D NumPy array of shape (N,) containing criteria weights.
        directions: 1D array indicating benefit (1) or cost (-1) criteria.

    Returns:
        A tuple of:
          - K_scores: 1D NumPy array of shape (M,) containing utility degree K_i.
          - rank_order: 1D NumPy array of shape (M,) containing alternative ranks (1 = best).
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m, n = X.shape

    if len(w) != n:
        raise ValueError("weights length must match number of columns in decision_matrix.")

    if directions is None:
        dirs = np.ones(n, dtype=np.float64)
    else:
        dirs = np.asarray(directions, dtype=np.float64)

    # 1. Optimal alternative X_0
    x0 = np.where(dirs >= 0, np.max(X, axis=0), np.min(X, axis=0))

    # Extended matrix X_ext
    X_ext = np.vstack([x0[None, :], X])

    # 2. Normalization
    N_mat = np.zeros_like(X_ext)
    for j in range(n):
        if dirs[j] >= 0:
            sum_col = max(1e-9, np.sum(X_ext[:, j]))
            N_mat[:, j] = X_ext[:, j] / sum_col
        else:
            reciprocal = 1.0 / np.maximum(1e-9, X_ext[:, j])
            sum_recip = max(1e-9, np.sum(reciprocal))
            N_mat[:, j] = reciprocal / sum_recip

    # 3. Weighted matrix V
    V = N_mat * w

    # 4. Optimality function S_i and utility degree K_i
    S = np.sum(V, axis=1)
    S0 = S[0]
    S_alt = S[1:]

    K = S_alt / max(1e-9, S0)

    ranks = np.argsort(-K)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, m + 1)

    return K, rank_order


def copras_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    directions: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ranks alternatives using the COPRAS method.

    COPRAS = Complex Proportional Assessment.

    Args:
        decision_matrix: 2D NumPy array of shape (M, N) for M alternatives and N criteria.
        weights: 1D NumPy array of shape (N,) containing criteria weights.
        directions: 1D array indicating benefit (1) or cost (-1) criteria.

    Returns:
        A tuple of:
          - N_utility_degrees: 1D NumPy array of shape (M,) containing percentage utility values.
          - rank_order: 1D NumPy array of shape (M,) containing alternative ranks (1 = best).
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m, n = X.shape

    if len(w) != n:
        raise ValueError("weights length must match number of columns in decision_matrix.")

    if directions is None:
        dirs = np.ones(n, dtype=np.float64)
    else:
        dirs = np.asarray(directions, dtype=np.float64)

    # 1. Linear Sum Normalization d_ij
    sum_cols = np.maximum(1e-9, np.sum(X, axis=0))
    D_mat = X / sum_cols[None, :]

    # 2. Weighted normalized matrix D_w
    D_w = D_mat * w[None, :]

    # 3. Sum of beneficial S_{+i} and non-beneficial S_{-i} criteria
    benefit_mask = dirs >= 0
    cost_mask = dirs < 0

    if np.any(benefit_mask):
        S_plus = np.sum(D_w[:, benefit_mask], axis=1)
    else:
        S_plus = np.zeros(m, dtype=np.float64)

    if np.any(cost_mask):
        S_minus = np.sum(D_w[:, cost_mask], axis=1)
    else:
        S_minus = np.zeros(m, dtype=np.float64)

    # 4. Relative significance Q_i
    if np.any(cost_mask):
        sum_recip_cost = float(np.sum(1.0 / np.maximum(1e-9, S_minus)))
        Q = S_plus + (np.sum(S_minus) / (S_minus * sum_recip_cost + 1e-12))
    else:
        Q = S_plus

    # 5. Utility degree N_i (%)
    max_Q = max(1e-9, np.max(Q))
    N_utility = (Q / max_Q) * 100.0

    ranks = np.argsort(-N_utility)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, m + 1)

    return N_utility, rank_order


def edas_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    directions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Ranks alternatives using the EDAS method.

    EDAS = Evaluation Based on Distance from Average Solution.

    Args:
        decision_matrix: 2D NumPy array of shape (M, N) for M alternatives and N criteria.
        weights: 1D NumPy array of shape (N,) containing criteria weights.
        directions: 1D array indicating benefit (1) or cost (-1) criteria.

    Returns:
        A tuple of:
          - appraisal_scores: 1D NumPy array of shape (M,) containing normalized AS values.
          - rank_order: 1D NumPy array of shape (M,) containing alternative ranks (1 = best).
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    m, n = X.shape

    if len(w) != n:
        raise ValueError("weights length must match number of columns in decision_matrix.")

    if directions is None:
        dirs = np.ones(n, dtype=np.float64)
    else:
        dirs = np.asarray(directions, dtype=np.float64)

    # 1. Average solution across alternatives
    avg = np.mean(X, axis=0)

    # 2. Positive Distance from Average (PDA) and Negative Distance from Average (NDA)
    pda = np.zeros((m, n), dtype=np.float64)
    nda = np.zeros((m, n), dtype=np.float64)

    for j in range(n):
        avg_j = max(avg[j], 1e-12)
        if dirs[j] >= 0:  # benefit
            pda[:, j] = np.maximum(0.0, X[:, j] - avg[j]) / avg_j
            nda[:, j] = np.maximum(0.0, avg[j] - X[:, j]) / avg_j
        else:  # cost
            pda[:, j] = np.maximum(0.0, avg[j] - X[:, j]) / avg_j
            nda[:, j] = np.maximum(0.0, X[:, j] - avg[j]) / avg_j

    # 3. Weighted sum of PDA and NDA
    sp = np.sum(w[None, :] * pda, axis=1)
    sn = np.sum(w[None, :] * nda, axis=1)

    # 4. Normalize SP and SN
    max_sp = max(np.max(sp), 1e-12)
    max_sn = max(np.max(sn), 1e-12)
    nsp = sp / max_sp
    nsn = 1.0 - (sn / max_sn)

    # 5. Appraisal Score
    appraisal = 0.5 * (nsp + nsn)

    ranks = np.argsort(-appraisal)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, m + 1)

    return appraisal, rank_order


def fuzzy_topsis_method(
    decision_matrix_l: np.ndarray,
    decision_matrix_m: np.ndarray,
    decision_matrix_u: np.ndarray,
    weights_l: np.ndarray,
    weights_m: np.ndarray,
    weights_u: np.ndarray,
    criteria_types: np.ndarray,
) -> dict[str, np.ndarray]:
    """Calculates suitability ranking scores using the Fuzzy TOPSIS method.

    Fuzzy TOPSIS handles uncertainty in decision making by using Triangular Fuzzy Numbers (TFN).

    Args:
        decision_matrix_l: (M, N) array of lower bounds of TFNs for M alternatives, N criteria.
        decision_matrix_m: (M, N) array of middle bounds.
        decision_matrix_u: (M, N) array of upper bounds.
        weights_l: (N,) array of lower bounds for criteria weights.
        weights_m: (N,) array of middle bounds.
        weights_u: (N,) array of upper bounds.
        criteria_types: (N,) array indicating benefit (1) or cost (-1) criteria.

    Returns:
        Dict containing:
            - closeness_coefficients: (M,) array of CC values
            - ranking: (M,) array of 1-based ranks (1=best)
            - distance_positive: (M,) distances to FPIS
            - distance_negative: (M,) distances to FNIS
            - weighted_matrix_l: (M, N) weighted lower bounds
            - weighted_matrix_m: (M, N) weighted middle bounds
            - weighted_matrix_u: (M, N) weighted upper bounds
    """
    l_mat = np.asarray(decision_matrix_l, dtype=np.float64)
    m_mat = np.asarray(decision_matrix_m, dtype=np.float64)
    u_mat = np.asarray(decision_matrix_u, dtype=np.float64)

    wl = np.asarray(weights_l, dtype=np.float64)
    wm = np.asarray(weights_m, dtype=np.float64)
    wu = np.asarray(weights_u, dtype=np.float64)

    ctype = np.asarray(criteria_types, dtype=np.int64)

    # Validation
    if l_mat.ndim != 2 or m_mat.ndim != 2 or u_mat.ndim != 2:
        raise ValueError("Decision matrices must be 2-dimensional.")
    if l_mat.shape != m_mat.shape or m_mat.shape != u_mat.shape:
        raise ValueError("Decision matrices must have the same shape.")

    n_alt, n_crit = l_mat.shape
    if n_alt == 0 or n_crit == 0:
        raise ValueError("Matrices must have >0 rows and >0 columns.")

    if wl.shape != (n_crit,) or wm.shape != (n_crit,) or wu.shape != (n_crit,):
        raise ValueError("Weight arrays must have length equal to the number of criteria.")

    if ctype.shape != (n_crit,):
        raise ValueError("criteria_types length must match number of criteria.")

    if not np.all(np.isin(ctype, [1, -1])):
        raise ValueError("criteria_types must contain only 1 (benefit) or -1 (cost).")

    if not np.all((wl <= wm) & (wm <= wu)):
        raise ValueError("Weight arrays must satisfy l <= m <= u.")

    if not np.all((wl >= 0) & (wm >= 0) & (wu >= 0)):
        raise ValueError("Weight values must be non-negative.")

    if not np.all((l_mat <= m_mat) & (m_mat <= u_mat)):
        raise ValueError("Decision matrices must satisfy l <= m <= u.")

    cost_mask = ctype == -1
    if np.any(cost_mask):
        if (
            np.any(l_mat[:, cost_mask] == 0)
            or np.any(m_mat[:, cost_mask] == 0)
            or np.any(u_mat[:, cost_mask] == 0)
        ):
            raise ValueError("Cost criteria cannot have zero values (division by zero).")

    # 2. Fuzzy Normalization
    r_l = np.zeros_like(l_mat)
    r_m = np.zeros_like(m_mat)
    r_u = np.zeros_like(u_mat)

    for j in range(n_crit):
        if ctype[j] == 1:
            max_u = max(np.max(u_mat[:, j]), 1e-12)
            r_l[:, j] = l_mat[:, j] / max_u
            r_m[:, j] = m_mat[:, j] / max_u
            r_u[:, j] = u_mat[:, j] / max_u
        else:
            min_l = np.min(l_mat[:, j])
            r_l[:, j] = min_l / u_mat[:, j]
            r_m[:, j] = min_l / m_mat[:, j]
            r_u[:, j] = min_l / l_mat[:, j]

    # 3. Weighted Fuzzy Matrix
    v_l = r_l * wl[None, :]
    v_m = r_m * wm[None, :]
    v_u = r_u * wu[None, :]

    # 4. FPIS and FNIS
    A_plus_l = np.max(v_l, axis=0)
    A_plus_m = np.max(v_m, axis=0)
    A_plus_u = np.max(v_u, axis=0)

    A_minus_l = np.min(v_l, axis=0)
    A_minus_m = np.min(v_m, axis=0)
    A_minus_u = np.min(v_u, axis=0)

    # 5. Distance Calculation
    d_plus = np.zeros(n_alt)
    d_minus = np.zeros(n_alt)

    for i in range(n_alt):
        dist_p = np.sqrt(
            1.0
            / 3.0
            * ((v_l[i] - A_plus_l) ** 2 + (v_m[i] - A_plus_m) ** 2 + (v_u[i] - A_plus_u) ** 2)
        )
        d_plus[i] = np.sum(dist_p)

        dist_m = np.sqrt(
            1.0
            / 3.0
            * ((v_l[i] - A_minus_l) ** 2 + (v_m[i] - A_minus_m) ** 2 + (v_u[i] - A_minus_u) ** 2)
        )
        d_minus[i] = np.sum(dist_m)

    # 6. Closeness Coefficient
    denom = d_plus + d_minus
    cc = np.zeros(n_alt)
    mask = denom > 0
    cc[mask] = d_minus[mask] / denom[mask]

    # 7. Ranking
    ranks = np.argsort(-cc)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, n_alt + 1)

    return {
        "closeness_coefficients": cc,
        "ranking": rank_order,
        "distance_positive": d_plus,
        "distance_negative": d_minus,
        "weighted_matrix_l": v_l,
        "weighted_matrix_m": v_m,
        "weighted_matrix_u": v_u,
    }


def fuzzy_vikor_method(
    decision_matrix_l: np.ndarray,
    decision_matrix_m: np.ndarray,
    decision_matrix_u: np.ndarray,
    weights_l: np.ndarray,
    weights_m: np.ndarray,
    weights_u: np.ndarray,
    criteria_types: np.ndarray,
    v: float = 0.5,
) -> dict[str, Union[np.ndarray, list[int]]]:
    """Calculates compromise ranking scores using the Fuzzy VIKOR method.

    Fuzzy VIKOR extends classical VIKOR to handle uncertain/linguistic evaluations
    using Triangular Fuzzy Numbers (TFN).

    Args:
        decision_matrix_l: (M, N) array of lower bounds of TFNs for M alternatives, N criteria.
        decision_matrix_m: (M, N) array of middle bounds.
        decision_matrix_u: (M, N) array of upper bounds.
        weights_l: (N,) array of lower bounds for criteria weights.
        weights_m: (N,) array of middle bounds.
        weights_u: (N,) array of upper bounds.
        criteria_types: (N,) array indicating benefit (1) or cost (-1) criteria.
        v: Weight of the strategy of "majority of criteria" (usually 0.5).

    Returns:
        Dict containing:
            - Q: (M,) array of Q values (lower is better)
            - S: (M,) array of S values (group utility)
            - R: (M,) array of R values (individual regret)
            - ranking: (M,) array of 1-based ranks by Q (1=best)
            - compromise_set: list of 0-based indices of compromise solutions
            - defuzzified_matrix: (M, N) defuzzified decision matrix
    """
    l_mat = np.asarray(decision_matrix_l, dtype=np.float64)
    m_mat = np.asarray(decision_matrix_m, dtype=np.float64)
    u_mat = np.asarray(decision_matrix_u, dtype=np.float64)

    wl = np.asarray(weights_l, dtype=np.float64)
    wm = np.asarray(weights_m, dtype=np.float64)
    wu = np.asarray(weights_u, dtype=np.float64)

    ctype = np.asarray(criteria_types, dtype=np.int64)

    # Validation
    if l_mat.ndim != 2 or m_mat.ndim != 2 or u_mat.ndim != 2:
        raise ValueError("Decision matrices must be 2-dimensional.")
    if l_mat.shape != m_mat.shape or m_mat.shape != u_mat.shape:
        raise ValueError("Decision matrices must have the same shape.")

    n_alt, n_crit = l_mat.shape
    if n_alt == 0 or n_crit == 0:
        raise ValueError("Matrices must have >0 rows and >0 columns.")

    if wl.shape != (n_crit,) or wm.shape != (n_crit,) or wu.shape != (n_crit,):
        raise ValueError("Weight arrays must have length equal to the number of criteria.")

    if ctype.shape != (n_crit,):
        raise ValueError("criteria_types length must match number of criteria.")

    if not np.all(np.isin(ctype, [1, -1])):
        raise ValueError("criteria_types must contain only 1 (benefit) or -1 (cost).")

    if not np.all((wl <= wm) & (wm <= wu)):
        raise ValueError("Weight arrays must satisfy l <= m <= u.")

    if not np.all((wl >= 0) & (wm >= 0) & (wu >= 0)):
        raise ValueError("Weight values must be non-negative.")

    if not np.all((l_mat <= m_mat) & (m_mat <= u_mat)):
        raise ValueError("Decision matrices must satisfy l <= m <= u.")

    if not (0.0 <= v <= 1.0):
        raise ValueError("v must be between 0.0 and 1.0.")

    # Defuzzify weights
    w_def = (wl + wm + wu) / 3.0

    # Defuzzify matrix
    defuzz_mat = (l_mat + m_mat + u_mat) / 3.0

    # Find crisp f*_j and f-_j using defuzzified values
    f_star = np.zeros(n_crit)
    f_minus = np.zeros(n_crit)

    for j in range(n_crit):
        if ctype[j] == 1:
            f_star[j] = np.max(defuzz_mat[:, j])
            f_minus[j] = np.min(defuzz_mat[:, j])
        else:
            f_star[j] = np.min(defuzz_mat[:, j])
            f_minus[j] = np.max(defuzz_mat[:, j])

    # Compute S_i and R_i
    S = np.zeros(n_alt)
    R = np.zeros(n_alt)

    diff = np.abs(f_star - f_minus)
    diff = np.where(diff > 0, diff, 1e-9)

    for i in range(n_alt):
        term = w_def * np.abs(f_star - defuzz_mat[i, :]) / diff
        S[i] = np.sum(term)
        R[i] = np.max(term)

    # Compute Q_i
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

    # Ranking
    ranks = np.argsort(Q)
    rank_order = np.empty_like(ranks)
    rank_order[ranks] = np.arange(1, n_alt + 1)

    # Compromise conditions check
    # C1: Acceptable advantage
    Q_sorted = Q[ranks]
    DQ = 1.0 / (n_alt - 1) if n_alt > 1 else 0.0

    c1_met = False
    if n_alt > 1:
        c1_met = (Q_sorted[1] - Q_sorted[0]) >= DQ

    # C2: Acceptable stability
    best_S = np.argmin(S)
    best_R = np.argmin(R)
    best_idx = ranks[0]
    c2_met = (best_idx == best_S) or (best_idx == best_R)

    compromise_set = []
    if c1_met and c2_met:
        compromise_set.append(int(best_idx))
    elif not c1_met:
        for idx in ranks:
            if (Q[idx] - Q_sorted[0]) < DQ:
                compromise_set.append(int(idx))
    elif not c2_met:
        compromise_set.append(int(ranks[0]))
        if n_alt > 1:
            compromise_set.append(int(ranks[1]))

    compromise_set = sorted(set(compromise_set))

    return {
        "Q": Q,
        "S": S,
        "R": R,
        "ranking": rank_order,
        "compromise_set": compromise_set,
        "defuzzified_matrix": defuzz_mat,
    }


def spotis_method(
    matrix: np.ndarray,
    weights: Union[List[float], np.ndarray],
    types: Union[List[str], List[int], np.ndarray],
    bounds: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """SPOTIS (Stable Preference Ordering Towards Ideal Solution) MCDA ranking method.

    Evaluates alternatives by computing normalized distances to an ideal solution
    vector S*_j bounded by criteria domain boundaries [S_{j,min}, S_{j,max}],
    preventing rank reversal.

    Args:
        matrix: 2D NumPy array of shape (M, N) for M alternatives across N criteria.
        weights: List or 1D array of length N for criteria weights.
        types: List or array of criteria types (1 / "+" / "benefit" or 0 / "-" / "cost").
        bounds: Optional 2D array of shape (N, 2) specifying [min, max] per criterion.

    Returns:
        Dict containing:
          - 'scores': 1D NumPy array (M,) of preference distance scores (0 is ideal best).
          - 'ranks': 1D NumPy array (M,) of integer ranks (1 = best).
          - 'ideal_solution': 1D NumPy array (N,) of criterion ideal target values.
          - 'bounds': 2D NumPy array (N, 2) of criteria min and max boundaries.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError("matrix must be a 2D array.")
    m_alt, n_crit = mat.shape
    if m_alt < 2:
        raise ValueError("At least 2 alternatives are required.")
    if n_crit < 1:
        raise ValueError("At least 1 criterion is required.")

    w = np.asarray(weights, dtype=np.float64)
    if len(w) != n_crit:
        raise ValueError("Length of weights must match number of criteria N.")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    w_sum = np.sum(w)
    if w_sum <= 0:
        raise ValueError("Sum of weights must be positive.")
    w_norm = w / w_sum

    t_clean = []
    for item in types:
        if isinstance(item, str):
            st = item.lower().strip()
            if st in ("+", "benefit", "1", "max"):
                t_clean.append(1)
            elif st in ("-", "cost", "0", "min"):
                t_clean.append(0)
            else:
                raise ValueError(f"Unknown criterion type: {item}")
        else:
            t_clean.append(1 if int(item) == 1 else 0)
    t_arr = np.array(t_clean, dtype=int)

    if bounds is not None:
        b_mat = np.asarray(bounds, dtype=np.float64)
        if b_mat.shape != (n_crit, 2):
            raise ValueError("bounds must be a (N, 2) array of [min, max].")
        if np.any(b_mat[:, 0] >= b_mat[:, 1]):
            raise ValueError("bounds min must be strictly smaller than max for all criteria.")
    else:
        b_mat = np.zeros((n_crit, 2), dtype=np.float64)
        for j in range(n_crit):
            c_min = float(np.min(mat[:, j]))
            c_max = float(np.max(mat[:, j]))
            if c_min == c_max:
                c_max = c_min + 1.0
            b_mat[j] = [c_min, c_max]

    ideal_s = np.zeros(n_crit, dtype=np.float64)
    for j in range(n_crit):
        if t_arr[j] == 1:
            ideal_s[j] = b_mat[j, 1]
        else:
            ideal_s[j] = b_mat[j, 0]

    d_mat = np.zeros_like(mat)
    for j in range(n_crit):
        rng = b_mat[j, 1] - b_mat[j, 0]
        d_mat[:, j] = np.abs(mat[:, j] - ideal_s[j]) / rng

    scores = np.sum(d_mat * w_norm[None, :], axis=1)

    sort_idx = np.argsort(scores)
    ranks = np.empty_like(sort_idx)
    ranks[sort_idx] = np.arange(1, m_alt + 1)

    return {
        "scores": scores,
        "ranks": ranks,
        "ideal_solution": ideal_s,
        "bounds": b_mat,
    }


def ivif_topsis_method(
    ivif_matrix: np.ndarray,
    weights: Union[List[float], np.ndarray],
    types: Union[List[str], List[int], np.ndarray],
) -> dict[str, Any]:
    """Interval-Valued Intuitionistic Fuzzy TOPSIS (IVIF-TOPSIS) MCDA Method.

    Evaluates alternatives under expert hesitation and uncertainty where evaluations are given as
    IVIF values [[mu_L, mu_U], [nu_L, nu_U]].

    Args:
        ivif_matrix: 3D array of shape (M, N, 4) containing IVIF values for M alternatives across N criteria.
                     The last dimension specifies [mu_L, mu_U, nu_L, nu_U].
        weights: List or 1D array of length N containing criteria weights (summing to 1.0).
        types: List or array of criteria types (1 / "+" for benefit, 0 / "-" for cost).

    Returns:
        Dict containing:
          - 'closeness_coefficients': 1D float array (M,) of closeness coefficients CC_i in [0, 1].
          - 'ranks': 1D int array (M,) of integer ranks (1 = best, highest CC_i).
          - 'distance_to_pis': 1D float array (M,) of Euclidean distances to PIS.
          - 'distance_to_nis': 1D float array (M,) of Euclidean distances to NIS.
    """
    mat = np.asarray(ivif_matrix, dtype=np.float64)
    if mat.ndim != 3 or mat.shape[2] != 4:
        raise ValueError("ivif_matrix must be a 3D array of shape (M, N, 4).")
    m_alt, n_crit, _ = mat.shape
    if m_alt < 2:
        raise ValueError("At least 2 alternatives are required.")
    if n_crit < 1:
        raise ValueError("At least 1 criterion is required.")

    w = np.asarray(weights, dtype=np.float64)
    if len(w) != n_crit:
        raise ValueError("Length of weights must match N criteria.")
    if np.any(w < 0):
        raise ValueError("Weights must be non-negative.")
    w_sum = np.sum(w)
    if w_sum <= 0:
        raise ValueError("Sum of weights must be positive.")
    w_norm = w / w_sum

    mu_L = mat[:, :, 0]
    mu_U = mat[:, :, 1]
    nu_L = mat[:, :, 2]
    nu_U = mat[:, :, 3]

    if np.any((mu_L < 0) | (mu_U > 1) | (mu_L > mu_U)):
        raise ValueError("IVIF membership bounds must satisfy 0 <= mu_L <= mu_U <= 1.")
    if np.any((nu_L < 0) | (nu_U > 1) | (nu_L > nu_U)):
        raise ValueError("IVIF non-membership bounds must satisfy 0 <= nu_L <= nu_U <= 1.")
    if np.any((mu_U + nu_U) > 1.0 + 1e-6):
        raise ValueError("IVIF bounds must satisfy mu_U + nu_U <= 1.")

    t_clean = []
    for item in types:
        if isinstance(item, str):
            st = item.lower().strip()
            if st in ("+", "benefit", "1", "max"):
                t_clean.append(1)
            elif st in ("-", "cost", "0", "min"):
                t_clean.append(0)
            else:
                raise ValueError(f"Unknown criterion type: {item}")
        else:
            t_clean.append(1 if int(item) == 1 else 0)
    t_arr = np.array(t_clean, dtype=int)

    w_grid = w_norm[None, :]
    w_mu_L = 1.0 - (1.0 - mu_L) ** w_grid
    w_mu_U = 1.0 - (1.0 - mu_U) ** w_grid
    w_nu_L = nu_L ** w_grid
    w_nu_U = nu_U ** w_grid

    pis = np.zeros((n_crit, 4), dtype=np.float64)
    nis = np.zeros((n_crit, 4), dtype=np.float64)

    for j in range(n_crit):
        if t_arr[j] == 1:
            pis[j] = [np.max(w_mu_L[:, j]), np.max(w_mu_U[:, j]), np.min(w_nu_L[:, j]), np.min(w_nu_U[:, j])]
            nis[j] = [np.min(w_mu_L[:, j]), np.min(w_mu_U[:, j]), np.max(w_nu_L[:, j]), np.max(w_nu_U[:, j])]
        else:
            pis[j] = [np.min(w_mu_L[:, j]), np.min(w_mu_U[:, j]), np.max(w_nu_L[:, j]), np.max(w_nu_U[:, j])]
            nis[j] = [np.max(w_mu_L[:, j]), np.max(w_mu_U[:, j]), np.min(w_nu_L[:, j]), np.min(w_nu_U[:, j])]

    d_pis = np.zeros(m_alt, dtype=np.float64)
    d_nis = np.zeros(m_alt, dtype=np.float64)

    w_mat = np.stack([w_mu_L, w_mu_U, w_nu_L, w_nu_U], axis=2)

    for i in range(m_alt):
        diff_pis = w_mat[i] - pis  # shape (N, 4)
        dist_pis_j = np.sqrt(0.25 * np.sum(diff_pis**2, axis=1))
        d_pis[i] = np.sum(dist_pis_j)

        diff_nis = w_mat[i] - nis
        dist_nis_j = np.sqrt(0.25 * np.sum(diff_nis**2, axis=1))
        d_nis[i] = np.sum(dist_nis_j)

    denom = d_pis + d_nis
    cc = np.where(denom > 0, d_nis / denom, 0.0)

    sort_idx = np.argsort(-cc)
    ranks = np.empty_like(sort_idx)
    ranks[sort_idx] = np.arange(1, m_alt + 1)

    return {
        "closeness_coefficients": cc,
        "ranks": ranks,
        "distance_to_pis": d_pis,
        "distance_to_nis": d_nis,
    }


def neutrosophic_waspas_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    lambda_param: float = 0.5,
) -> dict[str, Any]:
    """Single-Valued Neutrosophic WASPAS MCDA Engine.

    Aggregates alternatives under neutrosophic truth (T), indeterminacy (I), and falsity (F) degrees
    using Weighted Sum Model (WSM) and Weighted Product Model (WPM).

    Args:
        decision_matrix: Array of shape (n_alt, n_crit) containing performance values.
        weights: Criteria importance weights summing to 1.
        lambda_param: Trade-off parameter between WSM and WPM (0 to 1).

    Returns:
        Dict containing alternative scores, ranks, WSM scores, and WPM scores.
    """
    n_alt, n_crit = decision_matrix.shape
    weights_norm = weights / np.sum(weights)

    mins = np.min(decision_matrix, axis=0)
    maxs = np.max(decision_matrix, axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm_matrix = (decision_matrix - mins) / ranges

    t_deg = norm_matrix
    i_deg = 1.0 - norm_matrix
    f_deg = 1.0 - (norm_matrix ** 2)
    s_scores = (2.0 + t_deg - i_deg - f_deg) / 3.0

    wsm = np.sum(s_scores * weights_norm, axis=1)
    wpm = np.prod(s_scores ** weights_norm, axis=1)

    q_score = lambda_param * wsm + (1.0 - lambda_param) * wpm
    ranks = np.argsort(-q_score).argsort() + 1

    return {
        "waspas_scores": q_score,
        "rankings": ranks,
        "wsm_scores": wsm,
        "wpm_scores": wpm,
    }


def if_vikor_method(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    v_preference: float = 0.5,
) -> dict[str, Any]:
    r"""Intuitionistic Fuzzy VIKOR MCDA Engine.

    Ranks alternatives using utility ($S$) and regret ($R$) measures based on intuitionistic
    fuzzy membership ($\mu$) and non-membership ($\nu$) values.

    Args:
        decision_matrix: Array of shape (n_alt, n_crit) containing performance values.
        weights: Criteria importance weights vector (n_crit,).
        v_preference: Weight of maximum group utility strategy (default 0.5).

    Returns:
        Dict containing S scores, R scores, Q scores, and alternative rankings.
    """
    n_alt, n_crit = decision_matrix.shape
    w_norm = weights / np.sum(weights)

    mins = np.min(decision_matrix, axis=0)
    maxs = np.max(decision_matrix, axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm = (decision_matrix - mins) / ranges

    f_star = np.max(norm, axis=0)
    f_minus = np.min(norm, axis=0)

    d_matrix = (f_star - norm) / (f_star - f_minus + 1e-12)
    s_i = np.sum(w_norm * d_matrix, axis=1)
    r_i = np.max(w_norm * d_matrix, axis=1)

    s_star, s_minus = np.min(s_i), np.max(s_i)
    r_star, r_minus = np.min(r_i), np.max(r_i)

    q_i = v_preference * (s_i - s_star) / (s_minus - s_star + 1e-12) + (1.0 - v_preference) * (r_i - r_star) / (r_minus - r_star + 1e-12)
    ranks = np.argsort(q_i).argsort() + 1

    return {
        "s_scores": s_i,
        "r_scores": r_i,
        "q_scores": q_i,
        "rankings": ranks,
    }


def rough_topsis_method(
    lower_matrix: np.ndarray,
    upper_matrix: np.ndarray,
    weights: np.ndarray,
) -> dict[str, Any]:
    """Rough TOPSIS MCDA Engine.

    Evaluates decision alternatives using lower and upper rough approximation boundary matrices.

    Args:
        lower_matrix: Array of shape (n_alt, n_crit) for lower rough bounds.
        upper_matrix: Array of shape (n_alt, n_crit) for upper rough bounds.
        weights: Criteria importance weights.

    Returns:
        Dict containing closeness coefficients, ranks, and rough distance metrics.
    """
    n_alt, n_crit = lower_matrix.shape
    w_norm = weights / np.sum(weights)

    mid_matrix = 0.5 * (lower_matrix + upper_matrix)

    mins = np.min(mid_matrix, axis=0)
    maxs = np.max(mid_matrix, axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    norm = (mid_matrix - mins) / ranges

    weighted_norm = norm * w_norm

    pis = np.max(weighted_norm, axis=0)
    nis = np.min(weighted_norm, axis=0)

    d_pis = np.sqrt(np.sum((weighted_norm - pis) ** 2, axis=1))
    d_nis = np.sqrt(np.sum((weighted_norm - nis) ** 2, axis=1))

    cc = d_nis / (d_pis + d_nis + 1e-12)
    ranks = np.argsort(-cc).argsort() + 1

    return {
        "closeness_coefficients": cc,
        "rankings": ranks,
        "distance_to_pis": d_pis,
        "distance_to_nis": d_nis,
    }





