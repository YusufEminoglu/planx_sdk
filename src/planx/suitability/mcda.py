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







