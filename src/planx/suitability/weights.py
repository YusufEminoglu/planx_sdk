# -*- coding: utf-8 -*-
"""MCDA weighting engines (AHP, Entropy, CRITIC, PCA)."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np


def ahp_weights(matrix: Union[list[list[float]], np.ndarray]) -> Tuple[np.ndarray, float]:
    """Calculates weights and Consistency Ratio (CR) using the Analytic Hierarchy Process (AHP).

    Args:
        matrix: Square positive pairwise comparison matrix.

    Returns:
        Tuple of:
          - weights: 1D NumPy array of normalized weights.
          - cr: Consistency Ratio (CR). A value <= 0.10 is typically considered acceptable.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("AHP matrix must be square")
    n = m.shape[0]
    if n == 0:
        raise ValueError("AHP matrix cannot be empty")
    if np.any(m <= 0):
        raise ValueError("AHP matrix values must be positive.")

    # Calculate eigenvalues and eigenvectors
    vals, vecs = np.linalg.eig(m)
    max_idx = int(np.argmax(vals.real))
    lam = float(vals[max_idx].real)

    # Extract the corresponding eigenvector
    w = np.abs(vecs[:, max_idx].real)
    if w.sum() == 0:
        w = np.ones(n, dtype=float)
    w = w / w.sum()

    # Consistency Index and Consistency Ratio
    ri_map = {
        1: 0.0,
        2: 0.0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49,
    }
    ri = ri_map.get(n, 1.49)
    ci = (lam - n) / max(1.0, n - 1)
    cr = 0.0 if ri == 0.0 else ci / ri
    return w, cr


def decision_matrix_from_layers(
    layers: List[np.ndarray], nodata: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Flattens a list of spatial layers (NumPy arrays) into a 2D decision matrix.

    Only pixel locations that are valid (finite and not equal to nodata) in ALL
    layers are included in the decision matrix.

    Args:
        layers: List of NumPy arrays of identical shapes.
        nodata: Optional value representing no-data to be excluded.

    Returns:
        Tuple of:
          - decision_matrix: 2D NumPy array of shape (M, N) where M is the number of
            valid pixels and N is the number of layers (criteria).
          - valid_mask: Boolean NumPy array of the same shape as the input layers,
            where True represents pixels included in the decision matrix.
    """
    if not layers:
        raise ValueError("At least one layer must be provided")

    shape = layers[0].shape
    valid = np.ones(shape, dtype=bool)

    flat_layers = []
    for i, lyr in enumerate(layers):
        if lyr.shape != shape:
            raise ValueError(f"Layer at index {i} has shape {lyr.shape}, expected {shape}")

        arr = np.asarray(lyr, dtype=np.float64)
        m = np.isfinite(arr)
        if nodata is not None:
            m &= arr != nodata

        valid &= m
        flat_layers.append(arr)

    # Gather valid cells into decision matrix: shape (M, N)
    m_count = np.sum(valid)
    n_count = len(layers)
    decision_matrix = np.empty((m_count, n_count), dtype=np.float64)

    for i in range(n_count):
        decision_matrix[:, i] = flat_layers[i][valid]

    return decision_matrix, valid


def entropy_weights(decision_matrix: np.ndarray) -> np.ndarray:
    """Calculates weights for criteria using Shannon's Entropy method.

    Args:
        decision_matrix: 2D NumPy array of shape (M, N) representing M alternatives
            and N criteria.

    Returns:
        1D NumPy array of shape (N,) containing normalized criteria weights.
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("decision_matrix must be a 2D array")

    m_alt, n_crit = X.shape
    if m_alt == 0 or n_crit == 0:
        return np.ones(n_crit, dtype=np.float64) / max(1, n_crit)

    # Normalize each criterion to [0, 1] range
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    denom = maxs - mins
    denom = np.where(denom <= 1e-12, 1.0, denom)

    Z = (X - mins) / denom
    Z = np.clip(Z, 0.0, 1.0)

    # Proportions P_ij
    # Add small epsilon to avoid log(0)
    P = Z + 1e-12
    col_sums = np.sum(P, axis=0, keepdims=True)
    P = P / col_sums

    # Calculate entropy
    k_entropy = 1.0 / np.log(max(2, m_alt))
    entropy_values = -k_entropy * np.sum(P * np.log(P), axis=0)

    # Diversification degree
    d = 1.0 - entropy_values
    d = np.where(np.isfinite(d), d, 0.0)

    # Normalize to get weights
    d_sum = np.sum(d)
    if d_sum <= 1e-12:
        return np.ones(n_crit, dtype=np.float64) / n_crit

    return d / d_sum


def critic_weights(
    decision_matrix: np.ndarray, directions: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates weights using the CRITIC (Criteria Importance Through Intercriteria Correlation)
    method.

    Args:
        decision_matrix: 2D NumPy array of shape (M, N) representing M alternatives
            and N criteria.
        directions: Optional list or array of length N specifying criterion direction.
            1 for benefit criteria (higher is better), -1 for cost criteria (lower is better).
            Defaults to all benefit.

    Returns:
        Tuple of:
          - weights: 1D NumPy array of shape (N,) containing normalized criteria weights.
          - std_devs: 1D NumPy array of shape (N,) containing standard deviations of normalizations.
          - contrast_sums: 1D NumPy array of shape (N,) containing sum of (1 - correlation)
            for each criterion.
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("decision_matrix must be a 2D array")

    m_alt, n_crit = X.shape
    if m_alt == 0 or n_crit == 0:
        return (
            np.ones(n_crit, dtype=np.float64) / max(1, n_crit),
            np.zeros(n_crit, dtype=np.float64),
            np.ones(n_crit, dtype=np.float64),
        )

    if directions is None:
        dirs = np.ones(n_crit, dtype=np.int32)
    else:
        dirs = np.asarray(directions, dtype=np.int32)
        if dirs.shape[0] != n_crit:
            raise ValueError(
                f"directions length ({dirs.shape[0]}) must match criteria count ({n_crit})"
            )

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    denom = maxs - mins
    denom = np.where(denom <= 1e-12, 1.0, denom)

    # Normalize based on benefit / cost criteria
    Z = np.empty_like(X, dtype=np.float64)
    for j in range(n_crit):
        if dirs[j] >= 0:
            Z[:, j] = (X[:, j] - mins[j]) / denom[j]
        else:
            Z[:, j] = (maxs[j] - X[:, j]) / denom[j]
    Z = np.clip(Z, 0.0, 1.0)

    # Standard deviation of each normalized criterion
    sigma = np.std(Z, axis=0)

    # Intercriteria correlation
    # handle case when variance is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(Z, rowvar=False)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    np.fill_diagonal(corr, 1.0)

    # Conflict/contrast sum: sum(1 - r_jk)
    conflict = np.sum(1.0 - corr, axis=1)

    # Criterion importance score
    c_score = sigma * conflict
    c_score = np.where(np.isfinite(c_score), c_score, 0.0)

    # Normalize to get weights
    c_sum = np.sum(c_score)
    if c_sum <= 1e-12:
        return (
            np.ones(n_crit, dtype=np.float64) / n_crit,
            sigma,
            conflict,
        )

    return c_score / c_sum, sigma, conflict


def pca_weights(decision_matrix: np.ndarray) -> np.ndarray:
    """Calculates proxy weights for criteria using Principal Component Analysis (PCA).

    Weights are calculated from the absolute loadings of the first principal
    component (PC1), weighted by its explained variance ratio.

    Args:
        decision_matrix: 2D NumPy array of shape (M, N) representing M alternatives
            and N criteria.

    Returns:
        1D NumPy array of shape (N,) containing normalized criteria weights.
    """
    X = np.asarray(decision_matrix, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("decision_matrix must be a 2D array")

    m_alt, n_crit = X.shape
    if m_alt < 3 or n_crit == 0:
        return np.ones(n_crit, dtype=np.float64) / max(1, n_crit)

    # Standardize features
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    Z = (X - mu) / sd

    # Compute correlation/covariance matrix and eigen-decompose
    cov = np.cov(Z, rowvar=False)
    # Ensure cov is 2D even for single criterion
    if cov.ndim == 0:
        cov = cov.reshape((1, 1))
    elif cov.ndim == 1:
        cov = cov.reshape((cov.shape[0], cov.shape[0]))

    vals, vecs = np.linalg.eigh(cov)

    # Sort in descending order of eigenvalues
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    if vals[0] <= 1e-12:
        return np.ones(n_crit, dtype=np.float64) / n_crit

    # Use absolute loadings of the first principal component
    loadings = np.abs(vecs[:, 0])

    # Weighted by PC1 explained variance ratio
    total_variance = np.sum(np.maximum(vals, 0.0))
    evr_pc1 = float(vals[0] / max(1e-12, total_variance))

    scores = loadings * max(evr_pc1, 1e-6)

    scores_sum = np.sum(scores)
    if scores_sum <= 1e-12:
        return np.ones(n_crit, dtype=np.float64) / n_crit

    return scores / scores_sum


def bwm_weights(
    best_to_others: np.ndarray,
    others_to_worst: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Calculates MCDA criterion weights using the Best-Worst Method (BWM).

    Args:
        best_to_others: 1D array A_B of preference scores of Best criterion over others (1..9).
        others_to_worst: 1D array A_W of preference scores of others over Worst criterion (1..9).

    Returns:
        A tuple of:
          - weights: 1D NumPy array of shape (N,) containing normalized criteria weights.
          - consistency_index: Float indicating consistency score xi (0 = perfectly consistent).
    """
    a_B = np.asarray(best_to_others, dtype=np.float64)
    a_W = np.asarray(others_to_worst, dtype=np.float64)
    n = len(a_B)

    if len(a_W) != n:
        raise ValueError("best_to_others and others_to_worst must have equal length.")
    if np.any(a_B < 1.0) or np.any(a_W < 1.0):
        raise ValueError("Preference scores in BWM must be greater than or equal to 1.0.")

    from scipy.optimize import linprog

    c = np.zeros(n + 1)
    c[-1] = 1.0

    A_ub = []
    b_ub = []

    best_idx = int(np.argmin(a_B))
    worst_idx = int(np.argmin(a_W))

    for j in range(n):
        row1 = np.zeros(n + 1)
        row1[best_idx] = 1.0
        row1[j] = -a_B[j]
        row1[-1] = -1.0
        A_ub.append(row1)
        b_ub.append(0.0)

        row2 = np.zeros(n + 1)
        row2[best_idx] = -1.0
        row2[j] = a_B[j]
        row2[-1] = -1.0
        A_ub.append(row2)
        b_ub.append(0.0)

        row3 = np.zeros(n + 1)
        row3[j] = 1.0
        row3[worst_idx] = -a_W[j]
        row3[-1] = -1.0
        A_ub.append(row3)
        b_ub.append(0.0)

        row4 = np.zeros(n + 1)
        row4[j] = -1.0
        row4[worst_idx] = a_W[j]
        row4[-1] = -1.0
        A_ub.append(row4)
        b_ub.append(0.0)

    A_eq = [np.append(np.ones(n), 0.0)]
    b_eq = [1.0]

    bounds = [(0.0, 1.0) for _ in range(n)] + [(0.0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        weights = res.x[:n]
        weights = np.maximum(0.0, weights)
        w_sum = np.sum(weights)
        if w_sum > 0:
            weights = weights / w_sum
        return weights, float(res.x[-1])

    raw_weights = 1.0 / (a_B + 1e-9)
    weights = raw_weights / np.sum(raw_weights)
    return weights, 1.0


def fuzzy_ahp_weights(fuzzy_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Calculates MCDA criterion weights using Fuzzy AHP (Chang's extent analysis method).

    Args:
        fuzzy_matrix: 3D NumPy array of shape (N, N, 3) where fuzzy_matrix[i, j] = (l, m, u)
                      representing triangular fuzzy preference of criterion i over criterion j.

    Returns:
        A tuple of:
          - weights: 1D NumPy array of shape (N,) containing normalized crisp criteria weights.
          - consistency_index: Consistency index float (0 = consistent).
    """
    mat = np.asarray(fuzzy_matrix, dtype=np.float64)
    if mat.ndim != 3 or mat.shape[2] != 3 or mat.shape[0] != mat.shape[1]:
        raise ValueError("fuzzy_matrix must be of shape (N, N, 3).")

    n = mat.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64), 0.0

    # Calculate fuzzy synthetic extent for each criterion i:
    # S_i = sum_j(M_ij) (+) [sum_i sum_j M_ij]^-1
    row_sums = np.sum(mat, axis=1)  # (N, 3) -> (l_i, m_i, u_i)
    total_sum = np.sum(row_sums, axis=0)  # (3,) -> (l_tot, m_tot, u_tot)

    inv_total = np.array([1.0 / total_sum[2], 1.0 / total_sum[1], 1.0 / total_sum[0]])

    synthetic_extents = row_sums * inv_total  # (N, 3) -> (l_S, m_S, u_S)

    # Degree of possibility V(M2 >= M1)
    def possibility(m2: np.ndarray, m1: np.ndarray) -> float:
        l1, m1_val, u1 = m1
        l2, m2_val, u2 = m2
        if m2_val >= m1_val:
            return 1.0
        elif l1 >= u2:
            return 0.0
        else:
            return float((l1 - u2) / ((m2_val - u2) - (m1_val - l1)))

    d_min = np.zeros(n, dtype=np.float64)
    for i in range(n):
        possibilities = []
        for k in range(n):
            if i != k:
                possibilities.append(possibility(synthetic_extents[i], synthetic_extents[k]))
        d_min[i] = min(possibilities) if possibilities else 1.0

    d_sum = np.sum(d_min)
    if d_sum > 0:
        weights = d_min / d_sum
    else:
        weights = np.ones(n, dtype=np.float64) / n

    # Consistency indicator from middle crisp matrix
    crisp_m = mat[:, :, 1]
    _, ci = ahp_weights(crisp_m)

    return weights, ci


def fucom_weights(
    comparative_priorities: np.ndarray,
    comparative_ranks: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Calculates MCDA criterion weights using the Full Consistency Method (FUCOM).

    FUCOM determines criterion weights with N-1 comparative steps and minimal deviation chi.

    Args:
        comparative_priorities: 1D array of shape (N-1,) of comparative priority ratios.
        comparative_ranks: 1D array of shape (N,) indicating ranking index order of criteria.

    Returns:
        A tuple of:
          - weights: 1D NumPy array of shape (N,) containing normalized criteria weights.
          - chi_deviation: Float indicating deviation score chi (0 = fully consistent).
    """
    phi = np.asarray(comparative_priorities, dtype=np.float64)
    ranks = np.asarray(comparative_ranks, dtype=np.int64)
    n = len(ranks)

    if len(phi) != n - 1:
        raise ValueError("comparative_priorities length must be N-1.")

    from scipy.optimize import linprog

    c = np.zeros(n + 1)
    c[-1] = 1.0  # Minimize chi

    A_ub = []
    b_ub = []

    # Condition 1: |w_{k} / w_{k+1} - phi_k| <= chi
    # Condition 2: |w_{k} / w_{k+2} - (phi_k * phi_{k+1})| <= chi
    for k in range(n - 1):
        idx_k = ranks[k]
        idx_k1 = ranks[k + 1]

        # w_k - phi_k * w_{k+1} - chi <= 0
        row1 = np.zeros(n + 1)
        row1[idx_k] = 1.0
        row1[idx_k1] = -phi[k]
        row1[-1] = -1.0
        A_ub.append(row1)
        b_ub.append(0.0)

        # -w_k + phi_k * w_{k+1} - chi <= 0
        row2 = np.zeros(n + 1)
        row2[idx_k] = -1.0
        row2[idx_k1] = phi[k]
        row2[-1] = -1.0
        A_ub.append(row2)
        b_ub.append(0.0)

        if k < n - 2:
            idx_k2 = ranks[k + 2]
            phi_trans = phi[k] * phi[k + 1]

            row3 = np.zeros(n + 1)
            row3[idx_k] = 1.0
            row3[idx_k2] = -phi_trans
            row3[-1] = -1.0
            A_ub.append(row3)
            b_ub.append(0.0)

            row4 = np.zeros(n + 1)
            row4[idx_k] = -1.0
            row4[idx_k2] = phi_trans
            row4[-1] = -1.0
            A_ub.append(row4)
            b_ub.append(0.0)

    A_eq = [np.append(np.ones(n), 0.0)]
    b_eq = [1.0]

    bounds = [(0.0, 1.0) for _ in range(n)] + [(0.0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        weights = np.maximum(0.0, res.x[:n])
        w_sum = np.sum(weights)
        if w_sum > 0:
            weights = weights / w_sum
        return weights, float(res.x[-1])

    # Fallback to direct cumulative product
    w_unnorm = np.ones(n, dtype=np.float64)
    for k in range(n - 1):
        w_unnorm[ranks[k + 1]] = w_unnorm[ranks[k]] / max(1e-9, phi[k])
    weights = w_unnorm / np.sum(w_unnorm)
    return weights, 0.0


def dematel_method(direct_influence_matrix: np.ndarray) -> dict:
    """Calculates DEMATEL causal structure matrix.

    DEMATEL = Decision Making Trial and Evaluation Laboratory.

    Args:
        direct_influence_matrix: 2D NumPy array of shape (N, N) for direct influence scores Z_ij.

    Returns:
        Dict containing DEMATEL outputs:
          - normalized_matrix: 2D NumPy array X of shape (N, N).
          - total_influence_matrix: 2D NumPy array T of shape (N, N).
          - prominence: 1D NumPy array D + R (N,) indicating criteria importance.
          - relation: 1D NumPy array D - R (N,) indicating cause (+) or effect (-).
          - cause_effect_class: List of strings ("Cause" or "Effect").
    """
    Z = np.asarray(direct_influence_matrix, dtype=np.float64)
    n, cols = Z.shape

    if n != cols:
        raise ValueError("direct_influence_matrix must be square (N, N).")

    if n == 0:
        return {
            "normalized_matrix": np.zeros((0, 0)),
            "total_influence_matrix": np.zeros((0, 0)),
            "prominence": np.zeros(0),
            "relation": np.zeros(0),
            "cause_effect_class": [],
        }

    # Normalization scale factor s
    row_sums = np.sum(Z, axis=1)
    col_sums = np.sum(Z, axis=0)
    max_sum = max(np.max(row_sums), np.max(col_sums))
    s = 1.0 / max(1e-9, max_sum)

    # Normalized matrix X
    X = Z * s

    # Total influence matrix T = X (I - X)^-1
    I_mat = np.eye(n, dtype=np.float64)
    T = X @ np.linalg.inv(I_mat - X)

    # Dispatch (D) and Receive (R) sums
    D = np.sum(T, axis=1)
    R = np.sum(T, axis=0)

    prominence = D + R
    relation = D - R

    cause_effect = ["Cause" if rel >= 0 else "Effect" for rel in relation]

    return {
        "normalized_matrix": X,
        "total_influence_matrix": T,
        "prominence": prominence,
        "relation": relation,
        "cause_effect_class": cause_effect,
    }




