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
