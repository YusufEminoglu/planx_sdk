# -*- coding: utf-8 -*-
"""Travel demand modeling engine routines."""

from __future__ import annotations

import numpy as np


def trip_generation(
    pop: np.ndarray, jobs: np.ndarray, p_rate: float, a_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate productions and attractions from zone population and jobs."""
    P = (pop * p_rate).astype(np.float64)
    A = (jobs * a_rate).astype(np.float64)
    return P, A


def gravity(
    P: np.ndarray,
    A: np.ndarray,
    cost: np.ndarray,
    beta: float,
    kind: str = "exp",
    max_iter: int = 100,
    tol: float = 1e-4,
) -> tuple[np.ndarray, int, float]:
    """Doubly constrained gravity model using Furness/IPF balancing.

    Returns:
        flow_matrix: 2D array of shape (N, M)
        iterations: number of iterations run
        error: final maximum absolute difference from P/A totals
    """
    P = np.asarray(P, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    if P.ndim != 1 or A.ndim != 1:
        raise ValueError("P and A must be one-dimensional arrays")
    if cost.ndim != 2 or cost.shape != (len(P), len(A)):
        raise ValueError("cost shape must be (len(P), len(A))")
    if not np.all(np.isfinite(P)) or np.any(P < 0):
        raise ValueError("P must contain finite, non-negative values")
    if not np.all(np.isfinite(A)) or np.any(A < 0):
        raise ValueError("A must contain finite, non-negative values")
    if not np.all(np.isfinite(cost)) or np.any(cost < 0):
        raise ValueError("cost must contain finite, non-negative values")
    if kind not in {"exp", "power"}:
        raise ValueError("kind must be 'exp' or 'power'")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be a finite, non-negative value")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite, positive value")

    N, M = cost.shape
    if P.sum() == 0 or A.sum() == 0:
        return np.zeros((N, M), dtype=np.float64), 0, 0.0

    # Scale A to match P to ensure convergence
    P_sum = P.sum()
    A_sum = A.sum()
    A = A * (P_sum / A_sum)

    # Deterrence matrix
    if kind == "power":
        F = np.power(np.maximum(cost, 1e-6), -beta)
    else:
        F = np.exp(-beta * cost)

    # Balancing
    r = np.ones(N, dtype=np.float64)
    s = np.ones(M, dtype=np.float64)

    iters = 0
    diff = 1.0

    for iteration in range(1, max_iter + 1):
        iters = iteration
        denom_r = F @ s
        denom_r[denom_r < 1e-12] = 1e-12
        r = P / denom_r

        denom_s = F.T @ r
        denom_s[denom_s < 1e-12] = 1e-12
        s = A / denom_s

        T = (r[:, None] * F) * s[None, :]

        row_diff = np.abs(T.sum(axis=1) - P).max()
        col_diff = np.abs(T.sum(axis=0) - A).max()
        diff = max(row_diff, col_diff)

        if diff < tol:
            break

    return T, iters, float(diff)


def mode_split(times: list[np.ndarray], betas: list[float], asc: list[float]) -> list[np.ndarray]:
    """Compute multinomial logit shares per OD pair."""
    K = len(times)
    if K == 0:
        raise ValueError("at least one travel-time array is required")
    if len(betas) != K or len(asc) != K:
        raise ValueError("times, betas, and asc must have identical lengths")
    time_arrays = [np.asarray(values, dtype=np.float64) for values in times]
    shape = time_arrays[0].shape
    if any(values.shape != shape for values in time_arrays):
        raise ValueError("all travel-time arrays must have identical shapes")
    if any(not np.all(np.isfinite(values)) for values in time_arrays):
        raise ValueError("travel times must contain only finite values")
    coefficients = np.asarray(betas, dtype=np.float64)
    constants = np.asarray(asc, dtype=np.float64)
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(constants)):
        raise ValueError("betas and asc must contain only finite values")

    utils = []
    for k in range(K):
        u = constants[k] + coefficients[k] * time_arrays[k]
        utils.append(u)

    max_util = np.maximum.reduce(utils)

    exps = []
    for k in range(K):
        exps.append(np.exp(utils[k] - max_util))

    total_exp = sum(exps)
    total_exp[total_exp < 1e-12] = 1e-12

    shares = []
    for k in range(K):
        shares.append(exps[k] / total_exp)

    return shares
