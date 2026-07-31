# -*- coding: utf-8 -*-
"""Origin-Destination (OD) Demand Matrix Estimation & Furness Matrix Balancing Engines."""

from __future__ import annotations

from typing import Any

import numpy as np


def gravity_model_od_estimation(
    origins_population: np.ndarray,
    destinations_attraction: np.ndarray,
    distance_matrix: np.ndarray,
    beta_decay: float = 0.002,
) -> dict[str, Any]:
    """Estimates Unconstrained / Doubly-Constrained Gravity Model OD Matrix.

    T_ij = k * (P_i^a * A_j^b) * exp(-beta * d_ij)

    Args:
        origins_population: 1D array of origin zone trip generations (P_i).
        destinations_attraction: 1D array of destination zone trip attractions (A_j).
        distance_matrix: 2D travel distance / cost matrix (N_o, N_d).
        beta_decay: Friction distance decay parameter.

    Returns:
        Dict containing estimated OD matrix (N_o, N_d), total trips, and mean trip length.
    """
    p = np.asarray(origins_population, dtype=np.float64)
    a = np.asarray(destinations_attraction, dtype=np.float64)
    d = np.asarray(distance_matrix, dtype=np.float64)

    friction = np.exp(-beta_decay * d)
    raw_od = np.outer(p, a) * friction

    total_p = np.sum(p)
    raw_sum = np.sum(raw_od)
    k = total_p / max(raw_sum, 1e-12) if raw_sum > 0 else 1.0

    od_matrix = raw_od * k
    mean_dist = float(np.sum(od_matrix * d) / max(np.sum(od_matrix), 1e-12))

    return {
        "od_matrix": od_matrix,
        "total_trips": float(np.sum(od_matrix)),
        "mean_trip_distance": mean_dist,
    }


def furness_matrix_balancing(
    initial_matrix: np.ndarray,
    origin_totals: np.ndarray,
    dest_totals: np.ndarray,
    max_iter: int = 30,
) -> dict[str, Any]:
    """Balances OD matrix to match target origin and destination totals (Furness / Fratar Method).

    Args:
        initial_matrix: 2D seed matrix (N_o, N_d).
        origin_totals: Target 1D array of row sums.
        dest_totals: Target 1D array of column sums.
        max_iter: Maximum balancing iterations.

    Returns:
        Dict containing balanced OD matrix, row error %, and col error %.
    """
    mat = np.asarray(initial_matrix, dtype=np.float64).copy()
    row_target = np.asarray(origin_totals, dtype=np.float64)
    col_target = np.asarray(dest_totals, dtype=np.float64)

    for _ in range(max_iter):
        # Row balancing
        row_sums = np.sum(mat, axis=1)
        row_factors = np.where(row_sums > 0, row_target / np.maximum(row_sums, 1e-12), 1.0)
        mat *= row_factors[:, None]

        # Column balancing
        col_sums = np.sum(mat, axis=0)
        col_factors = np.where(col_sums > 0, col_target / np.maximum(col_sums, 1e-12), 1.0)
        mat *= col_factors[None, :]

    row_err = float(np.mean(np.abs(np.sum(mat, axis=1) - row_target) / np.maximum(row_target, 1.0)))
    col_err = float(np.mean(np.abs(np.sum(mat, axis=0) - col_target) / np.maximum(col_target, 1.0)))

    return {
        "balanced_od_matrix": mat,
        "row_error_ratio": row_err,
        "col_error_ratio": col_err,
        "total_balanced_trips": float(np.sum(mat)),
    }
