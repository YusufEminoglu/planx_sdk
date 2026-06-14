# -*- coding: utf-8 -*-
"""Facility location and optimal placement algorithms for suitability analysis."""

from __future__ import annotations

from typing import Optional

import numpy as np


def greedy_mclp(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_pop: np.ndarray,
    max_distance: float,
    k: int,
    existing_coords: Optional[np.ndarray] = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Solves the Maximal Covering Location Problem (MCLP) using a greedy heuristic.

    At each step, selects the candidate site that covers the largest amount
    of currently uncovered population within the max_distance.

    Args:
        candidate_coords: NumPy array of shape (C, 2) containing candidate site coordinates.
        demand_coords: NumPy array of shape (D, 2) containing demand point coordinates.
        demand_pop: NumPy array of shape (D,) containing population at each demand point.
        max_distance: Maximum distance for coverage.
        k: Number of facilities to select.
        existing_coords: Optional NumPy array of shape (E, 2) containing existing facility coords.

    Returns:
        Tuple of:
          - selected_indices: List of indices of selected candidate sites in order.
          - pop_added: NumPy array of shape (k_actual,) of additional population
            covered at each step.
          - cum_covered: NumPy array of shape (k_actual,) of cumulative population
            covered after each step.
    """
    cand = np.asarray(candidate_coords, dtype=np.float64)
    dem = np.asarray(demand_coords, dtype=np.float64)
    pop = np.asarray(demand_pop, dtype=np.float64)

    if cand.ndim != 2 or cand.shape[1] != 2:
        raise ValueError("candidate_coords must be of shape (C, 2)")
    if dem.ndim != 2 or dem.shape[1] != 2:
        raise ValueError("demand_coords must be of shape (D, 2)")
    if pop.ndim != 1 or pop.shape[0] != dem.shape[0]:
        raise ValueError("demand_pop must be a 1D array of length D")

    c_count = cand.shape[0]
    d_count = dem.shape[0]

    # Precompute pairwise distances between candidates and demand points: shape (C, D)
    dists = np.sqrt(
        (cand[:, None, 0] - dem[None, :, 0]) ** 2 + (cand[:, None, 1] - dem[None, :, 1]) ** 2
    )

    # Boolean matrix: True if candidate c covers demand d
    coverage_matrix = dists <= max_distance

    # Track coverage status of demand points (True if covered)
    covered = np.zeros(d_count, dtype=bool)

    # Pre-cover demand with existing shelters
    if existing_coords is not None and len(existing_coords) > 0:
        exist = np.asarray(existing_coords, dtype=np.float64)
        if exist.ndim != 2 or exist.shape[1] != 2:
            raise ValueError("existing_coords must be of shape (E, 2)")
        exist_dists = np.sqrt(
            (exist[:, None, 0] - dem[None, :, 0]) ** 2 + (exist[:, None, 1] - dem[None, :, 1]) ** 2
        )
        covered = np.any(exist_dists <= max_distance, axis=0)

    selected_indices: list[int] = []
    pop_added: list[float] = []
    cum_covered: list[float] = []

    for _ in range(k):
        gains = np.zeros(c_count)
        for c in range(c_count):
            if c in selected_indices:
                gains[c] = -1.0
                continue
            gains[c] = np.sum(pop[coverage_matrix[c] & ~covered])

        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]

        if best_gain <= 0:
            break

        selected_indices.append(best_idx)
        pop_added.append(best_gain)

        covered |= coverage_matrix[best_idx]
        cum_covered.append(np.sum(pop[covered]))

    return selected_indices, np.array(pop_added), np.array(cum_covered)
