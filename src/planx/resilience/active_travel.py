# -*- coding: utf-8 -*-
"""Urban active transport equity, spatial mismatch, and accessibility metrics.

Features inspired by GSD Urban Theory Lab (Harvard), Center for Geographic Analysis
(Harvard), and the Transport Studies Unit (Oxford).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def job_housing_spatial_mismatch(
    residential_pop: np.ndarray,
    job_capacity: np.ndarray,
    travel_cost_matrix: np.ndarray,
    cutoff: float,
    decay_method: str = "linear",
) -> np.ndarray:
    """Calculates the Job-Housing Spatial Mismatch Index (SMI) for each zone.

    Measures the spatial mismatch between the local residential labor supply and low-skill
    or service job accessibility using active travel time/distance.

    Formula:
        AccessJobs_i = Sum_j (Job_j * f(d_ij))
        AccessPop_i = Sum_j (Pop_j * f(d_ij))
        SMI_i = ln( (AccessJobs_i + epsilon) / (AccessPop_i + epsilon) )

    Args:
        residential_pop: 1D array of shape (M,) representing zone population.
        job_capacity: 1D array of shape (N,) representing zone jobs.
        travel_cost_matrix: 2D array of shape (M, N) containing active travel costs.
        cutoff: Travel cost threshold (e.g. 15-minute walking distance).
        decay_method: 'exponential', 'linear', or 'none'.

    Returns:
        1D NumPy array of shape (M,) containing zone spatial mismatch index.
    """
    pop = np.asarray(residential_pop, dtype=np.float64)
    jobs = np.asarray(job_capacity, dtype=np.float64)
    dists = np.asarray(travel_cost_matrix, dtype=np.float64)

    m, n = dists.shape
    if pop.shape != (m,):
        raise ValueError(f"residential_pop shape ({pop.shape}) must match matrix rows ({m})")
    if jobs.shape != (n,):
        raise ValueError(f"job_capacity shape ({jobs.shape}) must match matrix columns ({n})")
    if cutoff <= 0:
        raise ValueError("cutoff must be greater than 0")

    mask = (dists <= cutoff) & np.isfinite(dists)
    decay = np.zeros_like(dists)
    method_lower = decay_method.lower().replace(" ", "_").replace("-", "_")

    if method_lower in ("none", "uniform"):
        decay = np.ones_like(dists)
    elif method_lower == "linear":
        decay = 1.0 - (dists / cutoff)
        decay = np.clip(decay, 0.0, 1.0)
    elif method_lower == "exponential":
        beta = 3.0 / cutoff
        decay = np.exp(-beta * dists)
    else:
        raise ValueError(f"Unknown decay method: {decay_method}")

    decay[~mask] = 0.0
    decay[~np.isfinite(dists)] = 0.0

    # Calculate accessibility
    # AccessJobs = Sum_j (Job_j * f(d_ij)) -> shape (M,)
    access_jobs = np.sum(decay * jobs[None, :], axis=1)

    # AccessPop = Sum_j (Pop_j * f(d_ij)) -> shape (M,)
    # For population accessibility, we sum population from all zones i to j
    access_pop = np.sum(decay * pop[None, :], axis=1)

    epsilon = 1e-6
    smi = np.log((access_jobs + epsilon) / (access_pop + epsilon))
    return smi


def active_travel_equity_gini(
    accessibility_scores: np.ndarray,
    population_weights: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculates active travel accessibility inequality using Gini coefficient and Lorenz curve.

    Args:
        accessibility_scores: 1D array of active travel accessibility index.
        population_weights: 1D array of population per zone.

    Returns:
        Tuple containing:
            - gini: Gini coefficient float [0.0, 1.0].
            - cum_pop: 1D array of cumulative population share (Lorenz X-axis).
            - cum_access: 1D array of cumulative accessibility share (Lorenz Y-axis).
    """
    acc = np.asarray(accessibility_scores, dtype=np.float64)
    pop = np.asarray(population_weights, dtype=np.float64)

    if len(acc) != len(pop):
        raise ValueError("accessibility_scores and population_weights must have identical length")

    total_pop = np.sum(pop)
    if total_pop <= 0:
        return 0.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # Sort both arrays by accessibility score ascending
    idx = np.argsort(acc)
    sorted_acc = acc[idx]
    sorted_pop = pop[idx]

    # Cumulative shares
    cum_pop = np.cumsum(sorted_pop) / total_pop
    total_acc_weighted = np.sum(sorted_acc * sorted_pop)

    if total_acc_weighted <= 0:
        return 0.0, np.array([0.0, 1.0]), np.array([0.0, 1.0])

    cum_access = np.cumsum(sorted_acc * sorted_pop) / total_acc_weighted

    # Insert origin (0, 0)
    cum_pop = np.insert(cum_pop, 0, 0.0)
    cum_access = np.insert(cum_access, 0, 0.0)

    # Calculate Gini via area under the Lorenz Curve (trapezoidal rule)
    # Area = Sum_i 0.5 * (cum_access_i + cum_access_{i-1}) * (cum_pop_i - cum_pop_{i-1})
    dx = np.diff(cum_pop)
    y_mean = (cum_access[1:] + cum_access[:-1]) / 2.0
    area = np.sum(dx * y_mean)

    gini = float(1.0 - 2.0 * area)
    return max(0.0, min(1.0, gini)), cum_pop, cum_access


def transport_mismatch_index(
    accessibility: np.ndarray,
    vulnerable_population: np.ndarray,
) -> float:
    """Computes vulnerable population mismatch index against active travel accessibility.

    Scores how much transport-deprived (low accessibility) zones are populated
    by vulnerable groups (e.g. low income, elderly, carless).

    Formula:
        Mismatch = Sum ( Vuln_i * (100.0 - Normalise(Access_i)) ) / Sum ( Vuln_i )

    Args:
        accessibility: 1D array of active travel accessibility.
        vulnerable_population: 1D array of vulnerable population weights.

    Returns:
        Index score [0.0 to 100.0] representing vulnerable population mismatch (higher is worse).
    """
    acc = np.clip(np.asarray(accessibility, dtype=np.float64), 0.0, None)
    vuln = np.asarray(vulnerable_population, dtype=np.float64)

    if len(acc) != len(vuln):
        raise ValueError("accessibility and vulnerable_population must have identical length")

    sum_vuln = np.sum(vuln)
    if sum_vuln <= 0:
        return 0.0

    # Min-max normalize accessibility to [0.0, 100.0]
    min_acc = np.min(acc)
    max_acc = np.max(acc)
    diff = max_acc - min_acc
    if diff <= 0:
        diff = 1.0
    norm_acc = (acc - min_acc) / diff * 100.0

    # Calculate mismatch: weight vulnerable population by lack of accessibility
    mismatch_weights = 100.0 - norm_acc
    total_mismatch = np.sum(vuln * mismatch_weights)

    return float(total_mismatch / sum_vuln)
