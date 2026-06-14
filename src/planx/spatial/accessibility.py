# -*- coding: utf-8 -*-
"""Spatial accessibility engines for urban planning and resilience."""

from __future__ import annotations

from typing import Optional

import numpy as np


def gravity_accessibility(
    dists: np.ndarray,
    destinations_weight: np.ndarray,
    decay_method: str = "exponential",
    beta: float = 0.05,
    cutoff: Optional[float] = None,
) -> np.ndarray:
    """Calculates the gravity-based accessibility index (Hansen Index) for origins.

    Args:
        dists: NumPy array of shape (M, N) containing distances/costs from M
            origins to N destinations.
        destinations_weight: NumPy array of shape (N,) containing weights/
            attractiveness of destinations.
        decay_method: One of 'exponential', 'power', 'gaussian', or 'linear'.
        beta: Decay parameter (beta).
        cutoff: Optional maximum travel cost. Destinations beyond this cost are ignored.

    Returns:
        NumPy array of shape (M,) containing accessibility index for each origin.
    """
    d = np.asarray(dists, dtype=np.float64)
    w = np.asarray(destinations_weight, dtype=np.float64)

    if d.ndim != 2:
        raise ValueError("dists must be a 2D array of shape (M, N)")
    if w.ndim != 1 or w.shape[0] != d.shape[1]:
        raise ValueError(
            "destinations_weight length must match the number of " f"destinations ({d.shape[1]})."
        )

    # Apply cutoff if specified
    mask = np.ones_like(d, dtype=bool)
    if cutoff is not None:
        mask = d <= cutoff

    # Calculate decay factor
    decay = np.zeros_like(d)
    method_lower = decay_method.lower().replace(" ", "_").replace("-", "_")

    with np.errstate(divide="ignore", invalid="ignore"):
        if method_lower == "exponential":
            decay = np.exp(-beta * d)
        elif method_lower == "power":
            # Avoid division by zero
            safe_d = np.where(d > 0, d, 1e-9)
            decay = safe_d ** (-beta)
        elif method_lower == "gaussian":
            decay = np.exp(-0.5 * (d / beta) ** 2) if beta > 0 else np.zeros_like(d)
        elif method_lower == "linear":
            if cutoff is None or cutoff <= 0:
                raise ValueError("linear decay requires a positive cutoff value")
            decay = 1.0 - (d / cutoff)
            decay = np.clip(decay, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown decay method: {decay_method}")

    # Set decay to 0 for elements exceeding cutoff or which are infinite
    decay[~mask] = 0.0
    decay[~np.isfinite(d)] = 0.0

    # Accessibility index is the sum of weighted decay values
    return np.sum(decay * w[None, :], axis=1)


def cumulative_opportunities(
    dists: np.ndarray,
    destinations_weight: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Calculates the cumulative opportunities accessibility index.

    Args:
        dists: NumPy array of shape (M, N) containing distances/costs from M
            origins to N destinations.
        destinations_weight: NumPy array of shape (N,) containing weights/
            attractiveness of destinations.
        cutoff: Maximum travel cost threshold.

    Returns:
        NumPy array of shape (M,) containing the sum of opportunities within cutoff.
    """
    d = np.asarray(dists, dtype=np.float64)
    w = np.asarray(destinations_weight, dtype=np.float64)

    if d.ndim != 2:
        raise ValueError("dists must be a 2D array of shape (M, N)")
    if w.ndim != 1 or w.shape[0] != d.shape[1]:
        raise ValueError(
            "destinations_weight length must match the number of " f"destinations ({d.shape[1]})."
        )

    in_range = (d <= cutoff) & np.isfinite(d)
    return np.sum(in_range * w[None, :], axis=1)


def enhanced_2sfca(
    dists: np.ndarray,
    supply: np.ndarray,
    demand: np.ndarray,
    cutoff: float,
    decay_method: str = "none",
    beta: float = 1.0,
) -> np.ndarray:
    """Calculates spatial accessibility using the Enhanced 2-Step Floating Catchment Area method.

    E2SFCA measures accessibility to services (supply) by population (demand) within
    a threshold distance (cutoff), applying optional decay weights (Gaussian, Exponential,
    or Linear) based on distance.

    Args:
        dists: NumPy array of shape (M, N) containing distances/costs from M
            origins (demand points) to N destinations (facilities/supply points).
        supply: NumPy array of shape (N,) containing capacity/supply values of N destinations.
        demand: NumPy array of shape (M,) containing population/demand at M origins.
        cutoff: Catchment threshold distance (d0).
        decay_method: Weight decay function: 'gaussian', 'exponential', 'linear', or 'none'.
            If 'none', acts as standard 2SFCA (uniform weight within cutoff).
        beta: Decay function parameter (e.g., standard deviation for Gaussian, or rate for
            Exponential).

    Returns:
        NumPy array of shape (M,) containing the accessibility score for each origin.
    """
    d = np.asarray(dists, dtype=np.float64)
    s = np.asarray(supply, dtype=np.float64)
    p = np.asarray(demand, dtype=np.float64)

    m, n = d.shape
    if s.shape != (n,):
        raise ValueError(f"supply length ({s.shape[0]}) must match number of destinations ({n})")
    if p.shape != (m,):
        raise ValueError(f"demand length ({p.shape[0]}) must match number of origins ({m})")

    if cutoff <= 0:
        raise ValueError("cutoff must be greater than 0")

    # Catchment mask
    mask = (d <= cutoff) & np.isfinite(d)

    # Compute decay weights
    method_lower = decay_method.lower().replace(" ", "_").replace("-", "_")
    with np.errstate(divide="ignore", invalid="ignore"):
        if method_lower == "none":
            W = np.ones_like(d)
        elif method_lower == "gaussian":
            W = np.exp(-0.5 * (d / beta) ** 2) if beta > 0.0 else np.zeros_like(d)
        elif method_lower == "exponential":
            W = np.exp(-beta * d)
        elif method_lower == "linear":
            W = np.clip(1.0 - (d / cutoff), 0.0, 1.0)
        else:
            raise ValueError(f"Unknown decay method: {decay_method}")

    # Enforce cutoff and mask nans/infs
    W[~mask] = 0.0
    W[~np.isfinite(d)] = 0.0

    # Step 1: Compute weighted demand at each supply location
    # sum_k (P_k * W_kj) -> shape (N,)
    weighted_demand = np.sum(W * p[:, None], axis=0)

    # Calculate R_j (supply-to-demand ratio)
    R = np.zeros(n, dtype=np.float64)
    valid_demand = weighted_demand > 0.0
    R[valid_demand] = s[valid_demand] / weighted_demand[valid_demand]

    # Step 2: Sum supply-to-demand ratios at each origin
    # A_i = sum_j (R_j * W_ij) -> shape (M,)
    A = np.sum(W * R[None, :], axis=1)

    return A


def spatial_equity_gini(accessibility: np.ndarray, population: np.ndarray) -> float:
    """Calculates the population-weighted Gini coefficient of accessibility.

    A Gini coefficient of 0 indicates perfect equality (all individuals have
    identical accessibility), while 1 indicates perfect inequality.

    Args:
        accessibility: NumPy array of shape (M,) containing accessibility scores.
        population: NumPy array of shape (M,) containing population weights.

    Returns:
        Float value representing the Gini coefficient [0.0, 1.0].
    """
    a = np.asarray(accessibility, dtype=np.float64)
    p = np.asarray(population, dtype=np.float64)

    if len(a) != len(p):
        raise ValueError("accessibility and population arrays must have the same length")
    if np.sum(p) <= 0:
        return 0.0

    # Calculate the mean accessibility
    mean_a = np.average(a, weights=p)
    if mean_a <= 0:
        return 0.0

    # Vectorized double sum of absolute differences
    abs_diff = np.abs(a[:, None] - a[None, :])
    pop_prod = p[:, None] * p[None, :]

    numerator = np.sum(abs_diff * pop_prod)
    denominator = 2.0 * np.sum(p) * np.sum(p * a)

    if denominator <= 0:
        return 0.0

    return float(numerator / denominator)
