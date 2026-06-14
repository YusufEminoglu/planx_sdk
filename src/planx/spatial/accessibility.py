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
