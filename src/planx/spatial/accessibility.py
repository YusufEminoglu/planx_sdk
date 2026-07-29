# -*- coding: utf-8 -*-
"""Spatial accessibility engines for urban planning and resilience."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

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
            f"destinations_weight length must match the number of destinations ({d.shape[1]})."
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
            f"destinations_weight length must match the number of destinations ({d.shape[1]})."
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


def service_area_coverage(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    facilities: np.ndarray,
    thresholds: List[float],
    node_population: Optional[np.ndarray] = None,
) -> Dict[float, Dict[str, Union[np.ndarray, float]]]:
    """Calculates network-based service areas (isochrones) and population coverage.

    Finds the reachable nodes and total covered population from facility locations
    within multiple travel cost/distance thresholds.

    Args:
        indptr: CSR indptr array of shape (n + 1,)
        adj: CSR adj array of shape (E,)
        weights: CSR edge weights array of shape (E,)
        n: Number of nodes in the graph.
        facilities: 1D array of facility node indices.
        thresholds: List of cost/distance thresholds sorted in ascending order.
        node_population: Optional 1D array of shape (n,) containing population at each node.

    Returns:
        A dictionary mapping each threshold value to a dictionary containing:
          - "reachable_nodes": 1D NumPy array of node indices reachable within the threshold.
          - "population_covered": Float representing the total population covered.
          - "coverage_fraction": Float representing the fraction of total population covered.
    """
    fac = np.asarray(facilities, dtype=np.int64)
    if len(fac) == 0:
        raise ValueError("facilities list/array cannot be empty")

    if node_population is None:
        pop = np.ones(n, dtype=np.float64)
    else:
        pop = np.asarray(node_population, dtype=np.float64)
        if pop.shape != (n,):
            raise ValueError(f"node_population shape {pop.shape} must match number of nodes {n}")

    total_pop = float(np.sum(pop))
    if total_pop <= 0:
        total_pop = 1.0

    sorted_thresholds = sorted(thresholds)
    max_threshold = sorted_thresholds[-1] if sorted_thresholds else 0.0

    # Run multi_source Dijkstra from facilities up to max_threshold
    from .paths import multi_source

    dists, _ = multi_source(indptr, adj, weights, n, fac, cutoff=max_threshold)

    results: Dict[float, Dict[str, Union[np.ndarray, float]]] = {}
    for t in thresholds:
        t_val = float(t)
        reachable = (dists <= t_val) & np.isfinite(dists)
        reachable_nodes = np.where(reachable)[0]
        pop_covered = np.sum(pop[reachable])

        results[t_val] = {
            "reachable_nodes": reachable_nodes,
            "population_covered": float(pop_covered),
            "coverage_fraction": float(pop_covered / total_pop),
        }

    return results


def huff_gravity_model(
    dists: np.ndarray,
    destinations_weight: np.ndarray,
    decay_method: str = "power",
    exponent: float = 2.0,
    beta: float = 0.05,
) -> np.ndarray:
    """Calculates choice probabilities for origins choosing destinations using
    the Huff Gravity Model.

    Formula: P_ij = (W_j * f(d_ij)) / Sum_k (W_k * f(d_ik))

    Args:
        dists: NumPy array of shape (M, N) containing distances/costs from M
            origins to N destinations.
        destinations_weight: NumPy array of shape (N,) containing attractiveness/
            capacity weights of N destinations.
        decay_method: One of 'power' or 'exponential'.
        exponent: Friction exponent parameter for 'power' decay (e.g. 2.0).
            f(d) = d ** (-exponent)
        beta: Decay parameter for 'exponential' decay.
            f(d) = exp(-beta * d)

    Returns:
        NumPy array of shape (M, N) containing the probability of each origin M
        choosing destination N. Each row sums to 1.0 (or 0.0 if all destinations
        are inaccessible).
    """
    d = np.asarray(dists, dtype=np.float64)
    w = np.asarray(destinations_weight, dtype=np.float64)

    if d.ndim != 2:
        raise ValueError("dists must be a 2D array of shape (M, N)")
    if w.ndim != 1 or w.shape[0] != d.shape[1]:
        raise ValueError(
            f"destinations_weight length must match the number of destinations ({d.shape[1]})."
        )

    # Calculate decay factor
    decay = np.zeros_like(d)
    method_lower = decay_method.lower().replace(" ", "_").replace("-", "_")

    with np.errstate(divide="ignore", invalid="ignore"):
        if method_lower == "power":
            # Avoid division by zero by setting small distance to epsilon
            safe_d = np.where(d > 0, d, 1e-9)
            decay = safe_d ** (-exponent)
        elif method_lower == "exponential":
            decay = np.exp(-beta * d)
        else:
            raise ValueError(f"Unknown decay method: {decay_method}")

    # Set decay to 0 for infinite distances
    decay[~np.isfinite(d)] = 0.0

    # Calculate utility: W_j * f(d_ij)
    utility = decay * w[None, :]

    # Sum of utilities for each origin (row sum)
    row_sum = np.sum(utility, axis=1, keepdims=True)

    # Calculate probabilities
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = utility / row_sum
        probs = np.where(row_sum > 0, probs, 0.0)

    return probs


def kernel_density_2sfca(
    dists: np.ndarray,
    supply: np.ndarray,
    demand: np.ndarray,
    cutoff: float,
    kernel: str = "quartic",
) -> np.ndarray:
    """Calculates spatial accessibility using the Kernel Density 2-Step
    Floating Catchment Area (KD2SFCA) method.

    KD2SFCA uses continuous kernel functions (e.g. quartic or Gaussian) to weight
    demand and supply within a catchment cutoff distance d0.

    Args:
        dists: NumPy array of shape (M, N) containing distances/costs from M
            origins (demand points) to N destinations (facilities/supply points).
        supply: NumPy array of shape (N,) containing capacity/supply values of N destinations.
        demand: NumPy array of shape (M,) containing population/demand at M origins.
        cutoff: Catchment threshold distance (d0).
        kernel: Kernel type: 'quartic', 'gaussian', or 'epanechnikov'.

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

    # Ratio within cutoff
    ratio = d / cutoff
    mask = (d <= cutoff) & np.isfinite(d)

    kernel_lower = kernel.lower().replace(" ", "_").replace("-", "_")
    W = np.zeros_like(d)

    with np.errstate(divide="ignore", invalid="ignore"):
        if kernel_lower == "quartic":
            W = (15.0 / 16.0) * (1.0 - ratio**2) ** 2
        elif kernel_lower == "gaussian":
            num = np.exp(-0.5 * ratio**2) - np.exp(-0.5)
            den = 1.0 - np.exp(-0.5)
            W = num / den
        elif kernel_lower == "epanechnikov":
            W = 0.75 * (1.0 - ratio**2)
        else:
            raise ValueError(f"Unknown kernel type: {kernel}")

    # Enforce cutoff
    W[~mask] = 0.0
    W[~np.isfinite(d)] = 0.0
    W = np.clip(W, 0.0, None)

    # Step 1: Compute weighted demand at each supply location
    weighted_demand = np.sum(W * p[:, None], axis=0)

    # Calculate R_j (supply-to-demand ratio)
    R = np.zeros(n, dtype=np.float64)
    valid_demand = weighted_demand > 0.0
    R[valid_demand] = s[valid_demand] / weighted_demand[valid_demand]

    # Step 2: Sum supply-to-demand ratios at each origin
    A = np.sum(W * R[None, :], axis=1)

    return A


def three_step_2sfca(
    dists: np.ndarray,
    supply: np.ndarray,
    demand: np.ndarray,
    cutoff: float,
    decay_method: str = "none",
    beta: float = 1.0,
) -> np.ndarray:
    """Calculates spatial accessibility using the Three-Step Floating Catchment Area (3SFCA) method.

    Introduces selection probabilities based on distance competition between multiple facilities
    to address 2SFCA demand overestimation.

    Args:
        dists: NumPy array of shape (M, N) containing distances/costs from M
            origins (demand points) to N destinations (facilities/supply points).
        supply: NumPy array of shape (N,) containing capacity/supply values.
        demand: NumPy array of shape (M,) containing population/demand values.
        cutoff: Catchment threshold distance (d0).
        decay_method: Weight decay function: 'gaussian', 'exponential', 'linear', or 'none'.
        beta: Decay parameter.

    Returns:
        1D NumPy array of shape (M,) containing accessibility scores.
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

    mask = (d <= cutoff) & np.isfinite(d)
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

    W[~mask] = 0.0
    W[~np.isfinite(d)] = 0.0

    # Step 1: Calculate Selection Probability G_ij (M, N)
    # Sum of W_ik across destinations for each origin
    sum_w_orig = np.sum(W, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(sum_w_orig > 0, W / sum_w_orig, 0.0)

    # Step 2: Compute weighted demand at supply locations R_j
    # sum_k (G_kj * P_k * W_kj) -> shape (N,)
    weighted_demand = np.sum(G * p[:, None] * W, axis=0)

    R = np.zeros(n, dtype=np.float64)
    valid_demand = weighted_demand > 0.0
    R[valid_demand] = s[valid_demand] / weighted_demand[valid_demand]

    # Step 3: Sum selection-adjusted supply-to-demand ratios at each origin
    # A_i = sum_j (G_ij * R_j * W_ij) -> shape (M,)
    A = np.sum(G * R[None, :] * W, axis=1)

    return A


def calculate_15m_city_score(
    amenity_distances: np.ndarray,
    amenity_categories: list[str],
    category_weights: dict[str, float],
    max_threshold: float = 1200.0,
) -> np.ndarray:
    """Calculates the 15-Minute City accessibility index (0-100) for location points.

    Measures the degree to which essential amenities are reachable within a given walking
    distance threshold, weighted by category importances.

    For each origin i:
        Score_i = sum_k (Weight_k * I(distance_i,k <= max_threshold)) * 100
        where I is the indicator function (1.0 if reachable, else 0.0).

    Args:
        amenity_distances: NumPy array of shape (M, N) containing distances (in meters)
            from M origins to N nearest amenities.
        amenity_categories: List of strings of length N matching the category label
            of each amenity.
        category_weights: Dictionary mapping category labels to importance weights
            (will be normalized).
        max_threshold: Distance threshold representing the maximum walking limit (default: 1200m).

    Returns:
        1D NumPy array of shape (M,) containing 15-minute city scores in range [0, 100].
    """
    dists = np.asarray(amenity_distances, dtype=np.float64)
    if dists.ndim != 2:
        raise ValueError("amenity_distances must be a 2D array of shape (M, N)")

    m, n = dists.shape

    if len(amenity_categories) != n:
        raise ValueError(
            f"amenity_categories length ({len(amenity_categories)}) "
            f"must match columns of dists ({n})"
        )

    if n == 0:
        return np.zeros(m, dtype=np.float64)

    # Group categories and compute weights per column
    col_weights = np.zeros(n, dtype=np.float64)

    # Normalize category weights dictionary
    unique_cats = set(amenity_categories)
    sum_w = sum(category_weights.get(cat, 0.0) for cat in unique_cats)
    if sum_w <= 0.0:
        norm_weights = {cat: 1.0 / len(unique_cats) for cat in unique_cats}
    else:
        norm_weights = {cat: category_weights.get(cat, 0.0) / sum_w for cat in unique_cats}

    # Count amenities per category to split weights equally within category members
    cat_counts: dict[str, int] = {}
    for cat in amenity_categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    for idx, cat in enumerate(amenity_categories):
        col_weights[idx] = norm_weights.get(cat, 0.0) / cat_counts[cat]

    # Indicator matrix (M, N) indicating if amenity is within threshold
    within_threshold = (dists <= max_threshold) & np.isfinite(dists)

    # Compute weighted score (M,)
    scores = np.sum(within_threshold * col_weights[None, :], axis=1) * 100.0

    return np.clip(scores, 0.0, 100.0)


def transit_frequency_accessibility(
    demand_coords: np.ndarray,
    stop_coords: np.ndarray,
    headways_minutes: np.ndarray,
    num_routes: np.ndarray,
    catchment_radius: float = 800.0,
    decay_function: str = "gaussian",
    headway_benchmark: float = 10.0,
    route_diversity_weight: float = 0.3,
) -> dict[str, np.ndarray]:
    """Calculates the Public Transit Frequency Accessibility Index.

    Args:
        demand_coords: NumPy array of shape (N, 2) containing demand point coordinates.
        stop_coords: NumPy array of shape (S, 2) containing transit stop coordinates.
        headways_minutes: NumPy array of shape (S,) containing average headway in minutes per stop.
        num_routes: NumPy array of shape (S,) containing number of distinct routes
            serving each stop.
        catchment_radius: Maximum distance threshold in coordinate units.
        decay_function: One of 'gaussian', 'exponential', or 'linear'.
        headway_benchmark: Ideal headway in minutes.
        route_diversity_weight: Weight for route diversity component, between 0.0 and 1.0.

    Returns:
        A dictionary containing:
            - 'accessibility_index': (N,) float array in [0, 1]
            - 'num_stops_in_catchment': (N,) int array
            - 'nearest_stop_distance': (N,) float array
            - 'mean_headway_in_catchment': (N,) float array, NaN if no stops reachable
    """
    from scipy.spatial.distance import cdist

    dem = np.asarray(demand_coords, dtype=np.float64)
    stops = np.asarray(stop_coords, dtype=np.float64)
    headways = np.asarray(headways_minutes, dtype=np.float64)
    routes = np.asarray(num_routes, dtype=np.int64)

    if dem.ndim != 2 or dem.shape[1] != 2:
        raise ValueError(f"demand_coords must be a 2D array of shape (N, 2), got {dem.shape}")
    if stops.ndim != 2 or stops.shape[1] != 2:
        raise ValueError(f"stop_coords must be a 2D array of shape (S, 2), got {stops.shape}")

    s_count = stops.shape[0]
    if headways.ndim != 1 or headways.shape[0] != s_count:
        raise ValueError(
            f"headways_minutes length ({headways.shape[0]}) must match number of stops ({s_count})"
        )
    if routes.ndim != 1 or routes.shape[0] != s_count:
        raise ValueError(
            f"num_routes length ({routes.shape[0]}) must match number of stops ({s_count})"
        )
    if np.any(headways <= 0):
        raise ValueError("headways_minutes must be > 0")
    if np.any(routes < 1):
        raise ValueError("num_routes must be >= 1")
    if catchment_radius <= 0:
        raise ValueError("catchment_radius must be > 0")
    if headway_benchmark <= 0:
        raise ValueError("headway_benchmark must be > 0")
    if not (0.0 <= route_diversity_weight <= 1.0):
        raise ValueError("route_diversity_weight must be between 0 and 1")

    if s_count == 0:
        n_count = dem.shape[0]
        return {
            "accessibility_index": np.zeros(n_count, dtype=np.float64),
            "num_stops_in_catchment": np.zeros(n_count, dtype=int),
            "nearest_stop_distance": np.full(n_count, np.inf),
            "mean_headway_in_catchment": np.full(n_count, np.nan),
        }

    dists = cdist(dem, stops, metric="euclidean")
    w = np.zeros_like(dists)
    decay_lower = decay_function.lower().replace(" ", "_").replace("-", "_")
    beta = catchment_radius / 3.0
    mask = dists <= catchment_radius

    if decay_lower == "gaussian":
        w = np.exp(-0.5 * (dists / beta) ** 2)
    elif decay_lower == "exponential":
        w = np.exp(-dists / beta)
    elif decay_lower == "linear":
        w = np.clip(1.0 - (dists / catchment_radius), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown decay_function: {decay_function}")

    w[~mask] = 0.0
    f_s = np.clip(headway_benchmark / headways, 0.0, 1.0)
    r_s = 1.0 - np.exp(-routes / 3.0)
    q_s = (1.0 - route_diversity_weight) * f_s + route_diversity_weight * r_s
    sum_w = np.sum(w, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        a_i = np.where(sum_w > 0, np.sum(w * q_s[None, :], axis=1) / sum_w, 0.0)
        mean_headway = np.where(sum_w > 0, np.sum(w * headways[None, :], axis=1) / sum_w, np.nan)

    return {
        "accessibility_index": a_i,
        "num_stops_in_catchment": np.sum(mask, axis=1).astype(int),
        "nearest_stop_distance": np.min(dists, axis=1),
        "mean_headway_in_catchment": mean_headway,
    }


def network_voronoi_allocation(
    graph_sparse: np.ndarray,
    facility_indices: np.ndarray,
    demand_values: Optional[np.ndarray] = None,
    impedance_cutoff: Optional[float] = None,
) -> Dict[str, Union[np.ndarray, float]]:
    """Allocates every node in a sparse graph to its nearest facility using
    shortest-path network Voronoi tessellation.

    Args:
        graph_sparse: (V, V) scipy sparse CSR matrix of edge weights (distances/costs).
            Can also be a dense numpy array which we convert.
        facility_indices: (F,) array of node indices that are facilities (0-based).
        demand_values: Optional (V,) array of demand/population at each node.
            Default is uniform (all ones).
        impedance_cutoff: Optional maximum travel cost. Nodes beyond this from all
            facilities are unassigned.

    Returns:
        Dict with keys:
        - `assigned_facility`: (V,) int array, facility index each node is assigned to
          (-1 = unassigned)
        - `travel_cost`: (V,) float array, shortest distance to assigned facility
          (np.inf if unassigned)
        - `facility_demand`: (F,) float array, total demand served by each facility
        - `facility_node_count`: (F,) int array, number of nodes assigned to each facility
        - `facility_mean_cost`: (F,) float array, mean travel cost per facility
        - `facility_max_cost`: (F,) float array, max travel cost per facility
        - `coverage_ratio`: float, fraction of total demand within cutoff (1.0 if no cutoff)
    """
    from scipy.sparse import csr_matrix, issparse
    from scipy.sparse.csgraph import shortest_path

    if issparse(graph_sparse):
        g = graph_sparse.tocsr()  # type: ignore[attr-defined]
    else:
        g_arr = np.asarray(graph_sparse, dtype=np.float64)
        if g_arr.ndim != 2 or g_arr.shape[0] != g_arr.shape[1]:
            raise ValueError("graph_sparse must be a 2D square matrix (V, V)")
        g = csr_matrix(g_arr)

    v = g.shape[0]
    fac = np.asarray(facility_indices, dtype=np.int64)
    if fac.ndim != 1:
        raise ValueError("facility_indices must be a 1D array")
    if fac.size == 0:
        raise ValueError("facility_indices must have at least 1 facility")
    if np.any((fac < 0) | (fac >= v)):
        raise ValueError("facility_indices must be in range [0, V)")

    if demand_values is None:
        dem = np.ones(v, dtype=np.float64)
    else:
        dem = np.asarray(demand_values, dtype=np.float64)
        if dem.ndim != 1 or dem.shape[0] != v:
            raise ValueError(f"demand_values must be 1D with length V ({v})")
        if np.any(dem < 0):
            raise ValueError("demand_values must be non-negative")

    if impedance_cutoff is not None and impedance_cutoff <= 0:
        raise ValueError("impedance_cutoff must be > 0")

    dist_matrix = shortest_path(
        csgraph=g,
        method="D",
        directed=True,
        return_predecessors=False,
        unweighted=False,
        indices=fac,
    )

    min_dist_idx = np.argmin(dist_matrix, axis=0)
    travel_cost = np.min(dist_matrix, axis=0)

    assigned_facility = min_dist_idx.copy()

    if impedance_cutoff is not None:
        unassigned_mask = travel_cost > impedance_cutoff
        assigned_facility[unassigned_mask] = -1
        travel_cost[unassigned_mask] = np.inf
    else:
        unassigned_mask = np.isinf(travel_cost)
        assigned_facility[unassigned_mask] = -1

    f = len(fac)
    facility_demand = np.zeros(f, dtype=np.float64)
    facility_node_count = np.zeros(f, dtype=np.int64)
    facility_mean_cost = np.zeros(f, dtype=np.float64)
    facility_max_cost = np.zeros(f, dtype=np.float64)

    for i in range(f):
        assigned_nodes = assigned_facility == i
        node_count = np.sum(assigned_nodes)
        facility_node_count[i] = node_count

        if node_count > 0:
            facility_demand[i] = np.sum(dem[assigned_nodes])
            facility_mean_cost[i] = np.mean(travel_cost[assigned_nodes])
            facility_max_cost[i] = np.max(travel_cost[assigned_nodes])
        else:
            facility_mean_cost[i] = np.nan
            facility_max_cost[i] = np.nan

    total_demand = np.sum(dem)
    if total_demand > 0:
        assigned_demand = np.sum(facility_demand)
        coverage_ratio = float(assigned_demand / total_demand)
    else:
        coverage_ratio = 1.0

    return {
        "assigned_facility": assigned_facility,
        "travel_cost": travel_cost,
        "facility_demand": facility_demand,
        "facility_node_count": facility_node_count,
        "facility_mean_cost": facility_mean_cost,
        "facility_max_cost": facility_max_cost,
        "coverage_ratio": coverage_ratio,
    }


def healthcare_equity_index(
    demand_coords: np.ndarray,
    facility_coords: np.ndarray,
    facility_capacities: np.ndarray,
    population_groups: np.ndarray,
    group_weights: np.ndarray,
    catchment_distance: float = 5000.0,
) -> Dict[str, Any]:
    """Computes socio-spatial healthcare accessibility equity across vulnerable demographic groups.

    Uses E2SFCA and Gini/Atkinson equity decomposition.

    Args:
        demand_coords: NumPy array of shape (N, 2) containing demand point locations.
        facility_coords: NumPy array of shape (M, 2) containing healthcare facility locations.
        facility_capacities: NumPy array of shape (M,) with capacity/beds/staff per facility (> 0).
        population_groups: NumPy array of shape (N, G) containing population counts for G
            demographic/vulnerability groups (e.g., elderly, low-income).
        group_weights: NumPy array of shape (G,) with vulnerability weighting factors (>= 0).
        catchment_distance: Maximum catchment distance threshold. Defaults to 5000.0.

    Returns:
        Dictionary containing:
            - `accessibility_scores`: (N,) float array of accessibility scores A_i.
            - `weighted_accessibility`: (N,) float array of weighted accessibility.
            - `gini_coefficient`: Float representing the overall spatial Gini coefficient (0 to 1).
            - `group_accessibility_mean`: (G,) float array of mean access per group.
            - `group_deficit`: (G,) float array of relative deficit per group.
            - `equity_index`: Float overall score in [0, 1] (1 = perfectly equitable access).
    """
    from scipy.spatial.distance import cdist

    dem_coords = np.asarray(demand_coords, dtype=np.float64)
    fac_coords = np.asarray(facility_coords, dtype=np.float64)
    caps = np.asarray(facility_capacities, dtype=np.float64)
    pops = np.asarray(population_groups, dtype=np.float64)
    weights = np.asarray(group_weights, dtype=np.float64)

    if dem_coords.ndim != 2 or dem_coords.shape[1] != 2:
        raise ValueError("demand_coords must be of shape (N, 2)")
    if fac_coords.ndim != 2 or fac_coords.shape[1] != 2:
        raise ValueError("facility_coords must be of shape (M, 2)")

    n_demands = dem_coords.shape[0]
    n_facs = fac_coords.shape[0]

    if caps.shape != (n_facs,):
        raise ValueError("facility_capacities must be of shape (M,)")
    if np.any(caps <= 0):
        raise ValueError("facility_capacities must be positive (> 0)")

    if pops.ndim != 2 or pops.shape[0] != n_demands:
        raise ValueError("population_groups must be of shape (N, G)")
    if np.any(pops < 0):
        raise ValueError("population_groups must be non-negative")

    n_groups = pops.shape[1]
    if weights.shape != (n_groups,):
        raise ValueError("group_weights must be of shape (G,)")
    if np.any(weights < 0):
        raise ValueError("group_weights must be non-negative")

    if catchment_distance <= 0:
        raise ValueError("catchment_distance must be positive")

    # Step 1: E2SFCA Access Ratios
    dists = cdist(dem_coords, fac_coords)

    # Distance decay zones
    W = np.zeros_like(dists)
    c1 = 0.33 * catchment_distance
    c2 = 0.66 * catchment_distance
    c3 = 1.0 * catchment_distance

    mask1 = dists <= c1
    mask2 = (dists > c1) & (dists <= c2)
    mask3 = (dists > c2) & (dists <= c3)

    W[mask1] = 1.0
    W[mask2] = 0.6
    W[mask3] = 0.2

    pop_total = np.sum(pops, axis=1)

    # Compute provider-to-population ratio R_m for each facility m
    weighted_demand = np.sum(W * pop_total[:, None], axis=0)
    R = np.zeros(n_facs, dtype=np.float64)
    valid_demand = weighted_demand > 0.0
    R[valid_demand] = caps[valid_demand] / weighted_demand[valid_demand]

    # Compute accessibility score A_i for each demand point i
    A_i = np.sum(W * R[None, :], axis=1)

    # Step 2: Equity & Mismatch Decomposition
    # Weighted accessibility score per location considering vulnerability
    vul_weight_i = np.sum(pops * weights[None, :], axis=1)
    A_weighted_i = np.zeros_like(A_i)
    valid_vul = vul_weight_i > 0.0
    A_weighted_i[valid_vul] = A_i[valid_vul] / vul_weight_i[valid_vul]

    # Overall Spatial Gini Coefficient of A_i across population
    gini = spatial_equity_gini(A_i, pop_total)
    equity_index = 1.0 - gini

    # Group Accessibility Deficit
    total_mean = np.average(A_i, weights=pop_total) if np.sum(pop_total) > 0 else 0.0

    group_means = np.zeros(n_groups, dtype=np.float64)
    group_deficits = np.zeros(n_groups, dtype=np.float64)

    for g in range(n_groups):
        pop_g = pops[:, g]
        sum_pop_g = np.sum(pop_g)
        if sum_pop_g > 0:
            mean_g = np.average(A_i, weights=pop_g)
            group_means[g] = mean_g
            if total_mean > 0:
                group_deficits[g] = (total_mean - mean_g) / total_mean
            else:
                group_deficits[g] = 0.0
        else:
            group_means[g] = 0.0
            group_deficits[g] = 0.0

    return {
        "accessibility_scores": A_i,
        "weighted_accessibility": A_weighted_i,
        "gini_coefficient": float(gini),
        "group_accessibility_mean": group_means,
        "group_deficit": group_deficits,
        "equity_index": float(equity_index),
    }
