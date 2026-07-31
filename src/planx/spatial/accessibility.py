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


def calculate_multimodal_15m_city(
    demand_coords: np.ndarray,
    amenity_coords_dict: dict[str, np.ndarray],
    modal_speeds_kmh: dict[str, float] | None = None,
    modal_weights: dict[str, float] | None = None,
    max_travel_time_minutes: float = 15.0,
) -> dict[str, Any]:
    """Computes multi-modal 15-minute city accessibility scores across urban services.

    Args:
        demand_coords: (N, 2) NumPy array of demand point locations.
        amenity_coords_dict: Dictionary mapping amenity category name to (M_c, 2) coordinates array.
        modal_speeds_kmh: Dictionary mapping mode to speed in km/h. Default:
            {'walk': 4.5, 'bike': 15.0, 'transit': 20.0}.
        modal_weights: Dictionary mapping mode to weight [0, 1] summing to 1. Default:
            {'walk': 0.6, 'bike': 0.3, 'transit': 0.1}.
        max_travel_time_minutes: Target threshold in minutes. Default: 15.0.

    Returns:
        Dictionary containing:
        - `city_15m_score`: (N,) float array in [0, 100] representing overall index.
        - `category_scores`: dict mapping category name -> (N,) float array in [0, 100].
        - `gini_equity_score`: float Gini coefficient of the scores.
        - `threshold_compliance_pct`: float percentage of locations meeting target score (>= 75.0).
    """
    from scipy.spatial.distance import cdist

    dem = np.asarray(demand_coords, dtype=np.float64)
    if dem.ndim != 2 or dem.shape[1] != 2:
        raise ValueError(f"demand_coords must be a 2D array of shape (N, 2), got {dem.shape}")

    n_points = dem.shape[0]
    if n_points == 0:
        raise ValueError("demand_coords cannot be empty")

    if not amenity_coords_dict:
        raise ValueError("amenity_coords_dict cannot be empty")

    if max_travel_time_minutes <= 0:
        raise ValueError("max_travel_time_minutes must be positive")

    if modal_speeds_kmh is None:
        modal_speeds_kmh = {"walk": 4.5, "bike": 15.0, "transit": 20.0}
    if modal_weights is None:
        modal_weights = {"walk": 0.6, "bike": 0.3, "transit": 0.1}

    for mode, speed in modal_speeds_kmh.items():
        if speed <= 0:
            raise ValueError(f"Speed for mode {mode} must be positive, got {speed}")

    weight_sum = 0.0
    for mode, weight in modal_weights.items():
        if weight < 0 or weight > 1:
            raise ValueError(f"Weight for mode {mode} must be between 0 and 1, got {weight}")
        weight_sum += weight

    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"Modal weights must sum to 1.0, got {weight_sum}")

    common_modes = set(modal_speeds_kmh.keys()).intersection(set(modal_weights.keys()))
    if not common_modes:
        raise ValueError("No common modes between speeds and weights")

    category_scores: dict[str, np.ndarray] = {}

    for cat, coords in amenity_coords_dict.items():
        coords_arr = np.asarray(coords, dtype=np.float64)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != 2:
            raise ValueError(f"Coordinates for category '{cat}' must be shape (M, 2)")
        if coords_arr.shape[0] == 0:
            raise ValueError(f"Coordinates for category '{cat}' cannot be empty")

        dists = cdist(dem, coords_arr, metric="euclidean")
        min_dist_m = np.min(dists, axis=1)
        min_dist_km = min_dist_m / 1000.0

        cat_score = np.zeros(n_points, dtype=np.float64)

        for mode in common_modes:
            speed = modal_speeds_kmh[mode]
            weight = modal_weights[mode]

            travel_time_min = (min_dist_km / speed) * 60.0
            mode_score = np.clip(1.0 - travel_time_min / max_travel_time_minutes, 0.0, 1.0)
            cat_score += weight * mode_score

        category_scores[cat] = cat_score * 100.0

    all_scores = np.stack(list(category_scores.values()), axis=0)
    city_15m_score = np.mean(all_scores, axis=0)

    threshold_compliance_pct = float(np.mean(city_15m_score >= 75.0) * 100.0)
    gini_equity_score = spatial_equity_gini(city_15m_score, np.ones(n_points, dtype=np.float64))

    return {
        "city_15m_score": city_15m_score,
        "category_scores": category_scores,
        "gini_equity_score": gini_equity_score,
        "threshold_compliance_pct": threshold_compliance_pct,
    }


def huff_retail_market_share(
    origin_coords: np.ndarray,
    store_coords: np.ndarray,
    store_attractiveness: np.ndarray,
    origin_populations: np.ndarray | None = None,
    distance_exponent: float = 2.0,
) -> dict[str, Any]:
    """Computes Huff Gravity Model market share probability matrices and expected retail sales.

    Args:
        origin_coords: NumPy array of shape (N, 2) containing demand / origin zone locations.
        store_coords: NumPy array of shape (M, 2) containing competing retail store locations.
        store_attractiveness: NumPy array of shape (M,) containing floor area / store score (> 0).
        origin_populations: Optional NumPy array of shape (N,) containing population /
            purchasing power per origin zone. Default is uniform (1.0).
        distance_exponent: Distance decay exponent lambda. Default is 2.0.

    Returns:
        Dict with keys:
            - 'probability_matrix': (N, M) float array of choice probabilities P_{i,j}.
            - 'store_captured_customers': (M,) float array of expected customers C_j.
            - 'store_market_shares': (M,) float array of total market share S_j [0, 1].
            - 'trade_area_zone_counts': (M,) int array count of zones with P >= 0.5.
    """
    from scipy.spatial.distance import cdist

    orig = np.asarray(origin_coords, dtype=np.float64)
    stores = np.asarray(store_coords, dtype=np.float64)
    attr = np.asarray(store_attractiveness, dtype=np.float64)

    if orig.ndim != 2 or orig.shape[1] != 2:
        raise ValueError(f"origin_coords must be a 2D array of shape (N, 2), got {orig.shape}")
    if stores.ndim != 2 or stores.shape[1] != 2:
        raise ValueError(f"store_coords must be a 2D array of shape (M, 2), got {stores.shape}")

    n_orig = orig.shape[0]
    n_stores = stores.shape[0]

    if attr.ndim != 1 or attr.shape[0] != n_stores:
        raise ValueError(
            f"store_attractiveness length ({attr.shape[0]}) must "
            f"match number of stores ({n_stores})"
        )

    if np.any(attr <= 0):
        raise ValueError("store_attractiveness must be > 0")

    if distance_exponent <= 0:
        raise ValueError("distance_exponent must be > 0")

    if origin_populations is None:
        pops = np.ones(n_orig, dtype=np.float64)
    else:
        pops = np.asarray(origin_populations, dtype=np.float64)
        if pops.ndim != 1 or pops.shape[0] != n_orig:
            raise ValueError(
                f"origin_populations length ({pops.shape[0]}) must "
                f"match number of origins ({n_orig})"
            )
        if np.any(pops < 0):
            raise ValueError("origin_populations must be non-negative")

    if n_orig == 0 or n_stores == 0:
        return {
            "probability_matrix": np.zeros((n_orig, n_stores), dtype=np.float64),
            "store_captured_customers": np.zeros(n_stores, dtype=np.float64),
            "store_market_shares": np.zeros(n_stores, dtype=np.float64),
            "trade_area_zone_counts": np.zeros(n_stores, dtype=int),
        }

    # Distance matrix D (N, M). Avoid division by zero by clipping D >= 1.0.
    dists = cdist(orig, stores, metric="euclidean")
    dists = np.clip(dists, 1.0, None)

    # Utility U_{i,j} = store_attractiveness_j / (D_{i,j} ^ distance_exponent)
    utility = attr[None, :] / (dists**distance_exponent)

    # Choice Probability Matrix P_{i,j} = U_{i,j} / sum_k(U_{i,k})
    sum_utility = np.sum(utility, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob_matrix = np.where(sum_utility > 0, utility / sum_utility, 0.0)

    # Expected customers per store C_j = sum_i(origin_populations_i * P_{i,j})
    captured = np.sum(pops[:, None] * prob_matrix, axis=0)

    # Total Market Share per store S_j = C_j / sum(origin_populations)
    total_pop = np.sum(pops)
    if total_pop > 0:
        market_shares = captured / total_pop
    else:
        market_shares = np.zeros(n_stores, dtype=np.float64)

    # Primary Trade Area (zones where P_{i,j} >= 0.50)
    trade_areas = np.sum(prob_matrix >= 0.50, axis=0).astype(int)

    return {
        "probability_matrix": prob_matrix,
        "store_captured_customers": captured,
        "store_market_shares": market_shares,
        "trade_area_zone_counts": trade_areas,
    }


def parking_spatial_mismatch_index(
    demand_coords: np.ndarray,
    parking_facility_coords: np.ndarray,
    parking_capacities: np.ndarray,
    zone_parking_demand: np.ndarray,
    walk_threshold_m: float = 400.0,
) -> dict[str, Any]:
    """Evaluates spatial mismatch between urban parking supply and parking demand.

    Uses a walking distance decay and capacity-constrained occupancy modeling to
    allocate effective reachable parking supply per zone.

    Args:
        demand_coords: NumPy array of shape (N, 2) containing origin/demand zone centroids.
        parking_facility_coords: NumPy array of shape (M, 2) containing parking facility locations.
        parking_capacities: NumPy array of shape (M,) containing total parking spaces
            per facility (> 0).
        zone_parking_demand: NumPy array of shape (N,) containing parking spaces
            required per zone (> 0).
        walk_threshold_m: Maximum comfortable walking distance in meters. Default is 400.0.

    Returns:
        Dictionary containing:
            - 'mismatch_ratios': (N,) float array R_i (supply / demand)
            - 'reachable_supply': (N,) float array S_i
            - 'deficit_zones_count': int count of zones where R_i < 1.0
            - 'surplus_zones_count': int count of zones where R_i >= 1.0
            - 'total_parking_deficit': float sum of (demand - supply) for deficit zones
            - 'mismatch_gini': float Gini coefficient of mismatch ratios [0, 1]
    """
    from scipy.spatial.distance import cdist

    dem_coords = np.asarray(demand_coords, dtype=np.float64)
    fac_coords = np.asarray(parking_facility_coords, dtype=np.float64)
    caps = np.asarray(parking_capacities, dtype=np.float64)
    demand = np.asarray(zone_parking_demand, dtype=np.float64)

    if dem_coords.ndim != 2 or dem_coords.shape[1] != 2:
        raise ValueError("demand_coords must be a 2D array of shape (N, 2)")
    if fac_coords.ndim != 2 or fac_coords.shape[1] != 2:
        raise ValueError("parking_facility_coords must be a 2D array of shape (M, 2)")

    n_demands = dem_coords.shape[0]
    m_facs = fac_coords.shape[0]

    if caps.ndim != 1 or caps.shape[0] != m_facs:
        raise ValueError(
            f"parking_capacities length ({caps.shape[0]}) must match number of "
            f"parking facilities ({m_facs})"
        )
    if np.any(caps <= 0):
        raise ValueError("parking_capacities must be positive (> 0)")

    if demand.ndim != 1 or demand.shape[0] != n_demands:
        raise ValueError(
            f"zone_parking_demand length ({demand.shape[0]}) must match number of "
            f"demand zones ({n_demands})"
        )
    if np.any(demand <= 0):
        raise ValueError("zone_parking_demand must be positive (> 0)")

    if walk_threshold_m <= 0:
        raise ValueError("walk_threshold_m must be positive (> 0)")

    if n_demands == 0:
        return {
            "mismatch_ratios": np.array([], dtype=np.float64),
            "reachable_supply": np.array([], dtype=np.float64),
            "deficit_zones_count": 0,
            "surplus_zones_count": 0,
            "total_parking_deficit": 0.0,
            "mismatch_gini": 0.0,
        }

    if m_facs == 0:
        ratios = np.zeros(n_demands, dtype=np.float64)
        reachable = np.zeros(n_demands, dtype=np.float64)
        return {
            "mismatch_ratios": ratios,
            "reachable_supply": reachable,
            "deficit_zones_count": n_demands,
            "surplus_zones_count": 0,
            "total_parking_deficit": float(np.sum(demand)),
            "mismatch_gini": 0.0,
        }

    dists = cdist(dem_coords, fac_coords, metric="euclidean")

    W = np.zeros_like(dists)
    mask = dists <= walk_threshold_m
    W[mask] = 1.0 - (dists[mask] / walk_threshold_m) ** 2

    S_i = np.sum(W * caps[None, :], axis=1)

    safe_demand = np.maximum(demand, 1e-6)
    R_i = S_i / safe_demand

    deficit_mask = R_i < 1.0
    surplus_mask = R_i >= 1.0

    deficit_zones_count = int(np.sum(deficit_mask))
    surplus_zones_count = int(np.sum(surplus_mask))

    deficits = np.maximum(0.0, demand - S_i)
    total_parking_deficit = float(np.sum(deficits))

    mismatch_gini = spatial_equity_gini(R_i, demand)

    return {
        "mismatch_ratios": R_i,
        "reachable_supply": S_i,
        "deficit_zones_count": deficit_zones_count,
        "surplus_zones_count": surplus_zones_count,
        "total_parking_deficit": total_parking_deficit,
        "mismatch_gini": mismatch_gini,
    }


def ev_charging_accessibility_index(
    zone_demand: np.ndarray,
    zone_coords: np.ndarray,
    station_coords: np.ndarray,
    station_chargers_kw: np.ndarray,
    station_types: Optional[np.ndarray] = None,
    transformer_capacity_kw: Optional[np.ndarray] = None,
    decay_beta: float = 0.1,
) -> dict[str, Any]:
    """Calculates EV Charging Station Spatial Accessibility and Grid Stress Index.

    Combines 2SFCA accessibility decay with station charging capacities (kW),
    optional charger power types (L2 AC vs L3 DC Fast), and local transformer grid capacity limits.

    Args:
        zone_demand: 1D array of EV charging demand per zone of shape (N,).
        zone_coords: 2D array of zone coordinates (N, 2).
        station_coords: 2D array of station coordinates (M, 2).
        station_chargers_kw: 1D array of station total power capacity (kW) of shape (M,).
        station_types: Optional 1D array of 0 (L2 AC) or 1 (L3 DC Fast) of shape (M,).
        transformer_capacity_kw: Optional 1D array of station grid transformer limits (kW) of shape (M,).
        decay_beta: Distance decay exponential parameter (default 0.1).

    Returns:
        Dict containing:
          - 'accessibility_score': 1D array (N,) of zone EV accessibility scores.
          - 'grid_stress_ratio': 1D array (M,) of station grid stress demand/capacity ratios.
          - 'station_capacity_ratios': 1D array (M,) of station capacity-to-demand ratios.
          - 'spatial_gini': Float Gini inequality coefficient across zone scores.
          - 'equity_index': Float (1.0 - Gini) spatial equity index.
          - 'l2_accessibility': 1D array (N,) (if station_types provided, else None).
          - 'dc_fast_accessibility': 1D array (N,) (if station_types provided, else None).
    """
    z_dem = np.asarray(zone_demand, dtype=np.float64)
    z_xy = np.asarray(zone_coords, dtype=np.float64)
    s_xy = np.asarray(station_coords, dtype=np.float64)
    s_kw = np.asarray(station_chargers_kw, dtype=np.float64)

    n_zones = len(z_dem)
    m_stations = len(s_kw)

    if z_xy.shape != (n_zones, 2):
        raise ValueError("zone_coords shape must be (N, 2).")
    if s_xy.ndim != 2 or s_xy.shape[1] != 2 or s_xy.shape[0] != m_stations:
        raise ValueError("station_coords shape must be (M, 2) matching station_chargers_kw.")
    if np.any(z_dem < 0):
        raise ValueError("zone_demand values must be non-negative.")
    if np.any(s_kw < 0):
        raise ValueError("station_chargers_kw values must be non-negative.")
    if decay_beta <= 0:
        raise ValueError("decay_beta must be positive.")

    eff_capacity = np.copy(s_kw)
    if transformer_capacity_kw is not None:
        t_kw = np.asarray(transformer_capacity_kw, dtype=np.float64)
        if len(t_kw) != m_stations:
            raise ValueError("transformer_capacity_kw length must equal number of stations M.")
        eff_capacity = np.minimum(eff_capacity, t_kw)

    from scipy.spatial.distance import cdist
    dists = cdist(z_xy, s_xy, metric="euclidean")
    f_decay = np.exp(-decay_beta * dists)

    d_stations = np.sum(z_dem[:, None] * f_decay, axis=0)

    safe_demand = np.maximum(d_stations, 1e-6)
    r_j = eff_capacity / safe_demand

    grid_stress = d_stations / np.maximum(eff_capacity, 1e-6)

    a_i = np.sum(r_j[None, :] * f_decay, axis=1)

    l2_access = None
    dc_access = None
    if station_types is not None:
        s_types = np.asarray(station_types, dtype=int)
        if len(s_types) != m_stations:
            raise ValueError("station_types length must equal number of stations M.")

        l2_mask = s_types == 0
        dc_mask = s_types == 1

        r_l2 = np.where(l2_mask, r_j, 0.0)
        r_dc = np.where(dc_mask, r_j, 0.0)

        l2_access = np.sum(r_l2[None, :] * f_decay, axis=1)
        dc_access = np.sum(r_dc[None, :] * f_decay, axis=1)

    gini = spatial_equity_gini(a_i, z_dem)
    equity = float(1.0 - gini)

    return {
        "accessibility_score": a_i,
        "grid_stress_ratio": grid_stress,
        "station_capacity_ratios": r_j,
        "spatial_gini": gini,
        "equity_index": equity,
        "l2_accessibility": l2_access,
        "dc_fast_accessibility": dc_access,
    }


def multimodal_transit_isochrone_profiler(
    origin_coord: np.ndarray,
    destination_coords: np.ndarray,
    transit_stop_coords: np.ndarray,
    transit_headways_min: np.ndarray,
    transit_travel_times: np.ndarray,
    walk_speed_kmh: float = 4.8,
    transfer_penalty_min: float = 5.0,
    max_time_budget_min: float = 45.0,
) -> dict[str, Any]:
    """Generates Multi-Modal Transit Travel Time Isochrones and Reachability Metrics.

    Evaluates shortest multi-modal travel times from an origin location to destinations,
    incorporating walking access/egress speeds, transit headway initial waiting times,
    in-vehicle travel times between stops, and transfer penalty friction.

    Args:
        origin_coord: 1D array of origin coordinate [x, y].
        destination_coords: 2D array of shape (N, 2) for target destinations/centroids.
        transit_stop_coords: 2D array of shape (S, 2) for public transit stop locations.
        transit_headways_min: 1D array of shape (S,) for transit service headways in minutes.
        transit_travel_times: 2D array of shape (S, S) for transit in-vehicle travel times in minutes.
        walk_speed_kmh: Walking speed in km/h (default 4.8 km/h).
        transfer_penalty_min: Fixed penalty per transfer in minutes (default 5.0 min).
        max_time_budget_min: Maximum time threshold budget in minutes (default 45.0 min).

    Returns:
        Dict containing:
          - 'travel_times_min': 1D float array (N,) of total multi-modal travel times in minutes.
          - 'reachable_mask': 1D bool array (N,) indicating reachability within max_time_budget_min.
          - 'isochrone_bands': 1D int array (N,) of band codes (1: <=15m, 2: 15-30m, 3: 30-45m, 4: 45-60m, 0: >60m/unreachable).
          - 'mode_used': List of str ("direct_walk" or "multimodal_transit") per destination.
          - 'reachable_count': Int count of reachable destinations.
          - 'coverage_ratio': Float fraction of destinations reachable within time budget.
    """
    orig_raw = np.asarray(origin_coord, dtype=np.float64)
    if orig_raw.ndim != 1 or orig_raw.shape != (2,):
        raise ValueError("origin_coord must be a 1D array of shape (2,).")
    orig_xy = orig_raw
    dest_xy = np.asarray(destination_coords, dtype=np.float64)
    stops_xy = np.asarray(transit_stop_coords, dtype=np.float64)
    headways = np.asarray(transit_headways_min, dtype=np.float64)
    t_matrix = np.asarray(transit_travel_times, dtype=np.float64)
    n_dests = len(dest_xy)
    if dest_xy.ndim != 2 or dest_xy.shape[1] != 2:
        raise ValueError("destination_coords must be a 2D array of shape (N, 2).")
    s_stops = len(stops_xy)
    if stops_xy.ndim != 2 or stops_xy.shape[1] != 2:
        raise ValueError("transit_stop_coords must be a 2D array of shape (S, 2).")
    if len(headways) != s_stops:
        raise ValueError("transit_headways_min length must match S stops.")
    if t_matrix.shape != (s_stops, s_stops):
        raise ValueError("transit_travel_times shape must be (S, S).")
    if walk_speed_kmh <= 0:
        raise ValueError("walk_speed_kmh must be positive.")
    if transfer_penalty_min < 0:
        raise ValueError("transfer_penalty_min must be non-negative.")
    if max_time_budget_min <= 0:
        raise ValueError("max_time_budget_min must be positive.")

    from scipy.spatial.distance import cdist
    walk_speed_mpm = (walk_speed_kmh * 1000.0) / 60.0

    d_direct = np.sqrt(np.sum((dest_xy - orig_xy) ** 2, axis=1))
    t_direct_walk = d_direct / walk_speed_mpm

    d_access = np.sqrt(np.sum((stops_xy - orig_xy) ** 2, axis=1))
    t_access_walk = d_access / walk_speed_mpm
    t_initial_wait = 0.5 * headways
    t_stop_arrival = t_access_walk + t_initial_wait

    t_stops_net = np.copy(t_matrix)
    np.fill_diagonal(t_stops_net, 0.0)

    for k in range(s_stops):
        for i in range(s_stops):
            for j in range(s_stops):
                if t_stops_net[i, k] > 0 and t_stops_net[k, j] > 0:
                    cand = t_stops_net[i, k] + t_stops_net[k, j] + transfer_penalty_min
                    if t_stops_net[i, j] == 0 or cand < t_stops_net[i, j]:
                        t_stops_net[i, j] = cand

    t_reach_stop = np.zeros(s_stops, dtype=np.float64)
    for s in range(s_stops):
        times_to_s = t_stop_arrival + t_stops_net[:, s]
        t_reach_stop[s] = np.min(times_to_s)

    d_egress = cdist(stops_xy, dest_xy, metric="euclidean")
    t_egress_walk = d_egress / walk_speed_mpm

    t_dest_transit = np.min(t_reach_stop[:, None] + t_egress_walk, axis=0)

    total_travel_times = np.minimum(t_direct_walk, t_dest_transit)

    modes_used = []
    for i in range(n_dests):
        if t_direct_walk[i] <= t_dest_transit[i]:
            modes_used.append("direct_walk")
        else:
            modes_used.append("multimodal_transit")

    reachable = total_travel_times <= max_time_budget_min
    reachable_cnt = int(np.sum(reachable))
    coverage = float(reachable_cnt / n_dests) if n_dests > 0 else 0.0

    isochrone_bands = np.zeros(n_dests, dtype=int)
    for i in range(n_dests):
        t = total_travel_times[i]
        if t <= 15.0:
            isochrone_bands[i] = 1
        elif t <= 30.0:
            isochrone_bands[i] = 2
        elif t <= 45.0:
            isochrone_bands[i] = 3
        elif t <= 60.0:
            isochrone_bands[i] = 4
        else:
            isochrone_bands[i] = 0

    return {
        "travel_times_min": total_travel_times,
        "reachable_mask": reachable,
        "isochrone_bands": isochrone_bands,
        "mode_used": modes_used,
        "reachable_count": reachable_cnt,
        "coverage_ratio": coverage,
    }


def ev_cvrp_multi_depot_routing(
    depot_coords: np.ndarray,
    customer_coords: np.ndarray,
    customer_demands: np.ndarray,
    charger_coords: Optional[np.ndarray] = None,
    vehicle_capacity: float = 100.0,
    battery_capacity_kwh: float = 60.0,
    energy_consumption_kwh_km: float = 0.25,
) -> dict[str, Any]:
    """Multi-Depot Electric Vehicle Capacitated Vehicle Routing Problem (EV-CVRP).

    Constructs vehicle routes from multiple depots serving customer demands, incorporating
    vehicle payload capacities, battery SOC constraints, and en-route charging stations.

    Args:
        depot_coords: 2D array of shape (D, 2) for depot locations.
        customer_coords: 2D array of shape (C, 2) for customer delivery locations.
        customer_demands: 1D array of shape (C,) for customer delivery payloads (> 0).
        charger_coords: Optional 2D array of shape (R, 2) for fast charger locations.
        vehicle_capacity: Maximum payload capacity per vehicle (> 0, default 100.0).
        battery_capacity_kwh: Vehicle battery energy capacity in kWh (default 60.0).
        energy_consumption_kwh_km: Fleet energy depletion rate in kWh/km (default 0.25).

    Returns:
        Dict containing:
          - 'routes': List of dicts per vehicle route with keys:
                      'depot_index', 'stops', 'load_used', 'total_distance_km', 'energy_consumed_kwh', 'recharge_events_count'.
          - 'total_distance_km': Float total travel distance across all routes.
          - 'total_energy_kwh': Float total energy consumed in kWh across fleet.
          - 'vehicles_used_count': Int number of active vehicle routes deployed.
          - 'unserviced_customers_count': Int count of unserviced customers.
    """
    d_coords = np.asarray(depot_coords, dtype=np.float64)
    c_coords = np.asarray(customer_coords, dtype=np.float64)
    c_dem = np.asarray(customer_demands, dtype=np.float64)

    n_cust = len(c_coords)

    if d_coords.ndim != 2 or d_coords.shape[1] != 2:
        raise ValueError("depot_coords must be a 2D array of shape (D, 2).")
    if c_coords.ndim != 2 or c_coords.shape[1] != 2:
        raise ValueError("customer_coords must be a 2D array of shape (C, 2).")
    if len(c_dem) != n_cust:
        raise ValueError("customer_demands length must match C customers.")
    if np.any(c_dem < 0):
        raise ValueError("customer_demands values must be non-negative.")
    if vehicle_capacity <= 0:
        raise ValueError("vehicle_capacity must be positive.")
    if battery_capacity_kwh <= 0:
        raise ValueError("battery_capacity_kwh must be positive.")
    if energy_consumption_kwh_km <= 0:
        raise ValueError("energy_consumption_kwh_km must be positive.")

    from scipy.spatial.distance import cdist

    r_coords = np.empty((0, 2), dtype=np.float64)
    if charger_coords is not None:
        r_coords = np.asarray(charger_coords, dtype=np.float64)
        if r_coords.ndim != 2 or r_coords.shape[1] != 2:
            raise ValueError("charger_coords must be a 2D array of shape (R, 2).")

    unvisited = set(range(n_cust))
    routes: list[dict[str, Any]] = []

    while unvisited:
        cust_list = list(unvisited)
        sub_c_coords = c_coords[cust_list]
        d_dep_cust = cdist(d_coords, sub_c_coords, metric="euclidean")

        best_d_idx, best_sub_c = np.unravel_index(np.argmin(d_dep_cust), d_dep_cust.shape)
        start_cust = cust_list[best_sub_c]

        current_depot = best_d_idx
        current_load = 0.0
        current_soc_kwh = battery_capacity_kwh
        current_pos = d_coords[current_depot]

        route_stops = []
        route_dist = 0.0
        recharge_cnt = 0

        curr_cust = start_cust
        while curr_cust is not None and curr_cust in unvisited:
            dem_val = c_dem[curr_cust]
            if current_load + dem_val > vehicle_capacity:
                break

            step_dist = float(np.linalg.norm(c_coords[curr_cust] - current_pos))
            energy_req = step_dist * energy_consumption_kwh_km

            dist_to_depot = float(np.linalg.norm(c_coords[curr_cust] - d_coords[current_depot]))
            energy_to_depot = dist_to_depot * energy_consumption_kwh_km

            if current_soc_kwh < energy_req + energy_to_depot:
                if len(r_coords) > 0:
                    d_pos_chg = cdist(current_pos[None, :], r_coords, metric="euclidean")[0]
                    chg_idx = int(np.argmin(d_pos_chg))
                    chg_dist = float(d_pos_chg[chg_idx])
                    if current_soc_kwh >= chg_dist * energy_consumption_kwh_km:
                        route_dist += chg_dist
                        current_pos = r_coords[chg_idx]
                        current_soc_kwh = battery_capacity_kwh
                        recharge_cnt += 1
                        continue

            if current_soc_kwh < energy_req:
                break

            unvisited.remove(curr_cust)
            route_stops.append(int(curr_cust))
            current_load += dem_val
            current_soc_kwh -= energy_req
            route_dist += step_dist
            current_pos = c_coords[curr_cust]

            if not unvisited:
                break

            rem_cust = list(unvisited)
            d_curr_rem = cdist(current_pos[None, :], c_coords[rem_cust], metric="euclidean")[0]
            next_sub = int(np.argmin(d_curr_rem))
            next_cand = rem_cust[next_sub]

            if current_load + c_dem[next_cand] <= vehicle_capacity:
                curr_cust = next_cand
            else:
                break

        ret_dist = float(np.linalg.norm(current_pos - d_coords[current_depot]))
        route_dist += ret_dist
        tot_energy = route_dist * energy_consumption_kwh_km

        routes.append({
            "depot_index": int(current_depot),
            "stops": route_stops,
            "load_used": float(current_load),
            "total_distance_km": float(route_dist),
            "energy_consumed_kwh": float(tot_energy),
            "recharge_events_count": recharge_cnt,
        })

        if not route_stops:
            unvisited.pop()

    tot_dist = float(np.sum([r["total_distance_km"] for r in routes]))
    tot_energy = float(np.sum([r["energy_consumed_kwh"] for r in routes]))

    return {
        "routes": routes,
        "total_distance_km": tot_dist,
        "total_energy_kwh": tot_energy,
        "vehicles_used_count": len(routes),
        "unserviced_customers_count": len(unvisited),
    }


def micromobility_equity_index(
    vehicle_counts: np.ndarray,
    transit_distances: np.ndarray,
    vulnerability_weights: np.ndarray,
) -> dict[str, Any]:
    """First-Mile / Last-Mile Micro-Mobility Equity Index.

    Evaluates dockless micro-mobility device availability relative to public transit hubs
    weighted by socio-economic vulnerability factors and calculates equity distribution metrics.

    Args:
        vehicle_counts: Array of available bikes/scooters per zone.
        transit_distances: Distance (meters) to nearest transit hub per zone.
        vulnerability_weights: Socio-economic vulnerability weight per zone (0-1).

    Returns:
        Dict containing zone equity scores, overall micro-mobility equity index, and Gini coefficient.
    """
    dist_decay = np.exp(-transit_distances / 500.0)
    supply_score = vehicle_counts * dist_decay

    equity_scores = supply_score / (vulnerability_weights + 1e-6)

    sorted_s = np.sort(equity_scores)
    n = len(sorted_s)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * sorted_s) - (n + 1) * np.sum(sorted_s)) / (n * np.sum(sorted_s) + 1e-12))

    return {
        "zone_equity_scores": equity_scores,
        "mean_equity_score": float(np.mean(equity_scores)),
        "equity_gini_index": max(0.0, float(gini)),
        "low_access_zones_count": int(np.sum(equity_scores < np.median(equity_scores) * 0.5)),
    }


def transit_fleet_electrification_scheduler(
    bus_arrival_times_hr: np.ndarray,
    energy_needed_kwh: np.ndarray,
    charger_power_kw: float = 150.0,
    max_grid_power_kw: float = 1000.0,
) -> dict[str, Any]:
    """Public Transit Fleet Electrification & Charging Scheduler.

    Optimizes electric bus depot charging schedules subject to grid peak demand caps.

    Args:
        bus_arrival_times_hr: Array of bus depot arrival timestamps in hours.
        energy_needed_kwh: Energy required to full charge per bus (kWh).
        charger_power_kw: Output power per fast charger dispenser (kW).
        max_grid_power_kw: Substation transformer maximum power cap (kW).

    Returns:
        Dict containing peak power demand (kW), total charging hours, and schedule feasibility.
    """
    if charger_power_kw <= 0:
        raise ValueError("charger_power_kw must be positive.")

    n_buses = len(bus_arrival_times_hr)
    charge_times_hr = energy_needed_kwh / charger_power_kw

    concurrent_buses = int(np.floor(max_grid_power_kw / charger_power_kw))
    peak_demand = min(n_buses * charger_power_kw, max_grid_power_kw)
    total_energy = float(np.sum(energy_needed_kwh))

    return {
        "peak_power_demand_kw": float(peak_demand),
        "total_energy_delivered_kwh": total_energy,
        "max_simultaneous_buses": concurrent_buses,
        "total_fleet_charge_time_hr": float(np.sum(charge_times_hr) / max(concurrent_buses, 1)),
        "grid_cap_compliant": bool(peak_demand <= max_grid_power_kw),
    }





