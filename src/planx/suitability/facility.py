# -*- coding: utf-8 -*-
"""Facility location and optimal placement algorithms for suitability analysis."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, cast

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


def greedy_p_median(
    candidate_coords: Optional[np.ndarray] = None,
    demand_coords: Optional[np.ndarray] = None,
    demand_pop: Optional[np.ndarray] = None,
    p: int = 1,
    dists: Optional[np.ndarray] = None,
    existing_indices: Optional[List[int]] = None,
    existing_coords: Optional[np.ndarray] = None,
) -> Tuple[List[int], np.ndarray]:
    """Solves the p-Median problem using a greedy heuristic.

    Finds a subset of p facilities from candidates to minimize the sum of weighted
    distances from demand points to their nearest selected facility.
    Supports either coordinates or a precomputed distance matrix.

    Args:
        candidate_coords: Optional array of shape (C, 2) containing candidate coordinates.
        demand_coords: Optional array of shape (D, 2) containing demand coordinates.
        demand_pop: Optional array of shape (D,) containing demand population/weights.
            If omitted, equal weights are assumed.
        p: Number of facilities to select.
        dists: Optional precomputed distance matrix of shape (C, D) from C candidates
            to D demands. If provided, candidate_coords and demand_coords are ignored.
        existing_indices: Optional list of indices of already selected candidates.
        existing_coords: Optional array of shape (E, 2) of existing facility coordinates.
            Used only if candidate_coords and demand_coords are provided.

    Returns:
        Tuple of:
          - selected_indices: List of selected candidate indices in order of greedy selection.
          - total_costs: NumPy array of shape (actual_p,) containing the total weighted distance
            cost after selecting each facility.
    """
    if dists is not None:
        d_mat = np.asarray(dists, dtype=np.float64).copy()
        if d_mat.ndim != 2:
            raise ValueError("dists must be a 2D array of shape (C, D)")
        c_count, d_count = d_mat.shape
    else:
        if candidate_coords is None or demand_coords is None:
            raise ValueError("Must provide either dists or both candidate_coords and demand_coords")
        cand = np.asarray(candidate_coords, dtype=np.float64)
        dem = np.asarray(demand_coords, dtype=np.float64)
        if cand.ndim != 2 or cand.shape[1] != 2:
            raise ValueError("candidate_coords must be of shape (C, 2)")
        if dem.ndim != 2 or dem.shape[1] != 2:
            raise ValueError("demand_coords must be of shape (D, 2)")
        c_count = cand.shape[0]
        d_count = dem.shape[0]
        # Compute Euclidean distance matrix
        d_mat = np.sqrt(
            (cand[:, None, 0] - dem[None, :, 0]) ** 2 + (cand[:, None, 1] - dem[None, :, 1]) ** 2
        )

    if demand_pop is None:
        pop = np.ones(d_count, dtype=np.float64)
    else:
        pop = np.asarray(demand_pop, dtype=np.float64)
        if pop.ndim != 1 or pop.shape[0] != d_count:
            raise ValueError(f"demand_pop must be a 1D array of length D ({d_count})")

    if p <= 0:
        raise ValueError("p must be greater than 0")

    # Initialize min_dists with infinity (or existing facilities)
    min_dists = np.full(d_count, np.inf)

    # Set up existing facilities if any
    selected_indices: List[int] = []

    if existing_indices is not None:
        for idx in existing_indices:
            idx = int(idx)
            if idx < 0 or idx >= c_count:
                raise ValueError("existing_indices must be within valid range [0, C)")
            selected_indices.append(idx)
            min_dists = np.minimum(min_dists, d_mat[idx])

    if existing_coords is not None and len(existing_coords) > 0 and dists is None:
        exist = np.asarray(existing_coords, dtype=np.float64)
        if exist.ndim != 2 or exist.shape[1] != 2:
            raise ValueError("existing_coords must be of shape (E, 2)")
        exist_dists = np.sqrt(
            (exist[:, None, 0] - dem[None, :, 0]) ** 2 + (exist[:, None, 1] - dem[None, :, 1]) ** 2
        )
        min_exist = np.min(exist_dists, axis=0)
        min_dists = np.minimum(min_dists, min_exist)

    total_costs: List[float] = []

    for _ in range(p):
        # Calculate costs for all candidates
        # np.minimum is shape (C, D)
        candidates_min = np.minimum(min_dists[None, :], d_mat)
        # Sum along axis 1 (multiply by pop)
        costs = np.sum(candidates_min * pop[None, :], axis=1)

        # Mark already selected candidates with infinity cost so they aren't chosen again
        if selected_indices:
            costs[selected_indices] = np.inf

        best_idx = int(np.argmin(costs))
        best_cost = costs[best_idx]

        if best_cost == np.inf or len(selected_indices) >= c_count:
            break

        selected_indices.append(best_idx)
        min_dists = np.minimum(min_dists, d_mat[best_idx])
        total_costs.append(float(np.sum(min_dists * pop)))

    newly_selected = [
        idx for idx in selected_indices if existing_indices is None or idx not in existing_indices
    ]

    return newly_selected, np.array(total_costs)


def greedy_lscp(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_pop: Optional[np.ndarray] = None,
    max_distance: float = 1000.0,
    target_coverage: float = 1.0,
    existing_coords: Optional[np.ndarray] = None,
) -> Tuple[List[int], float]:
    """Solves the Location Set Covering Problem (LSCP) using a greedy heuristic.

    Minimizes the number of selected facilities such that at least target_coverage
    fraction of the total population is covered within max_distance.

    Args:
        candidate_coords: NumPy array of shape (C, 2) containing candidate coordinates.
        demand_coords: NumPy array of shape (D, 2) containing demand coordinates.
        demand_pop: Optional NumPy array of shape (D,) containing demand population.
            If omitted, all demand points are weighted equally (population = 1).
        max_distance: Maximum distance for coverage.
        target_coverage: Fraction of total population/demand that must be covered [0.0, 1.0].
            Defaults to 1.0 (100% coverage).
        existing_coords: Optional NumPy array of shape (E, 2) of existing facility coordinates.

    Returns:
        Tuple of:
          - selected_indices: List of selected candidate site indices.
          - final_coverage_fraction: The actual fraction of population covered.
    """
    cand = np.asarray(candidate_coords, dtype=np.float64)
    dem = np.asarray(demand_coords, dtype=np.float64)

    if cand.ndim != 2 or cand.shape[1] != 2:
        raise ValueError("candidate_coords must be of shape (C, 2)")
    if dem.ndim != 2 or dem.shape[1] != 2:
        raise ValueError("demand_coords must be of shape (D, 2)")

    c_count = cand.shape[0]
    d_count = dem.shape[0]

    if demand_pop is None:
        pop = np.ones(d_count, dtype=np.float64)
    else:
        pop = np.asarray(demand_pop, dtype=np.float64)
        if pop.ndim != 1 or pop.shape[0] != d_count:
            raise ValueError(f"demand_pop must be a 1D array of length D ({d_count})")

    total_pop = float(np.sum(pop))
    if total_pop <= 0:
        total_pop = 1.0

    target_pop = total_pop * target_coverage

    # Precompute pairwise distances: shape (C, D)
    dists = np.sqrt(
        (cand[:, None, 0] - dem[None, :, 0]) ** 2 + (cand[:, None, 1] - dem[None, :, 1]) ** 2
    )
    coverage_matrix = dists <= max_distance

    covered = np.zeros(d_count, dtype=bool)

    # Pre-cover demand with existing facilities
    if existing_coords is not None and len(existing_coords) > 0:
        exist = np.asarray(existing_coords, dtype=np.float64)
        if exist.ndim != 2 or exist.shape[1] != 2:
            raise ValueError("existing_coords must be of shape (E, 2)")
        exist_dists = np.sqrt(
            (exist[:, None, 0] - dem[None, :, 0]) ** 2 + (exist[:, None, 1] - dem[None, :, 1]) ** 2
        )
        covered = np.any(exist_dists <= max_distance, axis=0)

    current_covered_pop = np.sum(pop[covered])
    selected_indices: List[int] = []

    # If we already meet the target, return immediately
    if current_covered_pop >= target_pop:
        return selected_indices, current_covered_pop / total_pop

    while current_covered_pop < target_pop and len(selected_indices) < c_count:
        gains = np.zeros(c_count)
        for c in range(c_count):
            if c in selected_indices:
                gains[c] = -1.0
                continue
            gains[c] = np.sum(pop[coverage_matrix[c] & ~covered])

        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]

        # If no more population can be covered, stop
        if best_gain <= 0:
            break

        selected_indices.append(best_idx)
        covered |= coverage_matrix[best_idx]
        current_covered_pop = np.sum(pop[covered])

    return selected_indices, current_covered_pop / total_pop


def capacitated_location_allocation(
    facility_coords: np.ndarray,
    facility_capacities: np.ndarray,
    demand_coords: np.ndarray,
    demand_pop: np.ndarray,
    max_distance: Optional[float] = None,
) -> tuple[dict[int, list[int]], np.ndarray, np.ndarray]:
    """Assigns demand points to their nearest available facility respecting capacity limits.

    Uses a greedy heuristic: demand points are sorted by their distance to the nearest
    facility, and each demand point is assigned to its closest facility that has enough
    remaining capacity.

    Args:
        facility_coords: NumPy array of shape (F, 2) containing facility coordinates.
        facility_capacities: NumPy array of shape (F,) containing capacity limits.
        demand_coords: NumPy array of shape (D, 2) containing demand point coordinates.
        demand_pop: NumPy array of shape (D,) containing population/demand at each point.
        max_distance: Optional maximum distance for assignment. Demands further than
            this from a facility cannot be assigned to it.

    Returns:
        Tuple of:
          - allocations: Dictionary mapping facility index (int) to list of assigned
            demand point indices (list of ints).
          - unassigned: 1D NumPy array containing indices of demand points that could
            not be assigned.
          - usage: 1D NumPy array of shape (F,) containing the total allocated population
            at each facility.
    """
    fac = np.asarray(facility_coords, dtype=np.float64)
    fac_caps = np.asarray(facility_capacities, dtype=np.float64).copy()
    dem = np.asarray(demand_coords, dtype=np.float64)
    pop = np.asarray(demand_pop, dtype=np.float64)

    if fac.ndim != 2 or fac.shape[1] != 2:
        raise ValueError("facility_coords must be of shape (F, 2)")
    if fac_caps.ndim != 1 or fac_caps.shape[0] != fac.shape[0]:
        raise ValueError("facility_capacities must be a 1D array of length F")
    if dem.ndim != 2 or dem.shape[1] != 2:
        raise ValueError("demand_coords must be of shape (D, 2)")
    if pop.ndim != 1 or pop.shape[0] != dem.shape[0]:
        raise ValueError("demand_pop must be a 1D array of length D")

    f_count = fac.shape[0]
    d_count = dem.shape[0]

    if f_count == 0 or d_count == 0:
        return {}, np.arange(d_count, dtype=np.int64), np.zeros(f_count)

    # Compute Euclidean distance matrix: shape (F, D)
    dists = np.sqrt(
        (fac[:, None, 0] - dem[None, :, 0]) ** 2 + (fac[:, None, 1] - dem[None, :, 1]) ** 2
    )

    # Find the minimum distance to any facility for each demand point to sort them
    min_dists = np.min(dists, axis=0)
    # Sort demand indices by their minimum distance to any facility (closest first)
    sorted_demand_indices = np.argsort(min_dists)

    allocations: dict[int, list[int]] = {i: [] for i in range(f_count)}
    unassigned: list[int] = []
    usage = np.zeros(f_count, dtype=np.float64)

    cutoff = max_distance if max_distance is not None else np.inf

    for d_idx in sorted_demand_indices:
        d_pop = pop[d_idx]
        # Get distances from this demand to all facilities
        d_dists = dists[:, d_idx]

        # Sort facilities by distance to this demand point
        sorted_fac_indices = np.argsort(d_dists)

        assigned = False
        for f_idx in sorted_fac_indices:
            dist = d_dists[f_idx]
            if dist > cutoff:
                # Since facilities are sorted by distance, all subsequent ones are also too far
                break

            # Check if facility has enough capacity left
            if fac_caps[f_idx] >= d_pop:
                # Allocate
                allocations[f_idx].append(int(d_idx))
                fac_caps[f_idx] -= d_pop
                usage[f_idx] += d_pop
                assigned = True
                break

        if not assigned:
            unassigned.append(int(d_idx))

    return allocations, np.array(unassigned, dtype=np.int64), usage


def mclp_distance_decay(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_weights: np.ndarray,
    max_distance: float,
    k: int,
    decay_method: str = "exponential",
    beta: float = 0.002,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Solves Maximal Covering Location Problem with continuous distance decay.

    Args:
        candidate_coords: (C, 2) NumPy array of candidate facility coordinates.
        demand_coords: (D, 2) NumPy array of demand point coordinates.
        demand_weights: (D,) NumPy array of population/demand weights.
        max_distance: Service cutoff distance threshold float.
        k: Maximum number of facilities to select int.
        decay_method: Distance decay kernel ("exponential", "gaussian", or "linear").
        beta: Decay parameter float.

    Returns:
        A tuple of:
          - selected_indices: List of selected candidate facility indices.
          - added_coverage: NumPy array of effective population added by each facility.
          - cumulative_coverage: NumPy array of cumulative effective population covered.
    """
    candidates = np.asarray(candidate_coords, dtype=np.float64)
    demands = np.asarray(demand_coords, dtype=np.float64)
    weights = np.asarray(demand_weights, dtype=np.float64)

    c_count = len(candidates)
    d_count = len(demands)
    if c_count == 0 or d_count == 0 or k <= 0:
        return [], np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    diffs = demands[:, None, :] - candidates[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))

    if decay_method == "exponential":
        decay_factors = np.exp(-beta * dists)
    elif decay_method == "gaussian":
        decay_factors = np.exp(-0.5 * (dists / max(1e-9, max_distance / 2.0)) ** 2)
    elif decay_method == "linear":
        decay_factors = np.maximum(0.0, 1.0 - (dists / max(1e-9, max_distance)))
    else:
        raise ValueError("Unsupported decay_method. Use exponential, gaussian, or linear.")

    decay_factors = np.where(dists <= max_distance, decay_factors, 0.0)

    selected: list[int] = []
    added_cov: list[float] = []
    cum_cov: list[float] = []
    current_best_coverage = np.zeros(d_count, dtype=np.float64)

    available = set(range(c_count))

    for _ in range(min(k, c_count)):
        best_candidate = -1
        best_added = -1.0

        for c_idx in available:
            candidate_cov = decay_factors[:, c_idx]
            new_cov = np.maximum(current_best_coverage, candidate_cov)
            added = float(np.sum((new_cov - current_best_coverage) * weights))
            if added > best_added:
                best_added = added
                best_candidate = c_idx

        if best_candidate == -1 or best_added <= 0:
            break

        selected.append(best_candidate)
        available.remove(best_candidate)

        current_best_coverage = np.maximum(current_best_coverage, decay_factors[:, best_candidate])
        added_cov.append(best_added)
        cum_cov.append(float(np.sum(current_best_coverage * weights)))

    return selected, np.array(added_cov, dtype=np.float64), np.array(cum_cov, dtype=np.float64)


def pareto_facility_location(
    candidate_coords: np.ndarray,
    demand_coords: np.ndarray,
    demand_weights: np.ndarray,
    k: int,
    num_samples: int = 20,
) -> list[dict]:
    """Computes Pareto-optimal facility location configurations for coverage and travel equity.

    Evaluates trade-offs between Total Covered Demand (max) and Average Travel Distance (min).

    Args:
        candidate_coords: (C, 2) NumPy array of candidate facility coordinates.
        demand_coords: (D, 2) NumPy array of demand coordinates.
        demand_weights: (D,) NumPy array of population/demand weights.
        k: Number of facilities to select int.
        num_samples: Number of random multi-objective sampling configurations.

    Returns:
        List of dicts representing Pareto-optimal facility configurations:
          - selected_indices: List of candidate facility indices.
          - total_coverage: Total demand population covered.
          - avg_distance: Average demand travel distance.
          - gini_inequality: Gini coefficient of travel distances across demand points.
    """
    candidates = np.asarray(candidate_coords, dtype=np.float64)
    demands = np.asarray(demand_coords, dtype=np.float64)
    weights = np.asarray(demand_weights, dtype=np.float64)

    c_count = len(candidates)
    d_count = len(demands)
    if c_count == 0 or d_count == 0 or k <= 0:
        return []

    k_sel = min(k, c_count)

    diffs = demands[:, None, :] - candidates[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))  # (D, C)

    configs = []
    # Include greedy MCLP baseline configuration
    greedy_mclp_sel, _, _ = greedy_mclp(
        candidates, demands, weights, max_distance=float(np.max(dists)), k=k_sel
    )
    if greedy_mclp_sel:
        configs.append(tuple(sorted(greedy_mclp_sel)))

    # Sample random subset configurations
    import itertools

    if c_count <= 10:
        for combo in itertools.combinations(range(c_count), k_sel):
            configs.append(combo)
    else:
        np.random.seed(42)
        for _ in range(num_samples):
            combo = tuple(sorted(np.random.choice(c_count, size=k_sel, replace=False)))
            configs.append(combo)

    unique_configs = list(set(configs))
    evaluations = []

    for cfg in unique_configs:
        cfg_dists = dists[:, list(cfg)]
        min_dists = np.min(cfg_dists, axis=1)

        tot_cov = float(np.sum(weights[min_dists < np.inf]))
        avg_d = float(np.sum(min_dists * weights) / max(1e-9, np.sum(weights)))

        # Gini calculation
        sorted_d = np.sort(min_dists)
        cum_w = np.cumsum(weights[np.argsort(min_dists)])
        cum_w_norm = cum_w / cum_w[-1]
        sum_sd = max(1e-9, float(np.sum(sorted_d)))
        gini = float(1.0 - 2.0 * float(np.sum(sorted_d * (1.0 - cum_w_norm))) / sum_sd)

        evaluations.append(
            {
                "selected_indices": list(cfg),
                "total_coverage": tot_cov,
                "avg_distance": avg_d,
                "gini_inequality": max(0.0, min(1.0, gini)),
            }
        )

    # Filter non-dominated Pareto front (Maximize coverage, Minimize avg_distance)
    pareto_front = []
    for i, eval_i in enumerate(evaluations):
        cov_i = cast(float, eval_i["total_coverage"])
        dist_i = cast(float, eval_i["avg_distance"])
        dominated = False
        for k_idx, eval_k in enumerate(evaluations):
            if i != k_idx:
                cov_k = cast(float, eval_k["total_coverage"])
                dist_k = cast(float, eval_k["avg_distance"])
                if cov_k >= cov_i and dist_k <= dist_i and (cov_k > cov_i or dist_k < dist_i):
                    dominated = True
                    break
        if not dominated:
            pareto_front.append(eval_i)

    return pareto_front


def evaluate_tod_node_suitability(
    station_transit_frequency: np.ndarray,
    surrounding_population_density: np.ndarray,
    land_use_mix_entropy: np.ndarray,
    walkability_pedestrian_score: np.ndarray,
    parking_supply_ratio: np.ndarray,
) -> dict[str, Any]:
    """Evaluates Transit-Oriented Development (TOD) suitability across transit station nodes.

    Uses 3D/5D TOD design principles (Density, Diversity, Design, Destination accessibility,
    Distance) to calculate a multi-criteria TOD suitability score.

    Args:
        station_transit_frequency: (S,) transit trips per hour at station (> 0).
        surrounding_population_density: (S,) residents/jobs per ha within 800m (> 0).
        land_use_mix_entropy: (S,) Shannon land-use mix entropy index [0, 1].
        walkability_pedestrian_score: (S,) pedestrian infrastructure quality score [0, 100].
        parking_supply_ratio: (S,) park-and-ride / parking spaces per transit user.

    Returns:
        Dict with keys:
          - 'tod_scores': (S,) float array in [0, 100]
          - 'tier_1_count': int
          - 'tier_2_count': int
          - 'tier_3_count': int
          - 'tod_ranking': (S,) int array 1-based ranks
    """
    freq = np.asarray(station_transit_frequency, dtype=np.float64)
    density = np.asarray(surrounding_population_density, dtype=np.float64)
    entropy = np.asarray(land_use_mix_entropy, dtype=np.float64)
    walkability = np.asarray(walkability_pedestrian_score, dtype=np.float64)
    parking = np.asarray(parking_supply_ratio, dtype=np.float64)

    if freq.ndim != 1:
        raise ValueError("station_transit_frequency must be a 1D array")

    if np.any(freq <= 0):
        raise ValueError("station_transit_frequency must be > 0.")
    if np.any(density <= 0):
        raise ValueError("surrounding_population_density must be > 0.")
    if np.any((entropy < 0) | (entropy > 1)):
        raise ValueError("land_use_mix_entropy must be in [0, 1].")
    if np.any((walkability < 0) | (walkability > 100)):
        raise ValueError("walkability_pedestrian_score must be in [0, 100].")

    s_freq = freq / np.max(freq) if np.max(freq) > 0 else np.zeros_like(freq)
    s_density = density / np.max(density) if np.max(density) > 0 else np.zeros_like(density)

    p_parking = np.exp(-0.5 * np.maximum(0, parking - 0.2))

    tod_scores = (
        s_freq * 0.25
        + s_density * 0.25
        + entropy * 0.20
        + (walkability / 100.0) * 0.20
        + p_parking * 0.10
    ) * 100.0

    tier_1_count = int(np.sum(tod_scores >= 75))
    tier_2_count = int(np.sum((tod_scores >= 50) & (tod_scores < 75)))
    tier_3_count = int(np.sum(tod_scores < 50))

    # Standard competition ranking or dense ranking? For now simple argsort
    order = np.argsort(-tod_scores)
    tod_ranking = np.empty_like(order)
    tod_ranking[order] = np.arange(1, len(tod_scores) + 1)

    return {
        "tod_scores": tod_scores,
        "tier_1_count": tier_1_count,
        "tier_2_count": tier_2_count,
        "tier_3_count": tier_3_count,
        "tod_ranking": tod_ranking.astype(int),
    }


def ev_fleet_charging_location_allocation(
    fleet_origins: np.ndarray,
    fleet_destinations: np.ndarray,
    candidate_depots: np.ndarray,
    num_depots_to_select: int,
    depot_power_capacities_kw: Optional[np.ndarray] = None,
    fleet_soc_depletion_rate: float = 0.2,
    max_detour_km: float = 15.0,
) -> dict[str, Any]:
    """EV Fleet Charging Station Multi-Objective Location-Allocation Engine.

    Selects optimal charging depots from candidates to serve fleet trips, minimizing detour distances
    and honoring depot power capacity constraints.

    Args:
        fleet_origins: 2D array (N, 2) of trip origin coordinates.
        fleet_destinations: 2D array (N, 2) of trip destination coordinates.
        candidate_depots: 2D array (M, 2) of candidate depot locations.
        num_depots_to_select: Number of depots p to choose (1 <= p <= M).
        depot_power_capacities_kw: Optional 1D array (M,) of max kW capacity per depot.
        fleet_soc_depletion_rate: Energy consumption in kWh per km (default 0.2).
        max_detour_km: Maximum allowable detour threshold in km (default 15.0).

    Returns:
        Dict containing:
          - 'selected_depot_indices': 1D int array of selected candidate indices.
          - 'trip_allocations': 1D int array (N,) mapping each trip to selected depot index (-1 if unassigned).
          - 'fleet_coverage_ratio': Float fraction of trips successfully allocated.
          - 'mean_detour_km': Float mean detour distance for allocated trips.
          - 'depot_power_utilization_kw': 1D float array of allocated charging power per selected depot.
          - 'total_detour_km': Float total extra detour distance across fleet.
    """
    f_orig = np.asarray(fleet_origins, dtype=np.float64)
    f_dest = np.asarray(fleet_destinations, dtype=np.float64)
    c_dep = np.asarray(candidate_depots, dtype=np.float64)

    n_trips = len(f_orig)
    m_candidates = len(c_dep)

    if f_orig.ndim != 2 or f_orig.shape[1] != 2:
        raise ValueError("fleet_origins must be a 2D array of shape (N, 2).")
    if f_dest.shape != f_orig.shape:
        raise ValueError("fleet_destinations shape must match fleet_origins shape.")
    if c_dep.ndim != 2 or c_dep.shape[1] != 2:
        raise ValueError("candidate_depots must be a 2D array of shape (M, 2).")
    if not (1 <= num_depots_to_select <= m_candidates):
        raise ValueError("num_depots_to_select must be between 1 and M.")
    if fleet_soc_depletion_rate <= 0:
        raise ValueError("fleet_soc_depletion_rate must be positive.")
    if max_detour_km <= 0:
        raise ValueError("max_detour_km must be positive.")

    if depot_power_capacities_kw is not None:
        p_cap = np.asarray(depot_power_capacities_kw, dtype=np.float64)
        if len(p_cap) != m_candidates:
            raise ValueError("depot_power_capacities_kw length must equal M.")
    else:
        p_cap = np.full(m_candidates, 1e9, dtype=np.float64)

    from scipy.spatial.distance import cdist
    d_orig_dep = cdist(f_orig, c_dep, metric="euclidean")
    d_dep_dest = cdist(c_dep, f_dest, metric="euclidean").T
    d_base = np.sqrt(np.sum((f_orig - f_dest) ** 2, axis=1))

    detours = (d_orig_dep + d_dep_dest) - d_base[:, None]

    selected_depots: list[int] = []
    candidates_remaining = list(range(m_candidates))

    for _ in range(num_depots_to_select):
        best_cand = -1
        best_score = -1.0

        for cand in candidates_remaining:
            temp_sel = selected_depots + [cand]
            sub_detours = detours[:, temp_sel]
            min_det = np.min(sub_detours, axis=1)
            coverage = np.sum(min_det <= max_detour_km)
            total_det = np.sum(np.where(min_det <= max_detour_km, min_det, 0.0))
            score = coverage * 1e5 - total_det

            if score > best_score:
                best_score = score
                best_cand = cand

        if best_cand != -1:
            selected_depots.append(best_cand)
            candidates_remaining.remove(best_cand)

    sel_arr = np.array(selected_depots, dtype=int)

    allocations = np.full(n_trips, -1, dtype=int)
    depot_allocated_kw = np.zeros(len(sel_arr), dtype=np.float64)
    charge_kw_per_trip = 50.0

    for i in range(n_trips):
        sub_d = detours[i, sel_arr]
        sorted_dep_idx = np.argsort(sub_d)

        for dep_i in sorted_dep_idx:
            cand_idx = sel_arr[dep_i]
            det = sub_d[dep_i]

            if det <= max_detour_km:
                if depot_allocated_kw[dep_i] + charge_kw_per_trip <= p_cap[cand_idx]:
                    allocations[i] = cand_idx
                    depot_allocated_kw[dep_i] += charge_kw_per_trip
                    break

    serviced_mask = allocations != -1
    covered_count = int(np.sum(serviced_mask))
    coverage_ratio = float(covered_count / n_trips) if n_trips > 0 else 0.0

    assigned_detours = []
    for i in range(n_trips):
        if allocations[i] != -1:
            c_idx = allocations[i]
            assigned_detours.append(float(detours[i, c_idx]))

    mean_det = float(np.mean(assigned_detours)) if assigned_detours else 0.0
    tot_det = float(np.sum(assigned_detours)) if assigned_detours else 0.0

    return {
        "selected_depot_indices": sel_arr,
        "trip_allocations": allocations,
        "fleet_coverage_ratio": coverage_ratio,
        "mean_detour_km": mean_det,
        "depot_power_utilization_kw": depot_allocated_kw,
        "total_detour_km": tot_det,
    }


def tod_spatial_diversity_index(
    landuse_ratios_matrix: np.ndarray,
    far_intensities: np.ndarray,
    transit_distances: np.ndarray,
) -> dict[str, Any]:
    """Transit-Oriented Development (TOD) Spatial Diversity Profiler.

    Calculates Shannon land-use mix entropy, floor-area ratio (FAR) intensity, and transit
    catchment score for urban nodes.

    Args:
        landuse_ratios_matrix: Matrix of shape (N, K) with land use proportion per zone.
        far_intensities: Array of shape (N,) containing average Floor-Area Ratio.
        transit_distances: Array of shape (N,) containing distance (meters) to transit station.

    Returns:
        Dict containing Shannon entropy scores, TOD diversity index scores, and high-readiness count.
    """
    n, k = landuse_ratios_matrix.shape
    p = np.clip(landuse_ratios_matrix, 1e-12, 1.0)
    p_norm = p / np.sum(p, axis=1, keepdims=True)

    entropy = -np.sum(p_norm * np.log(p_norm), axis=1) / np.log(max(k, 2))

    transit_score = np.exp(-transit_distances / 400.0)

    tod_score = entropy * 0.4 + (far_intensities / (np.max(far_intensities) + 1e-6)) * 0.3 + transit_score * 0.3

    return {
        "shannon_entropy_scores": entropy,
        "tod_diversity_scores": tod_score,
        "mean_tod_score": float(np.mean(tod_score)),
        "high_readiness_nodes_count": int(np.sum(tod_score > 0.7)),
    }


def logistics_microhub_location_allocation(
    demand_coords: np.ndarray,
    demand_volumes: np.ndarray,
    candidate_hub_coords: np.ndarray,
    num_hubs_to_select: int = 3,
    max_cargo_bike_range_km: float = 5.0,
) -> dict[str, Any]:
    """Urban Logistics Last-Mile Micro-Hub Location-Allocation Engine.

    Selects optimal cargo bike distribution micro-hubs minimizing delivery distances under LEZ constraints.

    Args:
        demand_coords: Array of shape (N, 2) for delivery destination points.
        demand_volumes: Array of shape (N,) for parcel volume per point.
        candidate_hub_coords: Array of shape (M, 2) for candidate micro-hub locations.
        num_hubs_to_select: Number of micro-hubs to select (<= M).
        max_cargo_bike_range_km: Maximum cargo bike delivery radius (km).

    Returns:
        Dict containing selected hub indices, demand point allocations, and fleet delivery distance.
    """
    from scipy.spatial.distance import cdist

    m_cands = len(candidate_hub_coords)
    if num_hubs_to_select < 1 or num_hubs_to_select > m_cands:
        raise ValueError("num_hubs_to_select must be between 1 and M.")

    dists = cdist(demand_coords, candidate_hub_coords) / 1000.0

    selected = []
    unselected = list(range(m_cands))

    for _ in range(num_hubs_to_select):
        best_cand = -1
        best_cost = float("inf")
        for c in unselected:
            trial = selected + [c]
            min_d = np.min(dists[:, trial], axis=1)
            cost = float(np.sum(min_d * demand_volumes))
            if cost < best_cost:
                best_cost = cost
                best_cand = c
        selected.append(best_cand)
        unselected.remove(best_cand)

    sub_d = dists[:, selected]
    allocations = np.argmin(sub_d, axis=1)
    min_dists = np.min(sub_d, axis=1)

    in_range_ratio = float(np.mean(min_dists <= max_cargo_bike_range_km))
    tot_dist = float(np.sum(min_dists * demand_volumes))

    return {
        "selected_hub_indices": np.array(selected, dtype=int),
        "demand_allocations": allocations,
        "total_delivery_vkt": tot_dist,
        "cargo_bike_range_coverage_ratio": in_range_ratio,
    }



