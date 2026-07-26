# -*- coding: utf-8 -*-
"""Facility location and optimal placement algorithms for suitability analysis."""

from __future__ import annotations

from typing import List, Optional, Tuple, cast

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

        current_best_coverage = np.maximum(
            current_best_coverage, decay_factors[:, best_candidate]
        )
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

        evaluations.append({
            "selected_indices": list(cfg),
            "total_coverage": tot_cov,
            "avg_distance": avg_d,
            "gini_inequality": max(0.0, min(1.0, gini)),
        })

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


