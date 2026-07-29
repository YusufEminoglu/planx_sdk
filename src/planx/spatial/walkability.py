# -*- coding: utf-8 -*-
"""Advanced urban mobility, walkability, and active transport routing models.

Features inspired by CityForm Lab (MIT), Senseable City Lab (MIT), Future Cities
Laboratory (Singapore/TU Delft/ETH), and the Transport Studies Unit (Oxford).
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .paths import many_to_many

INF = float("inf")


def simulate_thermal_comfort_pet(
    air_temp: np.ndarray,
    rel_humidity: np.ndarray,
    wind_speed: np.ndarray,
    solar_radiation: np.ndarray,
    sky_view_factor: np.ndarray,
    canopy_cover: np.ndarray,
    solar_absorption: float = 0.7,
) -> np.ndarray:
    """Estimates Physiological Equivalent Temperature (PET) index values for street segments.

    Calculates Mean Radiant Temperature (Tmrt) and thermal comfort index using empirical
    microclimate relationships from Singapore Future Cities Laboratory.

    Formula:
        T_mrt = T_air + solar_absorption * I_solar * (1.0 - Canopy) * SVF
        PET = T_air + 0.344 * (T_mrt - T_air) - 0.05 * v_wind * (T_mrt - T_air) + Humidity_factor

    Args:
        air_temp: 1D NumPy array of air temperature (Celsius).
        rel_humidity: 1D NumPy array of relative humidity (0 to 100).
        wind_speed: 1D NumPy array of wind speed (m/s).
        solar_radiation: 1D NumPy array of incoming solar radiation (W/m2).
        sky_view_factor: 1D NumPy array of Sky View Factor (0.0 to 1.0).
        canopy_cover: 1D NumPy array of tree canopy coverage (0.0 to 1.0).
        solar_absorption: Solar absorption coefficient of human skin/clothing (default 0.7).

    Returns:
        1D NumPy array of estimated PET values (Celsius).
    """
    t_air = np.asarray(air_temp, dtype=np.float64)
    rh = np.asarray(rel_humidity, dtype=np.float64)
    v_wind = np.clip(
        np.asarray(wind_speed, dtype=np.float64), 0.1, None
    )  # guard against division or negative wind
    i_solar = np.asarray(solar_radiation, dtype=np.float64)
    svf = np.asarray(sky_view_factor, dtype=np.float64)
    canopy = np.asarray(canopy_cover, dtype=np.float64)

    # 1. Estimate Mean Radiant Temperature (Tmrt)
    # Tmrt scales with solar radiation adjusted by canopy shading and Sky View Factor
    t_mrt = t_air + (solar_absorption * i_solar * (1.0 - canopy) * svf) / 10.0

    # 2. Compute humidity factor (vapour pressure contribution)
    # Saturation vapour pressure (hPa) using Tetens equation
    e_sat = 6.11 * 10.0 ** (7.5 * t_air / (237.3 + t_air))
    e_actual = e_sat * (rh / 100.0)

    # Empirical PET approximation under light wind and solar exposure
    pet = t_air + 0.344 * (t_mrt - t_air) - 0.05 * v_wind * (t_mrt - t_air) + 0.07 * e_actual
    return np.clip(pet, -10.0, 60.0)


def thermal_comfort_routing(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    start_node: int,
    end_node: int,
    shade_factors: np.ndarray,
    heat_weights: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> Dict[str, Union[List[int], float]]:
    """Calculates a shade-optimized, thermal-comfort walking route between two nodes.

    Friction is dynamically adjusted based on shade/canopy cover and optional solar/heat
    exposure (e.g. sky view factor, surface temperature).

    Adjusted Edge Weight formula:
        W_adjusted = W * (1.0 + alpha * (1.0 - S) + (1.0 - alpha) * H)
    Where:
        W = original travel time/distance
        S = shade factor [0.0 (fully exposed) to 1.0 (fully shaded/canopy)]
        H = heat/solar weight (normalized [0.0 to 1.0])
        alpha = balance parameter between shade weight and heat weight [0.0 to 1.0]

    Args:
        indptr: CSR representation indptr array of shape (n + 1,).
        adj: CSR representation adj array of shape (E,).
        weights: CSR representation edge weights array of shape (E,).
        n: Total number of nodes in the network.
        start_node: Starting node index.
        end_node: Target destination node index.
        shade_factors: 1D array of shape (E,) representing the shade/canopy factor of each edge.
        heat_weights: Optional 1D array of shape (E,) representing the heat/solar exposure.
        alpha: Weight parameter balancing shade vs heat exposure [0, 1].

    Returns:
        Dict containing:
            - "path": List of node indices representing the optimal comfort path.
            - "shortest_path": List of node indices representing the absolute shortest path.
            - "comfort_distance": Original distance of the comfort path.
            - "shortest_distance": Original distance of the shortest path.
            - "comfort_index": Mean shade factor along the comfort path.
    """
    s = int(start_node)
    t = int(end_node)

    if s < 0 or s >= n or t < 0 or t >= n:
        raise ValueError("Start or end node is out of bounds")

    num_edges = len(adj)
    s_factors = np.asarray(shade_factors, dtype=np.float64)
    if len(s_factors) != num_edges:
        raise ValueError(
            f"shade_factors length ({len(s_factors)}) must match number of edges ({num_edges})"
        )

    if heat_weights is not None:
        h_weights = np.asarray(heat_weights, dtype=np.float64)
        if len(h_weights) != num_edges:
            raise ValueError(
                f"heat_weights length ({len(h_weights)}) must match number of edges ({num_edges})"
            )
    else:
        h_weights = np.zeros(num_edges, dtype=np.float64)

    # Compute adjusted weights for comfort routing
    adjusted_weights = weights * (1.0 + alpha * (1.0 - s_factors) + (1.0 - alpha) * h_weights)

    # Helper function to run Dijkstra and reconstruct path
    def get_path(edge_costs: np.ndarray) -> Tuple[List[int], float]:
        dist = np.full(n, np.inf)
        dist[s] = 0.0
        pred = {}
        heap = [(0.0, s)]
        visited = np.zeros(n, dtype=bool)

        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True

            if u == t:
                break

            for k in range(indptr[u], indptr[u + 1]):
                v = adj[k]
                if visited[v]:
                    continue
                nd = d + edge_costs[k]
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v] = (u, k)
                    heapq.heappush(heap, (nd, v))

        if not np.isfinite(dist[t]):
            return [], float("inf")

        path = []
        curr = t
        orig_dist = 0.0
        while curr != s:
            prev, edge_idx = pred[curr]
            path.append(curr)
            orig_dist += float(weights[edge_idx])
            curr = prev
        path.append(s)
        path.reverse()
        return path, orig_dist

    comfort_path, comfort_dist = get_path(adjusted_weights)
    shortest_path, shortest_dist = get_path(weights)

    # Compute average shade factor along the comfort path
    comfort_idx = 0.0
    if comfort_path:
        edge_shades = []
        node_to_idx = {comfort_path[i]: i for i in range(len(comfort_path))}
        for u in comfort_path[:-1]:
            for k in range(indptr[u], indptr[u + 1]):
                v = adj[k]
                if v in node_to_idx and node_to_idx[v] == node_to_idx[u] + 1:
                    edge_shades.append(s_factors[k])
                    break
        comfort_idx = float(np.mean(edge_shades)) if edge_shades else 1.0

    return {
        "path": comfort_path,
        "shortest_path": shortest_path,
        "comfort_distance": comfort_dist,
        "shortest_distance": shortest_dist,
        "comfort_index": comfort_idx,
    }


def reach_centrality_una(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    destinations: np.ndarray,
    destination_weights: np.ndarray,
    cutoff: float,
    decay_method: str = "exponential",
    beta: float = 0.002,
) -> np.ndarray:
    """Computes Reach Centrality for each origin node (MIT CityForm Lab UNA style).

    Reach measures the total weight of destinations reachable from each origin node
    within a network distance threshold, discounted by a decay function.

    Formula:
        Reach^r(i) = Sum_j (W_j * f(d_ij))

    Args:
        indptr: CSR indptr array of shape (n + 1,).
        adj: CSR adj array of shape (E,).
        weights: CSR edge weights array of shape (E,).
        n: Total number of nodes in the network.
        destinations: 1D array of destination node indices.
        destination_weights: 1D array of weights for each destination.
        cutoff: Network search distance cutoff (r).
        decay_method: 'exponential', 'power', 'linear', or 'none'.
        beta: Decay parameter.

    Returns:
        NumPy array of shape (n,) containing reach centrality values.
    """
    return gravity_centrality_una(
        indptr=indptr,
        adj=adj,
        weights=weights,
        n=n,
        destination_weights=destination_weights,
        destinations=destinations,
        cutoff=cutoff,
        decay_method=decay_method,
        beta=beta,
    )


def gravity_centrality_una(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    destination_weights: np.ndarray,
    destinations: np.ndarray,
    cutoff: float,
    decay_method: str = "exponential",
    beta: float = 0.002,
) -> np.ndarray:
    """Computes network Gravity Centrality for each node (MIT CityForm Lab UNA style).

    Formula:
        Gravity(i) = Sum_j (W_j * f(d_ij))
    Where:
        W_j = Attraction weight of destination node j
        d_ij = Shortest network distance from node i to destination j (restricted by cutoff)
        f(d) = Distance decay function: 'exponential', 'power', or 'linear'

    Args:
        indptr: CSR representation indptr array of shape (n + 1,).
        adj: CSR representation adj array of shape (E,).
        weights: CSR representation edge weights array of shape (E,).
        n: Total number of nodes in the network.
        destination_weights: 1D array of shape (D,) representing attraction weights.
        destinations: 1D array of shape (D,) representing node indices of destinations.
        cutoff: Maximum network distance threshold.
        decay_method: One of 'exponential', 'power', or 'linear'.
        beta: Decay parameter.

    Returns:
        NumPy array of shape (n,) containing gravity centrality values for each node.
    """
    dest = np.asarray(destinations, dtype=np.int64)
    dest_w = np.asarray(destination_weights, dtype=np.float64)

    if len(dest) != len(dest_w):
        raise ValueError("destinations and destination_weights must have identical length")

    # Retrieve distance matrix from all nodes to all destinations
    # Shape: (n, D)
    dmat = many_to_many(indptr, adj, weights, n, np.arange(n), cutoff=cutoff)
    dists_to_dest = dmat[:, dest]  # select columns corresponding to destinations

    decay = np.zeros_like(dists_to_dest)
    method_lower = decay_method.lower().replace(" ", "_").replace("-", "_")

    mask = (dists_to_dest <= cutoff) & np.isfinite(dists_to_dest)

    with np.errstate(divide="ignore", invalid="ignore"):
        if method_lower in ("none", "uniform"):
            decay = np.ones_like(dists_to_dest)
        elif method_lower == "exponential":
            decay = np.exp(-beta * dists_to_dest)
        elif method_lower == "power":
            safe_d = np.where(dists_to_dest > 0, dists_to_dest, 1e-9)
            decay = safe_d ** (-beta)
        elif method_lower == "linear":
            if cutoff <= 0:
                raise ValueError("linear decay requires a positive cutoff value")
            decay = 1.0 - (dists_to_dest / cutoff)
            decay = np.clip(decay, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown decay method: {decay_method}")

    decay[~mask] = 0.0
    decay[~np.isfinite(dists_to_dest)] = 0.0

    # Gravity centrality is the sum of weighted decay values
    return np.sum(decay * dest_w[None, :], axis=1)


def choice_centrality_una(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    origins: np.ndarray,
    destinations: np.ndarray,
    origin_weights: np.ndarray,
    destination_weights: np.ndarray,
    cutoff: float,
    decay_method: str = "none",
    beta: float = 0.0,
) -> np.ndarray:
    """Computes Weighted Choice (Betweenness) Centrality with search radius and decay (MIT style).

    Choice measures how many shortest paths from origins to destinations pass through each node.

    Formula:
        Choice^r(u) = Sum_s Sum_t ( (sigma_st(u) / sigma_st) * P_s * W_t * f(d_st) )

    Args:
        indptr: CSR indptr array of shape (n + 1,).
        adj: CSR adj array of shape (E,).
        weights: CSR edge weights array of shape (E,).
        n: Total number of nodes in the network.
        origins: 1D array of origin node indices.
        destinations: 1D array of destination node indices.
        origin_weights: 1D array of weights (population) for origins.
        destination_weights: 1D array of weights (jobs/attraction) for destinations.
        cutoff: Maximum network distance (search radius).
        decay_method: 'exponential', 'power', 'linear', or 'none'.
        beta: Decay parameter.

    Returns:
        NumPy array of shape (n,) containing choice centrality scores for each node.
    """
    origs = np.asarray(origins, dtype=np.int64)
    orig_w = np.asarray(origin_weights, dtype=np.float64)
    dests = np.asarray(destinations, dtype=np.int64)
    dest_w = np.asarray(destination_weights, dtype=np.float64)

    if len(origs) != len(orig_w):
        raise ValueError("origins and origin_weights must have identical length")
    if len(dests) != len(dest_w):
        raise ValueError("destinations and destination_weights must have identical length")

    dest_set = set(dests)
    dest_weight_map = {int(d): float(w) for d, w in zip(dests, dest_w)}

    choice = np.zeros(n, dtype=np.float64)
    method_lower = decay_method.lower().replace(" ", "_").replace("-", "_")

    if method_lower not in ("none", "exponential", "power", "linear"):
        raise ValueError(f"Unknown decay method: {decay_method}")
    if method_lower == "linear" and cutoff <= 0:
        raise ValueError("linear decay requires a positive cutoff value")

    # Helper function to compute decay factor
    def get_decay(dist_val: float) -> float:
        if dist_val > cutoff:
            return 0.0
        if method_lower == "none":
            return 1.0
        elif method_lower == "exponential":
            return float(np.exp(-beta * dist_val))
        elif method_lower == "power":
            safe_d = max(dist_val, 1e-9)
            return float(safe_d ** (-beta))
        else:  # linear
            return float(max(0.0, 1.0 - (dist_val / cutoff)))

    # Run Dijkstra for each origin
    for s_idx, s in enumerate(origs):
        s_val = int(s)
        p_s = float(orig_w[s_idx])

        dist = np.full(n, np.inf)
        dist[s_val] = 0.0
        sigma = np.zeros(n, dtype=np.float64)
        sigma[s_val] = 1.0

        # Dijkstra queues and order
        heap = [(0.0, s_val)]
        preds: list[list[int]] = [[] for _ in range(n)]
        order = []
        visited = np.zeros(n, dtype=bool)

        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            order.append(u)

            for k in range(indptr[u], indptr[u + 1]):
                v = adj[k]
                if visited[v]:
                    continue
                nd = d + weights[k]
                if nd > cutoff:
                    continue

                tol = 1e-9 * max(1.0, abs(nd))
                if nd < dist[v] - tol:
                    dist[v] = nd
                    sigma[v] = sigma[u]
                    preds[v] = [u]
                    heapq.heappush(heap, (nd, v))
                elif abs(nd - dist[v]) <= tol:
                    sigma[v] += sigma[u]
                    preds[v].append(u)

        # Backpropagation of dependencies (Brandes 2001 adaptation for OD choice)
        # delta[v] = sum_{w: v in pred[w]} (sigma_v/sigma_w) * (contrib + delta[w])
        delta = np.zeros(n, dtype=np.float64)

        # Precompute destination contributions for this source
        dest_contrib = np.zeros(n, dtype=np.float64)
        for u in order:
            if u in dest_set and u != s_val:
                d_val = float(dist[u])
                dest_contrib[u] = dest_weight_map[u] * get_decay(d_val)

        for w in reversed(order):
            for u in preds[w]:
                coeff = sigma[u] / sigma[w] if sigma[w] > 0 else 0.0
                # When moving backward, we accumulate the choices passing through node u
                c = coeff * (dest_contrib[w] + delta[w])
                delta[u] += c

        # Choice centrality is the accumulated flow
        for u in order:
            if u != s_val:
                choice[u] += p_s * delta[u]

    return choice


def classify_level_of_traffic_stress(
    speed_limit: np.ndarray,
    num_lanes: np.ndarray,
    has_bike_lane: np.ndarray,
    has_sidewalk: np.ndarray,
    daily_traffic: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Classifies street segments into Levels of Traffic Stress (LTS 1 to 4).

    LTS classes:
      - LTS 1: Comfortable for children (low speeds, low traffic volume, or separate path).
      - LTS 2: Comfortable for most mainstream adults (minor arterials, slow streets).
      - LTS 3: Comfortable for enthused/confident cyclists (multi-lane roads, moderate speeds).
      - LTS 4: High stress (multi-lane arterials, fast traffic, mixed flow).

    Args:
        speed_limit: 1D array of speed limits (km/h or mph).
        num_lanes: 1D array representing total number of traffic lanes.
        has_bike_lane: 1D boolean array indicating bike lane presence.
        has_sidewalk: 1D boolean array indicating pedestrian sidewalk presence.
        daily_traffic: Optional 1D array representing Average Daily Traffic (ADT) volume.

    Returns:
        1D NumPy array of LTS classes (integers 1, 2, 3, or 4).
    """
    speeds = np.asarray(speed_limit, dtype=np.float64)
    lanes = np.asarray(num_lanes, dtype=np.int64)
    bike = np.asarray(has_bike_lane, dtype=bool)
    sidewalk = np.asarray(has_sidewalk, dtype=bool)
    n = len(speeds)

    if daily_traffic is not None:
        traffic = np.asarray(daily_traffic, dtype=np.float64)
    else:
        traffic = np.zeros(n, dtype=np.float64)

    lts = np.full(n, 4, dtype=np.int64)  # default to high stress (LTS 4)

    for i in range(n):
        sp = speeds[i]
        ln = lanes[i]
        bk = bike[i]
        sw = sidewalk[i]
        tf = traffic[i]

        # LTS 1: Low speed, small street, sidewalks present
        if sp <= 30.0 and ln <= 2 and sw:
            if tf <= 3000.0 or bk:
                lts[i] = 1
                continue

        # LTS 2: Comfortable for standard active travelers
        if sp <= 40.0 and ln <= 2:
            if bk or sw:
                lts[i] = 2
                continue

        # LTS 3: Moderate stress
        if sp <= 50.0:
            if bk and ln <= 4:
                lts[i] = 3
                continue
            if ln <= 2 and tf <= 8000.0:
                lts[i] = 3
                continue

        # Otherwise defaults to LTS 4
        lts[i] = 4

    return lts


def identify_low_stress_islands(
    indptr: np.ndarray,
    adj: np.ndarray,
    n: int,
    edge_lts: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int], List[Tuple[int, int, int]]]:
    """Identifies low-stress islands (connected components) and key barriers/bridges.

    An island is a network component reachable using only low-stress edges (LTS <= 2).
    A barrier is a high-stress edge (LTS >= 3) whose mitigation would merge the largest islands.

    Args:
        indptr: CSR indptr array of shape (n + 1,).
        adj: CSR adj array of shape (E,).
        n: Total number of nodes.
        edge_lts: 1D array of shape (E,) representing LTS level for each edge.

    Returns:
        Tuple containing:
            - island_labels: 1D array of shape (n,) containing island component IDs.
            - island_sizes: Dict mapping island ID to number of nodes in that island.
            - barriers: List of tuples (edge_idx, u, v) of high-stress segments sorted by
              the size of the islands they connect (highest impact first).
    """
    lts = np.asarray(edge_lts, dtype=np.int64)
    island_labels = np.full(n, -1, dtype=np.int64)
    curr_island = 0

    # 1. Identify connected components using BFS on low-stress edges (LTS <= 2)
    for start in range(n):
        if island_labels[start] != -1:
            continue

        # Start BFS
        queue = [start]
        island_labels[start] = curr_island
        head = 0

        while head < len(queue):
            u = queue[head]
            head += 1

            for k in range(indptr[u], indptr[u + 1]):
                v = adj[k]
                if island_labels[v] == -1 and lts[k] <= 2:
                    island_labels[v] = curr_island
                    queue.append(v)

        curr_island += 1

    # Calculate island sizes
    island_sizes: dict[int, int] = {}
    for lbl in island_labels:
        island_sizes[int(lbl)] = island_sizes.get(int(lbl), 0) + 1

    # 2. Identify high-stress barrier edges that span between different islands
    barrier_edges = []
    seen_connections = set()

    for u in range(n):
        l_u = island_labels[u]
        for k in range(indptr[u], indptr[u + 1]):
            v = adj[k]
            l_v = island_labels[v]

            if l_u != l_v and lts[k] >= 3:
                if (u, v) not in seen_connections:
                    seen_connections.add((u, v))
                    # Score is the sum of sizes of the two connected islands (mitigation potential)
                    score = island_sizes[l_u] + island_sizes[l_v]
                    barrier_edges.append((score, int(k), u, v))

    # Sort barrier edges by score in descending order (highest mitigation impact first)
    barrier_edges.sort(key=lambda x: x[0], reverse=True)

    sorted_barriers = [(item[1], item[2], item[3]) for item in barrier_edges]
    return island_labels, island_sizes, sorted_barriers


def active_mobility_permeability(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    low_stress_mask: np.ndarray,
) -> np.ndarray:
    """Calculates active mobility network permeability index (Oxford TSU & TU Delft style).

    Measures the ratio of reachable nodes via only low-stress active travel infrastructure
    versus the full reachable network within a travel budget.

    Args:
        indptr: CSR representation indptr array of shape (n + 1,).
        adj: CSR representation adj array of shape (E,).
        weights: CSR representation edge weights array of shape (E,).
        n: Total number of nodes in the network.
        low_stress_mask: 1D boolean array of shape (E,) indicating if an edge is low-stress.

    Returns:
        NumPy array of shape (n,) containing permeability scores [0, 100].
    """
    ls_mask = np.asarray(low_stress_mask, dtype=bool)
    num_edges = len(adj)
    if len(ls_mask) != num_edges:
        raise ValueError(
            f"low_stress_mask length ({len(ls_mask)}) must match number of edges ({num_edges})"
        )

    # Original distances matrix
    full_dists = many_to_many(indptr, adj, weights, n, np.arange(n))

    # Low-stress only network weights: set high-stress edges to infinite cost
    ls_weights = weights.copy()
    ls_weights[~ls_mask] = np.inf

    ls_dists = many_to_many(indptr, adj, ls_weights, n, np.arange(n))

    # Permeability is the percentage of reachable nodes on the full network
    # that remain reachable under low-stress conditions.
    permeability = np.zeros(n, dtype=np.float64)

    for i in range(n):
        reachable_full = np.isfinite(full_dists[i])
        reachable_full[i] = False  # exclude self
        count_full = np.sum(reachable_full)

        if count_full > 0:
            reachable_ls = np.isfinite(ls_dists[i])
            reachable_ls[i] = False
            # intersect reachable nodes
            count_ls_reachable = np.sum(reachable_ls & reachable_full)
            permeability[i] = (100.0 * count_ls_reachable) / count_full
        else:
            permeability[i] = 100.0

    return permeability


def calculate_walk_score(
    amenity_distances: np.ndarray,
    amenity_weights: np.ndarray,
    intersection_density: np.ndarray,
    avg_block_length: np.ndarray,
) -> np.ndarray:
    """Calculates Walk Score (0-100) for location points based on amenity accessibility.

    Applies standard distance decay and penalties for low intersection density
    and high average block lengths.

    Distance Decay (Walk Score Methodology):
        - <= 400m (0.25 miles): 100% value
        - 400m - 800m (0.5 miles): linear decay to 75%
        - 800m - 1200m (0.75 miles): linear decay to 60%
        - 1200m - 1600m (1.0 mile): linear decay to 50%
        - 1600m - 2400m (1.5 miles): linear decay to 12.5%
        - > 2400m: 0% value

    Intersection Density Penalty:
        - >= 200 intersections/sq mi: 0% penalty
        - 150 - 200: 1% penalty
        - 120 - 150: 2% penalty
        - 90 - 120: 3% penalty
        - 60 - 90: 4% penalty
        - < 60: 5% penalty

    Average Block Length Penalty:
        - <= 120m: 0% penalty
        - 120m - 150m: 1% penalty
        - 150m - 180m: 2% penalty
        - 180m - 200m: 3% penalty
        - 200m - 250m: 4% penalty
        - > 250m: 5% penalty

    Args:
        amenity_distances: NumPy array of shape (M, N) containing distances (in meters)
            from M origins to N nearest amenities.
        amenity_weights: NumPy array of shape (N,) containing category weights (must sum to 1.0).
        intersection_density: NumPy array of shape (M,) containing local intersection density.
        avg_block_length: NumPy array of shape (M,) containing average local block length in meters.

    Returns:
        1D NumPy array of shape (M,) containing estimated Walk Scores in range [0, 100].
    """
    dists = np.asarray(amenity_distances, dtype=np.float64)
    w = np.asarray(amenity_weights, dtype=np.float64)
    int_dens = np.asarray(intersection_density, dtype=np.float64)
    block_len = np.asarray(avg_block_length, dtype=np.float64)

    m, n = dists.shape
    if w.shape != (n,):
        raise ValueError(f"amenity_weights shape ({w.shape}) must match number of amenities ({n})")
    if int_dens.shape != (m,) or block_len.shape != (m,):
        raise ValueError(
            "intersection_density and avg_block_length must have length equal to number of origins"
        )

    # Normalize weights to sum to 1
    total_w = np.sum(w)
    if total_w > 0.0:
        w = w / total_w
    else:
        w = np.ones_like(w) / n

    # Compute distance decay factors (M, N)
    decay = np.zeros_like(dists)

    # <= 400m
    mask_400 = dists <= 400.0
    decay[mask_400] = 1.0

    # 400m - 800m
    mask_800 = (dists > 400.0) & (dists <= 800.0)
    decay[mask_800] = 1.0 - 0.25 * ((dists[mask_800] - 400.0) / 400.0)

    # 800m - 1200m
    mask_1200 = (dists > 800.0) & (dists <= 1200.0)
    decay[mask_1200] = 0.75 - 0.15 * ((dists[mask_1200] - 800.0) / 400.0)

    # 1200m - 1600m
    mask_1600 = (dists > 1200.0) & (dists <= 1600.0)
    decay[mask_1600] = 0.60 - 0.10 * ((dists[mask_1600] - 1200.0) / 400.0)

    # 1600m - 2400m
    mask_2400 = (dists > 1600.0) & (dists <= 2400.0)
    decay[mask_2400] = 0.50 - 0.375 * ((dists[mask_2400] - 1600.0) / 800.0)

    # Weighted amenity accessibility score (M,)
    raw_scores = np.sum(decay * w[None, :], axis=1) * 100.0

    # Calculate penalties (M,)
    # Intersection density penalty
    int_penalty = np.zeros(m, dtype=np.float64)
    int_penalty[int_dens < 200.0] = 0.01
    int_penalty[int_dens < 150.0] = 0.02
    int_penalty[int_dens < 120.0] = 0.03
    int_penalty[int_dens < 90.0] = 0.04
    int_penalty[int_dens < 60.0] = 0.05

    # Average block length penalty
    block_penalty = np.zeros(m, dtype=np.float64)
    block_penalty[block_len > 120.0] = 0.01
    block_penalty[block_len > 150.0] = 0.02
    block_penalty[block_len > 180.0] = 0.03
    block_penalty[block_len > 200.0] = 0.04
    block_penalty[block_len > 250.0] = 0.05

    total_penalty = int_penalty + block_penalty
    walk_scores = raw_scores * (1.0 - total_penalty)

    return np.clip(walk_scores, 0.0, 100.0)


def calculate_pedestrian_route_directness(
    network_distances: np.ndarray,
    origin_coords: np.ndarray,
    destination_coords: np.ndarray,
) -> np.ndarray:
    """Calculates the Pedestrian Route Directness (PRD) index for origin-destination pairs.

    PRD is the ratio of the network distance to the straight-line (Euclidean) distance.
    A value closer to 1.0 represents highly direct routes (good design/connectivity),
    whereas values > 1.5 indicate circuitous routes.

    PRD = d_network / d_euclidean

    Args:
        network_distances: NumPy array of shape (M, N) containing network path distances
            from M origins to N destinations.
        origin_coords: NumPy array of shape (M, 2) containing (X, Y) coordinates.
        destination_coords: NumPy array of shape (N, 2) containing (X, Y) coordinates.

    Returns:
        NumPy array of shape (M, N) containing PRD scores. Unreachable or collocated pairs
        (Euclidean distance of 0) will return NaN or 1.0 respectively.
    """
    net_d = np.asarray(network_distances, dtype=np.float64)
    origs = np.asarray(origin_coords, dtype=np.float64)
    dests = np.asarray(destination_coords, dtype=np.float64)

    m, n = net_d.shape
    if origs.shape != (m, 2):
        raise ValueError(
            f"origin_coords shape ({origs.shape}) must match number of origins ({m}, 2)"
        )
    if dests.shape != (n, 2):
        raise ValueError(
            f"destination_coords shape ({dests.shape}) must match number of destinations ({n}, 2)"
        )

    # Compute Euclidean distances (M, N)
    dx = origs[:, 0, None] - dests[None, :, 0]
    dy = origs[:, 1, None] - dests[None, :, 1]
    euclidean = np.sqrt(dx**2 + dy**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        prd = np.where(euclidean > 0.0, net_d / euclidean, 1.0)

    # Set unreachable routes (infinity network distance) to NaN
    prd[~np.isfinite(net_d)] = np.nan

    return prd


def calculate_average_route_circuity(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    node_xy: np.ndarray,
    num_samples: int = 100,
    seed: int | None = None,
) -> float:
    """Calculates the average network route circuity of a spatial network.

    Circuity is the ratio of the shortest network path distance to the straight-line
    (Euclidean) distance. Measures how circuitous/inefficient a network is due to layout
    or obstacles.

    Samples random origin-destination node pairs, computes shortest path network distances,
    and returns the average ratio across all reachable pairs.

    Args:
        indptr: CSR indptr array of shape (n + 1,)
        adj: CSR adj array of shape (E,)
        weights: CSR edge weights array of shape (E,)
        node_xy: NumPy array of shape (n, 2) containing node coordinates [X, Y].
        num_samples: Number of random node pairs to sample (default 100).
        seed: Random seed for sampling reproducibility.

    Returns:
        Average circuity ratio (float). Returns 1.0 if no valid pairs or paths exist.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    adj = np.asarray(adj, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    node_xy = np.asarray(node_xy, dtype=np.float64)

    n = len(indptr) - 1
    if n <= 1:
        return 1.0

    if node_xy.shape != (n, 2):
        raise ValueError(f"node_xy shape ({node_xy.shape}) must match number of nodes ({n})")
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")

    rng = np.random.RandomState(seed)

    # Sample random origin/destination pairs
    # Generate slightly more to account for self-pairs (u == v)
    u_raw = rng.randint(0, n, size=num_samples * 2)
    v_raw = rng.randint(0, n, size=num_samples * 2)

    valid_mask = u_raw != v_raw
    u = u_raw[valid_mask][:num_samples]
    v = v_raw[valid_mask][:num_samples]

    if len(u) == 0:
        return 1.0

    unique_sources = np.unique(u)
    # Compute distances from unique sources to all nodes
    dmat = many_to_many(indptr, adj, weights, n, unique_sources)

    # Map source node to row index in dmat
    source_to_row = {node: idx for idx, node in enumerate(unique_sources)}

    circuity_values = []
    for ui, vi in zip(u, v):
        row = source_to_row[ui]
        d_net = dmat[row, vi]
        if np.isinf(d_net):
            continue  # ignore unreachable pairs

        d_eucl = float(np.hypot(node_xy[ui, 0] - node_xy[vi, 0], node_xy[ui, 1] - node_xy[vi, 1]))
        if d_eucl > 0.0:
            circuity_values.append(d_net / d_eucl)

    if len(circuity_values) == 0:
        return 1.0

    return float(np.mean(circuity_values))


def profile_intersection_density_closeness(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    radius: float = 800.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Profiles intersection density and closeness using network-based catchments.

    Identifies walkable "town centers" or clusters of high connectivity by analyzing
    neighboring intersections within a given travel threshold (radius).

    For each intersection node i:
        - Density: Number of other intersections reachable within the travel radius.
        - Avg Distance: Mean network distance to all reachable intersections.
        - Profile Score: Density / Avg Distance (scaled to [0.0, 100.0]).

    Args:
        indptr: CSR indptr array of shape (n + 1,)
        adj: CSR adj array of shape (E,)
        weights: CSR edge weights array of shape (E,) representing travel distances/costs.
        radius: Travel network distance threshold (default 800.0 meters, ~10 mins walk).

    Returns:
        Tuple of:
            - density: 1D NumPy array of shape (n,) containing neighbor counts.
            - avg_distance: 1D NumPy array of shape (n,) containing average network distance.
            - profile_score: 1D NumPy array of shape (n,) containing combined scores in [0, 100].
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    adj = np.asarray(adj, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)

    n = len(indptr) - 1
    if n <= 0:
        return (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )

    if radius <= 0.0:
        raise ValueError("radius must be greater than 0.0")

    density = np.zeros(n, dtype=np.float64)
    avg_dist = np.zeros(n, dtype=np.float64)
    scores = np.zeros(n, dtype=np.float64)

    # Compute distances to all nodes within the network radius cutoff
    # Run in chunks to handle memory efficiently for large graphs
    chunk_size = 128
    all_nodes = np.arange(n, dtype=np.int64)

    for start in range(0, n, chunk_size):
        sources = all_nodes[start : start + chunk_size]
        dmat = many_to_many(indptr, adj, weights, n, sources, cutoff=radius)

        for idx, s in enumerate(sources):
            row_dists = dmat[idx]
            # Exclude the source node itself
            mask = (row_dists <= radius) & np.isfinite(row_dists)
            mask[s] = False

            k = int(np.sum(mask))
            density[s] = float(k)

            if k > 0:
                mean_d = float(np.mean(row_dists[mask]))
                avg_dist[s] = mean_d
                # score is density / mean distance
                scores[s] = float(k / mean_d) if mean_d > 0.0 else 0.0
            else:
                avg_dist[s] = 0.0
                scores[s] = 0.0

    # Normalize profile scores to range [0.0, 100.0]
    max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
    if max_score > 0.0:
        scores = (scores / max_score) * 100.0

    return density, avg_dist, scores


def street_network_morphometry(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    node_xy: np.ndarray,
) -> dict:
    """Calculates street network morphometric parameters and urban grain indicators.

    Args:
        indptr: CSR graph index pointer array (n + 1,).
        adj: CSR graph adjacency column indices array.
        weights: CSR edge weight/length array.
        node_xy: 2D NumPy array of node coordinates (n, 2).

    Returns:
        Dict containing morphometric indicators:
          - node_degrees: 1D NumPy array of nodal degrees (n,).
          - avg_link_length: Mean street segment length float.
          - meshedness_coefficient: Planar network meshedness alpha float [0, 1].
          - connectivity_index: Ratio of 3+-way intersections to total nodes float.
          - total_network_length: Sum of edge lengths float.
    """
    indptr_arr = np.asarray(indptr, dtype=np.int64)
    adj_arr = np.asarray(adj, dtype=np.int64)
    w_arr = np.asarray(weights, dtype=np.float64)
    n = len(indptr_arr) - 1

    if n <= 0:
        return {
            "node_degrees": np.zeros(0, dtype=np.int64),
            "avg_link_length": 0.0,
            "meshedness_coefficient": 0.0,
            "connectivity_index": 0.0,
            "total_network_length": 0.0,
        }

    degrees = np.diff(indptr_arr)
    num_edges = len(adj_arr) // 2

    tot_len = float(np.sum(w_arr) / 2.0)
    avg_len = float(tot_len / max(1, num_edges))

    if n > 2:
        max_faces = 2 * n - 5
        meshedness = max(0.0, min(1.0, (num_edges - n + 1) / max(1, max_faces)))
    else:
        meshedness = 0.0

    intersections = int(np.sum(degrees >= 3))
    conn_idx = float(intersections / n)

    return {
        "node_degrees": degrees,
        "avg_link_length": avg_len,
        "meshedness_coefficient": float(meshedness),
        "connectivity_index": conn_idx,
        "total_network_length": tot_len,
    }


def calculate_wind_comfort_lawson(
    building_heights: np.ndarray,
    street_widths: np.ndarray,
    ambient_wind_speed: float,
) -> dict:
    """Calculates Lawson pedestrian wind comfort and street canyon wind amplification.

    Args:
        building_heights: 1D array of average building heights H (m) along street segments.
        street_widths: 1D array of street canyon widths W (m) along street segments.
        ambient_wind_speed: Free stream wind speed at 10m reference height float (m/s).

    Returns:
        Dict containing wind comfort statistics:
          - wind_amplification_factor: 1D array of local wind speed ratios U_local/U_ambient.
          - local_wind_speed: 1D NumPy array of estimated street wind speeds (m/s).
          - lawson_class: List of Lawson comfort category strings.
    """
    h_arr = np.asarray(building_heights, dtype=np.float64)
    w_arr = np.asarray(street_widths, dtype=np.float64)
    n = len(h_arr)

    if len(w_arr) != n:
        raise ValueError("building_heights and street_widths must have equal length.")

    # Aspect ratio H/W
    aspect_ratio = h_arr / np.maximum(1e-9, w_arr)

    # Canyon wind speed amplification factor gamma
    amp_deep = 1.2 + 0.2 * np.minimum(aspect_ratio, 3.0)
    amp_shallow = 0.8 + 0.2 * aspect_ratio
    amp_factor = np.where(aspect_ratio > 1.5, amp_deep, amp_shallow)

    u_local = amp_factor * float(ambient_wind_speed)

    lawson_categories = []
    for u in u_local:
        if u < 1.8:
            lawson_categories.append("Sitting")
        elif u < 3.6:
            lawson_categories.append("Standing")
        elif u < 5.3:
            lawson_categories.append("Strolling")
        elif u < 7.6:
            lawson_categories.append("Business Walking")
        else:
            lawson_categories.append("Uncomfortable")

    return {
        "wind_amplification_factor": amp_factor,
        "local_wind_speed": u_local,
        "lawson_class": lawson_categories,
    }


def cul_de_sac_isolation_index(indptr: np.ndarray, adj: np.ndarray) -> dict:
    """Identifies dead-end street nodes (degree 1) and network cul-de-sac isolation ratio.

    Args:
        indptr: CSR graph index pointer 1D array of shape (N + 1,).
        adj: CSR graph adjacency column indices 1D array.

    Returns:
        Dict containing cul-de-sac isolation statistics:
          - cul_de_sac_nodes: 1D NumPy array of node IDs with degree == 1.
          - cul_de_sac_ratio: Float ratio of degree-1 nodes to total nodes.
          - total_dead_ends: Int count of dead-end nodes.
    """
    n = len(indptr) - 1
    if n == 0:
        return {
            "cul_de_sac_nodes": np.zeros(0, dtype=np.int64),
            "cul_de_sac_ratio": 0.0,
            "total_dead_ends": 0,
        }

    degrees = indptr[1:] - indptr[:-1]
    dead_ends = np.where(degrees == 1)[0]
    total_dead = len(dead_ends)
    ratio = float(total_dead / n)

    return {
        "cul_de_sac_nodes": dead_ends,
        "cul_de_sac_ratio": ratio,
        "total_dead_ends": total_dead,
    }


def street_orientation_rose_spectrum(
    segment_coords: list[tuple[tuple[float, float], tuple[float, float]]],
    n_bins: int = 36,
) -> dict:
    """Calculates street network orientation polar distribution and directional anisotropy index.

    Args:
        segment_coords: List of line endpoints [((x1, y1), (x2, y2)), ...].
        n_bins: Number of angular histogram bins int (default 36 for 10-degree intervals).

    Returns:
        Dict containing polar distribution statistics:
          - bin_edges_deg: 1D NumPy array of bin edges in degrees [0, 360].
          - orientation_counts: 1D NumPy array of segment counts per orientation bin.
          - orientation_lengths: 1D NumPy array of total length (m) per orientation bin.
          - anisotropy_index: Float orientation anisotropy score in [0, 1].
          - dominant_heading_deg: Float angle in degrees of dominant street grid orientation.
    """
    if len(segment_coords) == 0:
        bin_edges = np.linspace(0.0, 360.0, n_bins + 1)
        return {
            "bin_edges_deg": bin_edges,
            "orientation_counts": np.zeros(n_bins, dtype=np.int64),
            "orientation_lengths": np.zeros(n_bins, dtype=np.float64),
            "anisotropy_index": 0.0,
            "dominant_heading_deg": 0.0,
        }

    angles = []
    lengths = []
    for (x1, y1), (x2, y2) in segment_coords:
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        if length > 1e-9:
            angle_deg = math.degrees(math.atan2(dy, dx)) % 180.0
            angles.append(angle_deg)
            lengths.append(length)
            # Undirected line symmetry: add 180 deg opposite
            angles.append((angle_deg + 180.0) % 360.0)
            lengths.append(length)

    ang_arr = np.array(angles)
    len_arr = np.array(lengths)

    bin_edges = np.linspace(0.0, 360.0, n_bins + 1)
    counts, _ = np.histogram(ang_arr, bins=bin_edges)
    hist_lengths, _ = np.histogram(ang_arr, bins=bin_edges, weights=len_arr)

    dom_idx = int(np.argmax(hist_lengths))
    dom_heading = float((bin_edges[dom_idx] + bin_edges[dom_idx + 1]) / 2.0)

    total_len = np.sum(len_arr)
    if total_len > 0:
        p = hist_lengths / total_len
        p_clean = p[p > 0]
        entropy = -np.sum(p_clean * np.log(p_clean))
        max_entropy = np.log(n_bins)
        anisotropy = float(1.0 - (entropy / max(1e-9, max_entropy)))
    else:
        anisotropy = 0.0

    return {
        "bin_edges_deg": bin_edges,
        "orientation_counts": counts,
        "orientation_lengths": hist_lengths,
        "anisotropy_index": max(0.0, min(1.0, anisotropy)),
        "dominant_heading_deg": dom_heading,
    }


def pedestrian_level_of_service(
    sidewalk_width_m: np.ndarray,
    ped_flow_rate_per_min: np.ndarray,
    vehicle_lanes: np.ndarray,
    vehicle_speed_kmh: np.ndarray,
) -> dict:
    """Evaluates Pedestrian Level of Service (PLOS) based on HCM methodology.

    Computes a composite PLOS score from sidewalk geometry, pedestrian flow,
    and adjacent traffic characteristics, then classifies into LOS grades A-F.

    Args:
        sidewalk_width_m: 1D array of effective sidewalk widths in meters.
        ped_flow_rate_per_min: 1D array of pedestrian flow rate (peds/min/m).
        vehicle_lanes: 1D array of adjacent vehicle lane counts.
        vehicle_speed_kmh: 1D array of adjacent vehicle speed in km/h.

    Returns:
        Dict containing PLOS assessment:
          - plos_score: 1D NumPy array of composite PLOS scores.
          - los_grade: List of string LOS grades ("A" through "F").
          - space_per_ped_m2: 1D array of available space per pedestrian (m^2/ped).
    """
    sw = np.asarray(sidewalk_width_m, dtype=np.float64)
    flow = np.asarray(ped_flow_rate_per_min, dtype=np.float64)
    lanes = np.asarray(vehicle_lanes, dtype=np.float64)
    speed = np.asarray(vehicle_speed_kmh, dtype=np.float64)
    n = len(sw)

    if len(flow) != n or len(lanes) != n or len(speed) != n:
        raise ValueError("All input arrays must have equal length.")

    # Space per pedestrian (m^2/ped): effective width / max(flow, small eps)
    space = sw / np.maximum(flow, 1e-6)

    # Width factor: narrower sidewalks penalized
    fw = np.clip(sw / 3.0, 0.0, 1.0)

    # Traffic stress factor: more lanes and higher speed increase stress
    ft = 1.0 - np.clip((lanes * speed) / 300.0, 0.0, 1.0)

    # Flow comfort factor: lower density is better
    fc = np.clip(1.0 - flow / 75.0, 0.0, 1.0)

    # Composite PLOS score [0, 6]: higher = better
    plos = (0.4 * fw + 0.3 * ft + 0.3 * fc) * 6.0

    grades = []
    for val in plos.flat:
        if val >= 5.0:
            grades.append("A")
        elif val >= 4.0:
            grades.append("B")
        elif val >= 3.0:
            grades.append("C")
        elif val >= 2.0:
            grades.append("D")
        elif val >= 1.0:
            grades.append("E")
        else:
            grades.append("F")

    return {
        "plos_score": plos,
        "los_grade": grades,
        "space_per_ped_m2": space,
    }


def bike_network_low_stress_connectivity(
    edge_lts_scores: np.ndarray,
    edge_lengths_m: np.ndarray,
    graph_adj_matrix: np.ndarray,
    target_max_lts: int = 2,
) -> dict[str, Any]:
    """Evaluates bicycle stress levels and calculates connected low-stress island subgraphs.

    Args:
        edge_lts_scores: (E,) int array of Level of Traffic Stress per edge (1 to 4).
        edge_lengths_m: (E,) float array of edge lengths in meters.
        graph_adj_matrix: (V, V) sparse or dense adjacency matrix representing the street graph.
        target_max_lts: Max LTS threshold considered "low stress" (default 2).

    Returns:
        Dict with connectivity and permeability metrics.
    """
    lts = np.asarray(edge_lts_scores, dtype=np.int64)
    lengths = np.asarray(edge_lengths_m, dtype=np.float64)

    if np.any((lts < 1) | (lts > 4)):
        raise ValueError("LTS scores must be between 1 and 4")
    if target_max_lts < 1 or target_max_lts > 4:
        raise ValueError("target_max_lts must be between 1 and 4")
    if np.any(lengths < 0):
        raise ValueError("Edge lengths must be non-negative")

    is_low_stress = lts <= target_max_lts

    if sp.issparse(graph_adj_matrix):
        mat = graph_adj_matrix.tocoo()  # type: ignore
        v = mat.shape[0]
        rows = mat.row  # type: ignore
        cols = mat.col  # type: ignore
    else:
        mat = np.asarray(graph_adj_matrix)
        v = mat.shape[0]
        rows, cols = np.nonzero(mat)

    if len(rows) != len(lts):
        raise ValueError(
            f"Number of edges in graph_adj_matrix ({len(rows)}) "
            f"must match array length ({len(lts)})"
        )

    # Filter for low stress edges
    ls_rows = rows[is_low_stress]
    ls_cols = cols[is_low_stress]
    ls_data = np.ones(len(ls_rows), dtype=int)

    ls_adj = sp.coo_matrix((ls_data, (ls_rows, ls_cols)), shape=(v, v))

    n_components, labels = connected_components(ls_adj, directed=False)

    edge_labels = labels[ls_rows]
    ls_lengths = lengths[is_low_stress]
    island_lengths = np.bincount(edge_labels, weights=ls_lengths, minlength=n_components)
    island_node_counts = np.bincount(labels, minlength=n_components)

    total_low_stress_length = float(np.sum(ls_lengths))
    total_length = float(np.sum(lengths))

    if total_length > 0.0:
        low_stress_length_ratio = total_low_stress_length / total_length
    else:
        low_stress_length_ratio = 0.0

    if total_low_stress_length > 0.0:
        largest_island_length = float(np.max(island_lengths))
        llsc_ratio = largest_island_length / total_low_stress_length
        largest_island_id = int(np.argmax(island_lengths))
        largest_island_nodes = int(island_node_counts[largest_island_id])
    else:
        largest_island_length = 0.0
        llsc_ratio = 0.0
        largest_island_nodes = 0

    num_islands = int(np.sum(island_lengths > 0))

    return {
        "low_stress_length_ratio": low_stress_length_ratio,
        "num_low_stress_islands": num_islands,
        "largest_island_length_m": largest_island_length,
        "largest_island_node_count": largest_island_nodes,
        "llsc_ratio": llsc_ratio,
        "edge_is_low_stress": is_low_stress,
    }


def microclimate_pedestrian_route_optimizer(
    graph_adj_matrix: np.ndarray,
    edge_lengths_m: np.ndarray,
    edge_shade_ratios: np.ndarray,
    edge_lst_temperatures: np.ndarray,
    origin_node: int,
    destination_node: int,
    heat_sensitivity_weight: float = 0.5,
) -> dict[str, Any]:
    """Finds optimal cool pedestrian routes balancing shortest distance against thermal discomfort.

    Args:
        graph_adj_matrix: (V, V) sparse or dense adjacency matrix representing street network graph.
        edge_lengths_m: (E,) length of edges in meters.
        edge_shade_ratios: (E,) tree/building canopy shade ratio [0, 1].
        edge_lst_temperatures: (E,) Land Surface Temperature (Celsius) per edge.
        origin_node: Start node index.
        destination_node: Destination node index.
        heat_sensitivity_weight: Weight w in [0, 1] balancing distance vs thermal discomfort.

    Returns:
        Dict containing path_nodes, total_distance_m, mean_shade_ratio, mean_lst_temperature,
        thermal_impedance_score, and shortest_distance_comparison.
    """
    from scipy.sparse.csgraph import dijkstra

    csr = sp.csr_matrix(graph_adj_matrix)
    v_count = csr.shape[0]
    if csr.shape[0] != csr.shape[1]:
        raise ValueError("graph_adj_matrix must be square")

    lengths = np.asarray(edge_lengths_m, dtype=np.float64)
    shades = np.asarray(edge_shade_ratios, dtype=np.float64)
    lsts = np.asarray(edge_lst_temperatures, dtype=np.float64)
    e_count = csr.nnz

    if lengths.shape[0] != e_count:
        raise ValueError(f"edge_lengths_m must match number of edges ({e_count})")
    if shades.shape[0] != e_count:
        raise ValueError(f"edge_shade_ratios must match number of edges ({e_count})")
    if lsts.shape[0] != e_count:
        raise ValueError(f"edge_lst_temperatures must match number of edges ({e_count})")

    if origin_node < 0 or origin_node >= v_count:
        raise ValueError("origin_node out of bounds")
    if destination_node < 0 or destination_node >= v_count:
        raise ValueError("destination_node out of bounds")
    if not (0.0 <= heat_sensitivity_weight <= 1.0):
        raise ValueError("heat_sensitivity_weight must be between 0 and 1")
    if np.any((shades < 0.0) | (shades > 1.0)):
        raise ValueError("edge_shade_ratios must be between 0 and 1")

    m_e = (1.0 - 0.5 * shades) * (lsts / 30.0)
    c_e = lengths * ((1.0 - heat_sensitivity_weight) + heat_sensitivity_weight * m_e)

    csr_dist = csr.copy()
    csr_dist.data = lengths
    dist_shortest, pred_shortest = dijkstra(
        csgraph=csr_dist, directed=True, indices=origin_node, return_predecessors=True
    )
    shortest_dist_m = float(dist_shortest[destination_node])

    csr_therm = csr.copy()
    csr_therm.data = c_e
    dist_thermal, pred_thermal = dijkstra(
        csgraph=csr_therm, directed=True, indices=origin_node, return_predecessors=True
    )
    thermal_impedance_score = float(dist_thermal[destination_node])

    if np.isinf(thermal_impedance_score) or pred_thermal[destination_node] == -9999:
        return {
            "path_nodes": [],
            "total_distance_m": 0.0,
            "mean_shade_ratio": 0.0,
            "mean_lst_temperature": 0.0,
            "thermal_impedance_score": float("inf"),
            "shortest_distance_comparison": {
                "shortest_dist_m": float("inf"),
                "extra_distance_pct": 0.0,
            },
        }

    path = []
    curr = destination_node
    while curr != origin_node and curr >= 0:
        path.append(int(curr))
        curr = pred_thermal[curr]
    path.append(int(origin_node))
    path.reverse()

    total_dist = 0.0
    total_shade = 0.0
    total_lst = 0.0
    num_edges = len(path) - 1

    for i in range(num_edges):
        u = path[i]
        v = path[i + 1]
        start_idx = csr.indptr[u]
        end_idx = csr.indptr[u + 1]
        v_indices = csr.indices[start_idx:end_idx]
        idx = np.where(v_indices == v)[0]
        if len(idx) == 0:
            raise RuntimeError(f"Edge {u}->{v} not found in graph")
        edge_idx = start_idx + idx[0]

        total_dist += lengths[edge_idx]
        total_shade += shades[edge_idx]
        total_lst += lsts[edge_idx]

    mean_shade = float(total_shade / num_edges) if num_edges > 0 else 0.0
    mean_lst = float(total_lst / num_edges) if num_edges > 0 else 0.0
    extra_dist_pct = 0.0
    if shortest_dist_m > 0 and not np.isinf(shortest_dist_m):
        extra_dist_pct = float(((total_dist - shortest_dist_m) / shortest_dist_m) * 100.0)

    return {
        "path_nodes": path,
        "total_distance_m": float(total_dist),
        "mean_shade_ratio": mean_shade,
        "mean_lst_temperature": mean_lst,
        "thermal_impedance_score": thermal_impedance_score,
        "shortest_distance_comparison": {
            "shortest_dist_m": shortest_dist_m,
            "extra_distance_pct": extra_dist_pct,
        },
    }
