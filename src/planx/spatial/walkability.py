# -*- coding: utf-8 -*-
"""Advanced urban mobility, walkability, and active transport routing models.

Features inspired by CityForm Lab (MIT), Senseable City Lab (MIT), Future Cities
Laboratory (Singapore/TU Delft/ETH), and the Transport Studies Unit (Oxford).
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

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
        preds = [[] for _ in range(n)]
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
    island_sizes = {}
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
