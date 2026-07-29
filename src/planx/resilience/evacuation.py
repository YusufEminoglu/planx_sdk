# -*- coding: utf-8 -*-
"""Disaster evacuation route optimization engine."""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np


def evacuation_route_optimization(
    adj_matrix: np.ndarray,
    capacity_matrix: np.ndarray,
    origin_demands: dict[int, float],
    depot_nodes: list[int],
    vehicle_speed: float = 40.0,
) -> dict:
    """Calculates capacity-constrained evacuation routes, bottlenecks, and clearance time.

    Args:
        adj_matrix: (N, N) distance matrix between network nodes (inf or <=0 for no edge).
        capacity_matrix: (N, N) capacity matrix (vehicles/hour) for each directed edge.
        origin_demands: Dict mapping node index to number of evacuating vehicles/population.
        depot_nodes: List of safe destination depot node indices.
        vehicle_speed: Free-flow vehicle speed in km/h or distance units per hour.

    Returns:
        Dict containing evacuation metrics:
          - assigned_flows: (N, N) NumPy array of assigned vehicle flows per edge.
          - bottleneck_edges: List of tuples (u, v) where flow exceeds capacity.
          - edge_congestion_ratios: (N, N) NumPy array of flow-to-capacity ratios.
          - clearance_time_hours: Estimated maximum network clearance time in hours float.
          - unassigned_demand: Dict of remaining unassigned demand per origin node.
    """
    dist = np.asarray(adj_matrix, dtype=np.float64)
    cap = np.asarray(capacity_matrix, dtype=np.float64)
    n = dist.shape[0]

    if dist.shape != (n, n) or cap.shape != (n, n):
        raise ValueError("adj_matrix and capacity_matrix must be square 2D arrays of same shape.")

    if not depot_nodes:
        raise ValueError("At least one safe depot node must be specified.")

    flows = np.zeros((n, n), dtype=np.float64)
    unassigned = dict(origin_demands)

    depot_set = set(depot_nodes)

    def dijkstra_to_depot(start_node: int) -> tuple[float, list[int]]:
        """Dijkstra shortest path from start_node to nearest depot."""
        distances = {i: float("inf") for i in range(n)}
        distances[start_node] = 0.0
        predecessors: dict[int, int | None] = dict.fromkeys(range(n))
        pq = [(0.0, start_node)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > distances[u]:
                continue
            if u in depot_set:
                path = []
                curr: int | None = u
                while curr is not None:
                    path.append(curr)
                    curr = predecessors[curr]
                return d, path[::-1]

            for v in range(n):
                edge_d = dist[u, v]
                if edge_d > 0 and not np.isinf(edge_d):
                    new_d = d + edge_d
                    if new_d < distances[v]:
                        distances[v] = new_d
                        predecessors[v] = u
                        heapq.heappush(pq, (new_d, v))

        return float("inf"), []

    max_travel_time = 0.0

    for u_node, demand in list(origin_demands.items()):
        if demand <= 0 or u_node in depot_set:
            unassigned[u_node] = 0.0
            continue

        d_path_dist, path = dijkstra_to_depot(u_node)
        if not path or np.isinf(d_path_dist):
            continue

        for idx in range(len(path) - 1):
            u_edge, v_edge = path[idx], path[idx + 1]
            flows[u_edge, v_edge] += demand

        unassigned[u_node] = 0.0
        travel_time = d_path_dist / max(1e-9, vehicle_speed)
        if travel_time > max_travel_time:
            max_travel_time = travel_time

    congestion = np.zeros((n, n), dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        congestion = np.where(cap > 0, flows / cap, 0.0)

    bottlenecks = []
    max_queue_delay = 0.0

    for u in range(n):
        for v in range(n):
            if cap[u, v] > 0 and flows[u, v] > cap[u, v]:
                bottlenecks.append((u, v))
                queue_delay = (flows[u, v] - cap[u, v]) / cap[u, v]
                if queue_delay > max_queue_delay:
                    max_queue_delay = queue_delay

    clearance_time = max_travel_time + max_queue_delay

    return {
        "assigned_flows": flows,
        "bottleneck_edges": bottlenecks,
        "edge_congestion_ratios": congestion,
        "clearance_time_hours": float(clearance_time),
        "unassigned_demand": unassigned,
    }


def dynamic_evacuation_bottlenecks(
    origin_demands: np.ndarray,
    destination_capacities: np.ndarray,
    edge_list: np.ndarray,
    edge_capacities: np.ndarray,
    edge_free_flow_times: np.ndarray,
    time_horizon_steps: int = 24,
) -> dict[str, Any]:
    """
    Dynamic bottleneck and congestion propagation simulation for disaster evacuation networks.

    Args:
        origin_demands: (N,) evacuating vehicles/population per node.
        destination_capacities: (N,) safe shelter capacity per node (0 if not a shelter).
        edge_list: (E, 2) int array of (u, v) directed edges.
        edge_capacities: (E,) vehicles/step capacity per edge.
        edge_free_flow_times: (E,) steps needed to traverse edge under free flow.
        time_horizon_steps: T simulation time steps.

    Returns:
        Dict with keys:
        - `total_evacuated`: float total evacuees reached safety
        - `clearance_time_step`: int time step when evacuation completed / reached 95%
        - `edge_max_vcr`: (E,) float array of max volume-to-capacity ratios per edge
        - `edge_total_queues`: (E,) float array of cumulative queue length per edge
        - `critical_bottlenecks`: (K, 2) int array of top bottleneck edge indices
        - `time_series_evacuated`: (T,) float array of cumulative evacuees at each step
    """
    demands = np.asarray(origin_demands, dtype=np.float64)
    capacities = np.asarray(destination_capacities, dtype=np.float64)
    edges = np.asarray(edge_list, dtype=int)
    cap_e = np.asarray(edge_capacities, dtype=np.float64)
    fft_e = np.asarray(edge_free_flow_times, dtype=np.float64)

    N = demands.shape[0]
    E = edges.shape[0]

    if capacities.shape[0] != N:
        raise ValueError("origin_demands and destination_capacities must have same length.")
    if edges.size > 0 and edges.shape[1] != 2:
        raise ValueError("edge_list must be shape (E, 2).")
    if cap_e.shape[0] != E or fft_e.shape[0] != E:
        raise ValueError("edge_capacities and edge_free_flow_times must have length E.")

    if np.any(demands < 0) or np.any(capacities < 0) or np.any(cap_e < 0) or np.any(fft_e < 0):
        raise ValueError("Inputs cannot be negative.")

    if time_horizon_steps <= 0:
        raise ValueError("time_horizon_steps must be positive.")

    current_demand = demands.copy()
    rem_caps = capacities.copy()

    edge_max_vcr = np.zeros(E, dtype=np.float64)
    edge_total_queues = np.zeros(E, dtype=np.float64)
    time_series_evacuated = np.zeros(time_horizon_steps, dtype=np.float64)
    t_e = fft_e.copy()

    adj_rev: list[list[tuple[int, int]]] = [[] for _ in range(N)]
    for i, (u, v) in enumerate(edges):
        if 0 <= u < N and 0 <= v < N:
            adj_rev[v].append((u, i))
        else:
            raise ValueError(f"Edge ({u}, {v}) contains invalid node index.")

    total_evacuated = 0.0
    initial_total_demand = float(np.sum(demands))
    clearance_time_step = -1

    if initial_total_demand == 0.0:
        clearance_time_step = 0

    for t in range(time_horizon_steps):
        # 1. Admit to shelters
        for i in range(N):
            if rem_caps[i] > 0 and current_demand[i] > 0:
                admitted = min(current_demand[i], rem_caps[i])
                current_demand[i] -= admitted
                rem_caps[i] -= admitted
                total_evacuated += admitted

        # 2. Dijkstra from shelters
        dist = np.full(N, np.inf)
        next_edge = np.full(N, -1, dtype=int)

        pq: list[tuple[float, int]] = []
        for i in range(N):
            if rem_caps[i] > 0:
                dist[i] = 0.0
                heapq.heappush(pq, (0.0, i))

        while pq:
            d, v = heapq.heappop(pq)
            if d > dist[v]:
                continue
            for u, e_idx in adj_rev[v]:
                new_d = d + t_e[e_idx]
                if new_d < dist[u]:
                    dist[u] = new_d
                    next_edge[u] = e_idx
                    heapq.heappush(pq, (new_d, u))

        # 3. Calculate v_e
        v_e = np.zeros(E, dtype=np.float64)
        for u in range(N):
            if current_demand[u] > 0 and next_edge[u] != -1:
                v_e[next_edge[u]] += current_demand[u]

        # 4. Move evacuees
        new_demand = current_demand.copy()
        for e_idx in range(E):
            if v_e[e_idx] > 0:
                u, v = edges[e_idx]
                flow = min(v_e[e_idx], cap_e[e_idx])
                new_demand[u] -= flow
                new_demand[v] += flow
        current_demand = new_demand

        # 5. Metrics and BPR update
        for e_idx in range(E):
            c = cap_e[e_idx]
            if c <= 0:
                c = 1e-9
            ratio = v_e[e_idx] / c
            q = max(0.0, v_e[e_idx] - c)
            edge_total_queues[e_idx] += q
            if ratio > edge_max_vcr[e_idx]:
                edge_max_vcr[e_idx] = ratio
            t_e[e_idx] = fft_e[e_idx] * (1.0 + 0.15 * (ratio**4))

        # 6. Record
        time_series_evacuated[t] = total_evacuated
        if clearance_time_step == -1 and initial_total_demand > 0:
            if total_evacuated >= 0.95 * initial_total_demand:
                clearance_time_step = t

    top_k = min(5, E)
    if top_k > 0:
        sorted_indices = np.argsort(edge_max_vcr)[::-1]
        critical_bottlenecks = edges[sorted_indices[:top_k]]
    else:
        critical_bottlenecks = np.empty((0, 2), dtype=int)

    return {
        "total_evacuated": float(total_evacuated),
        "clearance_time_step": int(clearance_time_step),
        "edge_max_vcr": edge_max_vcr,
        "edge_total_queues": edge_total_queues,
        "critical_bottlenecks": critical_bottlenecks,
        "time_series_evacuated": time_series_evacuated,
    }
