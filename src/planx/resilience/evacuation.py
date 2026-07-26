# -*- coding: utf-8 -*-
"""Disaster evacuation route optimization engine."""

from __future__ import annotations

import heapq

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
