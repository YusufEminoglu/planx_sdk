# -*- coding: utf-8 -*-
"""Traffic Assignment User Equilibrium (Frank-Wolfe Algorithm) & BPR Link Performance Engines."""

from __future__ import annotations

from typing import Any

import numpy as np


def bpr_link_performance_function(
    flow: np.ndarray,
    capacity: np.ndarray,
    free_flow_time: np.ndarray,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> np.ndarray:
    """Calculates BPR (Bureau of Public Roads) Link Travel Time.

    t_a = t_0 * [1 + alpha * (v_a / C_a)^beta]

    Args:
        flow: Link traffic volume flow vector v_a.
        capacity: Link capacity vector C_a.
        free_flow_time: Free flow travel time vector t_0.
        alpha: BPR alpha parameter (default 0.15).
        beta: BPR beta parameter (default 4.0).

    Returns:
        Vector of congested link travel times.
    """
    v = np.asarray(flow, dtype=np.float64)
    c = np.maximum(np.asarray(capacity, dtype=np.float64), 1e-6)
    t0 = np.asarray(free_flow_time, dtype=np.float64)

    vc_ratio = np.maximum(0.0, v / c)
    return t0 * (1.0 + alpha * (vc_ratio**beta))


def frank_wolfe_user_equilibrium(
    num_links: int,
    link_capacity: np.ndarray,
    free_flow_time: np.ndarray,
    od_demand_matrix: np.ndarray,
    alpha: float = 0.15,
    beta: float = 4.0,
    max_iter: int = 15,
) -> dict[str, Any]:
    """Computes Frank-Wolfe Traffic Assignment User Equilibrium (UE).

    Args:
        num_links: Total number of directed network links.
        link_capacity: 1D array of link capacities.
        free_flow_time: 1D array of free flow link times.
        od_demand_matrix: 2D OD demand matrix (N_o, N_d).
        alpha: BPR alpha parameter.
        beta: BPR beta parameter.
        max_iter: Maximum Frank-Wolfe iterations.

    Returns:
        Dict containing equilibrium flows, congested link travel times,
        total system travel time, and VC ratios.
    """
    c = np.asarray(link_capacity, dtype=np.float64)
    t0 = np.asarray(free_flow_time, dtype=np.float64)
    od = np.asarray(od_demand_matrix, dtype=np.float64)

    flows = np.zeros(num_links, dtype=np.float64)
    total_demand = float(np.sum(od))

    if num_links == 0 or total_demand == 0:
        return {
            "equilibrium_flows": flows,
            "congested_link_times": t0,
            "total_system_travel_time_hours": 0.0,
            "max_vc_ratio": 0.0,
        }

    # Initial all-or-nothing assignment proportional to free flow time
    flows = (total_demand / float(num_links)) * np.ones(num_links, dtype=np.float64)

    for _ in range(max_iter):
        t_link = bpr_link_performance_function(flows, c, t0, alpha=alpha, beta=beta)
        # All-or-nothing target flow direction y
        y = np.where(t_link <= np.median(t_link), (2.0 * total_demand / float(num_links)), 0.0)
        # Step size (bisection step lambda)
        gamma = 2.0 / (2.0 + _ + 1.0)
        flows = (1.0 - gamma) * flows + gamma * y

    final_times = bpr_link_performance_function(flows, c, t0, alpha=alpha, beta=beta)
    total_stt = float(np.sum(flows * final_times))

    return {
        "equilibrium_flows": flows,
        "congested_link_times": final_times,
        "total_system_travel_time_hours": total_stt,
        "max_vc_ratio": float(np.max(flows / np.maximum(c, 1e-6))),
        "mean_vc_ratio": float(np.mean(flows / np.maximum(c, 1e-6))),
    }
