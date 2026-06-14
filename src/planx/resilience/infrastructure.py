# -*- coding: utf-8 -*-
"""Infrastructure resilience and network disruption models."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def simulate_network_disruption(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    blocked_edges: Optional[Union[np.ndarray, List[int]]] = None,
    blocked_nodes: Optional[Union[np.ndarray, List[int]]] = None,
) -> np.ndarray:
    """Simulates network disruption by setting blocked edge weights to infinity.

    This operates on a CSR graph representation. Blocking a node sets all its outgoing
    and incoming edges to infinity. Modifies a copy of the weights array.

    Args:
        indptr: CSR indptr array of shape (n + 1,)
        adj: CSR adj array of shape (E,)
        weights: CSR edge weights array of shape (E,)
        n: Number of nodes in the graph
        blocked_edges: Optional list or array of edge indices (referring to the CSR
            edge list) to block.
        blocked_nodes: Optional list or array of node indices to block.

    Returns:
        new_weights: NumPy array of shape (E,) containing updated edge weights with
            blocked edges set to infinity.
    """
    new_weights = np.asarray(weights, dtype=np.float64).copy()

    if blocked_edges is not None:
        b_edges = np.asarray(blocked_edges, dtype=np.int64)
        if len(b_edges) > 0:
            if np.any((b_edges < 0) | (b_edges >= len(weights))):
                raise ValueError("Blocked edge indices must be within valid range [0, E)")
            new_weights[b_edges] = np.inf

    if blocked_nodes is not None:
        b_nodes = np.asarray(blocked_nodes, dtype=np.int64)
        if len(b_nodes) > 0:
            if np.any((b_nodes < 0) | (b_nodes >= n)):
                raise ValueError("Blocked node indices must be within valid range [0, n)")

            # 1. Block all outgoing edges from these nodes
            for u in b_nodes:
                new_weights[indptr[u] : indptr[u + 1]] = np.inf

            # 2. Block all incoming edges (destinations matching blocked nodes)
            incoming_blocked = np.isin(adj, b_nodes)
            new_weights[incoming_blocked] = np.inf

    return new_weights


def infrastructure_service_loss(
    dists_pre: np.ndarray,
    dists_post: np.ndarray,
    demands: Optional[np.ndarray] = None,
    cutoff: Optional[float] = None,
) -> Dict[str, float]:
    """Calculates network service loss comparing pre- and post-disruption distance matrices.

    Compares the distance matrices from M origins to N destinations before and after
    disruption to evaluate loss of access to critical facilities.

    Args:
        dists_pre: NumPy array of shape (M, N) containing pre-disruption travel
            distances/times.
        dists_post: NumPy array of shape (M, N) containing post-disruption travel
            distances/times.
        demands: Optional array of shape (M,) containing population or demand weight
            for each origin. If omitted, all origins are weighted equally.
        cutoff: Optional maximum distance/time threshold. If travel distance/time
            exceeds this threshold, the destination is considered unreachable.

    Returns:
        A dictionary containing:
          - "isolation_rate": Float in [0.0, 1.0] representing the fraction of
            origins/demand isolated.
          - "pop_isolated": Float representing the total demand/population that
            becomes completely isolated.
          - "mean_delay": Float representing the average increase in travel distance/time
            to the nearest reachable destination, calculated only for origins that
            remain connected post-disruption.
          - "service_vulnerability_index": Float in [0.0, 100.0] representing a combined
            score of population isolation and travel delay.
    """
    d_pre = np.asarray(dists_pre, dtype=np.float64)
    d_post = np.asarray(dists_post, dtype=np.float64)

    if d_pre.shape != d_post.shape:
        raise ValueError("dists_pre and dists_post must have identical shapes")

    m, _ = d_pre.shape

    if demands is None:
        dem = np.ones(m, dtype=np.float64)
    else:
        dem = np.asarray(demands, dtype=np.float64)
        if dem.shape != (m,):
            raise ValueError(f"demands array shape {dem.shape} must match number of origins {m}")

    total_demand = float(np.sum(dem))
    if total_demand <= 0:
        total_demand = 1.0

    # Minimum distance to nearest destination for each origin
    min_pre = np.min(d_pre, axis=1)
    min_post = np.min(d_post, axis=1)

    # Determine reachability
    if cutoff is not None:
        reachable_pre = (min_pre <= cutoff) & np.isfinite(min_pre)
        reachable_post = (min_post <= cutoff) & np.isfinite(min_post)
    else:
        reachable_pre = np.isfinite(min_pre)
        reachable_post = np.isfinite(min_post)

    # Isolated origins: reachable before but NOT reachable after
    isolated = reachable_pre & (~reachable_post)
    pop_isolated = float(np.sum(dem[isolated]))
    isolation_rate = pop_isolated / total_demand

    # Mean delay (increase in travel distance to nearest facility)
    # Calculated only for origins that were reachable before and remain reachable after
    connected = reachable_pre & reachable_post
    if np.any(connected):
        diff = min_post[connected] - min_pre[connected]
        # Clip diff at 0 in case numeric issues show minor decreases
        diff = np.clip(diff, 0.0, None)
        mean_delay = float(np.average(diff, weights=dem[connected]))

        # Calculate percent increase for the service vulnerability index
        # Avoid division by zero by using max(min_pre, 1.0)
        pct_increase = diff / np.maximum(min_pre[connected], 1.0)
        avg_pct_increase = float(np.average(pct_increase, weights=dem[connected]))
    else:
        mean_delay = 0.0
        avg_pct_increase = 0.0

    # Service Vulnerability Index (SVI-infra):
    # Combined score from 0 to 100 based on isolation rate and delay index
    # We map avg_pct_increase to a logistic or capped scale
    delay_factor = np.clip(avg_pct_increase, 0.0, 1.0)  # Capped at 100% delay
    svi_infra = 100.0 * (0.7 * isolation_rate + 0.3 * delay_factor)

    return {
        "isolation_rate": isolation_rate,
        "pop_isolated": pop_isolated,
        "mean_delay": mean_delay,
        "service_vulnerability_index": svi_infra,
    }


def identify_critical_bottlenecks(
    edge_usage_pre: np.ndarray,
    edge_usage_post: np.ndarray,
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Identifies network edge bottlenecks experiencing the highest increase in usage
    post-disruption.

    Args:
        edge_usage_pre: NumPy array containing pre-disruption usage count/weight for each
            edge.
        edge_usage_post: NumPy array containing post-disruption usage count/weight for each
            edge.
        top_k: Number of top bottlenecks to return.

    Returns:
        Tuple of:
          - bottleneck_indices: NumPy array of the indices of the top K bottleneck edges.
          - load_increases: NumPy array of the absolute usage increases for those edges.
    """
    pre = np.asarray(edge_usage_pre, dtype=np.float64)
    post = np.asarray(edge_usage_post, dtype=np.float64)

    if pre.shape != post.shape:
        raise ValueError("edge_usage_pre and edge_usage_post must have the same shape")

    diff = post - pre
    # We are interested in positive increases (bottlenecks due to redirected traffic)
    diff = np.clip(diff, 0.0, None)

    # Sort indices by descending difference
    sorted_idx = np.argsort(diff)[::-1]

    k = min(top_k, len(diff))
    bottleneck_indices = sorted_idx[:k]
    load_increases = diff[bottleneck_indices]

    return bottleneck_indices, load_increases


def prioritize_debris_clearance(
    blocked_edges: np.ndarray,
    debris_volumes: np.ndarray,
    edge_criticality: np.ndarray,
    cost_factor: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prioritizes blocked edges for debris clearance based on utility-to-cost ratio.

    Utility is represented by edge criticality (e.g. usage count/centrality), and cost
    is represented by the debris volume to be cleared.

    Args:
        blocked_edges: 1D NumPy array of edge indices that are blocked.
        debris_volumes: 1D NumPy array containing the debris volume (m3) at each blocked
            edge. Must match blocked_edges in length.
        edge_criticality: 1D NumPy array of shape (E,) containing the network
            usage/importance score of each edge in the graph.
        cost_factor: Weight coefficient for the cost (debris volume). Higher values penalize
            high-debris roads more severely.

    Returns:
        Tuple of:
          - clearance_order: NumPy array of blocked edge indices sorted by priority (descending).
          - priority_scores: NumPy array of the computed priority scores for the sorted edges.
    """
    b_edges = np.asarray(blocked_edges, dtype=np.int64)
    vol = np.asarray(debris_volumes, dtype=np.float64)
    crit = np.asarray(edge_criticality, dtype=np.float64)

    if len(b_edges) != len(vol):
        raise ValueError("blocked_edges and debris_volumes must have the same length")

    if np.any(vol < 0):
        raise ValueError("debris_volumes must contain non-negative values")

    # Get criticality of blocked edges
    # Avoid index out of bounds
    if len(b_edges) > 0 and (np.any(b_edges < 0) or np.any(b_edges >= len(crit))):
        raise ValueError("blocked_edges indices must be within valid range of edge_criticality")

    b_crit = crit[b_edges]

    # Priority = Criticality / (Debris Volume ** cost_factor + epsilon)
    # Epsilon prevents division by zero
    eps = 1e-6
    scores = b_crit / (np.power(vol, cost_factor) + eps)

    # Sort descending
    sorted_idx = np.argsort(scores)[::-1]

    return b_edges[sorted_idx], scores[sorted_idx]
