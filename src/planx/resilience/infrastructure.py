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


def network_criticality_index(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    target_edges: Optional[Union[np.ndarray, List[int]]] = None,
    target_nodes: Optional[Union[np.ndarray, List[int]]] = None,
    origins: Optional[Union[np.ndarray, List[int]]] = None,
    destinations: Optional[Union[np.ndarray, List[int]]] = None,
) -> Dict[str, np.ndarray]:
    """Calculates the Network Criticality Index (NCI) for target edges or nodes.

    NCI measures the relative drop in global network efficiency (or increase in
    shortest path distances) when a specific edge or node is blocked.

    Args:
        indptr: CSR indptr array of shape (n + 1,)
        adj: CSR adj array of shape (E,)
        weights: CSR edge weights array of shape (E,)
        n: Number of nodes in the graph.
        target_edges: Optional list or array of edge indices to evaluate.
        target_nodes: Optional list or array of node indices to evaluate.
        origins: Optional list of origin nodes for shortest path calculations.
            If None, all nodes are used.
        destinations: Optional list of destination nodes. If None, all nodes are used.

    Returns:
        A dictionary containing:
          - "edges_nci": 1D NumPy array of NCI values for target_edges (if provided).
          - "nodes_nci": 1D NumPy array of NCI values for target_nodes (if provided).
    """
    from ..spatial.paths import many_to_many

    origs = np.arange(n) if origins is None else np.asarray(origins, dtype=np.int64)
    dests = np.arange(n) if destinations is None else np.asarray(destinations, dtype=np.int64)

    # Helper function to compute network efficiency
    def compute_efficiency(w_curr: np.ndarray) -> float:
        dists = many_to_many(indptr, adj, w_curr, n, sources=origs)
        dists = dists[:, dests]
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dists = 1.0 / dists
            inv_dists[~np.isfinite(inv_dists)] = 0.0
            for i, o in enumerate(origs):
                if o in dests:
                    inv_dists[i, np.where(dests == o)[0][0]] = 0.0
        return float(np.sum(inv_dists))

    eff_base = compute_efficiency(weights)

    results = {}

    # 1. Edges NCI
    if target_edges is not None:
        t_edges = np.asarray(target_edges, dtype=np.int64)
        nci_edges = np.zeros(len(t_edges), dtype=np.float64)
        if eff_base > 0:
            for idx, e in enumerate(t_edges):
                w_disrupted = weights.copy()
                w_disrupted[e] = np.inf
                # Find u and v for edge e
                u = int(np.searchsorted(indptr, e, side="right") - 1)
                v = int(adj[e])
                # Find counterpart edge v -> u and block it too
                for k in range(indptr[v], indptr[v + 1]):
                    if adj[k] == u:
                        w_disrupted[k] = np.inf
                        break
                eff_disrupted = compute_efficiency(w_disrupted)
                nci_edges[idx] = (eff_base - eff_disrupted) / eff_base
        results["edges_nci"] = nci_edges

    # 2. Nodes NCI
    if target_nodes is not None:
        t_nodes = np.asarray(target_nodes, dtype=np.int64)
        nci_nodes = np.zeros(len(t_nodes), dtype=np.float64)
        if eff_base > 0:
            for idx, u in enumerate(t_nodes):
                w_disrupted = weights.copy()
                w_disrupted[indptr[u] : indptr[u + 1]] = np.inf
                incoming = np.isin(adj, [u])
                w_disrupted[incoming] = np.inf
                eff_disrupted = compute_efficiency(w_disrupted)
                nci_nodes[idx] = (eff_base - eff_disrupted) / eff_base
        results["nodes_nci"] = nci_nodes

    return results


def debris_clearance_routing(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    blocked_edges: np.ndarray,
    debris_volumes: np.ndarray,
    edge_criticality: np.ndarray,
    depot_node: int,
    cost_factor: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Computes a greedy clearance route for blocked edges starting from a depot node.

    At each step, selects the next edge that maximizes:
    Score = Criticality / (DebrisVolume ** cost_factor * (Distance to start + edge_weight) + eps)

    Args:
        indptr: CSR indptr array of shape (n + 1,)
        adj: CSR adj array of shape (E,)
        weights: CSR edge weights array of shape (E,)
        n: Number of nodes in the graph.
        blocked_edges: 1D NumPy array of edge indices that are blocked.
        debris_volumes: 1D NumPy array of debris volumes matching blocked_edges.
        edge_criticality: 1D NumPy array of shape (E,) containing edge criticality.
        depot_node: Starting node index for the clearing vehicle.
        cost_factor: Weight coefficient for debris volume penalty.

    Returns:
        Tuple of:
          - clearance_order: 1D NumPy array of blocked edge indices in order of clearance.
          - total_distance: Float representing the total travel distance of the vehicle.
    """
    from ..spatial.paths import many_to_many

    b_edges = list(np.asarray(blocked_edges, dtype=np.int64))
    vols = list(np.asarray(debris_volumes, dtype=np.float64))
    crit = np.asarray(edge_criticality, dtype=np.float64)

    if len(b_edges) != len(vols):
        raise ValueError("blocked_edges and debris_volumes must have the same length")

    # Find u and v for all blocked edges
    edge_nodes = []
    for e in b_edges:
        u = int(np.searchsorted(indptr, e, side="right") - 1)
        v = int(adj[e])
        edge_nodes.append((u, v))

    curr_node = int(depot_node)
    clearance_order = []
    total_distance = 0.0
    eps = 1e-6

    # During routing, we use a copy of weights where blocked edges are not yet cleared.
    # When an edge is cleared, its weight returns to normal (which is stored in weights).
    # Initially, all blocked edges are set to infinity in the routing weights.
    w_routing = weights.copy()
    w_routing[b_edges] = np.inf

    # Keep track of indices
    remaining_indices = list(range(len(b_edges)))

    while remaining_indices:
        # Calculate distances from curr_node to all nodes
        dists = many_to_many(indptr, adj, w_routing, n, sources=[curr_node])[0]

        best_score = -1.0
        best_rem_idx = -1
        best_dist_to_u = 0.0

        for rem_idx in remaining_indices:
            e = b_edges[rem_idx]
            u, v = edge_nodes[rem_idx]
            vol = vols[rem_idx]
            c = crit[e]
            w_orig = float(weights[e])

            dist_to_u = dists[u]
            if not np.isfinite(dist_to_u):
                # If currently unreachable, try fallback using open/unblocked graph distance
                # or set a large penalty. We use 1e6 penalty to keep it solvable.
                dist_to_u = 1e6

            # Compute heuristics score
            cost = np.power(vol, cost_factor) * (dist_to_u + w_orig)
            score = c / (cost + eps)

            if score > best_score:
                best_score = score
                best_rem_idx = rem_idx
                best_dist_to_u = dist_to_u

        # Add to route
        e_next = b_edges[best_rem_idx]
        u_next, v_next = edge_nodes[best_rem_idx]
        w_orig_next = float(weights[e_next])

        clearance_order.append(e_next)
        if best_dist_to_u < 1e6:
            total_distance += best_dist_to_u + w_orig_next
        else:
            # Fallback path cost if unreachable
            total_distance += w_orig_next

        # Move vehicle to the end of the cleared edge
        curr_node = v_next

        # Clear the edge in routing weights (restore original weight)
        w_routing[e_next] = w_orig_next

        # Also find and restore the counterpart edge v -> u if it exists
        for k in range(indptr[v_next], indptr[v_next + 1]):
            if adj[k] == u_next:
                w_routing[k] = float(weights[k])
                break

        remaining_indices.remove(best_rem_idx)

    return np.array(clearance_order, dtype=np.int64), float(total_distance)
