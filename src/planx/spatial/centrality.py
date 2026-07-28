# -*- coding: utf-8 -*-
"""Centrality measures on CSR graphs.

* Closeness / harmonic closeness / straightness from chunked Dijkstra rows.
* Betweenness via Brandes (2001), weighted, with optional radius limiting
  (on the routing cost or on a second "prune" weight, as space syntax choice
  requires) and optional source sampling for very large networks.

Everything is exact pure-Python/NumPy; SciPy only accelerates the
distance-matrix part.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from . import paths

INF = float("inf")
_EPS = 1e-9


def closeness_straightness(
    indptr, adj, weights, n, node_xy=None, radius=None, chunk=128, cancel=None, progress=None
):
    """Per-node reach, farness, closeness (Wasserman-Faust), harmonic
    closeness and (if ``node_xy`` given) straightness centrality.

    Returns dict of float64 arrays.
    """
    reach = np.zeros(n, dtype=np.float64)
    farness = np.zeros(n, dtype=np.float64)
    harmonic = np.zeros(n, dtype=np.float64)
    straight = np.zeros(n, dtype=np.float64)
    all_nodes = np.arange(n, dtype=np.int64)
    for start in range(0, n, chunk):
        if cancel is not None and cancel():
            break
        idx = all_nodes[start : start + chunk]
        dmat = paths.many_to_many(indptr, adj, weights, n, idx, cutoff=radius)
        with np.errstate(divide="ignore", invalid="ignore"):
            for row, s in enumerate(idx):
                d = dmat[row]
                mask = np.isfinite(d)
                mask[s] = False
                r = int(mask.sum())
                reach[s] = r
                if r == 0:
                    continue
                ds = d[mask]
                farness[s] = ds.sum()
                harmonic[s] = (1.0 / ds).sum()
                if node_xy is not None:
                    eu = np.hypot(
                        node_xy[mask, 0] - node_xy[s, 0], node_xy[mask, 1] - node_xy[s, 1]
                    )
                    ok = ds > 0
                    straight[s] = float((eu[ok] / ds[ok]).mean()) if ok.any() else 0.0
        if progress is not None:
            progress(min(1.0, (start + chunk) / max(1, n)))
    # Wasserman-Faust closeness handles disconnected graphs gracefully.
    closeness = np.zeros(n, dtype=np.float64)
    pos = farness > 0
    if n > 1:
        closeness[pos] = (reach[pos] / farness[pos]) * (reach[pos] / (n - 1))
    out = {"reach": reach, "farness": farness, "closeness": closeness, "harmonic": harmonic}
    if node_xy is not None:
        out["straightness"] = straight
    return out


def eigenvector(indptr, adj, n, max_iter=200, tol=1e-10):
    """Eigenvector centrality by power iteration on the binary adjacency.

    Influence of a junction given the influence of its neighbours
    (Bonacich). Normalized so the maximum is 1. Converges to the dominant
    connected component; isolated parts get near-zero scores.
    """
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    x = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(max_iter):
        # Power-iterate on A + I: same dominant eigenvector as A, but the
        # +I shift breaks the +/-lambda tie on bipartite graphs (trees,
        # grids) where plain iteration oscillates forever.
        nxt = x.copy()
        for u in range(n):
            s = x[u]
            if s != 0.0:
                # np.add.at: parallel edges must each contribute
                np.add.at(nxt, adj[indptr[u] : indptr[u + 1]], s)
        norm = float(np.sqrt((nxt * nxt).sum()))
        if norm <= 0.0:
            return np.zeros(n, dtype=np.float64)
        nxt /= norm
        if float(np.abs(nxt - x).max()) < tol:
            x = nxt
            break
        x = nxt
    peak = float(x.max())
    return x / peak if peak > 0 else x


def brandes_betweenness(
    indptr,
    adj,
    weights,
    n,
    adj_edge=None,
    num_edges=0,
    w_prune=None,
    radius=None,
    sources=None,
    cancel=None,
    progress=None,
    collect_depth=False,
):
    """Weighted betweenness (Brandes 2001) with options used across PlanX.

    ``w_prune``/``radius``: prune the search once the accumulated prune
    weight exceeds ``radius`` (e.g. angular cost minimized within a metric
    radius for space syntax choice).
    ``sources``: subset of source nodes (results scaled by n/len(sources)).
    ``collect_depth``: also return per-source-reachability stats needed by
    space syntax integration (node count + total cost depth per source),
    sharing the same Dijkstra pass.

    Returns (node_bc, edge_bc or None, depth_stats or None). For an
    undirected graph each unordered pair is counted twice (s->t and t->s);
    callers divide by 2 when reporting pair-based conventions.
    """
    node_bc = np.zeros(n, dtype=np.float64)
    edge_bc = np.zeros(num_edges, dtype=np.float64) if adj_edge is not None else None
    if collect_depth:
        depth_nc = np.zeros(n, dtype=np.float64)
        depth_td = np.zeros(n, dtype=np.float64)

    src_list = range(n) if sources is None else [int(s) for s in sources]
    total_sources = n if sources is None else len(src_list)

    dist = np.empty(n, dtype=np.float64)
    prune_d = np.empty(n, dtype=np.float64)
    sigma = np.empty(n, dtype=np.float64)
    delta = np.empty(n, dtype=np.float64)

    for done, s in enumerate(src_list):
        if cancel is not None and cancel():
            break
        if progress is not None and done % 64 == 0:
            progress(done / max(1, total_sources))
        dist.fill(INF)
        prune_d.fill(INF)
        sigma.fill(0.0)
        dist[s] = 0.0
        prune_d[s] = 0.0
        sigma[s] = 1.0
        preds = [[] for _ in range(n)]
        order = []
        heap = [(0.0, 0.0, s)]
        visited = np.zeros(n, dtype=bool)
        while heap:
            d, p, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            order.append(u)
            for k in range(indptr[u], indptr[u + 1]):
                v = adj[k]
                if visited[v]:
                    continue
                if w_prune is not None and radius is not None:
                    np_ = p + w_prune[k]
                    if np_ > radius:
                        continue
                else:
                    np_ = p
                nd = d + weights[k]
                tol = _EPS * max(1.0, abs(nd))
                if nd < dist[v] - tol:
                    dist[v] = nd
                    prune_d[v] = np_
                    sigma[v] = sigma[u]
                    preds[v] = [(u, k)]
                    heapq.heappush(heap, (nd, np_, v))
                elif abs(nd - dist[v]) <= tol:
                    sigma[v] += sigma[u]
                    preds[v].append((u, k))
        if collect_depth:
            depth_nc[s] = len(order)  # includes the source
            depth_td[s] = sum(dist[v] for v in order[1:])
        delta.fill(0.0)
        for v in reversed(order):
            coeff = (1.0 + delta[v]) / sigma[v] if sigma[v] > 0 else 0.0
            for u, k in preds[v]:
                c = sigma[u] * coeff
                delta[u] += c
                if edge_bc is not None:
                    edge_bc[adj_edge[k]] += c
            if v != s:
                node_bc[v] += delta[v]

    if sources is not None and total_sources > 0 and total_sources < n:
        scale = n / float(total_sources)
        node_bc *= scale
        if edge_bc is not None:
            edge_bc *= scale
    depth = {"node_count": depth_nc, "total_depth": depth_td} if collect_depth else None
    return node_bc, edge_bc, depth


def network_criticality(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray,
    n: int,
    origins: list[int] | np.ndarray,
    destinations: list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates network criticality (edge usage/betweenness proxy via OD routing).

    For each origin, finds the shortest path to the nearest destination on the
    network, and counts how many times each edge (represented by its index in the
    CSR adjacency list) is used.

    Args:
        indptr: CSR indptr array of shape (n + 1,)
        adj: CSR adj array of shape (E,)
        weights: CSR edge weights array of shape (E,)
        n: Number of nodes
        origins: List/array of origin node indices
        destinations: List/array of destination node indices

    Returns:
        Tuple of:
          - edge_usage: NumPy array of shape (E,) containing the count of times each edge was used.
          - edge_criticality: NumPy array of shape (E,) containing normalized score [0, 100].
    """
    import heapq

    edge_usage = np.zeros(len(adj), dtype=np.int64)
    destinations_set = {int(d) for d in destinations}

    if not destinations_set:
        return edge_usage, np.zeros_like(edge_usage, dtype=np.float64)

    for s in origins:
        s = int(s)
        dist = np.full(n, INF)
        dist[s] = 0.0
        pred = {}
        heap = [(0.0, s)]
        visited = np.zeros(n, dtype=bool)

        nearest_dest = None

        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True

            if u in destinations_set:
                nearest_dest = u
                break

            for k in range(indptr[u], indptr[u + 1]):
                v = adj[k]
                if visited[v]:
                    continue
                nd = d + weights[k]
                if nd < dist[v]:
                    dist[v] = nd
                    pred[v] = (u, k)
                    heapq.heappush(heap, (nd, v))

        if nearest_dest is None:
            continue

        curr = nearest_dest
        while curr != s and curr in pred:
            prev, edge_idx = pred[curr]
            edge_usage[edge_idx] += 1
            curr = prev

    max_usage = int(np.max(edge_usage)) if len(edge_usage) > 0 else 0
    if max_usage > 0:
        edge_criticality = (100.0 * edge_usage) / max_usage
    else:
        edge_criticality = np.zeros_like(edge_usage, dtype=np.float64)

    return edge_usage, edge_criticality


def street_orientation_entropy(
    indptr: np.ndarray,
    adj: np.ndarray,
    node_xy: np.ndarray,
    num_bins: int = 36,
) -> tuple[float, np.ndarray]:
    """Calculates the Shannon entropy of street orientations (bearings).

    Quantifies the directional disorder/order of the street network. A grid-like city
    has low entropy (streets concentrated in specific cardinal directions), whereas an
    organic city has high entropy (streets distributed uniformly in all directions).

    Args:
        indptr: CSR indptr array of shape (n + 1,)
        adj: CSR adj array of shape (E,)
        node_xy: NumPy array of shape (n, 2) containing node coordinates [X, Y].
        num_bins: Number of bins to group bearings into (default 36, which is 10 degrees each).

    Returns:
        Tuple of:
            - entropy: Shannon entropy of orientations normalized to [0.0, 1.0].
            - bin_proportions: 1D NumPy array of shape (num_bins,) containing proportion of
              streets in each orientation bin.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    adj = np.asarray(adj, dtype=np.int64)

    n = len(indptr) - 1
    if node_xy.shape != (n, 2):
        raise ValueError(f"node_xy shape ({node_xy.shape}) must match number of nodes ({n})")
    if num_bins <= 0:
        raise ValueError("num_bins must be greater than 0")

    # Reconstruct source nodes for all edges
    source_nodes = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))

    # Calculate dx and dy for each directed edge
    coords_src = node_xy[source_nodes]
    coords_dst = node_xy[adj]
    dx = coords_dst[:, 0] - coords_src[:, 0]
    dy = coords_dst[:, 1] - coords_src[:, 1]

    # Ignore zero-length self-loops or duplicate location nodes
    valid = (dx != 0.0) | (dy != 0.0)
    if not np.any(valid):
        return 0.0, np.zeros(num_bins, dtype=np.float64)

    dx = dx[valid]
    dy = dy[valid]

    # Calculate compass bearings: 0 is North, 90 is East, 180 is South, 270 is West.
    # Standard compass bearing is measured clockwise from North.
    # In standard math coordinates: X is East (dx), Y is North (dy).
    # bearing = arctan2(dx, dy) in radians
    bearings = np.degrees(np.arctan2(dx, dy))
    bearings = (bearings + 360.0) % 360.0

    # Bin bearings into num_bins equal slices between [0, 360)
    # np.histogram finds counts of bearings falling in each bin
    bin_edges = np.linspace(0.0, 360.0, num_bins + 1)
    counts, _ = np.histogram(bearings, bins=bin_edges)

    # Convert counts to proportions
    total_edges = float(np.sum(counts))
    if total_edges <= 0.0:
        return 0.0, np.zeros(num_bins, dtype=np.float64)

    p = counts / total_edges
    # Calculate Shannon entropy: H = -sum(p_i * log(p_i))
    # Avoid log(0) using np.errstate or np.where
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 0.0, np.log(p), 0.0)
    h = -np.sum(p * log_p)

    # Normalize entropy to range [0.0, 1.0] by dividing by log(num_bins) (max possible entropy)
    max_h = np.log(num_bins)
    normalized_entropy = float(h / max_h) if max_h > 0.0 else 0.0

    return normalized_entropy, p


def pagerank_centrality(
    indptr: np.ndarray,
    adj: np.ndarray,
    weights: np.ndarray | None = None,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Calculates PageRank centrality on a spatial network.

    PageRank centrality measures the structural importance of network nodes under random walk
    dynamics. Outgoing transitions can be unweighted or weighted inversely by distance.

    Args:
        indptr: CSR indptr array of shape (n + 1,)
        adj: CSR adj array of shape (E,)
        weights: Optional CSR edge weights array of shape (E,) representing travel distances/costs.
            If provided, transition probabilities are weighted inversely by cost (1 / weight).
        alpha: Damping factor (default 0.85).
        max_iter: Maximum number of power iterations (default 100).
        tol: Convergence tolerance (default 1e-6).

    Returns:
        1D NumPy array of shape (n,) containing PageRank scores.
    """
    indptr = np.asarray(indptr, dtype=np.int64)
    adj = np.asarray(adj, dtype=np.int64)

    n = len(indptr) - 1
    if n <= 0:
        return np.zeros(0, dtype=np.float64)

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha damping factor must be in range [0.0, 1.0]")
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0")
    if tol <= 0.0:
        raise ValueError("tol tolerance must be greater than 0.0")

    # Construct source nodes for all edges
    source_nodes = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr))

    transition_probs = np.zeros(len(adj), dtype=np.float64)
    dangling_mask = np.zeros(n, dtype=bool)

    for u in range(n):
        start = indptr[u]
        end = indptr[u + 1]
        if start == end:
            dangling_mask[u] = True
            continue

        if weights is None:
            w_out = np.ones(end - start, dtype=np.float64)
        else:
            w_out = np.asarray(weights[start:end], dtype=np.float64)
            # Avoid division by zero, handle negative weights
            w_out = np.where(w_out > 0.0, 1.0 / w_out, 0.0)

        sum_w = np.sum(w_out)
        if sum_w <= 0.0:
            w_out = np.ones(end - start, dtype=np.float64)
            sum_w = float(end - start)

        transition_probs[start:end] = w_out / sum_w

    x = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(max_iter):
        x_new = np.zeros(n, dtype=np.float64)

        dangling_sum = np.sum(x[dangling_mask])

        # Matrix-vector multiplication Mx
        contributions = transition_probs * x[source_nodes]
        np.add.at(x_new, adj, contributions)

        # Apply damping factor and distribute dangling probability
        x_new = (alpha * x_new) + (alpha * dangling_sum / n) + ((1.0 - alpha) / n)

        # Check convergence
        err = np.sum(np.abs(x_new - x))
        x = x_new
        if err < tol:
            break

    return x


def angular_segment_centrality(
    segment_coords: list[tuple[tuple[float, float], tuple[float, float]]],
    radius: float = 800.0,
    radius_type: str = "angular",
) -> dict[str, np.ndarray]:
    """Calculates Space Syntax Angular Integration (NAIn) and Angular Choice (NACh) metrics.

    Args:
        segment_coords: List of line segment endpoints [((x1, y1), (x2, y2)), ...].
        radius: Cutoff radius threshold float.
        radius_type: "angular" (turn angle steps) or "metric" (Euclidean meter distance).

    Returns:
        Dict containing 1D NumPy arrays of segment metrics:
          - angular_integration: Mean depth inverse integration.
          - angular_choice: Betweenness choice count.
          - nain: Normalised Angular Integration.
          - nach: Normalised Angular Choice.
    """
    m = len(segment_coords)
    if m == 0:
        return {
            "angular_integration": np.zeros(0, dtype=np.float64),
            "angular_choice": np.zeros(0, dtype=np.float64),
            "nain": np.zeros(0, dtype=np.float64),
            "nach": np.zeros(0, dtype=np.float64),
        }

    vecs = []
    midpoints = []
    endpoint_map: dict[tuple[float, float], list[int]] = {}

    for idx, ((x1, y1), (x2, y2)) in enumerate(segment_coords):
        dx, dy = x2 - x1, y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        if length > 0:
            vecs.append((dx / length, dy / length))
        else:
            vecs.append((1.0, 0.0))

        midpoints.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        pt1 = (round(x1, 6), round(y1, 6))
        pt2 = (round(x2, 6), round(y2, 6))
        endpoint_map.setdefault(pt1, []).append(idx)
        endpoint_map.setdefault(pt2, []).append(idx)

    adj_matrix = np.full((m, m), float("inf"), dtype=np.float64)
    np.fill_diagonal(adj_matrix, 0.0)

    for seg_list in endpoint_map.values():
        if len(seg_list) > 1:
            for i in range(len(seg_list)):
                for k in range(i + 1, len(seg_list)):
                    u, v = seg_list[i], seg_list[k]
                    dot = max(-1.0, min(1.0, vecs[u][0] * vecs[v][0] + vecs[u][1] * vecs[v][1]))
                    angle_deg = math.degrees(math.acos(abs(dot)))
                    weight = angle_deg / 90.0
                    adj_matrix[u, v] = min(adj_matrix[u, v], weight)
                    adj_matrix[v, u] = min(adj_matrix[v, u], weight)

    from scipy.sparse.csgraph import dijkstra

    dist_matrix = dijkstra(csgraph=adj_matrix, directed=False)

    m_coords = np.array(midpoints)

    integration = np.zeros(m, dtype=np.float64)
    choice = np.zeros(m, dtype=np.float64)
    nain = np.zeros(m, dtype=np.float64)
    nach = np.zeros(m, dtype=np.float64)

    for i in range(m):
        if radius_type == "metric":
            eucl_dists = np.sqrt(np.sum((m_coords - m_coords[i]) ** 2, axis=1))
            valid = (eucl_dists <= radius) & np.isfinite(dist_matrix[i])
        else:
            valid = (dist_matrix[i] <= radius) & np.isfinite(dist_matrix[i])

        nc = np.sum(valid)
        td = np.sum(dist_matrix[i, valid])

        if nc > 1:
            mean_depth = td / (nc - 1)
            integration[i] = 1.0 / max(1e-9, mean_depth)
            nain[i] = (nc**1.2) / (td + 2.0)
        else:
            integration[i] = 0.0
            nain[i] = 0.0

    for s in range(m):
        for t in range(s + 1, m):
            d_st = dist_matrix[s, t]
            if not np.isinf(d_st) and d_st > 0:
                for v in range(m):
                    if v != s and v != t:
                        if np.isclose(dist_matrix[s, v] + dist_matrix[v, t], d_st):
                            choice[v] += 1.0

    for i in range(m):
        td = np.sum(dist_matrix[i, np.isfinite(dist_matrix[i])])
        nach[i] = math.log(choice[i] + 1.0) / math.log(td + 3.0)

    return {
        "angular_integration": integration,
        "angular_choice": choice,
        "nain": nain,
        "nach": nach,
    }


def axial_to_segment_conversion(
    axial_coords: list[tuple[tuple[float, float], tuple[float, float]]],
    tol: float = 1e-6,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Converts continuous axial lines into topological line segments split at all intersections.

    Args:
        axial_coords: List of line endpoints [((x1, y1), (x2, y2)), ...].
        tol: Coordinate snapping tolerance float.

    Returns:
        List of sub-segments split at intersection points [((x1, y1), (x2, y2)), ...].
    """
    if len(axial_coords) == 0:
        return []

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def line_intersection(
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        p4: tuple[float, float],
    ) -> tuple[float, float] | None:
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if abs(denom) < tol:
            return None

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        if 0.0 <= ua <= 1.0 and 0.0 <= ub <= 1.0:
            ix = x1 + ua * (x2 - x1)
            iy = y1 + ua * (y2 - y1)
            return (round(ix, 6), round(iy, 6))
        return None

    for i, (p1, p2) in enumerate(axial_coords):
        splits = [p1, p2]
        for k, (p3, p4) in enumerate(axial_coords):
            if i != k:
                intersect = line_intersection(p1, p2, p3, p4)
                if intersect is not None:
                    splits.append(intersect)

        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        o0, o1 = float(p1[0]), float(p1[1])
        vx, vy = float(dx), float(dy)

        def proj(
            pt: tuple[float, float],
            ox: float = o0,
            oy: float = o1,
            v_x: float = vx,
            v_y: float = vy,
        ) -> float:
            return (pt[0] - ox) * v_x + (pt[1] - oy) * v_y

        unique_splits = sorted(set(splits), key=proj)
        for idx in range(len(unique_splits) - 1):
            seg_start = unique_splits[idx]
            seg_end = unique_splits[idx + 1]
            dist_sq = (seg_end[0] - seg_start[0]) ** 2 + (seg_end[1] - seg_start[1]) ** 2
            if dist_sq > tol:
                segments.append((seg_start, seg_end))

    return segments
