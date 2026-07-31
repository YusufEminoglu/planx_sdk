# -*- coding: utf-8 -*-
"""Spatial weights matrix generation methods (e.g. k-NN, distance band)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def create_knn_weights(
    coords: np.ndarray,
    ids: list[int],
    k: int = 4,
    row_standardized: bool = True,
) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """Creates a k-Nearest Neighbors (k-NN) spatial weights matrix.

    Args:
        coords: NumPy array of shape (N, 2) containing point coordinates.
        ids: List of N unique integer IDs corresponding to the coordinates.
        k: Number of nearest neighbors to query (must be < N).
        row_standardized: If True, weights for each feature sum to 1.0.

    Returns:
        Tuple of:
          - neighbors: Dictionary mapping feature ID to list of neighboring feature IDs.
          - weights: Dictionary mapping feature ID to list of corresponding weights.
    """
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")
    if len(ids) != n:
        raise ValueError("ids length must match coordinates count")
    if k >= n:
        raise ValueError("k must be less than the number of points N")
    if k <= 0:
        raise ValueError("k must be greater than 0")

    # Build KDTree
    tree = cKDTree(pts)

    # Query nearest neighbors (k + 1 because the point itself is included at index 0)
    dists, indices = tree.query(pts, k=k + 1)

    neighbors_dict = {}
    weights_dict = {}

    for i, fid in enumerate(ids):
        # Ensure we take exactly k neighbors excluding self
        neigh_idx = [idx for idx in indices[i] if idx != i][:k]
        neigh_ids = [ids[idx] for idx in neigh_idx]
        neighbors_dict[fid] = neigh_ids

        # Set weights
        n_neighs = len(neigh_ids)
        if n_neighs > 0:
            if row_standardized:
                w = [1.0 / n_neighs] * n_neighs
            else:
                w = [1.0] * n_neighs
        else:
            w = []
        weights_dict[fid] = w

    return neighbors_dict, weights_dict


def create_distance_band_weights(
    coords: np.ndarray,
    ids: list[int],
    threshold: float,
    row_standardized: bool = True,
    binary: bool = True,
    power: float = 1.0,
) -> tuple[dict[int, list[int]], dict[int, list[float]]]:
    """Creates a Distance Band spatial weights matrix.

    Args:
        coords: NumPy array of shape (N, 2) containing point coordinates.
        ids: List of N unique integer IDs corresponding to the coordinates.
        threshold: Distance threshold. Points within this distance are neighbors.
        row_standardized: If True, weights for each feature sum to 1.0.
        binary: If True, weights are 1.0 (or 1/count). If False, inverse distance decay
            weights are used: 1.0 / (distance^power).
        power: Exponent for inverse distance weights (only used if binary is False).

    Returns:
        Tuple of:
          - neighbors: Dictionary mapping feature ID to list of neighboring feature IDs.
          - weights: Dictionary mapping feature ID to list of corresponding weights.
    """
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)

    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("coords must be of shape (N, 2)")
    if len(ids) != n:
        raise ValueError("ids length must match coordinates count")
    if threshold <= 0:
        raise ValueError("threshold must be greater than 0")

    # Build KDTree
    tree = cKDTree(pts)

    neighbors_dict = {}
    weights_dict = {}

    for i, fid in enumerate(ids):
        # Query points within threshold distance
        indices = tree.query_ball_point(pts[i], r=threshold)

        # Exclude self
        neigh_idx = [idx for idx in indices if idx != i]
        neigh_ids = [ids[idx] for idx in neigh_idx]
        neighbors_dict[fid] = neigh_ids

        n_neighs = len(neigh_ids)
        if n_neighs > 0:
            if binary:
                if row_standardized:
                    w = [1.0 / n_neighs] * n_neighs
                else:
                    w = [1.0] * n_neighs
            else:
                # Inverse distance weights
                neigh_pts = pts[neigh_idx]
                dists = np.sqrt(np.sum((neigh_pts - pts[i]) ** 2, axis=1))
                dists[dists < 1e-12] = 1e-12
                inv_dists = 1.0 / (dists**power)

                if row_standardized:
                    w_sum = np.sum(inv_dists)
                    w = list(inv_dists / w_sum if w_sum > 0 else inv_dists)
                else:
                    w = list(inv_dists)
        else:
            w = []
        weights_dict[fid] = w

    return neighbors_dict, weights_dict


def auto_distance_band(coords: list[tuple[float, float]] | np.ndarray) -> float:
    """Find smallest distance threshold guaranteeing every unit has >= 1 neighbor.

    Args:
        coords: List of (X, Y) coordinates or 2D array of shape (N, 2).

    Returns:
        Float threshold distance band.
    """
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return 0.0

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=2)
    return float(np.max(dists[:, 1]))


def build_weights(
    coords: list[tuple[float, float]] | np.ndarray,
    mode: str = "knn",
    k: int = 8,
    distance: float = 0.0,
    include_self: bool = False,
    row_standardize: bool = False,
    inverse_power: float = 1.0,
) -> list[list[tuple[int, float]]]:
    """Build a spatial-weights adjacency list for spatial analysis.

    Args:
        coords: List or array of (X, Y) coordinates.
        mode: "knn", "distance_band", or "inverse".
        k: Neighbor count for KNN mode.
        distance: Distance cutoff (0 for auto).
        include_self: Include self tuple (i, 1.0) if True.
        row_standardize: Normalize row weights to sum to 1.0 if True.
        inverse_power: Power exponent for inverse distance.

    Returns:
        List of adjacency lists where adj[i] is a list of (j, weight) tuples.
    """
    pts = np.asarray(coords, dtype=np.float64)
    n = len(pts)
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]

    if n < 2:
        if include_self:
            for i in range(n):
                adj[i].append((i, 1.0))
        return adj

    if mode == "knn":
        kk = max(1, min(k, n - 1))
        tree = cKDTree(pts)
        _, indices = tree.query(pts, k=kk + 1)
        for i in range(n):
            for idx in indices[i]:
                if idx != i:
                    adj[i].append((int(idx), 1.0))
    elif mode == "inverse":
        d = distance if distance > 0 else auto_distance_band(pts)
        d2cut = d * d
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d2 = float(np.sum((pts[i] - pts[j]) ** 2))
                if d2 <= d2cut:
                    dd = float(np.sqrt(d2))
                    w = 1.0 / (dd**inverse_power) if dd > 0 else 0.0
                    if w > 0:
                        adj[i].append((j, w))
    else:
        d = distance if distance > 0 else auto_distance_band(pts)
        d2cut = d * d
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if float(np.sum((pts[i] - pts[j]) ** 2)) <= d2cut:
                    adj[i].append((j, 1.0))

    if row_standardize:
        for i in range(n):
            s = sum(w for _, w in adj[i])
            if s > 0:
                adj[i] = [(j, w / s) for j, w in adj[i]]

    if include_self:
        for i in range(n):
            adj[i].append((i, 1.0))

    return adj


def neighbour_counts(adj: list[list[tuple[int, float]]]) -> list[int]:
    """Number of neighbors per unit.

    Args:
        adj: Adjacency list.

    Returns:
        List of neighbor count integers.
    """
    return [len(row) for row in adj]
