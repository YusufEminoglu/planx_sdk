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
