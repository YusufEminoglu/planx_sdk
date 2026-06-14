# -*- coding: utf-8 -*-
"""Spatial interpolation algorithms (e.g. IDW, nearest neighbor)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


def idw_to_points(
    source_coords: np.ndarray,
    source_values: np.ndarray,
    target_coords: np.ndarray,
    power: float = 2.0,
    max_points: Optional[int] = None,
    search_radius: Optional[float] = None,
) -> np.ndarray:
    """Interpolates values at target point locations using Inverse Distance Weighting (IDW).

    Uses a fast KDTree to identify neighboring source points and computes a weighted average.

    Args:
        source_coords: 2D NumPy array of shape (N, 2) containing coordinates of source points.
        source_values: 1D NumPy array of shape (N,) containing values at source points.
        target_coords: 2D NumPy array of shape (M, 2) containing coordinates of target points.
        power: Distance decay exponent (usually 2.0).
        max_points: Maximum number of nearest source points to consider. Defaults to 12.
        search_radius: Maximum search radius. Neighbors beyond this distance are ignored.

    Returns:
        1D NumPy array of shape (M,) containing the interpolated values.
        Points with no valid neighbors will return NaN.
    """
    src_xy = np.asarray(source_coords, dtype=np.float64)
    src_val = np.asarray(source_values, dtype=np.float64)
    tgt_xy = np.asarray(target_coords, dtype=np.float64)

    if src_xy.ndim != 2 or src_xy.shape[1] != 2:
        raise ValueError("source_coords must be of shape (N, 2)")
    if src_val.ndim != 1 or src_val.shape[0] != src_xy.shape[0]:
        raise ValueError("source_values must be a 1D array of length N")
    if tgt_xy.ndim != 2 or tgt_xy.shape[1] != 2:
        raise ValueError("target_coords must be of shape (M, 2)")
    if power <= 0:
        raise ValueError("power must be greater than 0")

    n_src = len(src_xy)
    n_tgt = len(tgt_xy)

    if n_src == 0 or n_tgt == 0:
        return np.full(n_tgt, np.nan)

    # Build KDTree for quick distance queries
    tree = cKDTree(src_xy)

    k = max_points if max_points is not None else min(n_src, 12)
    upper_bound = search_radius if search_radius is not None else np.inf

    # Query KDTree
    dists, indices = tree.query(tgt_xy, k=k, distance_upper_bound=upper_bound)

    # If k == 1, tree.query returns 1D arrays; reshape to 2D for consistency
    if k == 1:
        dists = dists[:, None]
        indices = indices[:, None]

    interpolated = np.empty(n_tgt, dtype=np.float64)

    for i in range(n_tgt):
        d = dists[i]
        idx = indices[i]

        # Filter out invalid neighbors (unreachable due to search_radius or out of bounds)
        valid = (d < np.inf) & (idx < n_src)
        if not np.any(valid):
            interpolated[i] = np.nan
            continue

        d_valid = d[valid]
        idx_valid = idx[valid]

        # Check if any distance is extremely close to zero (exact match)
        min_d_idx = np.argmin(d_valid)
        if d_valid[min_d_idx] < 1e-12:
            interpolated[i] = src_val[idx_valid[min_d_idx]]
            continue

        # Calculate weights
        weights = 1.0 / (d_valid**power)
        w_sum = np.sum(weights)

        if w_sum > 0:
            interpolated[i] = np.sum(weights * src_val[idx_valid]) / w_sum
        else:
            interpolated[i] = np.nan

    return interpolated


def idw_to_grid(
    source_coords: np.ndarray,
    source_values: np.ndarray,
    grid_bounds: tuple[float, float, float, float],
    cell_size: float,
    power: float = 2.0,
    max_points: Optional[int] = None,
    search_radius: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolates source points onto a regular 2D grid using Inverse Distance Weighting (IDW).

    Args:
        source_coords: 2D NumPy array of shape (N, 2) containing coordinates of source points.
        source_values: 1D NumPy array of shape (N,) containing values at source points.
        grid_bounds: Tuple of (xmin, ymin, xmax, ymax) specifying the bounding box.
        cell_size: Target grid cell size in map units.
        power: Distance decay exponent.
        max_points: Maximum number of nearest source points to consider. Defaults to 12.
        search_radius: Maximum search radius map units.

    Returns:
        Tuple of:
          - grid: 2D NumPy array of shape (rows, cols) containing interpolated values.
          - x_coords: 1D array of cell center X coordinates (cols,).
          - y_coords: 1D array of cell center Y coordinates (rows,).
    """
    xmin, ymin, xmax, ymax = grid_bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("Invalid grid bounds. Must be (xmin, ymin, xmax, ymax)")
    if cell_size <= 0:
        raise ValueError("cell_size must be greater than 0")

    # Generate cell centers
    x_coords = np.arange(xmin + cell_size / 2.0, xmax, cell_size)
    y_coords = np.arange(ymin + cell_size / 2.0, ymax, cell_size)

    # We want y_coords to go from top to bottom (standard raster alignment)
    y_coords = y_coords[::-1]

    cols = len(x_coords)
    rows = len(y_coords)

    if cols == 0 or rows == 0:
        return np.empty((0, 0)), x_coords, y_coords

    # Create grid coordinates
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_xy = np.column_stack((xx.ravel(), yy.ravel()))

    # Perform points interpolation
    flat_interpolated = idw_to_points(
        source_coords,
        source_values,
        grid_xy,
        power=power,
        max_points=max_points,
        search_radius=search_radius,
    )

    grid = flat_interpolated.reshape((rows, cols))
    return grid, x_coords, y_coords
