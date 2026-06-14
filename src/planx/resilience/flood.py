# -*- coding: utf-8 -*-
"""Flood risk and pluvial vulnerability models."""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.ndimage


def pluvial_flood_susceptibility(
    dem: np.ndarray,
    cell_size: float,
    neighborhood_radius: float = 150.0,
    drainage_dists: Optional[np.ndarray] = None,
    elevation_weight: float = 0.45,
    slope_weight: float = 0.30,
    drainage_weight: float = 0.25,
) -> tuple[np.ndarray, list[list[str]]]:
    """Calculates pluvial flood susceptibility scores across a DEM grid.

    This is a screening model that identifies low-lying, flat, and drainage-proximate
    areas using NumPy and SciPy image processing filters.

    Args:
        dem: 2D NumPy array containing elevation values. NaNs represent no-data.
        cell_size: Size of each grid cell in map units.
        neighborhood_radius: Radius in map units to calculate local relief.
        drainage_dists: Optional 2D NumPy array of the same shape as dem containing
            Euclidean distance to the closest drainage network.
        elevation_weight: Weight for relative low-elevation (0-1).
        slope_weight: Weight for flat / low-slope areas (0-1).
        drainage_weight: Weight for drainage proximity (0-1).

    Returns:
        Tuple of:
          - scores: 2D NumPy array containing the flood susceptibility score [0, 100].
          - risk_classes: List of lists of strings matching dem.shape containing risk category
            ('Low', 'Moderate', 'High', 'Very High').
    """
    dem_arr = np.asarray(dem, dtype=np.float64)
    shape = dem_arr.shape

    if dem_arr.ndim != 2:
        raise ValueError("DEM must be a 2D array")

    # Mask of valid pixels
    valid = np.isfinite(dem_arr)
    if not np.any(valid):
        return np.zeros_like(dem_arr), [["Low" for _ in range(shape[1])] for _ in range(shape[0])]

    # 1. Relative low elevation
    min_elev = np.min(dem_arr[valid])
    max_elev = np.max(dem_arr[valid])
    elev_range = max_elev - min_elev
    if elev_range <= 0:
        elev_range = 1.0

    rel_low = np.clip((1.0 - (dem_arr - min_elev) / elev_range) * 100.0, 0.0, 100.0)

    # 2. Slope proxy (local relief max - min in neighborhood)
    radius_pixels = max(1, int(round(neighborhood_radius / cell_size)))
    filter_size = 2 * radius_pixels + 1

    # To handle NaNs correctly in filters, we can temporarily fill them with inf/neg-inf
    dem_temp_max = dem_arr.copy()
    dem_temp_max[~valid] = -np.inf
    dem_temp_min = dem_arr.copy()
    dem_temp_min[~valid] = np.inf

    max_dem = scipy.ndimage.maximum_filter(dem_temp_max, size=filter_size)
    min_dem = scipy.ndimage.minimum_filter(dem_temp_min, size=filter_size)

    # Calculate relief
    relief = max_dem - min_dem
    relief[~np.isfinite(relief)] = 0.0

    # Flatness/slope proxy: 100 - (relief / radius) * 100
    slope_proxy = np.clip(100.0 - (relief / neighborhood_radius) * 100.0, 0.0, 100.0)

    # 3. Proximity to drainage
    if drainage_dists is not None:
        d_dists = np.asarray(drainage_dists, dtype=np.float64)
        if d_dists.shape != shape:
            raise ValueError("drainage_dists shape must match dem shape")
        drain_score = np.clip(100.0 - (d_dists / (4.0 * neighborhood_radius)) * 100.0, 0.0, 100.0)
    else:
        drain_score = np.full(shape, 50.0, dtype=np.float64)
        drainage_weight = 0.0  # Exclude from weighting if not provided

    # 4. Weighted linear combination
    weight_sum = elevation_weight + slope_weight + drainage_weight
    if weight_sum <= 0:
        weight_sum = 1.0

    scores = (
        rel_low * elevation_weight + slope_proxy * slope_weight + drain_score * drainage_weight
    ) / weight_sum

    scores = np.clip(scores, 0.0, 100.0)
    scores[~valid] = np.nan

    # 5. Risk classification
    risk_classes = []
    for r in range(shape[0]):
        row_classes = []
        for c in range(shape[1]):
            val = scores[r, c]
            if not np.isfinite(val):
                row_classes.append("Low")
            elif val >= 75.0:
                row_classes.append("Very High")
            elif val >= 55.0:
                row_classes.append("High")
            elif val >= 35.0:
                row_classes.append("Moderate")
            else:
                row_classes.append("Low")
        risk_classes.append(row_classes)

    return scores, risk_classes


def coastal_flood_inundation(
    dem: np.ndarray,
    water_level: float,
    sea_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates coastal flood inundation using a hydrologically connected bathtub model.

    Identifies cells that are flooded by a given water level rise. A cell is flooded if
    its elevation is less than or equal to the water level AND it is connected to a
    sea/water source cell via other flooded cells (8-connectivity).

    Args:
        dem: 2D NumPy array containing elevation values. NaNs represent no-data.
        water_level: Target water level/elevation for flooding (e.g. 2.0 meters).
        sea_mask: Optional 2D boolean array of the same shape as dem marking sea/water
            source cells. If omitted, all boundary cells with elevation <= 0 are used.

    Returns:
        Tuple of:
          - flooded: 2D boolean NumPy array where True indicates flooded cells.
          - water_depth: 2D NumPy array containing water depth (water_level - dem)
            for flooded cells, and 0.0 elsewhere (NaNs preserved).
    """
    dem_arr = np.asarray(dem, dtype=np.float64)
    shape = dem_arr.shape

    if dem_arr.ndim != 2:
        raise ValueError("DEM must be a 2D array")

    valid = np.isfinite(dem_arr)

    # 1. Identify potentially flooded cells (below or equal to water level)
    potential = (dem_arr <= water_level) & valid

    # 2. Determine sea mask/water source seeds
    if sea_mask is not None:
        seeds = np.asarray(sea_mask, dtype=bool)
        if seeds.shape != shape:
            raise ValueError("sea_mask shape must match dem shape")
    else:
        # Create default sea mask: boundary cells with elevation <= 0
        seeds = np.zeros(shape, dtype=bool)
        # Check boundary rows and columns
        if shape[0] > 0 and shape[1] > 0:
            seeds[0, :] = dem_arr[0, :] <= 0.0
            seeds[-1, :] = dem_arr[-1, :] <= 0.0
            seeds[:, 0] = dem_arr[:, 0] <= 0.0
            seeds[:, -1] = dem_arr[:, -1] <= 0.0
        seeds &= valid

    # Only keep seeds that are actually below the water level
    seeds = seeds & potential

    if not np.any(seeds):
        # No water source is flooded/active
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.float64)

    # 3. Connectivity analysis using 8-connectivity
    structure = np.ones((3, 3), dtype=bool)
    labeled_array, num_features = scipy.ndimage.label(potential, structure=structure)

    # Extract labels at seed locations
    seed_labels = labeled_array[seeds]
    unique_seed_labels = np.unique(seed_labels)
    unique_seed_labels = unique_seed_labels[unique_seed_labels > 0]

    # Mask flooded cells
    flooded = np.isin(labeled_array, unique_seed_labels)

    # 4. Calculate water depth: water_level - dem
    water_depth = np.zeros(shape, dtype=np.float64)
    water_depth[flooded] = water_level - dem_arr[flooded]
    water_depth[~valid] = np.nan

    return flooded, water_depth
