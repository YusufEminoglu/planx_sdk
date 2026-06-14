# -*- coding: utf-8 -*-
"""Wildfire risk and Wildland-Urban Interface (WUI) exposure models."""

from __future__ import annotations

import numpy as np


def _calculate_terrain_factors(dem: np.ndarray, cell_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculates slope and aspect in degrees using Horn's method."""
    padded = np.pad(dem, pad_width=1, mode="edge")

    # 3x3 window components
    z1 = padded[:-2, :-2]
    z2 = padded[:-2, 1:-1]
    z3 = padded[:-2, 2:]
    z4 = padded[1:-1, :-2]
    z6 = padded[1:-1, 2:]
    z7 = padded[2:, :-2]
    z8 = padded[2:, 1:-1]
    z9 = padded[2:, 2:]

    # Gradients
    dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * cell_size)
    dy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0 * cell_size)

    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_radians)

    # Aspect calculation (clockwise from North [0, 360])
    aspect_rad = np.arctan2(dy, -dx)
    aspect_deg = (270.0 + np.degrees(aspect_rad)) % 360.0

    # Set aspect to NaN for flat cells (slope < 1.0 degree)
    aspect_deg[slope_deg < 1.0] = np.nan

    return slope_deg, aspect_deg


def wildfire_risk_index(
    dem: np.ndarray,
    cell_size: float,
    vegetation_density: np.ndarray,
    hemisphere: str = "northern",
    slope_weight: float = 0.35,
    aspect_weight: float = 0.20,
    veg_weight: float = 0.45,
) -> tuple[np.ndarray, list[list[str]]]:
    """Calculates wildfire risk index and risk classes across a grid.

    Combines terrain slope (steeper slope increases spread speed), aspect (exposure to
    solar drying based on hemisphere), and vegetation/fuel density into a combined
    exposure index [0, 100].

    Args:
        dem: 2D NumPy array containing elevation values. NaNs represent no-data.
        cell_size: Size of each grid cell in map units. Must be > 0.
        vegetation_density: 2D NumPy array of same shape as dem containing fuel /
            vegetation cover share [0.0, 1.0].
        hemisphere: 'northern' or 'southern'. Determines aspect risk scoring.
        slope_weight: Weight for the terrain slope factor.
        aspect_weight: Weight for the terrain aspect factor.
        veg_weight: Weight for the vegetation/fuel factor.

    Returns:
        Tuple of:
          - scores: 2D NumPy array containing wildfire risk scores [0, 100].
          - risk_classes: List of lists of strings matching dem.shape containing risk category
            ('Low', 'Moderate', 'High', 'Very High').
    """
    dem_arr = np.asarray(dem, dtype=np.float64)
    shape = dem_arr.shape
    veg_arr = np.asarray(vegetation_density, dtype=np.float64)

    if dem_arr.ndim != 2:
        raise ValueError("DEM must be a 2D array")
    if veg_arr.shape != shape:
        raise ValueError("vegetation_density shape must match dem shape")
    if cell_size <= 0:
        raise ValueError("cell_size must be greater than 0")

    valid = np.isfinite(dem_arr)
    if not np.any(valid):
        return np.zeros_like(dem_arr), [["Low" for _ in range(shape[1])] for _ in range(shape[0])]

    # Calculate terrain factors
    slope_deg, aspect_deg = _calculate_terrain_factors(dem_arr, cell_size)

    # 1. Slope Score: linearly scale [0, 30] degrees to [0, 100]
    slope_score = np.clip(slope_deg / 30.0 * 100.0, 0.0, 100.0)

    # 2. Aspect Score: solar exposure dryness factor
    if hemisphere.lower() == "northern":
        # South-facing (180 deg) is max risk (100), North-facing (0/360) is min risk (0)
        aspect_score = (1.0 - np.cos(np.radians(aspect_deg))) * 50.0
    else:
        # Southern hemisphere: North-facing is max risk (100), South is min (0)
        aspect_score = (1.0 + np.cos(np.radians(aspect_deg))) * 50.0

    # Flat areas with undefined aspect get a neutral/low aspect score
    aspect_score[np.isnan(aspect_score)] = 0.0

    # 3. Vegetation Score: scale [0.0, 1.0] to [0, 100]
    veg_score = np.clip(veg_arr * 100.0, 0.0, 100.0)

    # Weighted Linear Combination
    weight_sum = slope_weight + aspect_weight + veg_weight
    if weight_sum <= 0.0:
        weight_sum = 1.0

    scores = (
        slope_score * slope_weight + aspect_score * aspect_weight + veg_score * veg_weight
    ) / weight_sum

    # Preserve NaNs from the original DEM
    scores[~valid] = np.nan

    # Classify scores
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
        row_classes = list(row_classes)
        risk_classes.append(row_classes)

    return scores, risk_classes
