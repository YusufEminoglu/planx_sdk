# -*- coding: utf-8 -*-
"""Landslide susceptibility and slope-stability screening models."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _calculate_slope_horn(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Calculates slope in degrees using Horn's method (8-neighbor estimator)."""
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

    # Horn's formulas for gradients
    dx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * cell_size)
    dy = ((z7 + 2 * z8 + z9) - (z1 + 2 * z2 + z3)) / (8.0 * cell_size)

    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_radians)


def landslide_susceptibility(
    dem: np.ndarray,
    cell_size: float,
    soil_susceptibility: Optional[np.ndarray] = None,
    lulc_susceptibility: Optional[np.ndarray] = None,
    slope_weight: float = 0.50,
    soil_weight: float = 0.25,
    lulc_weight: float = 0.25,
) -> tuple[np.ndarray, list[list[str]]]:
    """Calculates landslide susceptibility scores and risk classes across a DEM grid.

    Combines terrain slope (calculated using Horn's 8-neighbor method), soil stability,
    and land use/land cover (LULC) susceptibility into a combined index [0, 100].

    Args:
        dem: 2D NumPy array containing elevation values. NaNs represent no-data.
        cell_size: Size of each grid cell in map units (must be > 0).
        soil_susceptibility: Optional 2D NumPy array of the same shape as dem containing
            soil-related susceptibility scores [0, 100] (e.g. clay is high, rock is low).
        lulc_susceptibility: Optional 2D NumPy array of the same shape as dem containing
            land cover susceptibility scores [0, 100] (e.g. bare soil is high, forest is low).
        slope_weight: Weight for the terrain slope factor.
        soil_weight: Weight for the soil susceptibility factor.
        lulc_weight: Weight for the LULC susceptibility factor.

    Returns:
        Tuple of:
          - scores: 2D NumPy array containing the landslide susceptibility score [0, 100].
          - risk_classes: List of lists of strings matching dem.shape containing risk category
            ('Low', 'Moderate', 'High', 'Very High').
    """
    dem_arr = np.asarray(dem, dtype=np.float64)
    shape = dem_arr.shape

    if dem_arr.ndim != 2:
        raise ValueError("DEM must be a 2D array")
    if cell_size <= 0:
        raise ValueError("cell_size must be greater than 0")

    valid = np.isfinite(dem_arr)
    if not np.any(valid):
        return np.zeros_like(dem_arr), [["Low" for _ in range(shape[1])] for _ in range(shape[0])]

    # Calculate terrain slope in degrees
    slope_deg = _calculate_slope_horn(dem_arr, cell_size)

    # Normalize slope to a score [0, 100]
    # Slopes below 5 degrees have 0 susceptibility; slopes above 35 degrees have 100.
    slope_score = np.clip((slope_deg - 5.0) / (35.0 - 5.0) * 100.0, 0.0, 100.0)

    # Handle soil susceptibility
    if soil_susceptibility is not None:
        soil_arr = np.asarray(soil_susceptibility, dtype=np.float64)
        if soil_arr.shape != shape:
            raise ValueError("soil_susceptibility shape must match dem shape")
        soil_score = np.clip(soil_arr, 0.0, 100.0)
    else:
        soil_score = np.full(shape, 50.0, dtype=np.float64)
        soil_weight = 0.0

    # Handle LULC susceptibility
    if lulc_susceptibility is not None:
        lulc_arr = np.asarray(lulc_susceptibility, dtype=np.float64)
        if lulc_arr.shape != shape:
            raise ValueError("lulc_susceptibility shape must match dem shape")
        lulc_score = np.clip(lulc_arr, 0.0, 100.0)
    else:
        lulc_score = np.full(shape, 50.0, dtype=np.float64)
        lulc_weight = 0.0

    # Weighted linear combination
    weight_sum = slope_weight + soil_weight + lulc_weight
    if weight_sum <= 0.0:
        weight_sum = 1.0

    scores = (
        slope_score * slope_weight + soil_score * soil_weight + lulc_score * lulc_weight
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
        risk_classes.append(row_classes)

    return scores, risk_classes
