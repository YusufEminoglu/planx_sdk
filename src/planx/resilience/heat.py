# -*- coding: utf-8 -*-
"""Urban heat comfort and microclimate vulnerability models."""

from __future__ import annotations

from typing import Optional

import numpy as np


def urban_heat_comfort_risk(
    impervious_share: np.ndarray,
    building_share: np.ndarray,
    green_share: np.ndarray,
    cooling_dists: np.ndarray,
    vuln_counts: np.ndarray,
    cooling_distance: float = 400.0,
    w_imperv: float = 0.30,
    w_green: float = 0.30,
    w_build: float = 0.25,
    w_vuln: float = 0.15,
) -> tuple[np.ndarray, list[list[str]]]:
    """Calculates urban heat comfort risk scores across a grid.

    Combines hardscape share, building density, green space deficit, cooling distance,
    and vulnerable asset counts into a normalized 0-100 score.

    Args:
        impervious_share: 2D NumPy array of shape (R, C) containing hardscape area share [0, 1].
        building_share: 2D NumPy array of shape (R, C) containing building footprint share [0, 1].
        green_share: 2D NumPy array of shape (R, C) containing green area share [0, 1].
        cooling_dists: 2D NumPy array of shape (R, C) containing distance to nearest green/water.
        vuln_counts: 2D NumPy array of shape (R, C) containing number of vulnerable assets in cell.
        cooling_distance: Maximum walking distance to cooling area (threshold for normalization).
        w_imperv: Weight for impervious surfaces.
        w_green: Weight for green deficit & cooling distance.
        w_build: Weight for building density.
        w_vuln: Weight for vulnerable asset count.

    Returns:
        Tuple of:
          - scores: 2D NumPy array of shape (R, C) containing heat risk scores [0, 100].
          - risk_classes: List of lists of risk category strings matching the shape
            ('Low', 'Moderate', 'High', 'Very High').
    """
    imp = np.asarray(impervious_share, dtype=np.float64)
    bld = np.asarray(building_share, dtype=np.float64)
    grn = np.asarray(green_share, dtype=np.float64)
    dst = np.asarray(cooling_dists, dtype=np.float64)
    vuln = np.asarray(vuln_counts, dtype=np.float64)

    shape = imp.shape
    if bld.shape != shape or grn.shape != shape or dst.shape != shape or vuln.shape != shape:
        raise ValueError("All input arrays must have the same shape")

    # 1. Impervious score [0, 100]
    imp_score = imp * 100.0

    # 2. Building score [0, 100]
    bld_score = bld * 100.0

    # 3. Green deficit [0, 100]
    green_deficit = (1.0 - grn) * 100.0

    # 4. Cooling distance score [0, 100]
    cooling_score = np.clip((dst / cooling_distance) * 100.0, 0.0, 100.0)

    # Combined green deficit & cooling distance score
    green_score = (green_deficit + cooling_score) * 0.5

    # 5. Vulnerability score
    vuln_score = np.clip(vuln * 20.0, 0.0, 100.0)

    # Weighted sum
    weight_sum = w_imperv + w_green + w_build + w_vuln
    if weight_sum <= 0:
        weight_sum = 1.0

    scores = (
        imp_score * w_imperv + bld_score * w_build + green_score * w_green + vuln_score * w_vuln
    ) / weight_sum

    scores = np.clip(scores, 0.0, 100.0)

    # Risk classes
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


def urban_heat_island_intensity(
    albedo: np.ndarray,
    ndvi: np.ndarray,
    building_height: np.ndarray,
    building_footprint: np.ndarray,
    wind_speed: Optional[np.ndarray] = None,
    base_intensity: float = 2.0,
) -> np.ndarray:
    """Estimates Urban Heat Island (UHI) intensity (in degrees Celsius) across a grid.

    Calculates UHI intensity using a proxy model based on surface albedo,
    Normalized Difference Vegetation Index (NDVI) proxy, building heights,
    building footprints, and wind speed.

    Args:
        albedo: 2D NumPy array of shape (R, C) containing surface albedo values [0.0, 1.0].
        ndvi: 2D NumPy array of shape (R, C) containing NDVI values [-1.0, 1.0].
        building_height: 2D NumPy array of shape (R, C) containing average building height (meters).
        building_footprint: 2D NumPy array of shape (R, C) containing building
            footprint share [0.0, 1.0].
        wind_speed: Optional 2D NumPy array of shape (R, C) containing wind speed (m/s).
            If omitted, a constant wind speed of 1.5 m/s is assumed.
        base_intensity: Baseline temperature offset in degrees Celsius.

    Returns:
        2D NumPy array of shape (R, C) containing estimated UHI intensity
        offset (in degrees Celsius).
    """
    alb = np.asarray(albedo, dtype=np.float64)
    veg = np.asarray(ndvi, dtype=np.float64)
    bh = np.asarray(building_height, dtype=np.float64)
    bf = np.asarray(building_footprint, dtype=np.float64)

    shape = alb.shape
    if veg.shape != shape or bh.shape != shape or bf.shape != shape:
        raise ValueError("All input arrays must have the same shape")

    if wind_speed is None:
        wind = np.full(shape, 1.5, dtype=np.float64)
    else:
        wind = np.asarray(wind_speed, dtype=np.float64)
        if wind.shape != shape:
            raise ValueError("wind_speed array must have the same shape as other inputs")

    # 1. Albedo factor
    alb_contrib = 3.0 * (1.0 - alb)

    # 2. Vegetation cooling factor
    veg_index = np.clip((veg + 1.0) / 2.0, 0.0, 1.0)
    veg_contrib = 4.0 * (1.0 - veg_index) - 2.0 * veg_index

    # 3. Urban canyon / building volume factor
    vol_index = bh * bf
    vol_contrib = 4.0 * np.clip(vol_index / 30.0, 0.0, 1.0)

    # 4. Wind mitigation factor
    wind_cooling = 1.0 * np.log1p(np.clip(wind, 0.0, None))
    wind_cooling = np.clip(wind_cooling, 0.0, 2.0)

    # Calculate final UHI intensity offset (in C)
    uhi_intensity = base_intensity + alb_contrib + veg_contrib + vol_contrib - wind_cooling

    # UHI cannot be negative
    uhi_intensity = np.maximum(uhi_intensity, 0.0)

    return uhi_intensity
