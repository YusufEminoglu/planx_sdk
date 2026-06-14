# -*- coding: utf-8 -*-
"""Urban heat comfort and microclimate vulnerability models."""

from __future__ import annotations

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
