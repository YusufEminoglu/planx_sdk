# -*- coding: utf-8 -*-
"""Social vulnerability, equity, and demographics risk models."""

from __future__ import annotations

from typing import Any

import numpy as np


def social_vulnerability_index(
    indicators: dict[str, np.ndarray],
    weights: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """Calculates the Social Vulnerability Index (SVI) across spatial units.

    Each indicator is normalized using Min-Max normalization to the range [0.0, 100.0],
    and a weighted linear combination is computed.

    Args:
        indicators: Dictionary mapping indicator names (e.g., 'elderly', 'low_income')
            to 1D NumPy arrays containing the raw values for each spatial unit.
            All arrays must have the same length N.
        weights: Dictionary mapping indicator names to their weights.

    Returns:
        Tuple of:
          - scores: 1D NumPy array of shape (N,) containing the SVI score [0, 100].
          - classes: List of strings of length N containing risk category
            ('Low', 'Moderate', 'High', 'Very High').
    """
    keys = list(indicators.keys())
    if not keys:
        raise ValueError("At least one indicator must be provided")

    n = len(indicators[keys[0]])
    for k in keys:
        if len(indicators[k]) != n:
            raise ValueError(f"Indicator '{k}' length must match the others ({n})")

    # Normalize each indicator to [0, 100]
    norm_indicators = {}
    for k in keys:
        vals = np.asarray(indicators[k], dtype=np.float64)
        valid = np.isfinite(vals)
        if not np.any(valid):
            norm_indicators[k] = np.zeros(n)
            continue
        min_v = np.min(vals[valid])
        max_v = np.max(vals[valid])
        if max_v <= min_v:
            norm_indicators[k] = np.zeros(n)
        else:
            norm_indicators[k] = np.clip((vals - min_v) / (max_v - min_v) * 100.0, 0.0, 100.0)

    # Compute weighted average
    score_sum = np.zeros(n, dtype=np.float64)
    weight_sum = 0.0

    for k in keys:
        w = weights.get(k, 1.0)
        if w <= 0:
            continue
        score_sum += norm_indicators[k] * w
        weight_sum += w

    if weight_sum <= 0:
        weight_sum = 1.0

    scores = np.clip(score_sum / weight_sum, 0.0, 100.0)

    # Risk classes
    classes = []
    for val in scores:
        if not np.isfinite(val):
            classes.append("Low")
        elif val >= 75.0:
            classes.append("Very High")
        elif val >= 55.0:
            classes.append("High")
        elif val >= 35.0:
            classes.append("Moderate")
        else:
            classes.append("Low")

    return scores, classes


def urban_energy_vulnerability_index(
    median_income: np.ndarray,
    building_energy_efficiency_score: np.ndarray,
    climate_exposure_score: np.ndarray,
    vulnerable_demographics_ratio: np.ndarray,
) -> dict[str, Any]:
    """Computes composite urban energy vulnerability and fuel poverty risk indices.

    Args:
        median_income: (N,) median household income per zone (> 0).
        building_energy_efficiency_score: (N,) building energy efficiency rating
            [0, 100] (100 = highly efficient).
        climate_exposure_score: (N,) extreme temperature exposure score
            [0, 100] (100 = extreme heat/cold).
        vulnerable_demographics_ratio: (N,) ratio of low-income/elderly households [0, 1].

    Returns:
        Dictionary containing:
            - 'energy_vulnerability_index': (N,) float array in [0, 100]
            - 'income_burden_score': (N,) float array
            - 'inefficiency_burden_score': (N,) float array
            - 'high_risk_zones_count': int
            - 'moderate_risk_zones_count': int
            - 'low_risk_zones_count': int
            - 'overall_energy_poverty_gini': float [0, 1]
    """
    inc = np.asarray(median_income, dtype=np.float64)
    eff = np.asarray(building_energy_efficiency_score, dtype=np.float64)
    exp = np.asarray(climate_exposure_score, dtype=np.float64)
    dem = np.asarray(vulnerable_demographics_ratio, dtype=np.float64)

    if inc.shape != eff.shape or inc.shape != exp.shape or inc.shape != dem.shape:
        raise ValueError("All input arrays must have the same shape")

    if inc.size == 0:
        return {
            "energy_vulnerability_index": np.array([], dtype=np.float64),
            "income_burden_score": np.array([], dtype=np.float64),
            "inefficiency_burden_score": np.array([], dtype=np.float64),
            "high_risk_zones_count": 0,
            "moderate_risk_zones_count": 0,
            "low_risk_zones_count": 0,
            "overall_energy_poverty_gini": 0.0,
        }

    if np.any(inc <= 0):
        raise ValueError("median_income must be strictly positive (> 0)")
    if np.any((eff < 0) | (eff > 100)):
        raise ValueError("building_energy_efficiency_score must be in [0, 100]")
    if np.any((exp < 0) | (exp > 100)):
        raise ValueError("climate_exposure_score must be in [0, 100]")
    if np.any((dem < 0) | (dem > 1)):
        raise ValueError("vulnerable_demographics_ratio must be in [0, 1]")

    max_inc = np.max(inc)
    s_income = 1.0 - (inc / max_inc)
    s_ineff = 1.0 - (eff / 100.0)
    s_exposure = exp / 100.0

    evi = (s_income * 0.35 + s_ineff * 0.30 + s_exposure * 0.20 + dem * 0.15) * 100.0

    high_risk_count = int(np.sum(evi >= 65.0))
    moderate_risk_count = int(np.sum((evi >= 40.0) & (evi < 65.0)))
    low_risk_count = int(np.sum(evi < 40.0))

    evi_sorted = np.sort(evi)
    n = len(evi_sorted)
    sum_evi = np.sum(evi_sorted)

    if n > 0 and sum_evi > 0:
        index = np.arange(1, n + 1)
        gini_raw = (2.0 * np.sum(index * evi_sorted)) / (n * sum_evi) - (n + 1.0) / n
        gini_val = float(np.clip(gini_raw, 0.0, 1.0))
    else:
        gini_val = 0.0

    return {
        "energy_vulnerability_index": evi,
        "income_burden_score": s_income,
        "inefficiency_burden_score": s_ineff,
        "high_risk_zones_count": high_risk_count,
        "moderate_risk_zones_count": moderate_risk_count,
        "low_risk_zones_count": low_risk_count,
        "overall_energy_poverty_gini": gini_val,
    }
