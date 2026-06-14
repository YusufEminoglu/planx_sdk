# -*- coding: utf-8 -*-
"""Social vulnerability, equity, and demographics risk models."""

from __future__ import annotations

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
