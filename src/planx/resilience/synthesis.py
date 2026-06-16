# -*- coding: utf-8 -*-
"""Climate adaptation and multi-hazard synthesis models."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def _composite_risk_class(score: float) -> str:
    """Classifies a 0-100 score into standard composite risk classes."""
    if not np.isfinite(score):
        return "Low"
    if score >= 75.0:
        return "Very High"
    if score >= 55.0:
        return "High"
    if score >= 35.0:
        return "Moderate"
    return "Low"


def multi_hazard_composite(
    hazards: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[
    np.ndarray,
    Union[List[str], List[List[str]]],
    Union[List[str], List[List[str]]],
    np.ndarray,
    Union[List[List[str]], List[List[List[str]]]],
]:
    """Combines multiple hazard/risk score arrays into a composite index.

    Calculates the weighted composite score, risk categories, dominant hazard
    (highest weighted contributor), Shannon-entropy diversity index, and the
    top contributing hazard drivers for each unit.

    Args:
        hazards: Dictionary mapping hazard names to NumPy arrays of matching shapes.
            Arrays can be 1D or 2D. Values should be in range [0, 100].
        weights: Optional dictionary mapping hazard names to weights.
            Weights will be normalized. If omitted, equal weights are used.

    Returns:
        Tuple of:
          - scores: NumPy array of composite hazard scores [0, 100].
          - risk_classes: List (or list of lists) of risk category strings.
          - dominant_hazards: List (or list of lists) of dominant hazard names.
          - diversity_indices: NumPy array of Shannon entropy diversity indices [0, 100].
          - top_drivers: List of lists (or list of list of lists) of top hazard driver names.
    """
    keys = list(hazards.keys())
    if len(keys) < 2:
        raise ValueError("At least two hazard arrays must be provided")

    shape = hazards[keys[0]].shape
    for k in keys:
        if hazards[k].shape != shape:
            raise ValueError(
                f"Hazard '{k}' shape {hazards[k].shape} does not match '{keys[0]}' shape {shape}"
            )

    if weights is None:
        w_dict = dict.fromkeys(keys, 1.0)
    else:
        w_dict = {k: max(0.0, float(weights.get(k, 1.0))) for k in keys}

    # Flatten arrays for unified 1D computation
    flat_hazards = {k: np.asarray(hazards[k], dtype=np.float64).flatten() for k in keys}
    n_elements = len(flat_hazards[keys[0]])

    scores_flat = np.full(n_elements, np.nan, dtype=np.float64)
    diversity_flat = np.full(n_elements, 0.0, dtype=np.float64)
    classes_flat = []
    dominant_flat = []
    drivers_flat: List[List[str]] = []

    for i in range(n_elements):
        # Gather non-nan hazards
        active = {k: flat_hazards[k][i] for k in keys if np.isfinite(flat_hazards[k][i])}
        if not active:
            scores_flat[i] = np.nan
            classes_flat.append("Low")
            dominant_flat.append("")
            diversity_flat[i] = np.nan
            drivers_flat.append([])
            continue

        weighted_sum = 0.0
        weight_sum = 0.0
        contributions = []

        for k, val in active.items():
            w = w_dict[k]
            weighted_val = val * w
            weighted_sum += weighted_val
            weight_sum += w
            contributions.append((k, weighted_val))

        # 1. Composite Score
        score = weighted_sum / weight_sum if weight_sum > 0.0 else 0.0
        scores_flat[i] = score
        classes_flat.append(_composite_risk_class(score))

        # Sort contributions descending
        contributions.sort(key=lambda x: x[1], reverse=True)

        # 2. Dominant Hazard
        dominant_flat.append(contributions[0][0] if contributions[0][1] > 0 else "")

        # 3. Shannon Entropy / Diversity Index
        positive_contribs = [c[1] for c in contributions if c[1] > 0]
        n_pos = len(positive_contribs)
        total_p = sum(positive_contribs)

        if n_pos > 1 and total_p > 0:
            entropy = 0.0
            for val in positive_contribs:
                p = val / total_p
                entropy -= p * math.log(p)
            diversity_flat[i] = 100.0 * entropy / math.log(n_pos)
        else:
            diversity_flat[i] = 0.0

        # 4. Top Drivers (up to 3 positive drivers)
        drivers_flat.append([c[0] for c in contributions[:3] if c[1] > 0])

    # Reshape results to match input shape
    scores = scores_flat.reshape(shape)
    diversity = diversity_flat.reshape(shape)

    # Reshape list results
    def reshape_list(flat_list):
        if len(shape) == 1:
            return flat_list
        elif len(shape) == 2:
            rows, cols = shape
            return [flat_list[r * cols : (r + 1) * cols] for r in range(rows)]
        return flat_list

    return (
        scores,
        reshape_list(classes_flat),
        reshape_list(dominant_flat),
        diversity,
        reshape_list(drivers_flat),
    )


def equity_adjusted_priority(
    hazard_score: np.ndarray,
    svi_score: np.ndarray,
    equity_weight: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Union[List[str], List[List[str]]]]:
    """Calculates adaptation priority scores adjusted by social vulnerability.

    Applies SVI as an amplification factor to the hazard score, ensuring high-vulnerability,
    high-hazard areas rise to the top of intervention lists. The result is normalized
    back to [0, 100].

    Args:
        hazard_score: NumPy array (1D or 2D) of composite hazard exposure scores [0, 100].
        svi_score: NumPy array of Social Vulnerability Index scores [0, 100].
        equity_weight: Float in range [0.0, 1.0] indicating the weight of the equity boost.
            At 1.0, fully vulnerable units double in priority. At 0.0, SVI is ignored.

    Returns:
        Tuple of:
          - adjusted_score: Normalized equity-adjusted scores in range [0, 100].
          - adjusted_raw: Raw adjusted scores (hazard * factor).
          - factors: The calculated equity amplification factors (1 + weight * SVI / 100).
          - priority_classes: List (or list of lists) of priority category strings.
    """
    h = np.asarray(hazard_score, dtype=np.float64)
    s = np.asarray(svi_score, dtype=np.float64)

    if h.shape != s.shape:
        raise ValueError("hazard_score and svi_score must have the same shape")

    factors = 1.0 + equity_weight * (s / 100.0)
    adjusted_raw = h * factors

    max_possible = 100.0 * (1.0 + equity_weight)
    adjusted_score = np.clip(100.0 * adjusted_raw / max_possible, 0.0, 100.0)

    # Classify priority
    shape = h.shape
    flat_scores = adjusted_score.flatten()
    classes_flat = [_composite_risk_class(val) for val in flat_scores]

    classes: Union[List[str], List[List[str]]]
    if len(shape) == 1:
        classes = classes_flat
    elif len(shape) == 2:
        rows, cols = shape
        classes = [classes_flat[r * cols : (r + 1) * cols] for r in range(rows)]
    else:
        classes = classes_flat

    return adjusted_score, adjusted_raw, factors, classes
