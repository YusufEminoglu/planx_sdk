# -*- coding: utf-8 -*-
"""15-Minute City Multi-Dimensional Proximity & Equity Evaluation Engine for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class DimensionScore:
    """Score and nearest travel time for a single 15-minute city dimension."""

    dimension_name: str
    nearest_travel_time_min: float
    score: float  # 0.0 to 100.0
    matched_amenity_count: int


@dataclass
class ZoneAccessibilityScore:
    """Complete 15-minute score card for a single origin zone or parcel."""

    zone_id: str
    composite_15m_score: float  # 0.0 to 100.0
    dimension_scores: dict[str, DimensionScore]
    is_15m_compliant: bool  # True if composite_15m_score >= 60.0


@dataclass
class FifteenMinuteCityResult:
    """City-wide 15-minute evaluation result with equity indicators."""

    total_zones: int
    mean_composite_score: float
    median_composite_score: float
    compliance_rate: float  # Proportion of zones with score >= 60
    equity_gini: float  # 0.0 (perfect equity) to 1.0 (extreme inequality)
    hoover_index: float  # Proportion of accessibility that would need redistribution
    dimension_averages: dict[str, float]
    zone_scores: list[ZoneAccessibilityScore]
    deficit_zone_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_zones": self.total_zones,
            "mean_composite_score": round(self.mean_composite_score, 2),
            "median_composite_score": round(self.median_composite_score, 2),
            "compliance_rate": round(self.compliance_rate, 4),
            "equity_gini": round(self.equity_gini, 4),
            "hoover_index": round(self.hoover_index, 4),
            "dimension_averages": {k: round(v, 2) for k, v in self.dimension_averages.items()},
            "deficit_zones_count": len(self.deficit_zone_ids),
        }


DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "supplying": 0.25,  # Grocery, pharmacies, daily retail
    "caring": 0.20,     # Clinics, medical centers, pharmacy
    "learning": 0.20,   # Schools, libraries, nurseries
    "enjoying": 0.15,   # Parks, sports, green spaces
    "working": 0.10,    # Local offices, co-working
    "living": 0.10,     # Community spaces, public squares
}


def _decay_function(travel_time_min: float, threshold_min: float = 15.0) -> float:
    """Continuous logistic / Gaussian decay from 100 (at 0 min) to
    50 (at 15 min) and ~0 (at 30 min).
    """
    if travel_time_min <= 0:
        return 100.0
    # Sigmoidal smooth decay
    k = 0.20
    val = 100.0 / (1.0 + math.exp(k * (travel_time_min - threshold_min)))
    return max(0.0, min(100.0, val))


def evaluate_15min_city(
    origin_zones: list[dict[str, Any]],
    amenities: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    walking_speed_kmh: float = 4.8,
) -> FifteenMinuteCityResult:
    """Evaluate 15-minute city accessibility and equity metrics across zones.

    Args:
        origin_zones: List of dicts with 'id', 'x', 'y', optional 'population'.
        amenities: List of dicts with 'id', 'x', 'y', 'category'
            ('supplying', 'caring', 'learning', etc.).
        weights: Optional dimension weight dictionary summing to 1.0.
        walking_speed_kmh: Pedestrian walking speed (default: 4.8 km/h).

    Returns:
        FifteenMinuteCityResult with composite scores, dimension breakdowns, and equity indices.
    """
    dim_weights = weights or DEFAULT_DIMENSION_WEIGHTS
    # Normalize weights
    total_w = sum(dim_weights.values())
    norm_weights = {k: v / total_w for k, v in dim_weights.items()}

    # Group amenities by dimension
    amenities_by_dim: dict[str, list[tuple[float, float]]] = {d: [] for d in norm_weights}
    for am in amenities:
        cat = str(am.get("category", "")).lower()
        # Find matching dimension
        for dim in norm_weights:
            if dim in cat or cat in dim:
                amenities_by_dim[dim].append((float(am.get("x", 0.0)), float(am.get("y", 0.0))))
                break

    speed_m_per_min = (walking_speed_kmh * 1000.0) / 60.0
    zone_results: list[ZoneAccessibilityScore] = []
    composite_scores: list[float] = []
    dim_accum: dict[str, list[float]] = {d: [] for d in norm_weights}
    deficit_ids: list[str] = []

    for z in origin_zones:
        z_id = str(z.get("id", "z"))
        zx = float(z.get("x", 0.0))
        zy = float(z.get("y", 0.0))

        dim_scores: dict[str, DimensionScore] = {}
        weighted_comp = 0.0

        for dim, w in norm_weights.items():
            am_pts = amenities_by_dim[dim]
            if not am_pts:
                # No amenities in this category anywhere in the city
                min_time = 999.0
                score = 0.0
            else:
                # Find nearest amenity Euclidean distance (with Manhattan/circuity factor 1.25)
                min_dist = min(math.hypot(zx - ax, zy - ay) * 1.25 for ax, ay in am_pts)
                min_time = min_dist / max(1.0, speed_m_per_min)
                score = _decay_function(min_time, threshold_min=15.0)

            dim_scores[dim] = DimensionScore(
                dimension_name=dim,
                nearest_travel_time_min=round(min_time, 2),
                score=round(score, 2),
                matched_amenity_count=len(am_pts),
            )
            weighted_comp += score * w
            dim_accum[dim].append(score)

        comp_score = round(weighted_comp, 2)
        composite_scores.append(comp_score)
        is_compliant = comp_score >= 60.0
        if not is_compliant:
            deficit_ids.append(z_id)

        zone_results.append(
            ZoneAccessibilityScore(
                zone_id=z_id,
                composite_15m_score=comp_score,
                dimension_scores=dim_scores,
                is_15m_compliant=is_compliant,
            )
        )

    n_zones = max(1, len(origin_zones))
    mean_score = sum(composite_scores) / n_zones
    sorted_scores = sorted(composite_scores)
    median_score = sorted_scores[n_zones // 2]
    compliance_rate = sum(1 for s in composite_scores if s >= 60.0) / n_zones

    # Gini Coefficient calculation
    if sum(composite_scores) > 0:
        cum_scores = [0.0]
        for s in sorted_scores:
            cum_scores.append(cum_scores[-1] + s)
        total_sum = cum_scores[-1]
        b_area = sum(cum_scores[:-1]) / (total_sum * n_zones)
        gini = max(0.0, min(1.0, 1.0 - 2.0 * b_area))
    else:
        gini = 0.0

    # Hoover Redistribution Index
    if mean_score > 0:
        hoover = 0.5 * sum(abs(s - mean_score) for s in composite_scores) / (n_zones * mean_score)
    else:
        hoover = 0.0

    dim_averages = {
        d: (sum(scores) / n_zones if scores else 0.0) for d, scores in dim_accum.items()
    }

    return FifteenMinuteCityResult(
        total_zones=len(origin_zones),
        mean_composite_score=round(mean_score, 2),
        median_composite_score=round(median_score, 2),
        compliance_rate=round(compliance_rate, 4),
        equity_gini=round(gini, 4),
        hoover_index=round(hoover, 4),
        dimension_averages=dim_averages,
        zone_scores=zone_results,
        deficit_zone_ids=deficit_ids,
    )
