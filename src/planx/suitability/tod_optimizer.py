# -*- coding: utf-8 -*-
"""Transit-Oriented Development (TOD) Typology & Density Optimizer for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class TODStationScore:
    station_id: str
    tod_typology: str  # 'Regional Core', 'Urban Center', 'Neighborhood Station', 'Suburban Node'
    transit_frequency_score: float  # 0 to 100
    landuse_diversity_entropy: float  # Shannon entropy 0 to 1
    walkable_catchment_score: float  # 0 to 100
    composite_tod_index: float  # 0 to 100
    recommended_far_density: float  # Floor Area Ratio (e.g. 2.5 - 6.0)


@dataclass
class TODTypologyResult:
    total_stations_evaluated: int
    mean_tod_index: float
    high_potential_tod_count: int
    station_scores: list[TODStationScore]

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-ready dictionary."""
        return {
            "total_stations": self.total_stations_evaluated,
            "mean_tod_index": round(self.mean_tod_index, 2),
            "high_potential_count": self.high_potential_tod_count,
            "stations": [
                {
                    "id": s.station_id,
                    "typology": s.tod_typology,
                    "tod_index": round(s.composite_tod_index, 1),
                    "recommended_far": round(s.recommended_far_density, 2),
                }
                for s in self.station_scores
            ],
        }


def evaluate_tod_potential(
    transit_stations: list[dict[str, Any]],
) -> TODTypologyResult:
    """Evaluate TOD suitability typology and optimized FAR density envelopes for transit nodes."""
    scores: list[TODStationScore] = []
    tot_idx = 0.0
    high_cnt = 0

    for st in transit_stations:
        s_id = str(st.get("id", "station"))
        trips_per_hour = float(st.get("peak_trips_per_hour", 12.0))
        landuse_mix = st.get(
            "landuse_shares", {"residential": 0.5, "commercial": 0.3, "civic": 0.2}
        )
        ped_density = float(st.get("intersection_density_km2", 80.0))

        # Transit Score (normalized 0-100, 30 trips/hr = 100)
        t_score = min(100.0, (trips_per_hour / 30.0) * 100.0)

        # Land-use Shannon Entropy H = - \sum p_i ln(p_i) / ln(K)
        k = len(landuse_mix)
        if k > 1:
            entropy_sum = sum(p * math.log(max(1e-6, p)) for p in landuse_mix.values() if p > 0)
            h = -entropy_sum / math.log(k)
        else:
            h = 0.0
        entropy_score = max(0.0, min(1.0, h))

        # Walkability score (120 intersections/km2 = 100)
        w_score = min(100.0, (ped_density / 120.0) * 100.0)

        # Composite TOD index
        tod_idx = 0.40 * t_score + 0.35 * (entropy_score * 100.0) + 0.25 * w_score
        tot_idx += tod_idx

        # Typology classification
        if tod_idx >= 75.0:
            typology = "Regional Core"
            rec_far = 4.5 + (tod_idx - 75.0) * 0.06
            high_cnt += 1
        elif tod_idx >= 55.0:
            typology = "Urban Center"
            rec_far = 3.0 + (tod_idx - 55.0) * 0.05
            high_cnt += 1
        elif tod_idx >= 35.0:
            typology = "Neighborhood Station"
            rec_far = 2.0 + (tod_idx - 35.0) * 0.03
        else:
            typology = "Suburban Node"
            rec_far = 1.2

        scores.append(
            TODStationScore(
                station_id=s_id,
                tod_typology=typology,
                transit_frequency_score=t_score,
                landuse_diversity_entropy=entropy_score,
                walkable_catchment_score=w_score,
                composite_tod_index=tod_idx,
                recommended_far_density=rec_far,
            )
        )

    mean_idx = tot_idx / max(1, len(scores))

    return TODTypologyResult(
        total_stations_evaluated=len(scores),
        mean_tod_index=mean_idx,
        high_potential_tod_count=high_cnt,
        station_scores=scores,
    )
