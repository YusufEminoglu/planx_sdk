# -*- coding: utf-8 -*-
"""Universal Silver Mobility & Elderly Pedestrian Comfort Accessibility Index for PlanX."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class SeniorZoneProfile:
    zone_id: str
    senior_population_65plus: int
    mean_sidewalk_width_m: float
    curb_ramp_compliance_pct: float
    resting_bench_density_per_km: float
    max_walking_slope_pct: float
    pedestrian_crossing_time_seconds: float
    shade_coverage_pct: float


@dataclass
class AgingMobilityResult:
    overall_silver_index: float  # 0 to 100
    zone_grades: dict[str, str]  # 'A' (Excellent), 'B', 'C', 'D', 'F' (Inaccessible)
    mean_bench_spacing_meters: float
    steep_slopes_hazard_count: int
    key_deficiencies: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_silver_index": round(self.overall_silver_index, 1),
            "zone_grades": self.zone_grades,
            "mean_bench_spacing_m": round(self.mean_bench_spacing_meters, 1),
            "steep_slope_hazards": self.steep_slopes_hazard_count,
            "deficiencies_count": len(self.key_deficiencies),
        }


def calculate_silver_mobility_score(
    zones: Sequence[SeniorZoneProfile | dict[str, Any]],
) -> AgingMobilityResult:
    """Evaluate pedestrian urban environment friendliness for aging populations (WHO Age-Friendly Cities guidelines)."""
    scores = []
    grades = {}
    deficiencies = []
    steep_cnt = 0
    bench_densities = []

    for z_raw in zones:
        if isinstance(z_raw, dict):
            z = SeniorZoneProfile(
                zone_id=str(z_raw.get("zone_id", "Z")),
                senior_population_65plus=int(z_raw.get("seniors", 100)),
                mean_sidewalk_width_m=float(z_raw.get("sidewalk_width_m", 1.8)),
                curb_ramp_compliance_pct=float(z_raw.get("curb_ramp_pct", 80.0)),
                resting_bench_density_per_km=float(z_raw.get("benches_per_km", 4.0)),
                max_walking_slope_pct=float(z_raw.get("slope_pct", 4.0)),
                pedestrian_crossing_time_seconds=float(z_raw.get("crossing_time_s", 18.0)),
                shade_coverage_pct=float(z_raw.get("shade_pct", 30.0)),
            )
        else:
            z = z_raw

        bench_densities.append(z.resting_bench_density_per_km)

        # Dimension 1: Sidewalk Width (Optimal >= 2.0m)
        s_width = min(100.0, (z.mean_sidewalk_width_m / 2.0) * 100.0)

        # Dimension 2: Curb Ramps (Wheelchair & Rollator accessible)
        s_ramp = min(100.0, max(0.0, z.curb_ramp_compliance_pct))

        # Dimension 3: Rest & Seating (Optimal >= 8 benches/km ~ every 125m)
        s_bench = min(100.0, (z.resting_bench_density_per_km / 8.0) * 100.0)

        # Dimension 4: Slope Comfort (Penalize > 5% slope heavily)
        if z.max_walking_slope_pct <= 3.0:
            s_slope = 100.0
        elif z.max_walking_slope_pct <= 6.0:
            s_slope = 70.0
        else:
            s_slope = max(10.0, 100.0 - (z.max_walking_slope_pct - 3.0) * 15.0)
            steep_cnt += 1
            deficiencies.append({
                "zone_id": z.zone_id,
                "type": "STEEP_GRADIENT_BARRIER",
                "slope_pct": z.max_walking_slope_pct,
            })

        # Dimension 5: Shade Coverage
        s_shade = min(100.0, (z.shade_coverage_pct / 40.0) * 100.0)

        # Weighted aggregate score
        z_score = (s_width * 0.20 + s_ramp * 0.25 + s_bench * 0.25 + s_slope * 0.20 + s_shade * 0.10)
        scores.append(z_score)

        if z_score >= 85.0:
            grade = "A"
        elif z_score >= 70.0:
            grade = "B"
        elif z_score >= 55.0:
            grade = "C"
        elif z_score >= 40.0:
            grade = "D"
        else:
            grade = "F"

        grades[z.zone_id] = grade

    mean_score = sum(scores) / max(1, len(scores)) if scores else 0.0
    mean_benches = sum(bench_densities) / max(1, len(bench_densities)) if bench_densities else 0.0
    bench_spacing = (1000.0 / max(0.1, mean_benches)) if mean_benches > 0 else 10000.0

    return AgingMobilityResult(
        overall_silver_index=mean_score,
        zone_grades=grades,
        mean_bench_spacing_meters=bench_spacing,
        steep_slopes_hazard_count=steep_cnt,
        key_deficiencies=deficiencies,
    )
