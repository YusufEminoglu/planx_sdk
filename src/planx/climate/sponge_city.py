# -*- coding: utf-8 -*-
"""Sponge City & Low Impact Development (LID) Urban Hydrology Simulator for PlanX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LIDFacilityAssessment:
    # 'bioswale', 'permeable_pavement', 'rain_garden', 'green_roof', 'retention_pond'
    facility_type: str
    area_m2: float
    storage_volume_m3: float
    retention_efficiency_percent: float
    infiltrated_volume_m3: float


@dataclass
class SpongeCityResult:
    total_catchment_area_m2: float
    design_rainfall_depth_mm: float
    composite_curve_number_cn: float
    total_rainfall_volume_m3: float
    baseline_runoff_volume_m3: float
    lid_mitigated_runoff_volume_m3: float
    runoff_reduction_percent: float
    peak_discharge_reduction_percent: float
    facilities: list[LIDFacilityAssessment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_catchment_area_m2": round(self.total_catchment_area_m2, 1),
            "design_rainfall_depth_mm": round(self.design_rainfall_depth_mm, 1),
            "composite_cn": round(self.composite_curve_number_cn, 1),
            "baseline_runoff_m3": round(self.baseline_runoff_volume_m3, 2),
            "lid_runoff_m3": round(self.lid_mitigated_runoff_volume_m3, 2),
            "runoff_reduction_percent": round(self.runoff_reduction_percent, 2),
            "peak_reduction_percent": round(self.peak_discharge_reduction_percent, 2),
            "facilities_count": len(self.facilities),
        }


def _scs_runoff_depth(rainfall_mm: float, cn: float) -> float:
    """SCS-CN Runoff Equation: Q = (P - I_a)^2 / (P - I_a + S) where S = 25400/CN - 254."""
    if cn >= 100.0:
        return rainfall_mm
    s = (25400.0 / cn) - 254.0
    ia = 0.2 * s  # Initial abstraction
    if rainfall_mm <= ia:
        return 0.0
    return ((rainfall_mm - ia) ** 2) / (rainfall_mm - ia + s)


def simulate_sponge_city_runoff(
    landuse_areas_m2: dict[str, float],
    lid_facilities: list[dict[str, Any]],
    design_rainfall_depth_mm: float = 65.0,  # e.g. 10-year 24h design storm
) -> SpongeCityResult:
    """Simulate urban runoff retention and Low Impact Development (LID) sponge city performance.

    Args:
        landuse_areas_m2: Dict of landuse types and areas
            (e.g. {'impervious': 40000, 'green': 15000}).
        lid_facilities: List of dicts with 'type', 'area_m2',
            'storage_depth_mm', 'soil_infiltration_rate_mmh'.
        design_rainfall_depth_mm: Total precipitation in mm.
    """
    cn_table = {
        "impervious": 98.0,
        "roof": 98.0,
        "pavement": 98.0,
        "road": 98.0,
        "green": 65.0,
        "park": 60.0,
        "residential": 85.0,
        "commercial": 92.0,
    }

    tot_area = sum(landuse_areas_m2.values())
    if tot_area <= 0:
        tot_area = 10000.0
        landuse_areas_m2 = {"impervious": 6000.0, "green": 4000.0}

    # Area-weighted Curve Number
    weighted_cn_sum = sum(
        area * cn_table.get(k.lower(), 80.0) for k, area in landuse_areas_m2.items()
    )
    composite_cn = weighted_cn_sum / tot_area

    # Baseline Runoff
    p_mm = design_rainfall_depth_mm
    base_q_mm = _scs_runoff_depth(p_mm, composite_cn)
    base_runoff_m3 = (base_q_mm / 1000.0) * tot_area
    total_rain_m3 = (p_mm / 1000.0) * tot_area

    # Evaluate LID Facilities
    assessments: list[LIDFacilityAssessment] = []
    total_lid_retained_m3 = 0.0

    for lid in lid_facilities:
        f_type = str(lid.get("type", "bioswale"))
        f_area = float(lid.get("area_m2", 200.0))
        f_depth_mm = float(lid.get("storage_depth_mm", 150.0))
        f_infil_rate = float(lid.get("soil_infiltration_rate_mmh", 25.0))

        # Storage capacity + infiltration over 24h
        storage_vol_m3 = (f_depth_mm / 1000.0) * f_area
        infil_vol_m3 = (min(p_mm, f_infil_rate * 24.0) / 1000.0) * f_area
        total_capacity = storage_vol_m3 + infil_vol_m3

        retained = min(base_runoff_m3 * 0.40, total_capacity)
        total_lid_retained_m3 += retained

        eff = (retained / max(1e-4, (p_mm / 1000.0) * f_area * 5.0)) * 100.0

        assessments.append(
            LIDFacilityAssessment(
                facility_type=f_type,
                area_m2=f_area,
                storage_volume_m3=storage_vol_m3,
                retention_efficiency_percent=min(100.0, eff),
                infiltrated_volume_m3=infil_vol_m3,
            )
        )

    mitigated_runoff_m3 = max(0.0, base_runoff_m3 - total_lid_retained_m3)
    reduction_pct = ((base_runoff_m3 - mitigated_runoff_m3) / max(1e-4, base_runoff_m3)) * 100.0
    peak_reduction_pct = reduction_pct * 0.85  # Hydrograph peak attenuation factor

    return SpongeCityResult(
        total_catchment_area_m2=tot_area,
        design_rainfall_depth_mm=p_mm,
        composite_curve_number_cn=composite_cn,
        total_rainfall_volume_m3=total_rain_m3,
        baseline_runoff_volume_m3=base_runoff_m3,
        lid_mitigated_runoff_volume_m3=mitigated_runoff_m3,
        runoff_reduction_percent=reduction_pct,
        peak_discharge_reduction_percent=peak_reduction_pct,
        facilities=assessments,
    )
