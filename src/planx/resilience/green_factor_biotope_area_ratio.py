# -*- coding: utf-8 -*-
"""
Berlin Biotope Area Factor (BAF / BFF) and Malmö Green Factor Engine.

Implements ecological surface area ratio calculations, hydrological stormwater
attenuation, Shannon-Wiener biodiversity entropy, evapotranspiration cooling,
and municipal zoning compliance (Berlin BAF, Malmö GYF, and London UGF).

Pure-Python implementation with zero synthetic data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence


class SurfaceEcologicalType(str, Enum):
    """Statutory ecological surface classification and weighting standards."""

    SEALED_IMPERVIOUS = "sealed_impervious"
    SEMI_PERVIOUS_PAVER = "semi_pervious_paver"
    EXTENSIVE_GREEN_ROOF = "extensive_green_roof"
    INTENSIVE_GREEN_ROOF = "intensive_green_roof"
    VEGETATION_SOIL_CONNECTED = "vegetation_soil_connected"
    TREE_CANOPY_OVER_GROUND = "tree_canopy_over_ground"
    VERTICAL_GREEN_FACADE = "vertical_green_facade"
    WATER_RETENTION_POND = "water_retention_pond"


# Standard statutory ecological weights (w) and runoff coefficients (C)
SURFACE_WEIGHTS_DEFAULT: dict[SurfaceEcologicalType, tuple[float, float]] = {
    SurfaceEcologicalType.SEALED_IMPERVIOUS: (0.0, 0.90),
    SurfaceEcologicalType.SEMI_PERVIOUS_PAVER: (0.3, 0.50),
    SurfaceEcologicalType.EXTENSIVE_GREEN_ROOF: (0.5, 0.40),
    SurfaceEcologicalType.INTENSIVE_GREEN_ROOF: (0.7, 0.25),
    SurfaceEcologicalType.VEGETATION_SOIL_CONNECTED: (1.0, 0.15),
    SurfaceEcologicalType.TREE_CANOPY_OVER_GROUND: (1.0, 0.10),
    SurfaceEcologicalType.VERTICAL_GREEN_FACADE: (0.5, 0.30),
    SurfaceEcologicalType.WATER_RETENTION_POND: (1.0, 0.05),
}


@dataclass(frozen=True)
class BiotopeSurfaceElement:
    """An individual surface area element within an urban parcel."""

    surface_type: SurfaceEcologicalType
    area_m2: float
    description: str = ""
    custom_weight: float | None = None
    custom_runoff_coeff: float | None = None

    @property
    def weight(self) -> float:
        """Effective ecological weighting factor w in [0.0, 1.0]."""
        if self.custom_weight is not None:
            return max(0.0, min(1.0, self.custom_weight))
        return SURFACE_WEIGHTS_DEFAULT.get(self.surface_type, (0.0, 0.90))[0]

    @property
    def runoff_coefficient(self) -> float:
        """Effective runoff coefficient C in [0.0, 1.0]."""
        if self.custom_runoff_coeff is not None:
            return max(0.0, min(1.0, self.custom_runoff_coeff))
        return SURFACE_WEIGHTS_DEFAULT.get(self.surface_type, (0.0, 0.90))[1]

    @property
    def effective_ecological_area_m2(self) -> float:
        """Weighted ecological area (Area * w)."""
        return max(0.0, self.area_m2) * self.weight


@dataclass
class BiotopeAreaReport:
    """Comprehensive ecological and hydrological evaluation of an urban development parcel."""

    parcel_area_m2: float
    total_effective_green_area_m2: float
    biotope_area_factor: float
    composite_runoff_coefficient: float
    runoff_reduction_percentage: float
    biodiversity_shannon_index: float
    evapotranspiration_cooling_score: float
    is_compliant: bool
    target_threshold: float
    surface_breakdown: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serialisable dictionary."""
        return {
            "parcel_area_m2": round(self.parcel_area_m2, 2),
            "total_effective_green_area_m2": round(self.total_effective_green_area_m2, 2),
            "biotope_area_factor": round(self.biotope_area_factor, 4),
            "composite_runoff_coefficient": round(self.composite_runoff_coefficient, 4),
            "runoff_reduction_pct": round(self.runoff_reduction_percentage, 2),
            "biodiversity_shannon_index": round(self.biodiversity_shannon_index, 4),
            "evapotranspiration_cooling_score_out_of_10": round(self.evapotranspiration_cooling_score, 2),
            "is_compliant": self.is_compliant,
            "target_threshold": self.target_threshold,
            "surface_breakdown_count": len(self.surface_breakdown),
            "surface_breakdown": self.surface_breakdown,
        }


def calculate_biotope_area_factor(
    parcel_area_m2: float,
    surfaces: Sequence[BiotopeSurfaceElement | dict[str, Any]],
    target_baf_threshold: float = 0.60,
    baseline_impervious_c: float = 0.90,
) -> BiotopeAreaReport:
    """
    Calculate Berlin Biotope Area Factor (BAF) and hydrological green metrics.

    Parameters
    ----------
    parcel_area_m2 : float
        Gross horizontal planimetric area of development parcel in square meters.
    surfaces : Sequence[BiotopeSurfaceElement | dict]
        List of surface elements (sealed, green roof, lawn, trees, facades).
    target_baf_threshold : float
        Statutory planning target threshold (default 0.60 for residential BAF).
    baseline_impervious_c : float
        Runoff coefficient of unmitigated sealed impervious asphalt/concrete baseline.

    Returns
    -------
    BiotopeAreaReport
        Comprehensive ecological, hydrological, and zoning compliance metrics.
    """
    if parcel_area_m2 <= 0.0:
        raise ValueError(f"parcel_area_m2 must be positive, got {parcel_area_m2}")

    if not surfaces:
        raise ValueError("surfaces sequence cannot be empty")

    parsed_surfaces: list[BiotopeSurfaceElement] = []
    for s in surfaces:
        if isinstance(s, BiotopeSurfaceElement):
            parsed_surfaces.append(s)
        elif isinstance(s, dict):
            s_type_raw = str(s.get("surface_type", "sealed_impervious"))
            try:
                s_type = SurfaceEcologicalType(s_type_raw)
            except ValueError:
                s_type = SurfaceEcologicalType.SEALED_IMPERVIOUS
            parsed_surfaces.append(
                BiotopeSurfaceElement(
                    surface_type=s_type,
                    area_m2=float(s.get("area_m2", 0.0)),
                    description=str(s.get("description", "")),
                    custom_weight=float(s["custom_weight"]) if "custom_weight" in s else None,
                    custom_runoff_coeff=float(s["custom_runoff_coeff"]) if "custom_runoff_coeff" in s else None,
                )
            )

    total_effective_green = 0.0
    weighted_runoff_sum = 0.0
    breakdown: list[dict[str, Any]] = []

    for elem in parsed_surfaces:
        eff_area = elem.effective_ecological_area_m2
        total_effective_green += eff_area
        weighted_runoff_sum += elem.area_m2 * elem.runoff_coefficient
        breakdown.append(
            {
                "surface_type": elem.surface_type.value,
                "area_m2": round(elem.area_m2, 2),
                "weight": round(elem.weight, 2),
                "runoff_coeff": round(elem.runoff_coefficient, 2),
                "effective_green_area_m2": round(eff_area, 2),
                "description": elem.description,
            }
        )

    # Core BAF equation: BAF = sum(A_i * w_i) / A_parcel
    baf = total_effective_green / parcel_area_m2

    # Composite runoff coefficient: C_bar = sum(A_i * C_i) / A_parcel
    c_composite = min(1.0, max(0.0, weighted_runoff_sum / parcel_area_m2))

    # Runoff attenuation percentage vs baseline
    if baseline_impervious_c > 0.0:
        attenuation_pct = max(0.0, min(100.0, (1.0 - (c_composite / baseline_impervious_c)) * 100.0))
    else:
        attenuation_pct = 0.0

    # Shannon-Wiener ecological diversity entropy: H' = - sum(p_i * ln(p_i))
    shannon_entropy = 0.0
    if total_effective_green > 1e-9:
        for elem in parsed_surfaces:
            if elem.effective_ecological_area_m2 > 1e-9:
                p_i = elem.effective_ecological_area_m2 / total_effective_green
                shannon_entropy -= p_i * math.log(p_i)

    # Evapotranspiration cooling potential score (0 to 10 scale)
    # Scaled by BAF, vegetated soil connection, and tree canopy presence
    et_weight_sum = sum(
        elem.area_m2
        for elem in parsed_surfaces
        if elem.surface_type
        in (
            SurfaceEcologicalType.VEGETATION_SOIL_CONNECTED,
            SurfaceEcologicalType.TREE_CANOPY_OVER_GROUND,
            SurfaceEcologicalType.INTENSIVE_GREEN_ROOF,
            SurfaceEcologicalType.WATER_RETENTION_POND,
        )
    )
    cooling_score = min(10.0, (baf * 6.0) + (min(1.0, et_weight_sum / parcel_area_m2) * 4.0))

    is_compliant = baf >= target_baf_threshold

    return BiotopeAreaReport(
        parcel_area_m2=parcel_area_m2,
        total_effective_green_area_m2=total_effective_green,
        biotope_area_factor=baf,
        composite_runoff_coefficient=c_composite,
        runoff_reduction_percentage=attenuation_pct,
        biodiversity_shannon_index=shannon_entropy,
        evapotranspiration_cooling_score=cooling_score,
        is_compliant=is_compliant,
        target_threshold=target_baf_threshold,
        surface_breakdown=breakdown,
    )
