# -*- coding: utf-8 -*-
"""2D Urban Stormwater Flash Flood Inundation & Ponding Model for PlanX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class SurfaceDepressionCell:
    cell_id: str
    ground_elevation_m: float
    impervious_ratio: float  # 0.0 to 1.0
    drainage_inlet_capacity_m3s: float
    surface_area_m2: float = 100.0


@dataclass
class FloodInundationResult:
    peak_inundation_depth_m: float
    total_flooded_area_m2: float
    ponded_water_volume_m3: float
    submerged_cells_count: int
    critical_risk_flag: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to a JSON-ready dictionary."""
        return {
            "peak_depth_m": round(self.peak_inundation_depth_m, 2),
            "flooded_area_m2": round(self.total_flooded_area_m2, 1),
            "ponded_volume_m3": round(self.ponded_water_volume_m3, 1),
            "submerged_cells": self.submerged_cells_count,
            "is_critical": self.critical_risk_flag,
        }


def simulate_urban_flash_flood(
    surface_cells: Sequence[SurfaceDepressionCell | dict[str, Any]],
    storm_rainfall_mm: float = 85.0,  # 100-year return storm downpour
    storm_duration_hours: float = 2.0,
) -> FloodInundationResult:
    """Simulate 2D cell overland stormwater ponding, depression storage filling,
    and flood hazard depths.
    """
    cells: list[SurfaceDepressionCell] = []
    for c in surface_cells:
        if isinstance(c, dict):
            cells.append(
                SurfaceDepressionCell(
                    cell_id=str(c.get("id", "CELL")),
                    ground_elevation_m=float(c.get("elevation_m", 25.0)),
                    impervious_ratio=float(c.get("impervious", 0.75)),
                    drainage_inlet_capacity_m3s=float(c.get("drainage_capacity", 0.02)),
                    surface_area_m2=float(c.get("area_m2", 100.0)),
                )
            )
        else:
            cells.append(c)

    if not cells:
        return FloodInundationResult(0.0, 0.0, 0.0, 0, False)

    min_elev = min(c.ground_elevation_m for c in cells)
    storm_rain_m = storm_rainfall_mm / 1000.0

    total_ponded_vol = 0.0
    flooded_area = 0.0
    submerged_count = 0
    max_depth = 0.0

    for c in cells:
        # Runoff volume: Area * Rain * Impervious - Drainage Infiltration
        runoff_vol = c.surface_area_m2 * storm_rain_m * c.impervious_ratio
        drained_vol = c.drainage_inlet_capacity_m3s * (storm_duration_hours * 3600.0)
        net_ponding = max(0.0, runoff_vol - drained_vol)

        # Low-lying depression accumulation
        elev_diff = c.ground_elevation_m - min_elev
        if elev_diff < 1.0:
            net_ponding += net_ponding * 0.4  # Topographic sink accumulation

        if net_ponding > 0.1:
            depth = net_ponding / max(1.0, c.surface_area_m2)
            total_ponded_vol += net_ponding
            flooded_area += c.surface_area_m2
            submerged_count += 1
            max_depth = max(max_depth, depth)

    is_crit = (max_depth >= 0.30) or (submerged_count >= len(cells) * 0.5)

    return FloodInundationResult(
        peak_inundation_depth_m=max_depth,
        total_flooded_area_m2=flooded_area,
        ponded_water_volume_m3=total_ponded_vol,
        submerged_cells_count=submerged_count,
        critical_risk_flag=is_crit,
    )
