# -*- coding: utf-8 -*-
"""Green Infrastructure & Stormwater Retention Capacity Engines."""

from __future__ import annotations

from typing import Any


def stormwater_green_roof_retention_capacity(
    roof_area_m2: float,
    rainfall_depth_mm: float,
    retention_fraction: float = 0.65,
) -> dict[str, Any]:
    """Calculates Green Roof Runoff Retention Volume and Hydrograph Peak Attenuation.

    Args:
        roof_area_m2: Total green roof surface area in m^2.
        rainfall_depth_mm: Storm rainfall depth in mm.
        retention_fraction: Substrate stormwater retention fraction [0, 1].

    Returns:
        Dict containing retained runoff volume m^3, avoided runoff m^3, and peak reduction %.
    """
    total_rain_volume_m3 = (roof_area_m2 * rainfall_depth_mm) / 1000.0
    retained_volume_m3 = total_rain_volume_m3 * retention_fraction
    avoided_runoff_m3 = retained_volume_m3

    peak_attenuation_pct = retention_fraction * 100.0

    return {
        "total_rainfall_volume_m3": float(total_rain_volume_m3),
        "retained_runoff_volume_m3": float(retained_volume_m3),
        "avoided_runoff_m3": float(avoided_runoff_m3),
        "peak_reduction_percentage": float(peak_attenuation_pct),
    }
