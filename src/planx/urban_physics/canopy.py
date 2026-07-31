# -*- coding: utf-8 -*-
"""Urban Canopy Wind Drag, Roughness & Albedo Surface Microclimate Physics."""

from __future__ import annotations

from typing import Any

import numpy as np


def frontal_area_index_canopy(
    building_heights_m: np.ndarray,
    building_widths_m: np.ndarray,
    lot_area_m2: float,
    wind_direction_deg: float = 0.0,
) -> dict[str, Any]:
    """Calculates Frontal Area Index (lambda_F) and Roughness Length (z_0).

    Args:
        building_heights_m: 1D array of building heights (meters).
        building_widths_m: 1D array of building frontal widths (meters).
        lot_area_m2: Total lot/zone surface area in m^2.
        wind_direction_deg: Wind approach direction angle (degrees).

    Returns:
        Dict containing lambda_F, roughness_length_z0, and displacement_height_d0.
    """
    h_arr = np.asarray(building_heights_m, dtype=np.float64)
    w_arr = np.asarray(building_widths_m, dtype=np.float64)

    rad = np.radians(wind_direction_deg)
    eff_widths = w_arr * abs(np.cos(rad)) + w_arr * abs(np.sin(rad))

    frontal_area = float(np.sum(h_arr * eff_widths))
    lambda_f = frontal_area / max(lot_area_m2, 1e-6)

    h_mean = float(np.mean(h_arr)) if len(h_arr) > 0 else 0.0
    z0 = 0.5 * h_mean * lambda_f
    d0 = 0.7 * h_mean * (lambda_f**0.6)

    return {
        "frontal_area_index_lambda_f": lambda_f,
        "roughness_length_z0_m": z0,
        "displacement_height_d0_m": d0,
        "mean_building_height_m": h_mean,
    }


def surface_albedo_cooling_potential(
    current_albedo_grid: np.ndarray,
    target_albedo: float = 0.45,
    solar_irradiance_wm2: float = 800.0,
) -> dict[str, Any]:
    """Calculates Thermal Cooling Potential from Cool Roofs / Cool Pavements Albedo Uplift.

    Args:
        current_albedo_grid: 2D or 1D array of surface albedo values [0.05, 0.9].
        target_albedo: Target cool surface albedo (default 0.45).
        solar_irradiance_wm2: Peak solar irradiance in W/m^2.

    Returns:
        Dict containing temperature_drop_c_grid, mean_temperature_drop_c,
        and max_temperature_drop_c.
    """
    albedo = np.asarray(current_albedo_grid, dtype=np.float64)
    albedo_delta = np.maximum(0.0, target_albedo - albedo)

    h_c = 15.0  # Convective heat transfer coefficient W/(m^2 K)
    temp_drop = (albedo_delta * solar_irradiance_wm2) / h_c

    return {
        "temperature_drop_c_grid": temp_drop,
        "mean_temperature_drop_c": float(np.mean(temp_drop)),
        "max_temperature_drop_c": float(np.max(temp_drop)),
        "mean_albedo_uplift": float(np.mean(albedo_delta)),
    }
