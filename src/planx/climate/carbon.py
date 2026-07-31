# -*- coding: utf-8 -*-
"""Urban Canopy Carbon Sequestration & Ecosystem Services Engines."""

from __future__ import annotations

from typing import Any

import numpy as np


def carbon_sequestration_urban_canopy(
    tree_dbh_cm: np.ndarray,
    canopy_cover_ha: float,
    species_factor: float = 1.0,
) -> dict[str, Any]:
    """Estimates Annual CO2 Sequestration from Urban Forest Canopy.

    Args:
        tree_dbh_cm: 1D array of individual tree Diameter at Breast Height (DBH) in cm.
        canopy_cover_ha: Total canopy surface area in hectares.
        species_factor: Species biomass adjustment multiplier.

    Returns:
        Dict containing annual CO2 sequestration in tonnes, total biomass tonnes, and CO2 storage.
    """
    dbh = np.asarray(tree_dbh_cm, dtype=np.float64)
    if len(dbh) == 0:
        return {
            "annual_co2_sequestration_tonnes": 0.0,
            "total_aboveground_biomass_tonnes": 0.0,
            "total_carbon_storage_tonnes": 0.0,
        }

    # Allometric equation for aboveground biomass (W = a * DBH^b)
    biomass_kg = species_factor * 0.15 * (dbh**2.3)
    total_biomass_t = float(np.sum(biomass_kg) / 1000.0)

    # Carbon content is ~50% of dry biomass, CO2 equivalent is Carbon * 3.67
    total_carbon_t = total_biomass_t * 0.50
    annual_sequestration_t = float(total_carbon_t * 0.035)

    return {
        "annual_co2_sequestration_tonnes": annual_sequestration_t,
        "total_aboveground_biomass_tonnes": total_biomass_t,
        "total_carbon_storage_tonnes": total_carbon_t,
        "sequestration_rate_per_ha": annual_sequestration_t / max(canopy_cover_ha, 1e-4),
    }
