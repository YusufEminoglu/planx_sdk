# -*- coding: utf-8 -*-
"""Seismic vulnerability, building collapse, and debris volume simulation models."""

from __future__ import annotations

from typing import Optional

import numpy as np


def simulate_seismic_debris(
    building_areas: np.ndarray,
    building_floors: np.ndarray,
    building_years: np.ndarray,
    magnitude: float,
    floor_height: float = 3.0,
    debris_factor: float = 0.4,
    solid_volume_ratio: float = 0.3,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulates seismic structural collapse and debris volume using Monte Carlo.

    For each building, determines collapse probability based on age/construction year
    and scenario moment magnitude (Mw). Then simulates collapse and calculates
    estimated debris radius and excavation volume.

    Args:
        building_areas: NumPy array of shape (N,) containing footprint area (m2) of N buildings.
        building_floors: NumPy array of shape (N,) containing number of floors.
        building_years: NumPy array of shape (N,) containing construction years.
        magnitude: Scenario moment magnitude (Mw) (e.g. 7.0, 7.4).
        floor_height: Average floor height in meters.
        debris_factor: Coefficient determining horizontal debris spread radius (k).
        solid_volume_ratio: Ratio of solid volume of debris compared to bulk building volume.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
          - collapse_probs: NumPy array of shape (N,) of collapse probabilities.
          - collapsed: NumPy array of shape (N,) (binary 0 or 1) indicating collapse status.
          - debris_radii: NumPy array of shape (N,) of debris buffer radius (meters).
          - debris_volumes: NumPy array of shape (N,) of debris excavation volume (m3).
    """
    areas = np.asarray(building_areas, dtype=np.float64)
    floors = np.asarray(building_floors, dtype=np.float64)
    years = np.asarray(building_years, dtype=np.int64)

    n = len(areas)
    if len(floors) != n or len(years) != n:
        raise ValueError(
            "building_areas, building_floors, and building_years " "must have identical length"
        )

    # Determine base probability of collapse based on construction year
    base_probs = np.zeros(n, dtype=np.float64)
    base_probs[years <= 1985] = 0.85
    base_probs[(years > 1985) & (years <= 2000)] = 0.60
    base_probs[(years > 2000) & (years <= 2018)] = 0.25
    base_probs[years > 2018] = 0.05

    # Adjust base probability for scenario magnitude (Mw)
    # Mw 7.0 is the baseline; higher magnitude scale exponentially
    mag_factor = np.exp(0.8 * (magnitude - 7.0))
    collapse_probs = np.clip(base_probs * mag_factor, 0.0, 1.0)

    # Run stochastic Monte Carlo step
    rng = np.random.default_rng(seed)
    random_values = rng.random(n)
    collapsed = (random_values < collapse_probs).astype(np.int64)

    # Calculate building height: H = floors * floor_height
    heights = floors * floor_height

    # Debris radius (meters): E = H * debris_factor if collapsed, else 0
    debris_radii = np.where(collapsed == 1, heights * debris_factor, 0.0)

    # Debris volume (m3): V = area * H * solid_volume_ratio if collapsed, else 0
    debris_volumes = np.where(collapsed == 1, areas * heights * solid_volume_ratio, 0.0)

    return collapse_probs, collapsed, debris_radii, debris_volumes
