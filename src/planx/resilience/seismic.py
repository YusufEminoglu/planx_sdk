# -*- coding: utf-8 -*-
"""Seismic vulnerability, building collapse, and debris volume simulation models."""

from __future__ import annotations

from typing import Any, Optional

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
            "building_areas, building_floors, and building_years must have identical length"
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


def earthquake_building_collapse_casualty(
    building_types: list[str],
    story_counts: np.ndarray,
    occupancy: np.ndarray,
    pga_g: float,
) -> dict:
    """Calculates seismic building collapse probabilities and estimated casualties.

    Args:
        building_types: List of strings ("RC", "Masonry", "Timber", "Steel").
        story_counts: 1D array of building story counts.
        occupancy: 1D array of peak building occupants.
        pga_g: Peak Ground Acceleration float in g (e.g. 0.45g).

    Returns:
        Dict containing seismic vulnerability assessment:
          - collapse_probability: 1D NumPy array of collapse probability P_D in [0, 1].
          - expected_collapsed_buildings: Int expected number of collapsed structures.
          - estimated_fatalities: Float estimated fatalities.
          - estimated_injuries: Float estimated injuries.
    """
    stories = np.asarray(story_counts, dtype=np.float64)
    occ = np.asarray(occupancy, dtype=np.float64)
    n = len(building_types)

    if len(stories) != n or len(occ) != n:
        raise ValueError("building_types, story_counts, and occupancy must have equal length.")

    # Vulnerability fragility median alpha and beta by construction type
    # (PGA median in g for complete damage state)
    params = {
        "RC": (0.50, 0.40),
        "Masonry": (0.30, 0.45),
        "Timber": (0.65, 0.35),
        "Steel": (0.75, 0.35),
    }

    import scipy.stats as stats

    p_collapse = np.zeros(n, dtype=np.float64)

    for i in range(n):
        btype = building_types[i]
        alpha, beta = params.get(btype, (0.45, 0.40))
        # Story multiplier (taller buildings have higher vulnerability under long-period shaking)
        story_mult = 1.0 + 0.05 * max(0.0, stories[i] - 1.0)
        eff_pga = pga_g * story_mult

        p_collapse[i] = float(stats.norm.cdf((np.log(max(1e-4, eff_pga)) - np.log(alpha)) / beta))

    p_collapse = np.clip(p_collapse, 0.0, 1.0)

    exp_collapsed = int(np.sum(p_collapse >= 0.5))

    fatalities = float(np.sum(p_collapse * occ * 0.15))  # 15% fatality rate in collapse
    injuries = float(np.sum(p_collapse * occ * 0.35))  # 35% injury rate in collapse

    return {
        "collapse_probability": p_collapse,
        "expected_collapsed_buildings": exp_collapsed,
        "estimated_fatalities": fatalities,
        "estimated_injuries": injuries,
    }


def seismic_damage_loss_curve(
    pga_values_g: np.ndarray,
    building_counts: np.ndarray,
    replacement_values: np.ndarray,
    building_type: str = "c2_medium",
) -> dict[str, Any]:
    """Computes Hazus-compatible lognormal building damage state probabilities and loss curves.

    Args:
        pga_values_g: NumPy array of shape (P,) of PGA ground shaking values in g
            (e.g. 0.05 to 1.5g).
        building_counts: NumPy array of shape (N,) of number of buildings per asset class / region.
        replacement_values: NumPy array of shape (N,) of replacement cost ($) per asset class
            / region.
        building_type: Structural type key (e.g. 'c2_medium', 'w1', 'rm1', 's1', 'urm').

    Returns:
        Dict containing:
          - pga_values: (P,) copy of input PGA values
          - damage_state_probabilities: (P, 5) array of [None, Slight, Moderate, Extensive,
            Complete] probabilities
          - expected_loss_ratio: (P,) array of expected building loss ratio [0, 1]
          - total_economic_loss: (P,) array of total monetary losses ($)
          - building_collapse_count: (P,) expected count of collapsed buildings (Complete DS *
            building_counts)
    """
    pga_values = np.asarray(pga_values_g, dtype=np.float64)
    counts = np.asarray(building_counts, dtype=np.float64)
    values = np.asarray(replacement_values, dtype=np.float64)

    if np.any(pga_values < 0):
        raise ValueError("PGA values cannot be negative.")
    if np.any(counts < 0) or np.any(values < 0):
        raise ValueError("Building counts and replacement values cannot be negative.")

    params = {
        "w1": ([0.15, 0.25, 0.40, 0.70], 0.60),
        "c2_medium": ([0.20, 0.35, 0.60, 0.90], 0.64),
        "s1": ([0.22, 0.38, 0.65, 0.95], 0.62),
        "rm1": ([0.18, 0.30, 0.50, 0.80], 0.65),
        "urm": ([0.10, 0.18, 0.32, 0.55], 0.70),
    }

    if building_type not in params:
        raise ValueError(f"Unknown building_type: {building_type}")

    medians, beta = params[building_type]

    import scipy.stats as stats

    # Safe log of PGA to avoid log(0)
    pga_safe = np.where(pga_values > 0, pga_values, 1e-10)
    ln_pga = np.log(pga_safe)

    # Compute P(DS >= ds | PGA)
    p_ds = np.zeros((len(pga_values), 4), dtype=np.float64)
    for i, median in enumerate(medians):
        p_ds[:, i] = stats.norm.cdf((ln_pga - np.log(median)) / beta)

    # If PGA was 0, probabilities should be 0
    p_ds[pga_values == 0, :] = 0.0

    p_complete = p_ds[:, 3]
    p_extensive = p_ds[:, 2] - p_ds[:, 3]
    p_moderate = p_ds[:, 1] - p_ds[:, 2]
    p_slight = p_ds[:, 0] - p_ds[:, 1]
    p_none = 1.0 - p_ds[:, 0]

    damage_state_probabilities = np.column_stack(
        (p_none, p_slight, p_moderate, p_extensive, p_complete)
    )

    damage_ratios = np.array([0.0, 0.02, 0.10, 0.50, 1.00], dtype=np.float64)
    expected_loss_ratio = np.sum(damage_state_probabilities * damage_ratios, axis=1)

    total_asset_value = np.sum(counts * values)
    total_economic_loss = expected_loss_ratio * total_asset_value

    total_buildings = np.sum(counts)
    building_collapse_count = p_complete * total_buildings

    return {
        "pga_values": pga_values.copy(),
        "damage_state_probabilities": damage_state_probabilities,
        "expected_loss_ratio": expected_loss_ratio,
        "total_economic_loss": total_economic_loss,
        "building_collapse_count": building_collapse_count,
    }


def seismic_road_blockage_simulation(
    street_segment_coords: np.ndarray,
    street_widths_m: np.ndarray,
    adjacent_building_heights: np.ndarray,
    building_collapse_probabilities: np.ndarray,
    debris_expansion_factor: float = 0.5,
) -> dict[str, Any]:
    """Models post-earthquake structural collapse debris projection into street right-of-ways
    and evaluates road blockage probabilities for emergency response.

    Args:
        street_segment_coords: NumPy array of shape (S, 2, 2) containing start and end point
            coordinates per street segment.
        street_widths_m: NumPy array of shape (S,) of street width in meters. Must be > 0.
        adjacent_building_heights: NumPy array of shape (S,) of max height of adjacent
            buildings (m).
        building_collapse_probabilities: NumPy array of shape (S,) of probability of adjacent
            building collapse [0, 1].
        debris_expansion_factor: Fraction k of building height projected as debris radius.
            Default is 0.5.

    Returns:
        Dict containing:
          - blockage_probabilities: NumPy array of shape (S,) of float array P_block.
          - debris_extents_m: NumPy array of shape (S,) of float array W_debris.
          - blockage_ratios: NumPy array of shape (S,) of float array B_s.
          - blocked_segments_count: Integer count of blocked segments.
          - restricted_segments_count: Integer count of restricted segments.
          - open_segments_count: Integer count of open segments.
    """
    coords = np.asarray(street_segment_coords, dtype=np.float64)
    widths = np.asarray(street_widths_m, dtype=np.float64)
    heights = np.asarray(adjacent_building_heights, dtype=np.float64)
    probs = np.asarray(building_collapse_probabilities, dtype=np.float64)

    s = len(widths)
    if coords.shape != (s, 2, 2):
        raise ValueError(f"street_segment_coords must have shape ({s}, 2, 2)")
    if len(heights) != s or len(probs) != s:
        raise ValueError(
            "street_widths_m, adjacent_building_heights, and building_collapse_probabilities "
            "must have identical length"
        )
    if np.any(widths <= 0):
        raise ValueError("Street widths must be greater than 0.")
    if np.any(heights < 0):
        raise ValueError("Building heights cannot be negative.")
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("Building collapse probabilities must be in [0, 1].")

    # Debris extent W_debris_s = adjacent_building_heights_s * debris_expansion_factor
    debris_extents = heights * debris_expansion_factor

    # Blockage Ratio B_s = W_debris_s / max(street_widths_m_s, 1.0)
    safe_widths = np.maximum(widths, 1.0)
    blockage_ratios = debris_extents / safe_widths

    # Street Blockage Probability P_block_s = building_collapse_probabilities_s * min(1.0, B_s)
    blockage_probabilities = probs * np.minimum(1.0, blockage_ratios)

    # Street Status Classification
    blocked_mask = (blockage_probabilities >= 0.70) | (blockage_ratios >= 1.0)
    restricted_mask = (
        (blockage_probabilities >= 0.30) & (blockage_probabilities < 0.70) & (~blocked_mask)
    )
    open_mask = (blockage_probabilities < 0.30) & (~blocked_mask)

    return {
        "blockage_probabilities": blockage_probabilities,
        "debris_extents_m": debris_extents,
        "blockage_ratios": blockage_ratios,
        "blocked_segments_count": int(np.sum(blocked_mask)),
        "restricted_segments_count": int(np.sum(restricted_mask)),
        "open_segments_count": int(np.sum(open_mask)),
    }


def seismic_liquefaction_potential_index(
    pga_g: float,
    groundwater_depth_m: float,
    spt_n_values: np.ndarray,
    layer_depths_m: np.ndarray,
) -> dict[str, Any]:
    """Calculates Iwasaki Seismic Liquefaction Potential Index (LPI).

    LPI = sum_0^20 (10 - 0.5*z) * F_L * dz

    Args:
        pga_g: Peak Ground Acceleration in g units.
        groundwater_depth_m: Water table depth in meters.
        spt_n_values: 1D array of SPT-N soil blow counts at layer depths.
        layer_depths_m: 1D array of layer depths in meters (0 to 20m).

    Returns:
        Dict containing LPI score, severity classification, and liquefaction risk category.
    """
    n_vals = np.asarray(spt_n_values, dtype=np.float64)
    depths = np.asarray(layer_depths_m, dtype=np.float64)

    lpi = 0.0
    for i in range(len(depths)):
        z = depths[i]
        if z > 20.0 or z < groundwater_depth_m:
            continue

        w_z = 10.0 - 0.5 * z
        f_L = max(0.0, 1.0 - (n_vals[i] / max(1.0, pga_g * 100.0)))
        lpi += w_z * f_L

    lpi_val = float(lpi)
    if lpi_val >= 15.0:
        risk_class = "Very High"
    elif lpi_val >= 5.0:
        risk_class = "High"
    elif lpi_val > 0.0:
        risk_class = "Low"
    else:
        risk_class = "Very Low"

    return {
        "liquefaction_potential_index": lpi_val,
        "risk_classification": risk_class,
        "is_high_liquefaction_risk": lpi_val >= 5.0,
    }
